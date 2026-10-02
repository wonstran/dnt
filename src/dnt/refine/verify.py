"""Routing proposals by confidence band and by a VLM (spec 4.3, 7.2)."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass

from .events import ACCEPTED, Decision, Event, EventKind
from .vlm.prompts import RIDER_OPTIONS, build_prompt, options_for
from .vlm.runner import Question, Verdict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Band:
    """A stage's auto-accept and auto-reject thresholds."""

    accept_above: float
    reject_below: float

    @classmethod
    def of(cls, stage_cfg) -> Band:
        """Build a band from a stage config with ``accept_above`` / ``reject_below``."""
        return cls(float(stage_cfg.accept_above), float(stage_cfg.reject_below))


def band_route(score: float, band: Band) -> Decision | None:
    """Return AUTO_ACCEPT, AUTO_REJECT, or None when the score is in the uncertain band."""
    if score >= band.accept_above:
        return Decision.AUTO_ACCEPT
    if score < band.reject_below:
        return Decision.AUTO_REJECT
    return None


def decide(event: Event, decision: Decision, *, source: str, round: int = 0) -> None:
    """Record a decision; an accepted event's edit starts as its proposal (spec 4.1)."""
    event.decision = decision
    event.decision_history.append(
        {"decision": str(decision), "round": int(round), "source": source}
    )
    event.edit = (
        {"kind": str(event.kind), "params": copy.deepcopy(event.params)}
        if decision in ACCEPTED
        else None
    )


def route_without_vlm(events: list[Event], band: Band, *, round: int = 0) -> None:
    """Route by band only; expects UNDECIDED events.

    Overwrites any existing decision/edit and appends to the history. The uncertain band
    and unresolved rider subtypes become pending.
    """
    for ev in events:
        d = band_route(ev.algo_score, band)
        if (
            d is Decision.AUTO_ACCEPT
            and ev.kind is EventKind.RECLASS
            and ev.params.get("new_cls") is None
        ):
            ev.signals["needs_subtype"] = True
            d = Decision.HUMAN_PENDING
        decide(ev, Decision.HUMAN_PENDING if d is None else d, source="auto", round=round)


#: ``vlm.error`` of an event that was not asked because no evidence image could be made.
NO_EVIDENCE = "no evidence image"


@dataclass
class VLMRouting:
    """Everything `route_with_vlm` needs: the runner, the evidence builder, the config, fps."""

    runner: object
    evidence: object
    cfg: object
    fps: float


def _vlm_record(vlm: VLMRouting, verdict: Verdict | None, error: str | None) -> dict:
    return {
        "backend": vlm.runner.backend.name,
        "model": vlm.runner.backend.model,
        "answer": None if verdict is None else verdict.answer,
        "confidence": 0.0 if verdict is None else verdict.confidence,
        "votes": {} if verdict is None else verdict.votes,
        "reason": "" if verdict is None else verdict.reason,
        "evidence": None,
        "cached": False if verdict is None else verdict.cached,
        "error": error,
    }


def _screen_edit(event: Event, kind: str, **params) -> dict:
    return {"kind": kind, "params": {**params, "spans": copy.deepcopy(event.params.get("spans"))}}


def interpret(event: Event, verdict: Verdict, *, mode: str, cfg) -> tuple[Decision, dict | None]:
    """Map a verdict to ``(decision, redirected edit or None)`` (spec 7.2).

    A returned edit of ``None`` means "keep the proposal's edit" (or no edit when the decision
    is not an acceptance). Errors, ties, ``unsure`` and low confidence are ``HUMAN_PENDING``.
    """
    pending = (Decision.HUMAN_PENDING, None)
    ans = verdict.answer
    if verdict.error is not None or ans is None or ans == "unsure":
        return pending
    if not math.isfinite(verdict.confidence) or verdict.confidence < float(cfg.vlm.min_conf):
        return pending
    if mode == "subtype":
        if ans not in RIDER_OPTIONS or RIDER_OPTIONS[ans] not in cfg.reclass_map:
            return pending
        new_cls = int(cfg.reclass_map[RIDER_OPTIONS[ans]])
        return Decision.AUTO_ACCEPT, _screen_edit(event, "RECLASS", new_cls=new_cls)
    if event.kind is EventKind.SPLIT:
        return (Decision.VLM_ACCEPT if ans == "different" else Decision.VLM_REJECT), None
    if event.kind is EventKind.LINK:
        return (Decision.VLM_ACCEPT if ans == "same_individual" else Decision.VLM_REJECT), None
    if cfg.target == "person":
        if ans == "pedestrian":
            return Decision.VLM_REJECT, None
        if ans == "person_in_vehicle":
            return Decision.VLM_ACCEPT, _screen_edit(event, "DROP", reason="in_vehicle")
        if ans == "not_a_person":
            return Decision.VLM_ACCEPT, _screen_edit(event, "DROP", reason="static")
        if ans in RIDER_OPTIONS and RIDER_OPTIONS[ans] in cfg.reclass_map:
            new_cls = int(cfg.reclass_map[RIDER_OPTIONS[ans]])
            return Decision.VLM_ACCEPT, _screen_edit(event, "RECLASS", new_cls=new_cls)
        return pending
    if ans == "vehicle":
        return Decision.VLM_REJECT, None
    if ans == "part_or_duplicate_of_another_vehicle":
        if event.params.get("reason") == "duplicate":
            return Decision.VLM_ACCEPT, None
        return pending
    if ans == "not_a_vehicle":
        return Decision.VLM_ACCEPT, _screen_edit(event, "DROP", reason="static")
    return pending


def route_with_vlm(events: list[Event], band: Band, *, vlm: VLMRouting, round: int = 0) -> None:
    """Route ``events`` by band, sending the uncertain ones to the VLM (spec 4.3, 7.2, 7.4).

    Expects undecided events that already have ids. See ``interpret`` for the answer mapping.
    """
    cfg = vlm.cfg
    asked: list[tuple[int, Event, str, list[str]]] = []
    for i, ev in enumerate(events):
        d = band_route(ev.algo_score, band)
        options = options_for(ev, cfg.target)
        subtype = (
            d is Decision.AUTO_ACCEPT
            and ev.kind is EventKind.RECLASS
            and ev.params.get("new_cls") is None
        )
        if subtype:
            ev.signals["needs_subtype"] = True
        if options is not None and (d is None or subtype):
            asked.append((i, ev, "subtype" if subtype else "decide", options))
        else:
            decide(ev, Decision.HUMAN_PENDING if d is None else d, source="auto", round=round)
    if not asked:
        return
    mid = (band.accept_above + band.reject_below) / 2.0
    asked.sort(key=lambda t: (abs(t[1].algo_score - mid), t[0]))
    try:
        images = vlm.evidence.build_many([ev for _, ev, _, _ in asked])
    except Exception as err:  # VLM trouble never aborts a run (spec 7.4)
        log.warning("could not build the evidence images: %s", err)
        images = {}
    todo, questions = [], []
    for _, ev, mode, options in asked:
        image = images.get(ev.id)
        if image is None:
            ev.vlm = _vlm_record(vlm, None, NO_EVIDENCE)
            decide(ev, Decision.HUMAN_PENDING, source="auto", round=round)
            continue
        prompt = build_prompt(ev, cfg.target, options, fps=vlm.fps)
        questions.append(Question(f"{ev.kind}:{ev.id}", image, prompt, options))
        todo.append((ev, mode))
    verdicts = vlm.runner.ask_many(questions)
    for (ev, mode), verdict in zip(todo, verdicts, strict=True):
        decision, edit = interpret(ev, verdict, mode=mode, cfg=cfg)
        error = verdict.error
        if error is None and decision is Decision.HUMAN_PENDING and verdict.answer is not None:
            error = _pending_reason(ev, verdict, mode, cfg)
        ev.vlm = _vlm_record(vlm, verdict, error)
        source = "auto" if decision is Decision.AUTO_ACCEPT else "vlm"
        decide(ev, decision, source=source, round=round)
        if edit is not None:
            ev.edit = edit
        if mode == "subtype" and decision is Decision.AUTO_ACCEPT:
            ev.signals.pop("needs_subtype", None)


def _pending_reason(ev: Event, verdict: Verdict, mode: str, cfg) -> str | None:
    """Say why an answered event stayed pending (None for a plain `unsure`)."""
    ans = verdict.answer
    if ans == "unsure":
        return None
    if not math.isfinite(verdict.confidence):
        return "the confidence is not a finite number"
    if verdict.confidence < float(cfg.vlm.min_conf):
        return f"confidence {verdict.confidence:.2f} is below vlm.min_conf"
    if ans in RIDER_OPTIONS and RIDER_OPTIONS[ans] not in cfg.reclass_map:
        return f"reclass_map has no {RIDER_OPTIONS[ans]!r} entry"
    if mode == "subtype":
        return f"answered {ans!r}, which disagrees with the rider decision"
    return f"answered {ans!r}, which does not settle this event"
