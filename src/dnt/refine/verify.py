"""Routing proposals by confidence band (spec 4.3). VLM routing arrives in Plan 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .events import ACCEPTED, Decision, Event, EventKind


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
