import json

import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.verify import Band, VLMRouting, interpret, route_with_vlm, route_without_vlm
from dnt.refine.vlm import VLMTransientError
from dnt.refine.vlm.cache import AnswerCache
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.prompts import build_prompt, options_for
from dnt.refine.vlm.runner import Verdict, VLMRunner

BAND = Band(accept_above=0.875, reject_below=0.375)  # midpoint 0.625, exact in binary


def reply(answer, conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf, "reason": "because"})


class Images:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.asked = []

    def build_many(self, events):
        self.asked += [e.id for e in events]
        return {e.id: (None if e.id in self.missing else b"jpeg") for e in events}


def make(kind, stage, score, idx=1, tracks=(1,), **params):
    lineage = [[[t, 0, 99]] for t in tracks]
    ev = Event.propose(stage=stage, kind=kind, tracks=list(tracks), lineage=lineage,
                       frames=(0, 99), params=params, algo_score=score, signals={})
    ev.id = f"{stage}-r0-{idx:06d}"
    return ev


CREATED = []


@pytest.fixture(autouse=True)
def _close_runners():
    yield
    for r in CREATED:
        r.close()  # stop each runner's loop thread, so later tests see no stray threads
    CREATED.clear()


def routing(script, target="person", **vlm):
    cfg = RefineConfig.defaults(target)
    for k, v in vlm.items():
        setattr(cfg.vlm, k, v)
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "m"
    backend = FakeBackend(script)
    runner = VLMRunner(cfg.vlm, backend, None, sleep=lambda s: _noop())
    CREATED.append(runner)
    images = Images()
    return VLMRouting(runner, images, cfg, 10.0), backend, images


async def _noop():
    return None


def link(score, idx=1, gate="normal"):
    return make(EventKind.LINK, "link", score, idx, tracks=(1, 2), gap=[50, 60], gate=gate)


def test_banded_events_never_call_the_vlm():
    r, b, _ = routing({})
    hi, lo = link(0.95, 1), link(0.2, 2)
    route_with_vlm([hi, lo], BAND, vlm=r)
    assert (hi.decision, lo.decision) == (Decision.AUTO_ACCEPT, Decision.AUTO_REJECT)
    assert b.calls == [] and hi.vlm is None
    assert [h["source"] for h in hi.decision_history] == ["auto"]


@pytest.mark.parametrize(
    "answer,conf,decision,edit",
    [
        ("same_individual", 0.9, Decision.VLM_ACCEPT, True),
        ("different", 0.9, Decision.VLM_REJECT, False),
        ("unsure", 0.99, Decision.HUMAN_PENDING, False),
        ("same_individual", 0.5, Decision.HUMAN_PENDING, False),  # below min_conf 0.7
    ],
)
def test_link_answers(answer, conf, decision, edit):
    r, b, _ = routing({"*": reply(answer, conf)})
    ev = link(0.6)
    key, kind, params = ev.proposal_key, ev.kind, dict(ev.params)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision
    assert (ev.edit is not None) is edit
    if edit:
        assert ev.edit == {"kind": "LINK", "params": params}
    assert (ev.proposal_key, ev.kind, ev.params) == (key, kind, params)
    assert ev.vlm["answer"] == answer and ev.vlm["confidence"] == conf
    assert ev.vlm["backend"] == "fake" and ev.vlm["model"] == "fake-1"
    # a confident answer has no error; one below vlm.min_conf says why it stayed pending
    assert (ev.vlm["error"] is None) if conf >= 0.7 else ("min_conf" in ev.vlm["error"])
    assert ev.vlm["evidence"] is None and ev.vlm["cached"] is False
    assert ev.decision_history[-1]["source"] == "vlm"
    assert b.calls[0]["tag"] == f"LINK:{ev.id}" and b.calls[0]["options"][0] == "same_individual"


def test_split_answers_are_inverted():
    r, _, _ = routing({"SPLIT": [reply("different"), reply("same_individual")]},
                      max_concurrency=1)
    a = make(EventKind.SPLIT, "switch", 0.625, 1, cut_frame=40)  # asked first (closest to mid)
    b = make(EventKind.SPLIT, "switch", 0.5, 2, cut_frame=70)
    route_with_vlm([a, b], BAND, vlm=r)
    assert a.decision is Decision.VLM_ACCEPT and b.decision is Decision.VLM_REJECT


@pytest.mark.parametrize(
    "answer,decision,edit_kind,edit_params",
    [
        ("pedestrian", Decision.VLM_REJECT, None, None),
        ("person_in_vehicle", Decision.VLM_ACCEPT, "DROP", {"reason": "in_vehicle", "spans": None}),
        ("not_a_person", Decision.VLM_ACCEPT, "DROP", {"reason": "static", "spans": None}),
        ("cyclist", Decision.VLM_ACCEPT, "RECLASS", {"new_cls": 1, "spans": None}),
        ("motorcycle_rider", Decision.VLM_ACCEPT, "RECLASS", {"new_cls": 3, "spans": None}),
        ("scooter_rider", Decision.VLM_ACCEPT, "RECLASS", {"new_cls": 36, "spans": None}),
        ("unsure", Decision.HUMAN_PENDING, None, None),
    ],
)
def test_person_screen_answers_redirect_the_edit_not_the_proposal(
    answer, decision, edit_kind, edit_params
):
    r, _, _ = routing({"*": reply(answer)})
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason="static", spans=None)
    key, params = ev.proposal_key, dict(ev.params)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision
    if edit_kind is None:
        assert ev.edit is None
    else:
        assert ev.edit == {"kind": edit_kind, "params": edit_params}
    assert ev.kind is EventKind.DROP and ev.params == params and ev.proposal_key == key


def test_partial_spans_are_copied_into_the_redirected_edit():
    r, _, _ = routing({"*": reply("cyclist")})
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason="static", spans=[[10, 20]])
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.edit == {"kind": "RECLASS", "params": {"new_cls": 1, "spans": [[10, 20]]}}


def test_a_missing_reclass_map_key_leaves_the_event_pending():
    r, _, _ = routing({"*": reply("scooter_rider")})
    del r.cfg.reclass_map["scooter"]
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason="static", spans=None)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and "scooter" in ev.vlm["error"]


@pytest.mark.parametrize(
    "reason,answer,decision",
    [
        ("duplicate", "part_or_duplicate_of_another_vehicle", Decision.VLM_ACCEPT),
        ("static", "part_or_duplicate_of_another_vehicle", Decision.HUMAN_PENDING),
        ("duplicate", "vehicle", Decision.VLM_REJECT),
        ("static", "not_a_vehicle", Decision.VLM_ACCEPT),
    ],
)
def test_vehicle_screen_answers(reason, answer, decision):
    r, b, _ = routing({"*": reply(answer)}, target="vehicle")
    extra = {"of": 2} if reason == "duplicate" else {}
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason=reason, spans=None, **extra)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision
    assert b.calls[0]["options"][0] == "vehicle"
    if decision is Decision.VLM_ACCEPT and answer == "not_a_vehicle":
        assert ev.edit == {"kind": "DROP", "params": {"reason": "static", "spans": None}}
    if decision is Decision.VLM_ACCEPT and reason == "duplicate":
        assert ev.edit["params"]["reason"] == "duplicate" and ev.edit["params"]["of"] == 2


@pytest.mark.parametrize(
    "answer,decision,new_cls",
    [
        ("cyclist", Decision.AUTO_ACCEPT, 1),
        ("scooter_rider", Decision.AUTO_ACCEPT, 36),
        ("pedestrian", Decision.HUMAN_PENDING, None),
        ("not_a_person", Decision.HUMAN_PENDING, None),
        ("unsure", Decision.HUMAN_PENDING, None),
    ],
)
def test_rider_subtype_call_cannot_overturn_the_rider_decision(answer, decision, new_cls):
    r, b, _ = routing({"*": reply(answer)})
    ev = make(EventKind.RECLASS, "screen", 0.95, 1, new_cls=None, spans=None)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision and len(b.calls) == 1
    if new_cls is not None:
        assert ev.edit["params"]["new_cls"] == new_cls and ev.params["new_cls"] is None
        assert ev.decision_history[-1]["source"] == "auto"
    else:
        assert ev.edit is None and ev.signals["needs_subtype"] is True


def test_a_non_finite_confidence_never_accepts_an_edit():
    cfg = RefineConfig.defaults()
    for conf in (float("nan"), float("inf")):
        v = Verdict("different", conf, "", {"different": 1}, False, None)
        split = make(EventKind.SPLIT, "switch", 0.6, 1, cut_frame=40)
        assert interpret(split, v, mode="decide", cfg=cfg) == (Decision.HUMAN_PENDING, None)
        rider = make(EventKind.RECLASS, "screen", 0.95, 2, new_cls=None, spans=None)
        v2 = Verdict("cyclist", conf, "", {"cyclist": 1}, False, None)
        assert interpret(rider, v2, mode="subtype", cfg=cfg) == (Decision.HUMAN_PENDING, None)


def test_a_damaged_cache_entry_cannot_accept_an_edit(tmp_path):
    ev = make(EventKind.SPLIT, "switch", 0.6, 1, cut_frame=40)
    r, _, _ = routing({})  # nothing is scripted: any backend call fails
    prompt = build_prompt(ev, "person", options_for(ev, "person"), fps=10.0)
    key = AnswerCache.key(b"jpeg", prompt, options_for(ev, "person"), "fake", "fake-1", 0.0, 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"answer": "different", "confidence": float("nan"),
                                "reason": "r", "raw": "x"}))  # NaN literal, readable JSON
    r.runner.cache = AnswerCache(tmp_path)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and ev.edit is None
    assert r.runner.cache_hits == 0 and ev.vlm["error"].startswith("RuntimeError")


def test_a_rider_with_a_hinted_subtype_needs_no_call():
    r, b, _ = routing({})
    ev = make(EventKind.RECLASS, "screen", 0.95, 1, new_cls=3, spans=None)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.AUTO_ACCEPT and b.calls == []


def test_the_budget_goes_to_the_events_closest_to_the_band_midpoint():
    r, b, _ = routing({"*": reply("same_individual")}, max_calls=2, max_concurrency=1)
    scores = [0.5, 0.625, 0.75, 0.5625, 0.6875]  # distances from 0.625: .125 0 .125 1/16 1/16
    evs = [link(s, i + 1) for i, s in enumerate(scores)]
    route_with_vlm(evs, BAND, vlm=r)
    answered = [e.id for e in evs if e.decision is Decision.VLM_ACCEPT]
    assert answered == ["link-r0-000002", "link-r0-000004"]  # 0, then 1/16 (list order tie)
    left = [e for e in evs if e.decision is Decision.HUMAN_PENDING]
    assert len(left) == 3 and all(e.vlm["error"] == "budget" for e in left)
    assert [c["tag"] for c in b.calls] == ["LINK:link-r0-000002", "LINK:link-r0-000004"]


def test_orphans_and_position_records_are_never_asked():
    r, b, images = routing({})
    orphan = make(EventKind.DROP, "orphan", 0.5, 1, reason="orphan", spans=None)
    route_with_vlm([orphan], Band(0.7, 0.3), vlm=r)
    assert orphan.decision is Decision.HUMAN_PENDING and b.calls == [] and images.asked == []


def test_no_evidence_image_means_pending_without_a_call():
    r, b, images = routing({"*": reply("same_individual")})
    images.missing = {"link-r0-000001"}
    ev = link(0.6)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and b.calls == []
    assert ev.vlm["error"] == "no evidence image"


def test_a_failing_evidence_builder_leaves_events_pending_not_crashed():
    class Broken:
        def build_many(self, events):
            raise ValueError("cannot read frame 5")

    r, b, _ = routing({"*": reply("same_individual")})
    r.evidence = Broken()
    ev = link(0.6)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and b.calls == []
    assert ev.vlm["error"] == "no evidence image"


def test_backend_failures_leave_the_event_pending():
    r, _, _ = routing({"*": [RuntimeError("boom")]})
    ev = link(0.6)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and ev.edit is None
    assert ev.vlm["answer"] is None and ev.vlm["error"].startswith("RuntimeError")
    r2, _, _ = routing({"*": [VLMTransientError("429")]})
    ev2 = link(0.6)
    route_with_vlm([ev2], BAND, vlm=r2)
    assert ev2.decision is Decision.HUMAN_PENDING and ev2.vlm["error"].startswith("transient")


def test_the_without_vlm_path_is_unchanged():
    ev = link(0.6)
    route_without_vlm([ev], BAND)
    assert ev.decision is Decision.HUMAN_PENDING and ev.vlm is None
