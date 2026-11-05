from dnt.refine.apply import lineage_of_rows
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.link import run_link_stage
from dnt.refine.primitives import occlusion_flags
from dnt.refine.refiner import _Stages
from dnt.refine.reversal import ReversalSides
from dnt.refine.verify import Band, decide, route_without_vlm

from ._reversal import FPS, rows, work_of


def _merge(work, a, b, score):
    ga, gb = work[work["track"] == a], work[work["track"] == b]
    lo, hi = max(ga["frame"].min(), gb["frame"].min()), min(ga["frame"].max(), gb["frame"].max())
    ev = Event.propose(stage="dedup", kind=EventKind.MERGE, tracks=[a, b],
                       lineage=[lineage_of_rows(ga), lineage_of_rows(gb)], frames=(lo, hi),
                       params={"span": [int(lo), int(hi)]}, algo_score=score)
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    return ev


def _component_work():
    # A raw 1 (frames 0-10, then even to 28), X raw 2 (odd 11-49), B raw 1 (even 30-48, then 50-59)
    return work_of(rows(1, [*range(0, 11), *range(12, 30, 2)]), rows(2, range(11, 50, 2)),
                   rows(3, [*range(30, 50, 2), *range(50, 60)]), raw={3: 1})


def test_without_sides_dedup_joins_both_sides_through_a_component():
    work = _component_work()
    out = apply_merges(work, [_merge(work, 1, 2, 0.9), _merge(work, 2, 3, 0.8)],
                       RefineConfig.defaults("vehicle"))
    assert sorted(out.work["track"].unique()) == [1]


def test_dedup_refuses_the_join_that_would_put_both_sides_in_one_component():
    work = _component_work()
    evs = [_merge(work, 1, 2, 0.9), _merge(work, 2, 3, 0.8)]
    out = apply_merges(work, evs, RefineConfig.defaults("vehicle"),
                       reversal_sides=ReversalSides(cuts={1: [30]}))
    assert sorted(out.work["track"].unique()) == [1, 3]
    assert evs[0].applied and not evs[1].applied
    assert evs[1].signals["skipped_reason"] == "conflict"
    assert evs[1].signals["conflicts_with"] == {"tracks": [1, 3], "why": "reversal",
                                                "proposal_key": None}
    assert out.counts["conflict"] == 1


def _replacement_work():
    # review 21:42: A raw 1 frames 0-29 (0.8), B raw 1 frames 30-59 (0.9), a 0.95 copy of A
    return work_of(rows(1, range(0, 30), score=0.8), rows(2, range(30, 60), score=0.9),
                   rows(3, range(0, 30), score=0.95), raw={2: 1})


def _no_heading_cfg():
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.normal_heading_check = False  # only the cut protection may refuse the join
    return cfg


def _link(work, cfg, sides, excluded):
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    band = Band.of(cfg.link)
    return run_link_stage(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                          occluded=occ, route=lambda e: route_without_vlm(e, band),
                          excluded=excluded, reversal_sides=sides)


def test_a_replaced_side_is_relinked_without_carried_tags():
    cfg, work = _no_heading_cfg(), _replacement_work()
    evs = propose_merges(work, cfg, FPS, None)
    route_without_vlm(evs, Band.of(cfg.dedup))
    out = apply_merges(work, evs, cfg)
    assert set(out.work.loc[out.work["track"] == 1, "raw_id"]) == {3}  # all of A's rows lost
    (ev,) = _link(out.work, cfg, None, out.excluded).events
    assert ev.tracks == [1, 2] and ev.applied


def test_a_replaced_side_keeps_its_tag_and_is_not_relinked():
    cfg, work = _no_heading_cfg(), _replacement_work()
    sides = ReversalSides(cuts={1: [30]})
    evs = propose_merges(work, cfg, FPS, None)
    route_without_vlm(evs, Band.of(cfg.dedup))
    tags_before = sides.table_tags(work)
    out = apply_merges(work, evs, cfg, reversal_sides=sides)
    sides.absorb(out.absorbed, tags_before)
    assert (1, 0) in sides.carried[1]
    assert _link(out.work, cfg, sides, out.excluded).events == []


def test_the_pipeline_keeps_a_replaced_side_apart():
    cfg = _no_heading_cfg()
    cfg.switch.enabled = False
    cfg.screen.enabled = False  # screen could drop the copy before dedup sees it
    raw = work_of(rows(1, range(0, 30), score=0.8) + rows(1, range(30, 60), score=0.9),
                  rows(3, range(0, 30), score=0.95))
    stages = _Stages(cfg, FPS, None, None, None, None, {})
    out, events = stages.run(raw)
    (cut,) = [e for e in events if e.stage == "reversal"]
    assert cut.params["cut_frame"] == 30
    assert [e.tracks for e in events if e.stage == "dedup" and e.applied] == [[1, 3]]
    assert not [e for e in events if e.kind is EventKind.LINK and e.applied]
    assert out["track"].nunique() == 2
    assert (1, 0) in stages.reversal_sides.carried[1]
