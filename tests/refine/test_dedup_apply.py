import random

import pandas as pd
import pytest

from dnt.refine import apply as A
from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.events import Decision, Event, EventKind, Ledger
from dnt.refine.verify import Band, decide, route_without_vlm

from ._fixtures import random_tracks
from .test_dedup import FPS, _rows, _work

CFG = RefineConfig.defaults()


def _routed(work, cfg=CFG):
    evs = propose_merges(work, cfg, FPS)
    route_without_vlm(evs, Band.of(cfg.dedup))
    return evs


def _ev(work, a, b, score=0.9, decision=Decision.AUTO_ACCEPT, source="auto"):
    """A hand-made MERGE event between tracks ``a`` and ``b`` of ``work``."""
    la, lb = A.lineage(work, a), A.lineage(work, b)
    lo = max(la[0][1], lb[0][1])
    ev = Event.propose(
        stage="dedup", kind=EventKind.MERGE, tracks=[a, b], lineage=[la, lb],
        key_lineage=sorted([la, lb], key=lambda spans: tuple(spans[0])),
        frames=(lo, lo), params={"span": [lo, lo]}, algo_score=score,
    )
    ev.id = f"dedup-r0-{a}-{b}"
    decide(ev, decision, source=source)
    return ev


def test_an_applied_merge_drops_the_lower_score_rows_and_reports_them():
    # track 2 also sees frames 10 and 12 of track 1: 2 shared frames, under conflict_min_shared
    w = _work(_rows(1, range(0, 40, 2), score=0.9),
              _rows(2, sorted({*range(1, 40, 2), 10, 12}), score=0.8))
    ev = _ev(w, 1, 2)
    out = apply_merges(w, [ev], CFG)
    assert len(out.work) == 40 and out.work["track"].unique().tolist() == [1]
    assert ev.applied is True
    assert ev.signals["dropped_rows"] == 2
    assert ev.signals["dropped"] == [[2, 10, 10], [2, 12, 12]]
    assert ev.signals["merged_into"] == 1
    assert out.excluded == {2: {10, 12}}
    assert out.absorbed == {2: 1} and out.merged_reps == {1}
    assert out.counts == {"applied": 1, "redundant": 0, "conflict": 0, "dropped_rows": 2}


def test_a_fully_co_observed_hand_made_edge_is_refused_not_forced():
    w = _work(_rows(1, range(0, 20), score=0.9), _rows(2, range(10, 30), score=0.8))
    ev = _ev(w, 1, 2)
    out = apply_merges(w, [ev], CFG)
    assert ev.applied is False and ev.signals["skipped_reason"] == "conflict"
    assert ev.signals["conflicts_with"] == {"tracks": [1, 2], "why": "dense", "proposal_key": None}
    assert len(out.work) == 40 and out.absorbed == {} and out.counts["conflict"] == 1


def test_a_three_way_interleave_merges_into_the_earliest_track():
    w = _work(_rows(1, range(0, 120, 3)), _rows(2, range(1, 120, 3)), _rows(3, range(2, 120, 3)))
    evs = _routed(w)
    assert len(evs) == 3 and all(e.decision is Decision.AUTO_ACCEPT for e in evs)
    out = apply_merges(w, evs, CFG)
    assert len(out.work) == 120 and out.work["track"].unique().tolist() == [1]
    assert out.absorbed == {2: 1, 3: 1}
    assert out.counts["applied"] == 2 and out.counts["redundant"] == 1
    redundant = [e for e in evs if e.signals.get("skipped_reason") == "redundant"]
    assert len(redundant) == 1 and redundant[0].applied is False
    assert redundant[0].signals.get("dropped_rows", 0) == 0


def _cyclic():
    frames = {1: [*range(0, 120, 3), 60], 2: [*range(1, 120, 3), 60], 3: [*range(2, 120, 3), 60]}
    w = _work(*(_rows(t, sorted(set(fs))) for t, fs in frames.items()))
    return w, _routed(w)


def test_a_cyclic_group_is_attributed_once_whatever_the_proposal_order():
    results = []
    for seed in range(5):
        w, evs = _cyclic()
        random.Random(seed).shuffle(evs)
        out = apply_merges(w, evs, CFG)
        kept = out.work[out.work["frame"] == 60]
        assert len(kept) == 1 and int(kept["raw_id"].iloc[0]) == 1  # tie: the smaller raw ID wins
        assert sum(e.signals.get("dropped_rows", 0) for e in evs) == 2
        assert out.counts == {"applied": 2, "redundant": 1, "conflict": 0, "dropped_rows": 2}
        results.append((
            out.work.reset_index(drop=True),
            {e.proposal_key: (e.applied, e.signals.get("dropped_rows", 0)) for e in evs},
        ))
    first = results[0]
    for table_, per_event in results[1:]:
        pd.testing.assert_frame_equal(table_, first[0])
        assert per_event == first[1]


def _dense_pair():
    # tracks 1 and 3 are densely co-observed (a person and a ghost of the same person); track 2
    # interleaves with both, so A/B and B/C are both accepted and A/C has no event
    return _work(_rows(1, range(0, 60, 2)), _rows(2, range(1, 60, 2)), _rows(3, range(0, 60, 2)))


def test_a_densely_co_observed_pair_blocks_the_bridge_whatever_the_order():
    applied_keys = set()
    for seed in range(5):
        w = _dense_pair()
        evs = _routed(w)
        assert len(evs) == 2  # 1/3 is gated, so it has no event
        random.Random(seed).shuffle(evs)
        out = apply_merges(w, evs, CFG)
        (done,) = [e for e in evs if e.applied]
        (blocked,) = [e for e in evs if not e.applied]
        applied_keys.add(done.proposal_key)
        assert blocked.decision is Decision.AUTO_ACCEPT  # accepted, but not applied
        assert blocked.signals["skipped_reason"] == "conflict"
        assert blocked.signals["conflicts_with"] == {
            "tracks": [1, 3], "why": "dense", "proposal_key": None}
        track_of = out.work.groupby("raw_id")["track"].first()
        assert track_of[1] != track_of[3]
        assert out.counts["applied"] == 1 and out.counts["conflict"] == 1
    assert len(applied_keys) == 1  # the same edge wins under every proposal order


def _bridge(c_frames):
    # A = 0..49 (track 1), B = 50..99 (track 2), C = c_frames then 100..110 (track 3); the A/B and
    # B/C edges are accepted; A/C has no event but overlaps A on the first frames of c_frames
    w = _work(_rows(1, range(0, 50)), _rows(2, range(50, 100)),
              _rows(3, [*c_frames, *range(100, 111)], score=0.8))
    return w, [_ev(w, 1, 2, score=0.95), _ev(w, 2, 3, score=0.90)]


@pytest.mark.parametrize(
    "c_frames", [range(41, 50), range(43, 50)], ids=["below-span-gate", "below-min-observed"]
)
def test_dense_overlap_below_the_proposal_gates_still_blocks_a_bridge(c_frames):
    w, evs = _bridge(c_frames)
    out = apply_merges(w, evs, CFG)
    assert [e.applied for e in evs] == [True, False]
    assert evs[1].signals["conflicts_with"]["why"] == "dense"
    assert evs[1].signals["conflicts_with"]["tracks"] == [1, 3]
    assert out.absorbed == {2: 1}


def test_a_coincidence_on_a_couple_of_frames_does_not_block_a_bridge():
    w, evs = _bridge(range(48, 50))  # shared on 2 frames, fewer than conflict_min_shared (3)
    out = apply_merges(w, evs, CFG)
    assert [e.applied for e in evs] == [True, True]
    assert out.absorbed == {2: 1, 3: 1} and out.work["track"].unique().tolist() == [1]
    assert out.counts["dropped_rows"] == 2  # C's rows on 48 and 49 lose to A's higher score


def _chain(c_decision, c_source="auto"):
    w = _work(_rows(1, range(0, 20)), _rows(2, range(20, 40)), _rows(3, range(40, 60)))
    return w, [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.90),
               _ev(w, 1, 3, 0.50, decision=c_decision, source=c_source)]


@pytest.mark.parametrize(
    ("decision", "source"), [(Decision.VLM_REJECT, "vlm"), (Decision.HUMAN_REJECT, "human")]
)
def test_an_explicit_rejection_blocks_a_join(decision, source):
    w, evs = _chain(decision, source)
    apply_merges(w, evs, CFG)
    assert [e.applied for e in evs[:2]] == [True, False]
    why = evs[1].signals["conflicts_with"]
    assert why == {"tracks": [1, 3], "why": "rejected", "proposal_key": evs[2].proposal_key}


@pytest.mark.parametrize("decision", [Decision.AUTO_REJECT, Decision.HUMAN_PENDING])
def test_an_auto_reject_or_a_pending_pair_does_not_block_a_join(decision):
    w, evs = _chain(decision)
    out = apply_merges(w, evs, CFG)
    assert [e.applied for e in evs[:2]] == [True, True]
    assert out.absorbed == {2: 1, 3: 1}


def test_different_classes_block_a_join():
    w = _work(_rows(1, range(0, 20), cls=0), _rows(2, range(20, 40), cls=0),
              _rows(3, range(40, 60), cls=2))
    evs = [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.90)]
    apply_merges(w, evs, CFG)
    assert [e.applied for e in evs] == [True, False]
    assert evs[1].signals["conflicts_with"] == {"tracks": [1, 3], "why": "class",
                                                "proposal_key": None}
    CFG_GROUP = RefineConfig.defaults()
    CFG_GROUP.link.class_groups = [[0, 2]]
    evs = [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.90)]
    apply_merges(w, evs, CFG_GROUP)
    assert [e.applied for e in evs] == [True, True]


def test_pending_endpoints_and_merged_reps_use_the_representatives():
    w = _work(_rows(1, range(0, 20)), _rows(2, range(20, 40)), _rows(3, range(40, 60)))
    evs = [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.60, decision=Decision.HUMAN_PENDING)]
    out = apply_merges(w, evs, CFG)
    assert out.merged_reps == {1} and out.absorbed == {2: 1}
    assert out.pending_endpoints == {1, 3}  # track 2 is now track 1
    assert evs[1].applied is False and "skipped_reason" not in evs[1].signals


def test_nothing_to_do_leaves_the_table_alone():
    w = _work(_rows(1, range(0, 20)))
    out = apply_merges(w, [], CFG)
    assert out.work is w or out.work.equals(w)
    assert out.merged_reps == set() and out.absorbed == {} and out.excluded == {}
    empty = apply_merges(w.iloc[0:0], [], CFG)
    assert empty.work.empty and empty.counts["applied"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_tracks_keep_every_observation_accounted_for(seed):
    w = io.to_work(random_tracks(seed=seed)).work
    evs = _routed(w)
    out = apply_merges(w, evs, CFG)
    assert not out.work.duplicated(["track", "frame"]).any()
    original = set(zip(w["raw_id"], w["frame"], strict=True))
    kept = set(zip(out.work["raw_id"], out.work["frame"], strict=True))
    dropped = {
        (raw, f)
        for e in evs
        for raw, a, b in e.signals.get("dropped", [])
        for f in range(a, b + 1)
    }
    assert kept | dropped == original and not (kept & dropped)
    assert len(out.work) == len(w) - sum(e.signals.get("dropped_rows", 0) for e in evs)


def test_skipped_and_applied_events_round_trip_through_the_ledger(tmp_path):
    w = _dense_pair()
    evs = _routed(w)
    apply_merges(w, evs, CFG)
    Ledger({"format": "x"}, evs).write(tmp_path / "l.jsonl")
    back = Ledger.read(tmp_path / "l.jsonl").events
    assert [e.to_dict() for e in back] == [e.to_dict() for e in evs]
