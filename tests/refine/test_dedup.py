import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import propose_merges
from dnt.refine.events import EventKind
from dnt.refine.features import ArrayAppearance

from ._fixtures import table

FPS = 10.0


def _rows(track, frames, *, dx=0.0, cls=0, score=0.9, shift=None):
    """Rows of a 30 x 60 box on the path x = 100 + 3 * frame, plus ``dx``; ``shift`` per frame."""
    return [
        [f, track, 100.0 + 3.0 * f + dx + (shift or {}).get(f, 0.0), 100.0, 30.0, 60.0,
         score, cls, -1, -1]
        for f in frames
    ]


def _work(*row_lists):
    return io.to_work(table(*row_lists)).work


def _interleaved(n=120):
    return _work(_rows(1, range(0, n, 2)), _rows(2, range(1, n, 2)))


def _expected(fa, fb):
    lo, hi = max(min(fa), min(fb)), min(max(fa), max(fb))
    a = {f for f in fa if lo <= f <= hi}
    b = {f for f in fb if lo <= f <= hi}
    return len(a & b), len(a), len(b)


def test_an_interleaved_pair_becomes_one_merge_event_with_ordered_lineage():
    (ev,) = propose_merges(_interleaved(), RefineConfig.defaults(), FPS)
    assert ev.stage == "dedup" and ev.kind is EventKind.MERGE and ev.tracks == [1, 2]
    assert ev.lineage == [[[1, 0, 118]], [[2, 1, 119]]]
    assert ev.frames == (1, 118) and ev.params == {"span": [1, 118]}
    assert ev.algo_score == pytest.approx(1.0)
    s = ev.signals
    assert (s["shared"], s["n_a"], s["n_b"]) == (0, 59, 59)
    assert s["co_occupancy"] == 0.0 and s["comotion"] == pytest.approx(1.0)
    assert s["appearance"] is None


def test_the_evening_counts_score_above_accept():
    shared = [0, 30, 60, 90, 120, 150, 182]
    rest = [f for f in range(183) if f not in shared]
    a_only = [rest[i] for i in np.linspace(0, 175, 69).astype(int)]
    b_only = [f for f in rest if f not in set(a_only)][:100]
    fa, fb = sorted(shared + a_only), sorted(shared + b_only)
    work = _work(_rows(1, fa, shift={f: 300.0 for f in shared}), _rows(2, fb))
    cfg = RefineConfig.defaults()
    (ev,) = propose_merges(work, cfg, FPS)
    s = ev.signals
    assert (s["shared"], s["n_a"], s["n_b"]) == (7, 76, 107)
    assert s["co_occupancy"] == pytest.approx(7 / 76)
    assert ev.algo_score >= cfg.dedup.accept_above


@pytest.mark.parametrize("n_shared", [0, 1, 5, 12])
def test_co_occupancy_is_shared_over_the_sparser_track(n_shared):
    fa = list(range(0, 80, 2))
    fb = sorted(set(range(1, 80, 2)) | set(range(2, 2 + 2 * n_shared, 2)))
    (ev,) = propose_merges(_work(_rows(1, fa), _rows(2, fb)), RefineConfig.defaults(), FPS)
    shared, n_a, n_b = _expected(fa, fb)
    s = ev.signals
    assert shared == n_shared and (s["shared"], s["n_a"], s["n_b"]) == (shared, n_a, n_b)
    assert s["co_occupancy"] == pytest.approx(shared / min(n_a, n_b))


@pytest.mark.parametrize("dx", [200.0, 10.0, 1.6], ids=["apart", "overlapping", "duplicate"])
def test_densely_co_observed_pairs_are_never_candidates(dx):
    work = _work(_rows(1, range(60)), _rows(2, range(60), dx=dx))
    assert propose_merges(work, RefineConfig.defaults(), FPS) == []


def test_the_class_overlap_and_observation_gates():
    cfg = RefineConfig.defaults()
    mixed = _work(_rows(1, range(0, 120, 2), cls=0), _rows(2, range(1, 120, 2), cls=2))
    assert propose_merges(mixed, cfg, FPS) == []
    short = _work(_rows(1, range(0, 10, 2)), _rows(2, range(1, 10, 2)))
    cfg.dedup.min_observed = 1  # isolate the span gate: the overlap is 8 frames, under 1 s
    assert propose_merges(short, cfg, FPS) == []
    cfg.dedup.min_overlap_seconds = 0.5
    assert len(propose_merges(short, cfg, FPS)) == 1
    few = _work(_rows(1, range(0, 120, 2)), _rows(2, [1, 21, 41, 61, 81]))
    cfg = RefineConfig.defaults()  # the overlap is 81 frames; track 2 has 5 rows in it, under 8
    assert propose_merges(few, cfg, FPS) == []
    cfg.dedup.min_observed = 5
    assert len(propose_merges(few, cfg, FPS)) == 1


def test_vehicle_class_groups_merge_and_other_classes_do_not():
    cfg = RefineConfig.defaults("vehicle")
    car_truck = _work(_rows(1, range(0, 120, 2), cls=2), _rows(2, range(1, 120, 2), cls=7))
    assert len(propose_merges(car_truck, cfg, FPS)) == 1
    car_moto = _work(_rows(1, range(0, 120, 2), cls=2), _rows(2, range(1, 120, 2), cls=3))
    assert propose_merges(car_moto, cfg, FPS) == []


def test_disabled_empty_and_single_track_tables_give_no_events():
    cfg = RefineConfig.defaults()
    work = _interleaved()
    cfg.dedup.enabled = False
    assert propose_merges(work, cfg, FPS) == []
    cfg.dedup.enabled = True
    assert propose_merges(work.iloc[0:0], cfg, FPS) == []
    assert propose_merges(_work(_rows(1, range(40))), cfg, FPS) == []


def test_appearance_can_lower_a_score_but_not_veto_it():
    work = _interleaved()
    fa, fb = list(range(0, 120, 2)), list(range(1, 120, 2))
    same = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (60, 1))),
                            2: (fb, np.tile([1.0, 0.0], (60, 1)))})
    opposite = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (60, 1))),
                                2: (fb, np.tile([-1.0, 0.0], (60, 1)))})
    cfg = RefineConfig.defaults()
    (e_same,) = propose_merges(work, cfg, FPS, same)
    (e_opp,) = propose_merges(work, cfg, FPS, opposite)
    assert e_same.signals["appearance"] == pytest.approx(1.0)
    assert e_same.algo_score == pytest.approx(1.0)
    assert e_opp.signals["appearance"] == pytest.approx(-1.0)
    assert e_opp.algo_score == pytest.approx(cfg.dedup.appearance_floor)


def test_a_provider_without_one_tracks_embeddings_gives_a_motion_only_score():
    fa = list(range(0, 120, 2))
    only_one = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (60, 1)))})
    (ev,) = propose_merges(_interleaved(), RefineConfig.defaults(), FPS, only_one)
    assert ev.signals["appearance"] is None and ev.algo_score == pytest.approx(1.0)


def test_lineage_tracks_and_counts_correspond_and_the_key_ignores_work_ids():
    fa = list(range(0, 80, 2))  # 40 frames
    fb = [0, *range(1, 78, 2), 78]  # 41 frames, starts on the same frame as the other track

    def run(ids):
        work = _work(_rows(122, fa), _rows(125, fb))
        work["track"] = work["track"].map(ids)  # raw_id keeps 122 and 125
        (ev,) = propose_merges(work, RefineConfig.defaults(), FPS)
        return work, ev

    w1, e1 = run({122: 3, 125: 7})
    w2, e2 = run({122: 7, 125: 3})
    assert e1.proposal_key == e2.proposal_key
    assert e1.tracks == e2.tracks == [3, 7]  # the tie is broken by work ID, so the order flips
    assert e1.lineage != e2.lineage
    n = {122: 40, 125: 41}
    for work, ev in ((w1, e1), (w2, e2)):
        raws = [int(work.loc[work["track"] == t, "raw_id"].iloc[0]) for t in ev.tracks]
        assert [lin[0][0] for lin in ev.lineage] == raws
        assert [ev.signals["n_a"], ev.signals["n_b"]] == [n[r] for r in raws]


def _offset_alternating():
    n = 130
    fa, fb = list(range(0, n, 2)), list(range(1, n, 2))
    work = _work(_rows(1, fa), _rows(2, fb, dx=13.0))
    same = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (len(fa), 1))),
                            2: (fb, np.tile([1.0, 0.0], (len(fb), 1)))})
    return work, same


def test_an_alternating_look_alike_pair_with_offset_boxes_scores_above_accept():
    work, same = _offset_alternating()
    cfg = RefineConfig.defaults()
    (ev,) = propose_merges(work, cfg, FPS, same)
    assert 0.25 < ev.signals["comotion"] < 0.5
    assert ev.signals["interleave_relax"] == cfg.dedup.interleave_relax
    assert ev.algo_score >= cfg.dedup.accept_above


def test_the_interleave_relax_needs_appearance_evidence():
    work, _ = _offset_alternating()
    cfg = RefineConfig.defaults()
    (ev,) = propose_merges(work, cfg, FPS)
    assert ev.signals["interleave_relax"] == 0.0
    assert ev.algo_score < cfg.dedup.accept_above


def test_interleave_counts_owner_switches_not_a_single_handover():
    from dnt.refine.dedup import describe, interleave, overlap

    def sw(fa, fb):
        t = describe(_work(_rows(1, fa), _rows(2, fb)))
        return interleave(t[1], t[2], overlap(t[1], t[2]))

    assert sw(range(0, 20, 2), range(1, 21, 2)) == (17, 1.0)
    assert sw([0, 1, 2, 3, 10, 11, 12, 13], [4, 5, 6, 7, 14, 15])[0] == 1
