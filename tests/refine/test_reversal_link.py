import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.link import run_link_stage, score_candidates
from dnt.refine.primitives import occlusion_flags
from dnt.refine.reversal import ReversalSides
from dnt.refine.verify import Band, route_without_vlm

from ._fixtures import random_tracks
from ._reversal import FPS, rows, var_height, work_of


def _cfg(mode="scored"):
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.mode = mode
    return cfg


def _cands(work, sides=None, cfg=None):
    cfg = cfg or _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, _ = score_candidates(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                                occluded=occ, reversal_sides=sides)
    return {(c.i, c.j): c for c in cands}


def _link(work, sides=None, cfg=None):
    cfg = cfg or _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    band = Band.of(cfg.link)
    return run_link_stage(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                          occluded=occ, route=lambda e: route_without_vlm(e, band),
                          reversal_sides=sides)


def _split_by_hand(cut, h_of=lambda f: 10.0, n=60):
    """Raw 1 out and back, split at ``cut`` into work tracks 1 and 2 (both raw 1)."""
    return work_of(rows(1, range(cut), h_of=h_of), rows(2, range(cut, n), h_of=h_of), raw={2: 1})


def test_the_isolated_regression_relinks_at_23_without_the_guard_and_not_with_it():
    work = _split_by_hand(23, var_height)  # the cut the script's argmin made
    res = _link(work)
    (ev,) = res.events
    assert ev.tracks == [1, 2] and ev.applied and ev.algo_score == pytest.approx(0.975)
    assert ev.signals["heading"] is None
    assert (1, 2) not in _cands(work, ReversalSides(cuts={1: [23]}))
    assert _link(work, ReversalSides(cuts={1: [23]})).events == []


def test_legacy_mode_relinks_a_clean_cut_without_the_guard_and_not_with_it():
    work = _split_by_hand(30)
    assert [e.tracks for e in _link(work, cfg=_cfg("legacy")).events] == [[1, 2]]
    assert _link(work, ReversalSides(cuts={1: [30]}), cfg=_cfg("legacy")).events == []


def test_the_sides_stay_apart_after_rows_next_to_the_cut_are_dropped():
    work = work_of(rows(1, range(0, 27)), rows(2, range(33, 60)), raw={2: 1})
    assert (1, 2) not in _cands(work, ReversalSides(cuts={1: [30]}))


def test_a_re_split_side_may_rejoin_itself_but_not_the_other_side():
    work = work_of(rows(1, range(0, 30)), rows(2, range(30, 45)), rows(3, range(45, 60)),
                   raw={2: 1, 3: 1})
    cands = _cands(work, ReversalSides(cuts={1: [30]}))
    assert (2, 3) in cands
    assert (1, 2) not in cands and (1, 3) not in cands


def test_no_part_of_a_twice_reversed_track_links_to_another():
    def x(f):
        return 100.0 + (2.0 * f if f <= 30 else 120.0 - 2.0 * f if f <= 70 else 2.0 * f - 160.0)

    work = work_of(rows(1, range(0, 30), x), rows(2, range(30, 70), x), rows(3, range(70, 110), x),
                   raw={2: 1, 3: 1})
    cands = _cands(work, ReversalSides(cuts={1: [30, 70]}))
    assert not {(1, 2), (2, 3), (1, 3)} & set(cands)


@pytest.mark.parametrize("sides", [None, ReversalSides()])
def test_link_is_unchanged_without_cuts(sides):
    work = io.to_work(random_tracks(seed=0)).work
    base = _cands(work)
    got = _cands(work, sides)
    assert sorted(base) == sorted(got)
    assert all(base[k].score == got[k].score for k in base)
