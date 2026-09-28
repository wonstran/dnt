import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance
from dnt.refine.switch import propose_splits

from ._fixtures import box_rows, table

FPS = 10.0
A_, B_ = np.eye(8)[0], np.eye(8)[1]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _emb(frames, change_at, before=A_, after=B_):
    frames = list(frames)
    return frames, np.array([before if f < change_at else after for f in frames])


def test_swap_with_contact_splits_both_tracks_and_boosts():
    frames = range(100)
    w = _work(box_rows(1, frames, 100.0, 100.0, vx=4.0), box_rows(2, frames, 500.0, 100.0, vx=-4.0))
    app = ArrayAppearance({1: _emb(frames, 50), 2: _emb(frames, 50, before=B_, after=A_)})
    res = propose_splits(w, RefineConfig.defaults(), FPS, app)
    cuts = {e.tracks[0]: e for e in res.events}
    assert set(cuts) == {1, 2}
    for t, other in ((1, 2), (2, 1)):
        ev = cuts[t]
        assert ev.params["cut_frame"] == 50
        assert ev.signals["swap_with"] == other and ev.signals["swap_boost"] == pytest.approx(0.2)
        assert ev.algo_score >= 0.85 - 1e-9
        assert "contact" in ev.signals["gate"]


def test_appearance_drift_without_gate_has_no_event():
    frames = list(range(100))
    emb = np.array([np.cos(f / 60) * A_ + np.sin(f / 60) * B_ for f in frames])
    w = _work(box_rows(1, frames, 100.0, 100.0, vx=2.0))
    res = propose_splits(w, RefineConfig.defaults(), FPS, ArrayAppearance({1: (frames, emb)}))
    assert res.events == []


def test_motion_only_jump_after_gap_is_capped():
    rows = box_rows(1, range(0, 40), 100.0, 100.0, vx=2.0) + box_rows(1, range(45, 80), 490.0,
                                                                        100.0, vx=2.0)
    res = propose_splits(_work(rows), RefineConfig.defaults(), FPS, None)
    assert len(res.events) == 1
    ev = res.events[0]
    assert ev.params["cut_frame"] == 45 and ev.algo_score == pytest.approx(0.7)
    assert ev.signals["motion_only"] is True and "gap" in ev.signals["gate"]


def test_short_sides_have_no_event():
    short = box_rows(1, range(0, 8), 100.0, 100.0, vx=2.0)  # 0.8 s < 2 x min_side: skipped
    short[5][2] += 300
    late = box_rows(2, [*range(0, 27), 28, 29], 100.0, 300.0, vx=2.0)
    for r in late:
        if r[0] >= 28:  # jump after a gap, but only 0.2 s after it
            r[2] += 300
    res = propose_splits(_work(short, late), RefineConfig.defaults(), FPS, None)
    assert res.events == [] and res.candidates == {}


def _takeover(class_gate=True):
    rows = []
    for f in range(780, 861):
        truck = f >= 821
        cls = 5 if f == 821 else 7 if (f == 818 or f > 821) else 2
        rows.append([f, 12, 200.0 + (f - 780), 87.0, 95.0 if truck else 53.0,
                     60.0 if truck else 56.0, 0.9, cls, -1, -1])
    cfg = RefineConfig.defaults("vehicle")
    cfg.switch.class_change_gate = class_gate
    app = ArrayAppearance({12: _emb(range(780, 861), 821)})
    return propose_splits(_work(rows), cfg, FPS, app)


@pytest.mark.parametrize("class_gate", [True, False])
def test_takeover_by_untracked_object_splits_at_821(class_gate):
    res = _takeover(class_gate)
    assert [e.params["cut_frame"] for e in res.events] == [821]
    ev = res.events[0]
    assert ev.algo_score >= 0.5
    assert "size_jump" in ev.signals["gate"]
    assert ("class_change" in ev.signals["gate"]) is class_gate


def test_class_flicker_alone_has_no_event():
    rows = box_rows(1, range(100), 100.0, 100.0, vx=2.0, cls=2)
    rows[50][7] = 7
    frames = range(100)
    res = propose_splits(_work(rows), RefineConfig.defaults("vehicle"), FPS,
                         ArrayAppearance({1: (list(frames), np.tile(A_, (100, 1)))}))
    assert res.events == []


def test_weak_candidate_becomes_a_weak_cut():
    rows = box_rows(1, [f for f in range(100) if f not in (49,)], 100.0, 100.0, vx=2.0)
    for r in rows:
        if r[0] >= 50:
            r[4] = 30.0 * np.exp(0.24)
    res = propose_splits(_work(rows), RefineConfig.defaults(), FPS, None)
    assert res.events == [] and res.weak_cuts == {1: [50]}


def test_single_row_and_two_row_tracks_do_not_crash():
    w = _work(box_rows(1, [5], 0.0, 0.0), box_rows(2, [5, 6], 50.0, 0.0))
    assert propose_splits(w, RefineConfig.defaults(), FPS, None).events == []
