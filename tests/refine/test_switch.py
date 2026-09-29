import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance
from dnt.refine.switch import _bimodal_split, propose_splits

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


# ---- fix round 1: spec cluster-recall bimodality, amount-based sides, real gate tests -------


def _step_rows(width_ratio=1.0, flicker=False, n=100, at=50):
    rows = box_rows(1, range(n), 100.0, 100.0, vx=2.0, cls=2)
    for r in rows:
        if r[0] >= at:
            r[4] = 30.0 * width_ratio
    if flicker:
        rows[at][7] = 7
    return rows


def _step_app(n=100, at=50):
    return ArrayAppearance({1: _emb(range(n), at)})


def test_appearance_step_with_closed_gate_has_no_event():
    res = propose_splits(_work(_step_rows()), RefineConfig.defaults("vehicle"), FPS, _step_app())
    assert res.events == [] and res.candidates == {}


def test_class_flicker_opens_the_gate_for_an_appearance_step():
    cfg = RefineConfig.defaults("vehicle")
    res = propose_splits(_work(_step_rows(flicker=True)), cfg, FPS, _step_app())
    assert [e.params["cut_frame"] for e in res.events] == [50]
    assert "class_change" in res.events[0].signals["gate"]
    cfg.switch.class_change_gate = False
    res = propose_splits(_work(_step_rows(flicker=True)), cfg, FPS, _step_app())
    assert res.events == []


def test_size_jump_gate_threshold():
    cfg = RefineConfig.defaults()
    below = propose_splits(_work(_step_rows(width_ratio=1.4)), cfg, FPS, _step_app())
    assert below.events == []
    above = propose_splits(_work(_step_rows(width_ratio=1.6)), cfg, FPS, _step_app())
    assert [e.params["cut_frame"] for e in above.events] == [50]
    assert "size_jump" in above.events[0].signals["gate"]


def _repeat(vec, n):
    return [vec] * n


def _bimodal_events(labels_vecs, cut):
    n = len(labels_vecs)
    rows = box_rows(1, range(n), 100.0, 100.0, vx=2.0, cls=2)
    rows[cut][7] = 7  # the only gate condition: a class flicker at the cut
    app = ArrayAppearance({1: (list(range(n)), np.array(labels_vecs))})
    return propose_splits(_work(rows), RefineConfig.defaults("vehicle"), FPS, app)


def test_bimodal_uses_cluster_recall_in_time():
    seq = _repeat(A_, 95) + _repeat(B_, 10) + _repeat(A_, 5)
    res = _bimodal_events(seq, 95)
    assert [e.params["cut_frame"] for e in res.events] == [95]
    assert res.events[0].signals["bimodal"] > 0


def test_interleaved_clusters_are_not_bimodal():
    seq = [A_ if f % 2 == 0 else B_ for f in range(110)]
    frames = np.arange(110)
    assert _bimodal_split(frames, np.array(seq), 0.9) is None
    res = _bimodal_events(seq, 55)
    assert all(e.signals["bimodal"] == 0 for e in res.events)


def test_sparse_samples_over_a_long_span_are_skipped_by_amount():
    # 4 observed rows spread over 2.5 s: the amount (0.4 s) is under 2 x min_side_seconds
    rows = box_rows(1, [0, 8, 16, 24], 100.0, 100.0, vx=2.0)
    for r in rows:
        if r[0] >= 16:
            r[2] += 300
    res = propose_splits(_work(rows), RefineConfig.defaults(), FPS, None)
    assert res.events == [] and res.candidates == {}


def test_skip_rule_counts_sample_amount_with_the_track_stride():
    # Clean samples at 0, 5, 6..10 (median stride 1): 0.7 s < 2 x 0.5 s, although each side of
    # the jump at frame 6 alone (1.0 s and 0.5 s) would pass the side rule.
    frames = [0, 5, 6, 7, 8, 9, 10]
    rows = box_rows(1, frames, 100.0, 100.0, vx=2.0)
    for r in rows:
        if r[0] >= 6:
            r[2] += 300
    app = ArrayAppearance({1: (frames, np.tile(A_, (len(frames), 1)))})
    res = propose_splits(_work(rows), RefineConfig.defaults(), FPS, app)
    assert res.events == [] and res.candidates == {}


def _jump_at(frames_before, frames_after):
    rows = box_rows(1, [*frames_before, *frames_after], 100.0, 100.0, vx=2.0)
    cut = frames_after[0]
    for r in rows:
        if r[0] >= cut:
            r[2] += 300
    return _work(rows), cut


@pytest.mark.parametrize(("n_before", "n_after", "kept"), [
    (5, 36, True),   # 0.5 s before (boundary): kept
    (4, 36, False),  # 0.4 s before: discarded although the after side is long
    (36, 5, True),   # 0.5 s after (boundary): kept
    (36, 4, False),  # 0.4 s after: discarded although the before side is long
])
def test_each_side_needs_min_side_seconds_of_data(n_before, n_after, kept):
    before = list(range(n_before))
    after = list(range(n_before + 1, n_before + 1 + n_after))  # one-frame gap opens the gate
    w, cut = _jump_at(before, after)
    res = propose_splits(w, RefineConfig.defaults(), FPS, None)
    assert [e.params["cut_frame"] for e in res.events] == ([cut] if kept else [])
    assert res.candidates == ({1: [cut]} if kept else {})


def _swap_tracks(emb1, emb2, ids=(1, 2)):
    frames = range(100)
    w = _work(box_rows(ids[0], frames, 100.0, 100.0, vx=4.0),
              box_rows(ids[1], frames, 500.0, 100.0, vx=-4.0))
    app = ArrayAppearance({ids[0]: _emb(frames, 50, *emb1), ids[1]: _emb(frames, 50, *emb2)})
    return propose_splits(w, RefineConfig.defaults(), FPS, app)


@pytest.mark.parametrize("emb2", [(A_, B_), (np.eye(8)[2], np.eye(8)[3])])
def test_non_crossing_contacting_pair_gets_no_swap_boost(emb2):
    res = _swap_tracks((A_, B_), emb2)
    assert {e.tracks[0] for e in res.events} == {1, 2}
    assert all("swap_with" not in e.signals for e in res.events)


@pytest.mark.parametrize("ids", [(1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 1, 2), (2, 3, 1)])
def test_strongest_swap_pair_wins_regardless_of_track_order(ids):
    x, strong, weak = ids
    e2, e3 = np.eye(8)[2], np.eye(8)[3]
    weak_after = (e3 + 0.3 * A_) / np.linalg.norm(e3 + 0.3 * A_)
    frames = range(100)
    w = _work(box_rows(x, frames, 100.0, 100.0, vx=4.0),
              box_rows(strong, frames, 500.0, 100.0, vx=-4.0),
              box_rows(weak, frames, 500.0, 130.0, vx=-4.0))
    app = ArrayAppearance({x: _emb(frames, 50), strong: _emb(frames, 50, B_, A_),
                           weak: _emb(frames, 50, e2, weak_after)})
    res = propose_splits(w, RefineConfig.defaults(), FPS, app)
    ev = {e.tracks[0]: e for e in res.events}
    assert ev[x].signals["swap_with"] == strong
    assert ev[x].signals["swap_boost"] == pytest.approx(0.2)
