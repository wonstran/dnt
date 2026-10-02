import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance, CoarseArrayAppearance
from dnt.refine.switch import (
    _SILHOUETTE_MAX,
    _dense_rescore,
    _merge_dense,
    _relocate,
    _sides_ok,
    _silhouette,
    _span,
    propose_splits,
)

from ._fixtures import box_rows, random_tracks, table

FPS = 10.0
A_, B_ = np.eye(8)[0], np.eye(8)[1]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _gap_track(track=1):
    """Frames 0..119 minus frame 60 (the gap opens the gate near frame 61)."""
    return box_rows(track, [f for f in range(120) if f != 60], 100.0, 100.0, vx=2.0)


def _table(track=1, cut=64):
    frames = [f for f in range(120) if f != 60]
    return {track: (frames, np.array([A_ if f < cut else B_ for f in frames]))}


def test_dense_rescoring_moves_the_cut_to_the_true_switch_frame():
    work = _work(_gap_track())
    cfg = RefineConfig.defaults()
    dense = propose_splits(work, cfg, FPS, CoarseArrayAppearance(_table(), every=5))
    assert [e.params["cut_frame"] for e in dense.events] == [64]
    assert dense.candidates[1] == [64]
    ev = dense.events[0]
    assert ev.algo_score == pytest.approx(cfg.switch.w_app)  # app saturates, no motion break
    assert ev.signals["motion_only"] is False
    # the same coarse samples without dense access stay on the first frame of the plateau
    t = _table()
    f, e = t[1]
    coarse = propose_splits(work, cfg, FPS, ArrayAppearance({1: (f[::5], e[::5])}))
    assert [x.params["cut_frame"] for x in coarse.events] == [62]


def test_dense_samples_are_requested_only_around_candidates():
    class Recording(CoarseArrayAppearance):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.prefetched, self.dense_calls = [], []

        def prefetch_dense(self, windows):
            self.prefetched.append(list(windows))

        def dense_embeddings(self, raw_id, f0, f1):
            self.dense_calls.append((raw_id, f0, f1))
            return super().dense_embeddings(raw_id, f0, f1)

    quiet = box_rows(2, range(120), 400.0, 300.0, vx=1.0)
    table2 = {**_table(1), 2: (list(range(120)), np.tile(A_, (120, 1)))}
    app = Recording(table2, every=5)
    propose_splits(_work(_gap_track(1), quiet), RefineConfig.defaults(), FPS, app)
    assert len(app.prefetched) == 1 and app.prefetched[0]
    assert {w[0] for w in app.prefetched[0]} == {1}
    assert {c[0] for c in app.dense_calls} == {1}
    assert all(40 <= f0 <= f1 <= 80 for _, f0, f1 in app.prefetched[0])


def test_a_provider_that_is_dense_but_identical_changes_nothing():
    frames = list(range(100))
    rows = box_rows(1, range(50), 100.0, 100.0, vx=2.0, w=30.0) + box_rows(
        1, range(50, 100), 200.0, 100.0, vx=2.0, w=60.0
    )
    emb = np.array([A_ if f < 50 else B_ for f in frames])
    work = _work(rows)
    cfg = RefineConfig.defaults()
    plain = propose_splits(work, cfg, FPS, ArrayAppearance({1: (frames, emb)}))
    dense = propose_splits(work, cfg, FPS, CoarseArrayAppearance({1: (frames, emb)}, every=1))
    assert plain.events and len(plain.events) == len(dense.events)
    for a, b in zip(plain.events, dense.events, strict=True):
        assert a.params == b.params and a.algo_score == pytest.approx(b.algo_score, abs=1e-12)
        assert a.signals == pytest.approx(b.signals, abs=1e-12, nan_ok=True)


def _crafted(n, cuts, candidates, every=5):
    """A hand-built stage 1 info dict (gate open, no motion break) and its dense embeddings.

    The embedding is A before the first cut, B until the second, A again, and so on.
    """
    emb = np.tile(A_, (n, 1))
    state = 0
    for c in cuts:
        emb[c:] = B_ if state == 0 else A_
        state ^= 1
    frames = np.arange(n)
    info = {
        "frames": frames,
        "med": 0.0,
        "mad": 0.0,
        "bim": np.zeros(n),
        "fired": {"gap": np.ones(n, bool)},
        "mot": np.zeros(n),
        "A": np.zeros(n),
        "z": np.full(n, np.nan),
        "app": np.zeros(n),
        "S": np.zeros(n),
        "samples": frames[::every],
        "motion_only": False,
        "ef": frames[::every],
        "emb": emb[::every],
    }
    for c in candidates:
        info["S"][c] = 0.65
    return info, emb


def test_relocate_moves_to_the_best_eligible_row_and_stores_its_values():
    info, emb = _crafted(40, (12,), [10])
    sc = RefineConfig.defaults().switch
    ef = np.arange(40)
    assert _relocate(info, 10, sc, 5, 15, 5, ef, emb, lambda j: True) == 12
    assert info["S"][12] == pytest.approx(sc.w_app) and info["A"][12] == pytest.approx(1.0)

    info, emb = _crafted(40, (12,), [10])
    got = _relocate(info, 10, sc, 5, 15, 5, ef, emb, lambda j: j != 12)
    assert got in (11, 13) and info["S"][12] == 0.0  # the next best, and 12 was never stored

    info, emb = _crafted(40, (12,), [10])
    assert _relocate(info, 10, sc, 5, 15, 5, ef, emb, lambda j: j == 10) == 10
    assert info["S"][10] == pytest.approx(sc.w_app)  # rescored in place

    info, emb = _crafted(40, (12,), [10])
    empty = (np.empty(0, dtype=int), np.empty((0, 0)))
    assert _relocate(info, 10, sc, 5, 15, 5, *empty, lambda j: True) == 10  # no dense samples


def _rescored(cfg, n, cuts, candidates, nms, w=5, fps=10.0):
    info, emb = _crafted(n, cuts, candidates)
    app = CoarseArrayAppearance({1: (list(range(n)), emb)}, every=5)
    cands = {1: list(candidates)}
    _dense_rescore(app, {1: (info, [[1, 0, n - 1]])}, cands, cfg, w, fps, nms)
    return [int(info["frames"][i]) for i in cands[1]]


def test_a_move_cannot_leave_too_little_clean_data_on_a_side():
    cfg = RefineConfig.defaults()
    # coarse samples sit every 5 frames (0..40); the true cut is at 33. A cut at 33 leaves the
    # samples 35 and 40 on its right (2 x 5 frames = 1.0 s); a cut at 30 also keeps sample 30.
    cfg.switch.min_side_seconds = 0.5
    assert _rescored(cfg, 45, (33,), [30], nms=20) == [33]
    cfg.switch.min_side_seconds = 1.5  # needs three samples on the right: only 30 or earlier
    assert _rescored(cfg, 45, (33,), [30], nms=20) == [30]


def test_converging_candidates_keep_the_nms_spacing():
    cfg = RefineConfig.defaults()
    # cuts at 12 and 28 would pull the candidates at 10 and 30 to 16 frames apart
    assert _rescored(cfg, 60, (12, 28), [10, 30], nms=1) == [12, 28]
    kept = _rescored(cfg, 60, (12, 28), [10, 30], nms=20)
    assert len(kept) == 2 and kept[1] - kept[0] >= 20


def test_merge_dense_keeps_samples_sorted_and_unique():
    ef, emb = np.array([0, 5, 10]), np.eye(3)
    part_f, part_e = np.array([4, 5, 6]), np.array([[0.0, 1, 0], [0.0, 1, 0], [0, 0, 1.0]])
    f, e = _merge_dense(ef, emb, [(part_f, part_e), (np.empty(0, dtype=int), np.empty((0, 0)))])
    assert list(f) == [0, 4, 5, 6, 10] and e.shape == (5, 3)
    f2, e2 = _merge_dense(ef, emb, [])
    assert f2 is ef and e2 is emb


def test_silhouette_is_capped_and_equals_the_subsample():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal([1.0, 0.0], 0.05, (1500, 2)), rng.normal([0.0, 1.0], 0.05, (1500, 2))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.r_[np.zeros(1500, int), np.ones(1500, int)]
    pick = np.linspace(0, len(emb) - 1, _SILHOUETTE_MAX).astype(int)
    capped = _silhouette(emb, labels)
    assert capped > 0.9
    assert capped == _silhouette(emb[pick], labels[pick])


def test_candidates_are_moved_in_priority_order_not_frame_order():
    cfg = RefineConfig.defaults()
    # cuts at 12 and 19; candidates at 11 (weaker) and 21 (stronger); nms is 8 frames. The
    # stronger one moves first, to 19, which then keeps the weaker one off 12 (7 frames apart).
    info, emb = _crafted(60, (12, 19), [11, 21])
    info["S"][11] = 0.5
    info["S"][21] = 0.65
    app = CoarseArrayAppearance({1: (list(range(60)), emb)}, every=5)
    cands = {1: [11, 21]}
    _dense_rescore(app, {1: (info, [[1, 0, 59]])}, cands, cfg, 5, 10.0, 8)
    assert cands[1][1] == 19
    assert cands[1][1] - cands[1][0] >= 8


def test_dense_samples_are_merged_into_the_track_samples():
    cfg = RefineConfig.defaults()
    info, emb = _crafted(45, (33,), [30])
    app = CoarseArrayAppearance({1: (list(range(45)), emb)}, every=5)
    before = len(info["ef"])
    _dense_rescore(app, {1: (info, [[1, 0, 44]])}, {1: [30]}, cfg, 5, 10.0, 20)
    ef = info["ef"]
    assert len(ef) > before and 33 in ef and list(ef) == sorted(set(ef.tolist()))
    assert len(info["emb"]) == len(ef)


@pytest.mark.parametrize("gap", [12, 20, 40])
@pytest.mark.parametrize("every", [5, 1])
def test_a_switch_across_an_occlusion_gap_is_kept(gap, every):
    """The first row after a long gap has no dense 'before' window; its coarse score must count."""
    frames = [f for f in range(140) if not (56 <= f < 56 + gap)]
    work = _work(box_rows(1, frames, 100.0, 100.0, vx=2.0))
    rng = np.random.default_rng(gap)
    emb = np.array([A_ if f < 56 else B_ for f in frames]) + rng.normal(0, 0.05, (len(frames), 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cfg = RefineConfig.defaults()
    plain = propose_splits(work, cfg, FPS, ArrayAppearance({1: (frames, emb)}))
    dense = propose_splits(work, cfg, FPS, CoarseArrayAppearance({1: (frames, emb)}, every=every))
    assert plain.events and plain.events[0].params["cut_frame"] == 56 + gap
    got = [(e.params["cut_frame"], e.algo_score) for e in dense.events]
    assert got == pytest.approx([(e.params["cut_frame"], e.algo_score) for e in plain.events])
    assert dense.candidates[1] == plain.candidates[1]


@pytest.mark.parametrize("seed", range(15))
def test_dense_every_frame_matches_the_plain_provider_on_random_tracks(seed):
    rng = np.random.default_rng(seed)
    work = io.to_work(random_tracks(seed, n_objects=25, n_frames=300)).work
    tab = {}
    for rid, g in work.groupby("raw_id"):
        f = np.sort(g["frame"].to_numpy(int))
        e = rng.normal(size=8) + rng.normal(0, 0.5, (len(f), 8))
        if len(f) > 30 and rng.random() < 0.6:
            k = int(rng.integers(10, len(f) - 10))
            e[k:] = rng.normal(size=8) + rng.normal(0, 0.5, (len(f) - k, 8))
        tab[int(rid)] = (f, e)
    cfg = RefineConfig.defaults()
    a = propose_splits(work, cfg, FPS, ArrayAppearance(tab))
    b = propose_splits(work, cfg, FPS, CoarseArrayAppearance(tab, every=1))
    assert a.candidates == b.candidates and a.weak_cuts == b.weak_cuts
    got = [(e.params["cut_frame"], e.algo_score) for e in b.events]
    assert got == pytest.approx([(e.params["cut_frame"], e.algo_score) for e in a.events])


def test_relocate_without_a_baseline_returns_the_candidate_and_writes_nothing():
    info, emb = _crafted(40, (12,), [10])
    info["med"] = float("nan")
    sc = RefineConfig.defaults().switch
    assert _relocate(info, 10, sc, 5, 15, 5, np.arange(40), emb, lambda j: True) == 10
    assert info["S"][12] == 0.0 and info["A"][12] == 0.0


def test_relocate_keeps_the_candidate_row_when_it_has_no_dense_side_means():
    info, emb = _crafted(40, (12,), [10])
    info["A"][10] = 0.9
    sc = RefineConfig.defaults().switch
    ef = np.arange(40)
    # dense samples cover only the right of row 10, so its 'before' window is empty
    keep = ef >= 10
    got = _relocate(info, 10, sc, 5, 15, 5, ef[keep], emb[keep], lambda j: j in (10, 5))
    assert got == 10 and info["S"][10] == 0.65 and info["A"][10] == 0.9
    assert info["S"][5] == 0.0  # the other row was rescored to 0 and lost


def test_relocate_ties_go_to_the_nearest_then_the_earliest_row():
    info, emb = _crafted(40, (), [20])  # no change anywhere: every row scores the same
    sc = RefineConfig.defaults().switch
    ef = np.arange(40)
    assert _relocate(info, 20, sc, 15, 25, 5, ef, emb, lambda j: j in (17, 21)) == 21
    assert _relocate(info, 20, sc, 15, 25, 5, ef, emb, lambda j: j in (19, 21)) == 19


def test_span_starts_at_row_one_and_stops_at_the_last_row():
    info, _ = _crafted(40, (), [])
    assert _span(info, 2, 5, 5)[:2] == (1, 7)
    assert _span(info, 38, 5, 5)[:2] == (33, 39)


def test_sides_ok_checks_each_side_on_its_own():
    info, _ = _crafted(45, (), [])  # coarse samples at 0, 5, ..., 40
    assert _sides_ok(info, 22, 10.0, 1.0)
    assert not _sides_ok(info, 3, 10.0, 1.0)  # only the left side is short
    assert not _sides_ok(info, 43, 10.0, 1.0)  # only the right side is short


def test_relocate_returns_the_candidate_when_no_row_is_eligible():
    info, emb = _crafted(40, (12,), [10])
    sc = RefineConfig.defaults().switch
    assert _relocate(info, 10, sc, 5, 15, 5, np.arange(40), emb, lambda j: False) == 10
    assert info["S"][12] == 0.0


def test_relocate_counts_the_bimodal_split_score_of_a_row():
    info, emb = _crafted(40, (), [10])  # the embedding never changes
    info["bim"][13] = 1.0
    sc = RefineConfig.defaults().switch
    assert _relocate(info, 10, sc, 5, 15, 5, np.arange(40), emb, lambda j: True) == 13
    assert info["S"][13] == pytest.approx(sc.w_app)
