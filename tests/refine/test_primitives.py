import numpy as np
import pandas as pd
import pytest

from dnt.refine import primitives as P


def test_ramp_increasing_decreasing_and_clipped():
    assert P.ramp(0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert P.ramp(2.0, 0.0, 1.0) == 1.0
    assert P.ramp(0.2, 0.3, 0.05) == pytest.approx(0.4)  # decreasing ramp
    np.testing.assert_allclose(P.ramp(np.array([-1.0, 0.25, 9.0]), 0.0, 0.5), [0.0, 0.5, 1.0])


def test_cv_kalman_matches_interpolate_model():
    kf = P.cv_kalman(10.0, 25.0, 16.0)
    assert kf.F.shape == (8, 8) and kf.F[0, 1] == 1 and kf.F[1, 1] == 1
    assert kf.H[0, 0] == 1 and kf.H[1, 2] == 1 and kf.H[2, 4] == 1 and kf.H[3, 6] == 1
    np.testing.assert_allclose(np.diag(kf.R), [25.0, 25.0, 16.0, 16.0])
    np.testing.assert_allclose(np.diag(kf.P), np.full(8, 100.0))


def test_nis_is_chi2_4_on_model_consistent_simulation():
    rng = np.random.default_rng(3)
    kf = P.cv_kalman(10.0, 25.0, 16.0)
    x = np.array([500.0, 2.0, 300.0, -1.0, 60.0, 0.0, 120.0, 0.0])
    boxes = []
    for _ in range(3000):
        x = kf.F @ x + rng.multivariate_normal(np.zeros(8), kf.Q)
        z = kf.H @ x + rng.multivariate_normal(np.zeros(4), kf.R)
        boxes.append([z[0] - z[2] / 2, z[1] - z[3] / 2, z[2], z[3]])
    nis = P.kalman_nis(np.arange(3000), np.array(boxes))
    assert np.nanmean(nis[100:]) == pytest.approx(4.0, abs=0.4)


def test_nis_spikes_on_a_jump_after_a_gap():
    frames = np.r_[np.arange(0, 40), np.arange(45, 80)]
    x = 100 + 2.0 * frames + np.where(frames >= 45, 300.0, 0.0)
    boxes = np.column_stack([x, np.full(len(frames), 100.0), np.full(len(frames), 30.0),
                             np.full(len(frames), 60.0)])
    nis = P.kalman_nis(frames, boxes)
    i45 = int(np.flatnonzero(frames == 45)[0])
    assert nis[i45] > 18.47
    assert np.nanmax(nis[5:i45]) < 5.0


def test_speed_in_box_heights_per_second():
    frames = np.arange(10)
    boxes = np.column_stack([2.0 * frames, np.zeros(10), np.full(10, 30.0), np.full(10, 60.0)])
    v = P.speeds_hps(frames, boxes, fps=30.0)
    assert np.isnan(v[0])
    np.testing.assert_allclose(v[1:], 1.0)  # 2 px/frame * 30 fps / 60 px
    assert P.span_speed(frames, boxes, 30.0, 0.2, at="end") == pytest.approx(1.0)


def test_heading_smoothness_straight_vs_zigzag():
    f = np.arange(20)
    straight = np.column_stack([10.0 * f, np.zeros(20), np.full(20, 20.0), np.full(20, 40.0)])
    zig = straight.copy()
    zig[:, 1] = np.where(f % 2 == 0, 0.0, 30.0)
    zig[:, 0] = 0.0
    assert P.heading_smoothness(f, straight, 10.0) == pytest.approx(1.0)
    assert P.heading_smoothness(f, zig, 10.0) < 0.2


def test_majority_class_tie_goes_to_latest():
    assert P.majority_class([2, 2, 7, 7]) == 7
    assert P.majority_class([2, 2, 2, 7]) == 2
    assert P.majority_class([]) == -1


def test_iou_and_iob_matrices():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 5.0, 5.0], [0.0, 0.0, 20.0, 20.0]])
    iou = P.iou_matrix(a, b)
    assert iou.shape == (1, 3) and iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == 0.0
    assert P.iob_matrix(a, b)[0, 2] == pytest.approx(1.0)  # a fully inside the third box
    assert P.iou_matrix(np.empty((0, 4)), b).shape == (0, 3)


def test_frame_runs():
    assert P.frame_runs([5, 1, 2, 3, 7, 8]) == [(1, 3), (5, 5), (7, 8)]
    assert P.frame_runs([]) == []


def test_occlusion_flags_use_same_frame_boxes_and_context():
    work = pd.DataFrame({"frame": [0, 0, 1], "track": [1, 2, 1],
                         "x": [0.0, 2.0, 0.0], "y": [0.0, 0.0, 0.0],
                         "w": [10.0, 10.0, 10.0], "h": [10.0, 10.0, 10.0]})
    # the context box overlaps frame 1's row with IoU 0.38: below CONTEXT_MATCH_IOU, so it is
    # another object, not the row's own detection
    ctx = pd.DataFrame({"frame": [1], "track": [-1], "x": [5.0], "y": [0.0], "w": [10.0],
                        "h": [10.0], "cls": [2]})
    assert 0.3 <= P.iou_matrix([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 10.0, 10.0])[0, 0] < 0.5
    flags = P.occlusion_flags(work, ctx, 0.3)
    assert flags.tolist() == [True, True, True]
    assert P.occlusion_flags(work, None, 0.3).tolist() == [True, True, False]


def test_a_same_run_detection_context_does_not_flag_the_rows_own_boxes():
    # final-review I3 / R22: two pedestrians far apart, and a detection file of the same run
    # (their own boxes, jittered by a pixel) as context -> no row is occluded
    rng = np.random.default_rng(0)
    n = 40
    work = pd.DataFrame({"frame": np.repeat(np.arange(n), 2), "track": np.tile([1, 2], n),
                         "x": np.tile([100.0, 600.0], n) + np.repeat(np.arange(n), 2) * 2.0,
                         "y": 200.0, "w": 30.0, "h": 60.0})
    dets = work.assign(track=-1, cls=0, x=work["x"] + rng.uniform(-1, 1, len(work)),
                       y=work["y"] + rng.uniform(-1, 1, len(work)))
    ious = [P.iou_matrix(a, b)[0, 0] for a, b in zip(
        work[["x", "y", "w", "h"]].to_numpy()[:, None], dets[["x", "y", "w", "h"]].to_numpy()[:, None],
        strict=True)]
    assert min(ious) >= P.CONTEXT_MATCH_IOU  # every detection matches its row
    assert not P.occlusion_flags(work, dets, 0.3).any()  # 0% occluded (was 100%)
    assert P.context_duplicates(work, dets).all()
    assert P.drop_context_duplicates(work, dets).empty
    # a genuine occluder (a second box over row 10, which has its own detection) still flags it
    other = pd.DataFrame({"frame": [5], "track": [-1], "x": [100.0 + 10.0 + 8.0], "y": [200.0],
                          "w": [30.0], "h": [60.0], "cls": [0]})
    assert 0.3 <= P.iou_matrix(work.loc[10, ["x", "y", "w", "h"]].to_numpy(float),
                               other[["x", "y", "w", "h"]].to_numpy(float))[0, 0] < 0.9
    both = pd.concat([dets, other], ignore_index=True)
    flags = P.occlusion_flags(work, both, 0.3)
    assert flags[flags].index.tolist() == [10]
    assert P.context_duplicates(work, both).tolist() == [True] * len(dets) + [False]


# ---- final review I2 (ruling R6): own detections are matched one to one, at IoU >= 0.5 --------

_COLS = ["frame", "x", "y", "w", "h"]


def _boxes(*rows):
    """``(frame, x, y, w, h)`` rows; every box is 40x40 unless given."""
    return pd.DataFrame([r if len(r) == 5 else (*r, 40.0, 40.0) for r in rows],
                        columns=_COLS, dtype=float).astype({"frame": int})


def _iou(a, b):
    return float(P.iou_matrix(np.asarray(a, float)[1:], np.asarray(b, float)[1:])[0, 0])


def test_an_own_detection_below_the_old_cutoff_does_not_flag_its_row():
    row, det = (0, 0.0, 0.0), (0, 7.0, 0.0)  # a tracker box that lags its detection
    assert 0.5 <= _iou((*row, 40, 40), (*det, 40, 40)) < 0.9  # IoU 0.71
    work, ctx = _boxes(row), _boxes(det)
    assert P.occlusion_flags(work, ctx, 0.3).tolist() == [False]
    assert P.context_duplicates(work, ctx).tolist() == [True]
    assert P.drop_context_duplicates(work, ctx).empty


def test_a_different_box_below_the_match_iou_still_flags():
    row, other = (0, 0.0, 0.0), (0, 17.0, 0.0)
    assert 0.3 <= _iou((*row, 40, 40), (*other, 40, 40)) < 0.5  # IoU 0.41
    work, ctx = _boxes(row), _boxes(other)
    assert P.occlusion_flags(work, ctx, 0.3).tolist() == [True]
    assert P.context_duplicates(work, ctx).tolist() == [False]
    assert len(P.drop_context_duplicates(work, ctx)) == 1


def test_one_to_one_keeps_a_second_box_over_a_row_as_an_occluder():
    # the row's own detection (IoU 0.9) and another box (IoU 0.6): only the better one is own
    row, own, other = (0, 0.0, 0.0), (0, 2.0, 0.0), (0, 10.0, 0.0)
    assert _iou((*row, 40, 40), (*own, 40, 40)) >= 0.9
    assert 0.5 <= _iou((*row, 40, 40), (*other, 40, 40)) < 0.7
    work = _boxes(row)
    for ctx, own_mask in ((_boxes(own, other), [True, False]), (_boxes(other, own), [False, True])):
        assert P.context_duplicates(work, ctx).tolist() == own_mask
        assert P.occlusion_flags(work, ctx, 0.3).tolist() == [True]
        assert P.drop_context_duplicates(work, ctx)["x"].tolist() == [10.0]


def test_two_rows_and_two_detections_are_matched_crosswise():
    # row A's best box is d1, but the assignment with the largest total IoU gives d1 to row B
    # and d2 to row A, and both pairs reach 0.5; a per-row best match would leave d2 unmatched
    a, b, d1, d2 = (0, 0.0, 0.0), (0, -8.0, 0.0), (0, -2.0, 0.0), (0, 13.0, 0.0)
    m = {(r, d): _iou((*r, 40, 40), (*d, 40, 40)) for r in (a, b) for d in (d1, d2)}
    assert m[a, d1] > m[a, d2] >= 0.5 and m[b, d1] >= 0.5 > m[b, d2]
    assert m[a, d2] + m[b, d1] > m[a, d1] + m[b, d2]
    work, ctx = _boxes(a, b), _boxes(d1, d2)
    assert P.context_duplicates(work, ctx).tolist() == [True, True]
    assert P.drop_context_duplicates(work, ctx).empty


def test_own_detection_matching_with_empty_frames_and_tables():
    work = _boxes((0, 0.0, 0.0), (2, 0.0, 0.0))
    ctx = _boxes((1, 0.0, 0.0), (2, 1.0, 0.0), (3, 0.0, 0.0))  # frames 1 and 3 have no rows
    assert P.context_duplicates(work, ctx).tolist() == [False, True, False]
    assert P.occlusion_flags(work, ctx, 0.3).tolist() == [False, False]
    assert P.context_duplicates(work, None).shape == (0,)
    assert P.context_duplicates(work, ctx.iloc[:0]).shape == (0,)
    assert P.drop_context_duplicates(work, None) is None
    assert P.occlusion_flags(work, ctx.iloc[:0], 0.3).tolist() == [False, False]
    empty = work.iloc[:0]
    assert P.context_duplicates(empty, ctx).tolist() == [False, False, False]
    assert P.occlusion_flags(empty, ctx, 0.3).empty


def test_own_detection_matching_with_more_boxes_than_rows_and_more_rows_than_boxes():
    # one row, three boxes: its own detection, an occluder at IoU 0.6, and a box far away
    work = _boxes((0, 0.0, 0.0))
    ctx = _boxes((0, 10.0, 0.0), (0, 300.0, 0.0), (0, 1.0, 0.0))
    assert P.context_duplicates(work, ctx).tolist() == [False, False, True]
    assert P.occlusion_flags(work, ctx, 0.3).tolist() == [True]
    # three rows far apart, one box: it is only the nearest row's own detection
    work = _boxes((0, 0.0, 0.0), (0, 200.0, 0.0), (0, 400.0, 0.0))
    ctx = _boxes((0, 203.0, 0.0))
    assert P.context_duplicates(work, ctx).tolist() == [True]
    assert P.occlusion_flags(work, ctx, 0.3).tolist() == [False, False, False]
    # the same box far from every row is nobody's detection and occludes nobody
    assert P.context_duplicates(work, _boxes((0, 100.0, 0.0))).tolist() == [False]


def test_own_detection_matching_survives_coordinates_that_overflow():
    # x + w overflows to inf, so the IoU is NaN: the assignment must not reject the matrix
    for big in (1e200, 1e300, 1e308):
        work = _boxes((0, big, 0.0, big, 10.0))
        ctx = _boxes((0, big, 0.0, big, 10.0))
        with np.errstate(all="ignore"):
            assert P.context_duplicates(work, ctx).shape == (1,)
            assert P.drop_context_duplicates(work, ctx) is not None
            assert P.occlusion_flags(work, ctx, 0.3).shape == (1,)
