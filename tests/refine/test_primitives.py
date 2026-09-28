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
    ctx = pd.DataFrame({"frame": [1], "track": [-1], "x": [1.0], "y": [0.0], "w": [10.0],
                        "h": [10.0], "cls": [2]})
    flags = P.occlusion_flags(work, ctx, 0.3)
    assert flags.tolist() == [True, True, True]
    assert P.occlusion_flags(work, None, 0.3).tolist() == [True, True, False]
