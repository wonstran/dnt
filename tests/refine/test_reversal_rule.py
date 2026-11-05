import dataclasses
import warnings

import numpy as np
import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.reversal import (
    cut_in_run,
    find_reversals,
    hit_rows,
    keep_cuts,
    row_values,
    runs,
)

from ._reversal import FPS, boxes_at, out_and_back, path

RC = RefineConfig.defaults("vehicle").reversal


def cuts(frames, boxes, rc=RC, fps=FPS):
    return [r.row for r in find_reversals(frames, boxes, fps, rc)]


def test_a_clean_out_and_back_is_cut_at_the_turn_not_at_the_first_tied_row():
    t, cx = out_and_back()
    b = boxes_at(cx)
    assert hit_rows(t, b, FPS, RC).tolist() == list(range(23, 39))
    assert np.allclose(row_values(t, b, FPS, RC).cos[23:39], -1.0)  # every row: full reversal
    (rev,) = find_reversals(t, b, FPS, RC)
    assert (rev.row, rev.frame, rev.run) == (30, 30, (23, 38))
    assert rev.angle == pytest.approx(180.0)
    assert (rev.rows_before, rev.rows_after) == (30, 30)
    assert rev.speed_before == pytest.approx(2.0) and rev.speed_after == pytest.approx(2.0)


def test_noise_does_not_move_the_cut_but_moves_the_smallest_cosine():
    t, cx = out_and_back()
    cy = 100.0 + np.random.default_rng(0).normal(0.0, 0.5, 60)
    b = boxes_at(cx, cy)
    run = np.arange(23, 39)
    assert hit_rows(t, b, FPS, RC).tolist() == run.tolist()
    assert int(run[np.argmin(row_values(t, b, FPS, RC).cos[run])]) == 33  # the script's choice
    assert cuts(t, b) == [30]


def test_the_variable_height_fixture_is_cut_at_25():
    t, cx = out_and_back()
    h = np.where((t >= 15) & (t <= 30), 50.0, 10.0)
    b = boxes_at(cx, h=h)
    assert hit_rows(t, b, FPS, RC).tolist() == [23, 24, 25]
    assert cuts(t, b) == [25]


def test_near_the_start_the_stage_keeps_a_cut_the_script_drops():
    t, cx = out_and_back(n=30, turn=5)
    b = boxes_at(cx)
    assert hit_rows(t, b, FPS, RC).tolist() == list(range(3, 10))
    (rev,) = find_reversals(t, b, FPS, RC)
    assert (rev.row, rev.rows_before, rev.rows_after) == (5, 5, 25)


def test_near_the_end_the_cut_keeps_exactly_min_rows():
    t, cx = out_and_back(n=30, turn=25)
    b = boxes_at(cx)
    assert hit_rows(t, b, FPS, RC).tolist() == list(range(23, 28))
    assert cuts(t, b) == [25]


def test_two_reversals_give_two_cuts():
    t = np.arange(110)
    v = np.where(t <= 30, 2.0, np.where(t <= 70, -2.0, 2.0))
    b = boxes_at(path(v))
    assert [(int(r[0]), int(r[-1])) for r in runs(hit_rows(t, b, FPS, RC))] == [(23, 38), (63, 78)]
    assert cuts(t, b) == [30, 70]


def test_a_straight_track_with_a_gap_is_not_cut():
    frames = np.r_[0:40, 55:95]
    assert cuts(frames, boxes_at(100.0 + 2.0 * frames)) == []


def test_a_reversal_across_a_gap_is_cut_at_the_first_row_after_the_gap():
    frames = np.r_[0:30, 40:70]
    x = np.where(frames < 30, 2.0 * frames, 60.0 - 2.0 * (frames - 30))
    (rev,) = find_reversals(frames, boxes_at(100.0 + x), FPS, RC)
    assert (rev.row, rev.frame) == (30, 40)


def test_queue_jitter_is_not_cut_at_the_default_speed_but_is_at_0_2():
    t = np.arange(200)
    b = boxes_at(100.0 + 2.0 * np.sin(2.0 * np.pi * t / 40.0))
    assert cuts(t, b) == []
    assert cuts(t, b, dataclasses.replace(RC, min_speed=0.2))  # why the default is 0.5


@pytest.mark.parametrize(
    "name",
    ["stop_and_go", "right_angle", "parked"],
)
def test_motions_that_are_not_reversals_are_not_cut(name):
    if name == "stop_and_go":
        t = np.arange(160)
        b = boxes_at(path(np.tile(np.r_[np.full(20, 2.0), np.zeros(20)], 4)))
    elif name == "right_angle":
        t = np.arange(60)
        x = 100.0 + np.where(t < 30, 2.0 * t, 58.0)
        y = 100.0 - np.where(t < 30, 0.0, 2.0 * (t - 29))
        b = boxes_at(x, y)
    else:
        rng = np.random.default_rng(0)
        t = np.arange(100)
        b = boxes_at(100.0 + rng.uniform(-2, 2, 100), 100.0 + rng.uniform(-2, 2, 100))
    assert cuts(t, b) == []


def test_a_reversal_with_a_short_tail_has_hits_but_no_cut():
    t = np.arange(33)
    b = boxes_at(100.0 + np.where(t < 30, 2.0 * t, 58.0 - 2.0 * (t - 29)))
    assert len(hit_rows(t, b, FPS, RC)) > 0
    assert cuts(t, b) == []


def test_a_track_shorter_than_two_parts_has_no_hits():
    t = np.arange(9)
    b = boxes_at(100.0 + np.where(t < 5, 4.0 * t, 32.0 - 4.0 * t))
    assert hit_rows(t, b, FPS, RC).size == 0 and cuts(t, b) == []


def test_a_30_fps_out_and_back_is_cut_at_the_turn():
    t, cx = out_and_back(n=180, turn=90, speed=2.0 / 3.0)  # 2 box heights per second
    assert cuts(t, boxes_at(cx), fps=30.0) == [90]


def test_repeated_frames_give_no_division_warning():
    frames = np.r_[0, 0, 0, 0, 1:60]
    x = 100.0 + np.where(frames < 30, 2.0 * frames, 120.0 - 2.0 * frames)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (rev,) = find_reversals(frames, boxes_at(x), FPS, RC)
    assert rev.frame == 30


def test_keep_cuts_measures_each_part_from_the_previous_kept_cut():
    assert keep_cuts([3, 7, 20], 30, 5) == [7, 20]  # 3 leaves a 3-row head
    assert keep_cuts([10, 13, 20], 30, 5) == [10, 20]  # 13 leaves a 3-row middle
    assert keep_cuts([26], 30, 5) == []  # a 4-row tail
    assert keep_cuts([], 30, 5) == []


def test_cut_in_run_takes_the_earliest_of_tied_changes():
    change = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0 * (1 + 1e-12)])
    assert cut_in_run(np.array([4, 5, 6]), change) == 5
    assert cut_in_run(np.array([4, 5, 6]), np.array([0, 0, 0, 0, 1.0, 2.0, 3.0])) == 6
