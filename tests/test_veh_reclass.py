"""VehReClass: helpers (Task 5), VLM layer (Task 6) and the class (Task 7)."""

import math

import numpy as np
import pandas as pd
import pytest
from reclass_fakes import DET_FIELDS

from dnt.track import re_class
from dnt.track.re_class import BUS, CAR, TRUCK


def _sizes(values):
    return pd.DataFrame({"frame": range(len(values)), "w": values, "h": [1] * len(values)})


# ---- Task 5: size ----

@pytest.mark.parametrize("n_rows", [1, 2, 5, 10, 20])
def test_one_outlier_never_reaches_hv_size(n_rows):
    sizes = [100] * (n_rows - 1) + [1000]
    s = re_class._size(_sizes(sizes), 3)
    assert not s >= 500  # NaN for short tracks, 100 otherwise


def test_size_counts_k_rows_at_threshold():
    assert re_class._size(_sizes([500, 500, 500, 100]), 3) == 500
    assert math.isnan(re_class._size(_sizes([900, 900]), 3))


def test_size_uses_longer_side():
    rows = pd.DataFrame({"frame": [0, 1, 2], "w": [10, 10, 10], "h": [600, 700, 800]})
    assert re_class._size(rows, 3) == 600


# ---- Task 5: sampling ----

def _ramp(n):
    return pd.DataFrame({"frame": range(n), "x": 0, "y": 0, "w": range(1, n + 1), "h": 10})


def test_sample_largest_first_with_gap():
    assert [f for f, _ in re_class._spaced_sample(_ramp(20), 10, 5)] == [19, 14, 9, 4]


def test_sample_min_gap_zero_is_top_n():
    assert [f for f, _ in re_class._spaced_sample(_ramp(20), 3, 0)] == [19, 18, 17]


def test_sample_ties_go_to_earlier_frame():
    rows = pd.DataFrame({"frame": [9, 1, 5], "x": 0, "y": 0, "w": 5, "h": 5})
    assert re_class._spaced_sample(rows, 1, 0) == [(1, (0.0, 0.0, 5.0, 5.0))]


def test_sample_never_repeats_a_frame():
    rows = pd.DataFrame({"frame": [3, 3, 4], "x": 0, "y": 0, "w": [9, 8, 1], "h": 1})
    assert [f for f, _ in re_class._spaced_sample(rows, 5, 0)] == [3, 4]


def test_sample_short_and_empty():
    assert len(re_class._spaced_sample(_ramp(2), 10, 0)) == 2
    assert re_class._spaced_sample(_ramp(0), 10, 5) == []


# ---- Task 5: IoU and vote ----

def test_iou_basic_and_zero_area_is_zero():
    got = re_class._iou((0, 0, 10, 10), np.array([[0, 0, 10, 10], [5, 0, 10, 10], [0, 0, 0, 0]]))
    assert got.tolist() == pytest.approx([1.0, 50 / 150, 0.0])


def _dets(frame, *boxes):
    return pd.DataFrame(
        [{"frame": frame, "res": -1, "x": x, "y": y, "w": w, "h": h, "conf": c, "class": k}
         for x, y, w, h, c, k in boxes],
        columns=DET_FIELDS,
    )


BOX = (100.0, 100.0, 200.0, 100.0)


def test_vote_spec_example():
    boxes = {f: BOX for f in range(10)}
    by = {f: _dets(f, (100, 100, 200, 100, 0.8, BUS)) for f in range(4)}
    by |= {f: _dets(f, (100, 100, 200, 100, 0.6, TRUCK)) for f in (4, 5)}
    v = re_class._vote(boxes, by, (CAR, BUS, TRUCK), 0.5)
    assert (v.cls, v.support, v.n) == (BUS, pytest.approx(0.4), 10)


def test_vote_confidence_sum_beats_vote_count():
    # 4 bus votes at 0.3 (sum 1.2) lose to 3 truck votes at 0.9 (sum 2.7): a count-only
    # winner would pick bus
    boxes = {f: BOX for f in range(10)}
    by = {f: _dets(f, (100, 100, 200, 100, 0.3, BUS)) for f in range(4)}
    by |= {f: _dets(f, (100, 100, 200, 100, 0.9, TRUCK)) for f in (4, 5, 6)}
    v = re_class._vote(boxes, by, (CAR, BUS, TRUCK), 0.5)
    assert (v.cls, v.support, v.n) == (TRUCK, pytest.approx(0.3), 10)


def test_vote_equal_score_more_votes_wins():
    # truck: 2 x 0.45 = 0.9; bus: 1 x 0.9 = 0.9 (exactly equal in binary floating point).
    # Truck has more votes but the higher class ID, so this fails an implementation that
    # skips the vote-count tie break and goes straight to the lower class ID.
    by = {0: _dets(0, (100, 100, 200, 100, 0.45, TRUCK)),
          1: _dets(1, (100, 100, 200, 100, 0.45, TRUCK)),
          2: _dets(2, (100, 100, 200, 100, 0.9, BUS))}
    assert re_class._vote({f: BOX for f in range(3)}, by, (CAR, BUS, TRUCK), 0.5).cls == TRUCK


def test_vote_equal_score_and_votes_lower_class_wins():
    by = {0: _dets(0, (100, 100, 200, 100, 0.9, TRUCK)), 1: _dets(1, (100, 100, 200, 100, 0.9, BUS))}
    assert re_class._vote({0: BOX, 1: BOX}, by, (CAR, BUS, TRUCK), 0.5).cls == BUS
    by = {0: _dets(0, (100, 100, 200, 100, 0.9, BUS)), 1: _dets(1, (100, 100, 200, 100, 0.9, CAR))}
    assert re_class._vote({0: BOX, 1: BOX}, by, (CAR, BUS, TRUCK), 0.5).cls == CAR


def test_vote_one_stray_has_low_support():
    boxes = {f: BOX for f in range(10)}
    v = re_class._vote(boxes, {3: _dets(3, (100, 100, 200, 100, 0.99, BUS))}, (CAR, BUS, TRUCK), 0.5)
    assert v.cls == BUS and v.support == pytest.approx(0.1)


def test_vote_highest_conf_wins_frame_and_iou_gate():
    by = {0: _dets(0, (100, 100, 200, 100, 0.6, BUS), (100, 100, 200, 100, 0.8, TRUCK),
                   (0, 0, 20, 20, 0.99, CAR))}  # the car box does not overlap: ignored
    v = re_class._vote({0: BOX}, by, (CAR, BUS, TRUCK), 0.5)
    assert (v.cls, v.support) == (TRUCK, 1.0)


def test_vote_heavy_ignores_car():
    by = {0: _dets(0, (100, 100, 200, 100, 0.9, CAR))}
    v = re_class._vote({0: BOX}, by, (BUS, TRUCK), 0.5)
    assert v.cls is None and v.support == 0.0


def test_vote_no_frames():
    v = re_class._vote({}, {}, (CAR, BUS, TRUCK), 0.5)
    assert v.cls is None and v.n == 0 and math.isnan(v.support)
