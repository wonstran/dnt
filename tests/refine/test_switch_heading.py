"""Heading reversal: a vehicle cannot turn around between two frames, so it counts as a motion break."""

import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance
from dnt.refine.switch import _heading_reversal, propose_splits

from ._fixtures import box_rows, table

FPS = 10.0
A_, B_ = np.eye(8)[0], np.eye(8)[1]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _emb(frames, change_at):
    frames = list(frames)
    return frames, np.array([A_ if f < change_at else B_ for f in frames])


def _centers(xs, ys=None, w=20.0, h=10.0):
    """``boxes`` rows (x, y, w, h) whose centres are ``xs``, ``ys``."""
    xs = np.asarray(xs, dtype=float)
    ys = np.zeros_like(xs) if ys is None else np.asarray(ys, dtype=float)
    return np.column_stack([xs - w / 2, ys - h / 2, np.full_like(xs, w), np.full_like(xs, h)])


def _rev(frames, boxes, w=10, min_speed=10.0, both=False):
    out = _heading_reversal(np.asarray(frames), boxes, w, FPS, min_speed)
    return out if both else out[0]


# ---- the signal: -cos of the angle between the mean velocities before and after a row ----------


def test_an_out_and_back_track_reverses_at_the_turning_row():
    frames = np.arange(80)
    x = 100.0 + 3.0 * np.minimum(frames, 40) - 3.0 * np.maximum(frames - 40, 0)
    rev = _rev(frames, _centers(x))
    assert rev[40] == pytest.approx(1.0)  # +3 px/frame before, -3 px/frame after
    assert np.nanmax(np.delete(rev, range(36, 45))) < 0.5  # far from the turn: no reversal


def test_the_velocity_change_is_largest_at_the_turn():
    frames = np.arange(80)
    x = 100.0 + 3.0 * np.minimum(frames, 40) - 3.0 * np.maximum(frames - 40, 0)
    rev, change = _rev(frames, _centers(x), both=True)
    assert change[40] == pytest.approx(6.0)  # +3 -> -3 px/frame
    assert np.nanargmax(change) == 40
    assert (
        rev[38] == pytest.approx(1.0) and change[38] < change[40]
    )  # the same -cos, a smaller change


def test_a_straight_track_has_no_reversal_anywhere():
    frames = np.arange(60)
    rev = _rev(frames, _centers(100.0 + 3.0 * frames, 50.0 + 1.0 * frames))
    assert np.nanmax(rev) == pytest.approx(-1.0)
    assert np.isfinite(rev[10:50]).all()


def test_a_right_angle_turn_is_not_a_reversal():
    frames = np.arange(80)
    x = 100.0 + 3.0 * np.minimum(frames, 40)
    y = 100.0 + 3.0 * np.maximum(frames - 40, 0)
    assert _rev(frames, _centers(x, y))[40] == pytest.approx(0.0, abs=0.1)


def test_a_slow_reversal_is_undefined():
    # 0.5 px/frame is 5 px/s at 10 fps: below min_speed (10 px/s), so it is jitter, not heading
    frames = np.arange(80)
    x = 100.0 + 0.5 * np.minimum(frames, 40) - 0.5 * np.maximum(frames - 40, 0)
    assert np.isnan(_rev(frames, _centers(x))[40])
    assert _rev(frames, _centers(x), min_speed=4.0)[40] == pytest.approx(
        1.0
    )  # same data, lower bar


def test_a_stationary_track_is_undefined():
    frames = np.arange(60)
    assert np.isnan(_rev(frames, _centers(np.full(60, 100.0))[:, :])).all()


def test_rows_with_fewer_than_two_samples_on_a_side_are_undefined():
    frames = np.arange(20)
    rev = _rev(frames, _centers(100.0 + 3.0 * frames))
    assert np.isnan(rev[:2]).all() and np.isnan(rev[-1])  # rows 0-1 and the last row
    assert np.isfinite(rev[2:-1]).all()


def test_velocity_uses_frame_numbers_across_gaps():
    # every other frame observed: the same 3 px/frame, so the same reversal
    frames = np.arange(0, 160, 2)
    x = 100.0 + 3.0 * np.minimum(frames, 80) - 3.0 * np.maximum(frames - 80, 0)
    rev = _rev(frames, _centers(x), w=20)
    assert rev[40] == pytest.approx(1.0)  # frame 80


# ---- the score: a reversal is a motion break, but only where the opportunity gate is open ------


def _there_and_back(gap=True, vx=3.0):
    """One track that goes right, then left from where it stopped; a frame is missing at the cut."""
    cut = 50
    turn = 100.0 + vx * (cut - 1)  # where the first part ends
    start = cut + 1 if gap else cut  # the first frame of the second part
    first = box_rows(1, range(0, cut), 100.0, 100.0, vx=vx)
    return first + box_rows(1, range(start, 100), turn, 100.0, vx=-vx), start


def _events(rows, cut, appearance=True, cfg=None):
    frames = sorted({r[0] for r in rows})
    app = ArrayAppearance({1: _emb(frames, cut)}) if appearance else None
    res = propose_splits(_work(rows), cfg or RefineConfig.defaults("vehicle"), FPS, app)
    return {e.params["cut_frame"]: e for e in res.events}


def test_a_reversal_with_an_appearance_step_after_a_gap_is_accepted():
    rows, cut = _there_and_back()
    ev = _events(rows, cut)[cut]
    assert ev.signals["heading"] == pytest.approx(1.0) and ev.signals["heading_cos"] < -0.9
    assert ev.signals["mot"] == pytest.approx(1.0) and "gap" in ev.signals["gate"]
    assert ev.algo_score >= 0.9  # w_app * app + w_mot * 1


def test_without_the_heading_signal_the_same_track_stays_below_accept():
    rows, cut = _there_and_back()
    cfg = RefineConfig.defaults("vehicle")
    cfg.switch.ramps["heading"] = [2.0, 3.0]  # -cos never exceeds 1: the signal is off
    ev = _events(rows, cut, cfg=cfg)[cut]
    assert ev.signals["heading"] == 0.0 and ev.signals["mot"] < 0.5
    assert ev.algo_score < 0.9


def test_a_reversal_alone_is_not_a_split_without_a_gate():
    rows, cut = _there_and_back(gap=False)  # no gap, no contact, no size jump, same class
    assert _events(rows, cut) == {}


def test_a_straight_track_with_an_appearance_step_has_no_heading_boost():
    rows = box_rows(1, range(0, 50), 100.0, 100.0, vx=3.0) + box_rows(
        1, range(51, 100), 250.0, 100.0, vx=3.0
    )
    ev = _events(rows, 51)[51]
    assert ev.signals["heading"] == 0.0 and ev.signals["heading_cos"] > 0.9
    assert ev.algo_score < 0.9


def test_a_slow_reversal_does_not_count():
    rows, cut = _there_and_back(vx=0.5)
    ev = _events(rows, cut).get(cut)
    assert ev is None or (ev.signals["heading"] == 0.0 and ev.signals["heading_cos"] is None)


def test_motion_only_reversal_is_capped_and_left_pending():
    rows, cut = _there_and_back()
    events = _events(rows, cut, appearance=False)
    assert list(events) == [cut]  # at the turn, not at the first row of the saturated plateau
    ev = events[cut]
    assert ev.signals["motion_only"] is True and ev.signals["heading"] == pytest.approx(1.0)
    assert ev.algo_score == pytest.approx(0.7)


# ---- config -----------------------------------------------------------------------------------


def test_heading_reversal_is_on_for_vehicles_and_off_for_people_by_default():
    assert RefineConfig.defaults("vehicle").switch.heading_reversal is True
    assert RefineConfig.defaults("person").switch.heading_reversal is False
    assert RefineConfig.defaults().switch.heading_reversal is False
    # a saved config of either target overlays onto its own target's default
    assert RefineConfig.from_dict({"target": "vehicle"}).switch.heading_reversal is True
    assert RefineConfig.from_dict({"target": "person"}).switch.heading_reversal is False
    on = RefineConfig.from_dict({"target": "person", "switch": {"heading_reversal": True}})
    off = RefineConfig.from_dict({"target": "vehicle", "switch": {"heading_reversal": False}})
    assert on.switch.heading_reversal is True and off.switch.heading_reversal is False


def test_a_person_run_ignores_a_reversal_unless_it_is_turned_on():
    rows, cut = _there_and_back()
    person = RefineConfig.defaults("person")
    ev = _events(rows, cut, cfg=person)[cut]
    assert ev.signals["heading"] == 0.0 and ev.signals["heading_cos"] is None
    assert ev.algo_score < 0.9  # the appearance step alone, as before
    person.switch.heading_reversal = True
    assert _events(rows, cut, cfg=person)[cut].algo_score >= 0.9


def test_heading_settings_have_defaults_and_overlay_onto_them():
    sc = RefineConfig.defaults().switch
    assert sc.heading_min_speed == 10.0 and sc.ramps["heading"] == [0.3, 0.8]
    cfg = RefineConfig.from_dict({"switch": {"ramps": {"z_app": [1.0, 4.0]}}})  # an older YAML
    assert cfg.switch.ramps["heading"] == [0.3, 0.8] and cfg.switch.ramps["z_app"] == [1.0, 4.0]
    cfg = RefineConfig.from_dict(
        {"switch": {"heading_min_speed": 30, "ramps": {"heading": [0.0, 0.7]}}}
    )
    assert cfg.switch.heading_min_speed == 30 and cfg.switch.ramps["heading"] == [0.0, 0.7]


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_heading_min_speed_must_be_a_finite_number_of_at_least_zero(bad):
    cfg = RefineConfig.defaults()
    cfg.switch.heading_min_speed = bad
    with pytest.raises(ValueError, match=r"switch\.heading_min_speed"):
        cfg.validate()
