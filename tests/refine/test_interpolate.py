import numpy as np
import pandas as pd
import pytest

from dnt.refine.interpolate import interpolate_tracks_rts
from dnt.track.post_process import interpolate_tracks_rts as shim_interpolate

from ._fixtures import DATA, box_rows, load_raw, table


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("smooth", "name"), [(False, "interp"), (True, "interp_smooth")])
def test_matches_pre_move_baseline(tmp_path, seed, smooth, name):
    out = tmp_path / "o.csv"
    shim_interpolate(load_raw(seed), output_file=str(out), smooth_existing=smooth, verbose=False)
    assert out.read_bytes() == (DATA / f"{name}_{seed}.csv").read_bytes()


def test_shim_and_package_are_the_same_function():
    from dnt.track import interpolate_tracks_rts as track_level

    assert shim_interpolate is interpolate_tracks_rts is track_level


def _positional(df):
    return df.set_axis(range(df.shape[1]), axis=1)


def test_filled_rows_are_not_measurements_and_are_re_estimated():
    raw = table(box_rows(1, [f for f in range(10) if f != 5], 10.0, 20.0, vx=2.0))
    fake = raw.iloc[[0]].copy()
    fake[["frame", "x", "r3"]] = [5, 500.0, 1]  # a filled row at an absurd position
    out = interpolate_tracks_rts(_positional(pd.concat([raw, fake])), verbose=False)
    row5 = out[out["frame"] == 5].iloc[0]
    assert row5["interp"] == 1
    assert abs(row5["x"] - 20) <= 2  # re-estimated near 10 + 2*5, not 500


def test_filled_rows_outside_fillable_gaps_are_dropped():
    raw = table(box_rows(1, range(0, 5), 10.0, 20.0), box_rows(1, range(20, 25), 10.0, 20.0))
    filled = table(box_rows(1, range(5, 20), 10.0, 20.0))
    filled["r3"] = 1
    out = interpolate_tracks_rts(_positional(pd.concat([raw, filled])), max_gap=2, verbose=False)
    assert sorted(out["frame"]) == [*range(0, 5), *range(20, 25)]


def test_protected_gap_is_never_filled_and_smoothing_does_not_cross():
    raw = table(
        box_rows(1, range(0, 10), 100.0, 300.0, vy=-5.0),
        box_rows(1, range(73, 83), 60.0, 240.0, vx=-5.0),
    )
    kw = {"max_gap": 100, "verbose": False}
    filled = interpolate_tracks_rts(raw.copy(), **kw)
    assert set(range(10, 73)) <= set(filled["frame"])
    kept = interpolate_tracks_rts(raw.copy(), protected_gaps={1: [(9, 73)]}, **kw)
    assert not (set(range(10, 73)) & set(kept["frame"]))
    s_all = interpolate_tracks_rts(
        raw.copy(), protected_gaps={1: [(9, 73)]}, smooth_existing=True, **kw
    )
    s_head = interpolate_tracks_rts(raw[raw["frame"] < 10].copy(), smooth_existing=True, **kw)
    cols = ["frame", "x", "y", "w", "h"]
    pd.testing.assert_frame_equal(
        s_all[s_all["frame"] < 10][cols].reset_index(drop=True),
        s_head[cols].reset_index(drop=True),
    )


def test_two_protected_gaps_in_one_chain():
    raw = table(
        box_rows(1, range(0, 5), 0.0, 0.0),
        box_rows(1, range(20, 25), 0.0, 0.0),
        box_rows(1, range(40, 45), 0.0, 0.0),
    )
    out = interpolate_tracks_rts(
        raw, max_gap=100, protected_gaps={1: [(4, 20), (24, 40)]}, verbose=False
    )
    assert sorted(out["frame"]) == [*range(0, 5), *range(20, 25), *range(40, 45)]
    assert np.all(out["interp"] == 0)
