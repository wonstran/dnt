import pytest

from dnt.track.post_process import interpolate_tracks_rts as shim_interpolate

from ._fixtures import DATA, load_raw


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("smooth", "name"), [(False, "interp"), (True, "interp_smooth")])
def test_matches_pre_move_baseline(tmp_path, seed, smooth, name):
    out = tmp_path / "o.csv"
    shim_interpolate(load_raw(seed), output_file=str(out), smooth_existing=smooth, verbose=False)
    assert out.read_bytes() == (DATA / f"{name}_{seed}.csv").read_bytes()


def test_shim_and_package_are_the_same_function():
    from dnt.refine.interpolate import interpolate_tracks_rts
    from dnt.track import interpolate_tracks_rts as track_level

    assert shim_interpolate is interpolate_tracks_rts is track_level
