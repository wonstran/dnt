import pytest

from dnt.track.post_process import link_tracklets as shim_link

from ._fixtures import DATA, load_raw


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pre_move_baseline(tmp_path, seed):
    out = tmp_path / "o.csv"
    shim_link(load_raw(seed), output_file=str(out), max_gap=20, verbose=False)
    assert out.read_bytes() == (DATA / f"link_{seed}.csv").read_bytes()


def test_shim_and_package_are_the_same_function():
    from dnt.refine.link import link_tracklets
    from dnt.track import link_tracklets as track_level

    assert shim_link is link_tracklets is track_level


def test_gate_cost_rejects_and_scores_like_the_original():
    from dnt.refine.link import _legacy_gate_cost

    a = {
        "track": 1, "cls": 2, "t_end": 10, "end_c": (115.0, 130.0),
        "end_box": (100.0, 100.0, 30.0, 60.0), "area_end": 1800.0, "vx": 2.0, "vy": 0.0,
    }
    b = {
        "track": 2, "cls": 2, "t_start": 15, "start_c": (125.0, 130.0),
        "start_box": (110.0, 100.0, 30.0, 60.0),
    }
    kw = {
        "max_gap": 20, "size_ratio_max": 2.0, "dist_mult": 2.5, "iou_min": 0.05,
        "w_d": 1.0, "w_iou": 1.0, "w_s": 0.3,
    }
    assert _legacy_gate_cost(a, b, **kw) == pytest.approx(0.0, abs=1e-6)
    assert _legacy_gate_cost(a, {**b, "cls": 7}, **kw) is None
    assert _legacy_gate_cost(a, {**b, "cls": 7}, check_class=False, **kw) is not None
    assert _legacy_gate_cost(a, {**b, "t_start": 40}, **kw) is None
    _cost, terms = _legacy_gate_cost(a, b, detail=True, **kw)
    assert set(terms) == {"dist", "iou_pred", "w_ratio", "h_ratio"}
