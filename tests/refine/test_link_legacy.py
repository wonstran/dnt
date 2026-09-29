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


_GATE_A = {
    "track": 1, "cls": 2, "t_end": 10, "end_c": (115.0, 130.0),
    "end_box": (100.0, 100.0, 30.0, 60.0), "area_end": 1800.0, "vx": 2.0, "vy": 0.0,
}
# dt = 5, so the predicted center is (125, 130) and the predicted box (110, 100, 30, 60).
# b's box (111, 106, 36, 54) is centered at (129, 133): dist = 5, width ratio 1.2, height ratio 0.9.
_GATE_B = {
    "track": 2, "cls": 2, "t_start": 15, "start_c": (129.0, 133.0),
    "start_box": (111.0, 106.0, 36.0, 54.0),
}
_GATE_KW = {
    "max_gap": 20, "size_ratio_max": 2.0, "dist_mult": 2.5, "iou_min": 0.05,
    "w_d": 1.0, "w_iou": 1.0, "w_s": 0.3,
}


def _gate(b_over=None, a_over=None, **kw_over):
    from dnt.refine.link import _legacy_gate_cost

    a = {**_GATE_A, **(a_over or {})}
    b = {**_GATE_B, **(b_over or {})}
    return _legacy_gate_cost(a, b, **{**_GATE_KW, **kw_over})


def _far(dx):
    """A start box the same size as the end box whose center sits dx right of the prediction."""
    cx = 125.0 + dx
    return {"start_c": (cx, 130.0), "start_box": (cx - 15.0, 100.0, 30.0, 60.0)}


def test_gate_cost_rejects_and_scores_like_the_original():
    # Hand computed: dist / (sqrt(1800) + 1e-6) = 5 / 42.4264 = 0.117851; the predicted box
    # (110, 100, 30, 60) meets (111, 106, 36, 54) in 29 * 54 = 1566 of 1800 + 1944 - 1566 = 2178
    # (iou 0.719008, 1 - iou = 0.280992); size 0.3 * (|ln 1.2| + |ln 0.9|) = 0.086305.
    expected = 0.117851 + 0.280992 + 0.086305
    assert _gate() == pytest.approx(expected, abs=1e-5)
    assert _gate() == pytest.approx(0.4851474846927045, abs=1e-9)
    # The weights scale their own term only.
    assert _gate(w_iou=0.0, w_s=0.0) == pytest.approx(0.117851, abs=1e-5)
    assert _gate(w_d=0.0, w_s=0.0) == pytest.approx(0.280992, abs=1e-5)
    assert _gate(w_d=0.0, w_iou=0.0) == pytest.approx(0.086305, abs=1e-5)
    assert _gate(w_d=2.0) == pytest.approx(expected + 0.117851, abs=1e-5)
    # detail=True reports the raw terms next to the same cost.
    cost, terms = _gate(detail=True)
    assert cost == pytest.approx(expected, abs=1e-5)
    assert set(terms) == {"dist", "iou_pred", "w_ratio", "h_ratio"}
    assert terms["dist"] == pytest.approx(5.0)
    assert terms["iou_pred"] == pytest.approx(1566.0 / 2178.0)
    assert terms["w_ratio"] == pytest.approx(1.2) and terms["h_ratio"] == pytest.approx(0.9)


def test_gate_cost_rejects_each_gate():
    assert _gate({"track": 1}) is None  # same track
    assert _gate({"t_start": 10}) is None and _gate({"t_start": 9}) is None  # dt < 1
    assert _gate({"t_start": 11}) is not None  # dt = 1 is fine
    assert _gate(max_gap=5) is not None and _gate(max_gap=4) is None  # dt > max_gap
    assert _gate({"cls": 7}) is None
    assert _gate({"cls": 7}, check_class=False) is not None
    wide = {"start_box": (111.0, 106.0, 61.0, 54.0)}  # width ratio 2.03
    assert _gate(wide) is None
    assert _gate({"start_box": (111.0, 106.0, 14.0, 54.0)}) is None  # width ratio 0.47
    assert _gate(wide, size_ratio_max=2.1) is not None
    assert _gate({"start_box": (111.0, 106.0, 36.0, 121.0)}) is None  # height ratio 2.02
    assert _gate({"start_box": (111.0, 106.0, 36.0, 29.0)}) is None  # height ratio 0.48


def test_gate_cost_distance_gate_and_dist_growth():
    # The limit is dist_mult * sqrt(area) * (1 + dist_growth * dt) = 2.5 * 42.43 * 1.15 = 121.98
    # at dt = 5, so a dist of 120 passes and 124 fails; iou_min = 0 isolates the distance gate.
    kw = {"iou_min": 0.0}
    assert _gate(_far(120.0), **kw) is not None
    assert _gate(_far(124.0), **kw) is None
    assert _gate(_far(120.0), dist_growth=0.0, **kw) is None  # limit 106.07
    assert _gate(_far(124.0), dist_growth=0.1, **kw) is not None  # limit 159.1
    assert _gate(_far(100.0), dist_growth=0.0, **kw) is not None
    assert _gate(_far(100.0), dist_growth=0.0, dist_mult=2.0, **kw) is None  # limit 84.85


def test_gate_cost_predicted_box_iou_gate():
    # 60 px from the prediction: inside the distance limit but the predicted box misses it.
    assert _gate(_far(60.0)) is None
    cost = _gate(_far(60.0), iou_min=0.0)
    assert cost == pytest.approx(60.0 / 42.4264 + 1.0, abs=1e-4)  # iou 0, equal size
    assert _gate(_far(30.0)) is None  # the boxes only touch: iou 0 < iou_min
    assert _gate(_far(10.0)) is not None  # iou = 1200 / (3600 - 1200) = 0.5
