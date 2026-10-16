"""VehReClass: helpers (Task 5), VLM layer (Task 6) and the class (Task 7)."""

import json
import logging
import math

import numpy as np
import pandas as pd
import pytest
from reclass_fakes import DET_FIELDS, FakeDetector, FakeReader, vehicle_rows

from dnt.refine.config import VLMConfig
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import VLMRunner
from dnt.shared.files import read_track
from dnt.track import re_class
from dnt.track.re_class import BUS, CAR, TRUCK, VehReClass


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


# ---- Task 6: VLM layer ----


def _vlm(tmp_path, **kw):
    base = dict(backend="openai_compat", model="m", base_url="http://localhost:9/v1",
                cache_dir=str(tmp_path / "cache"))
    return VLMConfig(**(base | kw))


def _answer(ans, conf=0.9):
    return json.dumps({"answer": ans, "confidence": conf, "reason": "r"})


def test_tile_is_jpeg_and_none_without_crops():
    img = re_class._tile([np.zeros((40, 80, 3), np.uint8), np.zeros((20, 10, 3), np.uint8)])
    assert img[:2] == b"\xff\xd8"
    assert re_class._tile([]) is None


def test_tile_of_prescaled_crops_is_byte_identical():
    rng = np.random.default_rng(0)
    crops = [
        rng.integers(0, 255, (40, 80, 3), dtype=np.uint8),
        rng.integers(0, 255, (333, 17, 3), dtype=np.uint8),
        rng.integers(0, 255, (160, 90, 3), dtype=np.uint8),  # already tile height
        rng.integers(0, 255, (5, 400, 3), dtype=np.uint8),
    ]
    scaled = [re_class._fit_height(c) for c in crops]
    assert all(c.shape[0] == re_class._TILE_HEIGHT for c in scaled)
    assert re_class._tile(scaled) == re_class._tile(crops)


def test_prepare_vlm_resolves_endpoint_and_checks_it(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    seen = []
    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: seen.append(cfg.backend))
    vlm = VLMConfig(cache_dir=str(tmp_path), use="local", endpoints={
        "local": {"backend": "openai_compat", "model": "q", "base_url": "http://localhost:9/v1"}})
    cfg = re_class._prepare_vlm(vlm, None, check=True)
    assert (cfg.backend, cfg.model, seen) == ("openai_compat", "q", ["openai_compat"])
    assert vlm.use == "local"  # the caller's object is unchanged


@pytest.mark.parametrize("vlm", [
    VLMConfig(use="nope"),
    VLMConfig(backend="openai_compat", model="m", votes=0),
    VLMConfig(),  # resolves to backend none
])
def test_prepare_vlm_rejects(vlm):
    with pytest.raises(ValueError):
        re_class._prepare_vlm(vlm, None, check=False)


def test_detect_frames_warns_once_about_unread_frames(caplog):
    reader = FakeReader("v.mp4", bad={5, 10})
    with caplog.at_level(logging.WARNING, logger=re_class.log.name):
        by_frame, unread = re_class._detect_frames(FakeDetector(), reader, [0, 5, 10, 15])
    assert sorted(by_frame) == [0, 15] and unread == {5, 10}
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "2 of 4" in msgs[0] and "v.mp4" in msgs[0]


def test_detect_frames_silent_when_all_read(caplog):
    with caplog.at_level(logging.WARNING, logger=re_class.log.name):
        re_class._detect_frames(FakeDetector(), FakeReader("v.mp4"), [0, 5])
    assert caplog.records == []


def test_ask_vlm_warns_once_about_unreadable_crop_frames(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad={15, 20}))
    backend = FakeBackend({"*": _answer("bus")})
    with caplog.at_level(logging.WARNING, logger=re_class.log.name):
        re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", _requests(1))
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "2 of 3" in msgs[0] and "v.mp4" in msgs[0]


def test_prepare_vlm_missing_anthropic_key(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VehReClass") as e:
        re_class._prepare_vlm(VLMConfig(backend="anthropic", cache_dir=str(tmp_path)), None, check=True)
    msg = str(e.value)
    assert "ANTHROPIC_API_KEY" in msg and "vlm_api_key" in msg and "vlm.api_key_file" in msg
    assert "TrackRefiner" not in msg and "--vlm" not in msg


def test_prepare_vlm_missing_openai_key_names_local_server(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = VLMConfig(backend="openai_compat", model="m", api_key_env="MY_KEY",
                    cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="VehReClass") as e:
        re_class._prepare_vlm(cfg, None, check=True)
    assert "MY_KEY" in str(e.value) and "vlm.base_url" in str(e.value)


def _requests(*tracks, heavy=False):
    box = (100.0, 100.0, 300.0, 150.0)
    return {t: ([(10 * t + i * 5, box) for i in range(3)], heavy) for t in tracks}


def test_ask_vlm_answers_and_options(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    backend = FakeBackend({"vehclass:1": _answer("bus"), "vehclass:2": _answer("truck")})
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4",
                            _requests(1) | _requests(2, heavy=True))
    assert got[1] == re_class._VLMResult("bus", 0.9, None)
    assert got[2].answer == "truck"
    opts = {c["tag"]: c["options"] for c in backend.calls}
    assert opts == {"vehclass:1": ["car", "bus", "truck", "unclear"],
                    "vehclass:2": ["bus", "truck", "unclear"]}
    assert backend.closed == 1


def test_ask_vlm_keeps_original_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    backend = FakeBackend({"vehclass:1": _answer("bus"), "vehclass:2": _answer("truck")})
    reqs = {1.0: _requests(1)[1], 2: _requests(2)[2]}
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", reqs)
    assert set(got) == {1.0, 2} and got[1.0].answer == "bus" and got[2].answer == "truck"
    assert sorted(c["tag"] for c in backend.calls) == ["vehclass:1", "vehclass:2"]


def test_tag_id():
    assert [re_class._tag_id(v) for v in (1, 1.0, np.int64(3), np.float64(4.0), 2.5)] == [
        "1", "1", "3", "4", "2.5"]


def test_partial_crop_failure_still_asks(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad={15}))
    backend = FakeBackend({"*": _answer("bus")})
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", _requests(1))
    assert got[1].answer == "bus" and len(backend.calls) == 1


def test_track_without_usable_crop_is_not_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad={10, 15, 20}))
    backend = FakeBackend({"*": _answer("bus")})
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", _requests(1))
    assert got == {} and backend.calls == []


def test_ask_vlm_closes_runner_when_ask_many_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    closed = []
    monkeypatch.setattr(VLMRunner, "ask_many", lambda self, q: (_ for _ in ()).throw(RuntimeError("x")))
    real_close = VLMRunner.close
    monkeypatch.setattr(VLMRunner, "close", lambda self: (closed.append(1), real_close(self)))
    with pytest.raises(RuntimeError):
        re_class._ask_vlm(_vlm(tmp_path), lambda cfg: FakeBackend({}), "v.mp4", _requests(1))
    assert closed == [1]


def test_budget_is_per_call(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    cfg = _vlm(tmp_path, max_calls=1, max_concurrency=1)
    script = {"*": _answer("bus")}
    first = re_class._ask_vlm(cfg, lambda c: FakeBackend(script), "v.mp4", _requests(1, 2))
    assert sorted(r.error is None for r in first.values()) == [False, True]
    second = re_class._ask_vlm(cfg, lambda c: FakeBackend(script), "v.mp4", _requests(1, 2))
    assert all(r.error is None for r in second.values())  # one cached, one fresh call


# ---- Task 7: VehReClass ----

COLS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"")
    return str(p)


def _vr(monkeypatch, script=None, bad=(), **kw):
    det = FakeDetector(script)
    monkeypatch.setattr(re_class, "Detector", lambda **k: det)
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad=bad))
    return VehReClass(**kw), det


def _same(f, cls, conf=0.9, box=(100, 100, 200, 100)):
    return {f: [(*box, conf, cls)]}


def _script(frames, cls, conf=0.9, box=(100, 100, 200, 100)):
    out = {}
    for f in frames:
        out |= _same(f, cls, conf, box)
    return out


@pytest.mark.parametrize("kw", [
    {"vlm_crops": 0}, {"num_frames": 10.0}, {"hv_min_count": True}, {"hv_size": float("inf")},
    {"min_gap": -1}, {"iou_threshold": 0}, {"min_support": 1.5}, {"hv_size": 0},
])
def test_constructor_rejects(monkeypatch, kw):
    monkeypatch.setattr(re_class, "Detector", lambda **k: FakeDetector())
    with pytest.raises(ValueError):
        VehReClass(**kw)


def test_vlm_key_missing_fails_before_detector(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    built = []
    monkeypatch.setattr(re_class, "Detector", lambda **k: built.append(1))
    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VehReClass"):
        VehReClass(vlm=VLMConfig(backend="anthropic", cache_dir=str(tmp_path)))
    assert built == []


def test_api_key_with_backend_factory_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "Detector", lambda **k: FakeDetector())
    with pytest.raises(ValueError, match="vlm_api_key"):
        VehReClass(vlm=_vlm(tmp_path), vlm_api_key="k", vlm_backend_factory=lambda c: None)
    VehReClass(vlm_api_key="k")  # harmless without a vlm


def test_redetect_writes_cls_on_every_row(monkeypatch, video):
    tracks = pd.concat([
        vehicle_rows(1, range(0, 50), (100, 100, 200, 100)),
        vehicle_rows(2, range(0, 50), (900, 500, 60, 40)),
    ], ignore_index=True)
    tracks.loc[tracks.index[:3], "r3"] = 1  # filled rows still get the class
    # hv_size=150 flags track 1 (size 200) heavy, so its bus vote may decide; track 2 is small
    vr, _ = _vr(monkeypatch, _script(range(50), BUS), hv_size=150)
    out = vr.re_classify(tracks, video, verbose=False)
    assert out[out.track == 1].cls.unique().tolist() == [BUS]
    assert out[out.track == 2].cls.unique().tolist() == [CAR]  # nothing matched: fallback
    assert list(out.columns) == COLS
    pd.testing.assert_frame_equal(out.drop(columns="cls"), tracks.drop(columns="cls"))
    d = vr.decisions.set_index("track")
    assert d.loc[1, "layer"] == "redetect" and d.loc[2, "layer"] == "fallback"
    assert d.loc[1, "redetect_cls"] == BUS and pd.isna(d.loc[2, "redetect_cls"])
    assert str(vr.decisions["redetect_cls"].dtype) == "Int64"
    assert list(vr.decisions.columns) == [
        "track", "cls", "layer", "heavy", "size", "n", "support", "redetect_cls",
        "vlm_answer", "vlm_conf"]


def test_one_stray_detection_does_not_decide(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _same(5, BUS))  # frame 5 is sampled (frames 0, 5, ..., 45)
    out = vr.re_classify(vehicle_rows(1, range(100), (100, 100, 200, 100)), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]


def _boundary_script():
    # samples are frames 0, 5, ..., 45 (10 rows, equal area, min_gap 5): 4 bus votes at 0.3,
    # 3 truck votes at 0.9, 3 frames with nothing
    return (_script((0, 5, 10, 15), BUS, conf=0.3) | _script((20, 25, 30), TRUCK, conf=0.9))


def test_support_exactly_at_min_support_decides(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _boundary_script(), hv_size=150)  # heavy: a truck vote may decide
    out = vr.re_classify(vehicle_rows(1, range(0, 50, 5), (100, 100, 200, 100)), video,
                         verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [TRUCK]
    assert d.layer == "redetect" and d.n == 10 and d.support == 0.3


def test_support_just_below_min_support_goes_on(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _boundary_script(), min_support=0.31, hv_size=150)
    out = vr.re_classify(vehicle_rows(1, range(0, 50, 5), (100, 100, 200, 100)), video,
                         verbose=False)
    assert out.cls.unique().tolist() == [TRUCK]  # heavy fallback, not the vote
    assert vr.decisions.layer.iloc[0] == "fallback"
    assert vr.decisions.redetect_cls.iloc[0] == TRUCK  # recorded although it did not decide


def test_heavy_fallback_is_truck_and_car_not_allowed(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(100), CAR, box=(100, 100, 600, 300)), hv_size=500)
    out = vr.re_classify(vehicle_rows(1, range(100), (100, 100, 600, 300)), video, verbose=False)
    assert out.cls.unique().tolist() == [TRUCK]
    assert bool(vr.decisions.heavy.iloc[0])


def test_size_just_below_hv_size_is_not_heavy(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(100), CAR, box=(100, 100, 600, 300)), hv_size=601)
    out = vr.re_classify(vehicle_rows(1, range(100), (100, 100, 600, 300)), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]
    row = vr.decisions.iloc[0]
    assert not row.heavy and row["size"] == 600 and row.layer == "redetect"


def test_named_interp_and_headerless_file_agree(monkeypatch, video, tmp_path):
    real = vehicle_rows(1, range(0, 20), (100, 100, 200, 100), interp=0)
    filled = vehicle_rows(1, range(20, 30), (100, 100, 800, 400), interp=1)
    named = pd.concat([real, filled], ignore_index=True).drop(columns="r3")
    named = named[["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]]
    path = tmp_path / "t.txt"
    named.to_csv(path, header=False, index=False)
    script = _script(range(20), CAR) | _script(range(20, 30), BUS, box=(100, 100, 800, 400))
    vr, _ = _vr(monkeypatch, script, hv_size=500)
    a = vr.re_classify(named, video, verbose=False)
    da = vr.decisions.copy()
    b = vr.re_classify(str(path), video, verbose=False)
    assert a.cls.tolist() == b.cls.tolist() == [CAR] * 30
    pd.testing.assert_frame_equal(da, vr.decisions)
    assert list(a.columns) == list(named.columns) and list(b.columns) == COLS


def test_all_filled_track_falls_back(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(10), BUS))
    out = vr.re_classify(vehicle_rows(1, range(10), (100, 100, 200, 100), r3=1), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]
    row = vr.decisions.iloc[0]
    assert row.layer == "fallback" and not row.heavy and row.n == 0


def test_all_frames_unreadable_falls_back(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), bad=set(range(10)))
    out = vr.re_classify(vehicle_rows(1, range(10), (100, 100, 200, 100)), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]
    assert vr.decisions.n.iloc[0] == 0 and math.isnan(vr.decisions.support.iloc[0])


def test_unread_frame_leaves_only_tracks_that_sampled_it(monkeypatch, video):
    tracks = pd.concat([
        vehicle_rows(1, [0, 5, 10], (100, 100, 200, 100)),
        vehicle_rows(2, [5, 20], (400, 100, 200, 100)),
        vehicle_rows(3, [30, 40], (700, 100, 200, 100)),
    ], ignore_index=True)
    vr, det = _vr(monkeypatch, bad={5})
    vr.re_classify(tracks, video, verbose=False)
    assert vr.decisions.set_index("track").n.to_dict() == {1: 2, 2: 1, 3: 2}
    assert det.seen == [0, 10, 20, 30, 40]


def test_float_frame_column_matches_int(monkeypatch, video):
    tracks = vehicle_rows(1, range(30), (100, 100, 200, 100))
    vr, _ = _vr(monkeypatch, _script(range(30), BUS), hv_size=150)
    a = vr.re_classify(tracks, video, verbose=False)
    b = vr.re_classify(tracks.astype({"frame": float}), video, verbose=False)
    assert a.cls.tolist() == b.cls.tolist() == [BUS] * 30


def test_track_ids_subset_and_unknown(monkeypatch, video, caplog):
    tracks = pd.concat([vehicle_rows(1, range(10), (100, 100, 200, 100)),
                        vehicle_rows(2, range(10), (100, 100, 200, 100))], ignore_index=True)
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), hv_size=150)
    out = vr.re_classify(tracks, video, track_ids=[2, 99], verbose=False)
    assert out[out.track == 1].cls.unique().tolist() == [-1]
    assert out[out.track == 2].cls.unique().tolist() == [BUS]
    assert "1 track ID" in caplog.text


def test_track_ids_numpy_array(monkeypatch, video, caplog):
    tracks = vehicle_rows(1, range(10), (100, 100, 200, 100))
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), hv_size=150)
    out = vr.re_classify(tracks, video, track_ids=np.array([1, 7]), verbose=False)
    assert out.cls.unique().tolist() == [BUS] and "1 track ID" in caplog.text


def test_empty_inputs(monkeypatch, video, tmp_path):
    vr, _ = _vr(monkeypatch)
    empty = tmp_path / "e.txt"
    empty.write_bytes(b"")
    for src in (pd.DataFrame(columns=COLS), str(empty)):
        out_file = tmp_path / "o.txt"
        out = vr.re_classify(src, video, out_file=str(out_file), verbose=False)
        assert out.empty and out_file.stat().st_size == 0 and vr.decisions.empty
    with pytest.raises(FileNotFoundError):
        vr.re_classify(pd.DataFrame(columns=COLS), str(tmp_path / "missing.mp4"))


def test_out_file_round_trip(monkeypatch, video, tmp_path):
    tracks = vehicle_rows(1, range(10), (100, 100, 200, 100))
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), hv_size=150)
    out = vr.re_classify(tracks, video, out_file=str(tmp_path / "o.txt"), verbose=False)
    back = read_track(tmp_path / "o.txt")
    assert back[7].tolist() == out.cls.tolist()


def test_vlm_decides_undecided_track(monkeypatch, video, tmp_path):
    backend = FakeBackend({"vehclass:1": _answer("bus", 0.8), "vehclass:2": _answer("bus", 0.5)})
    tracks = pd.concat([vehicle_rows(1, range(30), (100, 100, 200, 100)),
                        vehicle_rows(2, range(30), (400, 100, 200, 100))], ignore_index=True)
    vr, _ = _vr(monkeypatch, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = vr.re_classify(tracks, video, verbose=False)
    d = vr.decisions.set_index("track")
    assert out[out.track == 1].cls.unique().tolist() == [BUS] and d.loc[1, "layer"] == "vlm"
    assert out[out.track == 2].cls.unique().tolist() == [CAR] and d.loc[2, "layer"] == "fallback"
    assert d.loc[2, "vlm_answer"] == "bus" and d.loc[2, "vlm_conf"] == 0.5  # below min_conf 0.7


@pytest.mark.parametrize("dtype", [int, float])
def test_vlm_with_int_or_float_track_ids(monkeypatch, video, tmp_path, dtype):
    backend = FakeBackend({"vehclass:1": _answer("bus", 0.8)})
    tracks = vehicle_rows(1, range(30), (100, 100, 200, 100)).astype({"track": dtype})
    vr, _ = _vr(monkeypatch, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = vr.re_classify(tracks, video, verbose=False)
    assert out.cls.unique().tolist() == [BUS]
    assert vr.decisions.layer.iloc[0] == "vlm"
    pd.testing.assert_series_equal(out["track"], tracks["track"])  # IDs and dtype unchanged
    assert [c["tag"] for c in backend.calls] == ["vehclass:1"]


def test_vlm_unclear_and_error_fall_back(monkeypatch, video, tmp_path, caplog):
    backend = FakeBackend({"vehclass:1": _answer("unclear"), "vehclass:2": RuntimeError("boom")})
    tracks = pd.concat([vehicle_rows(1, range(30), (100, 100, 200, 100)),
                        vehicle_rows(2, range(30), (400, 100, 600, 300))], ignore_index=True)
    vr, _ = _vr(monkeypatch, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend,
                hv_size=250)
    with caplog.at_level(logging.WARNING, logger=re_class.log.name):
        out = vr.re_classify(tracks, video, verbose=False)
    assert out[out.track == 1].cls.unique().tolist() == [CAR]
    assert out[out.track == 2].cls.unique().tolist() == [TRUCK]
    msgs = [r.getMessage() for r in caplog.records if r.name == re_class.log.name]
    assert len(msgs) == 1 and "track 2" in msgs[0] and "boom" in msgs[0]
    d = vr.decisions.set_index("track")
    assert d.loc[1, "vlm_answer"] == "unclear" and d.loc[1, "vlm_conf"] == 0.9
    assert d.loc[2, "vlm_answer"] is None and math.isnan(d.loc[2, "vlm_conf"])
    assert d.loc[2, "layer"] == "fallback"


def test_no_vlm_runner_when_nothing_pending(monkeypatch, video, tmp_path):
    built = []
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), vlm=_vlm(tmp_path),
                vlm_backend_factory=lambda c: built.append(1), hv_size=150)
    vr.re_classify(vehicle_rows(1, range(10), (100, 100, 200, 100)), video, verbose=False)
    assert built == []


# ---- light vs heavy: re-detection alone never promotes an unflagged track ----

VAN = (100, 100, 200, 100)


def test_unflagged_truck_vote_goes_to_vlm_with_all_options(monkeypatch, video, tmp_path):
    # a large van: RT-DETR calls it truck in most frames, but nothing flags it heavy
    backend = FakeBackend({"vehclass:20": _answer("car", 0.9)})
    vr, _ = _vr(monkeypatch, _script(range(0, 50, 5), TRUCK, conf=0.9, box=VAN),
                vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = vr.re_classify(vehicle_rows(20, range(50), VAN), video, verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [CAR]
    assert d.layer == "vlm" and d.redetect_cls == TRUCK and d.support == 1.0
    assert [c["options"] for c in backend.calls] == [["car", "bus", "truck", "unclear"]]


def test_unflagged_truck_vote_without_vlm_falls_back_to_car(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(0, 50, 5), TRUCK, conf=0.9, box=VAN))
    out = vr.re_classify(vehicle_rows(37, range(50), VAN), video, verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [CAR]
    assert d.layer == "fallback" and d.redetect_cls == TRUCK and not d.heavy


def test_heavy_truck_vote_decides_without_vlm(monkeypatch, video, tmp_path):
    built = []
    vr, _ = _vr(monkeypatch, _script(range(0, 50, 5), TRUCK, conf=0.9, box=VAN), hv_size=150,
                vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: built.append(1))
    out = vr.re_classify(vehicle_rows(1, range(50), VAN), video, verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [TRUCK]
    assert d.layer == "redetect" and d.redetect_cls == TRUCK and built == []


def test_unflagged_bus_vote_goes_to_vlm(monkeypatch, video, tmp_path):
    backend = FakeBackend({"vehclass:1": _answer("bus", 0.8)})
    vr, _ = _vr(monkeypatch, _script(range(0, 50, 5), BUS, conf=0.9, box=VAN),
                vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = vr.re_classify(vehicle_rows(1, range(50), VAN), video, verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [BUS]
    assert d.layer == "vlm" and d.redetect_cls == BUS and len(backend.calls) == 1


def test_unflagged_car_vote_still_decides(monkeypatch, video, tmp_path):
    built = []
    vr, _ = _vr(monkeypatch, _script(range(0, 50, 5), CAR, conf=0.9, box=VAN),
                vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: built.append(1))
    out = vr.re_classify(vehicle_rows(1, range(50), VAN), video, verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [CAR]
    assert d.layer == "redetect" and d.redetect_cls == CAR and built == []
