"""VehReClass: helpers (Task 5), VLM layer (Task 6) and the class (Task 7)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from reclass_fakes import DET_FIELDS, FakeReader

from dnt.refine.config import VLMConfig
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import VLMRunner
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


def test_prepare_vlm_missing_anthropic_key(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        re_class._prepare_vlm(VLMConfig(backend="anthropic", cache_dir=str(tmp_path)), None, check=True)


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
