"""PedReClass results are pinned, and the shared reading helpers tolerate bad frames."""

import pandas as pd
from reclass_fakes import FakeDetector, FakeReader

from dnt.track import re_class


def _tracks():
    t1 = pd.DataFrame({"frame": [9, 1, 5], "track": 1, "x": 100, "y": 100, "w": 50, "h": 50})
    t2 = pd.DataFrame({
        "frame": [20, 21, 22, 23, 24], "track": 2, "x": 300, "y": 300,
        "w": [10, 20, 30, 40, 50], "h": 50,
    })
    t3 = pd.DataFrame({"frame": [40, 41], "track": 3, "x": 500, "y": 500, "w": [30, 60], "h": 60})
    return pd.concat([t1, t2, t3], ignore_index=True)


def test_reclass_results_are_pinned(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    det = FakeDetector({
        9: [(110, 110, 20, 20, 0.9, 1)],   # inside track 1 at frame 9 (the tie winner)
        1: [(110, 110, 20, 20, 0.9, 36)],  # frame 1 must NOT be sampled with num_frames=1
        20: [(305, 305, 5, 5, 0.9, 36)],   # track 2's smallest box: not among its top 3
        41: [(510, 510, 20, 20, 0.9, 36)],  # track 3's largest box
    })
    monkeypatch.setattr(re_class, "Detector", lambda **kw: det)
    monkeypatch.setattr(re_class, "FrameReader", FakeReader, raising=False)
    rc = re_class.PedReClass(num_frames=1, threshold=0.75)
    out = rc.re_classify(_tracks(), str(video), verbose=False)
    assert out.values.tolist() == [[1, 1, 1.0], [2, 0, 0.0], [3, 36, 1.0]]


def test_read_frames_continues_after_a_bad_middle_frame():
    reader = FakeReader("v", bad={5})
    got = [(f, img is None) for f, img in re_class._read_frames(reader, [9, 5, 1, 9])]
    assert got == [(1, False), (5, True), (9, False)]
    assert reader.requests == [[1], [5], [9]]


def test_detect_frames_once_each_in_order():
    det = FakeDetector({1: [(0, 0, 5, 5, 0.5, 2)]})
    by_frame, unread = re_class._detect_frames(det, FakeReader("v", bad={4}), [7, 1, 4, 7])
    assert det.seen == [1, 7]
    assert unread == {4}
    assert sorted(by_frame) == [1, 7]
    assert len(by_frame[1]) == 1 and by_frame[7].empty


def test_dets_for_empty_has_columns():
    out = re_class._dets_for({}, [3])
    assert out.empty and list(out.columns) == re_class._DET_FIELDS


# ---- riders: bicycle, motorcycle and skateboard/scooter evidence, optional VLM ----

import json  # noqa: E402

import pytest  # noqa: E402

from dnt.refine.config import VLMConfig  # noqa: E402
from dnt.refine.hints import read_reclass_hints  # noqa: E402
from dnt.refine.vlm.fake import FakeBackend  # noqa: E402

BIKE, MOTO, BOARD = 1, 3, 36
RIDER_OPTIONS = ["pedestrian", "cyclist", "motorcycle_rider", "scooter_rider", "unclear"]


def _answer(ans, conf=0.9):
    return json.dumps({"answer": ans, "confidence": conf, "reason": "r"})


def _vlm(tmp_path, **kw):
    base = dict(backend="openai_compat", model="m", base_url="http://localhost:9/v1",
                cache_dir=str(tmp_path / "cache"))
    return VLMConfig(**(base | kw))


def _walker(frames=(0, 1, 2)):
    return pd.DataFrame({"frame": list(frames), "track": 1, "x": 100, "y": 100, "w": 50, "h": 100})


def _inside(cls, conf=0.9):
    return (110, 110, 20, 20, conf, cls)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"")
    return str(p)


def _rc(monkeypatch, script, **kw):
    det = FakeDetector(script)
    monkeypatch.setattr(re_class, "Detector", lambda **k: det)
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    return re_class.PedReClass(num_frames=3, **kw)


def test_a_motorcycle_overlap_makes_a_motorcycle_rider(monkeypatch, video):
    rc = _rc(monkeypatch, {f: [_inside(MOTO)] for f in range(3)})
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out.values.tolist() == [[1, MOTO, 1.0]]


def test_a_strong_single_class_overlap_skips_the_vlm(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("pedestrian")})
    rc = _rc(monkeypatch, {f: [_inside(BIKE)] for f in range(3)},
             vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out["cls"].tolist() == [BIKE] and backend.calls == []
    assert rc.decisions["layer"].tolist() == ["overlap"]


def test_no_overlap_stays_pedestrian_without_the_vlm(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("cyclist")})
    rc = _rc(monkeypatch, {}, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out["cls"].tolist() == [0] and backend.calls == []
    assert rc.decisions["layer"].tolist() == ["default"]


def test_two_overlapping_classes_go_to_the_vlm_with_the_rider_options(monkeypatch, video, tmp_path):
    backend = FakeBackend({"riderclass:1": _answer("motorcycle_rider", 0.8)})
    script = {f: [_inside(BIKE), (115, 115, 20, 20, 0.9, MOTO)] for f in range(3)}
    rc = _rc(monkeypatch, script, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out["cls"].tolist() == [MOTO]
    assert out["avg_score"].tolist() == [1.0]  # the overlap, when it is above the VLM confidence
    assert [c["options"] for c in backend.calls] == [RIDER_OPTIONS]
    assert rc.decisions.loc[0, "layer"] == "vlm"


def test_a_weak_overlap_is_sent_to_the_vlm_and_it_can_say_pedestrian(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("pedestrian", 0.9)})
    rc = _rc(monkeypatch, {0: [_inside(BIKE)]},  # 1 of 3 frames: support 1/3 < 0.5
             vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out.values.tolist() == [[1, 0, 0.0]]
    assert len(backend.calls) == 1


def test_a_weak_overlap_without_a_vlm_keeps_the_old_behaviour(monkeypatch, video):
    rc = _rc(monkeypatch, {0: [_inside(BIKE)]})
    assert rc.re_classify(_walker(), video, verbose=False)["cls"].tolist() == [BIKE]


@pytest.mark.parametrize("reply", [_answer("unclear", 0.95), _answer("scooter_rider", 0.3)])
def test_an_unclear_or_unsure_vlm_falls_back_to_the_best_overlap(monkeypatch, video, tmp_path, reply):
    backend = FakeBackend({"*": reply})
    script = {f: [_inside(BIKE, 0.9), (115, 115, 20, 20, 0.5, MOTO)] for f in range(3)}
    rc = _rc(monkeypatch, script, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out["cls"].tolist() in ([BIKE], [MOTO])  # a tie on IoB goes to the first match_class
    assert rc.decisions.loc[0, "layer"] == "overlap"


def test_the_vlm_options_follow_match_class(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("cyclist", 0.9)})
    rc = _rc(monkeypatch, {0: [_inside(BIKE)]}, match_class=[BIKE],
             vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    rc.re_classify(_walker(), video, verbose=False)
    assert backend.calls[0]["options"] == ["pedestrian", "cyclist", "unclear"]


def test_a_scooter_answer_maps_to_the_skateboard_class(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("scooter_rider", 0.9)})
    rc = _rc(monkeypatch, {0: [_inside(BIKE)]},
             vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out.values.tolist() == [[1, BOARD, 0.9]]


def test_the_output_reads_back_as_refine_hints(monkeypatch, video, tmp_path):
    rc = _rc(monkeypatch, {f: [_inside(MOTO)] for f in range(3)})
    out_file = tmp_path / "hints.csv"
    rc.re_classify(_walker(), video, out_file=str(out_file), verbose=False)
    hints = read_reclass_hints(out_file, {1})
    assert (hints[1].cls, hints[1].avg_score) == (MOTO, 1.0)


def test_a_missing_vlm_key_fails_before_the_detector(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    built = []
    monkeypatch.setattr(re_class, "Detector", lambda **k: built.append(1))
    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"PedReClass: vlm\.backend"):
        re_class.PedReClass(vlm=VLMConfig(backend="anthropic", cache_dir=str(tmp_path)))
    assert built == []


def test_rider_map_can_send_cyclist_answers_back_to_pedestrian(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("cyclist", 0.9)})
    rc = _rc(monkeypatch, {0: [_inside(BIKE)]}, rider_map={"cyclist": 0, "scooter_rider": BOARD},
             vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = rc.re_classify(_walker(), video, verbose=False)
    assert out.values.tolist() == [[1, 0, 0.0]]
    assert backend.calls[0]["options"] == ["pedestrian", "cyclist", "scooter_rider", "unclear"]


def test_rider_map_rejects_unknown_answers():
    with pytest.raises(ValueError, match="rider_map"):
        re_class.PedReClass(rider_map={"jogger": 0})


# ---- ReClass routes to PedReClass, VehReClass and TwoWheelReClass ----


def test_reclass_defaults_to_the_pedestrian_reclassifier(monkeypatch, video):
    det = FakeDetector({f: [_inside(MOTO)] for f in range(3)})
    monkeypatch.setattr(re_class, "Detector", lambda **k: det)
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    rc = re_class.ReClass(num_frames=3)
    assert rc.kind == "ped" and isinstance(rc.impl, re_class.PedReClass)
    assert rc.re_classify(_walker(), video, verbose=False).values.tolist() == [[1, MOTO, 1.0]]
    assert rc.decisions["layer"].tolist() == ["overlap"]


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("veh", "VehReClass"),
        ("Vehicle", "VehReClass"),
        ("twowheel", "TwoWheelReClass"),
        ("two-wheel", "TwoWheelReClass"),
        ("mc", "TwoWheelReClass"),
    ],
)
def test_reclass_builds_the_class_for_its_kind(monkeypatch, kind, cls):
    monkeypatch.setattr(re_class, "Detector", lambda **k: FakeDetector())
    rc = re_class.ReClass(kind)
    assert type(rc.impl) is getattr(re_class, cls)
    assert rc.kind == {"VehReClass": "veh", "TwoWheelReClass": "twowheel"}[cls]


def test_reclass_forwards_arguments_and_calls_for_two_wheelers(monkeypatch, video):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    rc = re_class.ReClass("twowheel", scooter="keep")
    assert rc.impl.scooter == "keep"
    tracks = _walker().assign(score=-1.0, cls=BIKE, r3=-1, r4=-1)
    out = rc.re_classify(tracks, video, verbose=False)
    assert set(out["cls"]) == {BIKE} and rc.decisions["layer"].tolist() == ["prior"]


def test_reclass_rejects_an_unknown_kind_and_bad_arguments():
    with pytest.raises(ValueError, match="kind must be one of"):
        re_class.ReClass("boat")
    with pytest.raises(ValueError, match="invalid TwoWheelReClass settings"):
        re_class.ReClass("twowheel", scooter="x")
    with pytest.raises(TypeError):
        re_class.ReClass("twowheel", num_frames=3)
