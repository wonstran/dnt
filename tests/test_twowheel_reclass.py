"""TwoWheelReClass sorts two-wheeler tracks into bicycle and motorcycle (and drops scooters)."""

import json

import pandas as pd
import pytest
from reclass_fakes import FakeReader, vehicle_rows

from dnt.refine.config import VLMConfig
from dnt.refine.vlm.fake import FakeBackend
from dnt.track import TwoWheelReClass, re_class

BIKE, MOTO, BOARD = 1, 3, 36
N = 10
BOX = (100, 100, 50, 100)


def _answer(ans, conf=0.9):
    return json.dumps({"answer": ans, "confidence": conf, "reason": "r"})


def _vlm(tmp_path, **kw):
    base = dict(backend="openai_compat", model="m", base_url="http://localhost:9/v1",
                cache_dir=str(tmp_path / "cache"))
    return VLMConfig(**(base | kw))


def _track(tid=1, cls=-1, n=N, box=BOX):
    return vehicle_rows(tid, range(n), box, cls=cls)


def _dets(frames, box=BOX):
    """Detections of one box on each of ``frames``."""
    x, y, w, h = box
    return pd.DataFrame({"frame": list(frames), "x": x, "y": y, "w": w, "h": h})


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"")
    return str(p)


def _rc(monkeypatch, backend=None, tmp_path=None, **kw):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    if backend is not None:
        kw.update(vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    return TwoWheelReClass(**kw)


def test_a_clear_motorcycle_overlap_decides_without_the_vlm(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("cyclist")})
    rc = _rc(monkeypatch, backend, tmp_path)
    out = rc.re_classify(_track(), video, mc_dets=_dets(range(N)), verbose=False)
    assert set(out["cls"]) == {MOTO}
    assert backend.calls == []
    assert rc.decisions.loc[0, "layer"] == "prior"


def test_a_clear_bicycle_overlap_decides(monkeypatch, video):
    rc = _rc(monkeypatch)
    out = rc.re_classify(_track(), video, bike_dets=_dets(range(N)), verbose=False)
    assert set(out["cls"]) == {BIKE}


def test_existing_class_votes_decide_when_no_detection_overlaps(monkeypatch, video):
    rc = _rc(monkeypatch)
    out = rc.re_classify(_track(cls=MOTO), video, verbose=False)
    assert set(out["cls"]) == {MOTO}


def test_mixed_evidence_goes_to_the_vlm_and_a_cyclist_is_a_bicycle(monkeypatch, video, tmp_path):
    backend = FakeBackend({"riderclass:1": _answer("cyclist", 0.8)})
    rc = _rc(monkeypatch, backend, tmp_path)
    out = rc.re_classify(
        _track(), video, mc_dets=_dets(range(N)), bike_dets=_dets(range(N)), verbose=False
    )
    assert set(out["cls"]) == {BIKE}
    assert rc.decisions.loc[0, "layer"] == "vlm"
    assert len(backend.calls) == 1


def test_a_vlm_motorcycle_rider_is_a_motorcycle(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("motorcycle_rider", 0.8)})
    rc = _rc(monkeypatch, backend, tmp_path)
    out = rc.re_classify(
        _track(), video, mc_dets=_dets(range(6)), bike_dets=_dets(range(6)), verbose=False
    )
    assert set(out["cls"]) == {MOTO}


@pytest.mark.parametrize("ans", ["scooter_rider", "pedestrian"])
def test_a_vlm_scooter_or_pedestrian_drops_the_track(monkeypatch, video, tmp_path, ans):
    backend = FakeBackend({"*": _answer(ans, 0.9)})
    rc = _rc(monkeypatch, backend, tmp_path)
    tracks = pd.concat([_track(1), _track(2, cls=BIKE)], ignore_index=True)
    out = rc.re_classify(
        tracks, video, mc_dets=_dets(range(6)), bike_dets=_dets(range(6)), verbose=False
    )
    assert out["track"].unique().tolist() == [2]
    assert pd.isna(rc.decisions.set_index("track").loc[1, "cls"])


def test_scooter_keep_maps_the_scooter_answer_to_the_skateboard_class(monkeypatch, video, tmp_path):
    backend = FakeBackend({"*": _answer("scooter_rider", 0.9)})
    rc = _rc(monkeypatch, backend, tmp_path, scooter="keep")
    out = rc.re_classify(
        _track(), video, mc_dets=_dets(range(6)), bike_dets=_dets(range(6)), verbose=False
    )
    assert set(out["cls"]) == {BOARD}


def test_a_track_voted_as_a_scooter_is_dropped_or_kept(monkeypatch, video):
    tracks = _track(cls=BOARD)
    dropped = _rc(monkeypatch).re_classify(tracks, video, verbose=False)
    assert dropped.empty
    kept = _rc(monkeypatch, scooter="keep").re_classify(tracks, video, verbose=False)
    assert set(kept["cls"]) == {BOARD}


def test_an_unsure_or_unclear_vlm_falls_back_to_the_larger_overlap(monkeypatch, video, tmp_path):
    for reply in (_answer("cyclist", 0.2), _answer("unclear", 0.95)):
        backend = FakeBackend({"*": reply})
        rc = _rc(monkeypatch, backend, tmp_path)
        out = rc.re_classify(
            _track(), video, mc_dets=_dets(range(6)), bike_dets=_dets(range(4)), verbose=False
        )
        assert set(out["cls"]) == {MOTO}
        assert rc.decisions.loc[0, "layer"] == "fallback"


def test_without_a_vlm_an_ambiguous_track_falls_back(monkeypatch, video):
    rc = _rc(monkeypatch)
    out = rc.re_classify(
        _track(), video, mc_dets=_dets(range(4)), bike_dets=_dets(range(6)), verbose=False
    )
    assert set(out["cls"]) == {BIKE}
    assert rc.decisions.loc[0, "layer"] == "fallback"


def test_the_fallback_uses_the_class_votes_on_a_tie(monkeypatch, video):
    rc = _rc(monkeypatch)
    tracks = _track(cls=[MOTO] * 6 + [BIKE] * 4)
    assert set(rc.re_classify(tracks, video, verbose=False)["cls"]) == {MOTO}


def test_a_track_gets_one_class_on_every_row(monkeypatch, video):
    rc = _rc(monkeypatch)
    tracks = _track(cls=[BIKE, MOTO] * (N // 2))
    out = rc.re_classify(tracks, video, bike_dets=_dets(range(N)), verbose=False)
    assert out["cls"].nunique() == 1


def test_detection_files_are_read_headerless(monkeypatch, video, tmp_path):
    mc = tmp_path / "mc.txt"
    mc.write_text("".join(f"{f},-1,100,100,50,100,0.9,3\n" for f in range(N)))
    out = _rc(monkeypatch).re_classify(_track(), video, mc_dets=str(mc), verbose=False)
    assert set(out["cls"]) == {MOTO}


def test_output_file_keeps_the_input_columns_headerless(monkeypatch, video, tmp_path):
    out_file = tmp_path / "out.txt"
    _rc(monkeypatch).re_classify(
        _track(), video, bike_dets=_dets(range(N)), out_file=str(out_file), verbose=False
    )
    written = pd.read_csv(out_file, header=None)
    assert written.shape == (N, 10) and set(written[7]) == {BIKE}


def test_empty_tracks_return_empty(monkeypatch, video):
    out = _rc(monkeypatch).re_classify(_track().iloc[0:0], video, verbose=False)
    assert out.empty


def test_track_ids_limit_what_is_classified(monkeypatch, video):
    tracks = pd.concat([_track(1), _track(2)], ignore_index=True)
    out = _rc(monkeypatch).re_classify(
        tracks, video, track_ids=[1], bike_dets=_dets(range(N)), verbose=False
    )
    assert set(out.loc[out["track"] == 1, "cls"]) == {BIKE}
    assert set(out.loc[out["track"] == 2, "cls"]) == {-1}


@pytest.mark.parametrize(
    "kw", [{"scooter": "x"}, {"min_ratio": 0}, {"dominance": 0.5}, {"iob_threshold": 1.5}]
)
def test_bad_settings_raise(kw):
    with pytest.raises(ValueError):
        TwoWheelReClass(**kw)


def test_a_missing_key_names_the_class(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"TwoWheelReClass: vlm\.backend"):
        TwoWheelReClass(vlm=VLMConfig(backend="anthropic", cache_dir=str(tmp_path)))
