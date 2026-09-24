import inspect

import pandas as pd
import pytest
from fake_ultralytics import FakeModel

from dnt.detect.yolo import detector as det_mod
from dnt.detect.yolo.detector import DetectorModel


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(det_mod, "YOLO", FakeModel)
    monkeypatch.setattr(det_mod, "RTDETR", FakeModel)


@pytest.mark.parametrize(("name", "expected"), [
    ("yolo26x", DetectorModel.YOLO26x), ("yolo26x.pt", DetectorModel.YOLO26x), ("YOLO26x", DetectorModel.YOLO26x),
    ("rtdetr-x", DetectorModel.RTDETRx), ("RTDETRx", DetectorModel.RTDETRx), (DetectorModel.YOLO11n, DetectorModel.YOLO11n),
])
def test_model_strings_coerced(fake, name, expected):
    d = det_mod.Detector(model=name, device="cpu")
    assert d.model.path.endswith(expected.value)


def test_unknown_model_string_raises(fake):
    with pytest.raises(ValueError, match="Unknown detector model 'rtdetr'"):
        det_mod.Detector(model="rtdetr", device="cpu")


def test_detect_and_detect_frames_same_rounding(fake, synthetic_video):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    full = d.detect(str(video), verbose=False)
    some = d.detect_frames(str(video), [0, 1], verbose=False)
    cols = ["frame", "x", "y", "w", "h", "conf", "class"]
    pd.testing.assert_frame_equal(some[cols], full[full.frame.isin([0, 1])][cols].reset_index(drop=True))
    assert list(some.loc[0, ["x", "y", "w", "h"]]) == [10, 20, 40, 49]  # truncation, as detect() always did


def test_reclass_default_constructs(monkeypatch):
    from dnt.track import re_class

    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(re_class, "Detector", Recorder)
    re_class.ReClass()
    assert seen == {"model": DetectorModel.RTDETRx, "weights": None, "device": "auto"}


def test_reclass_with_real_detector_class(fake):
    from dnt.track.re_class import ReClass

    ReClass(device="cpu")


def test_reclass_positional_order_unchanged():
    """Ensure ReClass parameter order is unchanged for backward compatibility."""
    from dnt.track.re_class import ReClass

    params = list(inspect.signature(ReClass.__init__).parameters)[1:5]
    assert params == ["num_frames", "threshold", "model", "weights"]
