import pytest
from fake_ultralytics import FakeModel

from dnt.detect.yolo import segmentor as seg_mod
from dnt.detect.yolo.segmentor import SegmentorModel


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(seg_mod, "YOLO", FakeModel)


@pytest.mark.parametrize("method", ["segment", "segment_crop"])
def test_half_never_passed_directly_to_predict(fake, synthetic_video, method, monkeypatch):
    """`half=` must go through `_device.predict_precision_kwargs`, not straight to predict()."""
    # unrelated to this test: destroyAllWindows() needs GTK/Cocoa, unavailable in this headless env
    monkeypatch.setattr(seg_mod.cv2, "destroyAllWindows", lambda: None)
    video, _ = synthetic_video
    s = seg_mod.Segmentor(model=SegmentorModel.YOLO26nSeg, device="cpu", enable_half=True)
    if method == "segment":
        s.segment(str(video), verbose=False, end_frame=0)
    else:
        s.segment_crop(str(video), 0, [0, 0, 10, 10])
    seen = s.model.last_kwargs
    assert {k: seen[k] for k in s._precision_kwargs} == s._precision_kwargs
