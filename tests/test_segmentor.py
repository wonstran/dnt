from pathlib import Path

import numpy as np
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
    s = seg_mod.Segmentor(model=SegmentorModel.YOLO26nSeg, device="cpu", half=True, fast=False)
    if method == "segment":
        s.segment(str(video), verbose=False, end_frame=0)
    else:
        s.segment_crop(str(video), 0, [0, 0, 10, 10])
    seen = s.model.last_kwargs
    assert {k: seen[k] for k in s._precision_kwargs} == s._precision_kwargs


def test_segment_defaults_redirect_to_segment_fast(fake, synthetic_video, monkeypatch):
    video, _ = synthetic_video
    s = seg_mod.Segmentor(model=SegmentorModel.YOLO26nSeg, device="cpu", batch=4)
    calls = []
    monkeypatch.setattr(
        seg_mod.Segmentor, "segment_fast", lambda self, v, **kw: calls.append(kw) or [{"sentinel": 1}]
    )
    assert s.segment(str(video), verbose=False) == [{"sentinel": 1}]
    assert calls[0]["batch"] == 4


def test_segment_fast_false_uses_per_frame_pipeline(fake, synthetic_video, monkeypatch):
    video, _ = synthetic_video
    s = seg_mod.Segmentor(model=SegmentorModel.YOLO26nSeg, device="cpu", fast=False)

    def boom(self, *args, **kwargs):
        raise AssertionError("segment_fast() should not be called when self.fast is False")

    monkeypatch.setattr(seg_mod.Segmentor, "segment_fast", boom)
    s.segment(str(video), verbose=False)  # must not raise


def test_segment_fast_batches_and_writes_jsonl(fake, synthetic_video, tmp_path, monkeypatch):
    video, _ = synthetic_video
    s = seg_mod.Segmentor(model=SegmentorModel.YOLO26nSeg, device="cpu", batch=3)
    seen = []

    class Res:
        masks = type("M", (), {"xy": [np.zeros((2, 2))]})()
        boxes = type("B", (), {"cls": np.array([2.0]), "conf": np.array([0.9])})()

    class Pred:
        def preprocess(self, im0s):
            seen.append(len(im0s))
            return im0s

        def inference(self, im):
            return im

        def postprocess(self, preds, im, im0s):
            return [Res() for _ in im0s]

    monkeypatch.setattr(s, "_ensure_predictor", lambda: Pred())
    mask_file = tmp_path / "m_mask.txt"
    res = s.segment_fast(str(video), mask_file=str(mask_file), verbose=False)

    assert max(seen) == 3 and sum(seen) == len(res)
    assert [r["frame"] for r in res] == list(range(len(res)))
    assert not Path(f"{mask_file}.part").exists()
    assert len(mask_file.read_text().splitlines()) == len(res)


def test_resume_part_file_drops_last_line_and_partial_tail(tmp_path):
    part = tmp_path / "x.part"
    part.write_text('{"frame": 0}\n{"frame": 1}\n{"frame": 2}\n{"fra')
    assert seg_mod.Segmentor._resume_part_file(part) == 2
    assert part.read_text() == '{"frame": 0}\n{"frame": 1}\n'
    part.write_text('{"fra')
    assert seg_mod.Segmentor._resume_part_file(part) is None
    assert part.read_text() == ""
