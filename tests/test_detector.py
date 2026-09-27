import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
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
    # fast=False: FakeModel has no .predictor, so it can't go through detect_fast()'s
    # bootstrap; this test is about detect()'s traditional pipeline specifically anyway.
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)
    full = d.detect(str(video), verbose=False)
    some = d.detect_frames(str(video), [0, 1], verbose=False)
    cols = ["frame", "x", "y", "w", "h", "conf", "class"]
    pd.testing.assert_frame_equal(some[cols], full[full.frame.isin([0, 1])][cols].reset_index(drop=True))
    assert list(some.loc[0, ["x", "y", "w", "h"]]) == [10, 20, 40, 49]  # truncation, as detect() always did


def test_detect_does_not_print_output_file_name(fake, synthetic_video, tmp_path, capsys):
    video, _ = synthetic_video
    out = tmp_path / "scene_iou.txt"
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)  # FakeModel has no .predictor
    d.detect(str(video), iou_file=str(out), verbose=True)
    assert out.exists()
    assert str(out) not in capsys.readouterr().out


def test_detect_batch_does_not_print_output_file_name(fake, synthetic_video, tmp_path, capsys):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)
    files = d.detect_batch([str(video)], output_path=str(tmp_path), verbose=True)
    assert [p.endswith("scene_iou.txt") for p in files] == [True]
    assert "scene_iou.txt" not in capsys.readouterr().out


@pytest.mark.parametrize("method", ["detect", "detect_frames"])
def test_half_never_passed_directly_to_predict(fake, synthetic_video, method):
    """`half=` must go through `_device.predict_precision_kwargs`, not straight to predict()."""
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", half=True, fast=False)
    if method == "detect":
        d.detect(str(video), verbose=False)  # FakeModel has no .predictor
    else:
        d.detect_frames(str(video), [0], verbose=False)
    seen = d.model.last_kwargs
    assert {k: seen[k] for k in d._precision_kwargs} == d._precision_kwargs


def test_class_names_resolved_and_merged_with_classes(fake):
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", classes=[1], class_names=["car", "bus"])
    assert d.classes == [1, 2, 5]  # bicycle(1) + car(2) + bus(5), COCO order, deduped and sorted


def test_class_names_only(fake):
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", class_names=["person"])
    assert d.classes == [0]


def test_unknown_class_name_raises(fake):
    with pytest.raises(ValueError, match="Unknown class name 'not_a_class'"):
        det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", class_names=["not_a_class"])


def test_imgsz_classes_rect_default_to_ultralytics_defaults(fake):
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    assert (d.imgsz, d.classes, d.rect, d.agnostic_nms, d.embed) == (640, None, False, False, None)


@pytest.mark.parametrize("method", ["detect", "detect_frames"])
def test_imgsz_classes_rect_passed_to_predict(fake, synthetic_video, method):
    video, _ = synthetic_video
    d = det_mod.Detector(
        model=DetectorModel.YOLO26n,
        device="cpu",
        imgsz=1280,
        classes=[2, 5],
        rect=True,
        agnostic_nms=True,
        embed=[10],
        fast=False,  # FakeModel has no .predictor
    )
    if method == "detect":
        d.detect(str(video), verbose=False)
    else:
        d.detect_frames(str(video), [0], verbose=False)
    seen = d.model.last_kwargs
    assert (seen["imgsz"], seen["classes"], seen["rect"]) == (1280, [2, 5], True)
    assert (seen["agnostic_nms"], seen["embed"]) == (True, [10])


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


# --- Detector(fast=..., batch=...): dispatch to detect_fast(), no real model needed here ---


def test_detector_fast_batch_default_attributes(fake):
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    assert d.fast is True
    assert d.batch == 8


def test_detect_defaults_redirect_to_detect_fast(fake, synthetic_video, monkeypatch):
    """The instance default (fast=True) must actually redirect, not just be stored."""
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")

    calls = []

    def spy(self, input_video, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"sentinel": [1]})

    monkeypatch.setattr(det_mod.Detector, "detect_fast", spy)
    d.detect(str(video), verbose=False)  # fast/batch left at their instance defaults

    assert len(calls) == 1
    assert calls[0]["batch"] == 8  # d.batch's default, forwarded by detect()


def test_detector_fast_false_uses_traditional_pipeline(fake, synthetic_video, monkeypatch):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)

    def boom(self, *args, **kwargs):
        raise AssertionError("detect_fast() should not be called when self.fast is False")

    monkeypatch.setattr(det_mod.Detector, "detect_fast", boom)
    d.detect(str(video), verbose=False)  # must not raise


def test_detector_batch_is_forwarded_to_detect_fast(fake, synthetic_video, monkeypatch):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", batch=4)

    sentinel = pd.DataFrame({"sentinel": [1]})
    calls = []

    def spy(self, input_video, **kwargs):
        calls.append((input_video, kwargs))
        return sentinel

    monkeypatch.setattr(det_mod.Detector, "detect_fast", spy)
    out = d.detect(
        str(video), iou_file="x.txt", video_index=1, video_tot=3,
        start_frame=0, end_frame=5, verbose=False, message="foo",
    )

    assert out is sentinel
    assert len(calls) == 1
    got_video, got_kwargs = calls[0]
    assert got_video == str(video)
    assert got_kwargs == {
        "iou_file": "x.txt", "video_index": 1, "video_tot": 3,
        "start_frame": 0, "end_frame": 5, "verbose": False, "message": "foo", "batch": 4,
        "return_df": True,
    }


def test_mutating_fast_after_construction_takes_effect(fake, synthetic_video, monkeypatch):
    """detector.fast = False after construction must change detect()'s behavior."""
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")  # fast=True by default

    def boom(self, *args, **kwargs):
        raise AssertionError("detect_fast() should not be called once self.fast is False")

    monkeypatch.setattr(det_mod.Detector, "detect_fast", boom)
    d.fast = False
    d.detect(str(video), verbose=False)  # must not raise


def test_detect_with_show_raises_when_fast(fake, synthetic_video):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")  # fast=True by default
    with pytest.raises(ValueError, match="show"):
        d.detect(str(video), show=True, verbose=False)


def test_detect_with_show_does_not_raise_when_not_fast(fake):
    # self.fast = False must bypass the fast/show ValueError check entirely, not just skip
    # the redirect. A nonexistent path is enough: it proves no exception comes from the
    # fast/show conflict check without needing to exercise the real display/plot code
    # (which FakeModel's Results stub doesn't implement).
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)
    out = d.detect("no_such_file.mp4", show=True, verbose=False)
    assert out.empty


@pytest.mark.model
def test_detect_default_matches_detect_fast_end_to_end(synthetic_video):
    # exercises the real redirect wiring (not mocked), on a real model, at the new default.
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", conf=0.001, nms=0.99, max_det=300)

    via_detect = d.detect(str(video), start_frame=0, end_frame=7, verbose=False)  # d.fast/d.batch defaults
    direct = d.detect_fast(str(video), start_frame=0, end_frame=7, verbose=False, batch=8)

    pd.testing.assert_frame_equal(via_detect, direct)


# --- detect_fast(): real model required (no `fake`), so these download/load real weights ---


def _sorted_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Sort detections into a deterministic row order for comparing two runs."""
    return df.sort_values(["frame", "class", "x", "y"]).reset_index(drop=True)


@pytest.mark.model
def test_detect_fast_matches_detect(synthetic_video):
    video, _ = synthetic_video
    # The synthetic video has no real-world objects, so a realistic confidence threshold
    # finds nothing on either path; loosen conf/nms so some (junk) boxes survive on both,
    # giving detect_fast() something non-trivial to match against detect().
    # fast=False: this test is specifically about the traditional pipeline matching
    # detect_fast(), which the default (fast=True) would otherwise hide.
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", conf=0.001, nms=0.99, max_det=300, fast=False)

    full = d.detect(str(video), start_frame=0, end_frame=7, verbose=False)
    fast = d.detect_fast(str(video), start_frame=0, end_frame=7, verbose=False, batch=3)

    assert len(full) > 0  # otherwise this test isn't checking anything
    assert len(fast) == len(full)
    a, b = _sorted_numeric(full), _sorted_numeric(fast)
    assert list(a["class"]) == list(b["class"])
    for col in ("x", "y", "w", "h"):
        assert (a[col] - b[col]).abs().max() <= 1  # int-truncation rounding, not a real mismatch
    assert (a["conf"] - b["conf"]).abs().max() < 0.01


@pytest.mark.model
def test_detect_fast_matches_detect_rtdetr(synthetic_video):
    # RT-DETR (the model this speed work was about) has its own predictor/postprocess
    # path, distinct from YOLO's; check it separately.
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.RTDETRx, device="cpu", conf=0.001, nms=0.99, max_det=300, fast=False)

    full = d.detect(str(video), start_frame=0, end_frame=7, verbose=False)
    fast = d.detect_fast(str(video), start_frame=0, end_frame=7, verbose=False, batch=3)

    assert len(full) > 0
    assert len(fast) == len(full)
    a, b = _sorted_numeric(full), _sorted_numeric(fast)
    assert list(a["class"]) == list(b["class"])
    for col in ("x", "y", "w", "h"):
        assert (a[col] - b[col]).abs().max() <= 1
    assert (a["conf"] - b["conf"]).abs().max() < 0.01


@pytest.mark.model
def test_detect_fast_rejects_non_positive_batch():
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    with pytest.raises(ValueError, match="batch must be"):
        d.detect_fast("no_such_file.mp4", batch=0)


@pytest.mark.model
def test_detect_fast_missing_video_returns_empty():
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    out = d.detect_fast("no_such_file.mp4", verbose=False)
    assert out.empty
    assert list(out.columns) == d.DET_FIELDS


@pytest.mark.model
def test_detect_fast_writes_iou_file(synthetic_video, tmp_path):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", conf=0.001, nms=0.99, max_det=300)
    out_file = tmp_path / "scene_iou.txt"
    df = d.detect_fast(str(video), iou_file=str(out_file), start_frame=0, end_frame=7, verbose=False, batch=2)
    assert out_file.exists()
    assert len(df) > 0  # otherwise this test isn't checking anything
    written = pd.read_csv(out_file, header=None)
    assert len(written) == len(df)


@pytest.mark.model
def test_detect_fast_bootstraps_predictor_only_once(synthetic_video, monkeypatch):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", conf=0.05)

    calls = []
    original_predict = d.model.predict

    def spy(*args, **kwargs):
        calls.append(1)
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(d.model, "predict", spy)

    d.detect_fast(str(video), start_frame=0, end_frame=3, verbose=False, batch=2)
    d.detect_fast(str(video), start_frame=4, end_frame=7, verbose=False, batch=2)

    # exactly one throwaway model.predict() call to bootstrap the predictor (see
    # _ensure_predictor()); every other frame goes through predictor.inference() directly.
    assert len(calls) == 1


@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_detect_fast_batch_one_matches_detect_on_cuda(synthetic_video):
    # detect_fast()'s speed gain at batch=1 comes only from skipping model.predict()'s
    # per-frame setup, so it must reproduce detect() exactly there -- including on CUDA
    # with half=True, where batch>1 is documented to shift a small fraction of results
    # (FP16 batching numerics; see detect_fast()'s docstring). This is the guarantee that
    # distinguishes "still correct, just faster" from "a different, approximate result".
    video, _ = synthetic_video
    d = det_mod.Detector(
        model=DetectorModel.RTDETRx, device="cuda:0", half=True, conf=0.001, nms=0.99, max_det=300, fast=False
    )

    full = d.detect(str(video), start_frame=0, end_frame=19, verbose=False)
    fast = d.detect_fast(str(video), start_frame=0, end_frame=19, verbose=False, batch=1)

    assert len(full) > 0
    assert len(fast) == len(full)
    a, b = _sorted_numeric(full), _sorted_numeric(fast)
    assert list(a["class"]) == list(b["class"])
    for col in ("x", "y", "w", "h", "conf"):
        assert (a[col] - b[col]).abs().max() == 0


# --- streaming output, resume and return_df: a fake infer() stands in for the model ---


def _fake_prepare(calls=None):
    """prepare() for Detector._run_batched: deterministic boxes derived from each frame's pixels."""

    def prepare():
        def infer(im0s):
            if calls is not None:
                calls.append(len(im0s))
            out = []
            for im in im0s:
                if int(im[::7, ::7].sum()) % 3 == 0:  # some frames without detections
                    out.append(np.empty((0, 6), np.float32))
                    continue
                m = float(im.mean())
                out.append(np.array([[m - 90.3, m + 1.5, m + 10.7, m + 20.2, 0.876, 2],
                                     [m + 3.3, m - 0.25, m + 7.9, m + 9.1, 0.455, 0]], np.float32))
            return out

        return infer

    return prepare


def _run(d, video, iou_file=None, calls=None, **kw):
    kw.setdefault("verbose", False)
    return d._run_batched(str(video), _fake_prepare(calls), iou_file=iou_file, **kw)


def test_arrays_to_df_matches_results_to_df(fake):
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    rng = np.random.default_rng(0)
    frame_ids, arrays, dicts = [3, 4, 5, 9, 11], [], []
    for f in frame_ids:
        n = int(rng.integers(0, 5))
        a = np.column_stack([rng.uniform(-5, 600, (n, 4)), rng.uniform(0, 1, n),
                             rng.integers(0, 80, n)]).astype(np.float32)
        arrays.append(a)
        dicts += [{"frame": f, "res": -1, "x": float(x1), "y": float(y1), "x2": float(x2), "y2": float(y2),
                   "conf": float(cf), "class": int(c)} for x1, y1, x2, y2, cf, c in a]
    old, new = d._results_to_df(dicts), d._arrays_to_df(frame_ids, arrays)
    assert len(old) > 0
    pd.testing.assert_frame_equal(new, old)
    assert new.to_csv(index=False, header=False) == old.to_csv(index=False, header=False)


def test_run_batched_streams_file_equal_to_returned_df(fake, synthetic_video, tmp_path):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    out = tmp_path / "v_iou.txt"
    df = _run(d, video, str(out), batch=4, end_frame=40)
    assert len(df) > 0
    assert out.exists() and not Path(f"{out}.part").exists()
    assert out.read_text() == df.to_csv(index=False, header=False)


def test_run_batched_file_does_not_depend_on_batch(fake, synthetic_video, tmp_path):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    a, b = tmp_path / "a_iou.txt", tmp_path / "b_iou.txt"
    _run(d, video, str(a), batch=1, end_frame=30)
    _run(d, video, str(b), batch=7, end_frame=30)
    assert a.read_text() == b.read_text()


def test_run_batched_return_df_false_returns_none_and_writes_file(fake, synthetic_video, tmp_path):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    ref, out = tmp_path / "ref_iou.txt", tmp_path / "v_iou.txt"
    _run(d, video, str(ref), batch=4, end_frame=20)
    assert _run(d, video, str(out), batch=4, end_frame=20, return_df=False) is None
    assert out.read_text() == ref.read_text()


def test_run_batched_return_df_false_needs_iou_file(fake, synthetic_video):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    with pytest.raises(ValueError, match="iou_file"):
        _run(d, video, None, return_df=False)


def test_run_batched_resumes_from_part_file(fake, synthetic_video, tmp_path):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    ref = tmp_path / "ref_iou.txt"
    _run(d, video, str(ref), batch=4, end_frame=60)
    text = ref.read_text()
    lines = text.splitlines(keepends=True)
    cut = len(lines) // 2

    out = tmp_path / "v_iou.txt"
    Path(f"{out}.part").write_text("".join(lines[:cut]) + lines[cut][:5])  # "crashed" mid-line
    calls = []
    df = _run(d, video, str(out), batch=4, end_frame=60, calls=calls)

    assert out.read_text() == text
    assert not Path(f"{out}.part").exists()
    assert sum(calls) < 61  # only the remainder was detected again
    pd.testing.assert_frame_equal(df, pd.read_csv(ref, header=None, names=det_mod.Detector.DET_FIELDS))


def test_resume_part_file_edge_cases(tmp_path):
    resume = det_mod.Detector._resume_part_file
    p = tmp_path / "x_iou.txt.part"

    p.write_bytes(b"")
    assert resume(p) is None

    p.write_bytes(b"12,-1,3")  # only a partial line
    assert resume(p) is None and p.read_bytes() == b""

    p.write_bytes(b"4,-1,1,1,1,1,0.5,0\n4,-1,2,2,2,2,0.5,0\n")  # one frame only
    assert resume(p) == 4 and p.read_bytes() == b""

    head = b"1,-1,1,1,1,1,0.5,0\n1,-1,2,2,2,2,0.5,0\n"
    p.write_bytes(head + b"2,-1,3,3,3,3,0.5,0\n2,-1,4,4,4,4,0.5,0\n2,-1,5")
    assert resume(p) == 2 and p.read_bytes() == head


def test_resume_part_file_reads_further_back_than_one_block(tmp_path):
    # the last frame's rows span more than the 1 MiB tail read first, so it must look further back
    p = tmp_path / "x_iou.txt.part"
    head = b"0,-1,1,1,1,1,0.5,0\n" * 1000
    p.write_bytes(head + b"7,-1,100,200,30,40,0.25,2\n" * 50_000 + b"7,-1,1")
    assert p.stat().st_size > 1 << 20
    assert det_mod.Detector._resume_part_file(p) == 7
    assert p.read_bytes() == head


def test_detect_traditional_return_df_false(fake, synthetic_video, tmp_path):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)
    out = tmp_path / "a_iou.txt"
    assert d.detect(str(video), iou_file=str(out), verbose=False, return_df=False) is None
    assert out.exists()
    with pytest.raises(ValueError, match="iou_file"):
        d.detect(str(video), verbose=False, return_df=False)


def test_detect_batch_skips_return_df_and_clears_stale_part_on_overwrite(fake, synthetic_video, tmp_path,
                                                                        monkeypatch):
    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu")
    stale = tmp_path / "scene_iou.txt.part"
    seen = []

    def spy(self, input_video, **kwargs):
        seen.append((kwargs["return_df"], stale.exists()))

    monkeypatch.setattr(det_mod.Detector, "detect_fast", spy)

    stale.write_text("0,-1,1,1,1,1,0.5,0\n")
    d.detect_batch([str(video)], output_path=str(tmp_path), is_overwrite=True, verbose=False)
    assert seen == [(False, False)]  # regenerating: stale partial run removed, table not built

    stale.write_text("0,-1,1,1,1,1,0.5,0\n")
    d.detect_batch([str(video)], output_path=str(tmp_path), verbose=False)
    assert seen[-1] == (False, True)  # otherwise the partial run is kept to resume from

