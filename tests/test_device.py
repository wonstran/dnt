import pytest

from dnt import _device


@pytest.fixture
def cuda_only(monkeypatch):
    monkeypatch.setattr(_device, "_available", lambda b: b in ("cuda", "cpu"))


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(_device, "_available", lambda b: b == "cpu")


@pytest.fixture
def xpu_only(monkeypatch):
    monkeypatch.setattr(_device, "_available", lambda b: b in ("xpu", "cpu"))


def test_auto_prefers_cuda(cuda_only):
    assert _device.resolve_device("auto") == "cuda"
    assert _device.resolve_device(None) == "cuda"


def test_indexed_cuda_kept(cuda_only):
    assert _device.resolve_device("CUDA:1") == "cuda:1"


def test_unavailable_backend_falls_back_to_cpu(cpu_only):
    assert _device.resolve_device("cuda:0") == "cpu"
    assert _device.resolve_device("mps") == "cpu"


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Invalid device"):
        _device.resolve_device("tpu")


@pytest.mark.parametrize(("device", "half", "expected"), [
    ("cuda", True, True), ("cuda:1", True, True), ("cpu", True, False), ("mps", True, False), ("cuda:0", False, False),
])
def test_half_allowed(device, half, expected):
    assert _device.half_allowed(device, half) is expected


@pytest.mark.parametrize(("device", "expected"), [
    ("cuda", "0"), ("cuda:1", "1"), ("cpu", "cpu"), ("mps", "mps"),
    ("xpu", "cpu"), ("xpu:1", "cpu"),
])
def test_to_boxmot_device_mapping(device, expected):
    assert _device.to_boxmot_device(device) == expected


def test_xpu_host_tracks_on_cpu(xpu_only):
    from dnt.track.tracker import Tracker, _plan_tracker

    plan = _plan_tracker(Tracker().boxmot_config, device="auto")
    resolved = _device.resolve_device(plan.device)
    assert _device.to_boxmot_device(resolved) == "cpu"


@pytest.mark.parametrize(("half", "expected"), [(True, {"quantize": 16}), (False, {"quantize": None})])
def test_predict_precision_kwargs_uses_quantize_when_supported(monkeypatch, half, expected):
    monkeypatch.setattr(_device, "_quantize_supported", True)
    assert _device.predict_precision_kwargs(half) == expected


@pytest.mark.parametrize("half", [True, False])
def test_predict_precision_kwargs_falls_back_to_half(monkeypatch, half):
    monkeypatch.setattr(_device, "_quantize_supported", False)
    assert _device.predict_precision_kwargs(half) == {"half": half}


def test_quantize_field_supported_detects_installed_ultralytics(monkeypatch):
    monkeypatch.setattr(_device, "_quantize_supported", None)
    # whatever this installed Ultralytics actually supports, detection must not raise
    result = _device._quantize_field_supported()
    assert isinstance(result, bool)
    assert _device._quantize_supported is result  # cached


def test_detector_half_on_indexed_cuda(monkeypatch, cuda_only):
    from fake_ultralytics import FakeModel

    from dnt.detect.yolo import detector as det_mod

    monkeypatch.setattr(det_mod, "YOLO", FakeModel)
    d = det_mod.Detector(model=det_mod.DetectorModel.YOLO26n, device="cuda:0", half=True)
    assert d.device == "cuda:0"
    assert d.half is True
