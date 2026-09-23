import numpy as np
import torch

from dnt.detect.signal import detector as sig


def test_model_does_not_request_imagenet_weights(monkeypatch):
    seen = {}
    real = sig.models.resnet18

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(weights=None)

    monkeypatch.setattr(sig.models, "resnet18", spy)
    sig.Model(2)
    assert "weights" in seen and seen["weights"] is None


def test_weights_loaded_with_map_location(monkeypatch, tmp_path):
    path = tmp_path / "w.pt"
    torch.save(sig.Model(2).state_dict(), path)
    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(sig.torch, "load", spy)
    sig.SignalDetector(det_zones=[(0, 0, 10, 10)], weights=str(path), device="cpu")
    assert seen.get("map_location") == "cpu"


def test_last_short_batch(synthetic_video):
    video, _ = synthetic_video
    d = object.__new__(sig.SignalDetector)
    d.det_zones = [(0, 0, 10, 10), (10, 10, 10, 10)]
    d.batchsz = 64
    d.threshold = 0.5
    d.device = "cpu"
    d.predict = lambda batch: np.zeros((len(batch), len(d.det_zones)))
    df = d.detect(str(video))
    assert len(df) == 150 * 2
    assert (df.groupby("frame").size() == 2).all()
