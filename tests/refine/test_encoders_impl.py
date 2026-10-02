import math

import numpy as np
import pytest
import torch

from dnt._device import resolve_device
from dnt.refine.config import EncoderConfig
from dnt.refine.encoders import (
    default_reid_weights,
    make_encoder,
    parameters_digest,
    weights_digest,
)
from dnt.refine.encoders.dino import DinoEncoder, letterbox
from dnt.refine.encoders.reid import ReidEncoder

from ._fakes import install_fake_torchreid, install_fake_transformers


def _crop(channel, h=40, w=20):
    c = np.zeros((h, w, 3), np.uint8)
    c[..., channel] = 200
    return c


RED, BLUE = _crop(0), _crop(2)


@pytest.fixture
def fake_transformers(monkeypatch):
    return install_fake_transformers(monkeypatch)


@pytest.fixture
def fake_torchreid(monkeypatch):
    return install_fake_torchreid(monkeypatch)


def test_letterbox_keeps_the_aspect_ratio_and_pads_with_the_mean():
    out = letterbox(_crop(0, h=100, w=50))
    assert out.shape == (3, 224, 224) and out.dtype == np.float32
    assert np.all(out[:, :, :56] == 0) and np.all(out[:, :, 168:] == 0)  # padding = ImageNet mean
    assert np.abs(out[:, :, 56:168]).max() > 0


def test_letterbox_handles_a_tiny_crop():
    assert letterbox(np.full((2, 3, 3), 128, np.uint8)).shape == (3, 224, 224)


def test_dino_embeddings_are_unit_float32_and_input_dependent(fake_transformers):
    enc = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    e = enc.encode([RED, BLUE, RED])
    assert e.shape == (3, 8) and e.dtype == np.float32 and enc.dim == 8
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(e[0], e[2]) and not np.allclose(e[0], e[1])
    assert (enc.name, enc.model_name) == ("dino", "facebook/dinov2-small")
    assert len(enc.weights_sha) == 64
    assert fake_transformers["source"] == "facebook/dinov2-small"


def test_dino_embedding_follows_the_cls_token(fake_transformers):
    enc = DinoEncoder("m", None, "cpu", 2)
    pixels = torch.from_numpy(letterbox(RED))[None]
    cls = enc._model.proj(pixels.mean(dim=(2, 3)))  # the fake's CLS token (position 0)
    want = torch.nn.functional.normalize(cls, dim=1).detach().numpy()
    assert np.allclose(enc.encode([RED]), want, atol=1e-6)


def test_dino_batch_size_does_not_change_the_embeddings(fake_transformers):
    crops = [RED, BLUE, RED, BLUE, RED]
    a = DinoEncoder("m", None, "cpu", 1).encode(crops)
    b = DinoEncoder("m", None, "cpu", 5).encode(crops)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7])
def test_dino_forwards_at_most_batch_size_crops_per_call(fake_transformers, batch_size):
    enc = DinoEncoder("m", None, "cpu", batch_size)
    enc.encode([RED, BLUE] * 3 + [RED])  # 7 crops
    sizes = fake_transformers["batches"]
    assert max(sizes) <= batch_size and sum(sizes) == 7
    assert len(sizes) == math.ceil(7 / batch_size)


def test_dino_empty_input_gives_an_empty_matrix(fake_transformers):
    assert DinoEncoder("m", None, "cpu", 4).encode([]).shape == (0, 8)


def test_dino_weights_override_the_model_id(fake_transformers, tmp_path):
    enc = DinoEncoder("facebook/dinov2-small", str(tmp_path), "cpu", 2)
    assert fake_transformers["source"] == str(tmp_path)
    assert enc.weights_sha == parameters_digest(enc._model) and len(enc.weights_sha) == 64


def test_dino_weights_sha_identifies_the_loaded_parameters(fake_transformers):
    a = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    again = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    fake_transformers["seed"] = 1  # the same model name now resolves to other weights
    other = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    assert a.weights_sha == again.weights_sha == parameters_digest(a._model)
    assert other.weights_sha != a.weights_sha


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr("dnt._device._available", lambda backend: backend == "cpu")


@pytest.mark.parametrize("requested", ["cuda", "cuda:1", "xpu", "mps", "auto"])
def test_an_unavailable_accelerator_falls_back_to_cpu(fake_transformers, cpu_only, requested):
    enc = DinoEncoder("m", None, requested, 2)
    assert enc.device == resolve_device(requested) == "cpu"


@pytest.mark.parametrize("requested", ["cuda", "auto"])
def test_reid_falls_back_to_cpu_when_no_accelerator_exists(
    fake_torchreid, cpu_only, tmp_path, requested
):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    enc = ReidEncoder("osnet_x1_0", str(weights), requested, 2)
    assert enc.device == "cpu" and fake_torchreid["device"] == "cpu"


def test_reid_embeddings_and_arguments(fake_torchreid, tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    enc = ReidEncoder("osnet_x1_0", str(weights), "cpu", 2)
    e = enc.encode([RED, BLUE, RED])
    assert e.shape == (3, 4) and e.dtype == np.float32 and enc.dim == 4
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    assert fake_torchreid["model_name"] == "osnet_x1_0"
    assert fake_torchreid["model_path"] == str(weights)
    assert fake_torchreid["device"] == enc.device == "cpu"
    assert enc.weights_sha == weights_digest(weights)
    assert enc.encode([]).shape == (0, 4)


def test_reid_receives_bgr_arrays_for_rgb_crops(fake_torchreid, tmp_path):
    # the real torchreid extractor treats ndarray input as BGR; the fake mimics that
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    enc = ReidEncoder("osnet_x1_0", str(weights), "cpu", 2)
    red, blue = enc.encode([RED, BLUE])  # RGB crops: red has R only, blue has B only
    assert red[0] == red[:3].max() and red[0] > red[2]
    assert blue[2] == blue[:3].max() and blue[2] > blue[0]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7])
def test_reid_forwards_at_most_batch_size_crops_per_call(fake_torchreid, tmp_path, batch_size):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    enc = ReidEncoder("osnet_x1_0", str(weights), "cpu", batch_size)
    fake_torchreid["batches"].clear()  # drop the construction-time probe call
    enc.encode([RED, BLUE] * 3 + [RED])  # 7 crops
    sizes = fake_torchreid["batches"]
    assert max(sizes) <= batch_size and sum(sizes) == 7
    assert len(sizes) == math.ceil(7 / batch_size)


def test_reid_batch_size_does_not_change_the_embeddings(fake_torchreid, tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    crops = [RED, BLUE, RED, BLUE, RED]
    a = ReidEncoder("osnet_x1_0", str(weights), "cpu", 1).encode(crops)
    b = ReidEncoder("osnet_x1_0", str(weights), "cpu", 9).encode(crops)
    assert np.allclose(a, b, atol=1e-6)


def test_make_encoder_selects_by_kind(fake_transformers, fake_torchreid):
    assert isinstance(make_encoder(EncoderConfig(kind="dino", device="cpu")), DinoEncoder)
    enc = make_encoder(EncoderConfig(kind="reid", device="cpu"), "person")
    assert isinstance(enc, ReidEncoder)
    # `model` is still the DINOv2 default, so the OSNet default and the shipped weights are used
    assert fake_torchreid["model_name"] == "osnet_x1_0"
    assert fake_torchreid["model_path"] == str(default_reid_weights())
    named = make_encoder(EncoderConfig(kind="reid", model="osnet_ain_x1_0", device="cpu"))
    assert named.model_name == "osnet_ain_x1_0"


def test_make_encoder_reid_for_a_vehicle_uses_the_given_weights(fake_torchreid, tmp_path):
    w = tmp_path / "veri.pt"
    w.write_bytes(b"v")
    enc = make_encoder(EncoderConfig(kind="reid", weights=str(w), device="cpu"), "vehicle")
    assert fake_torchreid["model_path"] == str(w) and enc.weights_sha == weights_digest(w)
