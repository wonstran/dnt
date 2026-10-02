import numpy as np
import pytest

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


def test_dino_batch_size_does_not_change_the_embeddings(fake_transformers):
    crops = [RED, BLUE, RED, BLUE, RED]
    a = DinoEncoder("m", None, "cpu", 1).encode(crops)
    b = DinoEncoder("m", None, "cpu", 5).encode(crops)
    assert np.allclose(a, b, atol=1e-6)


def test_dino_empty_input_gives_an_empty_matrix(fake_transformers):
    assert DinoEncoder("m", None, "cpu", 4).encode([]).shape == (0, 8)


def test_dino_weights_override_the_model_id(fake_transformers, tmp_path):
    DinoEncoder("facebook/dinov2-small", str(tmp_path), "cpu", 2)
    assert fake_transformers["source"] == str(tmp_path)


def test_dino_weights_sha_identifies_the_loaded_parameters(fake_transformers):
    a = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    again = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    fake_transformers["seed"] = 1  # the same model name now resolves to other weights
    other = DinoEncoder("facebook/dinov2-small", None, "cpu", 2)
    assert a.weights_sha == again.weights_sha == parameters_digest(a._model)
    assert other.weights_sha != a.weights_sha


def test_an_unavailable_accelerator_falls_back_to_the_resolved_device(fake_transformers):
    enc = DinoEncoder("m", None, "cuda", 2)
    assert enc.device == resolve_device("cuda")


def test_reid_embeddings_and_arguments(fake_torchreid, tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    enc = ReidEncoder("osnet_x1_0", str(weights), "cpu", 2)
    e = enc.encode([RED, BLUE, RED])
    assert e.shape == (3, 4) and e.dtype == np.float32 and enc.dim == 4
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    assert fake_torchreid["model_name"] == "osnet_x1_0"
    assert fake_torchreid["model_path"] == str(weights)
    assert enc.weights_sha == weights_digest(weights)
    assert enc.encode([]).shape == (0, 4)


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
