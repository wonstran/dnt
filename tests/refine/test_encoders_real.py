import numpy as np
import pytest

from dnt.refine.config import EncoderConfig
from dnt.refine.encoders import make_encoder

pytestmark = pytest.mark.model


def _crops():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (96, 48, 3), dtype=np.uint8) for _ in range(3)]


def test_real_dino_shape_and_norm():
    pytest.importorskip("transformers")
    enc = make_encoder(EncoderConfig(kind="dino", device="cpu"))
    e = enc.encode(_crops())
    assert e.shape == (3, enc.dim) and enc.dim == 384
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)


def test_real_osnet_shape_and_norm():
    pytest.importorskip("torchreid")
    enc = make_encoder(EncoderConfig(kind="reid", device="cpu"), "person")
    e = enc.encode(_crops())
    assert e.shape == (3, enc.dim) and enc.dim == 512
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
