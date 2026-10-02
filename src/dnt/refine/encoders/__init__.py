"""Appearance encoders (spec 5.5): the protocol, a factory, and the dependency check.

Heavy libraries are imported only inside the encoder constructors, so importing this package
never needs ``transformers`` or ``torchreid``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..io import sha256_file

if TYPE_CHECKING:
    from ..config import EncoderConfig

#: For each encoder kind: the module it needs and the pip extra that provides it.
REQUIRES = {"dino": ("transformers", "refine-dino"), "reid": ("torchreid", "refine-reid")}
DEFAULT_REID_MODEL = "osnet_x1_0"


class AppearanceEncoder(Protocol):
    """Turns RGB crops into L2-normalized float32 embeddings."""

    name: str
    model_name: str
    preprocess_id: str
    weights_sha: str | None

    @property
    def dim(self) -> int:
        """Return the embedding width."""
        ...

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return an ``(N, dim)`` float32 array of unit-norm rows for ``N`` RGB crops."""
        ...


def default_reid_weights() -> Path:
    """Return the OSNet MSMT17 weights shipped with dnt (the ``reid`` default for pedestrians)."""
    return Path(__file__).resolve().parents[2] / "track" / "reid_weights" / "osnet_x1_0_msmt17.pt"


def weights_digest(path) -> str:
    """Return the SHA-256 of a weights file, or of a weights directory's files and their names."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(q for q in p.rglob("*") if q.is_file()):
            h.update(f.relative_to(p).as_posix().encode())
            h.update(b"\0")
            h.update(sha256_file(f).encode())
            h.update(b"\0")
        return h.hexdigest()
    raise ValueError(f"encoder weights not found: {p}")


def parameters_digest(model) -> str:
    """Return the SHA-256 of a torch module's parameters and buffers (names and float32 values).

    This identifies the weights a Hub model name resolved to, so the feature cache misses when
    the same name later gives different weights.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(b"\0")
        h.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def weights_identity(cfg: EncoderConfig, target: str = "person") -> str | None:
    """Return the digest of the local weights file the encoder would load, or ``None``.

    ``None`` means a Hub model, which is identified by its loaded parameters instead
    (``parameters_digest``). A long-lived ``TrackRefiner`` compares this value to notice a
    weights file that was replaced in place.
    """
    if cfg.kind == "none":
        return None
    if cfg.weights:
        return weights_digest(cfg.weights)
    if cfg.kind == "reid" and target == "person":
        return weights_digest(default_reid_weights())
    return None


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):  # a loaded module whose __spec__ is None
        return sys.modules.get(module) is not None


def check_encoder_dependencies(cfg: EncoderConfig) -> None:
    """Raise ``ImportError`` naming the pip extra if the encoder's package is not installed.

    Parameters
    ----------
    cfg : EncoderConfig
        Encoder settings; ``kind == "none"`` needs nothing.

    Raises
    ------
    ImportError
        If ``cfg.kind`` is ``dino`` or ``reid`` and its package cannot be imported.

    """
    if cfg.kind == "none":
        return
    module, extra = REQUIRES[cfg.kind]
    if not _importable(module):
        raise ImportError(
            f"encoder.kind={cfg.kind!r} needs the {module!r} package. Install it with "
            f"pip install 'dnt[{extra}]', or set encoder.kind: none to run without appearance."
        )


def make_encoder(cfg: EncoderConfig, target: str = "person") -> AppearanceEncoder:
    """Build the encoder selected by ``cfg.kind`` (spec 5.5).

    Parameters
    ----------
    cfg : EncoderConfig
        Encoder settings.
    target : str
        ``"person"`` or ``"vehicle"``. ``reid`` for a person defaults to the shipped OSNet
        MSMT17 weights; for a vehicle ``cfg.weights`` is required.

    Returns
    -------
    AppearanceEncoder
        A ready encoder (the model is loaded).

    Raises
    ------
    ValueError
        If ``cfg.kind`` is ``none``, or ``reid`` is requested for a vehicle without weights.

    """
    if cfg.kind == "dino":
        from .dino import DinoEncoder

        return DinoEncoder(cfg.model, cfg.weights, cfg.device, cfg.batch_size)
    if cfg.kind == "reid":
        weights = cfg.weights
        if weights is None:
            if target != "person":
                raise ValueError("encoder.weights is required for reid with the vehicle target")
            weights = str(default_reid_weights())
        # `model` defaults to a DINOv2 hub id; a torchreid model name never contains "/"
        model = DEFAULT_REID_MODEL if "/" in cfg.model else cfg.model
        from .reid import ReidEncoder

        return ReidEncoder(model, weights, cfg.device, cfg.batch_size)
    raise ValueError(f"no encoder for encoder.kind={cfg.kind!r}")
