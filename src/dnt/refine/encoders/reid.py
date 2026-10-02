"""torchreid appearance encoder (spec 5.5); needs the ``refine-reid`` extra (``torchreid``)."""

from __future__ import annotations

import contextlib
import importlib
import sys

import numpy as np

from . import weights_digest

#: Where ``FeatureExtractor`` lives: KaiyangZhou's deep-person-reid, then the PyPI repackaging.
_EXTRACTOR_MODULES = ("torchreid.utils", "torchreid.reid.utils")


def _feature_extractor_class():
    """Return torchreid's ``FeatureExtractor`` class from whichever layout is installed.

    Raises
    ------
    ImportError
        If neither ``torchreid.utils`` nor ``torchreid.reid.utils`` provides it.

    """
    tried = []
    for name in _EXTRACTOR_MODULES:
        try:
            return importlib.import_module(name).FeatureExtractor
        except (ImportError, AttributeError) as exc:
            tried.append(f"{name} ({exc})")
    tensorboard_hint = (
        " PyPI torchreid imports torch.utils.tensorboard but does not declare it: "
        "pip install tensorboard (the dnt[refine-reid] extra includes it)."
        if any("tensorboard" in t for t in tried)
        else ""
    )
    raise ImportError(
        "encoder.kind='reid' needs torchreid's FeatureExtractor, which was not found in "
        + "; ".join(tried)
        + "."
        + tensorboard_hint
        + " pip install 'dnt[refine-reid]' may install a repackaged torchreid without it; "
        "install the original instead with "
        "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git, "
        "or set encoder.kind: none to run without appearance."
    )


class ReidEncoder:
    """Person or vehicle re-identification embeddings from a torchreid feature extractor."""

    name = "reid"
    preprocess_id = "reid-torchreid-256x128-rgb-v2"  # v1 fed BGR

    def __init__(self, model: str, weights: str, device: str, batch_size: int):
        """Load the extractor.

        Parameters
        ----------
        model : str
            torchreid model name, for example ``osnet_x1_0``.
        weights : str
            Path of the weights file.
        device : str
            ``auto``, ``cpu``, ``cuda[:N]``, ``xpu`` or ``mps`` (unavailable ones fall back).
        batch_size : int
            Crops per forward pass.

        """
        from ..._device import resolve_device

        feature_extractor = _feature_extractor_class()

        self.model_name = model
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.weights_sha = weights_digest(weights)
        # torchreid prints "Successfully loaded pretrained weights ..." to stdout even with
        # verbose=False; send it to stderr so stdout stays clean (dnt-refine prints JSON there)
        with contextlib.redirect_stdout(sys.stderr):
            self._extractor = feature_extractor(
                model_name=model, model_path=str(weights), device=self.device, verbose=False
            )
        self._dim = int(self._embed([np.zeros((64, 32, 3), np.uint8)]).shape[1])

    @property
    def dim(self) -> int:
        """Return the embedding width."""
        return self._dim

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        # torchreid's FeatureExtractor turns an ndarray into an image with T.ToPILImage(), which
        # reads it as RGB, so our RGB crops go in unchanged
        feats = self._extractor(list(crops)).detach().cpu().numpy().astype(np.float32)
        return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return unit-norm float32 embeddings, one row per RGB crop."""
        if not len(crops):
            return np.empty((0, self._dim), dtype=np.float32)
        parts = [
            self._embed(crops[i : i + self.batch_size])
            for i in range(0, len(crops), self.batch_size)
        ]
        return np.concatenate(parts).astype(np.float32)
