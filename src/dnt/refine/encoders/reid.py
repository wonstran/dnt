"""torchreid appearance encoder (spec 5.5); needs the ``refine-reid`` extra (``torchreid``)."""

from __future__ import annotations

import numpy as np

from . import weights_digest


class ReidEncoder:
    """Person or vehicle re-identification embeddings from a torchreid feature extractor."""

    name = "reid"
    preprocess_id = "reid-torchreid-256x128-v1"

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
        from torchreid.utils import FeatureExtractor

        from ..._device import resolve_device

        self.model_name = model
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.weights_sha = weights_digest(weights)
        self._extractor = FeatureExtractor(
            model_name=model, model_path=str(weights), device=self.device, verbose=False
        )
        self._dim = int(self._embed([np.zeros((64, 32, 3), np.uint8)]).shape[1])

    @property
    def dim(self) -> int:
        """Return the embedding width."""
        return self._dim

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
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
