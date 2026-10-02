"""DINOv2 appearance encoder (spec 5.5); needs the ``refine-dino`` extra (``transformers``)."""

from __future__ import annotations

import numpy as np

from . import parameters_digest

SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(crop: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize an RGB crop to ``size`` on its long side, pad it to a square, and normalize it.

    The padding is the ImageNet mean, so it is exactly 0 after normalization.

    Parameters
    ----------
    crop : numpy.ndarray
        ``uint8`` RGB crop ``(h, w, 3)``.
    size : int
        Side of the square output.

    Returns
    -------
    numpy.ndarray
        float32 array ``(3, size, size)``.

    """
    import cv2

    h, w = crop.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(crop, (nw, nh), interpolation=interp).astype(np.float32) / 255.0
    canvas = np.broadcast_to(_MEAN, (size, size, 3)).copy()
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = img
    return ((canvas - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)


class DinoEncoder:
    """DINOv2 CLS-token embeddings (``facebook/dinov2-small`` by default)."""

    name = "dino"
    preprocess_id = "dino-letterbox224-imagenet-v1"

    def __init__(self, model: str, weights: str | None, device: str, batch_size: int):
        """Load the model onto the resolved device.

        Parameters
        ----------
        model : str
            Hugging Face model id.
        weights : str or None
            Local model directory used instead of ``model`` when given.
        device : str
            ``auto``, ``cpu``, ``cuda[:N]``, ``xpu`` or ``mps`` (unavailable ones fall back).
        batch_size : int
            Crops per forward pass.

        """
        import torch
        from transformers import AutoModel

        from ..._device import resolve_device

        self._torch = torch
        self.model_name = model
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        loaded = AutoModel.from_pretrained(weights or model).eval()
        # identify what the name resolved to, so a Hub model that changes under the same name
        # makes the feature cache miss
        self.weights_sha = parameters_digest(loaded)
        self._model = loaded.to(self.device)
        self._dim = int(self._model.config.hidden_size)

    @property
    def dim(self) -> int:
        """Return the embedding width."""
        return self._dim

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return unit-norm float32 CLS embeddings, one row per RGB crop."""
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch = torch.from_numpy(
                    np.stack([letterbox(c) for c in crops[i : i + self.batch_size]])
                ).to(self.device)
                feats = self._model(pixel_values=batch).last_hidden_state[:, 0]
                feats = torch.nn.functional.normalize(feats.float(), dim=1)
                out.append(feats.cpu().numpy())
        if not out:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)
