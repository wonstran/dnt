"""Device and precision resolution shared by Detector, Segmentor and Tracker."""

from __future__ import annotations

import torch

_VALID = ("auto", "cuda", "xpu", "mps", "cpu")
_AUTO_ORDER = ("cuda", "xpu", "mps", "cpu")


def _available(backend: str) -> bool:
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "xpu":
        return (
            hasattr(torch, "xpu")
            and hasattr(torch.xpu, "is_available")
            and torch.xpu.is_available()
        )
    if backend == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return backend == "cpu"


def resolve_device(device: str | None = "auto") -> str:
    """Resolve a requested device into one that is available on this host.

    Parameters
    ----------
    device : str or None
        ``"auto"`` (or None), ``"cuda"``, ``"cuda:N"``, ``"xpu[:N]"``, ``"mps"`` or ``"cpu"``.

    Returns
    -------
    str
        The requested device if its backend is available, ``"cpu"`` otherwise;
        for ``"auto"`` the first available of cuda, xpu, mps, cpu (no index).

    Raises
    ------
    ValueError
        If the backend name is not supported.

    """
    requested = "auto" if device is None else str(device).lower().strip()
    backend = requested.split(":", maxsplit=1)[0]
    if backend not in _VALID:
        raise ValueError(
            f"Invalid device={device!r}. Choose one of {sorted(_VALID)} "
            "or backend:index like 'cuda:0'."
        )
    if backend == "auto":
        return next(b for b in _AUTO_ORDER if _available(b))
    return requested if _available(backend) else "cpu"


def half_allowed(device: str, half: bool) -> bool:
    """Return True only when half precision is requested and the device is CUDA."""
    return bool(half) and str(device).startswith("cuda")


_quantize_supported: bool | None = None


def _quantize_field_supported() -> bool:
    """Return True if the installed Ultralytics `model.predict()` accepts `quantize=`."""
    global _quantize_supported
    if _quantize_supported is None:
        from ultralytics.cfg import DEFAULT_CFG

        _quantize_supported = hasattr(DEFAULT_CFG, "quantize")
    return _quantize_supported


def predict_precision_kwargs(half: bool) -> dict[str, bool | int | None]:
    """Build the Ultralytics `model.predict()` precision kwarg(s) for a resolved `half` flag.

    Some Ultralytics releases deprecated `half=` in favor of a unified `quantize=` scheme;
    passing `half=` there logs a warning on every single call (dnt calls `predict()` once per
    frame), while passing `quantize=` to an Ultralytics that predates that scheme raises. This
    detects which the installed version accepts and returns the matching kwarg(s).

    Parameters
    ----------
    half : bool
        A resolved half-precision flag (e.g. from `half_allowed()`).

    Returns
    -------
    dict
        Either ``{"quantize": 16 or None}`` or ``{"half": half}``, to be splatted into
        `model.predict(**predict_precision_kwargs(...))`.

    """
    if _quantize_field_supported():
        return {"quantize": 16 if half else None}
    return {"half": half}


def to_boxmot_device(device: str) -> str:
    """Convert a resolved dnt device to BoxMOT's format (``"cuda:1"`` -> ``"1"``).

    Parameters
    ----------
    device : str
        A resolved device string (e.g., "cuda", "cuda:1", "cpu", "mps", "xpu").

    Returns
    -------
    str
        BoxMOT's device format. CUDA devices map to their index (or "0" if no
        index is given), "mps" passes through unchanged, and every other value
        (including "cpu" and any "xpu"/"xpu:N") maps to "cpu" since BoxMOT has
        no XPU support.

    """
    if device.startswith("cuda"):
        return device.split(":", maxsplit=1)[1] if ":" in device else "0"
    if device == "mps":
        return "mps"
    return "cpu"
