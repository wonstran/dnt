# dnt.refine Track Refinement — Plan 2 of 4: Appearance Encoders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `TrackRefiner.refine` real appearance evidence. With a video and `encoder.kind` of `dino` or `reid`, it crops each box, embeds the crops, caches the embeddings in `OUT.features.npz`, and feeds them to stage 1 (ID-switch splits) and stage 3 (fragment links). This replaces P1's "motion-only fallback" warning. Without a video, or with `encoder.kind: none`, nothing changes.

**Architecture:**
- **P1's `Appearance` interface stays.** A new provider, `VideoAppearance`, implements it. It reads frames with `cv2`, crops boxes, skips occluded crops, and encodes in batches through an `AppearanceEncoder`. Heavy libraries (`transformers`, `torchreid`) are imported only inside the encoder constructors.
- **Coarse then dense (spec §5.3).** `clean_embeddings` returns only the coarse samples (every `encoder.sample_every`-th observed frame of a raw track, clean crops only). Stage 1 finds candidates on them, then asks a new, optional `dense_embeddings` method for every clean frame around each candidate, rescoring it and moving the cut to the best frame. A provider without `dense_embeddings` (P1's `ArrayAppearance`) behaves exactly as in P1.
- **Cache.** `FeatureStore` holds embeddings per (raw track id, frame) and is saved as a deterministic `.npz` under a SHA-256 key of every input that could change an embedding. A key mismatch, a missing file, or a corrupt file is a cache miss, never an error.

**Tech Stack:** Python >=3.11, numpy, pandas, OpenCV, torch (already required). Optional extras: `transformers` (DINOv2), `torchreid` (OSNet). pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-09-27-track-refinement-design.md`](../specs/2026-09-27-track-refinement-design.md) (rev. 7). Section references such as "§5.3" point there. Plan 1 is `2026-09-28-track-refinement-p1-core.md` and is merged; read its "Global Constraints" too.

## Plan series

| Plan | Scope | Status |
|---|---|---|
| P1 Core | Package, I/O, config, events and ledger, all four stages (motion-only), `TrackRefiner.refine`, `dnt-refine run` | merged (`13abfb3`) |
| **P2 (this plan)** | Frame reader and crops, `dino` / `reid` encoders, feature cache and key, dense re-sampling in stage 1, wiring the real `Appearance` into `refine`, deferred dependency checks, extras | §5.3, §5.5, §9 `encoder`, §10 rows on encoders |
| P3 Verification | Evidence images, VLM backends, answer cache, votes, review HTML | not written |
| P4 Replay and audit | `apply`, `audit`, decisions, rounds, feature-cache replay rules, reference case, quickstart | not written |

## Global Constraints

- **Python and dependencies.** `requires-python = ">=3.11"`. **No new required dependencies.** `transformers` and `torchreid` are optional extras only: `refine-dino = ["transformers>=4.40"]`, `refine-reid = ["torchreid"]`, `refine = ["transformers>=4.40", "torchreid"]`. They are never imported at `import dnt.refine` time.
- **Dependency rule (§2.2).** `dnt.refine` imports only `dnt.shared`, `dnt.engine`, `dnt` (for `__version__`; and the package-private `dnt._device.resolve_device`, which is not a forbidden module), and third-party libraries. It must never import `dnt.track`, `dnt.detect`, `dnt.label`, `dnt.filter`, or `boxmot`. The OSNet weights shipped under `dnt/track/reid_weights/` are reached by file path only. `tests/test_refine_independence.py` enforces this and must keep passing.
- **Frame numbers are 0-based video frame indexes** (`cv2` position), the same numbers as in the track file.
- **Encoder output.** `AppearanceEncoder.encode(crops) -> (N, D) float32, L2-normalized rows`. Crops are RGB `uint8` arrays.
- **Coarse samples (§5.3).** For each raw track, the observed frames in order, ordinal `0, k, 2k, ...` with `k = encoder.sample_every`; only crops whose box has IoU below `encoder.occlusion_iou` with every other box in the frame (input and context) count as clean. The occlusion mask is computed by `primitives.occlusion_flags` on the raw work table, so it does not depend on decisions.
- **Cache key (§5.3).** SHA-256 over: the tracks file's SHA-256; the video fingerprint (whole-file SHA-256, size, frame count); the context file's SHA-256 or `"none"`; encoder name, model name, a digest of the weights actually loaded (DINOv2: of the loaded model parameters, so a Hub model name that resolves to new weights misses; OSNet: of the weights file), and the preprocessing id; `sample_every`; `occlusion_iou`; the crop padding; `FEATURES_VERSION`.
- **Determinism.** The cache file is byte-deterministic for the same content (sorted rows, numpy's fixed zip timestamps). The same inputs give the same ledger and output with and without a cache hit (on CPU).
- **Lint.** Ruff rules `E,F,I,UP,B,SIM,RUF,D` (line length 100, numpy-style docstrings); ASCII only in code, comments and docstrings; `zip(..., strict=True)`. Never add entries to the per-file lint baseline in `pyproject.toml`. `ruff format src/dnt/refine` must stay clean.
- **Do not touch** `site/` (committed build output) or the version numbers.

## Review Focus

The spec is silent on these inputs, and each would bite a person running this on real clips. Each line has a test in the task that owns the code.

1. **A track with no clean crops** (every box overlaps another, or lies outside the frame). Stage 1 skips it, stage 3 treats appearance as unknown, and nothing raises or produces NaN. (Tasks 5 and 7)
2. **A video that ends early or cannot be read.** A `ValueError` that names the frame, never silently missing embeddings. (Tasks 2 and 5)
3. **A stale or damaged cache.** A cache built from a different video, context file, encoder, weights (including the same model name resolving to different weights, and a weights file replaced in place), crop setting or `FEATURES_VERSION`, and any `.npz` that is truncated, garbage, or readable but malformed (wrong rank, dtype, width, non-finite values, duplicate keys), are never reused and never crash a run. (Tasks 3, 4 and 7)
4. **A detection file of the same run as context.** Its boxes duplicate the tracks' own boxes. That must not make every crop look occluded. (Task 7)
5. **Model loading and memory.** `refine_batch` loads the encoder once, not once per file. Crops are encoded in bounded batches, never all held at once. A requested accelerator that is missing falls back to CPU, and the batch size never changes the embeddings. (Tasks 3, 5 and 7)

---

### Task 1: Encoder package skeleton, dependency check, and extras

**Files:**
- Create: `src/dnt/refine/encoders/__init__.py`
- Modify: `pyproject.toml` (add three extras under `[project.optional-dependencies]`)
- Test: `tests/refine/test_encoders.py`

**Interfaces:**
- Consumes: `dnt.refine.config.EncoderConfig` (`kind`, `model`, `weights`, `device`, `batch_size`), `dnt.refine.io.sha256_file`.
- Produces (later tasks rely on these exact names):
  - `AppearanceEncoder` (Protocol): attributes `name: str`, `model_name: str`, `preprocess_id: str`, `weights_sha: str | None`; property `dim: int`; method `encode(crops: list[np.ndarray]) -> np.ndarray`.
  - `REQUIRES: dict[str, tuple[str, str]]` mapping kind to `(module, extra)`.
  - `check_encoder_dependencies(cfg: EncoderConfig) -> None` (raises `ImportError`).
  - `make_encoder(cfg: EncoderConfig, target: str = "person") -> AppearanceEncoder`.
  - `default_reid_weights() -> Path`, `weights_digest(path) -> str`, `DEFAULT_REID_MODEL = "osnet_x1_0"`.
  - `parameters_digest(model) -> str`: SHA-256 of a torch module's `state_dict` (names and float32 values). Identifies the weights a Hub model name resolved to.
  - `weights_identity(cfg: EncoderConfig, target: str = "person") -> str | None`: the digest of the local weights file the encoder would load (`cfg.weights`, or the shipped OSNet file for `reid` pedestrians), else `None` (a Hub model). Lets a long-lived `TrackRefiner` notice a replaced file.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_encoders.py`:

```python
import importlib.machinery
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from dnt.refine.config import EncoderConfig
from dnt.refine.encoders import (
    check_encoder_dependencies,
    default_reid_weights,
    make_encoder,
    parameters_digest,
    weights_digest,
    weights_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def _present(monkeypatch, name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    monkeypatch.setitem(sys.modules, name, mod)


def test_missing_dino_package_names_the_extra_and_the_alternative(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-dino\]") as err:
        check_encoder_dependencies(EncoderConfig(kind="dino"))
    assert "encoder.kind: none" in str(err.value)


def test_missing_reid_package_names_its_own_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchreid", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-reid\]"):
        check_encoder_dependencies(EncoderConfig(kind="reid"))


def test_kind_none_needs_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torchreid", None)
    check_encoder_dependencies(EncoderConfig(kind="none"))


def test_an_installed_package_passes(monkeypatch):
    _present(monkeypatch, "transformers")
    check_encoder_dependencies(EncoderConfig(kind="dino"))


def test_a_module_without_a_spec_still_counts_as_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    check_encoder_dependencies(EncoderConfig(kind="dino"))  # find_spec would raise ValueError


def test_parameters_digest_follows_the_weights_not_the_name():
    import torch

    def model(seed):
        torch.manual_seed(seed)
        return torch.nn.Linear(3, 4)

    assert parameters_digest(model(0)) == parameters_digest(model(0))
    assert len(parameters_digest(model(0))) == 64
    assert parameters_digest(model(0)) != parameters_digest(model(1))


def test_weights_identity_is_the_digest_of_the_local_file_or_none(tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"one")
    assert weights_identity(EncoderConfig(kind="reid", weights=str(w))) == weights_digest(w)
    w.write_bytes(b"two")  # replaced in place
    assert weights_identity(EncoderConfig(kind="reid", weights=str(w))) == weights_digest(w)
    assert weights_identity(EncoderConfig(kind="reid")) == weights_digest(default_reid_weights())
    assert weights_identity(EncoderConfig(kind="reid"), "vehicle") is None
    assert weights_identity(EncoderConfig(kind="dino")) is None
    assert weights_identity(EncoderConfig(kind="none")) is None
    with pytest.raises(ValueError, match="not found"):
        weights_identity(EncoderConfig(kind="dino", weights=str(tmp_path / "gone")))


def test_make_encoder_rejects_none_and_vehicle_reid_without_weights():
    with pytest.raises(ValueError, match="kind='none'"):
        make_encoder(EncoderConfig(kind="none"))
    with pytest.raises(ValueError, match="vehicle"):
        make_encoder(EncoderConfig(kind="reid"), "vehicle")


def test_the_shipped_osnet_weights_are_found():
    assert default_reid_weights().is_file()


def test_weights_digest_hashes_files_and_directories(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    d = tmp_path / "model"
    d.mkdir()
    (d / "a.bin").write_bytes(b"1")
    (d / "b.bin").write_bytes(b"2")
    assert weights_digest(f) == weights_digest(f) and len(weights_digest(f)) == 64
    first = weights_digest(d)
    assert first == weights_digest(d)
    (d / "b.bin").write_bytes(b"3")
    assert weights_digest(d) != first
    with pytest.raises(ValueError, match="not found"):
        weights_digest(tmp_path / "missing")


def test_extras_are_declared_and_not_required():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = meta["optional-dependencies"]
    assert extras["refine-dino"] == ["transformers>=4.40"]
    assert extras["refine-reid"] == ["torchreid"]
    assert set(extras["refine"]) == {"transformers>=4.40", "torchreid"}
    required = " ".join(meta["dependencies"])
    assert "transformers" not in required and "torchreid" not in required


def test_importing_the_package_loads_no_encoder_library():
    code = (
        "import sys, dnt.refine, dnt.refine.encoders; "
        "bad = [m for m in ('transformers', 'torchreid') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_encoders.py -q`
Expected: collection error `No module named 'dnt.refine.encoders'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/encoders/__init__.py`:

```python
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
```

In `pyproject.toml`, add after the `docs = [...]` line:

```toml
refine-dino = ["transformers>=4.40"]
refine-reid = ["torchreid"]
refine = ["transformers>=4.40", "torchreid"]
```

The `.dino` and `.reid` modules are created in Task 3. `make_encoder` imports them only after
its own checks, so Task 1's tests (which stop at those checks) pass without them.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_encoders.py tests/test_refine_independence.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/encoders/__init__.py pyproject.toml tests/refine/test_encoders.py
git commit -m "feat(refine): add the encoder protocol, factory, dependency check and extras" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Frame reader and box crops

**Files:**
- Create: `src/dnt/refine/crops.py`
- Test: `tests/refine/test_crops.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Uses the `synthetic_video` fixture from `tests/conftest.py` (150 frames, 320x240, 25 fps, three moving boxes).
- Produces:
  - `CROP_PAD = 1.1`, `SEEK_GAP = 100`.
  - `crop_box(frame: np.ndarray, box, pad: float = CROP_PAD) -> np.ndarray | None`: the RGB `uint8` crop of `box = (x, y, w, h)` from a BGR frame, enlarged by `pad` and clipped to the frame; `None` if the box is empty, non-finite, or has fewer than 2 pixels left on an axis.
  - `FrameReader(path)`: context manager; `frames(wanted: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]` yields `(index, BGR frame)` for the sorted, unique indexes; raises `ValueError` naming the frame if it cannot be read.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_crops.py`:

```python
import numpy as np
import pytest

from dnt.refine.crops import CROP_PAD, FrameReader, crop_box


def _frame():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :, 2] = 200  # BGR: red
    return img


def test_crop_is_rgb_and_padded():
    crop = crop_box(_frame(), (50, 20, 40, 60))
    assert crop.dtype == np.uint8 and crop.shape[2] == 3
    assert crop[..., 0].min() == 200 and crop[..., 2].max() == 0  # R first
    # 40 x 60 enlarged by 1.1 -> about 44 x 66
    assert abs(crop.shape[1] - 40 * CROP_PAD) <= 2 and abs(crop.shape[0] - 60 * CROP_PAD) <= 2


def test_crop_is_padded_then_clipped_to_the_frame():
    # (-20, -10, 60, 40) enlarged by 1.1 spans x -23..43 and y -12..32; the frame keeps 0..43, 0..32
    assert crop_box(_frame(), (-20, -10, 60, 40)).shape == (32, 43, 3)
    # the same box away from the edges keeps its full padded size: 66 wide, 44 high
    assert crop_box(_frame(), (60, 30, 60, 40)).shape == (44, 66, 3)


@pytest.mark.parametrize(
    "box",
    [(500, 20, 40, 60), (10, 10, 0, 50), (10, 10, 50, -3), (np.nan, 1, 5, 5), (200, 100, 5, 5)],
)
def test_empty_or_outside_boxes_give_none(box):
    assert crop_box(_frame(), box) is None


def test_reader_yields_requested_frames_in_order(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        got = list(reader.frames([40, 3, 3, 10]))
    assert [f for f, _ in got] == [3, 10, 40]
    assert all(img.shape == (240, 320, 3) for _, img in got)


def test_seeking_and_sequential_reads_agree(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        sequential = {f: img for f, img in reader.frames(range(0, 150))}
    with FrameReader(video) as reader:
        sparse = {f: img for f, img in reader.frames([149, 5, 120])}  # gaps > SEEK_GAP seek
    for f, img in sparse.items():
        assert np.array_equal(img, sequential[f])


def test_a_second_call_may_go_backwards(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        later = dict(reader.frames([60]))
        earlier = dict(reader.frames([7]))
    assert set(later) == {60} and set(earlier) == {7}


def test_a_frame_past_the_end_names_the_frame(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader, pytest.raises(ValueError, match="frame 500"):
        list(reader.frames([500]))


def test_an_unopenable_video_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot open video"):
        FrameReader(tmp_path / "missing.mp4")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_crops.py -q`
Expected: collection error `No module named 'dnt.refine.crops'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/crops.py`:

```python
"""Frame access and box crops for appearance encoding (spec 5.3)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

#: A box is enlarged by this factor on each axis before it is cropped. Part of the cache key.
CROP_PAD = 1.1
#: A jump of more than this many frames is done by seeking; a shorter one by decoding on.
SEEK_GAP = 100


def crop_box(frame: np.ndarray, box, pad: float = CROP_PAD) -> np.ndarray | None:
    """Return the RGB crop of ``box`` enlarged by ``pad``, or ``None`` if nothing is left.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image, as read by OpenCV.
    box : sequence of float
        ``(x, y, w, h)`` in pixels.
    pad : float
        Enlargement of the box on each axis before clipping to the frame.

    Returns
    -------
    numpy.ndarray or None
        ``uint8`` array ``(h, w, 3)`` in RGB order, or ``None`` when the box is empty,
        non-finite, or has fewer than 2 pixels left on an axis after clipping.

    """
    height, width = frame.shape[:2]
    x, y, w, h = (float(v) for v in box)
    if not (np.isfinite([x, y, w, h]).all() and w > 0 and h > 0):
        return None
    cx, cy = x + w / 2.0, y + h / 2.0
    # round, not floor/ceil: 100 * 1.1 / 2 is 55.00000000000001, which would add a pixel
    x0 = max(0, round(cx - w * pad / 2.0))
    x1 = min(width, round(cx + w * pad / 2.0))
    y0 = max(0, round(cy - h * pad / 2.0))
    y1 = min(height, round(cy + h * pad / 2.0))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.ascontiguousarray(frame[y0:y1, x0:x1, ::-1])


class FrameReader:
    """Read chosen frames of a video, decoding on over short gaps and seeking over long ones."""

    def __init__(self, path):
        """Open ``path``; raise ``ValueError`` if OpenCV cannot."""
        import cv2

        self.path = str(path)
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        self._next = 0

    def __enter__(self) -> FrameReader:
        """Return the reader."""
        return self

    def __exit__(self, *exc) -> None:
        """Release the video."""
        self.close()

    def close(self) -> None:
        """Release the video."""
        self._cap.release()

    def frames(self, wanted: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, BGR frame)`` for each unique wanted index, in increasing order.

        Raises
        ------
        ValueError
            If a frame cannot be read (for example, it is past the end of the video).

        """
        for f in sorted({int(v) for v in wanted}):
            if f < self._next or f - self._next > SEEK_GAP:
                self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, f)
                self._next = f
            while self._next < f:
                if not self._cap.grab():
                    raise ValueError(
                        f"cannot read frame {f} of {self.path}: "
                        f"the video ends at frame {self._next}"
                    )
                self._next += 1
            ok, img = self._cap.read()
            if not ok or img is None:
                raise ValueError(f"cannot read frame {f} of {self.path}")
            self._next += 1
            yield f, img
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_crops.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/crops.py tests/refine/test_crops.py
git commit -m "feat(refine): add the frame reader and box crops" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: DINOv2 and torchreid encoders

**Files:**
- Create: `src/dnt/refine/encoders/dino.py`, `src/dnt/refine/encoders/reid.py`
- Test: `tests/refine/_fakes.py` (fake `transformers` / `torchreid` modules, shared with Task 7), `tests/refine/test_encoders_impl.py` (runs everywhere), `tests/refine/test_encoders_real.py` (`@pytest.mark.model`, opt-in)

**Interfaces:**
- Consumes: Task 1 (`weights_digest`, `AppearanceEncoder` shape), `dnt._device.resolve_device`.
- Produces:
  - `dino.letterbox(crop, size=224) -> np.ndarray` float32 `(3, size, size)`.
  - `dino.DinoEncoder(model, weights, device, batch_size)` and `reid.ReidEncoder(model, weights, device, batch_size)`. Each has `name` (`"dino"` / `"reid"`), `model_name`, `preprocess_id`, `weights_sha`, `device` (the resolved device string), `dim`, and `encode(crops)`. `DinoEncoder.weights_sha` is `parameters_digest` of the loaded model (so the same Hub name resolving to new weights changes it); `ReidEncoder.weights_sha` is the digest of its weights file.
  - `_fakes.install_fake_transformers(monkeypatch) -> dict` and `_fakes.install_fake_torchreid(monkeypatch) -> dict`: install importable fake modules (with a real `__spec__`) and return a dict the test can read (`source`, `loads`, ...) and change (`seed`, the fake DINO weights).

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/_fakes.py`:

```python
"""Fake ``transformers`` and ``torchreid`` modules for the encoder tests."""

from __future__ import annotations

import importlib.machinery
import sys
import types

import torch


def _module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec works on it
    return mod


class _FakeDino(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.config = types.SimpleNamespace(hidden_size=8)
        self.proj = torch.nn.Linear(3, 8, bias=False)

    def forward(self, pixel_values):
        feats = self.proj(pixel_values.mean(dim=(2, 3)))
        tokens = feats[:, None, :].repeat(1, 5, 1)
        return types.SimpleNamespace(last_hidden_state=tokens)


def install_fake_transformers(monkeypatch):
    """Install a fake ``transformers``; set ``seen["seed"]`` to change what the model name loads."""
    seen = {"seed": 0, "loads": 0}

    class AutoModel:
        @staticmethod
        def from_pretrained(source):
            seen["source"] = source
            seen["loads"] += 1
            return _FakeDino(seen["seed"])

    mod = _module("transformers")
    mod.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return seen


def install_fake_torchreid(monkeypatch):
    """Install a fake ``torchreid`` whose extractor embeds a crop by its mean color."""
    seen = {"n": 0, "loads": 0}

    class FeatureExtractor:
        def __init__(self, model_name, model_path, device, verbose=True):
            seen.update(model_name=model_name, model_path=model_path, device=device)
            seen["loads"] += 1

        def __call__(self, images):
            seen["n"] += len(images)
            return torch.tensor(
                [[float(im[..., c].mean()) + 1.0 for c in range(3)] + [1.0] for im in images]
            )

    utils = _module("torchreid.utils")
    utils.FeatureExtractor = FeatureExtractor
    top = _module("torchreid")
    top.utils = utils
    monkeypatch.setitem(sys.modules, "torchreid", top)
    monkeypatch.setitem(sys.modules, "torchreid.utils", utils)
    return seen
```

Create `tests/refine/test_encoders_impl.py`:

```python
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
```

Create `tests/refine/test_encoders_real.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_encoders_impl.py -q`
Expected: collection error `No module named 'dnt.refine.encoders.dino'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/encoders/dino.py`:

```python
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
            Local model directory or file used instead of ``model`` when given.
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
```

Create `src/dnt/refine/encoders/reid.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_encoders.py tests/refine/test_encoders_impl.py tests/test_refine_independence.py -q`
Expected: all pass. (`tests/refine/test_encoders_real.py` is deselected by the default `-m` filter; run it with `-m model` only where the libraries and weights are installed.)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/encoders/dino.py src/dnt/refine/encoders/reid.py tests/refine/test_encoders_impl.py tests/refine/test_encoders_real.py
git commit -m "feat(refine): add the DINOv2 and torchreid encoders" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Feature cache, cache key, and dense-capable array provider

**Files:**
- Modify (replace whole file): `src/dnt/refine/features.py`
- Test: `tests/refine/test_features_cache.py`

**Interfaces:**
- Consumes: `io.sha256_file`; P1's `Appearance`, `ArrayAppearance`, `track_embeddings` (kept unchanged in behavior).
- Produces:
  - `FEATURES_VERSION = 1`.
  - `DenseAppearance` (Protocol): `Appearance` plus `dense_embeddings(raw_id, f0, f1) -> (frames, emb)` (every clean observed frame in `[f0, f1]`) and `prefetch_dense(windows: Sequence[tuple[int, int, int]]) -> None`.
  - `CoarseArrayAppearance(table, every=5)`: an `ArrayAppearance` whose `clean_embeddings` returns every `every`-th stored sample (by ordinal within the raw track) and whose `dense_embeddings` returns all stored samples; `prefetch_dense` does nothing. For tests and callers with precomputed features.
  - `dense_track_embeddings(appearance, lineage, f0, f1) -> (frames, emb)`: dense samples within `[f0, f1]` across a track's lineage spans, sorted by frame.
  - `features_key(*, tracks_sha, video, context_sha, encoder, sample_every, occlusion_iou, crop_pad) -> str` (64 hex chars). `video` is the fingerprint dict `{sha256, size, frame_count}`; `encoder` is anything with `name`, `model_name`, `weights_sha`, `preprocess_id`.
  - `FeatureStore(key)`: `key`, `dirty`, `has(raw_id, frame)`, `put(raw_id, frame, emb)`, `get(raw_id, frames) -> (len(frames), D) array`, `__len__`, `save(path) -> sha256 str`, and classmethod `load(path, key, dim=None) -> FeatureStore | None` (None on a missing, unreadable, key-mismatched, or structurally damaged file: wrong rank or dtype, an embedding width other than `dim` when `dim` is given, values that are not finite **after conversion to float32**, duplicate `(raw_id, frame)` rows). `put` raises `ValueError` for an embedding whose width differs from the store's.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_features_cache.py`:

```python
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from dnt.refine import features, io
from dnt.refine.features import (
    ArrayAppearance,
    CoarseArrayAppearance,
    FeatureStore,
    dense_track_embeddings,
    features_key,
    track_embeddings,
)


def _enc(**kw):
    base = dict(name="stub", model_name="m", weights_sha=None, preprocess_id="p1")
    return SimpleNamespace(**{**base, **kw})


BASE = dict(
    tracks_sha="t",
    video={"sha256": "v", "size": 10, "frame_count": 5},
    context_sha=None,
    encoder=_enc(),
    sample_every=5,
    occlusion_iou=0.3,
    crop_pad=1.1,
)


def test_key_is_a_stable_sha256():
    k = features_key(**BASE)
    assert k == features_key(**BASE) and len(k) == 64 and int(k, 16) >= 0


@pytest.mark.parametrize(
    "change",
    [
        {"tracks_sha": "t2"},
        {"video": {"sha256": "v2", "size": 10, "frame_count": 5}},
        {"video": {"sha256": "v", "size": 11, "frame_count": 5}},
        {"video": {"sha256": "v", "size": 10, "frame_count": 6}},
        {"context_sha": "c"},
        {"encoder": _enc(name="other")},
        {"encoder": _enc(model_name="m2")},
        {"encoder": _enc(weights_sha="w")},
        {"encoder": _enc(preprocess_id="p2")},
        {"sample_every": 4},
        {"occlusion_iou": 0.2},
        {"crop_pad": 1.2},
    ],
)
def test_key_changes_with_every_input(change):
    assert features_key(**{**BASE, **change}) != features_key(**BASE)


def test_key_changes_with_the_features_version(monkeypatch):
    before = features_key(**BASE)
    monkeypatch.setattr(features, "FEATURES_VERSION", features.FEATURES_VERSION + 1)
    assert features_key(**BASE) != before


def _store(key="k"):
    s = FeatureStore(key)
    s.put(2, 7, [0.0, 1.0])
    s.put(1, 3, [1.0, 0.0])
    s.put(1, 4, [0.6, 0.8])
    return s


def test_store_has_put_get_len_and_dirty():
    s = FeatureStore("k")
    assert not s.dirty and len(s) == 0 and not s.has(1, 3)
    s.put(1, 3, [1.0, 0.0])
    assert s.dirty and s.has(1, 3) and not s.has(1, 4) and not s.has(2, 3) and len(s) == 1
    s.put(1, 4, [0.0, 1.0])
    got = s.get(1, [4, 3])
    assert got.shape == (2, 2) and got.dtype == np.float32 and got[0, 1] == 1.0


def test_store_round_trips_and_is_byte_deterministic(tmp_path):
    a, b = tmp_path / "a.features.npz", tmp_path / "b.features.npz"
    sha_a = _store().save(a)
    other = FeatureStore("k")  # same content inserted in another order
    other.put(1, 4, [0.6, 0.8])
    other.put(2, 7, [0.0, 1.0])
    other.put(1, 3, [1.0, 0.0])
    assert other.save(b) == sha_a == io.sha256_file(a)
    assert a.read_bytes() == b.read_bytes()
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = FeatureStore.load(a, "k")
    assert loaded is not None and not loaded.dirty and len(loaded) == 3
    assert np.allclose(loaded.get(1, [3, 4]), [[1.0, 0.0], [0.6, 0.8]])
    assert loaded.has(2, 7)


def test_an_empty_store_round_trips(tmp_path):
    p = tmp_path / "e.features.npz"
    FeatureStore("k").save(p)
    loaded = FeatureStore.load(p, "k")
    assert loaded is not None and len(loaded) == 0


def test_a_different_key_is_a_miss_logged_at_info(tmp_path, caplog):
    p = tmp_path / "f.features.npz"
    _store("k").save(p)
    with caplog.at_level(logging.INFO, logger="dnt.refine.features"):
        assert FeatureStore.load(p, "other") is None
    assert "different inputs" in caplog.text


def test_missing_garbage_and_truncated_files_are_misses(tmp_path):
    assert FeatureStore.load(tmp_path / "none.npz", "k") is None
    bad = tmp_path / "bad.features.npz"
    bad.write_bytes(b"not a zip at all")
    assert FeatureStore.load(bad, "k") is None
    good = tmp_path / "good.features.npz"
    _store().save(good)
    cut = tmp_path / "cut.features.npz"
    cut.write_bytes(good.read_bytes()[:60])
    assert FeatureStore.load(cut, "k") is None
    empty = tmp_path / "empty.features.npz"
    empty.write_bytes(b"")
    assert FeatureStore.load(empty, "k") is None


def _write_npz(path, **over):
    arrays = {
        "key": np.array("k"),
        "raw_id": np.array([1, 1], dtype=np.int64),
        "frame": np.array([3, 4], dtype=np.int64),
        "emb": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    arrays.update(over)
    np.savez(path, **arrays)


def test_the_helper_writes_a_loadable_archive(tmp_path):
    p = tmp_path / "ok.features.npz"
    _write_npz(p)
    loaded = FeatureStore.load(p, "k")
    assert loaded is not None and len(loaded) == 2 and loaded.has(1, 4)


@pytest.mark.parametrize(
    "over",
    [
        {"emb": np.array([1.0, 0.0], dtype=np.float32)},  # 1-D embeddings
        {"emb": np.array([["a", "b"], ["c", "d"]])},  # non-numeric
        {"emb": np.array([[1.0, 0.0]], dtype=np.float32)},  # wrong number of rows
        {"emb": np.zeros((2, 0), dtype=np.float32)},  # zero width with rows
        {"emb": np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)},  # not finite
        {"emb": np.array([[1.0, np.inf], [0.0, 1.0]], dtype=np.float32)},
        {"raw_id": np.array(1, dtype=np.int64)},  # 0-D
        {"frame": np.array([[3, 4]], dtype=np.int64)},  # 2-D
        {"frame": np.array([3.0, 4.0])},  # float frames
        {"raw_id": np.array(["a", "b"])},  # non-numeric ids
        {"frame": np.array([3, 3], dtype=np.int64)},  # duplicate (raw_id, frame)
        {"key": np.array(["k", "k"])},  # key is not a scalar string
        {"key": np.array(3)},
    ],
)
def test_readable_but_malformed_archives_are_misses(tmp_path, over):
    p = tmp_path / "bad.features.npz"
    _write_npz(p, **over)
    assert FeatureStore.load(p, "k") is None


def test_an_archive_with_a_missing_array_is_a_miss(tmp_path):
    p = tmp_path / "part.features.npz"
    np.savez(p, key=np.array("k"), raw_id=np.array([1]), frame=np.array([3]))
    assert FeatureStore.load(p, "k") is None


def test_an_archive_of_the_wrong_embedding_width_is_a_miss_for_that_encoder(tmp_path):
    p = tmp_path / "w.features.npz"
    _write_npz(p)  # width 2
    assert FeatureStore.load(p, "k", dim=3) is None
    assert FeatureStore.load(p, "k", dim=2) is not None
    assert FeatureStore.load(p, "k") is not None  # no encoder to compare with


def test_values_that_overflow_float32_are_a_miss(tmp_path):
    p = tmp_path / "big.features.npz"
    _write_npz(p, emb=np.array([[1e300, 0.0], [0.0, 1.0]]))  # finite as float64, inf as float32
    assert FeatureStore.load(p, "k", dim=2) is None
    ok = tmp_path / "f64.features.npz"
    _write_npz(ok, emb=np.array([[0.5, 0.25], [0.0, 1.0]]))  # float64 is fine when it converts
    loaded = FeatureStore.load(ok, "k", dim=2)
    assert loaded is not None and loaded.get(1, [3]).dtype == np.float32


def test_put_rejects_an_embedding_of_another_width():
    s = FeatureStore("k")
    s.put(1, 3, [1.0, 0.0])
    with pytest.raises(ValueError, match="width"):
        s.put(1, 4, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="width"):
        s.put(1, 5, [[1.0, 0.0]])  # not a vector


def _table():
    frames = list(range(20))
    emb = np.eye(4)[np.arange(20) % 4]
    return {1: (frames, emb)}


def test_coarse_provider_returns_every_kth_sample_and_dense_all():
    app = CoarseArrayAppearance(_table(), every=5)
    f, e = app.clean_embeddings(1, 0, 19)
    assert list(f) == [0, 5, 10, 15] and e.shape == (4, 4)
    f, e = app.clean_embeddings(1, 3, 12)
    assert list(f) == [5, 10]
    f, e = app.dense_embeddings(1, 3, 12)
    assert list(f) == list(range(3, 13))
    assert app.prefetch_dense([(1, 0, 5)]) is None
    assert app.clean_embeddings(9, 0, 5)[0].size == 0 and app.dense_embeddings(9, 0, 5)[0].size == 0
    with pytest.raises(ValueError, match="every"):
        CoarseArrayAppearance(_table(), every=0)


def test_plain_array_provider_has_no_dense_methods():
    assert not hasattr(ArrayAppearance(_table()), "dense_embeddings")


def test_dense_track_embeddings_follow_the_lineage_and_clip_to_the_window():
    app = CoarseArrayAppearance({1: (range(10), np.eye(3)[np.arange(10) % 3]),
                                 2: (range(10, 20), np.eye(3)[np.arange(10) % 3])}, every=5)
    lineage = [[1, 0, 9], [2, 10, 19]]
    f, e = dense_track_embeddings(app, lineage, 7, 12)
    assert list(f) == [7, 8, 9, 10, 11, 12] and e.shape == (6, 3)
    f, e = dense_track_embeddings(app, lineage, 30, 40)
    assert f.size == 0 and e.size == 0
    f, _ = track_embeddings(app, lineage)  # coarse only: ordinals 0 and 5 of each raw track
    assert list(f) == [0, 5, 10, 15]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_features_cache.py -q`
Expected: ImportError (`CoarseArrayAppearance`, `FeatureStore`, ... not defined).

- [ ] **Step 3: Implement**

Replace `src/dnt/refine/features.py` with:

```python
"""The appearance interface the stages use (spec 5.3), array providers, and the embedding cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .io import sha256_file

log = logging.getLogger(__name__)

#: Bumped whenever the crop or embedding code changes, so old caches are not reused.
FEATURES_VERSION = 1


class Appearance(Protocol):
    """Source of clean (unoccluded), L2-normalized embeddings per raw track and frame."""

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames, embeddings)`` for ``raw_id`` within ``[f0, f1]``, sorted by frame."""
        ...


class DenseAppearance(Appearance, Protocol):
    """An ``Appearance`` that can also give every clean frame around a stage 1 candidate.

    ``clean_embeddings`` then returns the coarse samples only. Stage 1 finds candidates on them
    and calls ``prefetch_dense`` once with every window it will need, then ``dense_embeddings``.
    """

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every clean observed frame of ``raw_id`` within ``[f0, f1]``, sorted."""
        ...

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Compute the embeddings of ``(raw_id, f0, f1)`` windows in one pass."""
        ...


class ArrayAppearance:
    """In-memory ``Appearance`` built from arrays (tests and callers with precomputed features)."""

    def __init__(self, table: Mapping[int, tuple[Sequence[int], np.ndarray]]):
        """Store ``{raw_id: (frames, embeddings)}``, sorted and L2-normalized."""
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for raw, (frames, emb) in table.items():
            f = np.asarray(list(frames), dtype=int)
            e = np.asarray(emb, dtype=float)
            if len(f) != len(e):
                raise ValueError(f"raw_id {raw}: frames and embeddings have different lengths")
            if e.ndim != 2 or e.shape[0] == 0:
                raise ValueError(f"raw_id {raw}: embeddings must be 2-D with at least one row")
            order = np.argsort(f, kind="stable")
            f, e = f[order], e[order]
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            self._t[int(raw)] = (f, e)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored samples of ``raw_id`` within ``[f0, f1]``."""
        if int(raw_id) not in self._t:
            return np.empty(0, dtype=int), np.empty((0, 0))
        f, e = self._t[int(raw_id)]
        m = (f >= f0) & (f <= f1)
        return f[m], e[m]


class CoarseArrayAppearance(ArrayAppearance):
    """``ArrayAppearance`` that behaves like a video provider: coarse by default, dense on request.

    ``clean_embeddings`` returns every ``every``-th stored sample (by ordinal within the raw
    track); ``dense_embeddings`` returns all stored samples.
    """

    def __init__(self, table: Mapping[int, tuple[Sequence[int], np.ndarray]], every: int = 5):
        """Store the samples; ``every`` is the coarse stride."""
        if int(every) < 1:
            raise ValueError("every must be at least 1")
        super().__init__(table)
        self._every = int(every)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the coarse samples of ``raw_id`` within ``[f0, f1]``."""
        if int(raw_id) not in self._t:
            return np.empty(0, dtype=int), np.empty((0, 0))
        f, e = self._t[int(raw_id)]
        m = (np.arange(len(f)) % self._every == 0) & (f >= f0) & (f <= f1)
        return f[m], e[m]

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every stored sample of ``raw_id`` within ``[f0, f1]``."""
        return ArrayAppearance.clean_embeddings(self, raw_id, f0, f1)

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Do nothing: the samples are already in memory."""
        return None


def _join(parts) -> tuple[np.ndarray, np.ndarray]:
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return np.empty(0, dtype=int), np.empty((0, 0))
    f = np.concatenate([p[0] for p in parts])
    e = np.vstack([p[1] for p in parts])
    order = np.argsort(f, kind="stable")
    return f[order], e[order]


def track_embeddings(appearance: Appearance, lineage) -> tuple[np.ndarray, np.ndarray]:
    """Return a track's clean (coarse) samples across its lineage spans, sorted by frame."""
    return _join([appearance.clean_embeddings(int(r), int(a), int(b)) for r, a, b in lineage])


def dense_track_embeddings(
    appearance: DenseAppearance, lineage, f0: int, f1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a track's dense samples within ``[f0, f1]`` across its lineage spans, sorted."""
    parts = []
    for r, a, b in lineage:
        lo, hi = max(int(a), int(f0)), min(int(b), int(f1))
        if lo <= hi:
            parts.append(appearance.dense_embeddings(int(r), lo, hi))
    return _join(parts)


def features_key(
    *, tracks_sha, video, context_sha, encoder, sample_every, occlusion_iou, crop_pad
) -> str:
    """Return the SHA-256 key of every input that can change an embedding (spec 5.3).

    Parameters
    ----------
    tracks_sha : str
        SHA-256 of the input track file.
    video : dict
        Video fingerprint with ``sha256``, ``size`` and ``frame_count``.
    context_sha : str or None
        SHA-256 of the context file, or ``None``.
    encoder : object
        Anything with ``name``, ``model_name``, ``weights_sha`` and ``preprocess_id``.
    sample_every, occlusion_iou, crop_pad : float
        Sampling stride, occlusion threshold, and crop padding.

    Returns
    -------
    str
        Hex digest.

    """
    parts = {
        "version": FEATURES_VERSION,
        "tracks": tracks_sha,
        "video": [video["sha256"], int(video["size"]), int(video["frame_count"])],
        "context": context_sha or "none",
        "encoder": [encoder.name, encoder.model_name, encoder.weights_sha, encoder.preprocess_id],
        "sample_every": int(sample_every),
        "occlusion_iou": float(occlusion_iou),
        "crop_pad": float(crop_pad),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _checked(ids, frames, emb, dim) -> np.ndarray | None:
    """Return the embeddings as float32 if the cache arrays are usable, else ``None``.

    Usable means: 1-D integer ids and frames, a 2-D float matrix with one row each, the width
    ``dim`` when it is given (and at least one column otherwise), finite values *after* the
    conversion to float32, and no duplicate ``(raw_id, frame)`` pair.
    """
    if ids.ndim != 1 or frames.ndim != 1 or emb.ndim != 2:
        return None
    if ids.dtype.kind not in "iu" or frames.dtype.kind not in "iu" or emb.dtype.kind != "f":
        return None
    if not (len(ids) == len(frames) == emb.shape[0]):
        return None
    if len(ids) and (emb.shape[1] == 0 or (dim is not None and emb.shape[1] != dim)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        out = emb.astype(np.float32)
    if not bool(np.isfinite(out).all()):
        return None
    if len(set(zip(ids.tolist(), frames.tolist(), strict=True))) != len(ids):
        return None
    return out


class FeatureStore:
    """Embeddings per (raw track id, frame), saved as a deterministic ``.npz`` under a key."""

    def __init__(self, key: str):
        """Create an empty store for cache key ``key``."""
        self.key = key
        self.dirty = False
        self._dim: int | None = None
        self._d: dict[int, dict[int, np.ndarray]] = {}

    def __len__(self) -> int:
        """Return the number of stored embeddings."""
        return sum(len(v) for v in self._d.values())

    def has(self, raw_id: int, frame: int) -> bool:
        """Return whether ``(raw_id, frame)`` has an embedding."""
        return int(frame) in self._d.get(int(raw_id), {})

    def put(self, raw_id: int, frame: int, emb) -> None:
        """Store one embedding; every embedding of a store has the same width."""
        e = np.asarray(emb, dtype=np.float32)
        if e.ndim != 1 or (self._dim is not None and e.shape[0] != self._dim):
            raise ValueError(
                f"embedding of shape {e.shape} does not match the store's width {self._dim}"
            )
        self._dim = int(e.shape[0])
        self._d.setdefault(int(raw_id), {})[int(frame)] = e
        self.dirty = True

    def get(self, raw_id: int, frames: Sequence[int]) -> np.ndarray:
        """Return the embeddings of ``frames`` of ``raw_id`` as a ``(len(frames), D)`` array."""
        rows = self._d[int(raw_id)]
        return np.stack([rows[int(f)] for f in frames])

    def save(self, path) -> str:
        """Write the store atomically and return the file's SHA-256."""
        ids, frames, embs = [], [], []
        for rid in sorted(self._d):
            for f in sorted(self._d[rid]):
                ids.append(rid)
                frames.append(f)
                embs.append(self._d[rid][f])
        dim = embs[0].shape[0] if embs else 0
        arrays = {
            "key": np.array(self.key),
            "raw_id": np.asarray(ids, dtype=np.int64),
            "frame": np.asarray(frames, dtype=np.int64),
            "emb": np.asarray(embs, dtype=np.float32).reshape(len(ids), dim),
        }
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, p)
        self.dirty = False
        return sha256_file(p)

    @classmethod
    def load(cls, path, key: str, dim: int | None = None) -> FeatureStore | None:
        """Return the stored cache, or ``None`` if it is missing, damaged, or for other inputs.

        Parameters
        ----------
        path : path-like
            The ``.features.npz`` file.
        key : str
            The cache key the file must carry.
        dim : int, optional
            The encoder's embedding width; a file with another width is a miss.

        """
        p = Path(path)
        if not p.is_file():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                key_array = z["key"]
                ids, frames, emb = z["raw_id"], z["frame"], z["emb"]
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
            log.info("feature cache %s is unreadable (%s); recomputing", p, err)
            return None
        if key_array.ndim != 0 or key_array.dtype.kind != "U" or str(key_array) != key:
            log.info("feature cache %s was built with different inputs; recomputing", p)
            return None
        emb32 = _checked(ids, frames, emb, dim)
        if emb32 is None:
            log.info("feature cache %s is malformed or has another width; recomputing", p)
            return None
        store = cls(key)
        for r, f, e in zip(ids.tolist(), frames.tolist(), emb32, strict=True):
            store._d.setdefault(r, {})[f] = e
        if len(emb32):
            store._dim = int(emb32.shape[1])
        return store
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_features_cache.py tests/refine/test_verify_features.py tests/refine/test_switch.py tests/refine/test_link_scoring.py -q`
Expected: all pass (P1's `ArrayAppearance` and `track_embeddings` behave as before).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/features.py tests/refine/test_features_cache.py
git commit -m "feat(refine): add the feature cache, its key, and a dense-capable array provider" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `VideoAppearance`, the real provider

**Files:**
- Create: `src/dnt/refine/video_appearance.py`, `tests/refine/_video.py` (test helpers), `tests/refine/test_video_appearance.py`

**Interfaces:**
- Consumes: Task 2 (`FrameReader`, `crop_box`, `CROP_PAD`), Task 4 (`FeatureStore`), Task 1 (an object with the `AppearanceEncoder` shape), P1's `work` table (columns `frame, track, x, y, w, h, ..., raw_id`) and `primitives.occlusion_flags` output (a bool Series indexed like `work`).
- Produces: `VideoAppearance(work, occluded, video_file, encoder, store, *, sample_every, batch_size, crop_pad=CROP_PAD)` implementing `DenseAppearance`:
  - `clean_embeddings(raw_id, f0, f1)`: the **coarse** clean samples only (ordinals `0, k, 2k, ...` of the raw track's observed frames, not occluded, crop not empty).
  - `dense_embeddings(raw_id, f0, f1)`: every clean observed frame in the window.
  - `prefetch_coarse()`: embeds every coarse sample of every raw track in one sequential pass over the video.
  - `prefetch_dense(windows)`: embeds the windows' clean frames in one pass; overlapping windows embed each frame once.
  - Both getters embed missing samples on demand, so they work without a prefetch. Samples already in the store are never re-embedded and never re-read.
  - Test helpers in `tests/refine/_video.py`: `make_color_video`, `video_rows`, `ColorEncoder`, `RED`, `BLUE`, `RED_DIR`, `BLUE_DIR`, `FPS`.

- [ ] **Step 1: Write the test helpers and the failing tests**

Create `tests/refine/_video.py`:

```python
"""A tiny colored-box video and a stub encoder for the appearance tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FPS = 10.0
WIDTH, HEIGHT = 320, 240
BACKGROUND = 90
RED = (0, 0, 200)  # BGR
BLUE = (200, 0, 0)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


RED_DIR = _unit(np.array([200.0, 0.0, 0.0]) - BACKGROUND)  # RGB, away from the background
BLUE_DIR = _unit(np.array([0.0, 0.0, 200.0]) - BACKGROUND)


def make_color_video(path, rows, n_frames, fps=FPS):
    """Write an mp4 where each ``(frame, x, y, w, h, bgr)`` row is a filled rectangle."""
    by_frame: dict[int, list] = {}
    for f, x, y, w, h, color in rows:
        by_frame.setdefault(int(f), []).append((x, y, w, h, color))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    try:
        for f in range(n_frames):
            img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
            for x, y, w, h, color in by_frame.get(f, []):
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
            writer.write(img)
    finally:
        writer.release()
    return Path(path)


def video_rows(track_rows, color):
    """Turn track rows ``[f, track, x, y, w, h, ...]`` into ``make_color_video`` rows."""
    return [(r[0], r[2], r[3], r[4], r[5], color) for r in track_rows]


class ColorEncoder:
    """Stub encoder: the direction of a crop's mean RGB color away from the gray background."""

    name = "stub"
    model_name = "stub-color"
    preprocess_id = "stub-v1"
    weights_sha = None
    dim = 3

    def __init__(self):
        self.calls = 0
        self.crops = 0
        self.max_batch = 0

    def encode(self, crops):
        self.calls += 1
        self.crops += len(crops)
        self.max_batch = max(self.max_batch, len(crops))
        out = []
        for c in crops:
            v = c.reshape(-1, 3).mean(axis=0) - BACKGROUND
            n = np.linalg.norm(v)
            out.append(v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0]))
        return np.asarray(out, dtype=np.float32).reshape(-1, 3)
```

Create `tests/refine/test_video_appearance.py`:

```python
import numpy as np
import pandas as pd
import pytest

from dnt.refine import io, video_appearance
from dnt.refine.features import FeatureStore
from dnt.refine.video_appearance import VideoAppearance

from ._fixtures import box_rows, table
from ._video import BLUE, RED, RED_DIR, ColorEncoder, make_color_video, video_rows


def _make(tmp_path, tracks, colors, *, occluded=None, every=5, batch=4, n_frames=60, enc=None,
          store=None):
    rows = [r for t in tracks for r in t]
    vrows = [v for t, c in zip(tracks, colors, strict=True) for v in video_rows(t, c)]
    video = make_color_video(tmp_path / "v.mp4", vrows, n_frames)
    work = io.to_work(table(*tracks)).work
    occ = pd.Series(False, index=work.index) if occluded is None else occluded(work)
    enc = enc or ColorEncoder()
    store = store or FeatureStore("k")
    app = VideoAppearance(work, occ, video, enc, store, sample_every=every, batch_size=batch)
    assert len(rows) == len(work)
    return app, enc, store, video


def _walker(track, frames, x0=100.0):
    return box_rows(track, frames, x0, 60.0, vx=2.0, w=40.0, h=80.0)


def test_coarse_samples_are_every_kth_observed_frame_with_the_right_color(tmp_path):
    app, enc, _, _ = _make(tmp_path, [_walker(1, range(60))], [RED])
    f, e = app.clean_embeddings(1, 0, 59)
    assert list(f) == list(range(0, 60, 5)) and e.shape == (12, 3)
    assert (e @ RED_DIR).min() > 0.98
    assert enc.crops == 12


def test_ordinals_count_observed_frames_not_frame_numbers(tmp_path):
    frames = list(range(10)) + list(range(20, 30))
    app, *_ = _make(tmp_path, [_walker(1, frames)], [RED])
    f, _ = app.clean_embeddings(1, 0, 59)
    assert list(f) == [0, 5, 20, 25]


def test_dense_returns_every_clean_frame_and_reuses_coarse_samples(tmp_path):
    app, enc, _, _ = _make(tmp_path, [_walker(1, range(60))], [RED])
    fc, ec = app.clean_embeddings(1, 10, 20)
    assert list(fc) == [10, 15, 20]
    before = enc.crops
    fd, ed = app.dense_embeddings(1, 10, 20)
    assert list(fd) == list(range(10, 21))
    assert enc.crops == before + 8  # only the 8 frames not already embedded
    assert np.array_equal(ed[[0, 5, 10]], ec)
    # the coarse view is unchanged by dense samples sitting in the store
    assert list(app.clean_embeddings(1, 10, 20)[0]) == [10, 15, 20]


def test_occluded_rows_are_not_embedded(tmp_path):
    app, *_ = _make(
        tmp_path, [_walker(1, range(60))], [RED], occluded=lambda w: w["frame"].between(20, 29)
    )
    f, _ = app.clean_embeddings(1, 0, 59)
    assert 20 not in f and 25 not in f and 30 in f and 15 in f
    f, _ = app.dense_embeddings(1, 15, 35)
    assert list(f) == [15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35]


def test_a_track_with_no_clean_crops_gives_empty_arrays(tmp_path):
    app, enc, *_ = _make(
        tmp_path, [_walker(1, range(30))], [RED], occluded=lambda w: w["frame"] >= 0
    )
    f, e = app.clean_embeddings(1, 0, 59)
    assert f.size == 0 and e.size == 0 and enc.calls == 0
    assert app.dense_embeddings(1, 0, 59)[0].size == 0
    assert app.clean_embeddings(99, 0, 59)[0].size == 0  # unknown raw id


def test_boxes_outside_the_frame_are_skipped_without_error(tmp_path):
    inside, outside = _walker(1, range(30)), _walker(2, range(30), x0=1000.0)
    app, enc, *_ = _make(tmp_path, [inside, outside], [RED, BLUE])
    assert app.clean_embeddings(2, 0, 29)[0].size == 0
    assert list(app.clean_embeddings(1, 0, 29)[0]) == [0, 5, 10, 15, 20, 25]
    assert enc.crops == 6
    assert app.dense_embeddings(2, 0, 29)[0].size == 0  # no retry storm on the unreadable ones
    assert enc.crops == 6


def test_samples_in_the_store_are_never_re_read(tmp_path, monkeypatch):
    app, enc, *_ = _make(tmp_path, [_walker(1, range(60))], [RED])
    first = app.clean_embeddings(1, 0, 59)
    opened = []
    real = video_appearance.FrameReader

    def counting(path):
        opened.append(path)
        return real(path)

    monkeypatch.setattr(video_appearance, "FrameReader", counting)
    calls = enc.calls
    again = app.clean_embeddings(1, 0, 59)
    assert opened == [] and enc.calls == calls
    assert np.array_equal(first[1], again[1])


def test_prefetch_coarse_reads_the_video_once_for_all_tracks(tmp_path, monkeypatch):
    tracks = [_walker(1, range(60)), _walker(2, range(10, 50), x0=200.0)]
    app, *_ = _make(tmp_path, tracks, [RED, BLUE])
    opened = []
    real = video_appearance.FrameReader
    monkeypatch.setattr(video_appearance, "FrameReader", lambda p: opened.append(p) or real(p))
    app.prefetch_coarse()
    assert len(opened) == 1
    app.clean_embeddings(1, 0, 59)
    app.clean_embeddings(2, 0, 59)
    assert len(opened) == 1


def test_prefetch_dense_embeds_overlapping_windows_once(tmp_path):
    app, enc, *_ = _make(tmp_path, [_walker(1, range(60))], [RED])
    app.prefetch_dense([(1, 10, 20), (1, 15, 25), (1, 10, 20)])
    assert enc.crops == 16  # frames 10..25 once each
    assert list(app.dense_embeddings(1, 10, 25)[0]) == list(range(10, 26))
    assert enc.crops == 16


def test_batches_are_bounded_and_batch_size_does_not_change_the_embeddings(tmp_path):
    tracks = [_walker(1, range(60))]
    small, enc_small, *_ = _make(tmp_path, tracks, [RED], batch=3, every=1)
    big, enc_big, *_ = _make(tmp_path, tracks, [RED], batch=64, every=1)
    a = small.clean_embeddings(1, 0, 59)[1]
    b = big.clean_embeddings(1, 0, 59)[1]
    assert enc_small.max_batch <= 3 and enc_big.max_batch <= 64 and enc_small.calls > enc_big.calls
    assert np.allclose(a, b, atol=1e-6)


def test_a_saved_store_serves_the_coarse_samples_without_the_video(tmp_path):
    app, _, store, _ = _make(tmp_path, [_walker(1, range(60))], [RED])
    first = app.clean_embeddings(1, 0, 59)
    path = tmp_path / "f.features.npz"
    store.save(path)
    loaded = FeatureStore.load(path, "k")
    work = io.to_work(table(_walker(1, range(60)))).work
    enc2 = ColorEncoder()
    app2 = VideoAppearance(
        work, pd.Series(False, index=work.index), tmp_path / "gone.mp4", enc2, loaded,
        sample_every=5, batch_size=4,
    )
    again = app2.clean_embeddings(1, 0, 59)
    assert enc2.calls == 0 and np.array_equal(first[1], again[1])
    with pytest.raises(ValueError, match="cannot open video"):
        app2.dense_embeddings(1, 0, 59)  # needs frames that were never embedded


def test_a_frame_past_the_end_of_the_video_names_the_frame(tmp_path):
    tracks = [_walker(1, [10, 500])]
    app, *_ = _make(tmp_path, tracks, [RED], n_frames=60, every=1)
    with pytest.raises(ValueError, match="frame 500"):
        app.clean_embeddings(1, 0, 600)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_video_appearance.py -q`
Expected: collection error `No module named 'dnt.refine.video_appearance'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/video_appearance.py`:

```python
"""``Appearance`` provider that embeds crops of a video, with an on-disk cache (spec 5.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .crops import CROP_PAD, FrameReader, crop_box
from .features import FeatureStore

#: Crops are encoded as soon as this many batches' worth are waiting, so memory stays bounded.
_FLUSH_BATCHES = 8


class VideoAppearance:
    """Embed the clean crops of raw tracks, on demand, and keep them in a ``FeatureStore``.

    A raw track's *coarse* samples are the observed frames with ordinal ``0, k, 2k, ...``
    (``k = sample_every``); *dense* samples are all of its observed frames. A sample is used
    only if its row is not occluded and its crop is not empty.
    """

    def __init__(
        self,
        work: pd.DataFrame,
        occluded: pd.Series,
        video_file,
        encoder,
        store: FeatureStore,
        *,
        sample_every: int,
        batch_size: int,
        crop_pad: float = CROP_PAD,
    ):
        """Index the raw tracks of ``work``.

        Parameters
        ----------
        work : pandas.DataFrame
            Raw work table with ``raw_id``, ``frame``, ``x``, ``y``, ``w``, ``h``.
        occluded : pandas.Series
            True for rows whose crop is occluded (``primitives.occlusion_flags``), indexed
            like ``work``.
        video_file : path-like
            The video the track file belongs to.
        encoder : AppearanceEncoder
            Embeds RGB crops.
        store : FeatureStore
            Holds, and receives, the embeddings.
        sample_every : int
            Coarse stride, in observed frames.
        batch_size : int
            Crops per ``encoder.encode`` call.
        crop_pad : float
            Box enlargement before cropping.

        """
        self.video_file = video_file
        self.encoder = encoder
        self.store = store
        self.batch_size = max(1, int(batch_size))
        self.crop_pad = float(crop_pad)
        self._unreadable: set[tuple[int, int]] = set()
        self._tr: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        w = work.sort_values(["raw_id", "frame"])
        w = w.assign(_occ=occluded.loc[w.index].to_numpy(bool))
        every = max(1, int(sample_every))
        for raw_id, g in w.groupby("raw_id", sort=True):
            self._tr[int(raw_id)] = (
                g["frame"].to_numpy(int),
                g[["x", "y", "w", "h"]].to_numpy(float),
                ~g["_occ"].to_numpy(bool),
                np.arange(len(g)) % every == 0,
            )

    def _rows(self, raw_id: int, f0: int, f1: int, dense: bool):
        tr = self._tr.get(int(raw_id))
        if tr is None:
            return None
        frames, boxes, clean, coarse = tr
        m = (frames >= f0) & (frames <= f1) & clean
        if not dense:
            m &= coarse
        return frames, boxes, np.flatnonzero(m)

    def _ensure(self, windows: Sequence[tuple[int, int, int]], dense: bool) -> None:
        need: dict[int, dict[tuple[int, int], np.ndarray]] = {}
        for raw_id, f0, f1 in windows:
            rows = self._rows(raw_id, f0, f1, dense)
            if rows is None:
                continue
            frames, boxes, idx = rows
            for i in idx:
                key = (int(raw_id), int(frames[i]))
                if key in self._unreadable or self.store.has(*key):
                    continue
                need.setdefault(key[1], {})[key] = boxes[i]
        if not need:
            return
        crops: list[np.ndarray] = []
        owners: list[tuple[int, int]] = []
        with FrameReader(self.video_file) as reader:
            for f, img in reader.frames(need):
                for key in sorted(need[f]):
                    crop = crop_box(img, need[f][key], self.crop_pad)
                    if crop is None:
                        self._unreadable.add(key)
                        continue
                    crops.append(crop)
                    owners.append(key)
                if len(crops) >= self.batch_size * _FLUSH_BATCHES:
                    self._flush(crops, owners)
        self._flush(crops, owners)

    def _flush(self, crops: list[np.ndarray], owners: list[tuple[int, int]]) -> None:
        for i in range(0, len(crops), self.batch_size):
            chunk = crops[i : i + self.batch_size]
            emb = self.encoder.encode(chunk)
            if len(emb) != len(chunk):
                raise ValueError(f"encoder returned {len(emb)} embeddings for {len(chunk)} crops")
            for (raw_id, frame), e in zip(owners[i : i + self.batch_size], emb, strict=True):
                self.store.put(raw_id, frame, e)
        crops.clear()
        owners.clear()

    def _collect(self, raw_id: int, f0: int, f1: int, dense: bool):
        rows = self._rows(raw_id, f0, f1, dense)
        if rows is None:
            return np.empty(0, dtype=int), np.empty((0, 0))
        frames, _, idx = rows
        keep = [int(frames[i]) for i in idx if self.store.has(raw_id, int(frames[i]))]
        if not keep:
            return np.empty(0, dtype=int), np.empty((0, 0))
        return np.asarray(keep, dtype=int), self.store.get(raw_id, keep)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the coarse clean samples of ``raw_id`` within ``[f0, f1]``, sorted by frame."""
        self._ensure([(raw_id, f0, f1)], dense=False)
        return self._collect(raw_id, f0, f1, dense=False)

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every clean observed frame of ``raw_id`` within ``[f0, f1]``, sorted."""
        self._ensure([(raw_id, f0, f1)], dense=True)
        return self._collect(raw_id, f0, f1, dense=True)

    def prefetch_coarse(self) -> None:
        """Embed every coarse sample of every raw track in one pass over the video."""
        windows = [(rid, int(tr[0][0]), int(tr[0][-1])) for rid, tr in self._tr.items()]
        self._ensure(windows, dense=False)

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Embed the clean frames of ``(raw_id, f0, f1)`` windows in one pass over the video."""
        self._ensure(list(windows), dense=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_video_appearance.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/video_appearance.py tests/refine/_video.py tests/refine/test_video_appearance.py
git commit -m "feat(refine): add the video-backed appearance provider" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Stage 1 dense rescoring and the silhouette cap

**Files:**
- Modify: `src/dnt/refine/switch.py`
- Test: `tests/refine/test_switch_dense.py`

**Interfaces:**
- Consumes: Task 4 (`dense_track_embeddings`, `CoarseArrayAppearance`), P1's `switch._score_track` info dict, `switch._side_means`, `switch._candidates`, `primitives.ramp`.
- Produces: no new public names. Behavior: when the `appearance` passed to `propose_splits` has a `dense_embeddings` method, after the candidates are chosen on the coarse samples, stage 1
  1. calls `appearance.prefetch_dense(windows)` once (if the method exists) with every `(raw_id, f0, f1)` window it will need,
  2. for each candidate, rescores the rows within `encoder.sample_every` observed rows of it on the dense samples, and **moves the candidate to the best eligible row** (highest `S`, then highest raw appearance change, then nearest, then earliest), so the cut lands on the true switch frame rather than anywhere on a coarse plateau. A row is eligible only if the two rules `_candidates` applied still hold for it: each side keeps at least `switch.min_side_seconds` of clean samples, and it is at least `switch.nms_seconds` from every other candidate (finalized ones at their new frame, the rest at their original frame). Candidates are handled in `_candidates`' priority order, and a candidate's own original row is always eligible, so a move can never break either rule,
  3. merges the dense samples into the track's samples, so swap confirmation also sees them.
  A provider without `dense_embeddings` (P1's `ArrayAppearance`) and motion-only runs behave exactly as before. `_silhouette` is capped at 600 evenly spaced samples (module constant `_SILHOUETTE_MAX`), which bounds its memory; it changes no result below 600 samples.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_switch_dense.py`:

```python
import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance, CoarseArrayAppearance
from dnt.refine.switch import (
    _SILHOUETTE_MAX,
    _dense_rescore,
    _merge_dense,
    _relocate,
    _silhouette,
    propose_splits,
)

from ._fixtures import box_rows, table

FPS = 10.0
A_, B_ = np.eye(8)[0], np.eye(8)[1]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _gap_track(track=1):
    """Frames 0..119 minus frame 60 (the gap opens the gate near frame 61)."""
    return box_rows(track, [f for f in range(120) if f != 60], 100.0, 100.0, vx=2.0)


def _table(track=1, cut=64):
    frames = [f for f in range(120) if f != 60]
    return {track: (frames, np.array([A_ if f < cut else B_ for f in frames]))}


def test_dense_rescoring_moves_the_cut_to_the_true_switch_frame():
    work = _work(_gap_track())
    cfg = RefineConfig.defaults()
    dense = propose_splits(work, cfg, FPS, CoarseArrayAppearance(_table(), every=5))
    assert [e.params["cut_frame"] for e in dense.events] == [64]
    assert dense.candidates[1] == [64]
    ev = dense.events[0]
    assert ev.algo_score == pytest.approx(cfg.switch.w_app)  # app saturates, no motion break
    assert ev.signals["motion_only"] is False
    # the same coarse samples without dense access stay on the first frame of the plateau
    t = _table()
    f, e = t[1]
    coarse = propose_splits(work, cfg, FPS, ArrayAppearance({1: (f[::5], e[::5])}))
    assert [x.params["cut_frame"] for x in coarse.events] == [62]


def test_dense_samples_are_requested_only_around_candidates():
    class Recording(CoarseArrayAppearance):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.prefetched, self.dense_calls = [], []

        def prefetch_dense(self, windows):
            self.prefetched.append(list(windows))

        def dense_embeddings(self, raw_id, f0, f1):
            self.dense_calls.append((raw_id, f0, f1))
            return super().dense_embeddings(raw_id, f0, f1)

    quiet = box_rows(2, range(120), 400.0, 300.0, vx=1.0)
    table2 = {**_table(1), 2: (list(range(120)), np.tile(A_, (120, 1)))}
    app = Recording(table2, every=5)
    propose_splits(_work(_gap_track(1), quiet), RefineConfig.defaults(), FPS, app)
    assert len(app.prefetched) == 1 and app.prefetched[0]
    assert {w[0] for w in app.prefetched[0]} == {1}
    assert {c[0] for c in app.dense_calls} == {1}
    assert all(40 <= f0 <= f1 <= 80 for _, f0, f1 in app.prefetched[0])


def test_a_provider_that_is_dense_but_identical_changes_nothing():
    frames = list(range(100))
    rows = box_rows(1, range(50), 100.0, 100.0, vx=2.0, w=30.0) + box_rows(
        1, range(50, 100), 200.0, 100.0, vx=2.0, w=60.0
    )
    emb = np.array([A_ if f < 50 else B_ for f in frames])
    work = _work(rows)
    cfg = RefineConfig.defaults()
    plain = propose_splits(work, cfg, FPS, ArrayAppearance({1: (frames, emb)}))
    dense = propose_splits(work, cfg, FPS, CoarseArrayAppearance({1: (frames, emb)}, every=1))
    assert plain.events and len(plain.events) == len(dense.events)
    for a, b in zip(plain.events, dense.events, strict=True):
        assert a.params == b.params and a.algo_score == pytest.approx(b.algo_score, abs=1e-12)
        assert a.signals == pytest.approx(b.signals, abs=1e-12, nan_ok=True)


def _crafted(n, cuts, candidates, every=5):
    """A hand-built stage 1 info dict (gate open, no motion break) and its dense embeddings.

    The embedding is A before the first cut, B until the second, A again, and so on.
    """
    emb = np.tile(A_, (n, 1))
    state = 0
    for c in cuts:
        emb[c:] = B_ if state == 0 else A_
        state ^= 1
    frames = np.arange(n)
    info = {
        "frames": frames,
        "med": 0.0,
        "mad": 0.0,
        "bim": np.zeros(n),
        "fired": {"gap": np.ones(n, bool)},
        "mot": np.zeros(n),
        "A": np.zeros(n),
        "z": np.full(n, np.nan),
        "app": np.zeros(n),
        "S": np.zeros(n),
        "samples": frames[::every],
        "motion_only": False,
        "ef": frames[::every],
        "emb": emb[::every],
    }
    for c in candidates:
        info["S"][c] = 0.65
    return info, emb


def test_relocate_moves_to_the_best_eligible_row_and_stores_its_values():
    info, emb = _crafted(40, (12,), [10])
    sc = RefineConfig.defaults().switch
    ef = np.arange(40)
    assert _relocate(info, 10, sc, 5, 15, 5, ef, emb, lambda j: True) == 12
    assert info["S"][12] == pytest.approx(sc.w_app) and info["A"][12] == pytest.approx(1.0)

    info, emb = _crafted(40, (12,), [10])
    got = _relocate(info, 10, sc, 5, 15, 5, ef, emb, lambda j: j != 12)
    assert got in (11, 13) and info["S"][12] == 0.0  # the next best, and 12 was never stored

    info, emb = _crafted(40, (12,), [10])
    assert _relocate(info, 10, sc, 5, 15, 5, ef, emb, lambda j: j == 10) == 10
    assert info["S"][10] == pytest.approx(sc.w_app)  # rescored in place

    info, emb = _crafted(40, (12,), [10])
    empty = (np.empty(0, dtype=int), np.empty((0, 0)))
    assert _relocate(info, 10, sc, 5, 15, 5, *empty, lambda j: True) == 10  # no dense samples


def _rescored(cfg, n, cuts, candidates, nms, w=5, fps=10.0):
    info, emb = _crafted(n, cuts, candidates)
    app = CoarseArrayAppearance({1: (list(range(n)), emb)}, every=5)
    cands = {1: list(candidates)}
    _dense_rescore(app, {1: (info, [[1, 0, n - 1]])}, cands, cfg, w, fps, nms)
    return [int(info["frames"][i]) for i in cands[1]]


def test_a_move_cannot_leave_too_little_clean_data_on_a_side():
    cfg = RefineConfig.defaults()
    # coarse samples sit every 5 frames (0..40); the true cut is at 33. A cut at 33 leaves the
    # samples 35 and 40 on its right (2 x 5 frames = 1.0 s); a cut at 30 also keeps sample 30.
    cfg.switch.min_side_seconds = 0.5
    assert _rescored(cfg, 45, (33,), [30], nms=20) == [33]
    cfg.switch.min_side_seconds = 1.5  # needs three samples on the right: only 30 or earlier
    assert _rescored(cfg, 45, (33,), [30], nms=20) == [30]


def test_converging_candidates_keep_the_nms_spacing():
    cfg = RefineConfig.defaults()
    # cuts at 12 and 28 would pull the candidates at 10 and 30 to 16 frames apart
    assert _rescored(cfg, 60, (12, 28), [10, 30], nms=1) == [12, 28]
    kept = _rescored(cfg, 60, (12, 28), [10, 30], nms=20)
    assert len(kept) == 2 and kept[1] - kept[0] >= 20


def test_merge_dense_keeps_samples_sorted_and_unique():
    ef, emb = np.array([0, 5, 10]), np.eye(3)
    part_f, part_e = np.array([4, 5, 6]), np.array([[0.0, 1, 0], [0.0, 1, 0], [0, 0, 1.0]])
    f, e = _merge_dense(ef, emb, [(part_f, part_e), (np.empty(0, dtype=int), np.empty((0, 0)))])
    assert list(f) == [0, 4, 5, 6, 10] and e.shape == (5, 3)
    f2, e2 = _merge_dense(ef, emb, [])
    assert f2 is ef and e2 is emb


def test_silhouette_is_capped_and_equals_the_subsample():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal([1.0, 0.0], 0.05, (1500, 2)), rng.normal([0.0, 1.0], 0.05, (1500, 2))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.r_[np.zeros(1500, int), np.ones(1500, int)]
    pick = np.linspace(0, len(emb) - 1, _SILHOUETTE_MAX).astype(int)
    capped = _silhouette(emb, labels)
    assert capped > 0.9
    assert capped == _silhouette(emb[pick], labels[pick])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_switch_dense.py -q`
Expected: ImportError (`_SILHOUETTE_MAX`, `_merge_dense`).

- [ ] **Step 3: Implement**

In `src/dnt/refine/switch.py`:

1. Change the import line `from .features import Appearance, track_embeddings` to
   `from .features import Appearance, dense_track_embeddings, track_embeddings`.

2. Replace the start of `_silhouette` (add a module constant above the function):

```python
_SILHOUETTE_MAX = 600


def _silhouette(emb: np.ndarray, labels: np.ndarray) -> float:
    if len(emb) > _SILHOUETTE_MAX:  # bounds the n x n distance matrix; deterministic subsample
        pick = np.linspace(0, len(emb) - 1, _SILHOUETTE_MAX).astype(int)
        emb, labels = emb[pick], labels[pick]
    d = 1.0 - emb @ emb.T
```

   (the rest of the function is unchanged).

3. In `_score_track`, add the baseline to the info dict. Replace

```python
    sil = None
    if not motion_only:
        change_a = _appearance_change(frames, ef, emb, w)
```

   with

```python
    sil = None
    med = mad = float("nan")
    if not motion_only:
        change_a = _appearance_change(frames, ef, emb, w)
```

   and replace the end of the returned dict

```python
        "ef": ef,
        "emb": emb,
    }
```

   with

```python
        "ef": ef,
        "emb": emb,
        "med": med,
        "mad": mad,
    }
```

4. Insert these helpers immediately before `def propose_splits(`:

```python
def _lineage_windows(lin, f0: int, f1: int) -> list[tuple[int, int, int]]:
    """Return the ``(raw_id, lo, hi)`` pieces of ``[f0, f1]`` that lie inside the lineage."""
    out = []
    for r, a, b in lin:
        lo, hi = max(int(a), f0), min(int(b), f1)
        if lo <= hi:
            out.append((int(r), lo, hi))
    return out


def _span(info, i: int, radius: int, w: int) -> tuple[int, int, int, int]:
    """Rows ``lo..hi`` around candidate row ``i``, and the frame range their sides need."""
    fr = info["frames"]
    lo, hi = max(1, i - radius), min(len(fr) - 1, i + radius)
    return lo, hi, int(fr[lo]) - w, int(fr[hi]) + w - 1


def _relocate(info, i: int, sc, lo: int, hi: int, w: int, ef_d, emb_d, eligible) -> int:
    """Rescore rows ``lo..hi`` on dense samples, store the best eligible row's values, return it.

    The best row has the highest ``S``, then the highest raw appearance change (``S`` saturates
    for strong changes), then the smallest distance from ``i``, then the earliest frame.
    ``eligible(j)`` says whether a row may hold the cut; the original row ``i`` always may.
    """
    if not np.isfinite(info["med"]):
        return i
    best = None
    for j in range(lo, hi + 1):
        if not eligible(j):
            continue
        m = _side_means(ef_d, emb_d, int(info["frames"][j]), w)
        if m is None:
            continue
        a = 1.0 - float(m[0] @ m[1])
        z = (a - info["med"]) / (1.4826 * info["mad"] + sc.mad_floor)
        app = max(float(ramp(z, *sc.ramps["z_app"])), float(info["bim"][j]))
        gate = any(bool(v[j]) for v in info["fired"].values())
        s = (sc.w_app * app + sc.w_mot * float(info["mot"][j])) if gate else 0.0
        key = (s, a, -abs(j - i), -j)
        if best is None or key > best[0]:
            best = (key, j, a, z, app, s)
    if best is None:
        return i
    _, j, a, z, app, s = best
    info["A"][j], info["z"][j], info["app"][j], info["S"][j] = a, z, app, s
    return j


def _sides_ok(info, t: int, fps: float, min_side: float) -> bool:
    """Return whether both sides of a cut at frame ``t`` keep ``min_side`` seconds of samples."""
    s = info["samples"]
    return (
        _covered_seconds(s[s < t], fps, info["motion_only"]) >= min_side
        and _covered_seconds(s[s >= t], fps, info["motion_only"]) >= min_side
    )


def _merge_dense(ef, emb, parts):
    """Merge dense ``(frames, embeddings)`` parts into a track's samples: sorted, one per frame."""
    fs, es = [], []
    if len(ef):
        fs.append(ef)
        es.append(emb)
    for f, e in parts:
        if len(f):
            fs.append(f)
            es.append(e)
    if not any(len(f) for f, _ in parts):
        return ef, emb
    f = np.concatenate(fs)
    e = np.vstack(es)
    order = np.argsort(f, kind="stable")
    f, e = f[order], e[order]
    keep = np.concatenate([[True], np.diff(f) > 0])
    return f[keep], e[keep]


def _dense_rescore(appearance, infos, cands, cfg: RefineConfig, w: int, fps: float, nms: int):
    """Rescore each candidate on dense embeddings and move it to its best row (spec 5.3).

    A move must keep the rules that chose the candidate: ``switch.min_side_seconds`` of samples
    on each side, and ``nms`` frames from every other candidate of the track.
    """
    if appearance is None or not hasattr(appearance, "dense_embeddings"):
        return
    sc = cfg.switch
    radius = max(1, int(cfg.encoder.sample_every))
    wanted = []
    for tid, idx in cands.items():
        info, lin = infos[tid]
        for i in idx:
            _, _, f0, f1 = _span(info, i, radius, w)
            wanted += _lineage_windows(lin, f0, f1)
    prefetch = getattr(appearance, "prefetch_dense", None)
    if prefetch is not None and wanted:
        prefetch(wanted)
    for tid, idx in cands.items():
        info, lin = infos[tid]
        fr = info["frames"]
        order = sorted(
            idx,
            key=lambda i, info=info, fr=fr: (
                -float(info["S"][i]),
                -float(info["A"][i]),
                -float(info["mot"][i]),
                int(fr[i]),
            ),
        )
        pos = {i: int(fr[i]) for i in idx}  # candidate -> its current frame
        moved, parts = [], []
        for i in order:
            lo, hi, f0, f1 = _span(info, i, radius, w)
            ef_d, emb_d = dense_track_embeddings(appearance, lin, f0, f1)
            others = [f for k, f in pos.items() if k != i]

            def eligible(j, info=info, fr=fr, others=others):
                t = int(fr[j])
                return all(abs(t - o) >= nms for o in others) and _sides_ok(
                    info, t, fps, sc.min_side_seconds
                )

            j = _relocate(info, i, sc, lo, hi, w, ef_d, emb_d, eligible)
            pos[i] = int(fr[j])
            moved.append(j)
            parts.append((ef_d, emb_d))
        cands[tid] = sorted(set(moved))
        info["ef"], info["emb"] = _merge_dense(info["ef"], info["emb"], parts)
```

5. In `propose_splits`, replace

```python
        idx = _candidates(info, fps, cfg, nms)
        infos[int(tid)] = (info, lin)
        cands[int(tid)] = idx
        if idx:
            result.candidates[int(tid)] = [int(info["frames"][i]) for i in idx]
```

   with

```python
        infos[int(tid)] = (info, lin)
        cands[int(tid)] = _candidates(info, fps, cfg, nms)
    _dense_rescore(appearance, infos, cands, cfg, w, fps, nms)
    for tid, idx in cands.items():
        if idx:
            result.candidates[tid] = [int(infos[tid][0]["frames"][i]) for i in idx]
```

   and add one sentence to the docstring paragraph of `propose_splits`: "With an appearance provider that has `dense_embeddings`, each candidate found on the coarse samples is rescored on dense samples and moved to its best frame (spec 5.3)."

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_switch_dense.py tests/refine/test_switch.py tests/refine/test_refiner.py -q`
Expected: all pass (existing switch and refiner tests use `ArrayAppearance` or no provider, so they take the old path).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/switch.py tests/refine/test_switch_dense.py
git commit -m "feat(refine): rescore stage 1 candidates on dense embeddings and cap the silhouette" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Wire the real provider into `TrackRefiner`, and the CLI error

**Files:**
- Modify: `src/dnt/refine/refiner.py`, `src/dnt/refine/cli.py`
- Modify (existing P1 tests that pass a video with the default `dino` encoder): `tests/refine/test_refiner.py`, `tests/refine/test_cli.py`
- Test: `tests/refine/test_appearance_refine.py`

**Interfaces:**
- Consumes: Tasks 1-6 (`check_encoder_dependencies`, `make_encoder`, `features_key`, `FeatureStore`, `VideoAppearance`, `CROP_PAD`, dense rescoring) and the test helpers of Task 5.
- Produces:
  - `TrackRefiner(config=None, config_yaml=None, device=None, *, appearance_factory=None, encoder_factory=None)`. `encoder_factory(encoder_cfg, target) -> AppearanceEncoder` replaces `make_encoder` (tests and custom encoders) and also skips the dependency check. The encoder is built once per `TrackRefiner` and reused by every `refine` call while the encoder settings **and the bytes of its local weights file** (`weights_identity`) are unchanged; a replaced weights file reloads the model and, through the cache key, misses the cache. A Hub model is not re-resolved by a running refiner; a new `TrackRefiner` loads whatever the name resolves to then, and the key follows the loaded parameters.
  - Behavior of `refine` with a video and `encoder.kind` of `dino` or `reid` (and no `appearance_factory`): dependencies are checked first (`ImportError` naming the extra, before any file is touched); the feature cache `OUT.features.npz` is reused if its key matches, else rebuilt; coarse samples are embedded in one video pass; the cache is written before the ledger; the ledger header records `inputs.features = {path, abs_path, sha256, cache_key}`. With no video, `kind: none`, or an `appearance_factory`, no cache is read or written and `inputs.features` stays `null`.
  - `dnt-refine run` exits 2 on `ImportError` too.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_appearance_refine.py`:

```python
import sys

import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.features import FeatureStore
from dnt.refine.refiner import TrackRefiner

from ._fakes import install_fake_torchreid, install_fake_transformers
from ._fixtures import box_rows, table
from ._video import BLUE, RED, ColorEncoder, make_color_video, video_rows


def _scene(tmp_path, n_frames=120, name="t.txt", video="v.mp4"):
    """One track: a red object for 60 frames, then a bigger blue one takes the ID over."""
    red = box_rows(1, range(60), 20.0, 100.0, vx=2.0, w=20.0, h=40.0)
    blue = box_rows(1, range(60, 120), 140.0, 100.0, vx=2.0, w=35.0, h=40.0)
    vid = make_color_video(
        tmp_path / video, video_rows(red, RED) + video_rows(blue, BLUE), n_frames
    )
    src = tmp_path / name
    table(red + blue).to_csv(src, index=False, header=False)
    return src, vid


def _cfg(**encoder):
    cfg = RefineConfig.defaults()
    cfg.link.enabled = False  # keep the two pieces apart: these tests are about stage 1
    for k, v in encoder.items():
        setattr(cfg.encoder, k, v)
    return cfg


def _run(src, out, video, enc, cfg=None, context=None):
    refiner = TrackRefiner(cfg or _cfg(), encoder_factory=lambda c, t: enc)
    refiner.refine(src, out, video_file=video, context_file=context, verbose=False)
    return refiner.last_result


def _header(res):
    return Ledger.read(res.ledger_path).header


def test_a_takeover_is_split_at_the_right_frame_from_appearance(tmp_path):
    src, video = _scene(tmp_path)
    res = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    splits = [e for e in res.events if e.kind is EventKind.SPLIT]
    assert len(splits) == 1
    ev = splits[0]
    assert ev.params["cut_frame"] == 60 and ev.signals["motion_only"] is False
    assert ev.algo_score >= RefineConfig.defaults().switch.accept_above
    assert ev.decision is Decision.AUTO_ACCEPT and ev.applied
    spans = res.tracks.groupby("track")["frame"].agg(["min", "max"]).to_numpy().tolist()
    assert spans == [[0, 59], [60, 119]]


def test_the_cache_is_written_recorded_and_reused(tmp_path):
    src, video = _scene(tmp_path)
    enc1 = ColorEncoder()
    r1 = _run(src, tmp_path / "o.txt", video, enc1)
    feats = tmp_path / "o.features.npz"
    rec = _header(r1)["inputs"]["features"]
    assert enc1.crops > 0 and feats.is_file()
    assert rec["sha256"] == io.sha256_file(feats) and len(rec["cache_key"]) == 64
    assert rec["path"] == str(feats)
    first_bytes = feats.read_bytes()

    enc2 = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video, enc2)
    assert enc2.calls == 0
    assert r2.tracks.equals(r1.tracks)
    assert _header(r2)["inputs"]["features"] == rec
    assert feats.read_bytes() == first_bytes
    assert [(e.id, e.kind, e.decision, e.algo_score) for e in r2.events] == [
        (e.id, e.kind, e.decision, e.algo_score) for e in r1.events
    ]


def test_a_different_video_is_a_cache_miss(tmp_path):
    src, video = _scene(tmp_path)
    r1 = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    key1 = _header(r1)["inputs"]["features"]["cache_key"]  # the next run rewrites this ledger
    _, video2 = _scene(tmp_path, n_frames=121, video="v2.mp4")
    enc = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video2, enc)
    assert enc.crops > 0
    assert _header(r2)["inputs"]["features"]["cache_key"] != key1


def test_a_different_encoder_setting_is_a_cache_miss(tmp_path):
    src, video = _scene(tmp_path)
    _run(src, tmp_path / "o.txt", video, ColorEncoder())
    enc = ColorEncoder()
    _run(src, tmp_path / "o.txt", video, enc, cfg=_cfg(sample_every=4))
    assert enc.crops > 0


def test_a_damaged_cache_is_rebuilt_not_fatal(tmp_path):
    src, video = _scene(tmp_path)
    feats = tmp_path / "o.features.npz"
    feats.write_bytes(b"garbage")
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc)
    assert enc.crops > 0 and len(res.tracks)
    key = _header(res)["inputs"]["features"]["cache_key"]
    assert FeatureStore.load(feats, key) is not None


def test_a_cache_with_the_right_key_but_the_wrong_width_is_rebuilt(tmp_path):
    import numpy as np

    src, video = _scene(tmp_path)
    r1 = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    rec = _header(r1)["inputs"]["features"]
    feats = tmp_path / "o.features.npz"
    store = FeatureStore.load(feats, rec["cache_key"])
    assert store is not None and len(store) > 0
    # same key, readable, but 5 columns where this encoder writes 3
    np.savez(
        feats,
        key=np.array(rec["cache_key"]),
        raw_id=np.array([1, 1], dtype=np.int64),
        frame=np.array([0, 5], dtype=np.int64),
        emb=np.ones((2, 5), dtype=np.float32),
    )
    enc = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video, enc)
    assert enc.crops > 0 and r2.tracks.equals(r1.tracks)
    assert FeatureStore.load(feats, rec["cache_key"], dim=3) is not None


def test_a_replaced_weights_file_reloads_the_encoder_and_misses_the_cache(tmp_path, monkeypatch):
    seen = install_fake_torchreid(monkeypatch)
    src, video = _scene(tmp_path)
    weights = tmp_path / "osnet.pt"
    weights.write_bytes(b"v1")
    refiner = TrackRefiner(_cfg(kind="reid", weights=str(weights), device="cpu"))

    def run():
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
        return _header(refiner.last_result)["inputs"]["features"]["cache_key"]

    key1 = run()
    assert seen["loads"] == 1
    assert run() == key1 and seen["loads"] == 1  # same weights: model and cache are reused
    weights.write_bytes(b"v2")  # replaced in place, same path, same refiner
    key3 = run()
    assert seen["loads"] == 2 and key3 != key1


def test_the_same_hub_model_name_with_new_weights_misses_the_cache(tmp_path, monkeypatch):
    seen = install_fake_transformers(monkeypatch)
    src, video = _scene(tmp_path)

    def run():
        refiner = TrackRefiner(_cfg(kind="dino", device="cpu"))  # a new refiner loads the model
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
        return _header(refiner.last_result)["inputs"]["features"]["cache_key"]

    key1 = run()
    assert run() == key1  # the name still resolves to the same weights: the cache is valid
    seen["seed"] = 1  # facebook/dinov2-small now resolves to other weights
    assert run() != key1


def test_a_detection_file_of_the_same_run_does_not_make_every_crop_occluded(tmp_path):
    src, video = _scene(tmp_path)
    raw = pd.read_csv(src, header=None)
    det = pd.DataFrame(
        {"f": raw[0], "res": -1, "x": raw[2], "y": raw[3], "w": raw[4], "h": raw[5],
         "conf": 0.9, "cls": 0}
    )
    ctx = tmp_path / "t_iou.txt"
    det.to_csv(ctx, index=False, header=False)
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc, context=ctx)
    assert enc.crops > 0
    assert [e.params["cut_frame"] for e in res.events if e.kind is EventKind.SPLIT] == [60]


def test_a_track_with_no_clean_crops_is_handled(tmp_path):
    rows = box_rows(1, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0) + box_rows(
        2, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0
    )
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 60)
    src = tmp_path / "t.txt"
    table(rows).to_csv(src, index=False, header=False)
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc)  # both boxes always overlap each other
    assert enc.calls == 0 and len(res.tracks) > 0


def test_kind_none_with_a_video_is_motion_only_and_writes_no_cache(tmp_path):
    src, video = _scene(tmp_path)
    enc = ColorEncoder()
    with_video = _run(src, tmp_path / "o.txt", video, enc, cfg=_cfg(kind="none"))
    assert enc.calls == 0 and _header(with_video)["inputs"]["features"] is None
    assert not (tmp_path / "o.features.npz").exists()
    refiner = TrackRefiner(_cfg(kind="none"))
    no_video = refiner.refine(src, tmp_path / "p.txt", fps=10, verbose=False)
    assert no_video.equals(with_video.tracks)


def test_a_missing_package_fails_before_any_file_is_written(tmp_path, monkeypatch):
    src, video = _scene(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", None)
    refiner = TrackRefiner(_cfg())  # kind "dino", no encoder_factory
    with pytest.raises(ImportError, match=r"dnt\[refine-dino\]"):
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]
    # the same call without a video does not need the package
    refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)


def test_refine_batch_builds_the_encoder_once(tmp_path):
    srcs, videos = [], []
    for name in ("a", "b"):
        s, v = _scene(tmp_path, name=f"{name}_track.txt", video=f"{name}.mp4")
        srcs.append(s)
        videos.append(v)
    built = []
    enc = ColorEncoder()

    def factory(cfg, target):
        built.append(target)
        return enc

    refiner = TrackRefiner(_cfg(), encoder_factory=factory)
    outs = refiner.refine_batch(srcs, video_files=videos, output_path=tmp_path / "out", verbose=False)
    assert len(outs) == 2 and built == ["person"]
    assert all((tmp_path / "out" / f"{n}_refined.features.npz").is_file() for n in ("a", "b"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_appearance_refine.py -q`
Expected: FAIL (`TrackRefiner` has no `encoder_factory`).

- [ ] **Step 3: Implement**

In `src/dnt/refine/refiner.py`:

1. Imports. Replace `from .features import Appearance` with `from .features import Appearance, FeatureStore, features_key` and add, in alphabetical position with the other relative imports:

```python
from .crops import CROP_PAD
from .encoders import check_encoder_dependencies, make_encoder, weights_identity
```

   and, after `from .verify import Band, decide, route_without_vlm`:

```python
from .video_appearance import VideoAppearance
```

2. `TrackRefiner.__init__`: add the keyword-only parameter and state. Replace

```python
        appearance_factory: Callable[..., Appearance | None] | None = None,
    ) -> None:
```

   with

```python
        appearance_factory: Callable[..., Appearance | None] | None = None,
        encoder_factory: Callable[..., object] | None = None,
    ) -> None:
```

   and replace `self.appearance_factory = appearance_factory` with

```python
        self.appearance_factory = appearance_factory
        self.encoder_factory = encoder_factory
        self._encoder_memo: tuple[tuple, object] | None = None
```

3. In `refine`, directly after `paths = output_paths(out)` insert:

```python
        if (
            video_file is not None
            and cfg.encoder.kind != "none"
            and self.appearance_factory is None
            and self.encoder_factory is None
        ):
            check_encoder_dependencies(cfg.encoder)  # before any processing (spec 5.5)
```

   and add to the `Raises` section of the `refine` docstring:

```
        ImportError
            If a video is given, ``encoder.kind`` is ``dino`` or ``reid``, and the encoder's
            package is not installed; the message names the pip extra.
```

4. Replace the line `appearance = self._appearance(work, video_file, ctx_boxes, fps_val)` with

```python
        appearance, store = self._appearance(
            work,
            video_file,
            ctx_boxes,
            fps_val,
            key_parts={
                "tracks_sha": track_sha,
                "video": (inputs["video"] or {}).get("fingerprint"),
                "context_sha": context_sha,
            },
            features_path=paths["features"],
        )
```

5. Directly after the `with tqdm(...) as pbar:` block (the line `work, id_map = renumber(work)` follows it), insert before `work, id_map = renumber(work)`:

```python
        if store is not None:
            sha = store.save(paths["features"])
            inputs["features"] = _file_record(paths["features"], sha, cache_key=store.key)
```

6. Replace the whole `_appearance` method with these two methods:

```python
    def _encoder(self):
        """Return the encoder, reused while the settings and the weights file are unchanged."""
        cfg = self.config.encoder
        # a weights file replaced in place under the same path must not reuse the old model
        identity = None
        if self.encoder_factory is None:
            identity = weights_identity(cfg, self.config.target)
        key = (
            cfg.kind,
            cfg.model,
            cfg.weights,
            cfg.device,
            cfg.batch_size,
            self.config.target,
            identity,
        )
        if self._encoder_memo is None or self._encoder_memo[0] != key:
            factory = self.encoder_factory or make_encoder
            self._encoder_memo = (key, factory(cfg, self.config.target))
        return self._encoder_memo[1]

    def _appearance(self, work, video, context, fps, *, key_parts, features_path):
        """Return ``(appearance, store)``; ``store`` is the feature cache to save, or None."""
        if self.appearance_factory is not None:
            app = self.appearance_factory(
                work=work, video=video, context=context, fps=fps, config=self.config
            )
            return app, None
        cfg = self.config.encoder
        if video is None or cfg.kind == "none":
            return None, None
        encoder = self._encoder()
        key = features_key(
            **key_parts,
            encoder=encoder,
            sample_every=cfg.sample_every,
            occlusion_iou=cfg.occlusion_iou,
            crop_pad=CROP_PAD,
        )
        loaded = FeatureStore.load(features_path, key, dim=encoder.dim)
        store = loaded if loaded is not None else FeatureStore(key)
        occluded = occlusion_flags(work, context, cfg.occlusion_iou)
        app = VideoAppearance(
            work,
            occluded,
            video,
            encoder,
            store,
            sample_every=cfg.sample_every,
            batch_size=cfg.batch_size,
            crop_pad=CROP_PAD,
        )
        app.prefetch_coarse()
        return app, store
```

   (A loaded store may be empty, which is falsy, so it is tested with `is not None`.)

In `src/dnt/refine/cli.py`, change `except (ValueError, FileNotFoundError) as exc:` to `except (ValueError, FileNotFoundError, ImportError) as exc:`.

7. Update the P1 tests that pass a video while the default encoder is `dino` (they are about motion-only behavior):
   - `tests/refine/test_refiner.py`: in the `_refine` helper, before `refiner = TrackRefiner(cfg)`, add
     ```python
         if kw.get("video_file") is not None:
             cfg = cfg if cfg is not None else RefineConfig.defaults()
             cfg.encoder.kind = "none"  # these tests are about the video's metadata, not appearance
     ```
     and in `test_video_supplies_fps_frame_size_and_fingerprint` replace `assert "motion-only" in caplog.text` with `assert "Plan 2" not in caplog.text`.
   - `tests/refine/test_cli.py`: change `_cfg` to accept encoder overrides and use it in the `--video` test:
     ```python
     def _cfg(tmp_path, name="c.yaml", **encoder):
         cfg = tmp_path / name
         c = RefineConfig.defaults()
         for k, v in encoder.items():
             setattr(c.encoder, k, v)
         c.to_yaml(cfg)
         return cfg
     ```
     and pass `_cfg(tmp_path, kind="none")` in the test that runs `--video` (the `_main(capsys, "run", src, "--video", video, ...)` call). Add this test (adding `import sys` and `from ._fixtures import box_rows, table` if they are not imported yet):
     ```python
     def test_missing_encoder_package_is_a_clean_error(tmp_path, capsys, synthetic_video, monkeypatch):
         monkeypatch.setitem(sys.modules, "transformers", None)
         video, _ = synthetic_video
         src = tmp_path / "t.txt"
         table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)).to_csv(src, index=False, header=False)
         code, stdout, err = _main(
             capsys, "run", src, "--video", video, "--config", _cfg(tmp_path),
             "--out", tmp_path / "o.txt",
         )
         assert code == 2 and "refine-dino" in err and stdout == ""
         assert not (tmp_path / "o.txt").exists()
     ```
   Run the whole `tests/refine` directory to find any other P1 test that passes a video with the default encoder (fix the same way: `encoder.kind = "none"`).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine tests/test_refine_independence.py -q`
Expected: all pass except `test_cli.py::test_docs_and_changelog_state_the_limitation_accurately`,
which already fails on `main` (Task 8 repairs it). Any other failure is a P1 test that passes a
video with the default encoder: fix it as above.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/refiner.py src/dnt/refine/cli.py tests/refine/test_appearance_refine.py tests/refine/test_refiner.py tests/refine/test_cli.py
git commit -m "feat(refine): use the video appearance provider in refine, with a feature cache" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Minimal-install test, docs, changelog, and the final gate

**Files:**
- Create: `tests/refine/test_minimal_install.py`, `docs/api/refine/appearance.md`
- Modify: `docs/api/refine/index.md`, `docs/changelog.md`, `CHANGELOG.md` (mirror of `docs/changelog.md`), `mkdocs.yml` (nav), `tests/refine/test_cli.py` (the docs test)

**Interfaces:**
- Consumes: everything above. The existing docs test `test_docs_and_changelog_state_the_limitation_accurately` pins phrases of the "Current limitations" note and of the changelog bullet that starts `- This release scores with motion only`. **It currently fails on `main`**: the 0.3.4 release commit renamed the `## Unreleased` heading to `## 0.3.4`, and the test splits on `## Unreleased`. This task repairs it (it reads the newest changelog section instead) and adds the new section.
- Produces: user-facing documentation of the appearance feature, and a green full suite.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_minimal_install.py`:

```python
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import sys
    for name in ("transformers", "torchreid", "openai", "anthropic"):
        sys.modules[name] = None  # a minimal install: none of the optional packages
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    from dnt.refine import RefineConfig, TrackRefiner

    out = Path(sys.argv[1])
    cfg = RefineConfig.defaults()  # encoder.kind is "dino", but the package is missing
    rows = [[f, 1, 100.0 + 2 * f, 100.0, 30.0, 60.0, 0.9, 0, -1, -1] for f in range(60)]
    src = out / "t.txt"
    pd.DataFrame(rows).to_csv(src, index=False, header=False)
    vid = out / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(60):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()

    refiner = TrackRefiner(cfg)
    refiner.refine(src, out / "a.txt", fps=10, verbose=False)  # no video: no package needed
    assert (out / "a.txt").is_file()
    try:
        refiner.refine(src, out / "b.txt", video_file=vid, verbose=False)
    except ImportError as err:
        assert "refine-dino" in str(err), err
    else:
        raise SystemExit("expected an ImportError for a video with encoder.kind dino")
    assert not (out / "b.txt").exists() and not (out / "b.ledger.jsonl").exists()
    cfg.encoder.kind = "none"
    refiner.refine(src, out / "c.txt", video_file=vid, verbose=False)
    assert (out / "c.txt").is_file() and not (out / "c.features.npz").exists()
    print("OK")
    """
)


def test_a_minimal_install_runs_without_an_encoder_and_fails_early_with_one_requested(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout
```

In `tests/refine/test_cli.py`, replace the first lines of `test_docs_and_changelog_state_the_limitation_accurately`

```python
    log = (ROOT / "docs/changelog.md").read_text()
    unreleased = log.split("## Unreleased", 1)[1].split("\n## ", 1)[0]
```

with

```python
    log = (ROOT / "docs/changelog.md").read_text()
    assert log.startswith("# Changelog\n\n## Unreleased\n")
    unreleased = log.split("## Unreleased", 1)[1].split("\n## ", 1)[0]
```

and append these lines at the end of that test (the existing `assert log.index("## Unreleased") < log.index("## 0.3.3")` stays):

```python
    # plan 2: the appearance feature is documented where users look for it
    assert "refine-dino" in index and "encoder.kind: none" in index and "features.npz" in index
    assert "refine-dino" in unreleased and "encoder_factory" in unreleased
    assert "api/refine/appearance.md" in (ROOT / "mkdocs.yml").read_text()
    assert (ROOT / "docs/api/refine/appearance.md").is_file()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_minimal_install.py tests/refine/test_cli.py -q`
Expected: `test_minimal_install` passes already (Task 7 did the work), and the docs test fails (`## Unreleased` is missing).

- [ ] **Step 3: Write the docs**

1. `docs/changelog.md`: insert this section directly under `# Changelog` (before `## 0.3.4 — 2026-10-01`), then copy the whole file to `CHANGELOG.md` (`cp docs/changelog.md CHANGELOG.md`; the test requires them to be identical):

```markdown
## Unreleased

### New
- `dnt.refine` looks at appearance. With a video and `encoder.kind` of `dino` (DINOv2, the
  default) or `reid` (torchreid OSNet), `refine` crops each box, skips crops that other boxes
  occlude, and embeds the rest. Stage 1 (ID-switch splits) and stage 3 (links) score with the
  embeddings, so a split or link that scores high enough is applied. The embeddings are cached
  next to the output as `OUT.features.npz`, and a rerun on the same inputs reuses them. New
  extras: `pip install 'dnt[refine-dino]'`, `'dnt[refine-reid]'`, or `'dnt[refine]'` for both.
- This release scores with motion only unless a video and an appearance encoder are given, and
  applies only the edits it is sure of: in-vehicle and duplicate false-track drops, rider
  reclasses whose subtype a ReClass hint settles, links across short gaps and static waits with
  a clear assignment margin, orphan drops, and filling. With an encoder, ID-switch splits and
  links are also scored by appearance and applied when they score high enough. Other edits are
  capped below auto-accept, recorded as `HUMAN_PENDING`, and not applied yet: ID-switch splits
  found from motion alone, links across occlusions, links with an ambiguous assignment margin,
  and false-track drops of static objects or of mixed tracks. Rider reclasses whose subtype no
  ReClass hint settles are pending too, however high they score, because only a hint can choose
  the subtype in this release. In-vehicle drops need a context file with the vehicles' boxes
  (`context_file=`, or `--context`); without one the in-vehicle cue is skipped. VLM
  verification, review pages, and applying review decisions follow in later releases.

### Changed
- `refine` with a video now needs the encoder's package (`pip install 'dnt[refine-dino]'`) or
  `encoder.kind: none`. Before, it logged a warning and ran on motion alone. `dnt-refine run`
  exits with code 2 and names the extra when the package is missing. `TrackRefiner` accepts
  `encoder_factory=` to supply your own encoder.

```

2. `docs/api/refine/index.md`: replace the whole `!!! note "Current limitations"` block with:

````markdown
!!! note "Current limitations"
    This release scores with motion only unless you give a video and an appearance encoder
    (see Appearance below), and cannot yet apply decisions made in review.
    Edits it is sure of are applied: in-vehicle and duplicate false-track drops, rider
    reclasses whose subtype a ReClass hint settles, links across short gaps and static waits
    with a clear assignment margin, orphan drops, and gap filling. With an encoder, ID-switch
    splits and links are also scored by appearance and applied when they score high enough.
    Other edits are proposed but never applied yet, because their scores are capped below
    auto-accept: ID-switch splits found from motion alone, links across occlusions, links with
    an ambiguous assignment margin, and false-track drops of static objects or of mixed tracks.
    Rider reclasses whose subtype no ReClass hint settles are pending too, however high they
    score, because only a hint can choose the subtype in this release. They appear in the
    ledger as `HUMAN_PENDING` and leave the tracks unchanged. In-vehicle drops need a context
    file with the vehicles' boxes (`context_file=`, or `--context`); without one the in-vehicle
    cue is skipped. VLM verification and applying review decisions follow in later releases.

## Appearance

With a video, `refine` also compares what tracks look like. It crops each box, skips crops that
another box overlaps, embeds the rest with a pretrained encoder, and uses the embeddings to
find ID switches (stage 1) and to score links (stage 3). Install an encoder first:

```bash
pip install 'dnt[refine-dino]'   # DINOv2 (the default, kind: dino)
pip install 'dnt[refine-reid]'   # torchreid OSNet (kind: reid)
```

```yaml
encoder:
  kind: dino        # dino | reid | none
  model: facebook/dinov2-small
  device: auto      # cuda, xpu, mps, then cpu
  sample_every: 5   # embed every 5th observed frame; stage 1 densifies around candidates
```

`encoder.kind: none`, or no video, scores with motion only and needs no extra. A video with the
default `dino` encoder and no package installed raises `ImportError` before any work starts.
The embeddings are saved as `OUT.features.npz` next to the output and reused when the track
file, video, context file, and encoder settings are unchanged. For vehicles, `reid` needs
`encoder.weights`. The cache key includes a digest of the weights that were actually loaded, so a
model that changes under the same name never reuses old embeddings.
````

   (keep the `::: dnt.refine` line at the end of the page).

3. `docs/api/refine/appearance.md`:

```markdown
# Appearance

::: dnt.refine.encoders

::: dnt.refine.features

::: dnt.refine.video_appearance
```

4. `mkdocs.yml`: after the line `- Linking: api/refine/link.md` add `- Appearance: api/refine/appearance.md` (same indentation).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine tests/test_refine_independence.py -q`
Expected: all pass.

- [ ] **Step 5: Gate**

```bash
.venv/bin/python -m pytest -q                      # whole default suite (about 8 minutes, timeout 600000 ms)
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src/dnt/refine
git status --short | grep -v '^??'                 # only the files named in this task are modified
```

Build the docs into a temporary directory outside the repository. `site/` is committed build output and must not change. The `with-pdf` plugin already fails the strict build on `main` ("No anchor" errors), so the check uses a temporary config without it:

```bash
T=$(mktemp -d) && .venv/bin/python - "$T" <<'EOF'
import pathlib
import re
import sys

t, root = pathlib.Path(sys.argv[1]), pathlib.Path.cwd()
s = pathlib.Path("mkdocs.yml").read_text()
s = re.sub(r"  - with-pdf:\n(?:      .*\n)+", "", s)
s = s.replace("docs_dir: docs", f"docs_dir: {root}/docs")
s = s.replace("site_dir: site", f"site_dir: {t}/site").replace("paths: [src]", f"paths: [{root}/src]")
s = s.replace("watch:\n  - src/dnt\n", f"watch:\n  - {root}/src/dnt\n")  # relative to the new config
(t / "mkdocs.yml").write_text(s)
EOF
.venv/bin/mkdocs build --strict -f "$T/mkdocs.yml"
git status --short site                            # empty
```

Expected: the build succeeds with no warnings; `site/` is untouched.

- [ ] **Step 6: Commit**

```bash
git add tests/refine/test_minimal_install.py tests/refine/test_cli.py docs/api/refine/appearance.md docs/api/refine/index.md docs/changelog.md CHANGELOG.md mkdocs.yml
git commit -m "docs(refine): document the appearance encoders; fix the changelog test after the 0.3.4 release" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review (run by the plan's author)

**Spec coverage.**
- §5.3 coarse samples, occlusion mask, dense re-sampling around stage 1 candidates, cache and key, fingerprint reuse: Tasks 4-7 (`features_key` takes the fingerprint P1 already computes).
- §5.5 encoders `dino`, `reid`, `none`, dependency checks deferred to the start of `refine`, extras: Tasks 1, 3, 7.
- §9 `encoder` block and §10 rows (no video, `kind: none` with video, missing extra): P1's config already holds the fields; Tasks 7-8 give the behavior and tests.
- §11.4 `test_features_cache` (Task 4 and 7), `test_minimal_install` (Task 8), `test_encoders_real` (Task 3). `test_fingerprint.py` (65 MiB files) is a P1 `io.video_fingerprint` test and is not repeated here.
- §12 extras `refine-dino`, `refine-reid`, `refine` (the union; `refine-vlm` joins it in Plan 3).
- Left to Plan 4 on purpose: `apply`-time cache rules (§4.2: cache hit/miss/invalid with and without the video), the `--features` flag, `features_recomputed` in the header. Left to Plan 3: evidence crops (§7.1 uses its own 1.5x padding).

**Placeholders.** None: every code step shows its code.

**Type consistency.** `AppearanceEncoder` attributes (`name`, `model_name`, `preprocess_id`, `weights_sha`, `dim`, `encode`) are the same in Task 1, the encoders (Task 3), the key (Task 4), the stub `ColorEncoder` (Task 5) and the refiner (Task 7). `VideoAppearance` implements `clean_embeddings`, `dense_embeddings`, `prefetch_dense` with the signatures of `DenseAppearance` (Task 4), which `switch._dense_rescore` (Task 6) duck-types with `hasattr`.

**Review Focus coverage.** (1) no clean crops: Task 5 `test_a_track_with_no_clean_crops_gives_empty_arrays`, Task 7 `test_a_track_with_no_clean_crops_is_handled`. (2) unreadable video: Task 2 and Task 5 tests. (3) stale or damaged cache: Task 3 (the same Hub name with new weights changes `weights_sha`), Task 4 (every key part; malformed-but-readable archives), Task 7 (a replaced weights file, a changed Hub model, a different video, a garbage file). (4) same-run detections as context: Task 7. (5) once-per-batch model load, bounded batches, device fallback, batch-size invariance: Task 7 `test_refine_batch_builds_the_encoder_once`, Task 5, Task 3.

**Known limits, stated plainly.** (a) A run that fails midway does not save the embeddings it computed; the next run recomputes them. (b) On a GPU, batch composition can shift embeddings in the last bits; the cache makes a rerun reproducible, and CPU runs are exact. (c) The default encoder is `dino` for both targets (spec §13 leaves OSNet as the alternative to compare on real clips). (d) The thresholds of §6 were chosen without appearance data; expect to tune `switch.*` and `link.*` on real clips after this lands. (e) A running `TrackRefiner` does not re-resolve a Hub model name; make a new one to pick up changed Hub weights (the key then follows the loaded parameters). `torchreid` on PyPI is an old release; if `pip install 'dnt[refine-reid]'` does not give a working `torchreid.utils.FeatureExtractor`, install it from its GitHub repository (the code only needs that class).
