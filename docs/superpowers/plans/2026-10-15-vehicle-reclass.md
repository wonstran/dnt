# Vehicle Re-classification (`VehReClass`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dnt.track.VehReClass`, which gives every vehicle track one class (car 2, bus 5 or truck 7) through a size → re-detection → VLM → fallback cascade, and writes it into the track file's `cls` column.

**Architecture:** Everything new lives in `src/dnt/track/re_class.py`: small pure helpers (size, spaced sampling, IoU, vote, tile), a tolerant frame-reading and one-pass detection helper that `ReClass` also moves onto, a VLM helper that reuses `dnt.refine`'s runner, backends and cache, and the `VehReClass` class that wires the layers together. Three small extractions in existing code make the reuse clean: `refine.io.filled_mask`, `refine.config.vlm_problems` and `Detector.detect_image`.

**Tech Stack:** Python 3.11–3.12, numpy, pandas, OpenCV, pytest, ruff. No new dependencies; the VLM backends stay behind the existing `refine-vlm` extra.

**Spec:** [docs/superpowers/specs/2026-10-15-vehicle-reclass-design.md](../specs/2026-10-15-vehicle-reclass-design.md) (rev. 2). Read §3, §4 and §5 before Task 5. Review and response: [review_2026-10-15-22-04-16.md](../../review_2026-10-15-22-04-16.md), [response_2026-10-15_22-40-00.md](../../response_2026-10-15_22-40-00.md).

**Plan revisions:** rev. 2 addresses [review 2026-10-15 22:36](../../review_2026-10-15-22-36-11.md) / [response](../../response_2026-10-15_23-05-00.md): VLM results keep the original track IDs (Task 6), and voting tests pin the confidence-weighted winner, the tie breakers and the `min_support` boundary (Tasks 5 and 7). Rev. 3 addresses [review 2026-10-15 22:48](../../review_2026-10-15-22-48-49.md) / [response](../../response_2026-10-15_23-20-00.md): the more-votes tie-break test now pits vote count against class ID.

## Global Constraints

- Run Python with the project venv: `.venv/bin/python -m pytest ...`. Record the default-suite count (`.venv/bin/python -m pytest -q`) before Task 1; it must only grow.
- `ruff check src tests tools` must stay clean; line length 100; rules `E,F,I,UP,B,SIM,RUF,D`. The per-file baseline in `pyproject.toml` may shrink but must not grow. New public functions and classes in `src/` need numpy-style docstrings.
- Imports are package-relative (`from ..refine.io import filled_mask`). `dnt.refine` must never import `dnt.track` (the direction is track → refine only).
- Class IDs are COCO: `CAR = 2`, `BUS = 5`, `TRUCK = 7`. Heavy-flagged tracks allow only bus and truck.
- Defaults, verbatim from the spec: `hv_size=None`, `hv_min_count=3`, `num_frames=10`, `min_gap=5`, `iou_threshold=0.5`, `min_support=0.3`, `vlm=None`, `vlm_crops=3`, `model=DetectorModel.RTDETRx`, `weights=None`, `device="auto"`. Layer 3's threshold is `vlm.min_conf` (VLMConfig default 0.7); there is no `vlm_min_conf` parameter.
- VLM answer options are exactly `car, bus, truck, unclear` (unflagged) and `bus, truck, unclear` (heavy). The question tag is `vehclass:<track id>`.
- Fallback: heavy → truck (7), otherwise car (2).
- Tests need no GPU and no network: fake detector, fake frame reader, `refine.vlm.fake.FakeBackend`, and `cache_dir` under `tmp_path`.
- When a step appends tests to an existing file, put any new `import` lines in that file's top import block, not next to the tests (ruff `E402`/`I001` apply to `tests/`; only `D` and `E501` are ignored there).
- Run `.venv/bin/ruff format` on every `src/` file you change before committing, then `ruff check`.
- Work on a branch: `git switch -c feature/veh-reclass` before Task 1.
- Commit trailer: end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

Inputs the spec implies but does not spell out, most likely to bite first. Each has a test in the task named in brackets.

1. **Float frame numbers and float track IDs.** A DataFrame whose `frame` or `track` column is float (`12.0`, `1.0`), as some post-processing produces, must classify exactly like the int version, including a track that reaches the VLM; the output keeps the input's `track` values and dtype. [Task 7, `test_float_frame_column_matches_int`, `test_vlm_with_int_or_float_track_ids`; Task 6, `test_ask_vlm_keeps_original_ids`]
2. **Every sampled frame unreadable** (track frames past the video's end). `n` is 0, support is NaN, the track falls back without an exception. [Task 7, `test_all_frames_unreadable_falls_back`]
3. **Duplicate `(track, frame)` rows.** Sampling never takes one frame twice, even with `min_gap=0`. [Task 5, `test_sample_never_repeats_a_frame`]
4. **Degenerate boxes** (`w` or `h` of 0, or a box outside the image). IoU is 0 rather than NaN, and a track with no usable crop is left out of the VLM batch instead of sending an empty image. [Task 5, `test_iou_zero_area_is_zero`; Task 6, `test_track_without_usable_crop_is_not_asked`]
5. **`track_ids` as a numpy array.** It works like a list, including the unknown-ID warning. [Task 7, `test_track_ids_numpy_array`]

## File Structure

| File | Responsibility |
|---|---|
| `src/dnt/refine/io.py` (modify) | `filled_mask(df)`: the filled-row rule, shared by `to_work` and `VehReClass`. |
| `src/dnt/refine/config.py` (modify) | `vlm_problems(vlm)`: VLM validation and endpoint resolution, shared by `RefineConfig.validate` and `VehReClass`. |
| `src/dnt/detect/yolo/detector.py` (modify) | `Detector.detect_image(image, frame)`; `detect_frames` calls the same per-image code. |
| `src/dnt/refine/vlm/prompts.py` (modify) | Vehicle-class options and prompt. |
| `src/dnt/track/re_class.py` (modify) | Shared helpers, `ReClass` on the batched detection helper, `VehReClass`. |
| `src/dnt/track/__init__.py` (modify) | Export `VehReClass`; drop the `ModuleNotFoundError` fallback. |
| `tests/support/reclass_fakes.py` (create) | `FakeReader`, `FakeDetector`, `vehicle_rows`. |
| `tests/refine/test_io.py`, `tests/refine/test_config.py`, `tests/test_detector.py`, `tests/refine/test_vlm_cache_prompts.py` (modify) | Tests for the extractions. |
| `tests/test_reclass.py` (create) | `ReClass` characterization and the shared reading helpers. |
| `tests/test_veh_reclass.py` (create) | `VehReClass` helpers, VLM layer and class. |
| `pyproject.toml`, `CLAUDE.md`, `docs/changelog.md` (modify) | Baseline shrink, notes, changelog. |

---

### Task 1: `refine.io.filled_mask`

**Files:**
- Modify: `src/dnt/refine/io.py` (add the function above `to_work`; use it inside `to_work`)
- Test: `tests/refine/test_io.py`

**Interfaces:**
- Produces: `filled_mask(df: pd.DataFrame) -> numpy.ndarray` (bool, one entry per row).

- [ ] **Step 1: Write the failing tests** (append to `tests/refine/test_io.py`)

```python
import pandas as pd

from dnt.refine.io import filled_mask


def test_filled_mask_r3_or_interp():
    df = pd.DataFrame({"r3": [1, -1, -1, 0], "interp": [0, 1, 0, 0]})
    assert filled_mask(df).tolist() == [True, True, False, False]


def test_filled_mask_missing_or_text_counts_as_real():
    df = pd.DataFrame({"interp": [None, "x", 1]})
    assert filled_mask(df).tolist() == [False, False, True]


def test_filled_mask_without_flag_columns():
    assert filled_mask(pd.DataFrame({"frame": [1, 2]})).tolist() == [False, False]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_io.py -k filled_mask -v`
Expected: FAIL with `ImportError: cannot import name 'filled_mask'`.

- [ ] **Step 3: Implement** (in `src/dnt/refine/io.py`, above `to_work`)

```python
def filled_mask(df: pd.DataFrame) -> np.ndarray:
    """Return a boolean array that is True where a row is a filled (interpolated) row.

    The flag is column 8: ``r3`` in the tracker layout and ``interp`` in interpolated output.
    A row is filled when either column holds 1; a missing or non-numeric value counts as real.

    Parameters
    ----------
    df : pandas.DataFrame
        A track table with named columns.

    Returns
    -------
    numpy.ndarray
        One ``bool`` per row.

    """
    filled = np.zeros(len(df), dtype=bool)
    for flag in ("r3", "interp"):
        if flag in df.columns:
            filled |= (pd.to_numeric(df[flag], errors="coerce").fillna(-1) == 1).to_numpy()
    return filled
```

In `to_work`, replace the four lines that build `filled` (from `filled = np.zeros(...)` through the `for flag in ("r3", "interp"):` loop) with:

```python
    filled = filled_mask(df)
```

- [ ] **Step 4: Run the refine I/O tests and the whole refine suite**

Run: `.venv/bin/python -m pytest tests/refine/test_io.py -v` then `.venv/bin/python -m pytest tests/refine -q`
Expected: all pass (the `to_work` behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/io.py tests/refine/test_io.py
git commit -m "refactor(refine): share the filled-row rule as io.filled_mask

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `refine.config.vlm_problems`

**Files:**
- Modify: `src/dnt/refine/config.py` (new function after the `VLMConfig` class; `RefineConfig.validate` calls it)
- Test: `tests/refine/test_config.py`

**Interfaces:**
- Produces: `vlm_problems(vlm: VLMConfig) -> tuple[list[str], VLMConfig]`. Never mutates its argument. The second value is a deep copy with blank `api_key_env`/`api_key_file`/`base_url` set to None and, when `use` names a known endpoint, that endpoint applied (`resolve()`); when the endpoints are broken it is the unresolved copy.

- [ ] **Step 1: Write the failing tests** (append to `tests/refine/test_config.py`)

```python
import copy

import pytest

from dnt.refine.config import RefineConfig, VLMConfig, vlm_problems


def test_vlm_problems_resolves_named_endpoint_without_mutating():
    vlm = VLMConfig(
        backend="none",
        endpoints={"local": {"backend": "openai_compat", "model": "qwen", "base_url": "http://localhost:8000/v1"}},
        use="local",
    )
    before = copy.deepcopy(vlm)
    problems, resolved = vlm_problems(vlm)
    assert problems == []
    assert (resolved.backend, resolved.model, resolved.use, resolved.endpoints) == (
        "openai_compat", "qwen", None, {}
    )
    assert vlm == before


def test_vlm_problems_unknown_endpoint():
    problems, _ = vlm_problems(VLMConfig(use="nope"))
    assert problems == ["vlm.use 'nope' is not in vlm.endpoints"]


def test_vlm_problems_bad_votes_and_blank_strings():
    problems, resolved = vlm_problems(VLMConfig(votes=0, api_key_env="  "))
    assert "vlm.votes must be an integer >= 1" in problems
    assert resolved.api_key_env is None


def test_refine_validate_still_reports_vlm_rules():
    cfg = RefineConfig.defaults("person")
    cfg.vlm.votes = 0
    cfg.vlm.backend = "openai_compat"
    with pytest.raises(ValueError) as exc:
        cfg.validate()
    assert "vlm.votes must be an integer >= 1" in str(exc.value)
    assert "vlm.model is required for this backend" in str(exc.value)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py -k "vlm_problems or still_reports" -v`
Expected: FAIL with `ImportError: cannot import name 'vlm_problems'`.

- [ ] **Step 3: Implement**

Add after the `VLMConfig` class:

```python
def vlm_problems(vlm: VLMConfig) -> tuple[list[str], VLMConfig]:
    """Check VLM settings and resolve the endpoint selected by ``use``.

    Parameters
    ----------
    vlm : VLMConfig
        The settings to check. They are not changed.

    Returns
    -------
    tuple of (list of str, VLMConfig)
        The broken rules (empty when the settings are valid), and a deep copy with blank
        ``api_key_env``, ``api_key_file`` and ``base_url`` set to None and, when the endpoints
        are valid, the chosen endpoint applied (see ``VLMConfig.resolve``).

    """
    out = copy.deepcopy(vlm)
    for name in ("api_key_env", "api_key_file", "base_url"):
        v = getattr(out, name)
        if isinstance(v, str) and not v.strip():
            setattr(out, name, None)  # "" from YAML: the default (variable, endpoint)
    p: list[str] = []
    endpoints_ok = True
    for ename, ep in out.endpoints.items():
        bad = set(ep) - _ENDPOINT_KEYS
        if bad:
            p.append(f"vlm.endpoints.{ename}: unknown keys {sorted(bad)}")
            endpoints_ok = False
    if out.use is not None and out.use not in out.endpoints:
        p.append(f"vlm.use '{out.use}' is not in vlm.endpoints")
        endpoints_ok = False
    vl = out.resolve() if endpoints_ok else out
    if vl.backend not in BACKENDS:
        p.append(f"vlm.backend must be one of {BACKENDS}")
    if vl.backend not in ("none", "anthropic") and not vl.model:
        p.append("vlm.model is required for this backend")

    def _num(x):
        return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)

    # --- MOVE HERE, unchanged, every check from RefineConfig.validate that starts at
    # --- `if not (isinstance(vl.votes, int) ...` and ends with the `vlm.api_key_file ... not the key`
    # --- elif. They already use the name `vl` and append to `p`.
    return p, vl
```

The comment block marks a verbatim move, not new code: cut the lines from `if not (isinstance(vl.votes, int) and not isinstance(vl.votes, bool) and vl.votes >= 1):` through `p.append("vlm.api_key_file must be the path of a file that holds the key, not the key")` out of `validate` and paste them in its place; then delete the three marker comment lines.

In `RefineConfig.validate`:
- keep the existing in-place blank-string loop on `self.vlm` at the top (other code relies on `validate` normalizing `self.vlm`);
- replace the block from `vlm = self.vlm` through `if vlm.backend not in BACKENDS: p.append(...)` with:

```python
        vlm_p, _ = vlm_problems(self.vlm)
        p.extend(vlm_p)
```

- delete the now-moved lines: `if vlm.backend not in ("none", "anthropic") and not vlm.model: ...`, `vl = vlm`, the local `_num`, and the moved checks.

- [ ] **Step 4: Run the config tests and the refine suite**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py -v` then `.venv/bin/python -m pytest tests/refine -q`
Expected: all pass. Only the order of messages inside one `ValueError` can change; no test compares the whole message.

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check src/dnt/refine/config.py tests/refine/test_config.py`
Expected: no errors.

```bash
git add src/dnt/refine/config.py tests/refine/test_config.py
git commit -m "refactor(refine): move VLM validation into config.vlm_problems

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `Detector.detect_image`

**Files:**
- Modify: `src/dnt/detect/yolo/detector.py` (`detect_frames`, new `detect_image` and `_predict_rows` next to it)
- Test: `tests/test_detector.py`

**Interfaces:**
- Produces: `Detector.detect_image(image: numpy.ndarray, frame: int = 0) -> pandas.DataFrame` with columns `Detector.DET_FIELDS` (`frame, res, x, y, w, h, conf, class`) and `detect_frames`'s rounding.

- [ ] **Step 1: Write the failing test** (append to `tests/test_detector.py`)

```python
def test_detect_image_matches_detect_frames(fake, synthetic_video):
    import cv2

    video, _ = synthetic_video
    d = det_mod.Detector(model=DetectorModel.YOLO26n, device="cpu", fast=False)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 3)
    ok, img = cap.read()
    cap.release()
    assert ok
    pd.testing.assert_frame_equal(
        d.detect_image(img, frame=3), d.detect_frames(str(video), [3], verbose=False)
    )
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_detector.py::test_detect_image_matches_detect_frames -v`
Expected: FAIL with `AttributeError: 'Detector' object has no attribute 'detect_image'`.

- [ ] **Step 3: Implement**

Add two methods to `Detector`, directly above `detect_frames`:

```python
    def _predict_rows(self, image: np.ndarray, frame: int) -> list[dict]:
        """Run the model on one BGR image and return per-box dicts (frame,res,x,y,x2,y2,conf,class)."""
        preds = self.model.predict(
            image,
            verbose=False,
            conf=self.conf,
            iou=self.nms,
            max_det=self.max_det,
            device=self.device,
            imgsz=self.imgsz,
            classes=self.classes,
            rect=self.rect,
            agnostic_nms=self.agnostic_nms,
            **self._precision_kwargs,
            embed=self.embed,
        )
        boxes = preds[0].boxes
        rows: list[dict] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
            confs = boxes.conf.cpu().numpy()  # (N,)
            clss = boxes.cls.cpu().numpy().astype(int)  # (N,)
            for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss, strict=True):
                rows.append({
                    "frame": frame,
                    "res": -1,
                    "x": float(x1),
                    "y": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": float(cf),
                    "class": int(c),
                })
        return rows

    def detect_image(self, image: np.ndarray, frame: int = 0) -> pd.DataFrame:
        """Run object detection on one image.

        Parameters
        ----------
        image : numpy.ndarray
            BGR image, as read by OpenCV.
        frame : int, optional
            Value written to the ``frame`` column. Default is 0.

        Returns
        -------
        pandas.DataFrame
            Columns ``frame, res, x, y, w, h, conf, class`` with ``detect_frames``'s rounding;
            empty, with those columns, when nothing is found.

        """
        return self._results_to_df(self._predict_rows(image, frame))
```

In `detect_frames`, replace everything from `preds = self.model.predict(` through the end of the `for (x1, y1, x2, y2), cf, c in zip(...)` loop with:

```python
            results.extend(self._predict_rows(frame, pos_frame))
```

- [ ] **Step 4: Run the detector tests**

Run: `.venv/bin/python -m pytest tests/test_detector.py -v`
Expected: all pass, including `test_detect_and_detect_frames_same_rounding` and the two `predict`-kwargs tests for `detect_frames`.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/detect/yolo/detector.py tests/test_detector.py
git commit -m "feat(detect): add Detector.detect_image for one-image detection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Shared reading and detection helpers; `ReClass` moves onto them

**Files:**
- Create: `tests/support/reclass_fakes.py`
- Create: `tests/test_reclass.py`
- Modify: `src/dnt/track/re_class.py`, `pyproject.toml` (drop the `re_class.py` E501 entry)

**Interfaces:**
- Consumes: `Detector.detect_image` (Task 3), `refine.crops.FrameReader`.
- Produces (module-private in `re_class.py`, used by Tasks 6–7):
  - `_DET_FIELDS: list[str]` (a copy of `Detector.DET_FIELDS`, taken at import time)
  - `_read_frames(reader, frames) -> Iterator[tuple[int, numpy.ndarray | None]]`: increasing order, unique frames, `None` for a frame that could not be read.
  - `_detect_frames(detector, reader, frames, *, verbose=False, desc="Detecting frames") -> tuple[dict[int, pandas.DataFrame], set[int]]`: detections per read frame, and the unread frames.
  - `_dets_for(by_frame: dict[int, DataFrame], frames) -> DataFrame`: the rows of the given frames (empty `_DET_FIELDS` table when none).
- Test support (`tests/support` is on `sys.path` via `tests/conftest.py`):
  - `FakeReader(path, bad=(), shape=(1080, 1920))` with `frames(wanted)`, context manager, `close()`, and `requests: list[list[int]]`.
  - `FakeDetector(script: dict[int, list[tuple]])` with `detect_image(image, frame=0)`, `detect_frames(video, frames, verbose=True)` and `seen: list[int]`; each script tuple is `(x, y, w, h, conf, cls)`.
  - `vehicle_rows(track, frames, box, **cols) -> pd.DataFrame` building `TRACK_FIELDS` rows.

- [ ] **Step 1: Create the test fakes** (`tests/support/reclass_fakes.py`)

```python
"""Fakes for ReClass and VehReClass tests: a frame reader and a scripted detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

DET_FIELDS = ["frame", "res", "x", "y", "w", "h", "conf", "class"]
TRACK_FIELDS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]


class FakeReader:
    """Mimic ``refine.crops.FrameReader``: a failed frame raises and ends that generator."""

    def __init__(self, path, bad=(), shape=(1080, 1920)):
        self.path = str(path)
        self.bad = set(bad)
        self.shape = shape
        self.requests: list[list[int]] = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True

    def frames(self, wanted):
        wanted = sorted({int(v) for v in wanted})
        self.requests.append(wanted)
        for f in wanted:
            if f in self.bad:
                raise ValueError(f"cannot read frame {f}")
            yield f, np.full((*self.shape, 3), f % 255, np.uint8)


class FakeDetector:
    """Return scripted boxes per frame; record every frame it is asked about."""

    DET_FIELDS = DET_FIELDS

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.seen: list[int] = []

    def _rows(self, frame):
        return [
            {"frame": frame, "res": -1, "x": x, "y": y, "w": w, "h": h, "conf": c, "class": k}
            for (x, y, w, h, c, k) in self.script.get(int(frame), [])
        ]

    def detect_image(self, image, frame=0):
        self.seen.append(int(frame))
        return pd.DataFrame(self._rows(frame), columns=DET_FIELDS)

    def detect_frames(self, input_video, frames, verbose=True):
        rows = [r for f in frames for r in self._rows(f)]
        self.seen.extend(int(f) for f in frames)
        return pd.DataFrame(rows, columns=DET_FIELDS)


def vehicle_rows(track, frames, box, **cols) -> pd.DataFrame:
    """Rows of one track at ``frames`` with box ``(x, y, w, h)``; ``cols`` override columns."""
    x, y, w, h = box
    df = pd.DataFrame({
        "frame": list(frames), "track": track, "x": x, "y": y, "w": w, "h": h,
        "score": -1.0, "cls": -1, "r3": -1, "r4": -1,
    })
    for k, v in cols.items():
        df[k] = v
    return df[TRACK_FIELDS + [c for c in cols if c not in TRACK_FIELDS]]
```

- [ ] **Step 2: Write the characterization test and the helper tests** (`tests/test_reclass.py`)

```python
"""ReClass results are pinned, and the shared reading helpers tolerate bad frames."""

import pandas as pd
from reclass_fakes import FakeDetector, FakeReader

from dnt.track import re_class


def _tracks():
    t1 = pd.DataFrame({"frame": [9, 1, 5], "track": 1, "x": 100, "y": 100, "w": 50, "h": 50})
    t2 = pd.DataFrame({
        "frame": [20, 21, 22, 23, 24], "track": 2, "x": 300, "y": 300,
        "w": [10, 20, 30, 40, 50], "h": 50,
    })
    t3 = pd.DataFrame({"frame": [40, 41], "track": 3, "x": 500, "y": 500, "w": [30, 60], "h": 60})
    return pd.concat([t1, t2, t3], ignore_index=True)


def test_reclass_results_are_pinned(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    det = FakeDetector({
        9: [(110, 110, 20, 20, 0.9, 1)],   # inside track 1 at frame 9 (the tie winner)
        1: [(110, 110, 20, 20, 0.9, 36)],  # frame 1 must NOT be sampled with num_frames=1
        20: [(305, 305, 5, 5, 0.9, 36)],   # track 2's smallest box: not among its top 3
        41: [(510, 510, 20, 20, 0.9, 36)],  # track 3's largest box
    })
    monkeypatch.setattr(re_class, "Detector", lambda **kw: det)
    monkeypatch.setattr(re_class, "FrameReader", FakeReader, raising=False)
    rc = re_class.ReClass(num_frames=1, threshold=0.75)
    out = rc.re_classify(_tracks(), str(video), verbose=False)
    assert out.values.tolist() == [[1, 1, 1.0], [2, 0, 0.0], [3, 36, 1.0]]


def test_read_frames_continues_after_a_bad_middle_frame():
    reader = FakeReader("v", bad={5})
    got = [(f, img is None) for f, img in re_class._read_frames(reader, [9, 5, 1, 9])]
    assert got == [(1, False), (5, True), (9, False)]
    assert reader.requests == [[1], [5], [9]]


def test_detect_frames_once_each_in_order():
    det = FakeDetector({1: [(0, 0, 5, 5, 0.5, 2)]})
    by_frame, unread = re_class._detect_frames(det, FakeReader("v", bad={4}), [7, 1, 4, 7])
    assert det.seen == [1, 7]
    assert unread == {4}
    assert sorted(by_frame) == [1, 7]
    assert len(by_frame[1]) == 1 and by_frame[7].empty


def test_dets_for_empty_has_columns():
    out = re_class._dets_for({}, [3])
    assert out.empty and list(out.columns) == re_class._DET_FIELDS
```

- [ ] **Step 3: Run the characterization test on the OLD code — it must pass**

Run: `.venv/bin/python -m pytest tests/test_reclass.py::test_reclass_results_are_pinned -v`
Expected: PASS (old `ReClass` calls `FakeDetector.detect_frames`). If it fails, stop: the pinned values are wrong, not the code. The helper tests fail with `AttributeError` (no `_read_frames`), which is expected.

- [ ] **Step 4: Implement the helpers and move `ReClass` onto them**

At the top of `src/dnt/track/re_class.py`:

```python
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..detect import Detector, DetectorModel
from ..engine.iob import iobs
from ..refine.crops import FrameReader

log = logging.getLogger(__name__)

#: Detection columns, copied at import so tests may replace ``Detector``.
_DET_FIELDS = list(Detector.DET_FIELDS)


def _read_frames(reader, frames: Iterable[int]) -> Iterator[tuple[int, np.ndarray | None]]:
    """Yield ``(frame, BGR image or None)`` for each unique frame, in increasing order.

    Each frame is requested on its own, so a frame the reader cannot decode yields ``None``
    and reading goes on: ``FrameReader`` seeks on the next request after a failure.
    """
    for f in sorted({int(v) for v in frames}):
        try:
            ((_, img),) = reader.frames([f])
        except ValueError:
            log.debug("cannot read frame %d of %s; it is left out", f, getattr(reader, "path", "?"))
            yield f, None
            continue
        yield f, img


def _detect_frames(
    detector, reader, frames: Iterable[int], *, verbose: bool = False, desc: str = "Detecting frames"
) -> tuple[dict[int, pd.DataFrame], set[int]]:
    """Detect each unique frame once, in increasing order.

    Returns the detections of every frame that was read, and the frames that were not.
    """
    wanted = sorted({int(v) for v in frames})
    by_frame: dict[int, pd.DataFrame] = {}
    unread: set[int] = set()
    pbar = tqdm(total=len(wanted), unit="frame", desc=desc) if verbose else None
    for f, img in _read_frames(reader, wanted):
        if img is None:
            unread.add(f)
        else:
            by_frame[f] = detector.detect_image(img, frame=f)
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()
    return by_frame, unread


def _dets_for(by_frame: dict[int, pd.DataFrame], frames: Iterable[int]) -> pd.DataFrame:
    """Return the detections of ``frames`` as one table (empty, with columns, when none)."""
    parts = [by_frame[int(f)] for f in frames if int(f) in by_frame]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=_DET_FIELDS)
    return pd.concat(parts, ignore_index=True)
```

Replace the body of `ReClass.re_classify` after the `track_ids` default (keep the two `raise` checks above it) with:

```python
        samples = {}
        for track_id in track_ids:
            target_track = tracks[tracks["track"] == track_id].copy()
            target_track["area"] = target_track["w"] * target_track["h"]
            target_track.sort_values(by="area", inplace=True, ascending=False)
            samples[track_id] = target_track.head(self.num_frames)

        frames = {int(f) for s in samples.values() for f in s["frame"]}
        with FrameReader(str(input_video)) as reader:
            by_frame, _ = _detect_frames(
                self.detector, reader, frames, verbose=verbose, desc="Re-classifying tracks"
            )

        results = []
        for track_id in track_ids:
            top_frames = samples[track_id]
            dets = _dets_for(by_frame, top_frames["frame"])

            matched = []
            for cls in self.match_class:
                match_dets = dets[dets["class"] == cls]
                hit, avg_score = self.match_mmv(top_frames, match_dets)
                if hit:
                    matched.append((cls, avg_score))

            if len(matched) > 0:
                cls, avg_score = max(matched, key=lambda x: x[1])
            else:
                cls = self.default_class
                avg_score = 0

            results.append([track_id, cls, round(avg_score, 2)])

        df = pd.DataFrame(results, columns=["track", "cls", "avg_score"])
        if out_file:
            df.to_csv(out_file, index=False)

        return df
```

The sample selection is the old code with `head(self.num_frames)` in place of the old conditional; `head` returns the whole table when it is shorter, so the result is identical. Behavior note for the changelog (Task 8): a video OpenCV cannot open now raises `ValueError` from `FrameReader`; before, `detect_frames` printed a message and every track silently got `default_class`.

In `pyproject.toml`, delete the line `"src/dnt/track/re_class.py" = ["E501"]`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_reclass.py tests/test_detector.py -v`
Expected: all pass. The characterization test passes unchanged on the new code.

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check src tests tools`
Expected: clean (the `re_class.py` E501 entry is gone, so any long line there must be wrapped).

```bash
git add tests/support/reclass_fakes.py tests/test_reclass.py src/dnt/track/re_class.py pyproject.toml
git commit -m "refactor(reclass): detect all sampled frames in one tolerant pass

ReClass now reads each frame once, in order, and keeps going past a frame
the video cannot decode. Its sampling and scoring are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Cascade helpers — size, spaced sampling, IoU, vote

**Files:**
- Modify: `src/dnt/track/re_class.py` (module-level helpers below `_dets_for`)
- Create: `tests/test_veh_reclass.py`

**Interfaces:**
- Produces (module-private, used by Task 7):
  - `CAR, BUS, TRUCK = 2, 5, 7`
  - `_size(rows: DataFrame, k: int) -> float`: the k-th largest `max(w, h)`; NaN with fewer than k rows.
  - `_spaced_sample(rows: DataFrame, n: int, min_gap: int) -> list[tuple[int, tuple[float, float, float, float]]]`: `(frame, (x, y, w, h))`, largest area first, ties to the earlier frame, frames at least `min_gap` apart and never repeated.
  - `_iou(box, boxes) -> numpy.ndarray`
  - `_Vote(NamedTuple)`: `cls: int | None`, `support: float`, `n: int`
  - `_vote(boxes: dict[int, tuple], by_frame: dict[int, DataFrame], allowed: tuple[int, ...], iou_threshold: float) -> _Vote`

- [ ] **Step 1: Write the failing tests** (`tests/test_veh_reclass.py`)

```python
"""VehReClass: helpers (Task 5), VLM layer (Task 6) and the class (Task 7)."""

import math

import numpy as np
import pandas as pd
import pytest
from reclass_fakes import DET_FIELDS, FakeDetector, FakeReader, vehicle_rows

from dnt.track import re_class
from dnt.track.re_class import BUS, CAR, TRUCK


def _sizes(values):
    return pd.DataFrame({"frame": range(len(values)), "w": values, "h": [1] * len(values)})


# ---- Task 5: size ----

@pytest.mark.parametrize("n_rows", [1, 2, 5, 10, 20])
def test_one_outlier_never_reaches_hv_size(n_rows):
    sizes = [100] * (n_rows - 1) + [1000]
    s = re_class._size(_sizes(sizes), 3)
    assert not s >= 500  # NaN for short tracks, 100 otherwise


def test_size_counts_k_rows_at_threshold():
    assert re_class._size(_sizes([500, 500, 500, 100]), 3) == 500
    assert math.isnan(re_class._size(_sizes([900, 900]), 3))


def test_size_uses_longer_side():
    rows = pd.DataFrame({"frame": [0, 1, 2], "w": [10, 10, 10], "h": [600, 700, 800]})
    assert re_class._size(rows, 3) == 600


# ---- Task 5: sampling ----

def _ramp(n):
    return pd.DataFrame({"frame": range(n), "x": 0, "y": 0, "w": range(1, n + 1), "h": 10})


def test_sample_largest_first_with_gap():
    assert [f for f, _ in re_class._spaced_sample(_ramp(20), 10, 5)] == [19, 14, 9, 4]


def test_sample_min_gap_zero_is_top_n():
    assert [f for f, _ in re_class._spaced_sample(_ramp(20), 3, 0)] == [19, 18, 17]


def test_sample_ties_go_to_earlier_frame():
    rows = pd.DataFrame({"frame": [9, 1, 5], "x": 0, "y": 0, "w": 5, "h": 5})
    assert re_class._spaced_sample(rows, 1, 0) == [(1, (0.0, 0.0, 5.0, 5.0))]


def test_sample_never_repeats_a_frame():
    rows = pd.DataFrame({"frame": [3, 3, 4], "x": 0, "y": 0, "w": [9, 8, 1], "h": 1})
    assert [f for f, _ in re_class._spaced_sample(rows, 5, 0)] == [3, 4]


def test_sample_short_and_empty():
    assert len(re_class._spaced_sample(_ramp(2), 10, 0)) == 2
    assert re_class._spaced_sample(_ramp(0), 10, 5) == []


# ---- Task 5: IoU and vote ----

def test_iou_basic_and_zero_area_is_zero():
    got = re_class._iou((0, 0, 10, 10), np.array([[0, 0, 10, 10], [5, 0, 10, 10], [0, 0, 0, 0]]))
    assert got.tolist() == pytest.approx([1.0, 50 / 150, 0.0])


def _dets(frame, *boxes):
    return pd.DataFrame(
        [{"frame": frame, "res": -1, "x": x, "y": y, "w": w, "h": h, "conf": c, "class": k}
         for x, y, w, h, c, k in boxes],
        columns=DET_FIELDS,
    )


BOX = (100.0, 100.0, 200.0, 100.0)


def test_vote_spec_example():
    boxes = {f: BOX for f in range(10)}
    by = {f: _dets(f, (100, 100, 200, 100, 0.8, BUS)) for f in range(4)}
    by |= {f: _dets(f, (100, 100, 200, 100, 0.6, TRUCK)) for f in (4, 5)}
    v = re_class._vote(boxes, by, (CAR, BUS, TRUCK), 0.5)
    assert (v.cls, v.support, v.n) == (BUS, pytest.approx(0.4), 10)


def test_vote_confidence_sum_beats_vote_count():
    # 4 bus votes at 0.3 (sum 1.2) lose to 3 truck votes at 0.9 (sum 2.7): a count-only
    # winner would pick bus
    boxes = {f: BOX for f in range(10)}
    by = {f: _dets(f, (100, 100, 200, 100, 0.3, BUS)) for f in range(4)}
    by |= {f: _dets(f, (100, 100, 200, 100, 0.9, TRUCK)) for f in (4, 5, 6)}
    v = re_class._vote(boxes, by, (CAR, BUS, TRUCK), 0.5)
    assert (v.cls, v.support, v.n) == (TRUCK, pytest.approx(0.3), 10)


def test_vote_equal_score_more_votes_wins():
    # truck: 2 x 0.45 = 0.9; bus: 1 x 0.9 = 0.9 (exactly equal in binary floating point).
    # Truck has more votes but the higher class ID, so this fails an implementation that
    # skips the vote-count tie break and goes straight to the lower class ID.
    by = {0: _dets(0, (100, 100, 200, 100, 0.45, TRUCK)),
          1: _dets(1, (100, 100, 200, 100, 0.45, TRUCK)),
          2: _dets(2, (100, 100, 200, 100, 0.9, BUS))}
    assert re_class._vote({f: BOX for f in range(3)}, by, (CAR, BUS, TRUCK), 0.5).cls == TRUCK


def test_vote_equal_score_and_votes_lower_class_wins():
    by = {0: _dets(0, (100, 100, 200, 100, 0.9, TRUCK)), 1: _dets(1, (100, 100, 200, 100, 0.9, BUS))}
    assert re_class._vote({0: BOX, 1: BOX}, by, (CAR, BUS, TRUCK), 0.5).cls == BUS
    by = {0: _dets(0, (100, 100, 200, 100, 0.9, BUS)), 1: _dets(1, (100, 100, 200, 100, 0.9, CAR))}
    assert re_class._vote({0: BOX, 1: BOX}, by, (CAR, BUS, TRUCK), 0.5).cls == CAR


def test_vote_one_stray_has_low_support():
    boxes = {f: BOX for f in range(10)}
    v = re_class._vote(boxes, {3: _dets(3, (100, 100, 200, 100, 0.99, BUS))}, (CAR, BUS, TRUCK), 0.5)
    assert v.cls == BUS and v.support == pytest.approx(0.1)


def test_vote_highest_conf_wins_frame_and_iou_gate():
    by = {0: _dets(0, (100, 100, 200, 100, 0.6, BUS), (100, 100, 200, 100, 0.8, TRUCK),
                   (0, 0, 20, 20, 0.99, CAR))}  # the car box does not overlap: ignored
    v = re_class._vote({0: BOX}, by, (CAR, BUS, TRUCK), 0.5)
    assert (v.cls, v.support) == (TRUCK, 1.0)


def test_vote_heavy_ignores_car():
    by = {0: _dets(0, (100, 100, 200, 100, 0.9, CAR))}
    v = re_class._vote({0: BOX}, by, (BUS, TRUCK), 0.5)
    assert v.cls is None and v.support == 0.0


def test_vote_no_frames():
    v = re_class._vote({}, {}, (CAR, BUS, TRUCK), 0.5)
    assert v.cls is None and v.n == 0 and math.isnan(v.support)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_veh_reclass.py -v`
Expected: FAIL with `ImportError: cannot import name 'BUS'`.

- [ ] **Step 3: Implement** (in `re_class.py`, below `_dets_for`; add `import math` and `from typing import NamedTuple` to the imports)

```python
CAR, BUS, TRUCK = 2, 5, 7


def _size(rows: pd.DataFrame, k: int) -> float:
    """Return the ``k``-th largest ``max(w, h)`` of ``rows``, or NaN with fewer than ``k`` rows."""
    if len(rows) < k:
        return math.nan
    sides = np.maximum(rows["w"].to_numpy(float), rows["h"].to_numpy(float))
    return float(np.sort(sides)[-k])


def _spaced_sample(
    rows: pd.DataFrame, n: int, min_gap: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Pick up to ``n`` rows, largest box first, with frames at least ``min_gap`` apart.

    Ties in area go to the earlier frame. A frame is never taken twice.
    """
    if rows.empty:
        return []
    order = rows.assign(_area=rows["w"] * rows["h"], _frame=rows["frame"].astype(int))
    order = order.sort_values(["_area", "_frame"], ascending=[False, True], kind="mergesort")
    taken: list[tuple[int, tuple[float, float, float, float]]] = []
    for f, x, y, w, h in order[["_frame", "x", "y", "w", "h"]].itertuples(index=False):
        if any(abs(f - t) < min_gap or f == t for t, _ in taken):
            continue
        taken.append((int(f), (float(x), float(y), float(w), float(h))))
        if len(taken) == n:
            break
    return taken


def _iou(box, boxes) -> np.ndarray:
    """IoU of one ``(x, y, w, h)`` box with each row of ``boxes``; 0 where the union is empty."""
    x, y, w, h = (float(v) for v in box)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    iw = np.clip(np.minimum(x + w, b[:, 0] + b[:, 2]) - np.maximum(x, b[:, 0]), 0, None)
    ih = np.clip(np.minimum(y + h, b[:, 1] + b[:, 3]) - np.maximum(y, b[:, 1]), 0, None)
    inter = iw * ih
    union = w * h + b[:, 2] * b[:, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class _Vote(NamedTuple):
    cls: int | None
    support: float
    n: int


def _vote(boxes, by_frame, allowed, iou_threshold) -> _Vote:
    """Vote for a class over the sampled frames (spec 5.2).

    Each frame's single vote goes to the allowed-class detection with the highest confidence
    among those overlapping the track box by at least ``iou_threshold``; it is weighted by that
    confidence. ``support`` is the winner's votes divided by the number of frames ``n``.
    """
    n = len(boxes)
    weight: dict[int, float] = {}
    count: dict[int, int] = {}
    for f, box in boxes.items():
        d = by_frame.get(int(f))
        if d is None or d.empty:
            continue
        d = d[d["class"].isin(allowed)]
        if d.empty:
            continue
        d = d[_iou(box, d[["x", "y", "w", "h"]].to_numpy(float)) >= iou_threshold]
        if d.empty:
            continue
        best = d.loc[d["conf"].astype(float).idxmax()]
        c = int(best["class"])
        weight[c] = weight.get(c, 0.0) + float(best["conf"])
        count[c] = count.get(c, 0) + 1
    if n == 0:
        return _Vote(None, math.nan, 0)
    if not count:
        return _Vote(None, 0.0, n)
    winner = min(count, key=lambda c: (-weight[c], -count[c], c))
    return _Vote(winner, count[winner] / n, n)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_veh_reclass.py -v`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check src/dnt/track/re_class.py tests/test_veh_reclass.py`

```bash
git add src/dnt/track/re_class.py tests/test_veh_reclass.py
git commit -m "feat(reclass): add size, spaced sampling and voting helpers for vehicles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: VLM layer — prompt, tile, preflight, ask

**Files:**
- Modify: `src/dnt/refine/vlm/prompts.py`, `src/dnt/track/re_class.py`
- Test: `tests/refine/test_vlm_cache_prompts.py`, `tests/test_veh_reclass.py`

**Interfaces:**
- Consumes: `vlm_problems` (Task 2), `_read_frames` (Task 4), `refine.crops.crop_box`, `refine.vlm` (`check_vlm_dependencies`, `resolve_api_key`, `missing_key_message`), `refine.vlm.runner` (`Question`, `VLMRunner`), `refine.vlm.cache.AnswerCache`.
- Produces:
  - `prompts.VEHICLE_CLASS = ["car", "bus", "truck", "unclear"]`, `prompts.HEAVY_CLASS = ["bus", "truck", "unclear"]`, `prompts.vehicle_class_prompt(options: list[str]) -> str`.
  - In `re_class.py`: `_tile(crops: list[numpy.ndarray]) -> bytes | None` (RGB crops in, JPEG out); `_prepare_vlm(vlm, api_key, *, check: bool) -> VLMConfig`; `_VLMResult(NamedTuple)`: `answer: str | None`, `confidence: float`, `error: str | None`; `_ask_vlm(cfg, backend_factory, input_video, requests: dict[TrackId, tuple[list[tuple[int, tuple]], bool]]) -> dict[TrackId, _VLMResult]` where each request is `(crop samples, heavy)` and the result keys are the **same objects** as the request keys (any hashable track ID: `int`, `float`, numpy scalar); `_tag_id(t) -> str` (display only).

- [ ] **Step 1: Write the failing prompt test** (append to `tests/refine/test_vlm_cache_prompts.py`)

```python
from dnt.refine.vlm.prompts import HEAVY_CLASS, VEHICLE_CLASS, vehicle_class_prompt


def test_vehicle_class_prompt_lists_options_and_reply_format():
    text = vehicle_class_prompt(VEHICLE_CLASS)
    assert "- car:" in text and "- bus:" in text and "- truck:" in text and "- unclear:" in text
    assert text.endswith('"reason": "<one short sentence>"}')
    assert "- car:" not in vehicle_class_prompt(HEAVY_CLASS)
```

- [ ] **Step 2: Implement the prompt** (`src/dnt/refine/vlm/prompts.py`)

Add after `SAME`:

```python
#: Vehicle class answers (``dnt.track.VehReClass``); heavy tracks are asked without ``car``.
VEHICLE_CLASS = ["car", "bus", "truck", "unclear"]
HEAVY_CLASS = ["bus", "truck", "unclear"]
```

Add to `_GLOSS`:

```python
    "car": "a passenger car, SUV or minivan",
    "bus": "a bus or coach that carries passengers",
    "truck": "a truck of any size, including pickup trucks, box trucks and semi-trailers",
    "unclear": "you cannot tell",
```

Extract the reply block that ends `build_prompt` into a helper and use it in both places (the text of `build_prompt` must stay byte-identical, since it is part of the VLM cache key):

```python
def _answer_lines(options: list[str]) -> list[str]:
    lines = ["", "Options:"]
    lines += [f"- {o}: {_GLOSS[o]}" for o in options]
    lines += [
        "",
        "Reply with only a JSON object, no other text:",
        '{"answer": "<one of the options>", "confidence": <number from 0 to 1>, '
        '"reason": "<one short sentence>"}',
    ]
    return lines


def vehicle_class_prompt(options: list[str]) -> str:
    """Return the vehicle-class question for ``options`` (deterministic text)."""
    body = (
        "The crops show one road vehicle at different moments of one track, left to right. "
        "Which kind of vehicle is it?"
    )
    return "\n".join([body, *_answer_lines(options)])
```

and end `build_prompt` with `return "\n".join([body, *_answer_lines(options)])` in place of its own `lines = [body, "", "Options:"] ...` block.

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_cache_prompts.py -v`
Expected: all pass, including the existing prompt tests.

- [ ] **Step 3: Write the failing VLM-layer tests** (append to `tests/test_veh_reclass.py`; the four imports go in the file's top import block)

```python
import json

from dnt.refine.config import VLMConfig
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import VLMRunner

# --- tests (below the existing Task 5 tests) ---


def _vlm(tmp_path, **kw):
    base = dict(backend="openai_compat", model="m", base_url="http://localhost:9/v1",
                cache_dir=str(tmp_path / "cache"))
    return VLMConfig(**(base | kw))


def _answer(ans, conf=0.9):
    return json.dumps({"answer": ans, "confidence": conf, "reason": "r"})


def test_tile_is_jpeg_and_none_without_crops():
    img = re_class._tile([np.zeros((40, 80, 3), np.uint8), np.zeros((20, 10, 3), np.uint8)])
    assert img[:2] == b"\xff\xd8"
    assert re_class._tile([]) is None


def test_prepare_vlm_resolves_endpoint_and_checks_it(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    seen = []
    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: seen.append(cfg.backend))
    vlm = VLMConfig(cache_dir=str(tmp_path), use="local", endpoints={
        "local": {"backend": "openai_compat", "model": "q", "base_url": "http://localhost:9/v1"}})
    cfg = re_class._prepare_vlm(vlm, None, check=True)
    assert (cfg.backend, cfg.model, seen) == ("openai_compat", "q", ["openai_compat"])
    assert vlm.use == "local"  # the caller's object is unchanged


@pytest.mark.parametrize("vlm", [
    VLMConfig(use="nope"),
    VLMConfig(backend="openai_compat", model="m", votes=0),
    VLMConfig(),  # resolves to backend none
])
def test_prepare_vlm_rejects(vlm):
    with pytest.raises(ValueError):
        re_class._prepare_vlm(vlm, None, check=False)


def test_prepare_vlm_missing_anthropic_key(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        re_class._prepare_vlm(VLMConfig(backend="anthropic", cache_dir=str(tmp_path)), None, check=True)


def _requests(*tracks, heavy=False):
    box = (100.0, 100.0, 300.0, 150.0)
    return {t: ([(10 * t + i * 5, box) for i in range(3)], heavy) for t in tracks}


def test_ask_vlm_answers_and_options(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    backend = FakeBackend({"vehclass:1": _answer("bus"), "vehclass:2": _answer("truck")})
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4",
                            _requests(1) | _requests(2, heavy=True))
    assert got[1] == re_class._VLMResult("bus", 0.9, None)
    assert got[2].answer == "truck"
    opts = {c["tag"]: c["options"] for c in backend.calls}
    assert opts == {"vehclass:1": ["car", "bus", "truck", "unclear"],
                    "vehclass:2": ["bus", "truck", "unclear"]}
    assert backend.closed == 1


def test_ask_vlm_keeps_original_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    backend = FakeBackend({"vehclass:1": _answer("bus"), "vehclass:2": _answer("truck")})
    reqs = {1.0: _requests(1)[1], 2: _requests(2)[2]}
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", reqs)
    assert set(got) == {1.0, 2} and got[1.0].answer == "bus" and got[2].answer == "truck"
    assert sorted(c["tag"] for c in backend.calls) == ["vehclass:1", "vehclass:2"]


def test_tag_id():
    assert [re_class._tag_id(v) for v in (1, 1.0, np.int64(3), np.float64(4.0), 2.5)] == [
        "1", "1", "3", "4", "2.5"]


def test_partial_crop_failure_still_asks(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad={15}))
    backend = FakeBackend({"*": _answer("bus")})
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", _requests(1))
    assert got[1].answer == "bus" and len(backend.calls) == 1


def test_track_without_usable_crop_is_not_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad={10, 15, 20}))
    backend = FakeBackend({"*": _answer("bus")})
    got = re_class._ask_vlm(_vlm(tmp_path), lambda cfg: backend, "v.mp4", _requests(1))
    assert got == {} and backend.calls == []


def test_ask_vlm_closes_runner_when_ask_many_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    closed = []
    monkeypatch.setattr(VLMRunner, "ask_many", lambda self, q: (_ for _ in ()).throw(RuntimeError("x")))
    real_close = VLMRunner.close
    monkeypatch.setattr(VLMRunner, "close", lambda self: (closed.append(1), real_close(self)))
    with pytest.raises(RuntimeError):
        re_class._ask_vlm(_vlm(tmp_path), lambda cfg: FakeBackend({}), "v.mp4", _requests(1))
    assert closed == [1]


def test_budget_is_per_call(monkeypatch, tmp_path):
    monkeypatch.setattr(re_class, "FrameReader", FakeReader)
    cfg = _vlm(tmp_path, max_calls=1, max_concurrency=1)
    script = {"*": _answer("bus")}
    first = re_class._ask_vlm(cfg, lambda c: FakeBackend(script), "v.mp4", _requests(1, 2))
    assert sorted(r.error is None for r in first.values()) == [False, True]
    second = re_class._ask_vlm(cfg, lambda c: FakeBackend(script), "v.mp4", _requests(1, 2))
    assert all(r.error is None for r in second.values())  # one cached, one fresh call
```

- [ ] **Step 4: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_veh_reclass.py -k "tile or prepare or ask_vlm or usable_crop or budget" -v`
Expected: FAIL with `AttributeError: module 'dnt.track.re_class' has no attribute '_tile'`.

- [ ] **Step 5: Implement** (in `re_class.py`, below `_vote`; add `import copy` and `import os` to the imports and `from ..refine.crops import FrameReader, crop_box`)

```python
_TILE_HEIGHT = 160
_TILE_GAP = 6
_JPEG_QUALITY = 85
_ANSWER_CLASS = {"car": CAR, "bus": BUS, "truck": TRUCK}


def _tile(crops: list[np.ndarray]) -> bytes | None:
    """Scale RGB crops to one height, place them side by side, and return JPEG bytes."""
    import cv2

    parts = []
    for c in crops:
        h, w = c.shape[:2]
        width = max(1, round(w * _TILE_HEIGHT / h))
        parts.append(cv2.resize(c, (width, _TILE_HEIGHT), interpolation=cv2.INTER_AREA))
    if not parts:
        return None
    width = sum(p.shape[1] for p in parts) + _TILE_GAP * (len(parts) + 1)
    canvas = np.full((_TILE_HEIGHT + 2 * _TILE_GAP, width, 3), 128, np.uint8)
    x = _TILE_GAP
    for p in parts:
        canvas[_TILE_GAP : _TILE_GAP + _TILE_HEIGHT, x : x + p.shape[1]] = p
        x += p.shape[1] + _TILE_GAP
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _prepare_vlm(vlm, api_key: str | None, *, check: bool):
    """Validate and resolve a copy of ``vlm`` (spec 3); with ``check``, preflight it.

    Raises ``ValueError`` for invalid settings, a config resolving to backend ``none``, or a
    missing key; ``ImportError`` when the backend's package is missing.
    """
    from ..refine.config import vlm_problems

    problems, cfg = vlm_problems(copy.deepcopy(vlm))
    if problems:
        raise ValueError("invalid vlm config: " + "; ".join(problems))
    if cfg.backend == "none":
        raise ValueError("vlm resolves to backend 'none'; pass vlm=None to turn layer 3 off")
    if check:
        from ..refine import vlm as vlm_mod

        vlm_mod.check_vlm_dependencies(cfg)
        _check_vlm_key(cfg, api_key)
    return cfg


def _check_vlm_key(cfg, api_key: str | None) -> None:
    """Raise ``ValueError`` when the backend needs a key and none is found (no network)."""
    from ..refine.vlm import missing_key_message, resolve_api_key

    if cfg.backend == "anthropic":
        if resolve_api_key(cfg, "ANTHROPIC_API_KEY", api_key) is None:
            raise ValueError(missing_key_message("anthropic", cfg.api_key_env or "ANTHROPIC_API_KEY"))
    elif cfg.backend == "openai_compat":
        key = resolve_api_key(cfg, "OPENAI_API_KEY", api_key)
        if key is None and not (cfg.base_url or os.environ.get("OPENAI_BASE_URL")):
            raise ValueError(
                missing_key_message("openai_compat", cfg.api_key_env or "OPENAI_API_KEY")
                + "; a local server needs vlm.base_url instead"
            )


class _VLMResult(NamedTuple):
    answer: str | None
    confidence: float
    error: str | None


def _ask_vlm(cfg, backend_factory, input_video: str, requests) -> dict[int, _VLMResult]:
    """Ask one question per track about a tile of its crops (spec 5.3).

    ``requests`` maps a track ID to ``(samples, heavy)``. A track with no usable crop is left
    out. The runner is built only when there is a question, and it is always closed.
    """
    from ..refine.vlm.cache import AnswerCache
    from ..refine.vlm.prompts import HEAVY_CLASS, VEHICLE_CLASS, vehicle_class_prompt
    from ..refine.vlm.runner import Question, VLMRunner

    want: dict[int, list[tuple[int, tuple]]] = {}
    for t, (samples, _) in requests.items():
        for f, box in samples:
            want.setdefault(int(f), []).append((t, box))
    crops: dict[int, list[tuple[int, np.ndarray]]] = {t: [] for t in requests}
    with FrameReader(str(input_video)) as reader:
        for f, img in _read_frames(reader, want):
            if img is None:
                continue
            for t, box in want[f]:
                c = crop_box(img, box)
                if c is not None:
                    crops[t].append((f, c))
    questions, asked = [], []
    for t, (_, heavy) in requests.items():
        image = _tile([c for _, c in sorted(crops[t], key=lambda fc: fc[0])])
        if image is None:
            log.debug("track %s has no usable crop for the VLM", t)
            continue
        options = list(HEAVY_CLASS if heavy else VEHICLE_CLASS)
        questions.append(
            Question(tag=f"vehclass:{_tag_id(t)}", image=image,
                     prompt=vehicle_class_prompt(options), options=options)
        )
        asked.append(t)  # the original ID, kept as is; the tag is only diagnostic text
    if not questions:
        return {}
    runner = VLMRunner(cfg, backend_factory(cfg), AnswerCache(cfg.cache_dir))
    try:
        verdicts = runner.ask_many(questions)
    finally:
        runner.close()
    out = {}
    for t, v in zip(asked, verdicts, strict=True):
        if v.error:
            log.warning("track %s: no VLM answer (%s); it falls back", t, v.error)
        out[t] = _VLMResult(v.answer, float(v.confidence), v.error)
    return out


def _tag_id(t) -> str:
    """Display a track ID for a question tag: an integral float such as ``1.0`` shows as ``1``."""
    if isinstance(t, numbers.Real) and not isinstance(t, bool) and float(t).is_integer():
        return str(int(t))
    return str(t)
```

The result is keyed by the **original** track ID (the key of `requests`), never by a value parsed back out of the tag, so a float `track` column such as `1.0` works. Add `import numbers` to the module imports in this task (Task 7 uses it too).

Check the field names of `Question` in `src/dnt/refine/vlm/runner.py` (`tag, image, prompt, options`) before running; they are the ones used above.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_veh_reclass.py tests/refine/test_vlm_cache_prompts.py tests/refine/test_vlm_runner.py -v`
Expected: all pass. `test_budget_is_per_call` is deterministic: `VLMRunner._run_batch` admits questions in list order before sending anything, so with `max_calls=1` the first question is admitted and the second gets a `budget` verdict; in the second call the first is a cache hit (free) and the second fits the fresh budget.

- [ ] **Step 7: Lint and commit**

Run: `.venv/bin/ruff check src tests`

```bash
git add src/dnt/refine/vlm/prompts.py src/dnt/track/re_class.py tests/refine/test_vlm_cache_prompts.py tests/test_veh_reclass.py
git commit -m "feat(reclass): add the VLM layer for vehicle classes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The `VehReClass` class

**Files:**
- Modify: `src/dnt/track/re_class.py` (class at the end of the module)
- Test: `tests/test_veh_reclass.py`

**Interfaces:**
- Consumes: everything from Tasks 1 and 4–6; `shared.files.read_track`, `shared.files.write_track`; `Tracker.TRACK_FIELDS` (import as `from .tracker import Tracker`).
- Produces: `VehReClass` per spec §3 with keyword-only extras `vlm_api_key: str | None = None` and `vlm_backend_factory: Callable[[VLMConfig], backend] | None = None` (the same two hooks `TrackRefiner` has; the factory skips the dependency and key preflight, as in `refine`). Attributes: `CAR`, `BUS`, `TRUCK`, `decisions: DataFrame` with columns `track, cls, layer, heavy, size, n, support, vlm_answer, vlm_conf`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_veh_reclass.py`; add `from dnt.shared.files import read_track` to the top import block and `VehReClass` to the existing `from dnt.track.re_class import ...` line)

```python
COLS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"")
    return str(p)


def _vr(monkeypatch, script=None, bad=(), **kw):
    det = FakeDetector(script)
    monkeypatch.setattr(re_class, "Detector", lambda **k: det)
    monkeypatch.setattr(re_class, "FrameReader", lambda p: FakeReader(p, bad=bad))
    return VehReClass(**kw), det


def _same(f, cls, conf=0.9, box=(100, 100, 200, 100)):
    return {f: [(*box, conf, cls)]}


def _script(frames, cls, conf=0.9, box=(100, 100, 200, 100)):
    out = {}
    for f in frames:
        out |= _same(f, cls, conf, box)
    return out


@pytest.mark.parametrize("kw", [
    {"vlm_crops": 0}, {"num_frames": 10.0}, {"hv_min_count": True}, {"hv_size": float("inf")},
    {"min_gap": -1}, {"iou_threshold": 0}, {"min_support": 1.5}, {"hv_size": 0},
])
def test_constructor_rejects(monkeypatch, kw):
    monkeypatch.setattr(re_class, "Detector", lambda **k: FakeDetector())
    with pytest.raises(ValueError):
        VehReClass(**kw)


def test_vlm_key_missing_fails_before_detector(monkeypatch, tmp_path):
    import dnt.refine.vlm as vlm_mod

    built = []
    monkeypatch.setattr(re_class, "Detector", lambda **k: built.append(1))
    monkeypatch.setattr(vlm_mod, "check_vlm_dependencies", lambda cfg: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        VehReClass(vlm=VLMConfig(backend="anthropic", cache_dir=str(tmp_path)))
    assert built == []


def test_redetect_writes_cls_on_every_row(monkeypatch, video):
    tracks = pd.concat([
        vehicle_rows(1, range(0, 50), (100, 100, 200, 100)),
        vehicle_rows(2, range(0, 50), (900, 500, 60, 40)),
    ], ignore_index=True)
    tracks.loc[tracks.index[:3], "r3"] = 1  # filled rows still get the class
    vr, _ = _vr(monkeypatch, _script(range(50), BUS))
    out = vr.re_classify(tracks, video, verbose=False)
    assert out[out.track == 1].cls.unique().tolist() == [BUS]
    assert out[out.track == 2].cls.unique().tolist() == [CAR]  # nothing matched: fallback
    assert list(out.columns) == COLS
    pd.testing.assert_frame_equal(out.drop(columns="cls"), tracks.drop(columns="cls"))
    d = vr.decisions.set_index("track")
    assert d.loc[1, "layer"] == "redetect" and d.loc[2, "layer"] == "fallback"
    assert list(vr.decisions.columns) == [
        "track", "cls", "layer", "heavy", "size", "n", "support", "vlm_answer", "vlm_conf"]


def test_one_stray_detection_does_not_decide(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _same(5, BUS))  # frame 5 is sampled (frames 0, 5, ..., 45)
    out = vr.re_classify(vehicle_rows(1, range(100), (100, 100, 200, 100)), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]


def _boundary_script():
    # samples are frames 0, 5, ..., 45 (10 rows, equal area, min_gap 5): 4 bus votes at 0.3,
    # 3 truck votes at 0.9, 3 frames with nothing
    return (_script((0, 5, 10, 15), BUS, conf=0.3) | _script((20, 25, 30), TRUCK, conf=0.9))


def test_support_exactly_at_min_support_decides(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _boundary_script())
    out = vr.re_classify(vehicle_rows(1, range(0, 50, 5), (100, 100, 200, 100)), video,
                         verbose=False)
    d = vr.decisions.iloc[0]
    assert out.cls.unique().tolist() == [TRUCK]
    assert d.layer == "redetect" and d.n == 10 and d.support == 0.3


def test_support_just_below_min_support_goes_on(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _boundary_script(), min_support=0.31)
    out = vr.re_classify(vehicle_rows(1, range(0, 50, 5), (100, 100, 200, 100)), video,
                         verbose=False)
    assert out.cls.unique().tolist() == [CAR]
    assert vr.decisions.layer.iloc[0] == "fallback"


def test_heavy_fallback_is_truck_and_car_not_allowed(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(100), CAR, box=(100, 100, 600, 300)), hv_size=500)
    out = vr.re_classify(vehicle_rows(1, range(100), (100, 100, 600, 300)), video, verbose=False)
    assert out.cls.unique().tolist() == [TRUCK]
    assert bool(vr.decisions.heavy.iloc[0])


def test_named_interp_and_headerless_file_agree(monkeypatch, video, tmp_path):
    real = vehicle_rows(1, range(0, 20), (100, 100, 200, 100), interp=0)
    filled = vehicle_rows(1, range(20, 30), (100, 100, 800, 400), interp=1)
    named = pd.concat([real, filled], ignore_index=True).drop(columns="r3")
    named = named[["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]]
    path = tmp_path / "t.txt"
    named.to_csv(path, header=False, index=False)
    script = _script(range(20), CAR) | _script(range(20, 30), BUS, box=(100, 100, 800, 400))
    vr, _ = _vr(monkeypatch, script, hv_size=500)
    a = vr.re_classify(named, video, verbose=False)
    da = vr.decisions.copy()
    b = vr.re_classify(str(path), video, verbose=False)
    assert a.cls.tolist() == b.cls.tolist() == [CAR] * 30
    pd.testing.assert_frame_equal(da, vr.decisions)
    assert list(a.columns) == list(named.columns) and list(b.columns) == COLS


def test_all_filled_track_falls_back(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(10), BUS))
    out = vr.re_classify(vehicle_rows(1, range(10), (100, 100, 200, 100), r3=1), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]
    row = vr.decisions.iloc[0]
    assert row.layer == "fallback" and not row.heavy and row.n == 0


def test_all_frames_unreadable_falls_back(monkeypatch, video):
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), bad=set(range(10)))
    out = vr.re_classify(vehicle_rows(1, range(10), (100, 100, 200, 100)), video, verbose=False)
    assert out.cls.unique().tolist() == [CAR]
    assert vr.decisions.n.iloc[0] == 0 and math.isnan(vr.decisions.support.iloc[0])


def test_unread_frame_leaves_only_tracks_that_sampled_it(monkeypatch, video):
    tracks = pd.concat([
        vehicle_rows(1, [0, 5, 10], (100, 100, 200, 100)),
        vehicle_rows(2, [5, 20], (400, 100, 200, 100)),
        vehicle_rows(3, [30, 40], (700, 100, 200, 100)),
    ], ignore_index=True)
    vr, det = _vr(monkeypatch, bad={5})
    vr.re_classify(tracks, video, verbose=False)
    assert vr.decisions.set_index("track").n.to_dict() == {1: 2, 2: 1, 3: 2}
    assert det.seen == [0, 10, 20, 30, 40]


def test_float_frame_column_matches_int(monkeypatch, video):
    tracks = vehicle_rows(1, range(30), (100, 100, 200, 100))
    vr, _ = _vr(monkeypatch, _script(range(30), BUS))
    a = vr.re_classify(tracks, video, verbose=False)
    b = vr.re_classify(tracks.astype({"frame": float}), video, verbose=False)
    assert a.cls.tolist() == b.cls.tolist() == [BUS] * 30


def test_track_ids_subset_and_unknown(monkeypatch, video, caplog):
    tracks = pd.concat([vehicle_rows(1, range(10), (100, 100, 200, 100)),
                        vehicle_rows(2, range(10), (100, 100, 200, 100))], ignore_index=True)
    vr, _ = _vr(monkeypatch, _script(range(10), BUS))
    out = vr.re_classify(tracks, video, track_ids=[2, 99], verbose=False)
    assert out[out.track == 1].cls.unique().tolist() == [-1]
    assert out[out.track == 2].cls.unique().tolist() == [BUS]
    assert "1 track ID" in caplog.text


def test_track_ids_numpy_array(monkeypatch, video, caplog):
    tracks = vehicle_rows(1, range(10), (100, 100, 200, 100))
    vr, _ = _vr(monkeypatch, _script(range(10), BUS))
    out = vr.re_classify(tracks, video, track_ids=np.array([1, 7]), verbose=False)
    assert out.cls.unique().tolist() == [BUS] and "1 track ID" in caplog.text


def test_empty_inputs(monkeypatch, video, tmp_path):
    vr, _ = _vr(monkeypatch)
    empty = tmp_path / "e.txt"
    empty.write_bytes(b"")
    for src in (pd.DataFrame(columns=COLS), str(empty)):
        out_file = tmp_path / "o.txt"
        out = vr.re_classify(src, video, out_file=str(out_file), verbose=False)
        assert out.empty and out_file.stat().st_size == 0 and vr.decisions.empty
    with pytest.raises(FileNotFoundError):
        vr.re_classify(pd.DataFrame(columns=COLS), str(tmp_path / "missing.mp4"))


def test_out_file_round_trip(monkeypatch, video, tmp_path):
    tracks = vehicle_rows(1, range(10), (100, 100, 200, 100))
    vr, _ = _vr(monkeypatch, _script(range(10), BUS))
    out = vr.re_classify(tracks, video, out_file=str(tmp_path / "o.txt"), verbose=False)
    back = read_track(tmp_path / "o.txt")
    assert back[7].tolist() == out.cls.tolist()


def test_vlm_decides_undecided_track(monkeypatch, video, tmp_path):
    backend = FakeBackend({"vehclass:1": _answer("bus", 0.8), "vehclass:2": _answer("bus", 0.5)})
    tracks = pd.concat([vehicle_rows(1, range(30), (100, 100, 200, 100)),
                        vehicle_rows(2, range(30), (400, 100, 200, 100))], ignore_index=True)
    vr, _ = _vr(monkeypatch, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = vr.re_classify(tracks, video, verbose=False)
    d = vr.decisions.set_index("track")
    assert out[out.track == 1].cls.unique().tolist() == [BUS] and d.loc[1, "layer"] == "vlm"
    assert out[out.track == 2].cls.unique().tolist() == [CAR] and d.loc[2, "layer"] == "fallback"
    assert d.loc[2, "vlm_answer"] == "bus" and d.loc[2, "vlm_conf"] == 0.5  # below min_conf 0.7


@pytest.mark.parametrize("dtype", [int, float])
def test_vlm_with_int_or_float_track_ids(monkeypatch, video, tmp_path, dtype):
    backend = FakeBackend({"vehclass:1": _answer("bus", 0.8)})
    tracks = vehicle_rows(1, range(30), (100, 100, 200, 100)).astype({"track": dtype})
    vr, _ = _vr(monkeypatch, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend)
    out = vr.re_classify(tracks, video, verbose=False)
    assert out.cls.unique().tolist() == [BUS]
    assert vr.decisions.layer.iloc[0] == "vlm"
    pd.testing.assert_series_equal(out["track"], tracks["track"])  # IDs and dtype unchanged
    assert [c["tag"] for c in backend.calls] == ["vehclass:1"]


def test_vlm_unclear_and_error_fall_back(monkeypatch, video, tmp_path):
    backend = FakeBackend({"vehclass:1": _answer("unclear"), "vehclass:2": RuntimeError("boom")})
    tracks = pd.concat([vehicle_rows(1, range(30), (100, 100, 200, 100)),
                        vehicle_rows(2, range(30), (400, 100, 600, 300))], ignore_index=True)
    vr, _ = _vr(monkeypatch, vlm=_vlm(tmp_path), vlm_backend_factory=lambda c: backend,
                hv_size=250)
    out = vr.re_classify(tracks, video, verbose=False)
    assert out[out.track == 1].cls.unique().tolist() == [CAR]
    assert out[out.track == 2].cls.unique().tolist() == [TRUCK]


def test_no_vlm_runner_when_nothing_pending(monkeypatch, video, tmp_path):
    built = []
    vr, _ = _vr(monkeypatch, _script(range(10), BUS), vlm=_vlm(tmp_path),
                vlm_backend_factory=lambda c: built.append(1))
    vr.re_classify(vehicle_rows(1, range(10), (100, 100, 200, 100)), video, verbose=False)
    assert built == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_veh_reclass.py -k "not tile and not prepare and not ask_vlm and not usable_crop and not budget" -v`
Expected: FAIL with `ImportError: cannot import name 'VehReClass'`.

- [ ] **Step 3: Implement** (end of `re_class.py`; add imports `from collections.abc import Callable`, `from ..refine.io import filled_mask`, `from ..shared.files import read_track, write_track`, `from .tracker import Tracker`)

```python
_DECISION_COLUMNS = ["track", "cls", "layer", "heavy", "size", "n", "support", "vlm_answer", "vlm_conf"]
_TRACK_COLUMNS = ("frame", "track", "x", "y", "w", "h", "cls")


def _setting_problems(hv_size, hv_min_count, num_frames, min_gap, iou_threshold, min_support,
                      vlm_crops) -> list[str]:
    def is_int(v):
        return isinstance(v, numbers.Integral) and not isinstance(v, bool)

    def is_num(v):
        return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)

    p = []
    if hv_size is not None and not (is_num(hv_size) and hv_size > 0):
        p.append("hv_size must be None or a finite number > 0")
    for name, v in (("hv_min_count", hv_min_count), ("num_frames", num_frames),
                    ("vlm_crops", vlm_crops)):
        if not (is_int(v) and v >= 1):
            p.append(f"{name} must be an integer >= 1")
    if not (is_int(min_gap) and min_gap >= 0):
        p.append("min_gap must be an integer >= 0")
    for name, v in (("iou_threshold", iou_threshold), ("min_support", min_support)):
        if not (is_num(v) and 0 < v <= 1):
            p.append(f"{name} must be a number in (0, 1]")
    return p


def _read_tracks(tracks) -> pd.DataFrame:
    """Return a copy of a track table, or read a headerless track file into one."""
    if isinstance(tracks, pd.DataFrame):
        if tracks.empty:
            return tracks.copy()
        missing = [c for c in _TRACK_COLUMNS if c not in tracks.columns]
        if missing:
            raise ValueError(f"tracks lacks columns {missing}")
        return tracks.copy()
    try:
        df = read_track(tracks)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=Tracker.TRACK_FIELDS)
    if df.shape[1] != len(Tracker.TRACK_FIELDS):
        raise ValueError(f"{tracks}: expected {len(Tracker.TRACK_FIELDS)} columns, got {df.shape[1]}")
    df.columns = Tracker.TRACK_FIELDS
    return df


class VehReClass:
    """Classify vehicle tracks as car, bus or truck with a size, re-detection and VLM cascade.

    Each track goes through the layers in order and stops at the first that decides (see the
    design spec ``2026-10-15-vehicle-reclass-design.md``):

    1. size: at least ``hv_min_count`` real rows with ``max(w, h) >= hv_size`` flag the track
       heavy, which allows only bus and truck from then on;
    2. re-detection: spaced, large frames are detected once, and each frame votes for the
       most confident allowed class overlapping the track box;
    3. VLM: a tile of crops is shown to the configured model;
    4. fallback: truck for a heavy track, car otherwise.

    Parameters
    ----------
    hv_size : float or None
        Heavy-vehicle size in pixels; None turns layer 1 off. Default None.
    hv_min_count : int
        Real rows that must reach ``hv_size``. Default 3.
    num_frames : int
        Frames sampled per track for re-detection. Default 10.
    min_gap : int
        Minimum distance in frames between samples (layers 2 and 3). Default 5.
    iou_threshold : float
        IoU a re-detection needs with the track box, in (0, 1]. Default 0.5.
    min_support : float
        Share of sampled frames the winning class needs, in (0, 1]. Default 0.3.
    vlm : dnt.refine.config.VLMConfig or None
        Layer 3 settings; None turns layer 3 off. Its ``min_conf`` is the answer threshold.
    vlm_crops : int
        Crops tiled into the one image per track. Default 3.
    model : DetectorModel or str
        Re-detection model. Default RT-DETR-x.
    weights : str or None
        Custom weights passed to ``Detector``.
    device : str
        Detector device. Default ``"auto"``.
    vlm_api_key : str or None
        API key; takes precedence over ``vlm.api_key_file`` and the environment.
    vlm_backend_factory : callable or None
        Builds the backend from the resolved ``VLMConfig`` (tests); skips the preflight.

    Raises
    ------
    ValueError
        For a broken setting, an invalid VLM config, one resolving to backend ``none``, or a
        missing API key.
    ImportError
        When the VLM backend's package is not installed.

    """

    CAR, BUS, TRUCK = CAR, BUS, TRUCK

    def __init__(
        self,
        hv_size: float | None = None,
        hv_min_count: int = 3,
        num_frames: int = 10,
        min_gap: int = 5,
        iou_threshold: float = 0.5,
        min_support: float = 0.3,
        vlm=None,
        vlm_crops: int = 3,
        model: DetectorModel | str = DetectorModel.RTDETRx,
        weights: str | None = None,
        device: str = "auto",
        *,
        vlm_api_key: str | None = None,
        vlm_backend_factory: Callable | None = None,
    ) -> None:
        """Validate the settings, prepare the VLM, then build the detector."""
        problems = _setting_problems(hv_size, hv_min_count, num_frames, min_gap, iou_threshold,
                                     min_support, vlm_crops)
        if problems:
            raise ValueError("invalid VehReClass settings: " + "; ".join(problems))
        self.hv_size = hv_size
        self.hv_min_count = hv_min_count
        self.num_frames = num_frames
        self.min_gap = min_gap
        self.iou_threshold = iou_threshold
        self.min_support = min_support
        self.vlm_crops = vlm_crops
        self._vlm_api_key = vlm_api_key
        self._vlm_backend_factory = vlm_backend_factory
        self.vlm = (
            None if vlm is None
            else _prepare_vlm(vlm, vlm_api_key, check=vlm_backend_factory is None)
        )
        self.detector = Detector(model=model, weights=weights, device=device)
        self.decisions = pd.DataFrame(columns=_DECISION_COLUMNS)

    def _backend_factory(self):
        if self._vlm_backend_factory is not None:
            return self._vlm_backend_factory
        from ..refine.vlm import make_backend

        return lambda cfg: make_backend(cfg, api_key=self._vlm_api_key)

    @staticmethod
    def _select(table: pd.DataFrame, track_ids) -> list:
        present = pd.unique(table["track"]).tolist()
        if track_ids is None:
            return present
        known = set(present)
        wanted = list(track_ids)
        ids = list(dict.fromkeys(t for t in wanted if t in known))
        unknown = sum(1 for t in wanted if t not in known)
        if unknown:
            log.warning("%d track ID(s) in track_ids are not in the tracks; ignored", unknown)
        return ids

    def re_classify(
        self,
        tracks: pd.DataFrame | str,
        input_video: str,
        track_ids: list | None = None,
        out_file: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Classify vehicle tracks and write the class into ``cls``.

        Parameters
        ----------
        tracks : pandas.DataFrame or str
            A track table (at least ``frame, track, x, y, w, h, cls``) or a headerless track
            file. Filled rows (``r3 == 1`` or ``interp == 1``) are never used as evidence.
        input_video : str
            The source video.
        track_ids : list or None
            Tracks to classify; None classifies all. Unknown IDs are ignored with a warning.
        out_file : str or None
            Where to write the result, headerless, in the input's column layout.
        verbose : bool
            Show a progress bar while detecting.

        Returns
        -------
        pandas.DataFrame
            The input table with ``cls`` set on every row of each classified track; the
            per-track reasons are in ``self.decisions``.

        Raises
        ------
        FileNotFoundError
            If ``input_video`` does not exist (checked first, even for empty input).
        ValueError
            If the video cannot be opened or the track input is malformed.

        """
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Video file not found: {input_video}")
        table = _read_tracks(tracks)
        self.decisions = pd.DataFrame(columns=_DECISION_COLUMNS)
        if table.empty:
            if out_file:
                write_track(table, out_file)
            return table

        ids = self._select(table, track_ids)
        real = table.loc[~filled_mask(table)]
        groups = dict(tuple(real.groupby("track", sort=False)))
        none = real.iloc[0:0]
        rows: dict = {}
        samples: dict = {}
        for t in ids:
            g = groups.get(t, none)
            size = _size(g, self.hv_min_count)
            heavy = self.hv_size is not None and size >= self.hv_size
            rows[t] = {"track": t, "cls": None, "layer": None, "heavy": bool(heavy), "size": size,
                       "n": 0, "support": math.nan, "vlm_answer": None, "vlm_conf": math.nan}
            samples[t] = _spaced_sample(g, self.num_frames, self.min_gap)

        frames = {f for s in samples.values() for f, _ in s}
        with FrameReader(str(input_video)) as reader:
            by_frame, unread = _detect_frames(self.detector, reader, frames, verbose=verbose,
                                              desc="Re-classifying vehicles")
        for t in ids:
            r = rows[t]
            allowed = (BUS, TRUCK) if r["heavy"] else (CAR, BUS, TRUCK)
            boxes = {f: b for f, b in samples[t] if f not in unread}
            v = _vote(boxes, by_frame, allowed, self.iou_threshold)
            r["n"], r["support"] = v.n, v.support
            if v.cls is not None and v.support >= self.min_support:
                r["cls"], r["layer"] = v.cls, "redetect"

        pending = {
            t: (_spaced_sample(groups[t], self.vlm_crops, self.min_gap), rows[t]["heavy"])
            for t in ids if rows[t]["layer"] is None and t in groups
        }
        if self.vlm is not None and pending:
            answers = _ask_vlm(self.vlm, self._backend_factory(), str(input_video), pending)
            for t, a in answers.items():
                r = rows[t]
                r["vlm_answer"], r["vlm_conf"] = a.answer, a.confidence
                ok = (a.error is None and a.answer in _ANSWER_CLASS
                      and math.isfinite(a.confidence) and a.confidence >= self.vlm.min_conf)
                if ok:
                    r["cls"], r["layer"] = _ANSWER_CLASS[a.answer], "vlm"

        for r in rows.values():
            if r["layer"] is None:
                r["cls"], r["layer"] = (TRUCK if r["heavy"] else CAR), "fallback"

        out = table.copy()
        cls_map = {t: r["cls"] for t, r in rows.items()}
        hit = out["track"].isin(list(cls_map))
        out.loc[hit, "cls"] = out.loc[hit, "track"].map(cls_map).astype(int)
        if out_file:
            write_track(out, out_file)
        self.decisions = pd.DataFrame(list(rows.values()), columns=_DECISION_COLUMNS)
        return out
```

If importing `Tracker` from `.tracker` creates an import cycle at test time, replace `Tracker.TRACK_FIELDS` with `from ..refine.io import TRACK_COLUMNS` (the same ten names) and say so in the commit message.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_veh_reclass.py tests/test_reclass.py -v`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/track/re_class.py`
Fix formatting with `.venv/bin/ruff format src/dnt/track/re_class.py` if needed.

```bash
git add src/dnt/track/re_class.py tests/test_veh_reclass.py
git commit -m "feat(reclass): add VehReClass, a size/re-detect/VLM cascade for vehicle classes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Export, docs and the full suite

**Files:**
- Modify: `src/dnt/track/__init__.py`, `src/dnt/track/re_class.py` (`ReClass` docstring fixes), `CLAUDE.md`, `docs/changelog.md`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write the failing import test** (append to `tests/test_imports.py`)

```python
def test_track_exports_both_reclassifiers():
    from dnt.track import ReClass, VehReClass

    assert ReClass is not None and VehReClass.TRUCK == 7
```

Run: `.venv/bin/python -m pytest tests/test_imports.py::test_track_exports_both_reclassifiers -v`
Expected: FAIL with `ImportError: cannot import name 'VehReClass'`.

- [ ] **Step 2: Update `src/dnt/track/__init__.py`**

Replace the `try:` / `except ModuleNotFoundError:` block with:

```python
from .re_class import ReClass as ReClass
from .re_class import VehReClass as VehReClass
```

and add `"VehReClass",` to `__all__` right after `"Tracker",`.

- [ ] **Step 3: Fix the `ReClass` docstring (review issue 8)**

In `ReClass.re_classify`'s docstring:
- the `avg_score` return line becomes: `Mean IoB over the frames whose best overlap reached the threshold, rounded to 2 decimals (0 when no class matched).`
- the example's import becomes `>>> from dnt.track import ReClass` and the example is marked `# doctest: +SKIP` on the `re_classify` line (it needs a video).

- [ ] **Step 4: Update `CLAUDE.md` and the changelog**

In `CLAUDE.md`, replace the line about `track/re_class.py` with:

```markdown
   - `track/re_class.py` has two re-classifiers. `ReClass` marks person tracks that ride a bicycle or skateboard (summary CSV `track, cls, avg_score`, used by `refine` as hints). `VehReClass` gives each vehicle track car (2), bus (5) or truck (7) through a size → re-detection → VLM → fallback cascade and writes the class into the track file's `cls`; its VLM layer reuses `dnt.refine.vlm`.
```

In `docs/changelog.md`, under `## Unreleased` → `### New`, add:

```markdown
- `dnt.track.VehReClass` classifies vehicle tracks as car, bus or truck and writes the class into
  `cls` on every row. Each track goes through up to three layers: a size check (`hv_size`, at
  least `hv_min_count` real rows) that limits a big vehicle to bus or truck; re-detection on 10
  large frames spaced at least 5 frames apart, where the winning class needs 30% of the frames;
  and, when `vlm=` is set, one VLM question per undecided track using `dnt.refine`'s backends and
  cache. Undecided tracks become truck if flagged heavy, car otherwise. `vr.decisions` shows
  which layer decided each track.
```

and under `### Changed` (create the heading if missing):

```markdown
- `ReClass` reads every sampled frame once, in order, instead of reopening the video per track,
  and skips a frame the video cannot decode instead of stopping. A video OpenCV cannot open now
  raises `ValueError`; before, every track silently got `default_class`.
- `dnt.track.ReClass` is always importable; the fallback that set it to `None` is removed.
```

- [ ] **Step 5: Run everything**

Run: `.venv/bin/python -m pytest -q` (compare with the count recorded before Task 1), `.venv/bin/ruff check src tests tools`, and `.venv/bin/mkdocs build --strict` if the docs build is part of your environment.
Expected: all tests pass; lint clean; the API page for `dnt.track.re_class` renders `VehReClass`.

- [ ] **Step 6: Commit**

```bash
git add src/dnt/track/__init__.py src/dnt/track/re_class.py CLAUDE.md docs/changelog.md tests/test_imports.py
git commit -m "feat(track): export VehReClass and document it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
