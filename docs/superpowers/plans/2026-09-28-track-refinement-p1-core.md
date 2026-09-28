# dnt.refine Track Refinement — Plan 1 of 4: Core Pipeline (motion-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `dnt.refine` subpackage's core, so that `dnt-refine run` / `TrackRefiner.refine` split ID switches, screen false tracks, link fragments, drop orphans, and fill gaps. Every edit is recorded in a JSONL ledger. This plan scores with motion only, and routes uncertain events to `HUMAN_PENDING`.

**Architecture:**
- **Stages propose; they never edit.** Each stage (`switch.py`, `screen.py`, `link.py`) is a pure function of a *work table* that returns proposed `Event`s.
- **Routing and applying are separate.** `verify.py` routes events by confidence band; `apply.py` is the only code that changes the table; `refiner.py` runs the stages in the spec's order (switch → screen → link → orphan → fill) and writes the outputs.
- **Existing code moves in, with shims.** `interpolate_tracks_rts` and `link_tracklets` move into `dnt.refine`, and `dnt.track.post_process` keeps re-exporting them. Characterization baselines captured before the move pin their behaviour.
- **Appearance is an injected `Appearance` protocol.** This plan implements all appearance math and tests it with in-memory embeddings. Plan 2 supplies the real video-backed provider.

**Tech Stack:** Python ≥3.11, numpy, pandas, scipy (`linear_sum_assignment`), filterpy (Kalman/RTS), OpenCV (video metadata), PyYAML, pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-09-27-track-refinement-design.md`](../specs/2026-09-27-track-refinement-design.md) (rev. 7, approved 2026-09-28). Section references such as "§6.3" point there. Read it alongside this plan.

## Plan series

The spec is delivered in four plans. Each one leaves `main`-mergeable, tested software. Later plans are written against the interfaces this plan creates.

| Plan | Scope | Spec sections |
|---|---|---|
| **P1 (this plan)** | Package skeleton, I/O, config, primitives, the move of `interpolate_tracks_rts` / `link_tracklets` (with their two changes), events and ledger file, apply, all four stages plus the orphan pass (full algorithms, including appearance math, driven by an injected `Appearance`), band routing without a VLM, `TrackRefiner.refine`, `dnt-refine run`, API docs | §1 criteria 1, 2, 4, 5, 7; §2; §3; §4.1; §4.2 (file format, keys, header; *not* replay); §4.3; §5.1–§5.4, §5.6; §6; §9; §10 (non-replay rows); §11.1 |
| P2 Appearance | Video frame reader, occlusion-masked crops, `dino` / `reid` / `none` encoders, feature cache and key, dense re-sampling around stage 1 candidates, deferred encoder dependency checks, wiring the real `Appearance` into `refine` (replacing P1's motion-only fallback) | §5.3, §5.5, criterion-1 extras |
| P3 Verification | Evidence images, prompts, VLM backends (`openai_compat`, `anthropic`, `fake`), answer cache, votes, budget, retries, redirected edits, rider subtype, review HTML | §7, §8.1, §8.3 VLM counters |
| P4 Replay and audit | `TrackRefiner.apply` / `Ledger.replay` with re-proposal, input verification, decisions, rounds and history, stage 3 passes on replay, feature-cache replay rules, `audit` / `audit-score`, CLI `apply` / `audit`, reference case end to end, `realdata` marker, quickstart docs | §4.2 replay, §8.2, §11.2 replay tests, §11.4 |

## Global Constraints

- **Python and dependencies.** `requires-python = ">=3.11"`. **No new required dependencies.** Heavy optional libraries (`transformers`, `torchreid`, `openai`, `anthropic`) are never imported by P1 code.
- **Dependency rule (§2.2).** `dnt.refine` imports only `dnt.shared`, `dnt.engine`, `dnt` (for `__version__`), and third-party libraries. It must never import `dnt.track`, `dnt.detect`, `dnt.label`, `dnt.filter`, or `boxmot`. `tests/test_refine_independence.py` enforces this.
- **Track file (§2.5).** Headerless 10 columns `frame, track, x, y, w, h, score, cls, r3, r4`.
  - Output column 8 is `interp` (`0` observed, `1` filled).
  - Integer columns are written as ints. Output is sorted by `frame, track`, and track IDs are renumbered contiguously from 1.
- **Filled input rows (`r3 == 1`) are removed on input.** Only observed rows reach the stages.
- **Durations are in seconds in config.** They are converted with `to_frames(seconds, fps) = max(1, round(seconds * fps))`. Speeds are in box heights per second (h/s).
- **There is no default frame rate.** Without video and without `fps`, `refine` raises `ValueError` before any processing (§5.1).
- **Config defaults and thresholds are copied verbatim from spec §6/§9.** Every threshold is a config field.
- **Lint.** Ruff rules `E,F,I,UP,B,SIM,RUF,D` apply, with line length 100 and numpy-style docstrings.
  - Use ASCII only in code, comments and docstrings (`>=`, `x`, `->`), because ruff's `RUF001-003` flags confusable Unicode.
  - `zip(...)` always gets `strict=True`.
  - **Never add entries to the legacy per-file baseline.** Task 5 removes the `src/dnt/track/post_process.py` entry.
- **Tests.** The default suite is CPU-only and needs no network: `.venv/bin/python -m pytest`. New tests live in `tests/refine/` (a package), except `tests/test_refine_independence.py`.
- **Existing tests must keep passing unchanged:**
  - `tests/test_post_process.py`
  - `tests/test_filter.py::test_filter_interpolate_uses_package_module`
  - `tests/test_pipeline_cpu.py`
  - `tests/test_imports.py`
- **Work on branch `feature/track-refinement`.** Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Run commands from the repository root with the venv:** `.venv/bin/python -m pytest …`, `.venv/bin/ruff check …`.

## Review Focus

These are the five inputs most likely to hurt a real user that the spec implies but does not test directly. Each has a pinned test in the task that owns the code:

1. **Raw track IDs that are large or sparse.** Examples are the `+10000` offset from two-pass vehicle tracking, or IDs like 3, 97, 12004. Split tails must get IDs that never collide, and lineage must keep the raw IDs. → Task 9, `test_split_tail_id_never_collides_with_sparse_ids`.
2. **Degenerate tracks.** A single-row track, or a tracker glitch that writes two rows for the same `(track, frame)`, must not crash any stage. Duplicates are dropped, keeping the first, with a warning. → Task 6, `test_duplicate_track_frame_rows_keep_first`; Task 11, `test_single_row_and_two_row_tracks_do_not_crash`; Task 12, `test_single_row_track_has_no_screen_event`.
3. **Large frame offsets.** Frame numbers starting at 1 (MOT) or at 120000 (clips cut from long recordings) must give the same scores as frames starting at 0. → Task 16, `test_frame_offset_does_not_change_scores`.
4. **A missing detection score (`score == -1`)**, from MOTChallenge files or trackers that don't write confidence, must not read as "low confidence" in the static cue. → Task 12, `test_static_cue_ignores_missing_scores`.
5. **Context that doesn't line up with the tracks.** A context file covering a different frame range, or an empty context file, gives zero context hits, not a crash or a spurious `DROP`. → Task 12, `test_context_with_no_overlapping_frames_is_harmless`.

## Corrections and decisions made while writing this plan

These are reported at handoff.

- **Dry run (2026-09-28).** The code blocks in this plan were assembled into a scratch copy of the repository and executed. The baselines were generated first, then every task was applied in order.
  - The full default suite passes, including every new test: 422 passed, 12 xfailed (pre-existing). This was re-run after the `dnt.refine` rename and the Tracker-style API.
  - `ruff check src tests` is clean.
  - `ruff format` reformats 13 of the new files. That is layout only, and Task 17 Step 5 runs it.
  - The two "verbatim" moves (Task 3's docstring, Task 5's nested helpers) were filled from the original source during the dry run.
- **Stage 1 z-score floor.** §6.1 writes the robust z-score as `(A_t - median) / (1.4826*MAD + eps)`. With ideal embeddings, MAD is 0, so any change saturates, and the appearance ramp creates plateaus. The plan uses a configurable floor `switch.mad_floor = 0.02` as `eps`. It also breaks ties between equal candidate scores by the raw appearance change `A_t`, then by the motion term, so the cut lands on the real change frame.
- **Implicit thresholds become config fields.** §6 says "each threshold is a config field", but §9's YAML lists only the main ones. The remaining constants named in §6 become fields with the spec's values. Each is listed in Task 7 with its spec source:
  - `switch.delta`, `nms_seconds`, `contact_iou`, `bimodal_purity`, `bimodal_silhouette_min`, `swap_boost`, and `ramps`;
  - `screen.in_vehicle_iob` and the other screen thresholds, plus `ramps`;
  - `link.static_speed`, `static_seconds`, `static_radius`, `overlap_frames`, `overlap_iou`, `k_embed`, `border_margin`, `speed_seconds`, `n_alternatives`, `heading_min_speed`;
  - `motion.moving_min`, and `orphan.ramp`.
- **`signals.replaces` holds the rejected event's `proposal_key`**, not its `id`. IDs are assigned by the runner, and the key is stable across rounds. That is what P4 needs.
- **Parity (criterion 7) compares groupings, not literal IDs.** `link_tracklets` keeps the earliest tracklet's ID, while `refine` renumbers. The parity test therefore compares *which input tracklets end up together*.
- **Rider subtype in P1.** With no VLM, an auto-accepted `RECLASS` whose `new_cls` is `None` becomes `HUMAN_PENDING` (§4.3, rider-subtype rule). A localized ReClass hint can still settle it.
- **Module shape (maintainer decision, 2026-09-28).** The package is `dnt.refine` rather than the roadmap's `dnt.post`, next to `dnt.detect` and `dnt.track`. `TrackRefiner` is used like `Tracker`: `config` or `config_yaml`, `device`, `refine(track_file, out_file, video_file=...)` returning a DataFrame (the full result is on `last_result`), and `refine_batch(...)`. The spec's section 2.4 was updated to match.
- **Appearance before Plan 2.** When a video is given but no `appearance_factory` is injected, P1 logs a WARNING and runs motion-only. Plan 2 replaces this fallback.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/dnt/refine/__init__.py` | create | Public API: `TrackRefiner`, `RefineConfig`, `RefineResult`, `interpolate_tracks_rts`, `link_tracklets` |
| `src/dnt/refine/io.py` | create | Column constants, `read_tracks` (dnt/MOT), `to_work`, `read_context`, `write_tracks`, `sha256_file`, `video_info`, `video_fingerprint` |
| `src/dnt/refine/config.py` | create | `RefineConfig` dataclass tree, per-target defaults, strict YAML round trip, validation, `to_frames` |
| `src/dnt/refine/primitives.py` | create | `ramp`, `cv_kalman`, `kalman_nis`, speeds, heading smoothness, majority class, IoU/IoB wrappers over `dnt.engine`, `frame_runs`, `occlusion_flags` |
| `src/dnt/refine/interpolate.py` | create (moved) | `interpolate_tracks_rts`, plus observed-only measurements and `protected_gaps` |
| `src/dnt/refine/link.py` | create (moved + new) | Legacy helpers and `link_tracklets`; stage 3 descriptors, gates, scoring, passes, chains, legacy mode |
| `src/dnt/refine/events.py` | create | `EventKind`, `Decision`, `Event`, `proposal_key`, `Ledger` read/write |
| `src/dnt/refine/apply.py` | create | Lineage, split, drop, reclass, chain merge, renumber, `apply_edit` |
| `src/dnt/refine/verify.py` | create | `Band`, `band_route`, `decide`, `route_without_vlm` |
| `src/dnt/refine/features.py` | create | `Appearance` protocol, `ArrayAppearance`, `track_embeddings` |
| `src/dnt/refine/hints.py` | create | `ReclassHint`, `read_reclass_hints` |
| `src/dnt/refine/switch.py` | create | Stage 1 proposals |
| `src/dnt/refine/screen.py` | create | Stage 2 proposals and the orphan pass |
| `src/dnt/refine/refiner.py` | create | `TrackRefiner`, `RefineResult`, `resolve_fps`, `fill_stage` |
| `src/dnt/refine/cli.py` | create | `dnt-refine run` |
| `src/dnt/track/post_process.py` | modify | Becomes a re-export shim |
| `src/dnt/filter/filter.py:833` | modify | Wrapper imports from `dnt.refine.interpolate` |
| `pyproject.toml` | modify | `[project.scripts] dnt-refine`; drop the `post_process.py` per-file ignore |
| `tests/refine/…` | create | Unit and integration tests, fixtures, characterization data |
| `tests/test_refine_independence.py` | create | §2.2 import rule |
| `docs/api/refine/*.md`, `mkdocs.yml`, `docs/api/track/post_process.md`, `docs/changelog.md` | create/modify | API docs and changelog |

The *work table* is the internal representation every stage uses. It is a DataFrame with columns `frame, track, x, y, w, h, score, cls, interp, r4, raw_id`:
- `raw_id` is the input track ID of the row, and never changes.
- `interp` is always 0 until stage 4.
- Functions in `apply.py` **preserve the DataFrame index**, except `renumber`. Row-aligned Series such as the occlusion flags therefore stay valid across stages.

---
### Task 1: Package skeleton, independence test, characterization baselines

**Files:**
- Create: `src/dnt/refine/__init__.py`
- Create: `tests/refine/__init__.py` (empty)
- Create: `tests/refine/_fixtures.py`
- Create: `tests/refine/make_baselines.py`
- Create: `tests/refine/data/raw_{0,1,2}.csv`, `interp_{0,1,2}.csv`, `interp_smooth_{0,1,2}.csv`, `link_{0,1,2}.csv` (generated)
- Test: `tests/test_refine_independence.py`

**Interfaces:**
- Produces:
  - `tests.refine._fixtures`:
    - `TRACK_COLUMNS: list[str]`
    - `box_rows(track, frames, x0, y0, *, vx=0.0, vy=0.0, w=30.0, h=60.0, cls=0, score=0.9) -> list[list]`
    - `table(*row_lists) -> pd.DataFrame`
    - `random_tracks(seed=0, n_objects=40, n_frames=300) -> pd.DataFrame`
    - `load_raw(seed) -> pd.DataFrame`
    - `DATA: Path`
  - The baseline CSVs in `tests/refine/data/`.

The baselines **must be generated before any code moves**. They record today's `interpolate_tracks_rts` and `link_tracklets` outputs, which Tasks 3–5 must reproduce byte for byte.

- [ ] **Step 1: Create the package and test package**

```python
# src/dnt/refine/__init__.py
"""Track refinement: switch splitting, false-track screening, linking, and gap filling."""
```

```python
# tests/refine/__init__.py
```

- [ ] **Step 2: Write the fixtures module**

```python
# tests/refine/_fixtures.py
"""Deterministic synthetic track tables for the dnt.refine tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]
DATA = Path(__file__).resolve().parent / "data"


def box_rows(track, frames, x0, y0, *, vx=0.0, vy=0.0, w=30.0, h=60.0, cls=0, score=0.9):
    """Rows of one constant-velocity box; position is relative to the first listed frame."""
    frames = [int(f) for f in frames]
    if not frames:
        return []
    f0 = frames[0]
    return [
        [f, track, x0 + vx * (f - f0), y0 + vy * (f - f0), w, h, score, cls, -1, -1]
        for f in frames
    ]


def table(*row_lists) -> pd.DataFrame:
    """Concatenate row lists into a 10-column raw track table."""
    rows = [r for rl in row_lists for r in rl]
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def random_tracks(seed: int = 0, n_objects: int = 40, n_frames: int = 300) -> pd.DataFrame:
    """Linear movers with random gaps; long gaps often restart the object under a new ID."""
    rng = np.random.default_rng(seed)
    rows = []
    next_id = 1
    for _ in range(n_objects):
        start = int(rng.integers(0, n_frames - 60))
        length = int(rng.integers(40, n_frames - start))
        x0, y0 = rng.uniform(0, 1500), rng.uniform(0, 900)
        vx, vy = rng.uniform(-6, 6), rng.uniform(-4, 4)
        w, h = rng.uniform(20, 80), rng.uniform(40, 120)
        cls = int(rng.choice([0, 2]))
        tid = next_id
        next_id += 1
        f = start
        while f < start + length:
            if rng.random() < 0.03:
                gap = int(rng.integers(2, 25))
                f += gap
                if gap > 8 and rng.random() < 0.6:
                    tid = next_id
                    next_id += 1
                continue
            j = rng.normal(0, 1.0, size=4)
            k = f - start
            rows.append([
                f, tid, round(x0 + vx * k + j[0], 1), round(y0 + vy * k + j[1], 1),
                round(w + j[2], 1), round(h + j[3], 1), round(float(rng.uniform(0.3, 0.95)), 2),
                cls, -1, -1,
            ])
            f += 1
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    return df.sort_values(["frame", "track"]).reset_index(drop=True)


def load_raw(seed: int) -> pd.DataFrame:
    """Read a saved random track table back with named columns (the baseline input form)."""
    return pd.read_csv(DATA / f"raw_{seed}.csv", header=None, names=TRACK_COLUMNS)
```

- [ ] **Step 3: Write the baseline generator**

```python
# tests/refine/make_baselines.py
"""Write characterization baselines for the dnt.refine move.

Run ONCE, before moving any code (plan Task 1), against the pre-move dnt.track.post_process:

    .venv/bin/python tests/refine/make_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _fixtures import DATA, load_raw, random_tracks  # noqa: E402

from dnt.track.post_process import interpolate_tracks_rts, link_tracklets  # noqa: E402


def main() -> None:
    """Write raw inputs and the current interpolate/link outputs for three seeds."""
    DATA.mkdir(exist_ok=True)
    for seed in (0, 1, 2):
        random_tracks(seed=seed).to_csv(DATA / f"raw_{seed}.csv", index=False, header=False)
        raw = load_raw(seed)
        interpolate_tracks_rts(
            raw.copy(), output_file=str(DATA / f"interp_{seed}.csv"), verbose=False
        )
        interpolate_tracks_rts(
            raw.copy(),
            output_file=str(DATA / f"interp_smooth_{seed}.csv"),
            smooth_existing=True,
            verbose=False,
        )
        link_tracklets(
            raw.copy(), output_file=str(DATA / f"link_{seed}.csv"), max_gap=20, verbose=False
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the baselines on the unmodified code**

Run: `.venv/bin/python tests/refine/make_baselines.py && ls tests/refine/data`
Expected: 12 CSV files (`raw_*`, `interp_*`, `interp_smooth_*`, `link_*` for seeds 0–2), each non-empty.

- [ ] **Step 5: Write the independence test**

```python
# tests/test_refine_independence.py
import json
import subprocess
import sys

FORBIDDEN = ("dnt.track", "dnt.detect", "dnt.label", "dnt.filter", "boxmot")


def test_refine_imports_nothing_forbidden():
    code = (
        "import importlib, json, pkgutil, sys\n"
        "import dnt.refine\n"
        "for m in pkgutil.walk_packages(dnt.refine.__path__, 'dnt.refine.'):\n"
        "    importlib.import_module(m.name)\n"
        f"forbidden = {FORBIDDEN!r}\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if any(m == p or m.startswith(p + '.') for p in forbidden))\n"
        "print(json.dumps(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
```

- [ ] **Step 6: Run it**

Run: `.venv/bin/python -m pytest tests/test_refine_independence.py -v`
Expected: PASS. The package is empty, so this pins the rule before any code arrives.

- [ ] **Step 7: Commit**

```bash
git add src/dnt/refine/__init__.py tests/refine tests/test_refine_independence.py
git commit -m "test: add dnt.refine skeleton, independence test, and pre-move baselines

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Numeric primitives

**Files:**
- Create: `src/dnt/refine/primitives.py`
- Test: `tests/refine/test_primitives.py`

**Interfaces:**
- Consumes: `dnt.engine.ious` (tlbr, pixel-inclusive), `dnt.engine.iobs` (xywh), `dnt.engine.cluster_by_gap`.
- Produces (all in `dnt.refine.primitives`):
  - `ramp(x, lo: float, hi: float) -> float | np.ndarray`
  - `cv_kalman(process_var=10.0, meas_var_pos=25.0, meas_var_size=16.0) -> filterpy.kalman.KalmanFilter`
  - `xywh_to_z(boxes) -> np.ndarray`
  - `kalman_nis(frames, boxes, *, process_var=10.0, meas_var_pos=25.0, meas_var_size=16.0) -> np.ndarray`
  - `box_centers(boxes) -> np.ndarray`
  - `rolling_height(h, window) -> np.ndarray`
  - `speeds_hps(frames, boxes, fps, window=15) -> np.ndarray`
  - `span_speed(frames, boxes, fps, seconds, *, at: str) -> float`
  - `heading_smoothness(frames, boxes, fps, *, window=15, moving_min=0.3) -> float`
  - `majority_class(values) -> int`
  - `iou_matrix(a, b) -> np.ndarray`
  - `iob_matrix(a, b) -> np.ndarray`
  - `frame_runs(frames) -> list[tuple[int, int]]`
  - `occlusion_flags(work: pd.DataFrame, context: pd.DataFrame | None, thr: float) -> pd.Series`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_primitives.py
import numpy as np
import pandas as pd
import pytest

from dnt.refine import primitives as P


def test_ramp_increasing_decreasing_and_clipped():
    assert P.ramp(0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert P.ramp(2.0, 0.0, 1.0) == 1.0
    assert P.ramp(0.2, 0.3, 0.05) == pytest.approx(0.4)  # decreasing ramp
    np.testing.assert_allclose(P.ramp(np.array([-1.0, 0.25, 9.0]), 0.0, 0.5), [0.0, 0.5, 1.0])


def test_cv_kalman_matches_interpolate_model():
    kf = P.cv_kalman(10.0, 25.0, 16.0)
    assert kf.F.shape == (8, 8) and kf.F[0, 1] == 1 and kf.F[1, 1] == 1
    assert kf.H[0, 0] == 1 and kf.H[1, 2] == 1 and kf.H[2, 4] == 1 and kf.H[3, 6] == 1
    np.testing.assert_allclose(np.diag(kf.R), [25.0, 25.0, 16.0, 16.0])
    np.testing.assert_allclose(np.diag(kf.P), np.full(8, 100.0))


def test_nis_is_chi2_4_on_model_consistent_simulation():
    rng = np.random.default_rng(3)
    kf = P.cv_kalman(10.0, 25.0, 16.0)
    x = np.array([500.0, 2.0, 300.0, -1.0, 60.0, 0.0, 120.0, 0.0])
    boxes = []
    for _ in range(3000):
        x = kf.F @ x + rng.multivariate_normal(np.zeros(8), kf.Q)
        z = kf.H @ x + rng.multivariate_normal(np.zeros(4), kf.R)
        boxes.append([z[0] - z[2] / 2, z[1] - z[3] / 2, z[2], z[3]])
    nis = P.kalman_nis(np.arange(3000), np.array(boxes))
    assert np.nanmean(nis[100:]) == pytest.approx(4.0, abs=0.4)


def test_nis_spikes_on_a_jump_after_a_gap():
    frames = np.r_[np.arange(0, 40), np.arange(45, 80)]
    x = 100 + 2.0 * frames + np.where(frames >= 45, 300.0, 0.0)
    boxes = np.column_stack([x, np.full(len(frames), 100.0), np.full(len(frames), 30.0),
                             np.full(len(frames), 60.0)])
    nis = P.kalman_nis(frames, boxes)
    i45 = int(np.flatnonzero(frames == 45)[0])
    assert nis[i45] > 18.47
    assert np.nanmax(nis[5:i45]) < 5.0


def test_speed_in_box_heights_per_second():
    frames = np.arange(10)
    boxes = np.column_stack([2.0 * frames, np.zeros(10), np.full(10, 30.0), np.full(10, 60.0)])
    v = P.speeds_hps(frames, boxes, fps=30.0)
    assert np.isnan(v[0])
    np.testing.assert_allclose(v[1:], 1.0)  # 2 px/frame * 30 fps / 60 px
    assert P.span_speed(frames, boxes, 30.0, 0.2, at="end") == pytest.approx(1.0)


def test_heading_smoothness_straight_vs_zigzag():
    f = np.arange(20)
    straight = np.column_stack([10.0 * f, np.zeros(20), np.full(20, 20.0), np.full(20, 40.0)])
    zig = straight.copy()
    zig[:, 1] = np.where(f % 2 == 0, 0.0, 30.0)
    zig[:, 0] = 0.0
    assert P.heading_smoothness(f, straight, 10.0) == pytest.approx(1.0)
    assert P.heading_smoothness(f, zig, 10.0) < 0.2


def test_majority_class_tie_goes_to_latest():
    assert P.majority_class([2, 2, 7, 7]) == 7
    assert P.majority_class([2, 2, 2, 7]) == 2
    assert P.majority_class([]) == -1


def test_iou_and_iob_matrices():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 5.0, 5.0], [0.0, 0.0, 20.0, 20.0]])
    iou = P.iou_matrix(a, b)
    assert iou.shape == (1, 3) and iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == 0.0
    assert P.iob_matrix(a, b)[0, 2] == pytest.approx(1.0)  # a fully inside the third box
    assert P.iou_matrix(np.empty((0, 4)), b).shape == (0, 3)


def test_frame_runs():
    assert P.frame_runs([5, 1, 2, 3, 7, 8]) == [(1, 3), (5, 5), (7, 8)]
    assert P.frame_runs([]) == []


def test_occlusion_flags_use_same_frame_boxes_and_context():
    work = pd.DataFrame({"frame": [0, 0, 1], "track": [1, 2, 1],
                         "x": [0.0, 2.0, 0.0], "y": [0.0, 0.0, 0.0],
                         "w": [10.0, 10.0, 10.0], "h": [10.0, 10.0, 10.0]})
    ctx = pd.DataFrame({"frame": [1], "track": [-1], "x": [1.0], "y": [0.0], "w": [10.0],
                        "h": [10.0], "cls": [2]})
    flags = P.occlusion_flags(work, ctx, 0.3)
    assert flags.tolist() == [True, True, True]
    assert P.occlusion_flags(work, None, 0.3).tolist() == [True, True, False]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_primitives.py -v`
Expected: FAIL with `ImportError: cannot import name 'primitives'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/primitives.py
"""Numeric primitives shared by the dnt.refine stages (spec section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import cluster_by_gap, iobs, ious


def ramp(x, lo: float, hi: float):
    """Map ``x`` linearly from ``[lo, hi]`` onto ``[0, 1]`` and clip; ``lo > hi`` decreases."""
    arr = np.asarray(x, dtype=float)
    out = (arr >= lo).astype(float) if hi == lo else np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def cv_kalman(process_var: float = 10.0, meas_var_pos: float = 25.0, meas_var_size: float = 16.0):
    """Return the constant-velocity Kalman filter shared by stage 1 and stage 4 (spec 5.2).

    State is ``[cx, vx, cy, vy, w, vw, h, vh]`` with one step per frame. The caller sets ``x``.
    """
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import KalmanFilter

    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8)
    for i in range(4):
        kf.F[2 * i, 2 * i + 1] = 1.0
    kf.H = np.zeros((4, 8))
    for i in range(4):
        kf.H[i, 2 * i] = 1.0
    q2 = Q_discrete_white_noise(dim=2, dt=1.0, var=process_var)
    kf.Q = np.zeros((8, 8))
    for i in range(4):
        kf.Q[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = q2
    kf.R = np.diag([meas_var_pos, meas_var_pos, meas_var_size, meas_var_size]).astype(float)
    kf.P = np.eye(8) * 100.0
    return kf


def xywh_to_z(boxes) -> np.ndarray:
    """Convert (N, 4) top-left ``x, y, w, h`` boxes to Kalman measurements ``cx, cy, w, h``."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0, b[:, 2], b[:, 3]])


def kalman_nis(
    frames, boxes, *, process_var: float = 10.0, meas_var_pos: float = 25.0,
    meas_var_size: float = 16.0,
) -> np.ndarray:
    """Return the normalized innovation squared at each observed row; NaN for the first row."""
    frames = np.asarray(frames, dtype=int)
    out = np.full(len(frames), np.nan)
    if len(frames) == 0:
        return out
    z = xywh_to_z(boxes)
    kf = cv_kalman(process_var, meas_var_pos, meas_var_size)
    kf.x = np.array([z[0, 0], 0.0, z[0, 1], 0.0, z[0, 2], 0.0, z[0, 3], 0.0])
    row_of = {int(f): i for i, f in enumerate(frames)}
    for f in range(int(frames[0]) + 1, int(frames[-1]) + 1):
        kf.predict()
        i = row_of.get(f)
        if i is None:
            continue
        y = z[i] - kf.H @ kf.x
        s = kf.H @ kf.P @ kf.H.T + kf.R
        out[i] = float(y @ np.linalg.solve(s, y))
        kf.update(z[i])
    return out


def box_centers(boxes) -> np.ndarray:
    """Return the (N, 2) centers of ``x, y, w, h`` boxes."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return b[:, :2] + b[:, 2:4] / 2.0


def rolling_height(h, window: int) -> np.ndarray:
    """Return the centered rolling median of box heights (spec 5.1)."""
    s = pd.Series(np.asarray(h, dtype=float))
    return s.rolling(max(int(window), 1), center=True, min_periods=1).median().to_numpy()


def speeds_hps(frames, boxes, fps: float, window: int = 15) -> np.ndarray:
    """Return per-row speed in box heights per second; NaN for the first row (spec 5.1)."""
    fr = np.asarray(frames, dtype=float)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = np.full(len(fr), np.nan)
    if len(fr) < 2:
        return out
    c = box_centers(b)
    ht = np.maximum(rolling_height(b[:, 3], window), 1.0)
    dist = np.linalg.norm(np.diff(c, axis=0), axis=1)
    out[1:] = dist / (ht[1:] * np.maximum(np.diff(fr), 1.0) / fps)
    return out


def span_speed(frames, boxes, fps: float, seconds: float, *, at: str) -> float:
    """Return the net speed (h/s) over the first or last ``seconds`` of rows; 0.0 if undefined."""
    fr = np.asarray(frames, dtype=int)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(fr) < 2:
        return 0.0
    span = seconds * fps
    if at == "end":
        sel = fr >= fr[-1] - span
    elif at == "start":
        sel = fr <= fr[0] + span
    else:
        raise ValueError(f"at must be 'start' or 'end', not {at!r}")
    f, bb = fr[sel], b[sel]
    if len(f) < 2 or f[-1] == f[0]:
        return 0.0
    c = box_centers(bb[[0, -1]])
    h = max(float(np.median(bb[:, 3])), 1.0)
    return float(np.linalg.norm(c[1] - c[0]) / (h * (f[-1] - f[0]) / fps))


def heading_smoothness(
    frames, boxes, fps: float, *, window: int = 15, moving_min: float = 0.3
) -> float:
    """Return ``1 - circular variance`` of heading over moving steps; NaN if under two steps."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(b) < 3:
        return float("nan")
    v = speeds_hps(frames, b, fps, window)
    d = np.diff(box_centers(b), axis=0)[v[1:] >= moving_min]
    if len(d) < 2:
        return float("nan")
    u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return float(np.linalg.norm(u.mean(axis=0)))


def majority_class(values) -> int:
    """Return the most frequent class; ties go to the class seen latest; -1 when empty."""
    vals = [int(v) for v in values]
    if not vals:
        return -1
    counts: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, c in enumerate(vals):
        counts[c] = counts.get(c, 0) + 1
        last[c] = i
    best = max(counts.values())
    return max((c for c, n in counts.items() if n == best), key=lambda c: last[c])


def _tlbr(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])


def iou_matrix(a, b) -> np.ndarray:
    """Return the (N, M) IoU of ``x, y, w, h`` boxes via ``dnt.engine.ious`` (spec 5.6)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return ious(_tlbr(a), _tlbr(b))


def iob_matrix(a, b) -> np.ndarray:
    """Return the (N, M) intersection over the area of each box in ``a`` (spec 5.6)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return iobs(a, b)[0]


def frame_runs(frames) -> list[tuple[int, int]]:
    """Return runs of consecutive frames as ``(first, last)`` pairs."""
    f = np.unique(np.asarray(frames, dtype=int))
    if f.size == 0:
        return []
    return [(int(r[0]), int(r[-1])) for r in cluster_by_gap(f, 1)]


def occlusion_flags(work: pd.DataFrame, context: pd.DataFrame | None, thr: float) -> pd.Series:
    """Return True where a row's box has IoU >= ``thr`` with another box in its frame (spec 5.3)."""
    flags = pd.Series(False, index=work.index)
    ctx: dict[int, np.ndarray] = {}
    if context is not None and len(context):
        ctx = {int(f): g[["x", "y", "w", "h"]].to_numpy(float) for f, g in context.groupby("frame")}
    for f, g in work.groupby("frame"):
        boxes = g[["x", "y", "w", "h"]].to_numpy(float)
        best = np.zeros(len(boxes))
        if len(boxes) > 1:
            m = iou_matrix(boxes, boxes)
            np.fill_diagonal(m, 0.0)
            best = m.max(axis=1)
        others = ctx.get(int(f))
        if others is not None and len(others):
            best = np.maximum(best, iou_matrix(boxes, others).max(axis=1))
        flags.loc[g.index] = best >= thr
    return flags
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_primitives.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/primitives.py tests/refine/test_primitives.py
git commit -m "feat(refine): add shared numeric primitives (ramp, CV Kalman, NIS, speeds, geometry)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Move `interpolate_tracks_rts` into `dnt.refine` (behaviour-preserving)

**Files:**
- Create: `src/dnt/refine/interpolate.py`
- Modify: `src/dnt/track/post_process.py` (delete lines 1–318, the module docstring, imports, and `interpolate_tracks_rts`; keep `link_tracklets` for Task 5)
- Modify: `src/dnt/filter/filter.py:833` (wrapper import)
- Test: `tests/refine/test_interpolate.py`

**Interfaces:**
- Consumes: `primitives.cv_kalman`.
- Produces: `dnt.refine.interpolate.interpolate_tracks_rts`, with the signature unchanged from 0.3.3. `dnt.track.post_process.interpolate_tracks_rts` and `dnt.track.interpolate_tracks_rts` remain the same object.

- [ ] **Step 1: Write the characterization test**

```python
# tests/refine/test_interpolate.py
import pandas as pd
import pytest

from dnt.track.post_process import interpolate_tracks_rts as shim_interpolate

from ._fixtures import DATA, load_raw


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("smooth", "name"), [(False, "interp"), (True, "interp_smooth")])
def test_matches_pre_move_baseline(tmp_path, seed, smooth, name):
    out = tmp_path / "o.csv"
    shim_interpolate(load_raw(seed), output_file=str(out), smooth_existing=smooth, verbose=False)
    assert out.read_bytes() == (DATA / f"{name}_{seed}.csv").read_bytes()


def test_shim_and_package_are_the_same_function():
    from dnt.refine.interpolate import interpolate_tracks_rts
    from dnt.track import interpolate_tracks_rts as track_level

    assert shim_interpolate is interpolate_tracks_rts is track_level
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_interpolate.py -v`
Expected: the baseline tests PASS (the old code is still in place), and `test_shim_and_package_are_the_same_function` FAILS with `ModuleNotFoundError: No module named 'dnt.refine.interpolate'`.

- [ ] **Step 3: Create `src/dnt/refine/interpolate.py`**

Move the function body verbatim into two helpers. The per-track body becomes `_rts_rows`, and the Kalman setup becomes a `cv_kalman` call. Keep the original docstring of `interpolate_tracks_rts` (`src/dnt/track/post_process.py` lines 30–123) as-is, and wrap any line over 100 characters.

```python
# src/dnt/refine/interpolate.py
"""Kalman RTS gap filling for track tables (spec 6.4).

Moved from ``dnt.track.post_process`` (which re-exports it) in dnt 0.4.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from tqdm import tqdm

from .primitives import cv_kalman

DEFAULT_COL_NAMES = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]


def _set_flag(row: dict, columns, add_interp_flag: bool, interp_col: str, value: int) -> None:
    if not add_interp_flag:
        return
    if "r3" in columns:
        row["r3"] = value
    else:
        row[interp_col] = value


def _rts_rows(
    g: pd.DataFrame, track_id, *, fill_gaps_only: bool, smooth_existing: bool,
    process_var: float, meas_var_pos: float, meas_var_size: float, max_gap: int,
    add_interp_flag: bool, interp_col: str,
) -> list[dict]:
    """Smooth one run of observed rows (sorted, unique frames) and return output rows."""
    from filterpy.kalman import rts_smoother

    frames_obs = g["frame"].astype(int).to_numpy()
    frame_start = int(frames_obs.min())
    frame_end = int(frames_obs.max())
    frames_full = np.arange(frame_start, frame_end + 1, dtype=int)
    observed_set = set(frames_obs.tolist())
    fillable_missing: set[int] = set()
    for f0, f1 in pairwise(frames_obs):
        gap = int(f1 - f0 - 1)
        if 0 < gap <= max_gap:
            fillable_missing.update(range(int(f0) + 1, int(f1)))

    cx = (g["x"].astype(float) + (g["w"].astype(float) / 2.0)).to_numpy()
    cy = (g["y"].astype(float) + (g["h"].astype(float) / 2.0)).to_numpy()
    ww = g["w"].astype(float).to_numpy()
    hh = g["h"].astype(float).to_numpy()
    z_map = {
        int(f): np.array([cx[i], cy[i], ww[i], hh[i]], dtype=float)
        for i, f in enumerate(frames_obs)
    }
    row_map = {int(row["frame"]): row for row in g.to_dict("records")}

    kf = cv_kalman(process_var, meas_var_pos, meas_var_size)
    z0 = z_map[frame_start]
    kf.x = np.array([z0[0], 0.0, z0[1], 0.0, z0[2], 0.0, z0[3], 0.0], dtype=float)
    xs, ps, fs, qs = [], [], [], []
    for f in frames_full:
        kf.predict()
        z = z_map.get(int(f))
        if z is not None:
            kf.update(z)
        xs.append(kf.x.copy())
        ps.append(kf.P.copy())
        fs.append(kf.F.copy())
        qs.append(kf.Q.copy())
    xs_s, _, _, _ = rts_smoother(np.asarray(xs), np.asarray(ps), np.asarray(fs), np.asarray(qs))

    if "cls" in g.columns and len(g["cls"].dropna()) > 0:
        cls_mode = g["cls"].mode()
        cls_fill = float(cls_mode.iloc[0]) if len(cls_mode) > 0 else -1
    else:
        cls_fill = -1
    has_score = "score" in g.columns and len(g["score"].dropna()) > 0
    score_fill = float(g["score"].mean()) if has_score else -1.0

    rows: list[dict] = []
    for i, frame in enumerate(frames_full.tolist()):
        sm_w = max(1.0, float(xs_s[i, 4]))
        sm_h = max(1.0, float(xs_s[i, 6]))
        sm_x = float(xs_s[i, 0]) - (sm_w / 2.0)
        sm_y = float(xs_s[i, 2]) - (sm_h / 2.0)
        if frame in observed_set:
            row = dict(row_map[frame])
            if smooth_existing or (not fill_gaps_only):
                row["x"], row["y"], row["w"], row["h"] = sm_x, sm_y, sm_w, sm_h
            _set_flag(row, g.columns, add_interp_flag, interp_col, 0)
            rows.append(row)
        elif frame in fillable_missing:
            row = {c: np.nan for c in g.columns}
            row["frame"] = frame
            row["track"] = track_id
            row["x"], row["y"], row["w"], row["h"] = sm_x, sm_y, sm_w, sm_h
            if "cls" in g.columns:
                row["cls"] = cls_fill
            if "score" in g.columns:
                row["score"] = score_fill
            _set_flag(row, g.columns, add_interp_flag, interp_col, 1)
            rows.append(row)
    return rows


def interpolate_tracks_rts(
    tracks: pd.DataFrame | None = None,
    track_file: str | None = None,
    output_file: str | None = None,
    col_names: list[str] | None = None,
    fill_gaps_only: bool = True,
    smooth_existing: bool = False,
    process_var: float = 10.0,
    meas_var_pos: float = 25.0,
    meas_var_size: float = 16.0,
    min_track_len: int = 2,
    max_gap: int = 30,
    add_interp_flag: bool = True,
    interp_col: str = "interp",
    verbose: bool = True,
    video_index: int | None = None,
    video_tot: int | None = None,
) -> pd.DataFrame:
    """<paste the original docstring from post_process.py lines 30-123 here, unchanged>"""
    if col_names is None:
        col_names = list(DEFAULT_COL_NAMES)
    if tracks is None:
        if not track_file:
            raise ValueError("Either `tracks` or `track_file` must be provided.")
        try:
            tracks = pd.read_csv(track_file, header=None)
        except pd.errors.EmptyDataError:
            tracks = pd.DataFrame(columns=col_names)
    if len(tracks) == 0:
        out = tracks.copy()
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out

    df = tracks.copy()
    required = ["frame", "track", "x", "y", "w", "h"]
    if all(c in df.columns for c in required):
        work = df.copy()
    else:
        if len(df.columns) < len(required):
            raise ValueError("tracks must include at least frame/track/x/y/w/h columns.")
        work = df.copy()
        work.columns = col_names[: len(df.columns)]
    work = work.sort_values(["track", "frame"]).reset_index(drop=True)

    output_rows: list[dict] = []
    grouped = list(work.groupby("track", sort=False))
    pbar = tqdm(total=len(grouped), unit=" tracks", disable=not verbose)
    if verbose:
        if video_index is not None and video_tot is not None:
            pbar.set_description_str(f"RTS interpolate {video_index} of {video_tot}")
        else:
            pbar.set_description_str("RTS interpolate")
    kw = {
        "fill_gaps_only": fill_gaps_only, "smooth_existing": smooth_existing,
        "process_var": process_var, "meas_var_pos": meas_var_pos,
        "meas_var_size": meas_var_size, "max_gap": max_gap,
        "add_interp_flag": add_interp_flag, "interp_col": interp_col,
    }
    for track_id, g in grouped:
        g = g.sort_values("frame").drop_duplicates("frame", keep="first").reset_index(drop=True)
        if len(g) < min_track_len:
            rows = g.to_dict("records")
            for r in rows:
                _set_flag(r, g.columns, add_interp_flag, interp_col, 0)
            output_rows.extend(rows)
        else:
            output_rows.extend(_rts_rows(g, track_id, **kw))
        pbar.update(1)
    pbar.close()

    out = pd.DataFrame(output_rows)
    if "r3" in out.columns:
        cols = list(out.columns)
        idx = cols.index("r3")
        out = out.rename(columns={"r3": interp_col})
        cols[idx] = interp_col
        out = out[cols]
    # Keep compatibility with legacy track file readers that enforce integer dtypes.
    for c in ["frame", "track", "x", "y", "w", "h", "cls", "r4", interp_col]:
        if c in out.columns:
            out[c] = out[c].fillna(-1).round().astype(int)
    if "score" in out.columns:
        out["score"] = out["score"].fillna(-1).astype(float)
    out = out.sort_values(["frame", "track"]).reset_index(drop=True)
    if output_file:
        out.to_csv(output_file, index=False, header=False)
    return out
```

The `"""<paste …>"""` line is not a placeholder to leave in. Replace it with the original numpy docstring text before running the tests.

- [ ] **Step 4: Replace the old implementation in `post_process.py`**

Delete lines 1–318 of `src/dnt/track/post_process.py` (the module docstring, the imports, and `interpolate_tracks_rts`), and put this at the top of the file, above `def link_tracklets`:

```python
"""Backward-compatible home of the track post-processing functions (moved to ``dnt.refine``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..refine.interpolate import interpolate_tracks_rts

__all__ = ["interpolate_tracks_rts", "link_tracklets"]
```

(`link_tracklets` still uses `np`, `pd` and `tqdm` until Task 5.)

- [ ] **Step 5: Point the Filter wrapper at the new module**

In `src/dnt/filter/filter.py`, inside `Filter.interpolate_tracks_rts`, change

```python
        from ..track.post_process import interpolate_tracks_rts as _interpolate_tracks_rts
```

to

```python
        from ..refine.interpolate import interpolate_tracks_rts as _interpolate_tracks_rts
```

and change its docstring reference to `:func:`dnt.refine.interpolate.interpolate_tracks_rts``.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_interpolate.py tests/test_post_process.py tests/test_filter.py tests/test_pipeline_cpu.py tests/test_refine_independence.py -v && .venv/bin/ruff check src/dnt/refine src/dnt/track/post_process.py src/dnt/filter/filter.py`
Expected: all PASS, and every baseline is byte-identical. Ruff is clean.

- [ ] **Step 7: Commit**

```bash
git add src/dnt/refine/interpolate.py src/dnt/track/post_process.py src/dnt/filter/filter.py tests/refine/test_interpolate.py
git commit -m "refactor: move interpolate_tracks_rts to dnt.refine; share the CV Kalman model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `interpolate_tracks_rts` — filled rows are not measurements; `protected_gaps`

**Files:**
- Modify: `src/dnt/refine/interpolate.py`
- Test: `tests/refine/test_interpolate.py` (append)

**Interfaces:**
- Produces: `interpolate_tracks_rts(..., protected_gaps: Mapping[int, Iterable[tuple[int, int]]] | None = None)`. This is a new keyword-only-by-convention parameter, added last, with default `None`.

- [ ] **Step 1: Write the failing tests**

Merge these imports into the import block at the top of `tests/refine/test_interpolate.py`, so the file's imports read:

```python
import numpy as np
import pandas as pd
import pytest

from dnt.refine.interpolate import interpolate_tracks_rts
from dnt.track.post_process import interpolate_tracks_rts as shim_interpolate

from ._fixtures import DATA, box_rows, load_raw, table
```

Then append the tests:

```python
# append to tests/refine/test_interpolate.py
def _positional(df):
    return df.set_axis(range(df.shape[1]), axis=1)


def test_filled_rows_are_not_measurements_and_are_re_estimated():
    raw = table(box_rows(1, [f for f in range(10) if f != 5], 10.0, 20.0, vx=2.0))
    fake = raw.iloc[[0]].copy()
    fake[["frame", "x", "r3"]] = [5, 500.0, 1]  # a filled row at an absurd position
    out = interpolate_tracks_rts(_positional(pd.concat([raw, fake])), verbose=False)
    row5 = out[out["frame"] == 5].iloc[0]
    assert row5["interp"] == 1
    assert abs(row5["x"] - 20) <= 2  # re-estimated near 10 + 2*5, not 500


def test_filled_rows_outside_fillable_gaps_are_dropped():
    raw = table(box_rows(1, range(0, 5), 10.0, 20.0), box_rows(1, range(20, 25), 10.0, 20.0))
    filled = table(box_rows(1, range(5, 20), 10.0, 20.0))
    filled["r3"] = 1
    out = interpolate_tracks_rts(_positional(pd.concat([raw, filled])), max_gap=2, verbose=False)
    assert sorted(out["frame"]) == [*range(0, 5), *range(20, 25)]


def test_protected_gap_is_never_filled_and_smoothing_does_not_cross():
    raw = table(box_rows(1, range(0, 10), 100.0, 300.0, vy=-5.0),
                box_rows(1, range(73, 83), 60.0, 240.0, vx=-5.0))
    kw = {"max_gap": 100, "verbose": False}
    filled = interpolate_tracks_rts(raw.copy(), **kw)
    assert set(range(10, 73)) <= set(filled["frame"])
    kept = interpolate_tracks_rts(raw.copy(), protected_gaps={1: [(9, 73)]}, **kw)
    assert not (set(range(10, 73)) & set(kept["frame"]))
    s_all = interpolate_tracks_rts(raw.copy(), protected_gaps={1: [(9, 73)]},
                                   smooth_existing=True, **kw)
    s_head = interpolate_tracks_rts(raw[raw["frame"] < 10].copy(), smooth_existing=True, **kw)
    cols = ["frame", "x", "y", "w", "h"]
    pd.testing.assert_frame_equal(s_all[s_all["frame"] < 10][cols].reset_index(drop=True),
                                  s_head[cols].reset_index(drop=True))


def test_two_protected_gaps_in_one_chain():
    raw = table(box_rows(1, range(0, 5), 0.0, 0.0), box_rows(1, range(20, 25), 0.0, 0.0),
                box_rows(1, range(40, 45), 0.0, 0.0))
    out = interpolate_tracks_rts(raw, max_gap=100, protected_gaps={1: [(4, 20), (24, 40)]},
                                 verbose=False)
    assert sorted(out["frame"]) == [*range(0, 5), *range(20, 25), *range(40, 45)]
    assert np.all(out["interp"] == 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_interpolate.py -v`
Expected: the new tests FAIL. The absurd row is used as a measurement, and `protected_gaps` is an unexpected keyword.

- [ ] **Step 3: Implement**

In `src/dnt/refine/interpolate.py`:

1. Add `from collections.abc import Iterable, Mapping` to the imports.
2. Add this helper above `interpolate_tracks_rts`:

```python
def _split_protected(g: pd.DataFrame, gaps: list[tuple[int, int]]) -> list[pd.DataFrame]:
    """Cut a track between consecutive observed frames that lie inside a protected gap."""
    if not gaps:
        return [g]
    frames = g["frame"].astype(int).to_numpy()
    cuts = [
        i for i in range(1, len(frames))
        if frames[i] - frames[i - 1] > 1
        and any(a <= frames[i - 1] and frames[i] <= b for a, b in gaps)
    ]
    bounds = [0, *cuts, len(frames)]
    return [g.iloc[s:e].reset_index(drop=True) for s, e in pairwise(bounds)]
```

3. Add the parameter after `video_tot`:

```python
    protected_gaps: Mapping[int, Iterable[tuple[int, int]]] | None = None,
```

and document both behaviours in the docstring's Parameters and Notes. `protected_gaps` lists gaps per track ID as `(last observed frame before, first observed frame after)`. The track is smoothed as independent segments at each one, so neither filling nor `smooth_existing` crosses it. Rows whose flag column (`interp_col`, or `r3` in the positional layout) equals 1 are not measurements: they are estimated again like missing frames, or dropped when outside a fillable gap.

4. Directly after `work = work.sort_values(["track", "frame"]).reset_index(drop=True)`, insert:

```python
    flag_col = interp_col if interp_col in work.columns else None
    if flag_col is None and "r3" in work.columns:
        flag_col = "r3"
    if flag_col is not None:
        is_filled = pd.to_numeric(work[flag_col], errors="coerce") == 1
        work = work.loc[~is_filled].reset_index(drop=True)
    if work.empty:
        out = tracks.iloc[0:0].copy()
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out
    protected = {
        int(k): [(int(a), int(b)) for a, b in v] for k, v in (protected_gaps or {}).items()
    }
```

5. Replace the `else:` branch of the per-track loop:

```python
        else:
            for seg in _split_protected(g, protected.get(int(track_id), [])):
                if len(seg) < min_track_len:
                    rows = seg.to_dict("records")
                    for r in rows:
                        _set_flag(r, seg.columns, add_interp_flag, interp_col, 0)
                    output_rows.extend(rows)
                else:
                    output_rows.extend(_rts_rows(seg, track_id, **kw))
```

- [ ] **Step 4: Run tests (new + baselines + legacy)**

Run: `.venv/bin/python -m pytest tests/refine/test_interpolate.py tests/test_post_process.py tests/test_filter.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS. The baselines are still byte-identical, because raw tracker files carry `r3 = -1` and no protected gaps.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/interpolate.py tests/refine/test_interpolate.py
git commit -m "feat(refine): interpolate ignores filled rows as measurements; add protected_gaps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Move `link_tracklets` into `dnt.refine.link` with shared legacy helpers

**Files:**
- Create: `src/dnt/refine/link.py`
- Modify: `src/dnt/track/post_process.py` (becomes a pure shim)
- Modify: `pyproject.toml` (remove the `"src/dnt/track/post_process.py" = ["E501"]` line)
- Test: `tests/refine/test_link_legacy.py`

**Interfaces:**
- Produces (in `dnt.refine.link`):
  - `LEGACY_COL_NAMES`
  - `_iou_xywh(a, b) -> float`
  - `_estimate_velocity(frames, cx, cy, k) -> tuple[float, float]`
  - `_DSU`
  - `_prepare_legacy(tracks, col_names) -> pd.DataFrame`
  - `_legacy_descriptors(df, vel_frames, pbar=None) -> dict[int, dict]`
  - `_legacy_gate_cost(a, b, *, max_gap, size_ratio_max, dist_mult, iou_min, w_d, w_iou, w_s, dist_growth=0.03, check_class=True, detail=False)`, which returns `float | tuple[float, dict] | None`
  - `_legacy_matches(stitchable, **gate_kw) -> list[tuple[int, int, float]]`
  - `link_tracklets(...)`, with the signature unchanged.

  Descriptor dicts keep the original keys: `track, cls, t_start, t_end, start_c, end_c, start_box, end_box, area_end, vx, vy, stitchable`.

- [ ] **Step 1: Write the characterization test**

```python
# tests/refine/test_link_legacy.py
import pytest

from dnt.track.post_process import link_tracklets as shim_link

from ._fixtures import DATA, load_raw


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pre_move_baseline(tmp_path, seed):
    out = tmp_path / "o.csv"
    shim_link(load_raw(seed), output_file=str(out), max_gap=20, verbose=False)
    assert out.read_bytes() == (DATA / f"link_{seed}.csv").read_bytes()


def test_shim_and_package_are_the_same_function():
    from dnt.refine.link import link_tracklets
    from dnt.track import link_tracklets as track_level

    assert shim_link is link_tracklets is track_level


def test_gate_cost_rejects_and_scores_like_the_original():
    from dnt.refine.link import _legacy_gate_cost

    a = {"track": 1, "cls": 2, "t_end": 10, "end_c": (115.0, 130.0), "end_box": (100.0, 100.0, 30.0, 60.0),
         "area_end": 1800.0, "vx": 2.0, "vy": 0.0}
    b = {"track": 2, "cls": 2, "t_start": 15, "start_c": (125.0, 130.0),
         "start_box": (110.0, 100.0, 30.0, 60.0)}
    kw = {"max_gap": 20, "size_ratio_max": 2.0, "dist_mult": 2.5, "iou_min": 0.05,
          "w_d": 1.0, "w_iou": 1.0, "w_s": 0.3}
    assert _legacy_gate_cost(a, b, **kw) == pytest.approx(0.0, abs=1e-6)
    assert _legacy_gate_cost(a, {**b, "cls": 7}, **kw) is None
    assert _legacy_gate_cost(a, {**b, "cls": 7}, check_class=False, **kw) is not None
    assert _legacy_gate_cost(a, {**b, "t_start": 40}, **kw) is None
    _cost, terms = _legacy_gate_cost(a, b, detail=True, **kw)
    assert set(terms) == {"dist", "iou_pred", "w_ratio", "h_ratio"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_link_legacy.py -v`
Expected: the baseline tests PASS. The other two FAIL with `ModuleNotFoundError: No module named 'dnt.refine.link'`.

- [ ] **Step 3: Create `src/dnt/refine/link.py` (legacy part)**

Keep `link_tracklets`'s full original numpy docstring (`post_process.py`, the text between its signature and `def _iou_xywh`), wrapping lines to 100 characters. Copy `_iou_xywh`, `_estimate_velocity` and `_DSU` verbatim from their nested definitions inside `link_tracklets`, then dedent them to module level. Their original docstrings stay. After dedenting, `_estimate_velocity`'s signature is 106 characters; wrap its parameters onto a second line.

```python
# src/dnt/refine/link.py
"""Stage 3: tracklet linking (spec 6.3), including the legacy ``link_tracklets``.

``link_tracklets`` moved here from ``dnt.track.post_process`` (which re-exports it) in dnt 0.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

LEGACY_COL_NAMES = ["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]


def _iou_xywh(a, b) -> float:
    ...  # verbatim from the nested definition


def _estimate_velocity(frames, cx, cy, k) -> tuple[float, float]:
    ...  # verbatim from the nested definition


class _DSU:
    ...  # verbatim from the nested definition


def _prepare_legacy(tracks: pd.DataFrame, col_names: list[str]) -> pd.DataFrame:
    df = tracks.copy()
    required = ["frame", "track", "x", "y", "w", "h"]
    if not all(c in df.columns for c in required):
        if len(df.columns) < 6:
            raise ValueError("tracks must include at least frame/track/x/y/w/h columns.")
        df.columns = col_names[: len(df.columns)]
    if "cls" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "cls"})
    if "interp" not in df.columns:
        df["interp"] = 0
    else:
        df["interp"] = pd.to_numeric(df["interp"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["frame", "track"]).reset_index(drop=True)
    df["cx"] = df["x"].astype(float) + (df["w"].astype(float) / 2.0)
    df["cy"] = df["y"].astype(float) + (df["h"].astype(float) / 2.0)
    df["area"] = df["w"].astype(float) * df["h"].astype(float)
    return df


def _legacy_descriptors(df: pd.DataFrame, vel_frames: int, pbar=None) -> dict[int, dict]:
    descriptors: dict[int, dict] = {}
    for tid, g in df.groupby("track", sort=False):
        g_real = g[g["interp"] != 1].sort_values("frame")
        if pbar is not None:
            pbar.update(1)
        if len(g_real) < 2:
            descriptors[int(tid)] = {"stitchable": False}
            continue
        start_row, end_row = g_real.iloc[0], g_real.iloc[-1]
        vx, vy = _estimate_velocity(
            g_real["frame"].to_numpy(), g_real["cx"].to_numpy(), g_real["cy"].to_numpy(), vel_frames
        )
        descriptors[int(tid)] = {
            "stitchable": True,
            "track": int(tid),
            "cls": int(end_row["cls"]) if "cls" in g_real.columns else -1,
            "t_start": int(g_real["frame"].iloc[0]),
            "t_end": int(g_real["frame"].iloc[-1]),
            "start_c": (float(start_row["cx"]), float(start_row["cy"])),
            "end_c": (float(end_row["cx"]), float(end_row["cy"])),
            "start_box": (float(start_row["x"]), float(start_row["y"]),
                          float(start_row["w"]), float(start_row["h"])),
            "end_box": (float(end_row["x"]), float(end_row["y"]),
                        float(end_row["w"]), float(end_row["h"])),
            "area_end": max(float(end_row["area"]), 1.0),
            "vx": vx,
            "vy": vy,
        }
    return descriptors


def _legacy_gate_cost(
    a: dict, b: dict, *, max_gap: int, size_ratio_max: float, dist_mult: float, iou_min: float,
    w_d: float, w_iou: float, w_s: float, dist_growth: float = 0.03, check_class: bool = True,
    detail: bool = False,
):
    """Return ``link_tracklets``'s cost for linking end ``a`` to start ``b``, or None if gated."""
    if a["track"] == b["track"]:
        return None
    dt = b["t_start"] - a["t_end"]
    if dt < 1 or dt > max_gap:
        return None
    if check_class and a["cls"] != b["cls"]:
        return None
    wi, hi = max(a["end_box"][2], 1.0), max(a["end_box"][3], 1.0)
    wj, hj = max(b["start_box"][2], 1.0), max(b["start_box"][3], 1.0)
    w_ratio, h_ratio = wj / wi, hj / hi
    if not (1.0 / size_ratio_max <= w_ratio <= size_ratio_max):
        return None
    if not (1.0 / size_ratio_max <= h_ratio <= size_ratio_max):
        return None
    pred_cx = a["end_c"][0] + a["vx"] * dt
    pred_cy = a["end_c"][1] + a["vy"] * dt
    sx, sy = b["start_c"]
    dist = float(np.hypot(pred_cx - sx, pred_cy - sy))
    if dist >= dist_mult * np.sqrt(a["area_end"]) * (1.0 + (dist_growth * dt)):
        return None
    iou = _iou_xywh((pred_cx - (wi / 2.0), pred_cy - (hi / 2.0), wi, hi), b["start_box"])
    if iou < iou_min:
        return None
    dist_norm = dist / (np.sqrt(a["area_end"]) + 1e-6)
    size_cost = abs(np.log(max(w_ratio, 1e-6))) + abs(np.log(max(h_ratio, 1e-6)))
    cost = (w_d * dist_norm) + (w_iou * (1.0 - iou)) + (w_s * size_cost)
    if detail:
        return cost, {"dist": dist, "iou_pred": iou, "w_ratio": w_ratio, "h_ratio": h_ratio}
    return cost


def _legacy_matches(stitchable: list[dict], **gate_kw) -> list[tuple[int, int, float]]:
    """Return ``link_tracklets``'s Hungarian matches as ``(end_track, start_track, cost)``."""
    ends = sorted(stitchable, key=lambda d: (d["t_end"], d["track"]))
    starts = sorted(stitchable, key=lambda d: (d["t_start"], d["track"]))
    inf = 1e9
    cost = np.full((len(ends), len(starts)), inf, dtype=float)
    for i, a in enumerate(ends):
        for j, b in enumerate(starts):
            c = _legacy_gate_cost(a, b, **gate_kw)
            if c is not None:
                cost[i, j] = c
    matches: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(ri, ci, strict=True) if cost[r, c] < inf]
    except Exception:
        used_r: set[int] = set()
        used_c: set[int] = set()
        pairs = sorted(np.argwhere(cost < inf), key=lambda rc: float(cost[rc[0], rc[1]]))
        for r, c in pairs:
            if int(r) in used_r or int(c) in used_c:
                continue
            used_r.add(int(r))
            used_c.add(int(c))
            matches.append((int(r), int(c)))
    return [(int(ends[r]["track"]), int(starts[c]["track"]), float(cost[r, c])) for r, c in matches]


def link_tracklets(
    tracks: pd.DataFrame | None = None,
    track_file: str | None = None,
    output_file: str | None = None,
    col_names: list[str] | None = None,
    max_gap: int = 20,
    vel_frames: int = 5,
    size_ratio_max: float = 2.0,
    dist_mult: float = 2.5,
    iou_min: float = 0.05,
    w_d: float = 1.0,
    w_iou: float = 1.0,
    w_s: float = 0.3,
    verbose: bool = True,
    video_index: int | None = None,
    video_tot: int | None = None,
) -> pd.DataFrame:
    """<paste the original link_tracklets docstring here, unchanged>"""
    if col_names is None:
        col_names = list(LEGACY_COL_NAMES)
    if tracks is None:
        if not track_file:
            raise ValueError("Either `tracks` or `track_file` must be provided.")
        tracks = pd.read_csv(track_file, header=None)
    if len(tracks) == 0:
        out = tracks.copy()
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out
    df = _prepare_legacy(tracks, col_names)
    n_tracks = df["track"].nunique()
    pbar = tqdm(total=n_tracks, unit=" tracklets", disable=not verbose)
    if verbose:
        if video_index is not None and video_tot is not None:
            pbar.set_description_str(f"Link tracklets {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Link tracklets")
    descriptors = _legacy_descriptors(df, vel_frames, pbar)
    pbar.close()
    stitchable = [d for d in descriptors.values() if d.get("stitchable", False)]
    if len(stitchable) <= 1:
        out = df.drop(columns=["cx", "cy", "area"])
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out
    matches = _legacy_matches(
        stitchable, max_gap=max_gap, size_ratio_max=size_ratio_max, dist_mult=dist_mult,
        iou_min=iou_min, w_d=w_d, w_iou=w_iou, w_s=w_s,
    )
    dsu = _DSU([int(d["track"]) for d in stitchable])
    for a_tid, b_tid, _ in matches:
        dsu.union(a_tid, b_tid)
    comps: dict[int, list[int]] = {}
    for d in stitchable:
        comps.setdefault(dsu.find(int(d["track"])), []).append(int(d["track"]))
    tstart_by_tid = {int(d["track"]): int(d["t_start"]) for d in stitchable}
    rep_map: dict[int, int] = {}
    for members in comps.values():
        rep = min(members, key=lambda t: (tstart_by_tid.get(t, 10**9), t))
        for t in members:
            rep_map[t] = rep
    for tid in df["track"].astype(int).unique().tolist():
        rep_map.setdefault(int(tid), int(tid))
    out = df.copy()
    out["track"] = out["track"].astype(int).map(rep_map).astype(int)
    out = out.drop(columns=["cx", "cy", "area"]).sort_values(["frame", "track"])
    out = out.reset_index(drop=True)
    if output_file:
        out.to_csv(output_file, index=False, header=False)
    return out
```

Replace the three `...  # verbatim …` bodies and both `"""<paste …>"""` docstrings with the original text before running the tests.

- [ ] **Step 4: Make `post_process.py` a pure shim; drop its lint baseline**

```python
# src/dnt/track/post_process.py  (entire file)
"""Backward-compatible home of the track post-processing functions (moved to ``dnt.refine``)."""

from ..refine.interpolate import interpolate_tracks_rts
from ..refine.link import link_tracklets

__all__ = ["interpolate_tracks_rts", "link_tracklets"]
```

In `pyproject.toml`, delete the line `"src/dnt/track/post_process.py" = ["E501"]`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/refine tests/test_post_process.py tests/test_filter.py tests/test_pipeline_cpu.py tests/test_imports.py tests/test_refine_independence.py -v && .venv/bin/ruff check src tests tools`
Expected: all PASS, and the link baselines are byte-identical. Ruff is clean with the smaller baseline.

- [ ] **Step 6: Commit**

```bash
git add src/dnt/refine/link.py src/dnt/track/post_process.py pyproject.toml tests/refine/test_link_legacy.py
git commit -m "refactor: move link_tracklets to dnt.refine.link with shared legacy gate helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 6: Track, context, and video I/O

**Files:**
- Create: `src/dnt/refine/io.py`
- Test: `tests/refine/test_io.py`

**Interfaces:**
- Produces (in `dnt.refine.io`):
  - `TRACK_COLUMNS`, `OUT_COLUMNS`, `WORK_COLUMNS`, `CONTEXT_COLUMNS`
  - `TrackInput(work: pd.DataFrame, n_filled_removed: int, n_duplicates_removed: int)`, a frozen dataclass
  - `empty_work() -> pd.DataFrame`
  - `to_work(df, *, source="tracks") -> TrackInput`
  - `read_tracks(path, *, fmt="dnt", class_id=0) -> TrackInput`
  - `read_context(path, fmt="auto") -> tuple[pd.DataFrame, str]`
  - `write_tracks(work, path) -> None`
  - `sha256_file(path) -> str`
  - `video_info(path) -> dict` with keys `fps, frame_count, width, height`
  - `video_fingerprint(path, frame_count) -> dict` with keys `sha256, size, frame_count`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_io.py
import hashlib
import logging

import pytest

from dnt.refine import io

from ._fixtures import box_rows, table


def _write(tmp_path, df, name="t.txt"):
    p = tmp_path / name
    df.to_csv(p, index=False, header=False)
    return p


def test_read_tracks_removes_filled_rows_and_builds_work_table(tmp_path):
    df = table(box_rows(7, range(5), 10.0, 20.0))
    df.loc[2, "r3"] = 1
    tin = io.read_tracks(_write(tmp_path, df))
    assert tin.n_filled_removed == 1
    assert list(tin.work.columns) == io.WORK_COLUMNS
    assert tin.work["frame"].tolist() == [0, 1, 3, 4]
    assert (tin.work["raw_id"] == 7).all() and (tin.work["interp"] == 0).all()


def test_duplicate_track_frame_rows_keep_first(tmp_path, caplog):
    df = table(box_rows(1, [0, 1, 1, 2], 10.0, 20.0))
    df.loc[2, "x"] = 999.0
    with caplog.at_level(logging.WARNING):
        tin = io.read_tracks(_write(tmp_path, df))
    assert tin.n_duplicates_removed == 1
    assert 999.0 not in tin.work["x"].tolist()
    assert "duplicate" in caplog.text


def test_read_tracks_errors(tmp_path):
    (tmp_path / "few.txt").write_text("1,2,3,4,5\n")
    with pytest.raises(ValueError, match="at least 6 columns"):
        io.read_tracks(tmp_path / "few.txt")
    (tmp_path / "bad.txt").write_text("0,1,1,1,1,1,0.9,0,-1,-1\n1,1,abc,1,1,1,0.9,0,-1,-1\n")
    with pytest.raises(ValueError, match="line 2"):
        io.read_tracks(tmp_path / "bad.txt")


def test_read_tracks_empty_file(tmp_path):
    (tmp_path / "e.txt").write_text("")
    tin = io.read_tracks(tmp_path / "e.txt")
    assert tin.work.empty and list(tin.work.columns) == io.WORK_COLUMNS


def test_read_mot_sets_class(tmp_path):
    (tmp_path / "m.txt").write_text("1,4,10,20,30,60,0.8,-1,-1,-1\n2,4,11,20,30,60,0.7,-1,-1,-1\n")
    tin = io.read_tracks(tmp_path / "m.txt", fmt="mot", class_id=0)
    assert tin.work["cls"].tolist() == [0, 0] and tin.work["score"].tolist() == [0.8, 0.7]


def test_read_context_detects_format(tmp_path):
    (tmp_path / "d.txt").write_text("0,-1,1,2,3,4,0.9,2\n")
    (tmp_path / "t.txt").write_text("0,5,1,2,3,4,0.9,7,-1,-1\n")
    (tmp_path / "x.txt").write_text("0,5,1,2,3,4,0.9\n")
    d, fd = io.read_context(tmp_path / "d.txt")
    t, ft = io.read_context(tmp_path / "t.txt")
    assert (fd, d["track"].tolist(), d["cls"].tolist()) == ("dets", [-1], [2])
    assert (ft, t["track"].tolist(), t["cls"].tolist()) == ("tracks", [5], [7])
    with pytest.raises(ValueError, match="7 columns"):
        io.read_context(tmp_path / "x.txt")


def test_write_tracks_format(tmp_path):
    work = io.read_tracks(_write(tmp_path, table(box_rows(2, [1, 0], 10.4, 20.6)))).work
    out = tmp_path / "o.txt"
    io.write_tracks(work, out)
    lines = out.read_text().splitlines()
    assert lines == ["0,2,10,21,30,60,0.9,0,0,-1", "1,2,10,21,30,60,0.9,0,0,-1"]
    io.write_tracks(io.empty_work(), tmp_path / "empty.txt")
    assert (tmp_path / "empty.txt").read_text() == ""


def test_sha256_and_video_info(tmp_path, synthetic_video):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert io.sha256_file(p) == hashlib.sha256(b"abc" * 1000).hexdigest()
    video, _ = synthetic_video
    info = io.video_info(video)
    assert info == {"fps": pytest.approx(25.0), "frame_count": 150, "width": 320, "height": 240}
    fp = io.video_fingerprint(video, info["frame_count"])
    assert fp["sha256"] == io.sha256_file(video) and fp["frame_count"] == 150
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'io' from 'dnt.refine'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/io.py
"""Track, context, and video I/O for dnt.refine (spec 2.5)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRACK_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]
OUT_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]
WORK_COLUMNS = [*OUT_COLUMNS, "raw_id"]
CONTEXT_COLUMNS = ["frame", "track", "x", "y", "w", "h", "cls"]
_INT_OUT = ["frame", "track", "x", "y", "w", "h", "cls", "interp", "r4"]
_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class TrackInput:
    """A work table plus counts of the input rows that were removed."""

    work: pd.DataFrame
    n_filled_removed: int
    n_duplicates_removed: int


def empty_work() -> pd.DataFrame:
    """Return an empty work table with the standard columns."""
    floats = {"x", "y", "w", "h", "score"}
    return pd.DataFrame({c: pd.Series(dtype=float if c in floats else int) for c in WORK_COLUMNS})


def _read_numeric_csv(path, min_cols: int) -> pd.DataFrame:
    path = Path(path)
    try:
        raw = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if raw.shape[1] < min_cols:
        raise ValueError(f"{path}: expected at least {min_cols} columns, found {raw.shape[1]}")
    num = raw.apply(pd.to_numeric, errors="coerce")
    bad = num.iloc[:, :min_cols].isna().any(axis=1).to_numpy()
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        text = ",".join(raw.iloc[i].fillna("").tolist())
        raise ValueError(f"{path}: non-numeric value on line {i + 1}: {text}")
    return num


def to_work(df: pd.DataFrame, *, source: str = "tracks") -> TrackInput:
    """Build a work table from a raw 6-10 column track table (positional or named)."""
    df = df.copy()
    if not all(c in df.columns for c in ("frame", "track", "x", "y", "w", "h")):
        df = df.iloc[:, : len(TRACK_COLUMNS)]
        df.columns = TRACK_COLUMNS[: df.shape[1]]
    for c, default in (("score", -1.0), ("cls", -1), ("r3", -1), ("r4", -1)):
        if c not in df.columns:
            df[c] = default
    filled = pd.to_numeric(df["r3"], errors="coerce").fillna(-1) == 1
    n_filled = int(filled.sum())
    df = df.loc[~filled]
    dup = df.duplicated(["track", "frame"], keep="first")
    n_dup = int(dup.sum())
    if n_dup:
        log.warning("%s: %d duplicate (track, frame) rows; kept the first of each", source, n_dup)
    df = df.loc[~dup]
    work = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "track": df["track"].astype(int),
        "x": df["x"].astype(float),
        "y": df["y"].astype(float),
        "w": df["w"].astype(float),
        "h": df["h"].astype(float),
        "score": pd.to_numeric(df["score"], errors="coerce").fillna(-1.0).astype(float),
        "cls": pd.to_numeric(df["cls"], errors="coerce").fillna(-1).astype(int),
        "interp": 0,
        "r4": pd.to_numeric(df["r4"], errors="coerce").fillna(-1).astype(int),
        "raw_id": df["track"].astype(int),
    })
    work = work.sort_values(["track", "frame"]).reset_index(drop=True)
    return TrackInput(work=work, n_filled_removed=n_filled, n_duplicates_removed=n_dup)


def read_tracks(path, *, fmt: str = "dnt", class_id: int = 0) -> TrackInput:
    """Read a dnt (10-column) or MOTChallenge track file into a work table (spec 2.5)."""
    raw = _read_numeric_csv(path, min_cols=6)
    if raw.empty:
        return TrackInput(work=empty_work(), n_filled_removed=0, n_duplicates_removed=0)
    if fmt == "dnt":
        df = raw.iloc[:, : len(TRACK_COLUMNS)].copy()
    elif fmt == "mot":
        df = pd.DataFrame({
            "frame": raw[0], "track": raw[1], "x": raw[2], "y": raw[3], "w": raw[4], "h": raw[5],
            "score": raw[6] if raw.shape[1] > 6 else -1.0, "cls": class_id, "r3": -1, "r4": -1,
        })
    else:
        raise ValueError(f"unknown track format {fmt!r}; expected 'dnt' or 'mot'")
    return to_work(df, source=str(path))


def read_context(path, fmt: str = "auto") -> tuple[pd.DataFrame, str]:
    """Read a context file (dnt tracks or detections) as boxes with classes (spec 2.5)."""
    raw = _read_numeric_csv(path, min_cols=6)
    if raw.empty:
        return pd.DataFrame(columns=CONTEXT_COLUMNS), ("tracks" if fmt == "auto" else fmt)
    ncol = raw.shape[1]
    if fmt == "auto":
        if ncol == 10:
            fmt = "tracks"
        elif ncol == 8:
            fmt = "dets"
        else:
            raise ValueError(
                f"{path}: context file has {ncol} columns; expected 8 (detections) or 10 (tracks)"
            )
    if fmt not in ("tracks", "dets"):
        raise ValueError(f"unknown context format {fmt!r}; expected 'auto', 'tracks' or 'dets'")
    if ncol < 8:
        raise ValueError(f"{path}: context file has {ncol} columns; the class is column 8")
    ctx = pd.DataFrame({
        "frame": raw[0].astype(int),
        "track": raw[1].astype(int) if fmt == "tracks" else -1,
        "x": raw[2].astype(float),
        "y": raw[3].astype(float),
        "w": raw[4].astype(float),
        "h": raw[5].astype(float),
        "cls": raw[7].astype(int),
    })
    return ctx, fmt


def write_tracks(work: pd.DataFrame, path) -> None:
    """Write a work table as a headerless 10-column dnt track file sorted by frame, track."""
    out = work.reindex(columns=OUT_COLUMNS).copy()
    for c in _INT_OUT:
        out[c] = pd.to_numeric(out[c]).fillna(-1).round().astype(int)
    out["score"] = pd.to_numeric(out["score"]).fillna(-1.0).astype(float)
    out = out.sort_values(["frame", "track"], kind="mergesort")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, header=False)


def sha256_file(path) -> str:
    """Return the SHA-256 of a whole file, read in 8 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def video_info(path) -> dict:
    """Return ``fps``, ``frame_count``, ``width`` and ``height`` of a video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def video_fingerprint(path, frame_count: int) -> dict:
    """Return the video fingerprint: whole-file SHA-256, size, and frame count (spec 5.3)."""
    p = Path(path)
    return {"sha256": sha256_file(p), "size": p.stat().st_size, "frame_count": int(frame_count)}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_io.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/io.py tests/refine/test_io.py
git commit -m "feat(refine): add track/context/video I/O with filled-row removal

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `RefineConfig`

**Files:**
- Create: `src/dnt/refine/config.py`
- Test: `tests/refine/test_config.py`

**Interfaces:**
- Produces (in `dnt.refine.config`):
  - `to_frames(seconds: float, fps: float) -> int`
  - The dataclasses `ContextConfig, HintsConfig, MotionConfig, EncoderConfig, SwitchConfig, ScreenConfig, LinkConfig, OrphanConfig, FillConfig, VLMConfig`, each with the fields below.
  - `RefineConfig`, with `defaults(target="person")`, `from_dict(data)`, `from_yaml(path)`, `to_dict()`, `to_yaml(path)`, and `validate()`.

The field names and defaults are part of the interface for every later task and plan. Fields not in the spec's §9 YAML come from §6's prose:

| Field | Spec source |
|---|---|
| `switch.delta` 0.5 | §6.1 "δ = 0.5 s" |
| `switch.nms_seconds` 1.0 | §6.1 "spacing ≥ 1 s" |
| `switch.contact_iou` 0.1 | §6.1 gate 1 |
| `switch.mad_floor` 0.02 | plan correction |
| `switch.bimodal_purity` 0.9 | §6.1 "at least 90%" |
| `switch.bimodal_silhouette_min` 0.25 | §6.1 |
| `switch.swap_boost` 0.2 | §6.1 |
| `switch.ramps` | §6.1 |
| `screen.in_vehicle_iob` 0.8 | §6.2 |
| `screen.move_together` 0.3 | §6.2 |
| `screen.rider_speed` 1.8 | §6.2 |
| `screen.twowheeler_iou` 0.3 | §6.2 |
| `screen.duplicate_iob` 0.7 | §6.2 |
| `screen.duplicate_min_frames` 10 | §6.2 |
| `screen.hotspot_radius` 0.5 | §6.2 |
| `screen.persistence_iou` 0.5 | §6.2 |
| `screen.ramps` | §6.2 table |
| `motion.moving_min` 0.3 | §6.2 persistence and "moving frames" |
| `link.static_speed` 0.2, `static_seconds` 0.5, `static_radius` 0.5 | §6.3 gate 1 |
| `link.overlap_frames` 2, `overlap_iou` 0.5 | §6.3 gate 2 |
| `link.k_embed` 5 | §6.3 `c_app` |
| `link.border_margin` 0.5 | §6.3 prior |
| `link.speed_seconds` 1.0 | §6.3 gate 9 |
| `link.heading_min_speed` 0.2 | §6.3 gate 8 |
| `link.n_alternatives` 2 | §6.3 |
| `orphan.ramp` [0.5, 0.1] | §6.2 orphan |

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_config.py
import pytest
import yaml

from dnt.refine.config import RefineConfig, to_frames


def test_to_frames():
    assert to_frames(1.0, 25) == 25 and to_frames(0.01, 10) == 1 and to_frames(0.5, 10) == 5


def test_target_defaults():
    p, v = RefineConfig.defaults("person"), RefineConfig.defaults("vehicle")
    assert p.class_ids == [0] and p.link.class_groups == []
    assert v.class_ids == [2, 5, 7] and v.link.class_groups == [[2, 7]]
    assert p.switch.size_gate == 1.5 and p.link.weights == {"mot": 0.45, "app": 0.40, "gap": 0.15}
    assert p.screen.ramps["R"] == [0.3, 0.05] and p.orphan.ramp == [0.5, 0.1]


def test_yaml_round_trip_keeps_int_keys(tmp_path):
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.max_gap = 2.0
    cfg.to_yaml(tmp_path / "c.yaml")
    back = RefineConfig.from_yaml(tmp_path / "c.yaml")
    assert back == cfg and 36 in back.hints.reclass_class_map


def test_partial_yaml_overlays_target_defaults(tmp_path):
    (tmp_path / "v.yaml").write_text("target: vehicle\nlink:\n  accept_above: 0.85\n")
    cfg = RefineConfig.from_yaml(tmp_path / "v.yaml")
    assert cfg.link.accept_above == 0.85 and cfg.link.class_groups == [[2, 7]]
    cfg2 = RefineConfig.from_dict({"screen": {"ramps": {"R": [0.4, 0.1]}}})
    assert cfg2.screen.ramps["R"] == [0.4, 0.1] and cfg2.screen.ramps["J"] == [0.03, 0.005]


@pytest.mark.parametrize("data, match", [
    ({"link": {"bogus": 1}}, "link.bogus"),
    ({"screen": {"ramps": {"Q": [0, 1]}}}, "screen.ramps.Q"),
    ({"switch": {"accept_above": 0.4, "reject_below": 0.5}}, "switch"),
    ({"screen": {"static_score_cap": 0.9}}, "static_score_cap"),
    ({"link": {"weights": {"mot": 0.5, "app": 0.5, "gap": 0.5}}}, "link.weights"),
    ({"link": {"occluded_score_cap": 0.9}}, "occluded_score_cap"),
    ({"link": {"ambiguous_cap": 0.9}}, "ambiguous_cap"),
    ({"link": {"max_gap": 9.0}}, "max_gap_occluded"),
    ({"link": {"class_groups": [[2, 7], [7, 5]]}}, "class_groups"),
    ({"screen": {"mixed_score_cap": 0.9}}, "mixed_score_cap"),
    ({"screen": {"segment_at": 0.6}}, "segment_at"),
    ({"target": "vehicle", "encoder": {"kind": "reid"}}, "weights"),
    ({"vlm": {"backend": "openai_compat"}}, "vlm.model"),
    ({"encoder": {"kind": "clip"}}, "encoder.kind"),
    ({"fps": 0}, "fps"),
    ({"screen": {"ramps": {"R": [0.3, 0.3]}}}, "screen.ramps.R"),
])
def test_validation_rules(data, match):
    with pytest.raises(ValueError, match=match):
        RefineConfig.from_dict(data)


def test_anthropic_backend_needs_no_model():
    RefineConfig.from_dict({"vlm": {"backend": "anthropic"}})


def test_from_yaml_rejects_non_mapping(tmp_path):
    (tmp_path / "l.yaml").write_text(yaml.safe_dump([1, 2]))
    with pytest.raises(ValueError, match="mapping"):
        RefineConfig.from_yaml(tmp_path / "l.yaml")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnt.refine.config'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/config.py
"""``RefineConfig``: every refinement setting, with per-target defaults (spec 9)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

TARGETS = ("person", "vehicle")
ENCODERS = ("dino", "reid", "none")
BACKENDS = ("none", "openai_compat", "anthropic")
_FIXED_KEY_DICTS = {"ramps", "weights", "weights_occluded", "legacy_weights", "reclass_map"}


def to_frames(seconds: float, fps: float) -> int:
    """Convert a duration in seconds to a whole number of frames (at least 1)."""
    return max(1, round(seconds * fps))


@dataclass(kw_only=True)
class ContextConfig:
    """Context file settings (spec 2.5)."""

    format: str = "auto"
    vehicle_classes: list[int] = field(default_factory=lambda: [2, 5, 7])
    twowheeler_classes: list[int] = field(default_factory=lambda: [1, 3])


@dataclass(kw_only=True)
class HintsConfig:
    """ReClass hint settings (spec 6.2)."""

    reclass_class_map: dict[int, str] = field(
        default_factory=lambda: {1: "cyclist", 3: "motorcycle", 36: "scooter"}
    )
    reclass_ramp: list[float] = field(default_factory=lambda: [0.75, 0.9])
    subtype_min: float = 1.0

    def __post_init__(self):
        """Normalize class keys to int (YAML and JSON may deliver strings)."""
        self.reclass_class_map = {int(k): str(v) for k, v in self.reclass_class_map.items()}


@dataclass(kw_only=True)
class MotionConfig:
    """Shared motion model settings (spec 5.1, 5.2)."""

    height_window: int = 15
    process_var: float = 10.0
    meas_var_pos: float = 25.0
    meas_var_size: float = 16.0
    moving_min: float = 0.3


@dataclass(kw_only=True)
class EncoderConfig:
    """Appearance encoder settings (spec 5.3, 5.5)."""

    kind: str = "dino"
    model: str = "facebook/dinov2-small"
    weights: str | None = None
    device: str = "auto"
    sample_every: int = 5
    occlusion_iou: float = 0.3
    batch_size: int = 64


@dataclass(kw_only=True)
class SwitchConfig:
    """Stage 1 settings (spec 6.1)."""

    enabled: bool = True
    accept_above: float = 0.90
    reject_below: float = 0.50
    window: float = 1.0
    min_side_seconds: float = 0.5
    nis_hi: float = 18.47
    w_app: float = 0.65
    w_mot: float = 0.35
    motion_only_cap: float = 0.70
    class_change_gate: bool = True
    size_gate: float = 1.5
    delta: float = 0.5
    nms_seconds: float = 1.0
    contact_iou: float = 0.1
    mad_floor: float = 0.02
    bimodal_purity: float = 0.9
    bimodal_silhouette_min: float = 0.25
    swap_boost: float = 0.2
    ramps: dict[str, list[float]] = field(default_factory=lambda: {
        "z_app": [2.0, 5.0], "silhouette": [0.25, 0.5], "jump": [0.15, 0.4], "cross": [0.0, 0.4],
    })


@dataclass(kw_only=True)
class ScreenConfig:
    """Stage 2 settings (spec 6.2)."""

    enabled: bool = True
    accept_above: float = 0.85
    reject_below: float = 0.40
    static_score_cap: float = 0.80
    vehicle_static_score_cap: float = 0.60
    mixed_score_cap: float = 0.75
    segment_at: float = 0.30
    in_vehicle_iob: float = 0.8
    move_together: float = 0.3
    rider_speed: float = 1.8
    twowheeler_iou: float = 0.3
    duplicate_iob: float = 0.7
    duplicate_min_frames: int = 10
    hotspot_radius: float = 0.5
    persistence_iou: float = 0.5
    ramps: dict[str, list[float]] = field(default_factory=lambda: {
        "R": [0.3, 0.05], "J": [0.03, 0.005], "C": [0.6, 0.3], "T": [2.0, 10.0],
        "H": [1.0, 4.0], "inside": [0.5, 0.9], "F": [0.3, 0.7], "S": [0.5, 0.9],
        "K": [0.2, 0.6], "D": [0.5, 0.9],
    })


@dataclass(kw_only=True)
class LinkConfig:
    """Stage 3 settings (spec 6.3)."""

    enabled: bool = True
    mode: str = "scored"
    accept_above: float = 0.80
    reject_below: float = 0.40
    max_gap: float = 1.0
    max_gap_static: float = 10.0
    max_gap_occluded: float = 8.0
    witness_iob: float = 0.5
    witness_min: float = 0.7
    max_heading_change: float = 120.0
    speed_factor: float = 1.5
    min_feasible_speed: float = 0.5
    occluded_score_cap: float = 0.75
    margin_min: float = 0.10
    ambiguous_cap: float = 0.75
    max_passes: int = 3
    weights_occluded: dict[str, float] = field(
        default_factory=lambda: {"mot": 0.25, "app": 0.60, "gap": 0.15}
    )
    class_groups: list[list[int]] = field(default_factory=list)
    size_ratio_max: float = 2.0
    dist_mult: float = 2.5
    dist_growth: float = 0.03
    iou_min: float = 0.05
    vel_frames: int = 5
    legacy_weights: dict[str, float] = field(
        default_factory=lambda: {"d": 1.0, "iou": 1.0, "s": 0.3}
    )
    legacy_cost_hi: float = 3.0
    weights: dict[str, float] = field(
        default_factory=lambda: {"mot": 0.45, "app": 0.40, "gap": 0.15}
    )
    static_speed: float = 0.2
    static_seconds: float = 0.5
    static_radius: float = 0.5
    overlap_frames: int = 2
    overlap_iou: float = 0.5
    k_embed: int = 5
    border_margin: float = 0.5
    speed_seconds: float = 1.0
    heading_min_speed: float = 0.2
    n_alternatives: int = 2


@dataclass(kw_only=True)
class OrphanConfig:
    """Orphan pass settings (spec 6.2)."""

    enabled: bool = True
    min_seconds: float = 0.5
    accept_above: float = 0.70
    reject_below: float = 0.30
    ramp: list[float] = field(default_factory=lambda: [0.5, 0.1])


@dataclass(kw_only=True)
class FillConfig:
    """Stage 4 settings (spec 6.4)."""

    enabled: bool = True
    max_gap: float | None = None
    smooth_existing: bool = False


@dataclass(kw_only=True)
class VLMConfig:
    """VLM verification settings (spec 7); used from Plan 3 on."""

    backend: str = "none"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    json_mode: bool = True
    min_conf: float = 0.7
    votes: int = 1
    vote_temperature: float = 0.7
    max_calls: int = 500
    max_concurrency: int = 4
    timeout_s: float = 60.0
    send_context_frames: bool = True
    cache_dir: str = "~/.cache/dnt/vlm"


@dataclass(kw_only=True)
class RefineConfig:
    """All refinement settings for one target (person or vehicle)."""

    target: str = "person"
    class_ids: list[int] = field(default_factory=lambda: [0])
    fps: float | None = None
    reclass_map: dict[str, int] = field(
        default_factory=lambda: {"cyclist": 1, "motorcycle": 3, "scooter": 36}
    )
    frame_size: list[int] | None = None
    context: ContextConfig = field(default_factory=ContextConfig)
    hints: HintsConfig = field(default_factory=HintsConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    orphan: OrphanConfig = field(default_factory=OrphanConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    @classmethod
    def defaults(cls, target: str = "person") -> RefineConfig:
        """Return the defaults for ``target`` (``person`` or ``vehicle``)."""
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
        cfg = cls(target=target)
        if target == "vehicle":
            cfg.class_ids = [2, 5, 7]
            cfg.link.class_groups = [[2, 7]]
        return cfg

    @classmethod
    def from_dict(cls, data: Mapping | None) -> RefineConfig:
        """Overlay ``data`` on the target's defaults; unknown keys raise ValueError."""
        data = dict(data or {})
        cfg = cls.defaults(data.get("target", "person"))
        _overlay(cfg, data, "")
        cfg.hints.reclass_class_map = {int(k): str(v)
                                       for k, v in cfg.hints.reclass_class_map.items()}
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path) -> RefineConfig:
        """Load a config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is not None and not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Return the config as plain Python data."""
        return asdict(self)

    def to_yaml(self, path) -> None:
        """Write the config to a YAML file."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def validate(self) -> None:
        """Raise ValueError listing every rule of spec 9 the config breaks."""
        p: list[str] = []
        if self.target not in TARGETS:
            p.append(f"target must be one of {TARGETS}")
        if self.encoder.kind not in ENCODERS:
            p.append(f"encoder.kind must be one of {ENCODERS}")
        if self.vlm.backend not in BACKENDS:
            p.append(f"vlm.backend must be one of {BACKENDS}")
        if self.link.mode not in ("scored", "legacy"):
            p.append("link.mode must be 'scored' or 'legacy'")
        if self.context.format not in ("auto", "tracks", "dets"):
            p.append("context.format must be 'auto', 'tracks' or 'dets'")
        for name in ("switch", "screen", "link", "orphan"):
            s = getattr(self, name)
            if not 0.0 <= s.reject_below < s.accept_above <= 1.0:
                p.append(f"{name}: need 0 <= reject_below < accept_above <= 1")
        sc, lc = self.screen, self.link
        if not sc.static_score_cap < sc.accept_above:
            p.append("screen.static_score_cap must be below screen.accept_above")
        if not sc.vehicle_static_score_cap < sc.accept_above:
            p.append("screen.vehicle_static_score_cap must be below screen.accept_above")
        if not sc.mixed_score_cap < sc.accept_above:
            p.append("screen.mixed_score_cap must be below screen.accept_above")
        if not sc.segment_at < self.switch.reject_below:
            p.append("screen.segment_at must be below switch.reject_below")
        for wname in ("weights", "weights_occluded"):
            if abs(sum(getattr(lc, wname).values()) - 1.0) > 1e-6:
                p.append(f"link.{wname} must sum to 1")
        if not lc.occluded_score_cap < lc.accept_above:
            p.append("link.occluded_score_cap must be below link.accept_above")
        if not lc.ambiguous_cap < lc.accept_above:
            p.append("link.ambiguous_cap must be below link.accept_above")
        if not lc.max_gap < lc.max_gap_occluded:
            p.append("link.max_gap must be below link.max_gap_occluded")
        seen: set[int] = set()
        for group in lc.class_groups:
            if seen & set(group):
                p.append("link.class_groups: a class appears in more than one group")
            seen |= set(group)
        if self.encoder.kind == "reid" and self.target == "vehicle" and not self.encoder.weights:
            p.append("encoder.weights is required for reid with the vehicle target")
        if self.vlm.backend not in ("none", "anthropic") and not self.vlm.model:
            p.append("vlm.model is required for this backend")
        ramps = {f"switch.ramps.{k}": v for k, v in self.switch.ramps.items()}
        ramps |= {f"screen.ramps.{k}": v for k, v in sc.ramps.items()}
        ramps |= {"orphan.ramp": self.orphan.ramp, "hints.reclass_ramp": self.hints.reclass_ramp}
        for name, (lo, hi) in ramps.items():
            if lo == hi:
                p.append(f"{name} needs lo != hi")
        if self.fps is not None and self.fps <= 0:
            p.append("fps must be positive")
        if self.frame_size is not None and (len(self.frame_size) != 2
                                            or min(self.frame_size) <= 0):
            p.append("frame_size must be [width, height] with positive values")
        if not self.class_ids:
            p.append("class_ids must not be empty")
        if not set(self.hints.reclass_class_map.values()) <= set(self.reclass_map):
            p.append("hints.reclass_class_map values must be keys of reclass_map")
        if p:
            raise ValueError("invalid refine config: " + "; ".join(p))


def _overlay(obj, data: Mapping, path: str) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{path}{key}' must be a mapping")
            _overlay(current, value, f"{path}{key}.")
        elif key in _FIXED_KEY_DICTS and isinstance(current, dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{path}{key}' must be a mapping")
            for sub in value:
                if sub not in current:
                    raise ValueError(f"unknown config key '{path}{key}.{sub}'")
            setattr(obj, key, {**current, **{k: (list(v) if isinstance(v, list | tuple) else v)
                                             for k, v in value.items()}})
        else:
            setattr(obj, key, value)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/config.py tests/refine/test_config.py
git commit -m "feat(refine): add RefineConfig with per-target defaults, strict YAML, validation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Events, proposal keys, and the ledger file

**Files:**
- Create: `src/dnt/refine/events.py`
- Test: `tests/refine/test_events.py`

**Interfaces:**
- Produces (in `dnt.refine.events`):
  - `EventKind(StrEnum)`: `DROP, RECLASS, SPLIT, LINK, FILL, SMOOTH`
  - `Decision(StrEnum)`: `AUTO_ACCEPT, AUTO_REJECT, VLM_ACCEPT, VLM_REJECT, HUMAN_PENDING, HUMAN_ACCEPT, HUMAN_REJECT`
  - `ACCEPTED`, `REJECTED` (frozensets of `Decision`)
  - `DEFINING_PARAMS: dict[EventKind, tuple[str, ...]]`
  - `clean_json(obj)`
  - `proposal_key(stage, kind, lineage, params) -> str`
  - `Event`, a dataclass with the spec §4.1 fields, where `decision` is `Decision | None` until routed, plus:
    - `Event.propose(*, stage, kind, tracks, lineage, frames, params, algo_score, signals=None, round=0) -> Event`
    - `to_dict()` and `from_dict(d)`
  - `Ledger(header: dict, events: list[Event])`, with `write(path)` and `Ledger.read(path)`
  - `assign_ids(events, stage, round, start=1) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_events.py
import math

from dnt.refine.events import Decision, Event, EventKind, Ledger, assign_ids, proposal_key


def _split(tracks=(4,), cut=821, lineage=((12, 780, 900),)):
    return Event.propose(stage="switch", kind=EventKind.SPLIT, tracks=list(tracks),
                         lineage=[[list(x) for x in lineage]], frames=(cut, cut),
                         params={"cut_frame": cut}, algo_score=0.7, signals={"mot": 1.0})


def test_key_ignores_track_numbering_but_not_the_cut():
    assert _split(tracks=(4,)).proposal_key == _split(tracks=(99,)).proposal_key
    assert _split(cut=821).proposal_key != _split(cut=822).proposal_key


def test_key_ignores_score_signals_and_edit():
    a = _split()
    b = _split()
    b.algo_score, b.signals, b.edit = 0.1, {"x": 1}, {"kind": "DROP", "params": {}}
    assert a.proposal_key == b.proposal_key
    k = proposal_key("screen", "DROP", [[[1, 0, 9]]], {"reason": "static", "spans": None})
    assert k != proposal_key("screen", "DROP", [[[1, 0, 9]]], {"reason": "in_vehicle",
                                                                "spans": None})


def test_round_trip_including_pending_and_nan(tmp_path):
    ev = _split()
    ev.signals["nis"] = float("nan")
    ev.decision = Decision.HUMAN_PENDING
    ev.decision_history.append({"decision": "HUMAN_PENDING", "round": 0, "source": "auto"})
    assign_ids([ev], "switch", 0)
    Ledger({"round": 0, "id_map": {3: 1}}, [ev]).write(tmp_path / "l.jsonl")
    header, *lines = (tmp_path / "l.jsonl").read_text().splitlines()
    assert '"round":0' in header and len(lines) == 1
    back = Ledger.read(tmp_path / "l.jsonl")
    got = back.events[0]
    assert got.id == "switch-r0-000001" and got.decision is Decision.HUMAN_PENDING
    assert got.frames == (821, 821) and got.kind is EventKind.SPLIT
    assert got.signals["nis"] is None and not math.isnan(got.algo_score)
    ev.signals["nis"] = None
    assert got == ev
    assert back.header["id_map"] == {"3": 1}


def test_write_is_deterministic(tmp_path):
    ev = _split()
    assign_ids([ev], "switch", 0)
    Ledger({"b": 1, "a": 2}, [ev]).write(tmp_path / "1.jsonl")
    Ledger({"a": 2, "b": 1}, [ev]).write(tmp_path / "2.jsonl")
    assert (tmp_path / "1.jsonl").read_bytes() == (tmp_path / "2.jsonl").read_bytes()


def test_assign_ids_continues_sequence():
    evs = [_split(), _split(cut=900)]
    assert assign_ids(evs, "switch", 1, start=5) == 7
    assert [e.id for e in evs] == ["switch-r1-000005", "switch-r1-000006"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnt.refine.events'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/events.py
"""Proposed edits, their decisions, and the JSONL ledger (spec 4.1, 4.2)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

import numpy as np


class EventKind(StrEnum):
    """What an event proposes to change."""

    DROP = "DROP"
    RECLASS = "RECLASS"
    SPLIT = "SPLIT"
    LINK = "LINK"
    FILL = "FILL"
    SMOOTH = "SMOOTH"


class Decision(StrEnum):
    """How an event was decided."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    AUTO_REJECT = "AUTO_REJECT"
    VLM_ACCEPT = "VLM_ACCEPT"
    VLM_REJECT = "VLM_REJECT"
    HUMAN_PENDING = "HUMAN_PENDING"
    HUMAN_ACCEPT = "HUMAN_ACCEPT"
    HUMAN_REJECT = "HUMAN_REJECT"


ACCEPTED = frozenset({Decision.AUTO_ACCEPT, Decision.VLM_ACCEPT, Decision.HUMAN_ACCEPT})
REJECTED = frozenset({Decision.AUTO_REJECT, Decision.VLM_REJECT, Decision.HUMAN_REJECT})
DEFINING_PARAMS: dict[EventKind, tuple[str, ...]] = {
    EventKind.SPLIT: ("cut_frame",),
    EventKind.DROP: ("reason", "spans", "of"),
    EventKind.RECLASS: ("new_cls", "spans"),
    EventKind.LINK: ("gap",),
    EventKind.FILL: ("gap",),
    EventKind.SMOOTH: (),
}


def clean_json(obj):
    """Return ``obj`` as strict-JSON-safe data (numpy -> Python, NaN/inf -> None)."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        items = sorted(obj) if isinstance(obj, set | frozenset) else obj
        return [clean_json(v) for v in items]
    if isinstance(obj, np.ndarray):
        return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _dumps(obj) -> str:
    return json.dumps(clean_json(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def proposal_key(stage: str, kind, lineage, params: dict) -> str:
    """Return the immutable key of a proposal (spec 4.2)."""
    k = EventKind(kind)
    defining = {name: params.get(name) for name in DEFINING_PARAMS[k]}
    return hashlib.sha256(_dumps([stage, k.value, lineage, defining]).encode()).hexdigest()


@dataclass
class Event:
    """One proposed edit, its decision, and its evidence (spec 4.1)."""

    id: str
    proposal_key: str
    round: int
    stage: str
    kind: EventKind
    tracks: list[int]
    lineage: list
    frames: tuple[int, int]
    params: dict
    edit: dict | None
    algo_score: float
    signals: dict
    decision: Decision | None
    decision_history: list[dict] = field(default_factory=list)
    vlm: dict | None = None
    applied: bool = False

    @classmethod
    def propose(
        cls, *, stage: str, kind, tracks, lineage, frames, params: dict, algo_score: float,
        signals: dict | None = None, round: int = 0,
    ) -> Event:
        """Create an undecided proposal and compute its key."""
        params = clean_json(dict(params))
        lineage = clean_json(lineage)
        return cls(
            id="", proposal_key=proposal_key(stage, kind, lineage, params), round=int(round),
            stage=stage, kind=EventKind(kind), tracks=[int(t) for t in tracks], lineage=lineage,
            frames=(int(frames[0]), int(frames[1])), params=params, edit=None,
            algo_score=float(algo_score), signals=clean_json(dict(signals or {})), decision=None,
        )

    @property
    def accepted(self) -> bool:
        """Whether the current decision accepts the event."""
        return self.decision in ACCEPTED

    def to_dict(self) -> dict:
        """Return the event as JSON-safe data."""
        return clean_json(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        """Rebuild an event written by ``to_dict``."""
        d = dict(d)
        d["kind"] = EventKind(d["kind"])
        d["decision"] = Decision(d["decision"]) if d.get("decision") else None
        d["frames"] = (int(d["frames"][0]), int(d["frames"][1]))
        return cls(**d)


@dataclass
class Ledger:
    """A header plus the events of one run (spec 4.2)."""

    header: dict
    events: list[Event]

    def write(self, path) -> None:
        """Write the header line and one line per event."""
        lines = [_dumps(self.header), *(_dumps(e.to_dict()) for e in self.events)]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path) -> Ledger:
        """Read a ledger written by ``write``."""
        lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        header = json.loads(lines[0])
        return cls(header=header, events=[Event.from_dict(json.loads(ln)) for ln in lines[1:]])


def assign_ids(events: list[Event], stage: str, round: int, start: int = 1) -> int:
    """Give unnumbered events IDs ``{stage}-r{round}-{seq:06d}``; return the next sequence."""
    seq = start
    for ev in events:
        if not ev.id:
            ev.id = f"{stage}-r{round}-{seq:06d}"
            seq += 1
    return seq
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_events.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean. (`round` shadows a builtin inside these signatures by design; it matches the spec's field name. If ruff flags `A002`, that rule is not selected, so no action is needed.)

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/events.py tests/refine/test_events.py
git commit -m "feat(refine): add events, immutable proposal keys, and the JSONL ledger

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Applying edits to the work table

**Files:**
- Create: `src/dnt/refine/apply.py`
- Test: `tests/refine/test_apply.py`

**Interfaces:**
- Consumes: `events.Event`, `events.EventKind`.
- Produces (in `dnt.refine.apply`). All of these preserve the DataFrame index except `renumber`.
  - `lineage_of_rows(rows) -> list[list[int]]`
  - `lineage(work, track) -> list[list[int]]`
  - `next_track_id(work) -> int`
  - `split_track(work, track, cut_frame, new_id)`
  - `drop_rows(work, track, spans=None)`
  - `reclass_rows(work, track, new_cls, spans=None)`
  - `merge_chains(work, pairs) -> tuple[pd.DataFrame, dict[int, int]]`
  - `renumber(work) -> tuple[pd.DataFrame, dict[int, int]]`
  - `apply_edit(work, event, *, new_id=None)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_apply.py
import pytest

from dnt.refine import apply as A
from dnt.refine import io
from dnt.refine.events import Event, EventKind

from ._fixtures import box_rows, table


def _work(*rows):
    return io.to_work(table(*rows)).work


def test_lineage_after_split_keeps_raw_ids():
    w = A.split_track(_work(box_rows(12, range(10, 20), 0.0, 0.0)), 12, 15, 13)
    assert A.lineage(w, 12) == [[12, 10, 14]] and A.lineage(w, 13) == [[12, 15, 19]]


def test_split_tail_id_never_collides_with_sparse_ids():
    w = _work(box_rows(3, range(5), 0.0, 0.0), box_rows(10004, range(5), 50.0, 0.0),
              box_rows(97, range(5), 99.0, 0.0))
    new = A.next_track_id(w)
    assert new == 10005
    w = A.split_track(w, 3, 2, new)
    assert sorted(w["track"].unique()) == [3, 97, 10004, 10005]
    assert (w.loc[w["track"] == 10005, "raw_id"] == 3).all()


def test_drop_and_reclass_with_spans_preserve_index():
    w = _work(box_rows(1, range(10), 0.0, 0.0))
    d = A.drop_rows(w, 1, [[5, 9]])
    assert d["frame"].tolist() == [0, 1, 2, 3, 4] and d.index.tolist() == [0, 1, 2, 3, 4]
    r = A.reclass_rows(w, 1, 3, [[0, 1]])
    assert r["cls"].tolist()[:3] == [3, 3, 0]
    assert A.drop_rows(w, 1).empty


def test_merge_chains_keeps_earliest_id_and_drops_overlap_rows():
    w = _work(box_rows(5, range(0, 10), 0.0, 0.0), box_rows(2, range(9, 20), 20.0, 0.0),
              box_rows(8, range(25, 30), 40.0, 0.0))
    m, rep = A.merge_chains(w, [(5, 2), (2, 8)])
    assert rep == {5: 5, 2: 5, 8: 5}
    assert m["frame"].tolist() == [*range(0, 20), *range(25, 30)]
    assert not m.duplicated(["track", "frame"]).any()


def test_renumber_is_contiguous_by_first_frame():
    w = _work(box_rows(50, range(5, 9), 0.0, 0.0), box_rows(7, range(0, 3), 0.0, 0.0))
    r, id_map = A.renumber(w)
    assert id_map == {7: 1, 50: 2} and sorted(r["track"].unique()) == [1, 2]


def test_apply_edit_uses_edit_not_proposal():
    w = _work(box_rows(1, range(10), 0.0, 0.0))
    ev = Event.propose(stage="screen", kind=EventKind.DROP, tracks=[1],
                       lineage=[A.lineage(w, 1)], frames=(0, 9),
                       params={"reason": "static", "spans": None}, algo_score=0.6)
    ev.edit = {"kind": "RECLASS", "params": {"new_cls": 1, "spans": None}}
    out = A.apply_edit(w, ev)
    assert len(out) == 10 and (out["cls"] == 1).all()
    ev.edit = None
    with pytest.raises(ValueError, match="no edit"):
        A.apply_edit(w, ev)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_apply.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/apply.py
"""The only code that edits the work table (spec 2.1). Every function preserves the index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .events import Event, EventKind


def lineage_of_rows(rows: pd.DataFrame) -> list[list[int]]:
    """Return ``[[raw_id, first_frame, last_frame], ...]`` for rows, ordered by first frame."""
    if rows.empty:
        return []
    g = rows.groupby("raw_id")["frame"].agg(["min", "max"]).reset_index().sort_values("min")
    return [[int(r), int(a), int(b)] for r, a, b in g[["raw_id", "min", "max"]].to_numpy()]


def lineage(work: pd.DataFrame, track: int) -> list[list[int]]:
    """Return the lineage of one track (spec 4.2)."""
    return lineage_of_rows(work.loc[work["track"] == track])


def next_track_id(work: pd.DataFrame) -> int:
    """Return an ID larger than every track ID in the table."""
    return int(work["track"].max()) + 1 if len(work) else 1


def _in_spans(frames: pd.Series, spans) -> np.ndarray:
    mask = np.zeros(len(frames), dtype=bool)
    for a, b in spans:
        mask |= ((frames >= a) & (frames <= b)).to_numpy()
    return mask


def split_track(work: pd.DataFrame, track: int, cut_frame: int, new_id: int) -> pd.DataFrame:
    """Give rows of ``track`` at or after ``cut_frame`` the ID ``new_id``."""
    out = work.copy()
    out.loc[(out["track"] == track) & (out["frame"] >= cut_frame), "track"] = int(new_id)
    return out


def drop_rows(work: pd.DataFrame, track: int, spans=None) -> pd.DataFrame:
    """Drop a track, or only its rows inside ``spans``."""
    mask = (work["track"] == track).to_numpy()
    if spans:
        mask &= _in_spans(work["frame"], spans)
    return work.loc[~mask]


def reclass_rows(work: pd.DataFrame, track: int, new_cls: int, spans=None) -> pd.DataFrame:
    """Set the class of a track, or of its rows inside ``spans``."""
    out = work.copy()
    mask = (out["track"] == track).to_numpy()
    if spans:
        mask &= _in_spans(out["frame"], spans)
    out.loc[mask, "cls"] = int(new_cls)
    return out


def merge_chains(
    work: pd.DataFrame, pairs: list[tuple[int, int]]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Merge linked tracks into chains that keep the earliest member's ID.

    Rows of a later member at or before the previous member's last frame (the small overlap
    that stage 3's gate 2 allows) are dropped.
    """
    if not pairs:
        return work, {}
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(int(j))] = find(int(i))
    groups: dict[int, list[int]] = {}
    for t in {int(t) for p in pairs for t in p}:
        groups.setdefault(find(t), []).append(t)
    first = work.groupby("track")["frame"].min()
    last = work.groupby("track")["frame"].max()
    rep_of: dict[int, int] = {}
    drop_idx: list = []
    for members in groups.values():
        members.sort(key=lambda t: (int(first[t]), t))
        prev_last = int(last[members[0]])
        for t in members[1:]:
            overlap = work.index[(work["track"] == t) & (work["frame"] <= prev_last)]
            drop_idx.extend(overlap.tolist())
            prev_last = max(prev_last, int(last[t]))
        for t in members:
            rep_of[t] = members[0]
    out = work.drop(index=drop_idx).copy()
    out["track"] = out["track"].map(lambda t: rep_of.get(int(t), int(t))).astype(int)
    return out.sort_values(["track", "frame"]), rep_of


def renumber(work: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """Renumber tracks 1..N by (first frame, old ID); return the table and old->new map."""
    if work.empty:
        return work.reset_index(drop=True), {}
    first = work.groupby("track")["frame"].min().reset_index().sort_values(["frame", "track"])
    id_map = {int(t): i + 1 for i, t in enumerate(first["track"])}
    out = work.copy()
    out["track"] = out["track"].map(id_map).astype(int)
    return out.sort_values(["frame", "track"]).reset_index(drop=True), id_map


def apply_edit(work: pd.DataFrame, event: Event, *, new_id: int | None = None) -> pd.DataFrame:
    """Apply an event's final ``edit`` (spec 4.1); LINK edits are applied by ``merge_chains``."""
    if event.edit is None:
        raise ValueError(f"event {event.id or event.proposal_key[:12]} has no edit to apply")
    kind = EventKind(event.edit["kind"])
    p = event.edit["params"]
    track = event.tracks[0]
    if kind is EventKind.SPLIT:
        return split_track(work, track, int(p["cut_frame"]),
                           next_track_id(work) if new_id is None else new_id)
    if kind is EventKind.DROP:
        return drop_rows(work, track, p.get("spans"))
    if kind is EventKind.RECLASS:
        if p.get("new_cls") is None:
            raise ValueError("a RECLASS edit needs new_cls")
        return reclass_rows(work, track, int(p["new_cls"]), p.get("spans"))
    raise ValueError(f"{kind} edits are applied at stage level, not by apply_edit")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_apply.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/apply.py tests/refine/test_apply.py
git commit -m "feat(refine): add apply (split, drop, reclass, chain merge, renumber, lineage)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Band routing and the `Appearance` protocol

**Files:**
- Create: `src/dnt/refine/verify.py`
- Create: `src/dnt/refine/features.py`
- Test: `tests/refine/test_verify_features.py`

**Interfaces:**
- Consumes: `events.*`.
- Produces:
  - In `verify`:
    - `Band(accept_above, reject_below)` with `Band.of(stage_cfg)`
    - `band_route(score, band) -> Decision | None`
    - `decide(event, decision, *, source, round=0)`, which sets `decision`, appends to `decision_history`, and sets `edit` to `{"kind", "params"}` on accept or `None` otherwise
    - `route_without_vlm(events, band, *, round=0)`
  - In `features`:
    - `Appearance`, a Protocol with `clean_embeddings(raw_id, f0, f1) -> tuple[np.ndarray, np.ndarray]`
    - `ArrayAppearance(table)`
    - `track_embeddings(appearance, lineage) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_verify_features.py
import numpy as np

from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.features import ArrayAppearance, track_embeddings
from dnt.refine.verify import Band, band_route, route_without_vlm


def _ev(kind, score, params):
    return Event.propose(stage="screen", kind=kind, tracks=[1], lineage=[[[1, 0, 9]]],
                         frames=(0, 9), params=params, algo_score=score)


def test_band_route():
    b = Band(0.85, 0.40)
    assert band_route(0.9, b) is Decision.AUTO_ACCEPT
    assert band_route(0.3, b) is Decision.AUTO_REJECT
    assert band_route(0.6, b) is None


def test_route_without_vlm_pending_and_edits():
    acc = _ev(EventKind.DROP, 0.9, {"reason": "in_vehicle", "spans": None})
    mid = _ev(EventKind.DROP, 0.6, {"reason": "static", "spans": None})
    rider = _ev(EventKind.RECLASS, 1.0, {"new_cls": None, "spans": None})
    hinted = _ev(EventKind.RECLASS, 1.0, {"new_cls": 3, "spans": None})
    route_without_vlm([acc, mid, rider, hinted], Band(0.85, 0.40))
    assert acc.decision is Decision.AUTO_ACCEPT and acc.edit["kind"] == "DROP"
    assert mid.decision is Decision.HUMAN_PENDING and mid.edit is None
    assert rider.decision is Decision.HUMAN_PENDING and rider.signals["needs_subtype"] is True
    assert hinted.decision is Decision.AUTO_ACCEPT
    assert acc.decision_history == [{"decision": "AUTO_ACCEPT", "round": 0, "source": "auto"}]


def test_array_appearance_normalizes_and_filters():
    app = ArrayAppearance({7: ([3, 1, 2], np.array([[0, 3.0], [2.0, 0], [0, 5.0]]))})
    f, e = app.clean_embeddings(7, 1, 2)
    assert f.tolist() == [1, 2]
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)
    assert app.clean_embeddings(99, 0, 10)[0].size == 0


def test_track_embeddings_follow_lineage():
    app = ArrayAppearance({1: (range(10), np.eye(10)), 2: (range(10, 20), np.eye(10))})
    f, e = track_embeddings(app, [[1, 5, 9], [2, 10, 11]])
    assert f.tolist() == [5, 6, 7, 8, 9, 10, 11] and e.shape == (7, 10)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_verify_features.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/verify.py
"""Routing proposals by confidence band (spec 4.3). VLM routing arrives in Plan 3."""

from __future__ import annotations

from dataclasses import dataclass

from .events import ACCEPTED, Decision, Event, EventKind


@dataclass(frozen=True)
class Band:
    """A stage's auto-accept and auto-reject thresholds."""

    accept_above: float
    reject_below: float

    @classmethod
    def of(cls, stage_cfg) -> Band:
        """Build a band from a stage config with ``accept_above`` / ``reject_below``."""
        return cls(float(stage_cfg.accept_above), float(stage_cfg.reject_below))


def band_route(score: float, band: Band) -> Decision | None:
    """Return AUTO_ACCEPT, AUTO_REJECT, or None when the score is in the uncertain band."""
    if score >= band.accept_above:
        return Decision.AUTO_ACCEPT
    if score < band.reject_below:
        return Decision.AUTO_REJECT
    return None


def decide(event: Event, decision: Decision, *, source: str, round: int = 0) -> None:
    """Record a decision; an accepted event's edit starts as its proposal (spec 4.1)."""
    event.decision = decision
    event.decision_history.append({"decision": str(decision), "round": int(round),
                                   "source": source})
    event.edit = ({"kind": str(event.kind), "params": dict(event.params)}
                  if decision in ACCEPTED else None)


def route_without_vlm(events: list[Event], band: Band, *, round: int = 0) -> None:
    """Route by band only; the uncertain band and unresolved rider subtypes become pending."""
    for ev in events:
        d = band_route(ev.algo_score, band)
        if (d is Decision.AUTO_ACCEPT and ev.kind is EventKind.RECLASS
                and ev.params.get("new_cls") is None):
            ev.signals["needs_subtype"] = True
            d = Decision.HUMAN_PENDING
        decide(ev, Decision.HUMAN_PENDING if d is None else d, source="auto", round=round)
```

```python
# src/dnt/refine/features.py
"""The appearance interface the stages use (spec 5.3); real providers arrive in Plan 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np


class Appearance(Protocol):
    """Source of clean (unoccluded), L2-normalized embeddings per raw track and frame."""

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames, embeddings)`` for ``raw_id`` within ``[f0, f1]``, sorted by frame."""
        ...


class ArrayAppearance:
    """In-memory ``Appearance`` built from arrays (tests and callers with precomputed features)."""

    def __init__(self, table: Mapping[int, tuple[Sequence[int], np.ndarray]]):
        """Store ``{raw_id: (frames, embeddings)}``, sorted and L2-normalized."""
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for raw, (frames, emb) in table.items():
            f = np.asarray(list(frames), dtype=int)
            e = np.asarray(emb, dtype=float)
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


def track_embeddings(appearance: Appearance, lineage) -> tuple[np.ndarray, np.ndarray]:
    """Return a track's clean samples across its lineage spans, sorted by frame."""
    parts = [appearance.clean_embeddings(int(r), int(a), int(b)) for r, a, b in lineage]
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return np.empty(0, dtype=int), np.empty((0, 0))
    f = np.concatenate([p[0] for p in parts])
    e = np.vstack([p[1] for p in parts])
    order = np.argsort(f, kind="stable")
    return f[order], e[order]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_verify_features.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/verify.py src/dnt/refine/features.py tests/refine/test_verify_features.py
git commit -m "feat(refine): add band routing and the Appearance protocol

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 11: Stage 1 — ID-switch proposals

**Files:**
- Create: `src/dnt/refine/switch.py`
- Test: `tests/refine/test_switch.py`

**Interfaces:**
- Consumes:
  - `config.RefineConfig`, `config.to_frames`
  - `primitives.kalman_nis`, `iou_matrix`, `ramp`
  - `features.Appearance`, `track_embeddings`
  - `apply.lineage_of_rows`
  - `events.Event`, `EventKind`
- Produces (in `dnt.refine.switch`):
  - `STAGE = "switch"`
  - `SwitchResult(events: list[Event], weak_cuts: dict[int, list[int]], candidates: dict[int, list[int]])`. `weak_cuts` are cut frames scoring in `[screen.segment_at, switch.reject_below)`, and `candidates` are all surviving local maxima, which Plan 2 densifies around.
  - `contact_flags(work, thr) -> pd.Series`
  - `propose_splits(work, cfg, fps, appearance=None) -> SwitchResult`
  - `SPLIT` events have `params={"cut_frame": t}`, `frames=(t, t)`, and `signals` keys `app, z_app, bimodal, silhouette, mot, nis, jump, gate (list of fired condition names), motion_only`. `swap_with`, `swap_cross` and `swap_boost` appear when a swap is confirmed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_switch.py
import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance
from dnt.refine.switch import propose_splits

from ._fixtures import box_rows, table

FPS = 10.0
A_, B_ = np.eye(8)[0], np.eye(8)[1]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _emb(frames, change_at, before=A_, after=B_):
    frames = list(frames)
    return frames, np.array([before if f < change_at else after for f in frames])


def test_swap_with_contact_splits_both_tracks_and_boosts():
    frames = range(100)
    w = _work(box_rows(1, frames, 100.0, 100.0, vx=4.0), box_rows(2, frames, 500.0, 100.0, vx=-4.0))
    app = ArrayAppearance({1: _emb(frames, 50), 2: _emb(frames, 50, before=B_, after=A_)})
    res = propose_splits(w, RefineConfig.defaults(), FPS, app)
    cuts = {e.tracks[0]: e for e in res.events}
    assert set(cuts) == {1, 2}
    for t, other in ((1, 2), (2, 1)):
        ev = cuts[t]
        assert ev.params["cut_frame"] == 50
        assert ev.signals["swap_with"] == other and ev.signals["swap_boost"] == pytest.approx(0.2)
        assert ev.algo_score >= 0.85 - 1e-9
        assert "contact" in ev.signals["gate"]


def test_appearance_drift_without_gate_has_no_event():
    frames = list(range(100))
    emb = np.array([np.cos(f / 60) * A_ + np.sin(f / 60) * B_ for f in frames])
    w = _work(box_rows(1, frames, 100.0, 100.0, vx=2.0))
    res = propose_splits(w, RefineConfig.defaults(), FPS, ArrayAppearance({1: (frames, emb)}))
    assert res.events == []


def test_motion_only_jump_after_gap_is_capped():
    rows = box_rows(1, range(0, 40), 100.0, 100.0, vx=2.0) + box_rows(1, range(45, 80), 490.0,
                                                                        100.0, vx=2.0)
    res = propose_splits(_work(rows), RefineConfig.defaults(), FPS, None)
    assert len(res.events) == 1
    ev = res.events[0]
    assert ev.params["cut_frame"] == 45 and ev.algo_score == pytest.approx(0.7)
    assert ev.signals["motion_only"] is True and "gap" in ev.signals["gate"]


def test_short_sides_have_no_event():
    short = box_rows(1, range(0, 8), 100.0, 100.0, vx=2.0)  # 0.8 s < 2 x min_side: skipped
    short[5][2] += 300
    late = box_rows(2, [*range(0, 27), 28, 29], 100.0, 300.0, vx=2.0)
    for r in late:
        if r[0] >= 28:  # jump after a gap, but only 0.2 s after it
            r[2] += 300
    res = propose_splits(_work(short, late), RefineConfig.defaults(), FPS, None)
    assert res.events == [] and res.candidates == {}


def _takeover(class_gate=True):
    rows = []
    for f in range(780, 861):
        truck = f >= 821
        cls = 5 if f == 821 else 7 if (f == 818 or f > 821) else 2
        rows.append([f, 12, 200.0 + (f - 780), 87.0, 95.0 if truck else 53.0,
                     60.0 if truck else 56.0, 0.9, cls, -1, -1])
    cfg = RefineConfig.defaults("vehicle")
    cfg.switch.class_change_gate = class_gate
    app = ArrayAppearance({12: _emb(range(780, 861), 821)})
    return propose_splits(_work(rows), cfg, FPS, app)


@pytest.mark.parametrize("class_gate", [True, False])
def test_takeover_by_untracked_object_splits_at_821(class_gate):
    res = _takeover(class_gate)
    assert [e.params["cut_frame"] for e in res.events] == [821]
    ev = res.events[0]
    assert ev.algo_score >= 0.5
    assert "size_jump" in ev.signals["gate"]
    assert ("class_change" in ev.signals["gate"]) is class_gate


def test_class_flicker_alone_has_no_event():
    rows = box_rows(1, range(100), 100.0, 100.0, vx=2.0, cls=2)
    rows[50][7] = 7
    frames = range(100)
    res = propose_splits(_work(rows), RefineConfig.defaults("vehicle"), FPS,
                         ArrayAppearance({1: (list(frames), np.tile(A_, (100, 1)))}))
    assert res.events == []


def test_weak_candidate_becomes_a_weak_cut():
    rows = box_rows(1, [f for f in range(100) if f not in (49,)], 100.0, 100.0, vx=2.0)
    for r in rows:
        if r[0] >= 50:
            r[4] = 30.0 * np.exp(0.24)
    res = propose_splits(_work(rows), RefineConfig.defaults(), FPS, None)
    assert res.events == [] and res.weak_cuts == {1: [50]}


def test_single_row_and_two_row_tracks_do_not_crash():
    w = _work(box_rows(1, [5], 0.0, 0.0), box_rows(2, [5, 6], 50.0, 0.0))
    assert propose_splits(w, RefineConfig.defaults(), FPS, None).events == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_switch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnt.refine.switch'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/switch.py
"""Stage 1: ID-switch proposals (spec 6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig, to_frames
from .events import Event, EventKind
from .features import Appearance, track_embeddings
from .primitives import iou_matrix, kalman_nis, ramp

STAGE = "switch"


@dataclass
class SwitchResult:
    """Stage 1 proposals plus the weaker cut points screening uses (spec 6.2)."""

    events: list[Event]
    weak_cuts: dict[int, list[int]] = field(default_factory=dict)
    candidates: dict[int, list[int]] = field(default_factory=dict)


def contact_flags(work: pd.DataFrame, thr: float) -> pd.Series:
    """Return True where a row's box has IoU > ``thr`` with another track's box that frame."""
    flags = pd.Series(False, index=work.index)
    for _, g in work.groupby("frame"):
        if len(g) < 2:
            continue
        boxes = g[["x", "y", "w", "h"]].to_numpy(float)
        m = iou_matrix(boxes, boxes)
        np.fill_diagonal(m, 0.0)
        flags.loc[g.index] = m.max(axis=1) > thr
    return flags


def _window_any(frames: np.ndarray, flags: np.ndarray, delta: int) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(flags.astype(int))])
    lo = np.searchsorted(frames, frames - delta, side="left")
    hi = np.searchsorted(frames, frames + delta, side="right")
    return (cs[hi] - cs[lo]) > 0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _side_means(ef, emb, t, w):
    b = emb[(ef >= t - w) & (ef < t)]
    a = emb[(ef >= t) & (ef < t + w)]
    if not len(b) or not len(a):
        return None
    return _unit(b.mean(axis=0)), _unit(a.mean(axis=0))


def _appearance_change(frames, ef, emb, w) -> np.ndarray:
    out = np.full(len(frames), np.nan)
    for i, t in enumerate(frames):
        m = _side_means(ef, emb, t, w)
        if m is not None:
            out[i] = 1.0 - float(m[0] @ m[1])
    return out


def _two_means(emb: np.ndarray, iters: int = 20) -> np.ndarray:
    centers = np.stack([emb[0], emb[-1]]).astype(float)
    labels = np.full(len(emb), -1)
    for _ in range(iters):
        new = np.argmax(emb @ centers.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                centers[k] = _unit(emb[labels == k].mean(axis=0))
    return labels


def _silhouette(emb: np.ndarray, labels: np.ndarray) -> float:
    d = 1.0 - emb @ emb.T
    vals = []
    for i in range(len(emb)):
        same = labels == labels[i]
        same[i] = False
        other = labels != labels[i]
        if not same.any() or not other.any():
            continue
        a, b = d[i, same].mean(), d[i, other].mean()
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def _bimodal_split(ef, emb, purity: float):
    """Return ``(k, silhouette)`` where sample ``k`` starts the second cluster, or None."""
    if len(ef) < 4:
        return None
    labels = _two_means(emb)
    if labels.min() == labels.max():
        return None
    best = (-1.0, 0)
    for k in range(1, len(ef)):
        first = int(np.bincount(labels[:k], minlength=2).argmax())
        score = min(float(np.mean(labels[:k] == first)), float(np.mean(labels[k:] != first)))
        if score > best[0]:
            best = (score, k)
    if best[0] < purity:
        return None
    return best[1], _silhouette(emb, labels)


def _score_track(g, contact, cfg: RefineConfig, fps, appearance, lin, delta, w):
    sc, mc = cfg.switch, cfg.motion
    frames = g["frame"].to_numpy(int)
    boxes = g[["x", "y", "w", "h"]].to_numpy(float)
    cls = g["cls"].to_numpy(int)
    n = len(frames)
    motion_only = appearance is None
    if motion_only:
        ef, emb = np.empty(0, dtype=int), np.empty((0, 0))
    else:
        ef, emb = track_embeddings(appearance, lin)
    samples = frames if motion_only else ef
    if n < 3 or len(samples) == 0:
        return None
    if (samples[-1] - samples[0] + 1) / fps < 2 * sc.min_side_seconds:
        return None
    nis = kalman_nis(frames, boxes, process_var=mc.process_var, meas_var_pos=mc.meas_var_pos,
                     meas_var_size=mc.meas_var_size)
    jump = np.zeros(n)
    ratio = np.maximum(boxes[1:, 2:4], 1e-6) / np.maximum(boxes[:-1, 2:4], 1e-6)
    jump[1:] = np.abs(np.log(ratio)).max(axis=1)
    mot = np.maximum(np.nan_to_num(ramp(nis, sc.nis_hi / 2.0, sc.nis_hi)),
                     ramp(jump, *sc.ramps["jump"]))
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(frames) > 1
    change = np.zeros(n, dtype=bool)
    change[1:] = cls[1:] != cls[:-1]
    fired = {
        "contact": _window_any(frames, contact, delta),
        "gap": _window_any(frames, gap, delta),
        "size_jump": _window_any(frames, jump >= np.log(sc.size_gate), delta),
    }
    if sc.class_change_gate:
        fired["class_change"] = _window_any(frames, change, delta)
    gate = np.zeros(n, dtype=bool)
    for v in fired.values():
        gate |= v
    a_raw = np.zeros(n)
    app = np.zeros(n)
    z = np.full(n, np.nan)
    bim = np.zeros(n)
    sil = None
    if not motion_only:
        change_a = _appearance_change(frames, ef, emb, w)
        fin = np.isfinite(change_a)
        if fin.any():
            med = float(np.median(change_a[fin]))
            mad = float(np.median(np.abs(change_a[fin] - med)))
            z = (change_a - med) / (1.4826 * mad + sc.mad_floor)
            app = np.nan_to_num(ramp(z, *sc.ramps["z_app"]))
        a_raw = np.nan_to_num(change_a)
        split = _bimodal_split(ef, emb, sc.bimodal_purity)
        if split is not None and split[1] >= sc.bimodal_silhouette_min:
            k, sil = split
            after = np.flatnonzero(frames > ef[k - 1])
            if len(after):
                i = int(after[0])
                bim[i] = ramp(sil, *sc.ramps["silhouette"])
                app[i] = max(app[i], bim[i])
        score = gate * (sc.w_app * app + sc.w_mot * mot)
    else:
        score = np.minimum(gate * mot, sc.motion_only_cap)
    score[0] = 0.0
    return {"frames": frames, "S": score, "A": a_raw, "app": app, "z": z, "bim": bim,
            "sil": sil, "mot": mot, "nis": nis, "jump": jump, "fired": fired,
            "samples": samples, "motion_only": motion_only, "ef": ef, "emb": emb}


def _candidates(info, fps, cfg: RefineConfig, nms: int) -> list[int]:
    sc = cfg.switch
    s, frames = info["S"], info["frames"]
    n = len(s)
    left = np.concatenate([[-np.inf], s[:-1]])
    right = np.concatenate([s[1:], [-np.inf]])
    peaks = [i for i in range(1, n) if s[i] > 0 and s[i] >= left[i] and s[i] >= right[i]]
    peaks.sort(key=lambda i: (-s[i], -info["A"][i], -info["mot"][i], frames[i]))
    kept: list[int] = []
    for i in peaks:
        if all(abs(int(frames[i]) - int(frames[j])) >= nms for j in kept):
            kept.append(i)
    samples = info["samples"]
    out = []
    for i in kept:
        t = frames[i]
        before, after = samples[samples < t], samples[samples >= t]
        if not len(before) or not len(after):
            continue
        if (before[-1] - before[0] + 1) / fps < sc.min_side_seconds:
            continue
        if (after[-1] - after[0] + 1) / fps < sc.min_side_seconds:
            continue
        out.append(i)
    return sorted(out)


def _pair_contact(work, ti, tj, t, delta, thr) -> bool:
    win = work[work["frame"].between(t - delta, t + delta)]
    a = win[win["track"] == ti]
    b = win[win["track"] == tj]
    m = a.merge(b, on="frame", suffixes=("_a", "_b"))
    for row in m.itertuples(index=False):
        box_a = [[row.x_a, row.y_a, row.w_a, row.h_a]]
        box_b = [[row.x_b, row.y_b, row.w_b, row.h_b]]
        if iou_matrix(box_a, box_b)[0, 0] > thr:
            return True
    return False


def propose_splits(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, appearance: Appearance | None = None
) -> SwitchResult:
    """Propose SPLIT events at likely ID switches (spec 6.1)."""
    sc = cfg.switch
    result = SwitchResult(events=[])
    if work.empty:
        return result
    contact = contact_flags(work, sc.contact_iou)
    delta = to_frames(sc.delta, fps)
    w = to_frames(sc.window, fps)
    nms = to_frames(sc.nms_seconds, fps)
    infos: dict[int, tuple[dict, list]] = {}
    cands: dict[int, list[int]] = {}
    for tid, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        lin = lineage_of_rows(g)
        info = _score_track(g, contact.loc[g.index].to_numpy(bool), cfg, fps, appearance, lin,
                            delta, w)
        if info is None:
            continue
        idx = _candidates(info, fps, cfg, nms)
        infos[int(tid)] = (info, lin)
        cands[int(tid)] = idx
        if idx:
            result.candidates[int(tid)] = [int(info["frames"][i]) for i in idx]

    boost: dict[tuple[int, int], tuple[float, int, float]] = {}
    if appearance is not None:
        flat = [(t, i) for t, idx in cands.items() for i in idx]
        for a in range(len(flat)):
            for b in range(a + 1, len(flat)):
                (ti, ii), (tj, jj) = flat[a], flat[b]
                if ti == tj:
                    continue
                fi = int(infos[ti][0]["frames"][ii])
                fj = int(infos[tj][0]["frames"][jj])
                if abs(fi - fj) > delta:
                    continue
                if not _pair_contact(work, ti, tj, (fi + fj) // 2, delta, sc.contact_iou):
                    continue
                mi = _side_means(infos[ti][0]["ef"], infos[ti][0]["emb"], fi, w)
                mj = _side_means(infos[tj][0]["ef"], infos[tj][0]["emb"], fj, w)
                if mi is None or mj is None:
                    continue
                (bi, ai), (bj, aj) = mi, mj
                cross = float(bi @ aj - bi @ ai + bj @ ai - bj @ aj)
                if cross > 0:
                    inc = sc.swap_boost * ramp(cross, *sc.ramps["cross"])
                    boost[(ti, ii)] = (inc, tj, cross)
                    boost[(tj, jj)] = (inc, ti, cross)

    for tid, idx in cands.items():
        info, lin = infos[tid]
        for i in idx:
            s = float(info["S"][i])
            extra = {}
            if (tid, i) in boost:
                inc, other, cross = boost[(tid, i)]
                s = min(1.0, s + inc)
                extra = {"swap_with": int(other), "swap_cross": cross, "swap_boost": inc}
            t = int(info["frames"][i])
            if s >= sc.reject_below:
                signals = {
                    "app": float(info["app"][i]), "z_app": float(info["z"][i]),
                    "bimodal": float(info["bim"][i]), "silhouette": info["sil"],
                    "mot": float(info["mot"][i]), "nis": float(info["nis"][i]),
                    "jump": float(info["jump"][i]),
                    "gate": [k for k, v in info["fired"].items() if v[i]],
                    "motion_only": info["motion_only"], **extra,
                }
                result.events.append(Event.propose(
                    stage=STAGE, kind=EventKind.SPLIT, tracks=[tid], lineage=[lin],
                    frames=(t, t), params={"cut_frame": t}, algo_score=s, signals=signals,
                ))
            elif s >= cfg.screen.segment_at:
                result.weak_cuts.setdefault(tid, []).append(t)
    return result
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_switch.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

If `test_weak_candidate_becomes_a_weak_cut` finds the NIS term pushing the score to at least 0.5, check the frame-50 values with `res.candidates` and the event's `signals`. Size changes ride on the `w` measurement, whose noise is `meas_var_size = 16`, so a +8 px width step should give NIS well under `nis_hi/2`. Adjust the fixture's width step, not the thresholds.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/switch.py tests/refine/test_switch.py
git commit -m "feat(refine): add stage 1 switch proposals with gates, appearance change, swaps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: ReClass hints and stage 2 — screening

**Files:**
- Create: `src/dnt/refine/hints.py`
- Create: `src/dnt/refine/screen.py`
- Test: `tests/refine/test_screen.py`

**Interfaces:**
- Consumes: `primitives.*`, `apply.lineage_of_rows`, `events.*`, `config.RefineConfig`.
- Produces:
  - In `hints`:
    - `ReclassHint(raw_id: int, cls: int, avg_score: float)`, a frozen dataclass
    - `read_reclass_hints(path, known_raw_ids) -> dict[int, ReclassHint]`
  - In `screen`:
    - `STAGE = "screen"`, `ORPHAN_STAGE = "orphan"`
    - `ScreenContext(boxes: pd.DataFrame | None = None, fmt: str | None = None, hints: dict[int, ReclassHint] = {}, split_raw_ids: set[int] = set())`
    - `propose_screen(work, cfg, fps, sctx, segment_cuts: dict[int, list[int]]) -> list[Event]`
  - `DROP` params: `{"reason": "static" | "in_vehicle" | "duplicate", "spans": [[f0, f1], ...] | None}`, plus `"of"` for `duplicate`.
  - `RECLASS` params: `{"new_cls": int | None, "spans": ... | None}`.
  - `signals` always includes `hypothesis`. Partial events add `partial: True` and `segments`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_screen.py
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind
from dnt.refine.hints import ReclassHint, read_reclass_hints
from dnt.refine.screen import ScreenContext, propose_screen
from dnt.refine.verify import Band, route_without_vlm

from ._fixtures import box_rows, table

FPS = 10.0


def _work(*rows):
    return io.to_work(table(*rows)).work


def _ctx_tracks(*rows):
    t = table(*rows)
    return pd.DataFrame({"frame": t.frame, "track": t.track, "x": t.x, "y": t.y, "w": t.w,
                         "h": t.h, "cls": t.cls}), "tracks"


def _ctx_dets(*rows):
    t = table(*rows)
    return pd.DataFrame({"frame": t.frame, "track": -1, "x": t.x, "y": t.y, "w": t.w, "h": t.h,
                         "cls": t.cls}), "dets"


def _screen(work, cfg=None, ctx=None, hints=None, split=None, cuts=None):
    cfg = cfg or RefineConfig.defaults()
    boxes, fmt = ctx if ctx else (None, None)
    sctx = ScreenContext(boxes=boxes, fmt=fmt, hints=hints or {}, split_raw_ids=split or set())
    evs = propose_screen(work, cfg, FPS, sctx, cuts or {})
    route_without_vlm(evs, Band.of(cfg.screen))
    return {e.tracks[0]: e for e in evs}


def _ped(track, frames, x0=100.0, vx=3.0):
    return box_rows(track, frames, x0, 110.0, vx=vx, w=20.0, h=40.0)


def _car(track, frames, x0=80.0, vx=3.0, cls=2):
    return box_rows(track, frames, x0, 100.0, vx=vx, w=80.0, h=60.0, cls=cls)


def test_static_low_conf_hotspot_is_capped_and_pending():
    rows = [box_rows(t, range(120 * (t - 1), 120 * t), 300.0, 200.0, score=0.35)
            for t in (1, 2, 3)]
    evs = _screen(_work(*rows))
    for t in (1, 2, 3):
        ev = evs[t]
        assert ev.kind is EventKind.DROP and ev.params["reason"] == "static"
        assert ev.algo_score == pytest.approx(0.8)
        assert ev.decision is Decision.HUMAN_PENDING


def test_static_cue_ignores_missing_scores():
    ev = _screen(_work(box_rows(1, range(120), 300.0, 200.0, score=-1)))[1]
    assert ev.signals["C"] is None and ev.params["reason"] == "static"


def test_person_inside_moving_car_is_dropped():
    ev = _screen(_work(_ped(1, range(50))), ctx=_ctx_tracks(_car(9, range(50))))[1]
    assert ev.params["reason"] == "in_vehicle" and ev.decision is Decision.AUTO_ACCEPT


def test_boarding_passenger_is_kept():
    evs = _screen(_work(_ped(1, range(50))), ctx=_ctx_tracks(_car(9, range(10))))
    assert 1 not in evs


def test_detection_context_uses_persistence():
    ev = _screen(_work(_ped(1, range(50))), ctx=_ctx_dets(_car(9, range(50))))[1]
    assert ev.params["reason"] == "in_vehicle" and ev.algo_score == pytest.approx(1.0)


def test_fast_smooth_person_is_a_rider_and_walker_is_not():
    evs = _screen(_work(_ped(1, range(50), vx=12.0), _ped(2, range(50), x0=900.0, vx=3.2)))
    assert evs[1].kind is EventKind.RECLASS and evs[1].params["new_cls"] is None
    assert evs[1].decision is Decision.HUMAN_PENDING  # subtype still needed
    assert 2 not in evs


def test_localized_hint_settles_subtype():
    ev = _screen(_work(_ped(1, range(50), vx=12.0)),
                 hints={1: ReclassHint(1, 3, 0.95)})[1]
    assert ev.params["new_cls"] == 3 and ev.signals["subtype_source"] == "reclass"
    assert ev.decision is Decision.AUTO_ACCEPT


def test_hint_is_unlocalized_after_split():
    walk = _ped(1, range(0, 50), vx=3.0)
    ride = _ped(2, range(50, 100), x0=250.0, vx=12.0)
    solo = _ped(5, range(0, 50), x0=2000.0, vx=12.0)
    w = _work(walk, ride, solo)
    w.loc[w["track"] == 2, "raw_id"] = 1  # track 2 is the tail of a split of raw track 1
    hints = {1: ReclassHint(1, 3, 0.95), 5: ReclassHint(5, 3, 0.95)}
    evs = _screen(w, hints=hints, split={1})
    assert 1 not in evs
    assert evs[2].params["new_cls"] is None and evs[2].signals["hint_unlocalized"]["cls"] == 3
    assert evs[5].params["new_cls"] == 3


def test_vehicle_duplicate_drops_the_smaller():
    big = box_rows(1, range(30), 100.0, 100.0, vx=2.0, w=120.0, h=80.0, cls=2)
    small = box_rows(2, range(30), 110.0, 110.0, vx=2.0, w=50.0, h=40.0, cls=2)
    evs = _screen(_work(big, small), cfg=RefineConfig.defaults("vehicle"))
    assert 1 not in evs
    assert evs[2].params == {"reason": "duplicate", "of": 1, "spans": None}


def test_pending_split_gives_partial_drop():
    w = _work(_ped(1, range(100)))
    ctx = _ctx_tracks(_car(9, range(50, 100), x0=80.0 + 150.0))
    ev = _screen(w, ctx=ctx, cuts={1: [50]})[1]
    assert ev.params["spans"] == [[50, 99]] and ev.signals["partial"] is True
    assert ev.algo_score == pytest.approx(0.75) and ev.decision is Decision.HUMAN_PENDING


def test_applied_split_drops_only_the_passenger_track():
    w = _work(_ped(1, range(50)), _ped(2, range(50, 100), x0=250.0))
    ctx = _ctx_tracks(_car(9, range(50, 100), x0=230.0))
    evs = _screen(w, ctx=ctx)
    assert 1 not in evs and evs[2].params["reason"] == "in_vehicle"
    assert evs[2].params["spans"] is None


def test_context_with_no_overlapping_frames_is_harmless():
    w = _work(_ped(1, range(50)))
    assert _screen(w, ctx=_ctx_tracks(_car(9, range(500, 550)))) == {}
    empty = pd.DataFrame(columns=["frame", "track", "x", "y", "w", "h", "cls"])
    assert _screen(w, ctx=(empty, "tracks")) == {}


def test_single_row_track_has_no_screen_event():
    assert _screen(_work(box_rows(1, [7], 0.0, 0.0))) == {}


def test_read_reclass_hints(tmp_path, caplog):
    p = tmp_path / "h.csv"
    p.write_text("track,cls,avg_score\n1,3,0.95\n99,1,0.8\n")
    hints = read_reclass_hints(p, {1, 2})
    assert hints == {1: ReclassHint(1, 3, 0.95)} and "unknown" in caplog.text
    (tmp_path / "bad.csv").write_text("id,cls\n1,3\n")
    with pytest.raises(ValueError, match="track, cls, avg_score"):
        read_reclass_hints(tmp_path / "bad.csv", {1})
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_screen.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `hints.py`**

```python
# src/dnt/refine/hints.py
"""Optional external cue files: ReClass hints (spec 2.5, 6.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)
_REQUIRED = ("track", "cls", "avg_score")


@dataclass(frozen=True)
class ReclassHint:
    """One ``ReClass.re_classify`` result for a raw track."""

    raw_id: int
    cls: int
    avg_score: float


def read_reclass_hints(path, known_raw_ids) -> dict[int, ReclassHint]:
    """Read ReClass output (header ``track, cls, avg_score``); ignore unknown track IDs."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"{path}: empty hints file; expected header track, cls, avg_score"
        ) from exc
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: hints file lacks {missing}; expected header track, cls, avg_score"
        )
    known = {int(k) for k in known_raw_ids}
    out: dict[int, ReclassHint] = {}
    unknown = 0
    for t, c, s in df[list(_REQUIRED)].itertuples(index=False):
        if int(t) not in known:
            unknown += 1
            continue
        out[int(t)] = ReclassHint(int(t), int(c), float(s))
    if unknown:
        log.warning("%s: ignored %d hint row(s) for unknown track IDs", path, unknown)
    return out
```

- [ ] **Step 4: Implement `screen.py`**

```python
# src/dnt/refine/screen.py
"""Stage 2: false-track screening, and the orphan pass (spec 6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig
from .events import Event, EventKind
from .hints import ReclassHint
from .primitives import box_centers, heading_smoothness, iob_matrix, iou_matrix, ramp, speeds_hps

STAGE = "screen"
ORPHAN_STAGE = "orphan"


@dataclass
class ScreenContext:
    """Inputs screening needs besides the work table."""

    boxes: pd.DataFrame | None = None
    fmt: str | None = None
    hints: dict[int, ReclassHint] = field(default_factory=dict)
    split_raw_ids: set[int] = field(default_factory=set)


@dataclass
class _Unit:
    track: int
    frames: np.ndarray
    boxes: np.ndarray
    score: np.ndarray
    hmed: float
    v: np.ndarray
    vel: np.ndarray
    localized_hint: ReclassHint | None
    unlocalized_hint: ReclassHint | None


def _pixel_velocity(frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    c = box_centers(boxes)
    vel = np.zeros_like(c)
    if len(c) > 1:
        vel[1:] = np.diff(c, axis=0) / np.maximum(np.diff(frames), 1)[:, None]
        vel[0] = vel[1]
    return vel


def _unit(track, rows: pd.DataFrame, cfg, fps, localized=None, unlocalized=None) -> _Unit:
    frames = rows["frame"].to_numpy(int)
    boxes = rows[["x", "y", "w", "h"]].to_numpy(float)
    return _Unit(
        track=int(track), frames=frames, boxes=boxes, score=rows["score"].to_numpy(float),
        hmed=max(float(np.median(boxes[:, 3])), 1.0),
        v=speeds_hps(frames, boxes, fps, cfg.motion.height_window),
        vel=_pixel_velocity(frames, boxes), localized_hint=localized, unlocalized_hint=unlocalized,
    )


class _ContextIndex:
    """Context boxes per frame, with pixel velocities for track contexts."""

    def __init__(self, sctx: ScreenContext):
        """Index ``sctx.boxes`` by frame."""
        self.fmt = sctx.fmt
        self.present = sctx.boxes is not None
        self.by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if not self.present or not len(sctx.boxes):
            return
        b = sctx.boxes.sort_values(["track", "frame"]).reset_index(drop=True)
        boxes = b[["x", "y", "w", "h"]].to_numpy(float)
        vel = np.zeros((len(b), 2))
        if self.fmt == "tracks":
            for _, g in b.groupby("track"):
                idx = g.index.to_numpy()
                vel[idx] = _pixel_velocity(g["frame"].to_numpy(int), boxes[idx])
        cls = b["cls"].to_numpy(int)
        for f, g in b.groupby("frame"):
            idx = g.index.to_numpy()
            self.by_frame[int(f)] = (boxes[idx], cls[idx], vel[idx])

    def at(self, frame: int, classes) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes, velocities)`` of context boxes of ``classes`` in ``frame``."""
        entry = self.by_frame.get(int(frame))
        if entry is None:
            return np.empty((0, 4)), np.empty((0, 2))
        boxes, cls, vel = entry
        keep = np.isin(cls, list(classes))
        return boxes[keep], vel[keep]


def _context_fraction(u: _Unit, ctx: _ContextIndex, classes, overlap: str, thr: float,
                      cfg: RefineConfig, fps: float) -> float:
    """Fraction of rows where a context box of ``classes`` overlaps ``u`` and moves with it."""
    overlap_fn = iob_matrix if overlap == "iob" else iou_matrix
    hits = np.zeros(len(u.frames), dtype=bool)
    for i, f in enumerate(u.frames):
        boxes, vel = ctx.at(f, classes)
        if not len(boxes):
            continue
        ov = overlap_fn(u.boxes[i : i + 1], boxes)[0]
        ok = ov >= thr
        if not ok.any():
            continue
        if ctx.fmt == "tracks":
            dv = np.linalg.norm(vel[ok] - u.vel[i], axis=1) / u.hmed * fps
            hits[i] = bool((dv < cfg.screen.move_together).any())
        else:
            if i == 0 or not np.isfinite(u.v[i]) or u.v[i] < cfg.motion.moving_min:
                continue
            prev_boxes, _ = ctx.at(u.frames[i - 1], classes)
            if not len(prev_boxes):
                continue
            prev_ok = prev_boxes[overlap_fn(u.boxes[i - 1 : i], prev_boxes)[0] >= thr]
            if len(prev_ok):
                persist = iou_matrix(boxes[ok], prev_ok) >= cfg.screen.persistence_iou
                hits[i] = bool(persist.any())
    return float(hits.mean()) if len(hits) else 0.0


def _static_center(u: _Unit, cfg: RefineConfig) -> tuple[float, float, np.ndarray]:
    c = box_centers(u.boxes)
    med = np.median(c, axis=0)
    r = float(np.percentile(np.linalg.norm(c - med, axis=1), 95) / u.hmed)
    return r, float(ramp(r, *cfg.screen.ramps["R"])), med


def _static(u: _Unit, cfg, fps, static_meds: dict[int, np.ndarray], cap: float):
    r = cfg.screen.ramps
    raw_r, r_ramp, med = _static_center(u, cfg)
    c = box_centers(u.boxes)
    if len(u.frames) > 1:
        step = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(u.frames), 1)
        jit = float(np.median(step) / u.hmed)
    else:
        jit = 0.0
    valid = u.score[u.score >= 0]
    conf = float(valid.mean()) if len(valid) else None
    dur = (u.frames[-1] - u.frames[0] + 1) / fps
    radius = cfg.screen.hotspot_radius * u.hmed
    near = sum(1 for t, m in static_meds.items()
               if t != u.track and np.linalg.norm(m - med) <= radius)
    hot = near + (1 if r_ramp >= 0.5 else 0)
    parts = [ramp(jit, *r["J"]), ramp(hot, *r["H"])]
    if conf is not None:
        parts.append(ramp(conf, *r["C"]))
    score = min(cap, r_ramp * ramp(dur, *r["T"]) * float(np.mean(parts)))
    signals = {"R": raw_r, "J": jit, "C": conf, "T": dur, "H": hot}
    return score, signals, {"reason": "static"}


def _in_vehicle(u: _Unit, ctx: _ContextIndex, cfg, fps):
    if not ctx.present:
        return None
    frac = _context_fraction(u, ctx, cfg.context.vehicle_classes, "iob",
                             cfg.screen.in_vehicle_iob, cfg, fps)
    return ramp(frac, *cfg.screen.ramps["inside"]), {"inside_frac": frac}, {"reason": "in_vehicle"}


def _rider(u: _Unit, ctx: _ContextIndex, cfg: RefineConfig, fps):
    sc, r = cfg.screen, cfg.screen.ramps
    v = u.v[1:]
    v = v[np.isfinite(v)]
    frac_fast = float(np.mean(v > sc.rider_speed)) if len(v) else 0.0
    smooth = heading_smoothness(u.frames, u.boxes, fps, window=cfg.motion.height_window,
                                moving_min=cfg.motion.moving_min)
    smooth = 0.0 if not np.isfinite(smooth) else smooth
    k = (_context_fraction(u, ctx, cfg.context.twowheeler_classes, "iou", sc.twowheeler_iou,
                           cfg, fps) if ctx.present else None)
    hint = u.localized_hint
    p = hint.avg_score if hint is not None and hint.cls in cfg.hints.reclass_class_map else None
    rp = ramp(p, *cfg.hints.reclass_ramp) if p is not None else 0.0
    score = max(ramp(frac_fast, *r["F"]) * ramp(smooth, *r["S"]),
                ramp(k, *r["K"]) if k is not None else 0.0, rp)
    signals = {"F": frac_fast, "S_smooth": smooth, "K": k, "P": p}
    new_cls = None
    if p is not None and rp >= cfg.hints.subtype_min:
        subtype = cfg.hints.reclass_class_map[hint.cls]
        new_cls = int(cfg.reclass_map[subtype])
        signals["subtype_source"] = "reclass"
        signals["subtype"] = subtype
    if u.unlocalized_hint is not None:
        signals["hint_unlocalized"] = {"cls": u.unlocalized_hint.cls,
                                       "avg_score": u.unlocalized_hint.avg_score}
    return score, signals, {"new_cls": new_cls}


def _duplicate(u: _Unit, others: dict[int, _Unit], cfg: RefineConfig, fps):
    sc = cfg.screen
    area_u = float(np.median(u.boxes[:, 2] * u.boxes[:, 3]))
    best = None
    for tid, o in others.items():
        if tid == u.track:
            continue
        common, iu, io_ = np.intersect1d(u.frames, o.frames, return_indices=True)
        if len(common) < sc.duplicate_min_frames:
            continue
        area_o = float(np.median(o.boxes[:, 2] * o.boxes[:, 3]))
        if area_u > area_o or (area_u == area_o and u.track < tid):
            continue
        ok = 0
        for a, b in zip(iu, io_, strict=True):
            iob = iob_matrix(u.boxes[a : a + 1], o.boxes[b : b + 1])[0, 0]
            dv = float(np.linalg.norm(u.vel[a] - o.vel[b]) / u.hmed * fps)
            ok += int(iob >= sc.duplicate_iob and dv < sc.move_together)
        frac = ok / len(common)
        s = ramp(frac, *sc.ramps["D"])
        if best is None or s > best[0]:
            best = (s, {"D": frac}, {"reason": "duplicate", "of": int(tid)})
    return best


def _segments(rows: pd.DataFrame, cuts: list[int]) -> list[pd.DataFrame]:
    f = rows["frame"].to_numpy(int)
    cuts = sorted({c for c in cuts if f[0] < c <= f[-1]})
    if not cuts:
        return []
    bounds = [f[0], *cuts, f[-1] + 1]
    segs = [rows[(rows["frame"] >= a) & (rows["frame"] < b)] for a, b in pairwise(bounds)]
    return [s for s in segs if len(s)]


def propose_screen(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, sctx: ScreenContext,
    segment_cuts: dict[int, list[int]],
) -> list[Event]:
    """Propose DROP / RECLASS events for false tracks (spec 6.2)."""
    if work.empty:
        return []
    sc = cfg.screen
    ctx = _ContextIndex(sctx)
    cap = sc.vehicle_static_score_cap if cfg.target == "vehicle" else sc.static_score_cap
    tracks = {int(t): g.sort_values("frame") for t, g in work.groupby("track", sort=True)}
    wholes = {t: _unit(t, g, cfg, fps) for t, g in tracks.items()}
    static_meds = {}
    for t, u in wholes.items():
        _, r_ramp, med = _static_center(u, cfg)
        if r_ramp >= 0.5:
            static_meds[t] = med

    def hyps(u):
        out = {"static": lambda: _static(u, cfg, fps, static_meds, cap)}
        if cfg.target == "person":
            out["in_vehicle"] = lambda: _in_vehicle(u, ctx, cfg, fps)
            out["rider"] = lambda: _rider(u, ctx, cfg, fps)
        else:
            out["duplicate"] = lambda: _duplicate(u, wholes, cfg, fps)
        return out

    events: list[Event] = []
    for t, g in tracks.items():
        lin = lineage_of_rows(g)
        raw_ids = {r for r, _, _ in lin}
        hint = sctx.hints.get(next(iter(raw_ids))) if len(raw_ids) == 1 else None
        segs = _segments(g, segment_cuts.get(t, []))
        localized = hint is not None and not segs and not (raw_ids & sctx.split_raw_ids)
        whole = _unit(t, g, cfg, fps, localized=hint if localized else None,
                      unlocalized=None if localized else hint)
        seg_units = [_unit(t, s, cfg, fps, unlocalized=hint) for s in segs]
        whole_h = hyps(whole)
        seg_h = [hyps(su) for su in seg_units]
        best = None
        for name, fn in whole_h.items():
            w = fn()
            cand = None
            if not seg_units:
                cand = None if w is None else (w[0], w[1], w[2], None)
            else:
                scores = [h[name]() for h in seg_h]
                sup = [(su, s) for su, s in zip(seg_units, scores, strict=True)
                       if s is not None and s[0] >= sc.reject_below]
                if sup and len(sup) == len(seg_units):
                    cand = None if w is None else (w[0], w[1], w[2], None)
                elif sup:
                    _, top = max(sup, key=lambda p: p[1][0])
                    spans = [[int(su.frames[0]), int(su.frames[-1])] for su, _ in sup]
                    segments = [[int(su.frames[0]), int(su.frames[-1]),
                                 None if s is None else float(s[0])]
                                for su, s in zip(seg_units, scores, strict=True)]
                    signals = {**top[1], "partial": True, "segments": segments}
                    cand = (min(top[0], sc.mixed_score_cap), signals, top[2], spans)
            if cand is not None and (best is None or cand[0] > best[1][0]):
                best = (name, cand)
        if best is None or best[1][0] < sc.reject_below:
            continue
        name, (score, signals, params, spans) = best
        kind = EventKind.RECLASS if name == "rider" else EventKind.DROP
        frames = ((spans[0][0], spans[-1][1]) if spans
                  else (int(g["frame"].iloc[0]), int(g["frame"].iloc[-1])))
        events.append(Event.propose(
            stage=STAGE, kind=kind, tracks=[t], lineage=[lin], frames=frames,
            params={**params, "spans": spans}, algo_score=score,
            signals={**signals, "hypothesis": name},
        ))
    return events
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_screen.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/dnt/refine/hints.py src/dnt/refine/screen.py tests/refine/test_screen.py
git commit -m "feat(refine): add stage 2 screening (static, in-vehicle, rider, duplicate, segments)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: The orphan pass

**Files:**
- Modify: `src/dnt/refine/screen.py` (append)
- Test: `tests/refine/test_orphan.py`

**Interfaces:**
- Produces: `screen.propose_orphans(work, cfg, fps, *, linked_tracks: set[int], pending_endpoints: set[int]) -> tuple[list[Event], list[int]]`. It returns the events and the deferred track IDs. Events use `stage="orphan"` and `params={"reason": "orphan", "spans": None}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_orphan.py
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.screen import propose_orphans

from ._fixtures import box_rows, table


def _work(*rows):
    return io.to_work(table(*rows)).work


def test_short_unlinked_track_is_an_orphan_and_long_is_not():
    w = _work(box_rows(1, range(2), 0.0, 0.0), box_rows(2, range(100), 50.0, 0.0),
              box_rows(3, range(3), 90.0, 0.0))
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                                    pending_endpoints=set())
    by = {e.tracks[0]: e for e in evs}
    assert set(by) == {1, 3} and deferred == []
    assert by[1].algo_score == pytest.approx(0.75)  # 0.2 s -> ramp(0.2; 0.5, 0.1)
    assert by[1].stage == "orphan" and by[1].params == {"reason": "orphan", "spans": None}


def test_linked_and_pending_tracks_are_skipped():
    w = _work(box_rows(1, [0], 0.0, 0.0), box_rows(2, [5], 50.0, 0.0))
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks={1},
                                    pending_endpoints={2})
    assert evs == [] and deferred == [2]


def test_nearly_long_enough_track_scores_below_reject():
    w = _work(box_rows(1, range(4), 0.0, 0.0))  # 0.4 s -> 0.25 < reject 0.3
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    assert evs == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_orphan.py -v`
Expected: FAIL with `ImportError: cannot import name 'propose_orphans'`.

- [ ] **Step 3: Implement (append to `screen.py`)**

```python
def propose_orphans(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, *, linked_tracks: set[int],
    pending_endpoints: set[int],
) -> tuple[list[Event], list[int]]:
    """Propose dropping short unlinked tracks; defer those with a pending link (spec 6.2)."""
    oc = cfg.orphan
    events: list[Event] = []
    deferred: list[int] = []
    for t, g in work.groupby("track", sort=True):
        t = int(t)
        if t in linked_tracks:
            continue
        observed = len(g) / fps
        if observed >= oc.min_seconds:
            continue
        if t in pending_endpoints:
            deferred.append(t)
            continue
        score = float(ramp(observed, *oc.ramp))
        if score < oc.reject_below:
            continue
        g = g.sort_values("frame")
        events.append(Event.propose(
            stage=ORPHAN_STAGE, kind=EventKind.DROP, tracks=[t], lineage=[lineage_of_rows(g)],
            frames=(int(g["frame"].iloc[0]), int(g["frame"].iloc[-1])),
            params={"reason": "orphan", "spans": None}, algo_score=score,
            signals={"observed_seconds": observed},
        ))
    return events, deferred
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_orphan.py tests/refine/test_screen.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/screen.py tests/refine/test_orphan.py
git commit -m "feat(refine): add the orphan pass with deferral for pending links

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 14: Stage 3 — descriptors, gates, scoring, and legacy mode

**Files:**
- Modify: `src/dnt/refine/link.py` (new imports at the top; new code appended after `link_tracklets`)
- Test: `tests/refine/test_link_scoring.py`

**Interfaces:**
- Consumes:
  - Task 5 helpers
  - `primitives.box_centers`, `iob_matrix`, `iou_matrix`, `majority_class`, `ramp`, `span_speed`
  - `features.track_embeddings`
  - `apply.lineage_of_rows`
  - `config.to_frames`
- Produces (in `dnt.refine.link`):
  - `STAGE = "link"`
  - `TrackDesc` (dataclass). Fields: `track, frames, boxes, cls_major, cls_last, h_end, h_start, vel, speed_static, speed_end, speed_start, end_clean, start_clean, lineage`. Properties: `t_s, t_e, start_box, end_box, start_c, end_c`. Method: `legacy() -> dict`.
  - `describe_tracks(work, cfg, fps, occluded) -> dict[int, TrackDesc]`
  - `Candidate(i, j, gate, g, score, signals)`, where `gate` is one of `"normal" | "static" | "occluded" | "overlap"`
  - `score_candidates(work, cfg, fps, *, appearance, context, frame_size, occluded) -> tuple[list[Candidate], dict[int, TrackDesc]]`
  - `legacy_link_events(work, cfg, fps) -> list[Event]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_link_scoring.py
import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.apply import merge_chains
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance
from dnt.refine.link import legacy_link_events, link_tracklets, score_candidates
from dnt.refine.primitives import occlusion_flags

from ._fixtures import box_rows, load_raw, table

A_ = np.eye(8)[0]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _cands(work, cfg=None, fps=25.0, app=None, frame_size=None):
    cfg = cfg or RefineConfig.defaults()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, _ = score_candidates(work, cfg, fps, appearance=app, context=None,
                                frame_size=frame_size, occluded=occ)
    return {(c.i, c.j): c for c in cands}


def _same_app(*spans):
    return ArrayAppearance({t: (list(fr), np.tile(A_, (len(fr), 1))) for t, fr in spans})


def _collinear(cls_a=0, cls_b=0):
    return _work(box_rows(1, range(40), 100.0, 100.0, vx=2.0, cls=cls_a),
                 box_rows(2, range(50, 90), 200.0, 100.0, vx=2.0, cls=cls_b))


def test_collinear_fragments_score_high():
    w = _collinear()
    with_app = _cands(w, app=_same_app((1, range(40)), (2, range(50, 90))))[(1, 2)]
    assert with_app.gate == "normal" and with_app.score == pytest.approx(1 - 0.15 * 11 / 25)
    motion = _cands(w)[(1, 2)]
    assert motion.score == pytest.approx(1 - 0.25 * 11 / 25) and motion.signals["motion_only"]


def test_high_cost_pair_merged_by_link_tracklets_is_not_auto_accepted():
    raw = table(box_rows(1, range(40), 100.0, 100.0), box_rows(2, range(45, 85), 125.0, 100.0))
    assert link_tracklets(raw.copy(), verbose=False)["track"].nunique() == 1
    c = _cands(io.to_work(raw).work)[(1, 2)]
    assert 0.40 <= c.score < 0.80


def test_static_gate_links_a_waiting_pedestrian():
    w = _work(box_rows(1, range(30), 100.0, 100.0), box_rows(2, range(90, 120), 100.0, 100.0))
    c = _cands(w, fps=10.0)[(1, 2)]
    assert c.gate == "static" and c.score == pytest.approx(1 - 0.25 * 61 / 100)  # g = 90 - 29
    cfg = RefineConfig.defaults()
    cfg.link.max_gap_static = 5.0
    assert (1, 2) not in _cands(w, cfg=cfg, fps=10.0)


def test_border_prior_scales_score():
    w = _collinear()
    inside = _cands(w, frame_size=(2000, 2000))[(1, 2)].score
    edge = _cands(w, frame_size=(215, 2000))[(1, 2)].score
    assert edge == pytest.approx(inside * 0.8)


def _turn(truck_frames=range(821, 883), j_start=(160.0, 240.0), j_v=(-5.0, 0.0),
          occluder=(100.0, 150.0, 300.0, 300.0)):
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), *j_start, vx=j_v[0], vy=j_v[1], w=50.0, h=50.0)]
    if truck_frames is not None:
        x, y, w, h = occluder
        rows.append(box_rows(99, truck_frames, x, y, vx=0.5, w=w, h=h))
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    return _cands(_work(*rows), fps=10.0, app=app)


def test_occluded_turn_is_witnessed_and_capped():
    c = _turn()[(1, 2)]
    assert c.gate == "occluded" and c.score == pytest.approx(0.75)
    assert c.signals["witness"] == pytest.approx(1.0) and c.signals["occluders"] == [99]
    assert c.signals["heading"] == pytest.approx(33.69, abs=0.1)


@pytest.mark.parametrize("kwargs", [
    {"truck_frames": None},
    {"truck_frames": range(821, 852)},
    {"j_start": (200.0, 400.0), "j_v": (0.0, 5.0)},
    {"j_start": (1200.0, 300.0), "occluder": (0.0, 0.0, 2000.0, 1000.0)},
])
def test_occluded_gate_rejects(kwargs):
    assert (1, 2) not in _turn(**kwargs)


def test_class_groups_decide_car_truck_links():
    w = _collinear(cls_a=2, cls_b=7)
    assert (1, 2) in _cands(w, cfg=RefineConfig.defaults("vehicle"))
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.class_groups = []
    assert (1, 2) not in _cands(w, cfg=cfg)


def _groups_by_raw(df, id_col):
    return {frozenset(g[id_col].tolist()) for _, g in df.groupby("track")}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_legacy_mode_matches_link_tracklets_grouping(seed):
    raw = load_raw(seed)
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    cfg.link.max_gap = 2.0  # 20 frames at 10 fps, the baseline's max_gap
    work = io.to_work(raw).work
    merged, _ = merge_chains(work, [tuple(e.tracks) for e in legacy_link_events(work, cfg, 10.0)])
    ours = {frozenset(set(g)) for g in merged.groupby("track")["raw_id"].apply(set)}
    out = link_tracklets(raw.copy(), max_gap=20, verbose=False)
    keyed = out.merge(raw[["frame", "x", "y", "w", "h", "track"]].rename(columns={"track": "orig"}),
                      on=["frame", "x", "y", "w", "h"])
    theirs = {frozenset(set(g)) for g in keyed.groupby("track")["orig"].apply(set)}
    assert ours == theirs
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_link_scoring.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_candidates'`.

- [ ] **Step 3: Implement**

Replace the import block at the top of `src/dnt/refine/link.py` (below the module docstring) with:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .apply import lineage_of_rows
from .config import RefineConfig, to_frames
from .events import Event, EventKind
from .features import Appearance, track_embeddings
from .primitives import box_centers, iob_matrix, iou_matrix, majority_class, ramp, span_speed
```

Then append:

```python
STAGE = "link"


@dataclass
class TrackDesc:
    """What stage 3 needs to know about one track (spec 6.3)."""

    track: int
    frames: np.ndarray
    boxes: np.ndarray
    cls_major: int
    cls_last: int
    h_end: float
    h_start: float
    vel: np.ndarray
    speed_static: float
    speed_end: float
    speed_start: float
    end_clean: np.ndarray
    start_clean: np.ndarray
    lineage: list

    @property
    def t_s(self) -> int:
        """First observed frame."""
        return int(self.frames[0])

    @property
    def t_e(self) -> int:
        """Last observed frame."""
        return int(self.frames[-1])

    @property
    def start_box(self) -> np.ndarray:
        """First box."""
        return self.boxes[0]

    @property
    def end_box(self) -> np.ndarray:
        """Last box."""
        return self.boxes[-1]

    @property
    def start_c(self) -> np.ndarray:
        """Center of the first box."""
        return box_centers(self.boxes[0])[0]

    @property
    def end_c(self) -> np.ndarray:
        """Center of the last box."""
        return box_centers(self.boxes[-1])[0]

    def legacy(self) -> dict:
        """Return the descriptor dict ``_legacy_gate_cost`` expects."""
        return {
            "track": self.track, "cls": self.cls_last, "t_start": self.t_s, "t_end": self.t_e,
            "start_c": tuple(map(float, self.start_c)), "end_c": tuple(map(float, self.end_c)),
            "start_box": tuple(map(float, self.start_box)),
            "end_box": tuple(map(float, self.end_box)),
            "area_end": max(float(self.end_box[2] * self.end_box[3]), 1.0),
            "vx": float(self.vel[0]), "vy": float(self.vel[1]),
        }


def describe_tracks(work, cfg: RefineConfig, fps: float, occluded) -> dict[int, TrackDesc]:
    """Build a ``TrackDesc`` per track; ``occluded`` is the row-aligned occlusion mask."""
    lc, hw = cfg.link, cfg.motion.height_window
    out: dict[int, TrackDesc] = {}
    for t, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy(int)
        boxes = g[["x", "y", "w", "h"]].to_numpy(float)
        c = box_centers(boxes)
        vx, vy = _estimate_velocity(frames, c[:, 0], c[:, 1], lc.vel_frames)
        occ = occluded.reindex(g.index, fill_value=False).to_numpy(bool)
        clean = np.flatnonzero(~occ)
        out[int(t)] = TrackDesc(
            track=int(t), frames=frames, boxes=boxes, cls_major=majority_class(g["cls"]),
            cls_last=int(g["cls"].iloc[-1]),
            h_end=max(float(np.median(boxes[-hw:, 3])), 1.0),
            h_start=max(float(np.median(boxes[:hw, 3])), 1.0),
            vel=np.array([vx, vy], dtype=float),
            speed_static=span_speed(frames, boxes, fps, lc.static_seconds, at="end"),
            speed_end=span_speed(frames, boxes, fps, lc.speed_seconds, at="end"),
            speed_start=span_speed(frames, boxes, fps, lc.speed_seconds, at="start"),
            end_clean=boxes[clean[-1]] if len(clean) else boxes[-1],
            start_clean=boxes[clean[0]] if len(clean) else boxes[0],
            lineage=lineage_of_rows(g),
        )
    return out


class _Occluders:
    """Boxes per frame from the work table (owner = track) and the context (owner = -1)."""

    def __init__(self, work, context):
        """Index boxes by frame."""
        parts = [work[["frame", "x", "y", "w", "h"]].assign(owner=work["track"].astype(int))]
        if context is not None and len(context):
            parts.append(context[["frame", "x", "y", "w", "h"]].assign(owner=-1))
        allb = pd.concat(parts, ignore_index=True)
        self.by_frame = {int(f): (g[["x", "y", "w", "h"]].to_numpy(float),
                                  g["owner"].to_numpy(int)) for f, g in allb.groupby("frame")}

    def witness(self, di: TrackDesc, dj: TrackDesc, iob_thr: float) -> tuple[float, list[int]]:
        """Return the fraction of gap frames whose hidden box is covered, and the occluders."""
        n = dj.t_s - di.t_e - 1
        if n <= 0:
            return 0.0, []
        covered, ids = 0, set()
        for f in range(di.t_e + 1, dj.t_s):
            entry = self.by_frame.get(f)
            if entry is None:
                continue
            boxes, owners = entry
            keep = (owners != di.track) & (owners != dj.track)
            if not keep.any():
                continue
            a = (f - di.t_e) / (dj.t_s - di.t_e)
            hidden = (1.0 - a) * di.end_box + a * dj.start_box
            hit = iob_matrix(hidden[None, :], boxes[keep])[0] >= iob_thr
            if hit.any():
                covered += 1
                ids.update(int(o) for o in owners[keep][hit])
        return covered / n, sorted(ids)


@dataclass
class Candidate:
    """A gated end->start pair and its score (spec 6.3)."""

    i: int
    j: int
    gate: str
    g: int
    score: float
    signals: dict = field(default_factory=dict)


def _class_ok(a: int, b: int, groups) -> bool:
    return a == b or any(a in grp and b in grp for grp in groups)


def _size_ok(a, b, r: float) -> bool:
    wr = max(float(b[2]), 1.0) / max(float(a[2]), 1.0)
    hr = max(float(b[3]), 1.0) / max(float(a[3]), 1.0)
    return 1.0 / r <= wr <= r and 1.0 / r <= hr <= r


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _c_app(di: TrackDesc, dj: TrackDesc, appearance: Appearance, k: int, cache: dict) -> float:
    def emb(d):
        if d.track not in cache:
            cache[d.track] = track_embeddings(appearance, d.lineage)
        return cache[d.track]

    fi, ei = emb(di)
    fj, ej = emb(dj)
    if not len(fi) or not len(fj):
        return 0.5
    ei_k = ei[fi <= di.t_e][-k:]
    ej_k = ej[fj >= dj.t_s][:k]
    if not len(ei_k) or not len(ej_k):
        return 0.5
    return float((1.0 - _unit(ei_k.mean(axis=0)) @ _unit(ej_k.mean(axis=0))) / 2.0)


def _prior(di: TrackDesc, dj: TrackDesc, frame_size, margin: float) -> int:
    if frame_size is None:
        return 1
    width, height = frame_size

    def inside(box, m):
        x, y, w, h = box
        return x >= m and y >= m and x + w <= width - m and y + h <= height - m

    return int(inside(di.end_box, margin * di.h_end) and inside(dj.start_box, margin * dj.h_start))


def _overlap_gate(di: TrackDesc, dj: TrackDesc, lc):
    shared = np.intersect1d(di.frames, dj.frames)
    if not len(shared) or not _size_ok(di.end_box, dj.start_box, lc.size_ratio_max):
        return None
    bi = di.boxes[np.searchsorted(di.frames, shared)]
    bj = dj.boxes[np.searchsorted(dj.frames, shared)]
    vals = np.array([iou_matrix(bi[k : k + 1], bj[k : k + 1])[0, 0] for k in range(len(shared))])
    if (vals < lc.overlap_iou).any():
        return None
    return "overlap", float(1.0 - vals.mean()), {"overlap_iou": float(vals.mean())}


def _gate(di: TrackDesc, dj: TrackDesc, g: int, cfg: RefineConfig, fps: float, gaps, occl):
    lc = cfg.link
    mg, mgs, mgo = gaps
    if -lc.overlap_frames <= g <= 0:
        return _overlap_gate(di, dj, lc)
    if g < 1:
        return None
    if g <= mg:
        res = _legacy_gate_cost(
            di.legacy(), dj.legacy(), max_gap=mg, size_ratio_max=lc.size_ratio_max,
            dist_mult=lc.dist_mult, iou_min=lc.iou_min, w_d=lc.legacy_weights["d"],
            w_iou=lc.legacy_weights["iou"], w_s=lc.legacy_weights["s"],
            dist_growth=lc.dist_growth, check_class=False, detail=True,
        )
        if res is None:
            return None
        cost, terms = res
        return "normal", float(ramp(cost, 0.0, lc.legacy_cost_hi)), {"legacy_cost": cost, **terms}
    if di.speed_static < lc.static_speed and g <= mgs:
        if not _size_ok(di.end_box, dj.start_box, lc.size_ratio_max):
            return None
        dist = float(np.linalg.norm(dj.start_c - di.end_c))
        radius = lc.static_radius * di.h_end
        if dist > radius:
            return None
        return "static", dist / radius, {"static_dist": dist}
    if g <= mgo:
        if not _size_ok(di.end_clean, dj.start_clean, lc.size_ratio_max):
            return None
        witness, ids = occl.witness(di, dj, lc.witness_iob)
        if witness < lc.witness_min:
            return None
        chord = dj.start_c - di.end_c
        clen = float(np.linalg.norm(chord))
        speed_i = float(np.linalg.norm(di.vel)) * fps / di.h_end
        heading = None
        if speed_i >= lc.heading_min_speed and clen > 0:
            cosang = float(di.vel @ chord / (np.linalg.norm(di.vel) * clen))
            heading = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            if heading > lc.max_heading_change:
                return None
        v_need = clen / (di.h_end * g / fps)
        v_ref = max(di.speed_end, dj.speed_start, lc.min_feasible_speed)
        if v_need > lc.speed_factor * v_ref:
            return None
        c_mot = (0.5 * v_need / (lc.speed_factor * v_ref)
                 + 0.5 * ((heading or 0.0) / lc.max_heading_change))
        return "occluded", float(c_mot), {"witness": witness, "occluders": ids,
                                          "v_need": v_need, "v_ref": v_ref, "heading": heading}
    return None


def score_candidates(
    work, cfg: RefineConfig, fps: float, *, appearance: Appearance | None, context,
    frame_size, occluded,
) -> tuple[list[Candidate], dict[int, TrackDesc]]:
    """Gate and score every end->start pair (spec 6.3)."""
    lc = cfg.link
    descs = describe_tracks(work, cfg, fps, occluded)
    if len(descs) < 2:
        return [], descs
    occl = _Occluders(work, context)
    gaps = (to_frames(lc.max_gap, fps), to_frames(lc.max_gap_static, fps),
            to_frames(lc.max_gap_occluded, fps))
    order = sorted(descs.values(), key=lambda d: (d.t_s, d.track))
    starts = np.array([d.t_s for d in order])
    motion_only = appearance is None
    cache: dict = {}
    cands: list[Candidate] = []
    for di in sorted(descs.values(), key=lambda d: d.track):
        lo = int(np.searchsorted(starts, di.t_e - lc.overlap_frames, side="left"))
        hi = int(np.searchsorted(starts, di.t_e + gaps[2], side="right"))
        for dj in order[lo:hi]:
            if dj.track == di.track or not _class_ok(di.cls_major, dj.cls_major, lc.class_groups):
                continue
            g = dj.t_s - di.t_e
            res = _gate(di, dj, g, cfg, fps, gaps, occl)
            if res is None:
                continue
            gate, c_mot, sig = res
            w = dict(lc.weights_occluded if gate == "occluded" else lc.weights)
            c_app = None
            if motion_only:
                total = w["mot"] + w["gap"]
                w = {"mot": w["mot"] / total, "gap": w["gap"] / total, "app": 0.0}
            else:
                c_app = _c_app(di, dj, appearance, lc.k_embed, cache)
            limit = {"normal": gaps[0], "overlap": gaps[0], "static": gaps[1],
                     "occluded": gaps[2]}[gate]
            c_gap = max(g, 0) / limit
            b = _prior(di, dj, frame_size, lc.border_margin)
            cost = w["mot"] * c_mot + w["gap"] * c_gap + w["app"] * (c_app or 0.0)
            s = float(np.clip((1.0 - cost) * (0.8 + 0.2 * b), 0.0, 1.0))
            if gate == "occluded":
                s = min(s, lc.occluded_score_cap)
            cands.append(Candidate(di.track, dj.track, gate, int(g), s, {
                **sig, "gate": gate, "g": int(g), "c_mot": c_mot, "c_app": c_app,
                "c_gap": c_gap, "b": b, "motion_only": motion_only,
            }))
    return cands, descs


def legacy_link_events(work, cfg: RefineConfig, fps: float) -> list[Event]:
    """Return ``link_tracklets``'s matches as LINK proposals (``link.mode: legacy``)."""
    lc = cfg.link
    if work.empty:
        return []
    df = _prepare_legacy(work[LEGACY_COL_NAMES], LEGACY_COL_NAMES)
    stitchable = [d for d in _legacy_descriptors(df, lc.vel_frames).values()
                  if d.get("stitchable")]
    if len(stitchable) <= 1:
        return []
    matches = _legacy_matches(
        stitchable, max_gap=to_frames(lc.max_gap, fps), size_ratio_max=lc.size_ratio_max,
        dist_mult=lc.dist_mult, iou_min=lc.iou_min, w_d=lc.legacy_weights["d"],
        w_iou=lc.legacy_weights["iou"], w_s=lc.legacy_weights["s"], dist_growth=lc.dist_growth,
    )
    by = {d["track"]: d for d in stitchable}
    events = []
    for a, b, cost in matches:
        t_e, t_s = by[a]["t_end"], by[b]["t_start"]
        events.append(Event.propose(
            stage=STAGE, kind=EventKind.LINK, tracks=[a, b],
            lineage=[lineage_of_rows(work[work["track"] == a]),
                     lineage_of_rows(work[work["track"] == b])],
            frames=(t_e, t_s), params={"gate": "legacy", "gap": [t_e, t_s]}, algo_score=1.0,
            signals={"legacy_cost": cost, "pass": 1},
        ))
    return events
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_link_scoring.py tests/refine/test_link_legacy.py tests/test_refine_independence.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/link.py tests/refine/test_link_scoring.py
git commit -m "feat(refine): add stage 3 gates and scoring (normal, static, occluded, overlap) + legacy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Stage 3 — assignment passes, frozen pairs, chains

**Files:**
- Modify: `src/dnt/refine/link.py` (append; the import block becomes the one shown in Step 3)
- Test: `tests/refine/test_link_passes.py`

**Interfaces:**
- Produces (in `dnt.refine.link`):
  - `assign_in_passes(cands, cfg, *, make_event: Callable[[Candidate, float, dict], Event], route: Callable[[list[Event]], None]) -> list[Event]`
  - `resolve_chains(accepted, descs, overlap_frames) -> tuple[list[tuple[int, int]], list[Event]]`
  - `LinkStageResult(events, accepted, pending_endpoints, skipped)`
  - `run_link_stage(work, cfg, fps, *, appearance, context, frame_size, occluded, route) -> LinkStageResult`
  - LINK event `signals` add `S_link, margin, pass, alternatives` (a list of `{"i", "j", "score"}`) and, when a rejection freed an endpoint, `replaces` (the rejected event's `proposal_key`). `params = {"gate": ..., "gap": [t_e, t_s]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_link_passes.py
import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.link import (
    Candidate,
    assign_in_passes,
    describe_tracks,
    resolve_chains,
    run_link_stage,
)
from dnt.refine.primitives import occlusion_flags
from dnt.refine.verify import Band, decide, route_without_vlm

from ._fixtures import box_rows, table


def _make_event(c, routed, extra):
    return Event.propose(stage="link", kind=EventKind.LINK, tracks=[c.i, c.j],
                         lineage=[[[c.i, 0, 9]], [[c.j, 20, 29]]], frames=(9, 20),
                         params={"gate": c.gate, "gap": [9, 20]}, algo_score=routed,
                         signals={**c.signals, **extra})


def _route(decisions):
    def route(evs):
        for ev in evs:
            decide(ev, decisions.get(tuple(ev.tracks), Decision.AUTO_ACCEPT), source="auto")
    return route


def _c(i, j, s):
    return Candidate(i, j, "normal", 5, s, {})


def test_rejected_edge_frees_endpoint_for_next_pass():
    evs = assign_in_passes([_c(1, 2, 0.9), _c(1, 3, 0.7)], RefineConfig.defaults(),
                           make_event=_make_event, route=_route({(1, 2): Decision.VLM_REJECT}))
    assert [e.tracks for e in evs] == [[1, 2], [1, 3]]
    assert evs[1].signals["pass"] == 2 and evs[1].signals["replaces"] == evs[0].proposal_key
    assert evs[1].decision is Decision.AUTO_ACCEPT


def test_pending_edge_reserves_its_endpoints():
    evs = assign_in_passes([_c(1, 2, 0.9), _c(1, 3, 0.7)], RefineConfig.defaults(),
                           make_event=_make_event,
                           route=_route({(1, 2): Decision.HUMAN_PENDING}))
    assert [e.tracks for e in evs] == [[1, 2]]


def test_accepted_pairs_are_frozen_in_later_passes():
    cands = [_c(1, 10, 0.9), _c(2, 10, 0.95), _c(2, 11, 0.85), _c(1, 12, 0.6)]
    evs = assign_in_passes(cands, RefineConfig.defaults(), make_event=_make_event,
                           route=_route({(2, 11): Decision.AUTO_REJECT}))
    assert sorted(e.tracks for e in evs) == [[1, 10], [2, 11]]
    assert not any(e.tracks == [2, 10] for e in evs)


def test_ambiguous_pair_is_capped_and_lists_its_alternative():
    cfg = RefineConfig.defaults()
    evs = assign_in_passes([_c(1, 2, 0.9), _c(1, 3, 0.9)], cfg, make_event=_make_event,
                           route=lambda e: route_without_vlm(e, Band.of(cfg.link)))
    assert len(evs) == 1
    ev = evs[0]
    assert ev.algo_score == pytest.approx(0.75) and ev.decision is Decision.HUMAN_PENDING
    assert ev.signals["margin"] == pytest.approx(0.0)
    assert len(ev.signals["alternatives"]) == 1


def test_clear_winner_keeps_its_score():
    evs = assign_in_passes([_c(1, 2, 0.9)], RefineConfig.defaults(), make_event=_make_event,
                           route=_route({}))
    assert evs[0].algo_score == pytest.approx(0.9) and evs[0].signals["alternatives"] == []


def _descs(*rows):
    w = io.to_work(table(*rows)).work
    return describe_tracks(w, RefineConfig.defaults(), 10.0, occlusion_flags(w, None, 0.3))


def _acc(i, j, score):
    ev = _make_event(_c(i, j, score), score, {})
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    ev.applied = True
    return ev


def test_chain_with_a_repeated_frame_skips_its_weakest_link():
    descs = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(11, 21), 30.0, 0.0),
                   box_rows(3, range(5, 31), 60.0, 0.0))
    a, b = _acc(1, 2, 0.9), _acc(2, 3, 0.8)
    pairs, skipped = resolve_chains([a, b], descs, overlap_frames=2)
    assert pairs == [(1, 2)] and skipped == [b]
    assert b.applied is False and b.signals["skipped_reason"] == "overlap"


def test_small_allowed_overlap_is_kept():
    descs = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(9, 21), 30.0, 0.0))
    pairs, skipped = resolve_chains([_acc(1, 2, 0.9)], descs, overlap_frames=2)
    assert pairs == [(1, 2)] and skipped == []


def test_run_link_stage_links_collinear_fragments():
    w = io.to_work(table(box_rows(1, range(40), 100.0, 100.0, vx=2.0),
                         box_rows(2, range(50, 90), 200.0, 100.0, vx=2.0))).work
    cfg = RefineConfig.defaults()
    res = run_link_stage(w, cfg, 25.0, appearance=None, context=None, frame_size=None,
                         occluded=occlusion_flags(w, None, 0.3),
                         route=lambda e: route_without_vlm(e, Band.of(cfg.link)))
    assert res.accepted == [(1, 2)] and res.events[0].applied is True
    assert res.pending_endpoints == set()
    assert np.isclose(res.events[0].algo_score, 1 - 0.25 * 11 / 25)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_link_passes.py -v`
Expected: FAIL with `ImportError: cannot import name 'assign_in_passes'`.

- [ ] **Step 3: Implement (update the imports, then append to `link.py`)**

The import block at the top of `link.py` becomes:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .apply import lineage_of_rows
from .config import RefineConfig, to_frames
from .events import ACCEPTED, REJECTED, Decision, Event, EventKind
from .features import Appearance, track_embeddings
from .primitives import box_centers, iob_matrix, iou_matrix, majority_class, ramp, span_speed
```

Append:

```python
@dataclass
class LinkStageResult:
    """Stage 3 output."""

    events: list[Event]
    accepted: list[tuple[int, int]]
    pending_endpoints: set[int]
    skipped: list[Event]


def _components(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(("e", i))] = find(("s", j))
    groups: dict = {}
    for e in edges:
        groups.setdefault(find(("e", e[0])), []).append(e)
    return sorted((sorted(v) for v in groups.values()), key=lambda v: v[0])


def _hungarian(comp: list[tuple[int, int]], avail: dict) -> list[tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment

    ends = sorted({e[0] for e in comp})
    starts = sorted({e[1] for e in comp})
    m = np.full((len(ends), len(starts)), -1.0)
    for i, j in comp:
        m[ends.index(i), starts.index(j)] = avail[(i, j)].score
    rows, cols = linear_sum_assignment(np.where(m >= 0, -m, 1e6))
    return [(ends[a], starts[b]) for a, b in zip(rows, cols, strict=True) if m[a, b] >= 0]


def assign_in_passes(
    cands: list[Candidate], cfg: RefineConfig, *,
    make_event: Callable[[Candidate, float, dict], Event],
    route: Callable[[list[Event]], None],
) -> list[Event]:
    """Assign pairs pass by pass; rejected edges free their endpoints (spec 6.3)."""
    lc = cfg.link
    edges = {(c.i, c.j): c for c in cands if c.score >= lc.reject_below}
    by_edge: dict[tuple[int, int], Event] = {}
    rejected_key: dict[int, str] = {}
    events: list[Event] = []
    for p in range(1, lc.max_passes + 1):
        live = [e for e, ev in by_edge.items() if ev.decision not in REJECTED]
        busy_end = {e[0] for e in live}
        busy_start = {e[1] for e in live}
        avail = {e: c for e, c in edges.items()
                 if e not in by_edge and e[0] not in busy_end and e[1] not in busy_start}
        if not avail:
            break
        chosen = [pair for comp in _components(sorted(avail)) for pair in _hungarian(comp, avail)]
        if not chosen:
            break
        new: list[Event] = []
        for i, j in chosen:
            c = avail[(i, j)]
            alts = sorted((a for e, a in avail.items() if e != (i, j) and (e[0] == i or e[1] == j)),
                          key=lambda a: (-a.score, a.i, a.j))
            margin = c.score - alts[0].score if alts else c.score
            routed = c.score if margin >= lc.margin_min else min(c.score, lc.ambiguous_cap)
            extra = {"S_link": c.score, "margin": margin, "pass": p,
                     "alternatives": [{"i": a.i, "j": a.j, "score": a.score}
                                      for a in alts[: lc.n_alternatives]]}
            replaced = rejected_key.get(i) or rejected_key.get(j)
            if replaced:
                extra["replaces"] = replaced
            ev = make_event(c, routed, extra)
            by_edge[(i, j)] = ev
            new.append(ev)
        route(new)
        events.extend(new)
        freed = [ev for ev in new if ev.decision in REJECTED]
        for ev in freed:
            for t in ev.tracks:
                rejected_key[int(t)] = ev.proposal_key
        if not freed:
            break
    return events


def _first_conflict(active: list[Event], descs: dict[int, TrackDesc], overlap_frames: int):
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in active:
        parent[find(e.tracks[1])] = find(e.tracks[0])
    chains: dict[int, list[Event]] = {}
    for e in active:
        chains.setdefault(find(e.tracks[0]), []).append(e)
    for evs in chains.values():
        members = sorted({t for e in evs for t in e.tracks}, key=lambda t: (descs[t].t_s, t))
        linked = {tuple(e.tracks) for e in evs}
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                x, y = members[a], members[b]
                shared = np.intersect1d(descs[x].frames, descs[y].frames)
                if not len(shared):
                    continue
                if (x, y) in linked and len(shared) <= overlap_frames + 1:
                    continue
                return min(evs, key=lambda e: (e.algo_score, e.proposal_key))
    return None


def resolve_chains(
    accepted: list[Event], descs: dict[int, TrackDesc], overlap_frames: int
) -> tuple[list[tuple[int, int]], list[Event]]:
    """Drop the weakest link of any chain that would repeat a frame (spec 6.3)."""
    active = list(accepted)
    skipped: list[Event] = []
    while True:
        bad = _first_conflict(active, descs, overlap_frames)
        if bad is None:
            return [(int(e.tracks[0]), int(e.tracks[1])) for e in active], skipped
        bad.applied = False
        bad.signals["skipped_reason"] = "overlap"
        active = [e for e in active if e is not bad]
        skipped.append(bad)


def run_link_stage(
    work, cfg: RefineConfig, fps: float, *, appearance: Appearance | None, context, frame_size,
    occluded, route: Callable[[list[Event]], None],
) -> LinkStageResult:
    """Propose, route, and resolve LINK events for the current work table."""
    if cfg.link.mode == "legacy":
        evs = legacy_link_events(work, cfg, fps)
        route(evs)
        acc = [e for e in evs if e.decision in ACCEPTED]
        for e in acc:
            e.applied = True
        return LinkStageResult(evs, [(e.tracks[0], e.tracks[1]) for e in acc], set(), [])
    cands, descs = score_candidates(work, cfg, fps, appearance=appearance, context=context,
                                    frame_size=frame_size, occluded=occluded)

    def make_event(c: Candidate, routed: float, extra: dict) -> Event:
        di, dj = descs[c.i], descs[c.j]
        return Event.propose(
            stage=STAGE, kind=EventKind.LINK, tracks=[c.i, c.j], lineage=[di.lineage, dj.lineage],
            frames=(di.t_e, dj.t_s), params={"gate": c.gate, "gap": [di.t_e, dj.t_s]},
            algo_score=routed, signals={**c.signals, **extra},
        )

    events = assign_in_passes(cands, cfg, make_event=make_event, route=route)
    acc = [e for e in events if e.decision in ACCEPTED]
    pairs, skipped = resolve_chains(acc, descs, cfg.link.overlap_frames)
    skipped_ids = {id(e) for e in skipped}
    for e in acc:
        e.applied = id(e) not in skipped_ids
    pending = {t for e in events if e.decision is Decision.HUMAN_PENDING for t in e.tracks}
    return LinkStageResult(events, pairs, pending, skipped)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_link_passes.py tests/refine/test_link_scoring.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/link.py tests/refine/test_link_passes.py
git commit -m "feat(refine): add stage 3 assignment passes, frozen pairs, and chain resolution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 16: `TrackRefiner` (refine, refine_batch) and stage 4

**Files:**
- Create: `src/dnt/refine/refiner.py`
- Test: `tests/refine/test_refiner.py`

**Interfaces:**
- Consumes: everything above.
- Produces (in `dnt.refine.refiner`):
  - `RefineResult(tracks, ledger_path, review_path, summary, events)`
  - `output_paths(out) -> dict[str, Path]`, with keys `ledger`, `review`, `features`
  - `resolve_fps(arg, cfg_fps, video_fps) -> tuple[float, str]`
  - `table_summary(work, fps) -> dict`
  - `fill_stage(work, cfg, fps, protected) -> tuple[pd.DataFrame, list[Event]]`
  - `TrackRefiner(config=None, config_yaml=None, device=None, *, appearance_factory=None)`. The shape follows `Tracker` (spec 2.4): pass `config` or `config_yaml`, not both, and `device` overrides `config.encoder.device`.
    - `refine(track_file, out_file, video_file=None, context_file=None, reclass_file=None, fps=None, fmt="dnt", video_index=None, video_tot=None, message="", verbose=True) -> pd.DataFrame`. It returns the refined table, like `Tracker.track()`. The full result is on `refiner.last_result: RefineResult`.
    - `refine_batch(track_files, video_files=None, output_path=None, context_files=None, reclass_files=None, fps=None, is_overwrite=False, is_report=True, message="", verbose=True) -> list[str]`. Files are paired by position. Outputs are `<output_path>/<base>_refined.txt`, where `<base>` is the stem without a trailing `_track`. Existing outputs are skipped unless `is_overwrite`, and skipped ones are still listed when `is_report`. `output_path` is required.
    - `appearance_factory(work=, video=, context=, fps=, config=)` returns `Appearance | None`. It is the seam Plan 2 fills.
  - Ledger header keys: `format, dnt_version, round, parent, config, inputs{tracks, video, context, hints, features}, fps, fps_source, frame_size, id_map, n_filled_input_rows_removed, smoothing, summary`.
  - Summary keys: `before, after, events` (`"stage/kind/decision" -> count`), `vlm`, `orphan_deferred` (final IDs), `filled_input_rows_removed`, `duplicate_input_rows_removed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_refiner.py
import logging
import sys

import numpy as np
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.refiner import TrackRefiner

from ._fixtures import box_rows, table


def _write(dirpath, df, name="t.txt"):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    df.to_csv(p, index=False, header=False)
    return p


def _refine(src, out, cfg=None, **kw):
    refiner = TrackRefiner(cfg)
    df = refiner.refine(src, out, verbose=False, **kw)
    assert df is refiner.last_result.tracks
    return refiner.last_result


def _crit1_rows(offset=0):
    rows = []
    for f in range(60):  # track 1: a car taken over by an untracked truck at frame 30
        truck = f >= 30
        cls = 5 if f == 30 else 7 if (f == 28 or f > 30) else 2
        rows.append([f + offset, 1, 100.0 + f, 100.0, 95.0 if truck else 53.0,
                     60.0 if truck else 56.0, 0.9, cls, -1, -1])
    rows += box_rows(2, [f + offset for f in range(120)], 400.0, 300.0, score=0.35)  # static
    rows += box_rows(3, [f + offset for f in range(40)], 600.0, 200.0, vx=3.0)  # fragment
    rows += box_rows(4, [f + offset for f in range(45, 90)], 735.0, 200.0, vx=3.0)
    return rows


def test_default_config_no_video_all_three_stages_score(tmp_path, monkeypatch):
    for name in ("transformers", "torchreid", "openai", "anthropic"):
        monkeypatch.setitem(sys.modules, name, None)  # a minimal install
    src = _write(tmp_path, table(_crit1_rows()))
    res = _refine(src, tmp_path / "o.csv", fps=10)
    assert {"switch", "screen", "link"} <= {e.stage for e in res.events}
    for e in res.events:
        if e.stage in ("switch", "screen"):
            assert e.decision is Decision.HUMAN_PENDING
    link = [e for e in res.events if e.stage == "link"]
    assert link[0].decision is Decision.AUTO_ACCEPT and link[0].applied
    header = Ledger.read(res.ledger_path).header
    assert header["fps"] == 10 and header["fps_source"] == "argument"
    assert header["inputs"]["tracks"]["sha256"] == io.sha256_file(src)
    out = pd.read_csv(tmp_path / "o.csv", header=None)
    assert out.shape[1] == 10
    assert sorted(out[1].unique()) == list(range(1, out[1].nunique() + 1))


def test_frame_offset_does_not_change_scores(tmp_path):
    def run(offset):
        d = tmp_path / f"o{offset}"
        res = _refine(_write(d, table(_crit1_rows(offset))), d / "o.csv", fps=10)
        return sorted((e.stage, str(e.kind), round(e.algo_score, 9)) for e in res.events)

    assert run(0) == run(120000)


def test_no_video_and_no_fps_fails_before_processing(tmp_path):
    src = _write(tmp_path, table(box_rows(1, range(5), 0.0, 0.0)))
    with pytest.raises(ValueError, match="fps="):
        _refine(src, tmp_path / "o.csv")
    assert not (tmp_path / "o.csv").exists()


def test_empty_track_file(tmp_path):
    (tmp_path / "e.txt").write_text("")
    res = _refine(tmp_path / "e.txt", tmp_path / "o.csv", fps=10)
    assert (tmp_path / "o.csv").read_text() == ""
    assert len(res.ledger_path.read_text().splitlines()) == 1 and res.events == []


def test_short_fragment_with_pending_link_is_deferred_not_dropped(tmp_path):
    rows = (box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0)
            + box_rows(2, [883], 160.0, 240.0, w=50.0, h=50.0)
            + box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=300.0))
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", fps=10)
    link = [e for e in res.events if e.stage == "link" and e.tracks == [1, 2]]
    assert link and link[0].decision is Decision.HUMAN_PENDING
    assert link[0].params["gate"] == "occluded"
    assert not any(e.stage == "orphan" for e in res.events)
    assert (pd.read_csv(tmp_path / "o.csv", header=None)[0] == 883).any()
    id_map = Ledger.read(res.ledger_path).header["id_map"]
    assert res.summary["orphan_deferred"] == [id_map["2"]]


def test_filled_input_rows_are_removed_and_refilled_once(tmp_path):
    df = table(box_rows(1, range(0, 10), 0.0, 0.0, vx=2.0),
               box_rows(1, range(15, 25), 30.0, 0.0, vx=2.0))
    filled = table(box_rows(1, range(10, 15), 20.0, 0.0))
    filled["r3"] = 1
    src = _write(tmp_path, pd.concat([df, filled]))
    res = _refine(src, tmp_path / "o.csv", fps=10)
    assert res.summary["filled_input_rows_removed"] == 5
    fills = [e for e in res.events if e.kind is EventKind.FILL]
    assert len(fills) == 1 and fills[0].params == {"gap": [9, 15], "n_rows": 5}
    assert fills[0].decision is Decision.AUTO_ACCEPT and fills[0].id == "fill-r0-000001"


def test_smoothing_writes_smooth_records(tmp_path):
    cfg = RefineConfig.defaults()
    cfg.fill.smooth_existing = True
    rng = np.random.default_rng(0)
    rows = box_rows(1, range(50), 100.0, 100.0, vx=2.0)
    for r in rows:
        r[2] += float(rng.normal(0, 3))
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", cfg=cfg, fps=10)
    sm = [e for e in res.events if e.kind is EventKind.SMOOTH]
    assert len(sm) == 1 and sm[0].params["n_rows"] > 0 and sm[0].signals["max_shift_px"] > 0
    assert Ledger.read(res.ledger_path).header["smoothing"] is True


def test_video_supplies_fps_frame_size_and_fingerprint(tmp_path, synthetic_video, caplog):
    video, _ = synthetic_video
    src = _write(tmp_path, table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)))
    with caplog.at_level(logging.WARNING):
        res = _refine(src, tmp_path / "o.csv", video_file=video)
    h = Ledger.read(res.ledger_path).header
    assert h["fps"] == pytest.approx(25.0) and h["fps_source"] == "video"
    assert h["frame_size"] == [320, 240]
    assert h["inputs"]["video"]["fingerprint"]["sha256"] == io.sha256_file(video)
    assert "motion-only" in caplog.text


def test_track_frames_beyond_the_video_are_rejected(tmp_path, synthetic_video):
    video, _ = synthetic_video
    src = _write(tmp_path, table(box_rows(1, range(495, 505), 0.0, 0.0)))
    with pytest.raises(ValueError, match="frame count"):
        _refine(src, tmp_path / "o.csv", video_file=video)


def test_refiner_api_matches_tracker(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg = RefineConfig.defaults("vehicle")
    cfg.to_yaml(cfg_file)
    refiner = TrackRefiner(config_yaml=str(cfg_file), device="cpu")
    assert refiner.config.target == "vehicle" and refiner.config.encoder.device == "cpu"
    with pytest.raises(ValueError, match="not both"):
        TrackRefiner(config=cfg, config_yaml=str(cfg_file))
    src = _write(tmp_path, table(box_rows(1, range(40), 100.0, 100.0, vx=2.0)))
    df = refiner.refine(src, tmp_path / "o.csv", fps=10, verbose=False)
    assert isinstance(df, pd.DataFrame) and refiner.last_result.ledger_path.exists()


def test_refine_batch_names_skips_and_overwrites(tmp_path):
    srcs = [_write(tmp_path / "in", table(box_rows(1, range(30), 0.0, 0.0, vx=2.0)), f"cam{i}_track.txt")
            for i in (1, 2)]
    out_dir = tmp_path / "out"
    refiner = TrackRefiner()
    with pytest.raises(ValueError, match="output_path"):
        refiner.refine_batch(srcs, fps=10)
    first = refiner.refine_batch(srcs, output_path=out_dir, fps=10, verbose=False)
    assert first == [str(out_dir / "cam1_refined.txt"), str(out_dir / "cam2_refined.txt")]
    assert (out_dir / "cam1_refined.ledger.jsonl").exists()
    stamp = (out_dir / "cam1_refined.txt").stat().st_mtime_ns
    assert refiner.refine_batch(srcs, output_path=out_dir, fps=10, verbose=False) == first
    assert refiner.refine_batch(srcs, output_path=out_dir, fps=10, is_report=False,
                                verbose=False) == []
    assert (out_dir / "cam1_refined.txt").stat().st_mtime_ns == stamp
    again = refiner.refine_batch(srcs[:1], output_path=out_dir, fps=10, is_overwrite=True,
                                 verbose=False)
    assert again == first[:1]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_refiner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnt.refine.refiner'`.

- [ ] **Step 3: Implement**

```python
# src/dnt/refine/refiner.py
"""``TrackRefiner``: runs the refinement stages in order and writes the outputs (spec 3)."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import __version__
from . import io
from .apply import apply_edit, lineage_of_rows, merge_chains, next_track_id, renumber
from .config import RefineConfig, to_frames
from .events import ACCEPTED, Decision, Event, EventKind, Ledger
from .features import Appearance
from .hints import read_reclass_hints
from .interpolate import interpolate_tracks_rts
from .link import run_link_stage
from .primitives import box_centers, frame_runs, majority_class, occlusion_flags, speeds_hps
from .screen import ScreenContext, propose_orphans, propose_screen
from .switch import propose_splits
from .verify import Band, decide, route_without_vlm

log = logging.getLogger(__name__)
LEDGER_FORMAT = "dnt.refine.ledger/1"


@dataclass
class RefineResult:
    """What ``TrackRefiner.refine`` produced."""

    tracks: pd.DataFrame
    ledger_path: Path
    review_path: Path | None
    summary: dict
    events: list[Event] = field(default_factory=list)


def output_paths(out) -> dict[str, Path]:
    """Return the ledger, review, and feature-cache paths that sit next to ``out`` (spec 2.5)."""
    out = Path(out)
    return {"ledger": out.with_suffix(".ledger.jsonl"),
            "review": out.with_suffix(".review.html"),
            "features": out.with_suffix(".features.npz")}


def resolve_fps(arg, cfg_fps, video_fps) -> tuple[float, str]:
    """Resolve the frame rate: argument, then config, then video (spec 5.1)."""
    explicit, source = (arg, "argument") if arg is not None else (cfg_fps, "config")
    if explicit is not None:
        explicit = float(explicit)
        if explicit <= 0:
            raise ValueError(f"fps must be positive, got {explicit}")
        if video_fps and abs(explicit - video_fps) / video_fps > 0.01:
            log.warning("fps %.3f (%s) differs from the video's %.3f by more than 1%%; "
                        "using %.3f", explicit, source, video_fps, explicit)
        return explicit, source
    if video_fps and video_fps > 0:
        return float(video_fps), "video"
    raise ValueError("no frame rate: pass fps=... (or --fps) when no video is given; "
                     "every threshold in seconds depends on it")


def _file_record(path, **extra) -> dict:
    p = Path(path)
    return {"path": str(path), "abs_path": str(p.resolve()), "sha256": io.sha256_file(p),
            **extra}


def table_summary(work: pd.DataFrame, fps: float) -> dict:
    """Summarize a work table for the ledger header (spec 8.3)."""
    if work.empty:
        return {"tracks": 0, "tracks_per_class": {}, "observed_rows": 0, "interpolated_rows": 0,
                "median_track_seconds": 0.0}
    per_track = work.groupby("track")["cls"].agg(majority_class)
    obs = work[work["interp"] == 0]
    dur = obs.groupby("track")["frame"].agg(lambda f: (f.max() - f.min() + 1) / fps)
    return {
        "tracks": int(work["track"].nunique()),
        "tracks_per_class": {str(k): int(v)
                             for k, v in per_track.value_counts().sort_index().items()},
        "observed_rows": len(obs),
        "interpolated_rows": int((work["interp"] == 1).sum()),
        "median_track_seconds": float(dur.median()) if len(dur) else 0.0,
    }


def _event_counts(events: list[Event]) -> dict[str, int]:
    c = Counter(f"{e.stage}/{e.kind}/{e.decision}" for e in events)
    return dict(sorted(c.items()))


def fill_stage(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, protected: dict[int, list[tuple[int, int]]]
) -> tuple[pd.DataFrame, list[Event]]:
    """Run stage 4; return the filled table and its FILL / SMOOTH records (spec 6.4)."""
    if work.empty:
        return work, []
    max_gap_s = cfg.fill.max_gap if cfg.fill.max_gap is not None else cfg.link.max_gap
    before = work[io.WORK_COLUMNS].reset_index(drop=True)
    out = interpolate_tracks_rts(
        tracks=before.copy(), fill_gaps_only=True, smooth_existing=cfg.fill.smooth_existing,
        process_var=cfg.motion.process_var, meas_var_pos=cfg.motion.meas_var_pos,
        meas_var_size=cfg.motion.meas_var_size, max_gap=to_frames(max_gap_s, fps),
        verbose=False, protected_gaps=protected,
    )
    out = out.sort_values(["track", "frame"]).reset_index(drop=True)
    out["raw_id"] = out.groupby("track")["raw_id"].ffill().astype(int)
    events: list[Event] = []
    for t, g in out.groupby("track", sort=True):
        obs = g[g["interp"] == 0]
        lin = lineage_of_rows(obs)
        for a, b in frame_runs(g.loc[g["interp"] == 1, "frame"]):
            f_before = int(obs.loc[obs["frame"] < a, "frame"].iloc[-1])
            f_after = int(obs.loc[obs["frame"] > b, "frame"].iloc[0])
            seg = g[(g["frame"] >= f_before) & (g["frame"] <= f_after)]
            boxes = seg[["x", "y", "w", "h"]].to_numpy(float)
            v = speeds_hps(seg["frame"].to_numpy(), boxes, fps, cfg.motion.height_window)
            ends = box_centers(boxes[[0, -1]])
            h = max(float(np.median(boxes[:, 3])), 1.0)
            ev = Event.propose(
                stage="fill", kind=EventKind.FILL, tracks=[int(t)], lineage=[lin],
                frames=(f_before, f_after),
                params={"gap": [f_before, f_after], "n_rows": int(b - a + 1)}, algo_score=1.0,
                signals={"gap_seconds": (f_after - f_before - 1) / fps,
                         "chord_h": float(np.linalg.norm(ends[1] - ends[0]) / h),
                         "max_fill_speed_h_s": float(np.nanmax(v[1:])) if len(v) > 1 else 0.0},
            )
            decide(ev, Decision.AUTO_ACCEPT, source="auto")
            ev.applied = True
            events.append(ev)
    if cfg.fill.smooth_existing:
        after = out.loc[out["interp"] == 0, ["track", "frame", "x", "y", "w", "h"]]
        m = before.merge(after, on=["track", "frame"], suffixes=("_0", "_1"))
        for t, g in m.groupby("track", sort=True):
            c0 = box_centers(g[["x_0", "y_0", "w_0", "h_0"]].to_numpy(float))
            c1 = box_centers(g[["x_1", "y_1", "w_1", "h_1"]].to_numpy(float))
            shift = np.linalg.norm(c1 - c0, axis=1)
            if not (shift > 0).any():
                continue
            k = int(np.argmax(shift))
            f = g["frame"].to_numpy(int)
            ev = Event.propose(
                stage="fill", kind=EventKind.SMOOTH, tracks=[int(t)],
                lineage=[lineage_of_rows(before[before["track"] == t])],
                frames=(int(f.min()), int(f.max())), params={"n_rows": int((shift > 0).sum())},
                algo_score=1.0,
                signals={"mean_shift_px": float(shift.mean()), "max_shift_px": float(shift[k]),
                         "max_shift_frame": int(f[k])},
            )
            decide(ev, Decision.AUTO_ACCEPT, source="auto")
            ev.applied = True
            events.append(ev)
    return out, events


class _Stages:
    """Runs stages 1-4 on a work table and collects their events (spec 3)."""

    def __init__(self, cfg: RefineConfig, fps: float, frame_size, appearance, ctx_boxes,
                 ctx_fmt, hints):
        """Hold the per-run inputs."""
        self.cfg, self.fps, self.frame_size = cfg, fps, frame_size
        self.appearance, self.ctx_boxes, self.ctx_fmt, self.hints = (
            appearance, ctx_boxes, ctx_fmt, hints)
        self.seq: Counter = Counter()
        self.events: list[Event] = []
        self.orphan_deferred: list[int] = []

    def _route(self, evs: list[Event], band: Band, stage: str) -> None:
        route_without_vlm(evs, band)
        for e in evs:
            self.seq[stage] += 1
            e.id = f"{stage}-r0-{self.seq[stage]:06d}"
        self.events.extend(evs)

    def run(self, work: pd.DataFrame, tick: Callable[[str], None] | None = None
            ) -> tuple[pd.DataFrame, list[Event]]:
        """Run every enabled stage in the spec's order; ``tick(name)`` follows each stage."""
        tick = tick or (lambda _name: None)
        occluded = occlusion_flags(work, self.ctx_boxes, self.cfg.encoder.occlusion_iou)
        work, split_raw, cuts = self._switch(work)
        tick("switch")
        work = self._screen(work, split_raw, cuts)
        tick("screen")
        work, protected, linked, pending = self._link(work, occluded)
        tick("link")
        work = self._orphans(work, linked, pending)
        tick("orphan")
        work = self._fill(work, protected)
        tick("fill")
        return work, self.events

    def _switch(self, work):
        cfg = self.cfg
        if not cfg.switch.enabled or work.empty:
            return work, set(), {}
        res = propose_splits(work, cfg, self.fps, self.appearance)
        self._route(res.events, Band.of(cfg.switch), "switch")

        def raw_at(track, frame):
            rows = work.loc[(work["track"] == track) & (work["frame"] == frame), "raw_id"]
            return int(rows.iloc[0])

        cut_points = [(raw_at(t, c), c) for t, cs in res.weak_cuts.items() for c in cs]
        accepted = []
        for ev in res.events:
            t, c = ev.tracks[0], int(ev.params["cut_frame"])
            if ev.decision in ACCEPTED:
                accepted.append(ev)
            elif ev.decision is Decision.HUMAN_PENDING:
                cut_points.append((raw_at(t, c), c))
        split_raw: set[int] = set()
        for ev in sorted(accepted, key=lambda e: -int(e.params["cut_frame"])):
            t = ev.tracks[0]
            split_raw |= set(work.loc[work["track"] == t, "raw_id"].astype(int).tolist())
            work = apply_edit(work, ev, new_id=next_track_id(work))
            ev.applied = True
        cuts: dict[int, list[int]] = {}
        for raw, c in cut_points:
            rows = work.loc[(work["raw_id"] == raw) & (work["frame"] == c), "track"]
            if len(rows):
                cuts.setdefault(int(rows.iloc[0]), []).append(c)
        return work, split_raw, cuts

    def _screen(self, work, split_raw, cuts):
        cfg = self.cfg
        if not cfg.screen.enabled or work.empty:
            return work
        sctx = ScreenContext(boxes=self.ctx_boxes, fmt=self.ctx_fmt, hints=self.hints,
                             split_raw_ids=split_raw)
        evs = propose_screen(work, cfg, self.fps, sctx, cuts)
        self._route(evs, Band.of(cfg.screen), "screen")
        for ev in evs:
            if ev.decision in ACCEPTED:
                work = apply_edit(work, ev)
                ev.applied = True
        return work

    def _link(self, work, occluded):
        cfg = self.cfg
        if not cfg.link.enabled or work.empty:
            return work, {}, set(), set()
        band = Band.of(cfg.link)
        res = run_link_stage(work, cfg, self.fps, appearance=self.appearance,
                             context=self.ctx_boxes, frame_size=self.frame_size,
                             occluded=occluded, route=lambda evs: self._route(evs, band, "link"))
        work, rep_of = merge_chains(work, res.accepted)
        protected: dict[int, list[tuple[int, int]]] = {}
        for ev in res.events:
            if ev.applied and ev.params.get("gate") == "occluded":
                rep = rep_of.get(ev.tracks[0], ev.tracks[0])
                protected.setdefault(rep, []).append(tuple(ev.params["gap"]))
        pending = {rep_of.get(t, t) for t in res.pending_endpoints}
        return work, protected, set(rep_of.values()), pending

    def _orphans(self, work, linked, pending):
        cfg = self.cfg
        if not cfg.orphan.enabled or work.empty:
            return work
        evs, self.orphan_deferred = propose_orphans(work, cfg, self.fps, linked_tracks=linked,
                                                    pending_endpoints=pending)
        self._route(evs, Band.of(cfg.orphan), "orphan")
        for ev in evs:
            if ev.decision in ACCEPTED:
                work = apply_edit(work, ev)
                ev.applied = True
        return work

    def _fill(self, work, protected):
        if not self.cfg.fill.enabled or work.empty:
            return work
        work, evs = fill_stage(work, self.cfg, self.fps, protected)
        for ev in evs:
            self.seq["fill"] += 1
            ev.id = f"fill-r0-{self.seq['fill']:06d}"
        self.events.extend(evs)
        return work


class TrackRefiner:
    """Refine track files after tracking; used like ``Detector`` and ``Tracker`` (spec 2.4)."""

    def __init__(
        self,
        config: RefineConfig | None = None,
        config_yaml: str | None = None,
        device: str | None = None,
        *,
        appearance_factory: Callable[..., Appearance | None] | None = None,
    ) -> None:
        """Configure once with ``config`` or ``config_yaml``; ``device`` sets ``encoder.device``."""
        if config is not None and config_yaml is not None:
            raise ValueError("pass config or config_yaml, not both")
        if config_yaml is not None:
            config = RefineConfig.from_yaml(config_yaml)
        self.config = config if config is not None else RefineConfig.defaults()
        if device is not None:
            self.config.encoder.device = device
        self.config.validate()
        self.appearance_factory = appearance_factory
        self.last_result: RefineResult | None = None

    def refine(
        self,
        track_file,
        out_file,
        video_file=None,
        context_file=None,
        reclass_file=None,
        fps: float | None = None,
        fmt: str = "dnt",
        video_index: int | None = None,
        video_tot: int | None = None,
        message: str | None = "",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Refine one track file into ``out_file`` and write its ledger (spec 2.4, 2.5).

        Returns the refined track table, like ``Tracker.track()``; ``self.last_result`` holds
        the ledger path, summary, and events.
        """
        cfg = self.config
        out = Path(out_file)
        paths = output_paths(out)
        vinfo = io.video_info(video_file) if video_file is not None else None
        fps_val, fps_src = resolve_fps(fps, cfg.fps, vinfo["fps"] if vinfo else None)
        if cfg.frame_size:
            frame_size = tuple(int(v) for v in cfg.frame_size)
        elif vinfo:
            frame_size = (vinfo["width"], vinfo["height"])
        else:
            frame_size = None
        tin = io.read_tracks(track_file, fmt=fmt, class_id=cfg.class_ids[0])
        work = tin.work
        if vinfo and len(work) and int(work["frame"].max()) > vinfo["frame_count"]:
            raise ValueError(
                f"track frame {int(work['frame'].max())} exceeds the video's frame count "
                f"{vinfo['frame_count']}; the track file does not belong to this video"
            )
        ctx_boxes, ctx_fmt = (io.read_context(context_file, cfg.context.format)
                              if context_file is not None else (None, None))
        hint_map = (read_reclass_hints(reclass_file, set(work["raw_id"].unique().tolist()))
                    if reclass_file is not None else {})
        inputs = {
            "tracks": _file_record(track_file, format=fmt),
            "video": None if video_file is None else {
                "path": str(video_file), "abs_path": str(Path(video_file).resolve()),
                "fingerprint": io.video_fingerprint(video_file, vinfo["frame_count"]),
                "frame_count": vinfo["frame_count"],
            },
            "context": None if context_file is None else _file_record(context_file,
                                                                        format=ctx_fmt),
            "hints": None if reclass_file is None else {"reclass": _file_record(reclass_file)},
            "features": None,
        }
        before = table_summary(work, fps_val)
        appearance = self._appearance(work, video_file, ctx_boxes, fps_val)
        stages = _Stages(cfg, fps_val, frame_size, appearance, ctx_boxes, ctx_fmt, hint_map)
        desc = ("Refining" if video_index is None or video_tot is None
                else f"Refining {video_index} of {video_tot}")
        if message:
            desc += f" {message}"
        with tqdm(total=5, desc=desc, unit=" stage", disable=not verbose) as pbar:
            work, events = stages.run(
                work, tick=lambda name: (pbar.set_postfix_str(name), pbar.update(1))
            )
        work, id_map = renumber(work)
        io.write_tracks(work, out)
        summary = {
            "before": before, "after": table_summary(work, fps_val),
            "events": _event_counts(events),
            "vlm": {"calls": 0, "cache_hits": 0, "failures": 0},
            "orphan_deferred": sorted(id_map[t] for t in stages.orphan_deferred if t in id_map),
            "filled_input_rows_removed": tin.n_filled_removed,
            "duplicate_input_rows_removed": tin.n_duplicates_removed,
        }
        header = {
            "format": LEDGER_FORMAT, "dnt_version": __version__, "round": 0, "parent": None,
            "config": cfg.to_dict(), "inputs": inputs, "fps": fps_val, "fps_source": fps_src,
            "frame_size": list(frame_size) if frame_size else None,
            "id_map": {str(k): v for k, v in id_map.items()},
            "n_filled_input_rows_removed": tin.n_filled_removed,
            "smoothing": cfg.fill.smooth_existing, "summary": summary,
        }
        Ledger(header, events).write(paths["ledger"])
        log.info("refined %s: %s", out, summary["events"])
        self.last_result = RefineResult(tracks=work, ledger_path=paths["ledger"],
                                        review_path=None, summary=summary, events=events)
        return work

    def refine_batch(
        self,
        track_files: Sequence,
        video_files: Sequence | None = None,
        output_path=None,
        context_files: Sequence | None = None,
        reclass_files: Sequence | None = None,
        fps: float | None = None,
        is_overwrite: bool = False,
        is_report: bool = True,
        message: str | None = "",
        verbose: bool = True,
    ) -> list[str]:
        """Refine several track files, like ``Tracker.track_batch`` (spec 2.4).

        Files are paired by position. Each output is ``<output_path>/<base>_refined.txt``,
        where ``<base>`` is the track file's stem without a trailing ``_track``. Existing
        outputs are skipped unless ``is_overwrite``; with ``is_report`` they are still listed.
        """
        if output_path is None:
            raise ValueError("refine_batch needs output_path: every run writes a ledger "
                             "next to its output")
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        def pick(seq, i):
            return seq[i] if seq is not None and i < len(seq) else None

        results: list[str] = []
        total = len(track_files)
        for i, track_file in enumerate(track_files):
            base = re.sub(r"_track$", "", Path(track_file).stem)
            out = out_dir / f"{base}_refined.txt"
            if out.exists() and not is_overwrite:
                if is_report:
                    results.append(str(out))
                continue
            self.refine(track_file, out, video_file=pick(video_files, i),
                        context_file=pick(context_files, i),
                        reclass_file=pick(reclass_files, i), fps=fps, video_index=i + 1,
                        video_tot=total, message=message, verbose=verbose)
            results.append(str(out))
        return results

    def _appearance(self, work, video, context, fps) -> Appearance | None:
        if self.appearance_factory is not None:
            return self.appearance_factory(work=work, video=video, context=context, fps=fps,
                                           config=self.config)
        if video is not None and self.config.encoder.kind != "none":
            log.warning("appearance encoders arrive in dnt.refine Plan 2; running motion-only")
        return None
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/refine/test_refiner.py -v && .venv/bin/ruff check src/dnt/refine`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/refiner.py tests/refine/test_refiner.py
git commit -m "feat(refine): add TrackRefiner.refine with stage 4, ledger header, and summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17: Public API, `dnt-refine run`, packaging, and docs

**Files:**
- Modify: `src/dnt/refine/__init__.py`
- Create: `src/dnt/refine/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Create: `docs/api/refine/index.md`, `refiner.md`, `config.md`, `events.md`, `interpolate.md`, `link.md`
- Modify: `mkdocs.yml` (nav), `docs/api/track/post_process.md`, `docs/changelog.md`
- Test: `tests/refine/test_cli.py`

**Interfaces:**
- Produces:
  - `dnt.refine.__all__ = ["RefineConfig", "RefineResult", "TrackRefiner", "interpolate_tracks_rts", "link_tracklets"]`
  - `dnt.refine.cli.main(argv=None) -> int`, which returns 0 on success and 2 on `ValueError` / `FileNotFoundError`
  - The console script `dnt-refine`

- [ ] **Step 1: Write the failing tests**

```python
# tests/refine/test_cli.py
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from dnt.refine.config import RefineConfig

from ._fixtures import box_rows, table

ROOT = Path(__file__).resolve().parents[2]


def _src(tmp_path):
    p = tmp_path / "t.txt"
    table(box_rows(1, range(40), 100.0, 100.0, vx=2.0),
          box_rows(2, range(46, 90), 192.0, 100.0, vx=2.0)).to_csv(p, index=False, header=False)
    return p


def _run(*args):
    return subprocess.run([sys.executable, "-m", "dnt.refine.cli", *map(str, args)],
                          capture_output=True, text=True)


def test_cli_run_writes_outputs(tmp_path):
    cfg = tmp_path / "c.yaml"
    RefineConfig.defaults().to_yaml(cfg)
    r = _run("run", _src(tmp_path), "--fps", 10, "--config", cfg, "--out", tmp_path / "o.csv")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.csv").exists() and (tmp_path / "o.ledger.jsonl").exists()
    assert json.loads(r.stdout)["ledger"].endswith("o.ledger.jsonl")


def test_cli_reports_errors_with_exit_code_2(tmp_path):
    cfg = tmp_path / "c.yaml"
    RefineConfig.defaults().to_yaml(cfg)
    r = _run("run", _src(tmp_path), "--config", cfg, "--out", tmp_path / "o.csv")
    assert r.returncode == 2 and "fps=" in r.stderr


def test_entry_point_is_declared():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["dnt-refine"] == "dnt.refine.cli:main"


def test_public_api():
    import dnt.refine as refine

    assert set(refine.__all__) == {"RefineConfig", "RefineResult", "TrackRefiner",
                                 "interpolate_tracks_rts", "link_tracklets"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/refine/test_cli.py -v`
Expected: FAIL (`No module named dnt.refine.cli`, `KeyError: 'scripts'`, and `__all__` is missing).

- [ ] **Step 3: Implement the public API and CLI**

```python
# src/dnt/refine/__init__.py
"""Track refinement: switch splitting, false-track screening, linking, and gap filling.

Design: docs/superpowers/specs/2026-09-27-track-refinement-design.md.
"""

from .config import RefineConfig
from .interpolate import interpolate_tracks_rts
from .link import link_tracklets
from .refiner import RefineResult, TrackRefiner

__all__ = ["RefineConfig", "RefineResult", "TrackRefiner", "interpolate_tracks_rts",
           "link_tracklets"]
```

```python
# src/dnt/refine/cli.py
"""Command-line entry point ``dnt-refine`` (spec 2.4)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .refiner import TrackRefiner


def build_parser() -> argparse.ArgumentParser:
    """Return the ``dnt-refine`` argument parser."""
    p = argparse.ArgumentParser(
        prog="dnt-refine",
        description="Refine a track file: split ID switches, screen false tracks, link "
                    "fragments, drop orphans, and fill gaps.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="refine a track file")
    run.add_argument("tracks", help="headerless dnt (10-column) or MOT track file")
    run.add_argument("--video", help="source video (frame rate, frame size, and later evidence)")
    run.add_argument("--fps", type=float, help="frame rate; required when there is no video")
    run.add_argument("--context", help="context file: dnt tracks (10 columns) or detections (8)")
    run.add_argument("--reclass-hints", dest="reclass_hints", help="ReClass output CSV")
    run.add_argument("--config", required=True, help="RefineConfig YAML")
    run.add_argument("--out", required=True, help="output track file")
    run.add_argument("--format", choices=("dnt", "mot"), default="dnt", help="input format")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run ``dnt-refine``; return the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        refiner = TrackRefiner(config_yaml=args.config)
        refiner.refine(
            args.tracks, args.out, video_file=args.video, context_file=args.context,
            reclass_file=args.reclass_hints, fps=args.fps, fmt=args.format, verbose=False,
        )
        res = refiner.last_result
    except (ValueError, FileNotFoundError) as exc:
        print(f"dnt-refine: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": args.out, "ledger": str(res.ledger_path),
                      "summary": res.summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

In `pyproject.toml`, after `[project.optional-dependencies]`, add:

```toml
[project.scripts]
dnt-refine = "dnt.refine.cli:main"
```

Re-install so the entry point exists: `.venv/bin/pip install -e . --no-deps -q`.

- [ ] **Step 4: Write the docs**

`docs/api/refine/index.md`:

````markdown
# Track Refinement (`dnt.refine`)

`dnt.refine` refines a track file after tracking: it splits ID switches, screens false tracks,
links fragments (including across static waits and witnessed occlusions), drops orphans, and
fills gaps. Every edit is recorded in a JSONL ledger next to the output.

Used like `Detector` and `Tracker`:

```python
from dnt.refine import RefineConfig, TrackRefiner

refiner = TrackRefiner(config=RefineConfig.defaults("vehicle"))  # or config_yaml="veh.yaml"
tracks = refiner.refine("cam1_track.txt", "cam1_refined.txt", video_file="cam1.mp4")
refiner.refine_batch(track_files, video_files=video_files, output_path="refined/")
```

Command line: `dnt-refine run TRACKS --fps 10 --config refine.yaml --out clean.csv`.

::: dnt.refine
````

`docs/api/refine/refiner.md`, `config.md`, `events.md`, `interpolate.md`, `link.md` each contain one heading and one directive. For example, `refiner.md` is:

```markdown
# Refiner

::: dnt.refine.refiner
```

The others use `::: dnt.refine.config` (heading "Configuration"), `::: dnt.refine.events` ("Events and Ledger"), `::: dnt.refine.interpolate` ("Interpolation"), and `::: dnt.refine.link` ("Linking").

In `mkdocs.yml`, insert this after the `- Tracking:` block (same indentation as `- Engine:`):

```yaml
      - Track Refinement:
          - api/refine/index.md
          - Refiner: api/refine/refiner.md
          - Configuration: api/refine/config.md
          - Events and Ledger: api/refine/events.md
          - Interpolation: api/refine/interpolate.md
          - Linking: api/refine/link.md
```

Replace `docs/api/track/post_process.md` with:

```markdown
# Post Processing

`interpolate_tracks_rts` and `link_tracklets` moved to [`dnt.refine`](../refine/index.md).
`dnt.track.post_process` still re-exports them.

::: dnt.refine.interpolate.interpolate_tracks_rts

::: dnt.refine.link.link_tracklets
```

At the top of `docs/changelog.md`, below `# Changelog`, add:

```markdown
## Unreleased

### New
- `dnt.refine` track refinement: `TrackRefiner` (`refine`, `refine_batch`, used like `Tracker`)
  and `dnt-refine run` split ID switches, screen
  false tracks, link fragments (including static waits and occlusion-witnessed gaps), drop
  orphans, and fill gaps. Every edit is recorded in a JSONL ledger next to the output. This
  release scores with motion only; appearance encoders, VLM verification, review pages, and
  replay follow.

### Changed
- `interpolate_tracks_rts` and `link_tracklets` moved to `dnt.refine`. `dnt.track.post_process`
  re-exports them, and `Filter.interpolate_tracks_rts` now calls `dnt.refine.interpolate`.
- `interpolate_tracks_rts` no longer uses rows flagged as filled (`interp`/`r3` equal to 1) as
  measurements, and accepts `protected_gaps`. Output for raw tracker files is unchanged.
```

- [ ] **Step 5: Run the tests, the whole suite, lint, and a docs build that does not touch `site/`**

Run:
```bash
.venv/bin/python -m pytest tests/refine/test_cli.py -v
.venv/bin/python -m pytest
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src/dnt/refine
.venv/bin/mkdocs --version && .venv/bin/mkdocs build --strict --site-dir "$(mktemp -d)"
```
Expected:
- The CLI tests PASS.
- The full default suite PASSES with no failures.
- Ruff is clean. If `ruff format --check` reports files, run `.venv/bin/ruff format src/dnt/refine` and re-run the tests.
- If mkdocs is installed, the strict build succeeds. (`site/` is committed build output; never regenerate it here.) If `mkdocs --version` fails, the docs extra is not installed. Note that in the handoff; it is not a failure.

- [ ] **Step 6: Commit**

```bash
git add src/dnt/refine/__init__.py src/dnt/refine/cli.py pyproject.toml docs/api/refine docs/api/track/post_process.md mkdocs.yml docs/changelog.md tests/refine/test_cli.py
git commit -m "feat(refine): add public API, dnt-refine run CLI, API docs, and changelog

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (Plan 1)

| Spec requirement | Task |
|---|---|
| §1 criterion 1 (minimal install, no video, `fps=`) | 16 `test_default_config_no_video_all_three_stages_score` |
| §1 criterion 2 (fixtures per stage) | 11, 12, 13, 14, 15 |
| §1 criterion 4 (independence) | 1 |
| §1 criterion 5 (legacy tests unchanged) | 3, 4, 5 |
| §1 criterion 7 (legacy parity) | 14 `test_legacy_mode_matches_link_tracklets_grouping` |
| §2.1 modules, §2.2 dependency rule | 1–17 |
| §2.3 move, shims, shared helpers, Filter wrapper | 3, 5 |
| §2.4 `refine`, `dnt-refine run` | 16, 17 |
| §2.5 input and output contract | 6, 16 |
| §3 order and routing between stages | 16 |
| §4.1 event model, §4.2 file format, keys, header, lineage | 8, 9, 16 |
| §4.3 bands and caps (static, mixed, occluded, ambiguous, rider subtype) | 10, 12, 14, 15 |
| §5.1, §5.2, §5.4, §5.6 | 2, 16 |
| §6.1 stage 1 | 11 |
| §6.2 stage 2 and orphans | 12, 13 |
| §6.3 stage 3, passes, chains, legacy | 14, 15 |
| §6.4 stage 4, `protected_gaps`, FILL / SMOOTH | 4, 16 |
| §9 config and validation | 7 |
| §10 rows not about replay or the VLM | 6, 7, 12, 16 |
| §11.1 | 11–15 |

Deferred by design, and listed in the Plan series table: §5.3 cache and encoders, and §5.5 (P2); §7 and §8.1 (P3); §4.2 replay, §8.2, and the §11.2 replay tests (P4).
