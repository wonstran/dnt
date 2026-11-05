# Vehicle Reversal Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `reversal` stage to `dnt.refine` that splits a vehicle track wherever its direction reverses sharply. Then make sure dedup and link never join the two sides again, whether directly, through a chain, or after a merge replaces a side's rows.

**Architecture:**
- **New module.** `src/dnt/refine/reversal.py` holds three things, all free of side effects:
  - the per-track rule: `row_values` → `hit_rows` → `runs` → `cut_in_run` → `keep_cuts` → `find_reversals`;
  - the stage's proposals, `propose_reversal_splits`;
  - the side bookkeeping: `ReversalSides` (the cuts, plus the side tags each track carries), `side_conflict`, and the link-chain resolver `resolve_reversal_chains`.
- **Pipeline.** `_Stages` in `refiner.py` runs the stage first and applies its splits. It then passes `ReversalSides` to dedup (`apply_merges`) and link (`score_candidates`, `legacy_link_events`, `run_link_stage`), and calls `absorb` after each of them.
- **Default.** Everything is off when `reversal.enabled` is false, and also when no cut was made.

**Tech Stack:** Python 3.11, numpy, pandas, pytest. Use the existing `dnt.refine` primitives (`mean_velocity`, `lineage_of_rows`, `Event.propose`, `decide`, `apply_edit`).

**Spec:** [docs/superpowers/specs/2026-11-04-vehicle-reversal-split-design.md](../specs/2026-11-04-vehicle-reversal-split-design.md) (rev. 5). Read §3.2 (the rule), §3.5 (hand-offs and side tags), §5 (config) and §6 (tests) before starting.

## Global Constraints

- Defaults (spec §5): `enabled` is `False`, and `True` in `RefineConfig.defaults("vehicle")`. `window = 2.0` s, `min_speed = 0.5` box heights/s, `min_angle = 135.0`°, `min_rows = 5`, `min_side_rows = 3`.
- Validation (spec §5): `window` > 0; `min_speed` finite and ≥ 0; `90 < min_angle ≤ 180`; `min_rows` an int ≥ 1; `min_side_rows` an int ≥ 2. Bools are not numbers. Every message names the field (`reversal.<field>`).
- Speeds are in **box heights per second**, unlike `switch.heading_min_speed` (pixels per second).
- Cut choice (spec §3.2 step 6): within a run, the largest `|v_a − v_b|` (pixels per frame). Changes within `1e-9` (relative) of the largest are ties, and the **earliest** of them wins.
- Part lengths (spec §3.2 step 7): a cut at row `i` is kept if `i − prev ≥ min_rows` and `n − i ≥ min_rows`, where `prev` is the previous **kept** cut (0 for the first).
- Events (spec §3.3):
  - `stage="reversal"`, `kind=SPLIT`, `tracks=[tid]`, `lineage=[lineage_of_rows(g)]`, `frames=(c, c)`, `params={"cut_frame": c}`, `algo_score=1.0`, decision `AUTO_ACCEPT`, IDs `reversal-r0-000001`, ….
  - `signals`: `angle`, `change`, `run`, `speed_before`, `speed_after`, `rows_before`, `rows_after`. After the split is applied they also hold `new_track` and `sides`.
- Side tags (spec §3.5):
  - A tag is `(r, k)` with `k = bisect_right(cuts[r], f0)` of a lineage span's first frame.
  - `tags(t) = own(t) ∪ carried[t]`.
  - `absorb(mapping, tags_before)` runs `carried[mapping.get(t, t)] |= tags_before[t]` for **every** `t` in `tags_before`.
- Skip markers:
  - A LINK dropped by the chain check gets `applied=False`, `skipped_reason="reversal"` and `conflicts_with={"raw": r, "sides": [k1, k2], "tracks": [u, v]}`.
  - A MERGE blocked in dedup gets `skipped_reason="conflict"` and `conflicts_with={"tracks": [u, v], "why": "reversal", "proposal_key": None}`.
- When `reversal_sides` is `None` or has no cuts, link and dedup behave exactly as before.
- Tags are never written into lineage, so proposal keys and evidence keep their current meaning.
- Lint: `ruff check src tests tools` passes, with line length 100 and numpy-style docstrings in `src`. Never add entries to the per-file baseline in `pyproject.toml`.
- Tests run with `.venv/bin/python -m pytest …`.
- Commit messages end with:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

## Review Focus

1. **A 30 fps clip.** `min_rows` is counted in rows, while `window` is in seconds. An out-and-back at 30 fps must still be cut at the turn row (90), not one row late. Test: Task 3, `test_a_30_fps_out_and_back_is_cut_at_the_turn`.
2. **A table where nothing can be split** (every track shorter than `2 * min_rows`). The stage must emit no events and leave every ID alone. Test: Task 6, `test_a_table_of_short_tracks_is_left_alone`.
3. **A track with two reversals.** Its middle part must join neither end in link. Test: Task 7, `test_no_part_of_a_twice_reversed_track_links_to_another`.
4. **One `TrackRefiner` reused across clips** (`refine_batch`). Cuts or tags from clip 1 must not change clip 2. Test: Task 6, `test_a_reused_refiner_does_not_carry_cuts_between_clips`.
5. **A hand-written YAML `reversal:` block with wrong types** (`min_rows: 5.0`, `min_rows: true`, `window: "2"`, `enabled: "yes"`). It must be rejected with the field name, not crash in the stage. Test: Task 2, `test_bad_values_are_rejected_with_the_field_name`.

---

### Task 1: Record the disabled-stage baseline with the pre-change code

This **must** run before any code change. Spec §6.2 ("Disabled") compares the new code with this baseline when the stage is off.

**Files:**
- Modify: `tests/refine/_fixtures.py` (append `reversal_scene`)
- Create: `tests/refine/make_reversal_baseline.py`
- Create (generated): `tests/refine/data/reversal_baseline/raw.csv`, `tracks.csv`, `ledger.jsonl`

**Interfaces:**
- Produces: `_fixtures.reversal_scene() -> pd.DataFrame` (a 10-column raw table), plus the three data files. Task 10 reads all of them.

- [ ] **Step 1: Check the tree is unchanged**

Run: `git status --short src/dnt/refine`
Expected: no output. If `src/dnt/refine` has changes, stop: the baseline must come from the committed pre-change code.

- [ ] **Step 2: Append the scene to `tests/refine/_fixtures.py`**

```python
def reversal_scene() -> pd.DataFrame:
    """A vehicle scene for the reversal stage: one ID goes out and comes back, plus others."""
    t = np.arange(60)
    cx = 100.0 + np.where(t < 30, 2.0 * t, 120.0 - 2.0 * t)
    rows = [[int(f), 1, float(x) - 5.0, 95.0, 10.0, 10.0, 0.9, 2, -1, -1] for f, x in zip(t, cx)]
    rows += box_rows(2, range(0, 120), 400.0, 300.0, w=40.0, h=30.0, cls=2, score=0.35)  # parked
    rows += box_rows(3, range(0, 40), 600.0, 200.0, vx=3.0, w=40.0, h=30.0, cls=2)  # fragment
    rows += box_rows(4, range(45, 90), 735.0, 200.0, vx=3.0, w=40.0, h=30.0, cls=2)  # its rest
    rows += box_rows(5, range(10, 100), 200.0, 500.0, vx=-4.0, w=50.0, h=40.0, cls=7)  # truck
    return table(rows)
```

- [ ] **Step 3: Create `tests/refine/make_reversal_baseline.py`**

```python
"""Write the reversal-stage baseline: the refine output of the code BEFORE the stage existed.

Run ONCE, before any reversal code is written (plan Task 1):

    .venv/bin/python tests/refine/make_reversal_baseline.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _fixtures import DATA, reversal_scene  # noqa: E402

from dnt.refine import RefineConfig, TrackRefiner  # noqa: E402

OUT = DATA / "reversal_baseline"


def main() -> None:
    """Refine the scene with the vehicle defaults, motion only, at 10 fps."""
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw.csv"
    reversal_scene().to_csv(raw, index=False, header=False)
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    refiner = TrackRefiner(cfg)
    refiner.refine(raw, OUT / "tracks.csv", fps=10, verbose=False)
    shutil.move(str(refiner.last_result.ledger_path), OUT / "ledger.jsonl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python tests/refine/make_reversal_baseline.py && ls tests/refine/data/reversal_baseline`
Expected: `ledger.jsonl  raw.csv  tracks.csv`. If refine wrote other side files (a review page, features), delete them; only these three are kept.

- [ ] **Step 5: Commit**

```bash
git add tests/refine/_fixtures.py tests/refine/make_reversal_baseline.py tests/refine/data/reversal_baseline
git commit -m "test(refine): record the pre-reversal baseline for the disabled-stage check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `ReversalConfig`

**Files:**
- Modify: `src/dnt/refine/config.py` (new dataclass after `SwitchConfig`, a field on `RefineConfig`, the vehicle defaults, and `validate`)
- Test: `tests/refine/test_reversal_config.py`

**Interfaces:**
- Produces: `dnt.refine.config.ReversalConfig` with fields `enabled: bool`, `window: float`, `min_speed: float`, `min_angle: float`, `min_rows: int` and `min_side_rows: int`, and `RefineConfig.reversal: ReversalConfig`.

- [ ] **Step 1: Write the failing tests**

```python
import re

import pytest

from dnt.refine.config import RefineConfig


def test_vehicle_defaults_turn_the_stage_on_and_person_defaults_leave_it_off():
    assert RefineConfig.defaults("vehicle").reversal.enabled is True
    assert RefineConfig.defaults("person").reversal.enabled is False


def test_defaults_are_the_script_values():
    rc = RefineConfig.defaults("vehicle").reversal
    assert (rc.window, rc.min_speed, rc.min_angle, rc.min_rows, rc.min_side_rows) == (
        2.0, 0.5, 135.0, 5, 3,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("window", 0), ("window", -1.0), ("window", "2"), ("window", True),
        ("min_speed", -0.1), ("min_speed", float("nan")), ("min_speed", True),
        ("min_angle", 90), ("min_angle", 181), ("min_angle", "135"),
        ("min_rows", 0), ("min_rows", 5.0), ("min_rows", True),
        ("min_side_rows", 1), ("min_side_rows", 3.0),
        ("enabled", "yes"),
    ],
)
def test_bad_values_are_rejected_with_the_field_name(key, value):
    with pytest.raises(ValueError, match=re.escape(f"reversal.{key}")):
        RefineConfig.from_dict({"target": "vehicle", "reversal": {key: value}})


def test_a_partial_block_overlays_the_target_defaults():
    cfg = RefineConfig.from_dict({"target": "vehicle", "reversal": {"min_angle": 150}})
    assert cfg.reversal.enabled is True
    assert cfg.reversal.min_angle == 150 and cfg.reversal.window == 2.0


def test_unknown_reversal_keys_raise():
    with pytest.raises(ValueError, match=re.escape("reversal.min_angel")):
        RefineConfig.from_dict({"reversal": {"min_angel": 150}})


def test_the_block_round_trips_through_yaml(tmp_path):
    cfg = RefineConfig.defaults("vehicle")
    cfg.reversal.min_rows = 7
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert RefineConfig.from_yaml(path).reversal.min_rows == 7
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_config.py -v`
Expected: FAIL with `AttributeError: 'RefineConfig' object has no attribute 'reversal'`.

- [ ] **Step 3: Implement the config**

In `src/dnt/refine/config.py`, add this after `class SwitchConfig`:

```python
@dataclass(kw_only=True)
class ReversalConfig:
    """Stage ``reversal`` settings (reversal spec 5).

    Speeds are in box heights per second, unlike ``switch.heading_min_speed`` (pixels per
    second).
    """

    enabled: bool = False
    window: float = 2.0
    min_speed: float = 0.5
    min_angle: float = 135.0
    min_rows: int = 5
    min_side_rows: int = 3
```

In `RefineConfig`, add the field just before `switch`:

```python
    reversal: ReversalConfig = field(default_factory=ReversalConfig)
```

In `RefineConfig.defaults`, inside `if target == "vehicle":`, add:

```python
            cfg.reversal.enabled = True  # two vehicles that share an ID: a reversal splits them
```

Add this module-level function next to `_dedup_problems`:

```python
def _reversal_problems(rc: ReversalConfig) -> list[str]:
    """Return the rules of the reversal block that its values break (reversal spec 5)."""

    def num(x) -> bool:
        return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)

    def whole(x) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    p: list[str] = []
    if not isinstance(rc.enabled, bool):
        p.append(f"reversal.enabled must be true or false, not {rc.enabled!r}")
    if not (num(rc.window) and rc.window > 0):
        p.append(f"reversal.window must be a positive number of seconds, not {rc.window!r}")
    if not (num(rc.min_speed) and rc.min_speed >= 0):
        p.append(
            "reversal.min_speed must be a number of box heights per second >= 0, "
            f"not {rc.min_speed!r}"
        )
    if not (num(rc.min_angle) and 90 < rc.min_angle <= 180):
        p.append(
            f"reversal.min_angle must be a number of degrees above 90 and at most 180, "
            f"not {rc.min_angle!r}"
        )
    if not (whole(rc.min_rows) and rc.min_rows >= 1):
        p.append(f"reversal.min_rows must be a whole number of rows >= 1, not {rc.min_rows!r}")
    if not (whole(rc.min_side_rows) and rc.min_side_rows >= 2):
        p.append(
            f"reversal.min_side_rows must be a whole number of rows >= 2, "
            f"not {rc.min_side_rows!r}"
        )
    return p
```

In `RefineConfig.validate`, right after `p.extend(_dedup_problems(self.dedup))`, add:

```python
        p.extend(_reversal_problems(self.reversal))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_config.py tests/refine/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/config.py tests/refine/test_reversal_config.py
git commit -m "feat(refine): add the reversal config block, on for vehicles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The reversal rule

**Files:**
- Create: `src/dnt/refine/reversal.py`
- Create: `tests/refine/_reversal.py` (shared synthetic helpers)
- Test: `tests/refine/test_reversal_rule.py`

**Interfaces:**
- Consumes: `ReversalConfig` (Task 2), `config.to_frames`, `primitives.mean_velocity`.
- Produces, in `dnt.refine.reversal`:
  - `STAGE = "reversal"`
  - `Reversal` (frozen dataclass: `row: int`, `frame: int`, `angle: float`, `change: float`, `run: tuple[int, int]` (frames), `speed_before: float`, `speed_after: float`, `rows_before: int`, `rows_after: int`)
  - `RowValues` (frozen dataclass: `cos`, `change`, `speed_before`, `speed_after`, each an `np.ndarray`)
  - `row_values(frames, boxes, fps, rc) -> RowValues`
  - `hit_rows(frames, boxes, fps, rc) -> np.ndarray`
  - `runs(hit) -> list[np.ndarray]`
  - `cut_in_run(run, change) -> int`
  - `keep_cuts(proposed, n, min_rows) -> list[int]`
  - `find_reversals(frames, boxes, fps, rc) -> list[Reversal]`

  In every function, `frames` is an int array sorted ascending and `boxes` is an `(n, 4)` array of `x, y, w, h`.
- Produces, in `tests/refine/_reversal.py`: `FPS`, `out_and_back`, `path`, `boxes_at`, `cx_out_and_back`, `var_height`, `rows`, `work_of`, `link_event`. Tasks 4 to 10 use them.

- [ ] **Step 1: Create the test helpers `tests/refine/_reversal.py`**

```python
"""Synthetic vehicle tracks and events for the reversal-stage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnt.refine import io
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.verify import decide

from ._fixtures import table

FPS = 10.0


def out_and_back(n: int = 60, turn: int = 30, speed: float = 2.0):
    """Frames and centre x of a box that moves right, then back left from frame ``turn``."""
    t = np.arange(n)
    return t, 100.0 + np.where(t < turn, speed * t, 2.0 * speed * turn - speed * t)


def path(vels) -> np.ndarray:
    """Positions starting at 100 that move by ``vels[k]`` into frame ``k`` (k >= 1)."""
    v = np.asarray(vels, dtype=float)
    return 100.0 + np.concatenate([[0.0], np.cumsum(v[1:])])


def boxes_at(cx, cy=100.0, h=10.0, w=10.0) -> np.ndarray:
    """``(n, 4)`` boxes centred on ``(cx, cy)``; ``cy`` and ``h`` broadcast."""
    cx = np.asarray(cx, dtype=float)
    cy = np.broadcast_to(np.asarray(cy, dtype=float), cx.shape)
    h = np.broadcast_to(np.asarray(h, dtype=float), cx.shape)
    return np.column_stack([cx - w / 2.0, cy - h / 2.0, np.full(cx.shape, w), h])


def cx_out_and_back(f, turn: int = 30, speed: float = 2.0) -> float:
    """Centre x at frame ``f`` of ``out_and_back``."""
    return 100.0 + (speed * f if f < turn else 2.0 * speed * turn - speed * f)


def var_height(f) -> float:
    """The review's height: 50 px on frames 15..30, 10 px elsewhere."""
    return 50.0 if 15 <= f <= 30 else 10.0


def rows(track, frames, cx_of=cx_out_and_back, *, h_of=lambda f: 10.0, w=10.0, cy=100.0,
         score=0.9, cls=2):
    """Raw rows of a box centred on ``(cx_of(f), cy)`` with height ``h_of(f)``."""
    return [
        [int(f), track, float(cx_of(f)) - w / 2.0, cy - h_of(f) / 2.0, w, float(h_of(f)),
         score, cls, -1, -1]
        for f in frames
    ]


def work_of(*row_lists, raw=None) -> pd.DataFrame:
    """Work table of the rows; ``raw`` maps a track ID to the raw ID its rows carry."""
    work = io.to_work(table(*row_lists)).work
    for t, r in (raw or {}).items():
        work.loc[work["track"] == t, "raw_id"] = r
    return work


def link_event(i, j, score, lineage_i=None, lineage_j=None) -> Event:
    """An accepted, applied LINK event ``i -> j``."""
    ev = Event.propose(
        stage="link",
        kind=EventKind.LINK,
        tracks=[i, j],
        lineage=[lineage_i or [[i, 0, 0]], lineage_j or [[j, 1, 1]]],
        frames=(0, 1),
        params={"gate": "normal", "gap": [0, 1]},
        algo_score=score,
    )
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    ev.applied = True
    return ev
```

- [ ] **Step 2: Write the failing rule tests `tests/refine/test_reversal_rule.py`**

Every expected number below was measured with a prototype of this exact rule on the production `mean_velocity`.

```python
import dataclasses
import warnings

import numpy as np
import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.reversal import (
    cut_in_run,
    find_reversals,
    hit_rows,
    keep_cuts,
    row_values,
    runs,
)

from ._reversal import FPS, boxes_at, out_and_back, path

RC = RefineConfig.defaults("vehicle").reversal


def cuts(frames, boxes, rc=RC, fps=FPS):
    return [r.row for r in find_reversals(frames, boxes, fps, rc)]


def test_a_clean_out_and_back_is_cut_at_the_turn_not_at_the_first_tied_row():
    t, cx = out_and_back()
    b = boxes_at(cx)
    assert hit_rows(t, b, FPS, RC).tolist() == list(range(23, 39))
    assert np.allclose(row_values(t, b, FPS, RC).cos[23:39], -1.0)  # every row: full reversal
    (rev,) = find_reversals(t, b, FPS, RC)
    assert (rev.row, rev.frame, rev.run) == (30, 30, (23, 38))
    assert rev.angle == pytest.approx(180.0)
    assert (rev.rows_before, rev.rows_after) == (30, 30)
    assert rev.speed_before == pytest.approx(2.0) and rev.speed_after == pytest.approx(2.0)


def test_noise_does_not_move_the_cut_but_moves_the_smallest_cosine():
    t, cx = out_and_back()
    cy = 100.0 + np.random.default_rng(0).normal(0.0, 0.5, 60)
    b = boxes_at(cx, cy)
    run = np.arange(23, 39)
    assert hit_rows(t, b, FPS, RC).tolist() == run.tolist()
    assert int(run[np.argmin(row_values(t, b, FPS, RC).cos[run])]) == 33  # the script's choice
    assert cuts(t, b) == [30]


def test_the_variable_height_fixture_is_cut_at_25():
    t, cx = out_and_back()
    h = np.where((t >= 15) & (t <= 30), 50.0, 10.0)
    b = boxes_at(cx, h=h)
    assert hit_rows(t, b, FPS, RC).tolist() == [23, 24, 25]
    assert cuts(t, b) == [25]


def test_near_the_start_the_stage_keeps_a_cut_the_script_drops():
    t, cx = out_and_back(n=30, turn=5)
    b = boxes_at(cx)
    assert hit_rows(t, b, FPS, RC).tolist() == list(range(3, 10))
    (rev,) = find_reversals(t, b, FPS, RC)
    assert (rev.row, rev.rows_before, rev.rows_after) == (5, 5, 25)


def test_near_the_end_the_cut_keeps_exactly_min_rows():
    t, cx = out_and_back(n=30, turn=25)
    b = boxes_at(cx)
    assert hit_rows(t, b, FPS, RC).tolist() == list(range(23, 28))
    assert cuts(t, b) == [25]


def test_two_reversals_give_two_cuts():
    t = np.arange(110)
    v = np.where(t <= 30, 2.0, np.where(t <= 70, -2.0, 2.0))
    b = boxes_at(path(v))
    assert [(int(r[0]), int(r[-1])) for r in runs(hit_rows(t, b, FPS, RC))] == [(23, 38), (63, 78)]
    assert cuts(t, b) == [30, 70]


def test_a_straight_track_with_a_gap_is_not_cut():
    frames = np.r_[0:40, 55:95]
    assert cuts(frames, boxes_at(100.0 + 2.0 * frames)) == []


def test_a_reversal_across_a_gap_is_cut_at_the_first_row_after_the_gap():
    frames = np.r_[0:30, 40:70]
    x = np.where(frames < 30, 2.0 * frames, 60.0 - 2.0 * (frames - 30))
    (rev,) = find_reversals(frames, boxes_at(100.0 + x), FPS, RC)
    assert (rev.row, rev.frame) == (30, 40)


def test_queue_jitter_is_not_cut_at_the_default_speed_but_is_at_0_2():
    t = np.arange(200)
    b = boxes_at(100.0 + 2.0 * np.sin(2.0 * np.pi * t / 40.0))
    assert cuts(t, b) == []
    assert cuts(t, b, dataclasses.replace(RC, min_speed=0.2))  # why the default is 0.5


@pytest.mark.parametrize(
    "name",
    ["stop_and_go", "right_angle", "parked"],
)
def test_motions_that_are_not_reversals_are_not_cut(name):
    if name == "stop_and_go":
        t = np.arange(160)
        b = boxes_at(path(np.tile(np.r_[np.full(20, 2.0), np.zeros(20)], 4)))
    elif name == "right_angle":
        t = np.arange(60)
        x = 100.0 + np.where(t < 30, 2.0 * t, 58.0)
        y = 100.0 - np.where(t < 30, 0.0, 2.0 * (t - 29))
        b = boxes_at(x, y)
    else:
        rng = np.random.default_rng(0)
        t = np.arange(100)
        b = boxes_at(100.0 + rng.uniform(-2, 2, 100), 100.0 + rng.uniform(-2, 2, 100))
    assert cuts(t, b) == []


def test_a_reversal_with_a_short_tail_has_hits_but_no_cut():
    t = np.arange(33)
    b = boxes_at(100.0 + np.where(t < 30, 2.0 * t, 58.0 - 2.0 * (t - 29)))
    assert len(hit_rows(t, b, FPS, RC)) > 0
    assert cuts(t, b) == []


def test_a_track_shorter_than_two_parts_has_no_hits():
    t = np.arange(9)
    b = boxes_at(100.0 + np.where(t < 5, 4.0 * t, 32.0 - 4.0 * t))
    assert hit_rows(t, b, FPS, RC).size == 0 and cuts(t, b) == []


def test_a_30_fps_out_and_back_is_cut_at_the_turn():
    t, cx = out_and_back(n=180, turn=90, speed=2.0 / 3.0)  # 2 box heights per second
    assert cuts(t, boxes_at(cx), fps=30.0) == [90]


def test_repeated_frames_give_no_division_warning():
    frames = np.r_[0, 0, 0, 0, 1:60]
    x = 100.0 + np.where(frames < 30, 2.0 * frames, 120.0 - 2.0 * frames)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (rev,) = find_reversals(frames, boxes_at(x), FPS, RC)
    assert rev.frame == 30


def test_keep_cuts_measures_each_part_from_the_previous_kept_cut():
    assert keep_cuts([3, 7, 20], 30, 5) == [7, 20]  # 3 leaves a 3-row head
    assert keep_cuts([10, 13, 20], 30, 5) == [10, 20]  # 13 leaves a 3-row middle
    assert keep_cuts([26], 30, 5) == []  # a 4-row tail
    assert keep_cuts([], 30, 5) == []


def test_cut_in_run_takes_the_earliest_of_tied_changes():
    change = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0 * (1 + 1e-12)])
    assert cut_in_run(np.array([4, 5, 6]), change) == 5
    assert cut_in_run(np.array([4, 5, 6]), np.array([0, 0, 0, 0, 1.0, 2.0, 3.0])) == 6
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_rule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnt.refine.reversal'`.

- [ ] **Step 4: Create `src/dnt/refine/reversal.py` with the rule**

```python
"""Stage ``reversal``: split vehicle tracks where the direction reverses (reversal spec).

Design: docs/superpowers/specs/2026-11-04-vehicle-reversal-split-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ReversalConfig, to_frames
from .primitives import mean_velocity

STAGE = "reversal"
_TIE_RTOL = 1e-9


@dataclass(frozen=True)
class Reversal:
    """One kept cut of a track (spec 3.2); ``run`` is the first and last frame of its run."""

    row: int
    frame: int
    angle: float
    change: float
    run: tuple[int, int]
    speed_before: float
    speed_after: float
    rows_before: int
    rows_after: int


@dataclass(frozen=True)
class RowValues:
    """Per-row values of the rule; ``cos`` and ``change`` are NaN where a row has no value."""

    cos: np.ndarray
    change: np.ndarray
    speed_before: np.ndarray
    speed_after: np.ndarray


def row_values(frames, boxes, fps: float, rc: ReversalConfig) -> RowValues:
    """Return each row's cosine, velocity change and side speeds (spec 3.2 steps 2-3).

    Parameters
    ----------
    frames : array_like of int
        The track's frames, sorted ascending.
    boxes : array_like
        ``(n, 4)`` boxes ``x, y, w, h`` in the same order.
    fps : float
        Frame rate; converts ``rc.window`` to frames and speeds to per second.
    rc : ReversalConfig
        The stage settings.

    Returns
    -------
    RowValues
        Speeds are in box heights per second, ``change`` in pixels per frame.

    """
    frames = np.asarray(frames, dtype=int)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(frames)
    cx = b[:, 0] + b[:, 2] / 2.0
    cy = b[:, 1] + b[:, 3] / 2.0
    h = b[:, 3]
    w = to_frames(rc.window, fps)
    cos = np.full(n, np.nan)
    change = np.full(n, np.nan)
    sb = np.zeros(n)
    sa = np.zeros(n)
    for i in range(1, n):
        t = frames[i]
        lo = int(np.searchsorted(frames, t - w, side="left"))
        hi = int(np.searchsorted(frames, t + w, side="left"))
        if i - lo < rc.min_side_rows or hi - i < rc.min_side_rows:
            continue
        if frames[lo] == frames[i - 1] or frames[i] == frames[hi - 1]:
            continue  # one distinct frame on a side has no slope (spec 8)
        vb = mean_velocity(frames[lo:i], cx[lo:i], cy[lo:i])
        va = mean_velocity(frames[i:hi], cx[i:hi], cy[i:hi])
        nb, na = float(np.hypot(*vb)), float(np.hypot(*va))
        if nb == 0.0 or na == 0.0:
            continue
        s_b = nb * fps / max(float(np.median(h[lo:i])), 1.0)
        s_a = na * fps / max(float(np.median(h[i:hi])), 1.0)
        if s_b < rc.min_speed or s_a < rc.min_speed:
            continue
        cos[i] = float(vb @ va) / (nb * na)
        change[i] = float(np.hypot(*(va - vb)))
        sb[i], sa[i] = s_b, s_a
    return RowValues(cos, change, sb, sa)


def _hits(v: RowValues, rc: ReversalConfig) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.flatnonzero(v.cos <= np.cos(np.radians(rc.min_angle)))


def hit_rows(frames, boxes, fps: float, rc: ReversalConfig) -> np.ndarray:
    """Return the rows that turn by at least ``rc.min_angle`` (spec 3.2 steps 1-4).

    These are the values the script's ``hit`` array holds; a track shorter than
    ``2 * rc.min_rows`` rows has none.
    """
    if len(frames) < 2 * rc.min_rows:
        return np.empty(0, dtype=int)
    return _hits(row_values(frames, boxes, fps, rc), rc)


def runs(hit: np.ndarray) -> list[np.ndarray]:
    """Split sorted hit rows into runs of consecutive rows (spec 3.2 step 5)."""
    hit = np.asarray(hit, dtype=int)
    if len(hit) == 0:
        return []
    return np.split(hit, np.flatnonzero(np.diff(hit) > 1) + 1)


def cut_in_run(run: np.ndarray, change: np.ndarray) -> int:
    """Return the run's row with the largest velocity change, the earliest of ties (step 6)."""
    c = np.asarray(change, dtype=float)[run]
    best = float(np.max(c))
    tied = np.flatnonzero(c >= best - _TIE_RTOL * max(1.0, abs(best)))
    return int(run[int(tied[0])])


def keep_cuts(proposed, n: int, min_rows: int) -> list[int]:
    """Keep, in row order, the cuts whose parts each keep ``min_rows`` rows (step 7)."""
    kept: list[int] = []
    prev = 0
    for i in sorted(int(x) for x in proposed):
        if i - prev >= min_rows and n - i >= min_rows:
            kept.append(i)
            prev = i
    return kept


def find_reversals(frames, boxes, fps: float, rc: ReversalConfig) -> list[Reversal]:
    """Return the kept cuts of one track, in frame order (spec 3.2).

    Parameters
    ----------
    frames, boxes, fps, rc
        As for ``row_values``.

    Returns
    -------
    list of Reversal
        The row of each cut is the first row of the part after it.

    """
    frames = np.asarray(frames, dtype=int)
    n = len(frames)
    if n < 2 * rc.min_rows:
        return []
    v = row_values(frames, boxes, fps, rc)
    proposed = {cut_in_run(r, v.change): r for r in runs(_hits(v, rc))}
    kept = keep_cuts(list(proposed), n, rc.min_rows)
    out: list[Reversal] = []
    for k, i in enumerate(kept):
        run = proposed[i]
        prev = kept[k - 1] if k else 0
        nxt = kept[k + 1] if k + 1 < len(kept) else n
        out.append(
            Reversal(
                row=i,
                frame=int(frames[i]),
                angle=float(np.degrees(np.arccos(np.clip(v.cos[i], -1.0, 1.0)))),
                change=float(v.change[i]),
                run=(int(frames[run[0]]), int(frames[run[-1]])),
                speed_before=float(v.speed_before[i]),
                speed_after=float(v.speed_after[i]),
                rows_before=i - prev,
                rows_after=nxt - i,
            )
        )
    return out
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_rule.py -v`
Expected: all PASS. If a measured number differs (for example a run boundary), do not edit the expected value. Recheck the implementation against spec §3.2: these values were measured on this exact rule.

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check src/dnt/refine/reversal.py tests/refine/_reversal.py tests/refine/test_reversal_rule.py`
Expected: `All checks passed!`

```bash
git add src/dnt/refine/reversal.py tests/refine/_reversal.py tests/refine/test_reversal_rule.py
git commit -m "feat(refine): find vehicle reversals: hit runs, the turn row, part lengths

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Parity with the script's hit rows

**Files:**
- Test: `tests/refine/test_reversal_parity.py`

**Interfaces:**
- Consumes: `hit_rows`, `row_values`, `runs`, `find_reversals` (Task 3).

- [ ] **Step 1: Write the parity test**

The oracle is the script's `_slope` and `find_reversals` (from `3a1-split_reversals.py`, as pasted in the design conversation), copied verbatim. It is cut at the line `hit = np.flatnonzero(cos <= cos_max)` and returns `hit`. The script's speed and window bookkeeping that only feeds later steps (`sp_b`, `sp_a`) is dropped.

```python
"""Parity with the Miami driver script ``3a1-split_reversals.py`` (reversal spec 6.3).

``_slope`` and the body of ``oracle_hit_rows`` are copied word for word from the script's
``_slope`` and ``find_reversals``, cut at ``hit = np.flatnonzero(cos <= cos_max)``.
"""

import numpy as np
import pandas as pd

from dnt.refine.config import RefineConfig
from dnt.refine.reversal import find_reversals, hit_rows, row_values, runs

WINDOW_S = 2.0
MIN_SPEED = 0.5
MIN_ANGLE = 135.0
MIN_ROWS = 5
MIN_SIDE_ROWS = 3
RC = RefineConfig.defaults("vehicle").reversal


def _slope(frames: np.ndarray, vals: np.ndarray) -> float:
    """Least-squares slope of ``vals`` over ``frames`` (units per frame)."""
    t = frames - frames.mean()
    den = float((t * t).sum())
    return float((t * (vals - vals.mean())).sum() / den) if den > 0 else 0.0


def oracle_hit_rows(g: pd.DataFrame, fps: float) -> np.ndarray:
    frames = g["frame"].to_numpy()
    n = len(frames)
    if n < 2 * MIN_ROWS:
        return np.empty(0, dtype=int)
    cx = (g["x"] + g["w"] / 2.0).to_numpy(float)
    cy = (g["y"] + g["h"] / 2.0).to_numpy(float)
    h = g["h"].to_numpy(float)
    w = max(1, round(WINDOW_S * fps))
    cos_max = np.cos(np.radians(MIN_ANGLE))
    cos = np.full(n, np.nan)
    for i in range(1, n):
        t = frames[i]
        lo = int(np.searchsorted(frames, t - w, side="left"))
        hi = int(np.searchsorted(frames, t + w, side="left"))
        if i - lo < MIN_SIDE_ROWS or hi - i < MIN_SIDE_ROWS:
            continue
        vb = np.array([_slope(frames[lo:i], cx[lo:i]), _slope(frames[lo:i], cy[lo:i])])
        va = np.array([_slope(frames[i:hi], cx[i:hi]), _slope(frames[i:hi], cy[i:hi])])
        nb, na = float(np.hypot(*vb)), float(np.hypot(*va))
        sb = nb * fps / max(float(np.median(h[lo:i])), 1.0)
        sa = na * fps / max(float(np.median(h[i:hi])), 1.0)
        if sb < MIN_SPEED or sa < MIN_SPEED:
            continue
        cos[i] = float(vb @ va) / (nb * na)
    return np.flatnonzero(cos <= cos_max)


def random_track(seed: int) -> pd.DataFrame:
    """A piecewise-linear 2-D path: 1-3 legs, random turns (often reversals), gaps and noise."""
    rng = np.random.default_rng(seed)
    h = float(rng.uniform(8, 40))
    w = 0.8 * h
    f, px, py = 0, 500.0, 500.0
    head = float(rng.uniform(0, 2 * np.pi))
    fr, xs, ys = [], [], []
    for leg in range(int(rng.integers(1, 4))):
        if leg:
            head += np.pi if rng.random() < 0.6 else float(rng.uniform(-np.pi, np.pi))
        sp = float(rng.uniform(0.1, 3.0)) * h / 10.0
        for _ in range(int(rng.integers(8, 50))):
            if rng.random() < 0.05:
                f += int(rng.integers(2, 12))
            px += sp * np.cos(head)
            py += sp * np.sin(head)
            fr.append(f)
            xs.append(px + rng.normal(0, 0.5))
            ys.append(py + rng.normal(0, 0.5))
            f += 1
    cx, cy = np.array(xs), np.array(ys)
    return pd.DataFrame({"frame": fr, "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h})


def _ref_keep(proposed, n, min_rows):
    """Independent reference of spec 3.2 step 7."""
    out, prev = [], 0
    for i in sorted(proposed):
        if i - prev < min_rows or n - i < min_rows:
            continue
        out.append(i)
        prev = i
    return out


def test_the_hit_rows_equal_the_scripts_on_random_tracks():
    with_hits = 0
    for seed in range(200):
        g = random_track(seed)
        boxes = g[["x", "y", "w", "h"]].to_numpy()
        expected = oracle_hit_rows(g, 10.0)
        assert hit_rows(g["frame"].to_numpy(), boxes, 10.0, RC).tolist() == expected.tolist(), seed
        with_hits += bool(len(expected))
    assert with_hits >= 60  # 80 of the 200 seeds hold a reversal; the test must exercise them


def test_cuts_follow_the_turn_rule_and_the_length_rule_on_the_stages_own_rows():
    for seed in range(200):
        g = random_track(seed)
        frames, boxes = g["frame"].to_numpy(), g[["x", "y", "w", "h"]].to_numpy()
        v = row_values(frames, boxes, 10.0, RC)
        proposed = []
        for run in runs(hit_rows(frames, boxes, 10.0, RC)):
            c = v.change[run]
            proposed.append(int(run[np.flatnonzero(c >= c.max() - 1e-9 * max(1.0, c.max()))[0]]))
        expected = _ref_keep(proposed, len(frames), RC.min_rows)
        assert [r.row for r in find_reversals(frames, boxes, 10.0, RC)] == expected, seed
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_parity.py -v`
Expected: both PASS. A prototype of Task 3's rule matched the oracle on all 200 seeds, with 80 of them holding hits.

- [ ] **Step 3: Commit**

```bash
git add tests/refine/test_reversal_parity.py
git commit -m "test(refine): pin parity of the reversal hit rows with the driver script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Side tags and the chain resolver

**Files:**
- Modify: `src/dnt/refine/reversal.py` (append)
- Test: `tests/refine/test_reversal_sides.py`

**Interfaces:**
- Consumes: `apply.lineage_of_rows`, `events.Event`.
- Produces, in `dnt.refine.reversal`:
  - `Tag = tuple[int, int]`
  - `side_conflict(tags) -> tuple[int, int, int] | None`
  - `ReversalSides` (dataclass). It has the fields `cuts: dict[int, list[int]]` (sorted cut frames) and `carried: dict[int, set[Tag]]`, and these methods:
    - `own(lineage) -> set[Tag]`
    - `tags(track, lineage) -> set[Tag]`
    - `table_tags(work, excluded=None) -> dict[int, set[Tag]]`
    - `absorb(mapping, tags_before) -> None`
  - `resolve_reversal_chains(active: list[Event], tags_of: dict[int, set[Tag]]) -> list[Event]`. It returns the dropped events in drop order and marks each one.

- [ ] **Step 1: Write the failing tests**

```python
from dnt.refine.reversal import ReversalSides, resolve_reversal_chains, side_conflict

from ._reversal import link_event, rows, work_of


def test_own_tags_number_the_sides_of_each_cut_raw_id():
    s = ReversalSides(cuts={1: [30, 70]})
    assert s.own([[1, 0, 29]]) == {(1, 0)}
    assert s.own([[1, 30, 69]]) == {(1, 1)}  # a span starting at a cut is after it
    assert s.own([[1, 70, 99]]) == {(1, 2)}
    assert s.own([[2, 0, 99]]) == set()  # raw 2 has no cuts
    assert s.own([[1, 0, 29], [2, 30, 40]]) == {(1, 0)}


def test_side_conflict_names_the_smallest_raw_and_its_two_smallest_sides():
    assert side_conflict({(3, 0), (3, 1), (1, 2), (1, 0), (1, 1)}) == (1, 0, 1)
    assert side_conflict({(1, 0), (2, 1)}) is None
    assert side_conflict(set()) is None


def test_absorb_keeps_the_representatives_own_tags():
    # the review's minimal case: track 3 merged into 1, only 1 had a side, and 1's own rows
    # are all gone (its lineage now names raw 3 only)
    s = ReversalSides(cuts={1: [30]})
    s.absorb({3: 1}, {1: {(1, 0)}, 3: set()})
    assert s.tags(1, [[3, 0, 29]]) == {(1, 0)}


def test_tags_carried_by_dedup_survive_a_later_link_map():
    s = ReversalSides(cuts={1: [30]})
    s.absorb({3: 1}, {1: {(1, 0)}, 2: {(1, 1)}, 3: set()})  # dedup
    before_link = {1: s.tags(1, [[3, 0, 29]]), 5: set()}
    s.absorb({1: 1, 5: 1}, before_link)  # link: 5 joined 1
    assert (1, 0) in s.tags(1, [[3, 0, 29], [5, 40, 60]])


def test_table_tags_read_each_tracks_lineage():
    work = work_of(rows(1, range(0, 30)), rows(2, range(30, 60)), rows(3, range(0, 10)),
                   raw={2: 1})
    s = ReversalSides(cuts={1: [30]})
    assert s.table_tags(work) == {1: {(1, 0)}, 2: {(1, 1)}, 3: set()}


def test_the_chain_check_drops_the_weakest_link_on_the_path_only():
    # 1 - 2 - 3 - 4 - 5; tracks 2 and 4 hold the two sides of raw 7's cut
    evs = [link_event(1, 2, 0.9), link_event(2, 3, 0.8), link_event(3, 4, 0.95),
           link_event(4, 5, 0.5)]
    tags = {1: set(), 2: {(7, 0)}, 3: set(), 4: {(7, 1)}, 5: set()}
    (dropped,) = resolve_reversal_chains(evs, tags)
    assert dropped.tracks == [2, 3]  # 4 -> 5 is weaker but not on the 2..4 path
    assert dropped.applied is False and dropped.signals["skipped_reason"] == "reversal"
    assert dropped.signals["conflicts_with"] == {"raw": 7, "sides": [0, 1], "tracks": [2, 4]}
    assert all(e.applied for e in evs if e is not dropped)


def test_every_path_between_the_sides_is_broken():
    # a diamond: 1-2-4 and 1-3-4, with 1 and 4 on opposite sides
    evs = [link_event(1, 2, 0.9), link_event(2, 4, 0.6), link_event(1, 3, 0.9),
           link_event(3, 4, 0.7)]
    tags = {1: {(7, 0)}, 2: set(), 3: set(), 4: {(7, 1)}}
    dropped = resolve_reversal_chains(evs, tags)
    assert [e.tracks for e in dropped] == [[2, 4], [3, 4]]


def test_a_two_way_pair_uses_its_weaker_link_and_no_conflict_drops_nothing():
    evs = [link_event(1, 2, 0.9), link_event(2, 1, 0.7, [[2, 0, 0]], [[1, 1, 1]])]
    (dropped,) = resolve_reversal_chains(evs, {1: {(7, 0)}, 2: {(7, 1)}})
    assert dropped.tracks == [2, 1]
    calm = [link_event(1, 2, 0.9)]
    assert resolve_reversal_chains(calm, {1: {(7, 0)}, 2: {(8, 1)}}) == []
    assert calm[0].applied


def test_a_track_that_holds_both_sides_alone_does_not_loop_forever():
    evs = [link_event(1, 2, 0.9)]
    assert resolve_reversal_chains(evs, {1: {(7, 0), (7, 1)}, 2: set()}) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_sides.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReversalSides'`.

- [ ] **Step 3: Implement**

Add these imports to `src/dnt/refine/reversal.py`. Keep the import block sorted; `ruff --fix` sorts it.

```python
import logging
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .apply import lineage_of_rows
from .events import Event
```

Below `STAGE`, add `log = logging.getLogger(__name__)`. Then append:

```python
Tag = tuple[int, int]


def side_conflict(tags) -> tuple[int, int, int] | None:
    """Return ``(r, k1, k2)`` for the smallest raw ID with two sides in ``tags``, else None.

    ``k1 < k2`` are that raw ID's two smallest sides (spec 3.5).
    """
    by: dict[int, set[int]] = {}
    for r, k in tags:
        by.setdefault(int(r), set()).add(int(k))
    for r in sorted(by):
        ks = sorted(by[r])
        if len(ks) > 1:
            return r, ks[0], ks[1]
    return None


@dataclass
class ReversalSides:
    """Reversal cuts and the side tags each work track carries (spec 3.5).

    A tag ``(r, k)`` is side ``k`` of raw ID ``r``: the frames from its ``k``-th cut to the next.
    A track's tags are its own (read from its lineage) plus those it ``carried`` from tracks
    merged into it, so a side survives a merge that drops all of its rows.
    """

    cuts: dict[int, list[int]] = field(default_factory=dict)
    carried: dict[int, set[Tag]] = field(default_factory=dict)

    def own(self, lineage) -> set[Tag]:
        """Return the tags of a lineage's spans of cut raw IDs."""
        out: set[Tag] = set()
        for raw, first, _last in lineage:
            cs = self.cuts.get(int(raw))
            if cs:
                out.add((int(raw), bisect_right(cs, int(first))))
        return out

    def tags(self, track: int, lineage) -> set[Tag]:
        """Return ``own(lineage)`` plus the tags ``track`` carries."""
        return self.own(lineage) | self.carried.get(int(track), set())

    def table_tags(self, work: pd.DataFrame, excluded=None) -> dict[int, set[Tag]]:
        """Return the tags of every track of ``work``."""
        return {
            int(t): self.tags(int(t), lineage_of_rows(g, excluded))
            for t, g in work.groupby("track", sort=True)
        }

    def absorb(self, mapping, tags_before) -> None:
        """Carry every pre-merge track's tags into the track it now belongs to.

        ``tags_before`` holds the tags of every track before the merges, representatives
        included; ``mapping`` maps absorbed work IDs to survivors (``rep_of``, ``absorbed``).
        """
        for track, tags in tags_before.items():
            rep = int(mapping.get(track, track))
            self.carried.setdefault(rep, set()).update(tags)


def _weaker_key(e: Event) -> tuple[float, str]:
    return (float(e.algo_score), e.proposal_key)


def _graph(active: list[Event]) -> dict[int, dict[int, Event]]:
    adj: dict[int, dict[int, Event]] = {}
    for e in active:
        a, b = int(e.tracks[0]), int(e.tracks[1])
        for x, y in ((a, b), (b, a)):
            nb = adj.setdefault(x, {})
            if y not in nb or _weaker_key(e) < _weaker_key(nb[y]):
                nb[y] = e  # two links between one pair: the path step uses the weaker
    return adj


def _bfs_path(adj, u: int, v: int) -> list[int]:
    prev: dict[int, int | None] = {u: None}
    queue = deque([u])
    while queue:
        x = queue.popleft()
        if x == v:
            break
        for y in sorted(adj[x]):
            if y not in prev:
                prev[y] = x
                queue.append(y)
    out = [v]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return out[::-1]


def _first_side_conflict(active: list[Event], tags_of):
    adj = _graph(active)
    seen: set[int] = set()
    for start in sorted(adj):  # the first unseen node is its component's smallest member
        if start in seen:
            continue
        comp, queue = [], deque([start])
        seen.add(start)
        while queue:
            x = queue.popleft()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        conflict = side_conflict(set().union(*(tags_of.get(t, set()) for t in comp)))
        if conflict is None:
            continue
        r, k1, k2 = conflict
        u = min(t for t in comp if (r, k1) in tags_of.get(t, set()))
        v = min(t for t in comp if (r, k2) in tags_of.get(t, set()))
        if u == v:  # one track holds both sides; no link can undo that (spec 3.5)
            log.warning("track %d holds both sides of raw %d's reversal cut", u, r)
            continue
        steps = _bfs_path(adj, u, v)
        on_path = [adj[a][b] for a, b in zip(steps, steps[1:], strict=False)]
        return min(on_path, key=_weaker_key), r, k1, k2, u, v
    return None


def resolve_reversal_chains(active: list[Event], tags_of) -> list[Event]:
    """Drop links until no chain of ``active`` links holds both sides of a cut (spec 3.5).

    Each round drops the weakest link, by ``(algo_score, proposal_key)``, on the path
    between the two smallest conflicting members of the first conflicting component.

    Parameters
    ----------
    active : list of Event
        Accepted LINK events that will be applied.
    tags_of : dict
        Work track ID to its tags (``ReversalSides.tags``), for every track in ``active``.

    Returns
    -------
    list of Event
        The dropped events, each with ``applied=False``, ``skipped_reason="reversal"`` and
        ``conflicts_with``.

    """
    active = list(active)
    dropped: list[Event] = []
    while (hit := _first_side_conflict(active, tags_of)) is not None:
        ev, r, k1, k2, u, v = hit
        ev.applied = False
        ev.signals["skipped_reason"] = "reversal"
        ev.signals["conflicts_with"] = {"raw": r, "sides": [k1, k2], "tracks": [u, v]}
        active = [e for e in active if e is not ev]
        dropped.append(ev)
    return dropped
```

The `u == v` branch can skip a component with a conflict, so a second conflict in that same component would also be skipped. That is acceptable, because dedup and link refuse any join that would put both sides on one track. Spec §3.5 states this as an impossibility.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_sides.py tests/refine/test_reversal_rule.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check src/dnt/refine/reversal.py tests/refine/test_reversal_sides.py`

```bash
git add src/dnt/refine/reversal.py tests/refine/test_reversal_sides.py
git commit -m "feat(refine): side tags for reversal cuts and a chain resolver over the link graph

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The stage in the pipeline

**Files:**
- Modify: `src/dnt/refine/reversal.py` (append `propose_reversal_splits`)
- Modify: `src/dnt/refine/refiner.py` (`_Stages.__init__`, `run`, the new `_reversal` and `_sides`, the stage numbering, the `Reporter` count)
- Test: `tests/refine/test_reversal_stage.py`

**Interfaces:**
- Consumes: `find_reversals`, `ReversalSides` (Tasks 3 and 5); `verify.decide`; `apply.apply_edit` and `next_track_id`.
- Produces:
  - `reversal.propose_reversal_splits(work, cfg, fps) -> list[Event]` (the events are undecided);
  - `_Stages.reversal_sides: ReversalSides`, which is set by `_reversal`;
  - `_Stages._sides() -> ReversalSides | None`, which returns `None` when there are no cuts;
  - `refiner._RUN_STAGES = 7`.

  Tasks 7 to 9 call `self._sides()`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind
from dnt.refine.refiner import TrackRefiner, _Stages
from dnt.refine.reversal import propose_reversal_splits

from ._fixtures import box_rows, table
from ._reversal import FPS, cx_out_and_back, rows, var_height, work_of


def _cfg():
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    return cfg


def _refine(tmp_path, df, cfg=None, name="t.txt", **kw):
    src = tmp_path / name
    df.to_csv(src, index=False, header=False)
    refiner = TrackRefiner(cfg or _cfg())
    out = refiner.refine(src, tmp_path / f"o_{name}", fps=FPS, verbose=kw.pop("verbose", False))
    return refiner, out


def test_events_have_the_spec_fields():
    work = work_of(rows(1, range(60)))
    (ev,) = propose_reversal_splits(work, _cfg(), FPS)
    assert (ev.stage, ev.kind, ev.tracks, ev.frames) == ("reversal", EventKind.SPLIT, [1], (30, 30))
    assert ev.params == {"cut_frame": 30} and ev.algo_score == 1.0 and ev.decision is None
    assert ev.lineage == [[[1, 0, 59]]]
    assert set(ev.signals) == {"angle", "change", "run", "speed_before", "speed_after",
                               "rows_before", "rows_after"}
    assert ev.signals["run"] == [23, 38]


def test_keys_do_not_depend_on_work_ids():
    work = work_of(rows(1, range(60)))
    moved = work.copy()
    moved["track"] += 1000
    (a,), (b,) = (propose_reversal_splits(w, _cfg(), FPS) for w in (work, moved))
    assert a.proposal_key == b.proposal_key and b.tracks == [1001]


def test_keys_identify_the_raw_track():
    (a,) = propose_reversal_splits(work_of(rows(1, range(60))), _cfg(), FPS)
    (b,) = propose_reversal_splits(work_of(rows(7, range(60))), _cfg(), FPS)
    assert a.proposal_key != b.proposal_key


def test_the_stage_splits_applies_and_records_the_sides():
    stages = _Stages(_cfg(), FPS, None, None, None, None, {})
    out, events = stages.run(work_of(rows(1, range(60))))
    (ev,) = [e for e in events if e.stage == "reversal"]
    assert ev.id == "reversal-r0-000001" and ev.decision is Decision.AUTO_ACCEPT and ev.applied
    assert ev.edit == {"kind": "SPLIT", "params": {"cut_frame": 30}}
    new = ev.signals["new_track"]
    assert ev.signals["sides"] == [[1, 0], [1, 1]]
    assert set(out["track"]) == {1, new}
    assert out.loc[out["track"] == 1, "frame"].max() == 29
    assert out.loc[out["track"] == new, "frame"].min() == 30
    assert stages.reversal_sides.cuts == {1: [30]}


def test_two_reversals_make_three_ids_with_three_sides():
    t = np.arange(110)
    x = {int(f): v for f, v in zip(t, 100.0 + np.r_[0.0, np.cumsum(
        np.where(t <= 30, 2.0, np.where(t <= 70, -2.0, 2.0))[1:])])}
    stages = _Stages(_cfg(), FPS, None, None, None, None, {})
    out, events = stages.run(work_of(rows(1, t, lambda f: x[int(f)])))
    revs = sorted((e for e in events if e.stage == "reversal"),
                  key=lambda e: e.params["cut_frame"])
    assert [e.params["cut_frame"] for e in revs] == [30, 70]
    assert [e.signals["sides"] for e in revs] == [[[1, 0], [1, 1]], [[1, 1], [1, 2]]]
    assert out["track"].nunique() == 3


def test_the_screen_stage_is_told_which_raw_ids_were_split(monkeypatch):
    seen = {}

    def fake_screen(work, cfg, fps, sctx, cuts):
        seen["raw"] = set(sctx.split_raw_ids)
        return []

    monkeypatch.setattr("dnt.refine.refiner.propose_screen", fake_screen)
    _Stages(_cfg(), FPS, None, None, None, None, {}).run(work_of(rows(1, range(60))))
    assert 1 in seen["raw"]


def test_the_person_target_leaves_tracks_alone():
    cfg = RefineConfig.defaults("person")
    stages = _Stages(cfg, FPS, None, None, None, None, {})
    _, events = stages.run(work_of(rows(1, range(60), cls=0)))
    assert not [e for e in events if e.stage == "reversal"]
    assert stages._sides() is None


def test_a_table_of_short_tracks_is_left_alone():
    work = work_of(rows(1, range(9)), rows(2, range(20, 29)))
    stages = _Stages(_cfg(), FPS, None, None, None, None, {})
    _, events = stages.run(work)
    assert not [e for e in events if e.stage == "reversal"] and stages._sides() is None


@pytest.mark.parametrize("h_of", [lambda f: 10.0, var_height])
def test_refine_writes_the_two_vehicles_as_two_tracks(tmp_path, h_of):
    refiner, out = _refine(tmp_path, table(rows(1, range(60), h_of=h_of)))
    assert out["track"].nunique() == 2
    ev = next(e for e in refiner.last_result.events if e.stage == "reversal")
    assert ev.params["cut_frame"] == (30 if h_of(0) == h_of(20) else 25)
    assert not [e for e in refiner.last_result.events if e.kind is EventKind.LINK and e.applied]


def test_progress_numbers_the_new_stage_first(tmp_path, capsys):
    _refine(tmp_path, table(rows(1, range(60))), verbose=True)
    err = capsys.readouterr().err
    assert "Stage 1/8 reversal" in err and "Stage 2/8 switch" in err and "Stage 8/8 Save" in err


def test_a_reused_refiner_does_not_carry_cuts_between_clips(tmp_path):
    clip1 = table(rows(1, range(60)))
    clip2 = table(box_rows(1, range(0, 60), 100.0, 100.0, vx=3.0, w=40.0, h=30.0, cls=2),
                  box_rows(5, range(61, 100), 100.0 + 3.0 * 61, 100.0, vx=3.0, w=40.0,
                           h=30.0, cls=2))
    refiner = TrackRefiner(_cfg())
    for i, df in enumerate((clip1, clip2)):
        p = tmp_path / f"c{i}.txt"
        df.to_csv(p, index=False, header=False)
        reused = refiner.refine(p, tmp_path / f"r{i}.txt", fps=FPS, verbose=False)
    _, fresh = _refine(tmp_path, clip2, name="fresh.txt")
    pd.testing.assert_frame_equal(reused.reset_index(drop=True), fresh.reset_index(drop=True))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_stage.py -v`
Expected: FAIL with `ImportError: cannot import name 'propose_reversal_splits'`.

- [ ] **Step 3: Add `propose_reversal_splits` to `reversal.py`**

Add to the imports: `from .config import RefineConfig, ReversalConfig, to_frames`, `from .events import Event, EventKind`, and `from .reporter import advance`. Then append:

```python
def propose_reversal_splits(work: pd.DataFrame, cfg: RefineConfig, fps: float) -> list[Event]:
    """Propose one undecided ``SPLIT`` event per kept cut of every track (spec 3.3).

    Parameters
    ----------
    work : pandas.DataFrame
        Work table from ``io.to_work``.
    cfg : RefineConfig
        ``cfg.reversal`` holds the thresholds.
    fps : float
        Video frame rate.

    Returns
    -------
    list of Event
        In track order, then frame order. The caller decides and applies them.

    """
    events: list[Event] = []
    if work.empty:
        return events
    for tid, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        found = find_reversals(
            g["frame"].to_numpy(int), g[["x", "y", "w", "h"]].to_numpy(float), fps, cfg.reversal
        )
        advance()
        lin = lineage_of_rows(g)
        for r in found:
            events.append(
                Event.propose(
                    stage=STAGE,
                    kind=EventKind.SPLIT,
                    tracks=[int(tid)],
                    lineage=[lin],
                    frames=(r.frame, r.frame),
                    params={"cut_frame": r.frame},
                    algo_score=1.0,
                    signals={
                        "angle": r.angle,
                        "change": r.change,
                        "run": list(r.run),
                        "speed_before": r.speed_before,
                        "speed_after": r.speed_after,
                        "rows_before": r.rows_before,
                        "rows_after": r.rows_after,
                    },
                )
            )
    return events
```

- [ ] **Step 4: Wire the stage into `refiner.py`**

Imports: add `from bisect import bisect_right`, and `from .reversal import ReversalSides, propose_reversal_splits`.

Below `LEDGER_FORMAT`, add the module constant:

```python
_RUN_STAGES = 7  # the stages _Stages.run runs: reversal, switch, screen, dedup, link, orphan, fill
```

In `_Stages.__init__`, after `self.merge_counts = ...`, add:

```python
        self.reversal_sides = ReversalSides()
```

In `_Stages.run`, change the stage numbering line inside `stage()` to:

```python
            rep.stage_start(name, i + rep.n_stages - _RUN_STAGES - 1, 0 if skipped else n0)
```

Then replace the stage calls with:

```python
        work, reversal_raw = stage(1, "reversal", self._reversal, work)
        work, split_raw, cuts = stage(2, "switch", self._switch, work)
        work = stage(3, "screen", self._screen, work, split_raw | reversal_raw, cuts)
        work = stage(4, "dedup", self._dedup, work)
        work, protected, linked, pending, rep_of = stage(5, "link", self._link, work, occluded)
        self._absorb(rep_of)
        linked = linked | {rep_of.get(t, t) for t in self.merged_reps}
        pending = pending | {rep_of.get(t, t) for t in self.merge_pending}
        work = stage(6, "orphan", self._orphans, work, linked, pending)
        work = stage(7, "fill", self._fill, work, protected)
        return work, self.events
```

Add these two methods to `_Stages`, before `_switch`:

```python
    def _sides(self) -> ReversalSides | None:
        """Return the reversal sides when the stage cut anything (reversal spec 3.5)."""
        return self.reversal_sides if self.reversal_sides.cuts else None

    def _reversal(self, work):
        cfg = self.cfg
        if not cfg.reversal.enabled or work.empty:
            return work, set()
        evs = propose_reversal_splits(work, cfg, self.fps)
        raw_of: dict[int, int] = {}
        for ev in evs:
            self.seq["reversal"] += 1
            ev.id = f"reversal-r0-{self.seq['reversal']:06d}"
            decide(ev, Decision.AUTO_ACCEPT, source="auto")  # a hard rule (reversal spec 3.3)
            t, c = ev.tracks[0], int(ev.params["cut_frame"])
            rows = work.loc[(work["track"] == t) & (work["frame"] == c), "raw_id"]
            raw_of[id(ev)] = int(rows.iloc[0])
        self.events.extend(evs)
        cuts: dict[int, list[int]] = {}
        for ev in sorted(evs, key=lambda e: -int(e.params["cut_frame"])):  # latest cut first
            new_id = next_track_id(work)
            work = apply_edit(work, ev, new_id=new_id)
            ev.applied = True
            ev.signals["new_track"] = new_id
            cuts.setdefault(raw_of[id(ev)], []).append(int(ev.params["cut_frame"]))
        cuts = {raw: sorted(cs) for raw, cs in cuts.items()}
        for ev in evs:
            raw = raw_of[id(ev)]
            k = bisect_right(cuts[raw], int(ev.params["cut_frame"]))
            ev.signals["sides"] = [[raw, k - 1], [raw, k]]
        self.reversal_sides = ReversalSides(cuts=cuts)
        return work, set(cuts)
```

In `TrackRefiner.refine`, change the `Reporter` line to:

```python
        n_stages = _RUN_STAGES + 1 if store is None else _RUN_STAGES + 2  # + Save (+ Prepare)
        rep = Reporter(enabled=verbose, tag=tag, n_stages=n_stages)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_stage.py tests/refine/test_dedup_stage.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check src/dnt/refine tests/refine/test_reversal_stage.py`

```bash
git add src/dnt/refine/reversal.py src/dnt/refine/refiner.py tests/refine/test_reversal_stage.py
git commit -m "feat(refine): run the reversal stage first and apply its splits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Link never pairs the two sides of a cut

**Files:**
- Modify: `src/dnt/refine/link.py` (`_legacy_matches`, `legacy_link_events`, `score_candidates`, `run_link_stage`)
- Modify: `src/dnt/refine/refiner.py` (`_Stages._link`)
- Test: `tests/refine/test_reversal_link.py`

**Interfaces:**
- Consumes: `ReversalSides`, `side_conflict` (Task 5); `_Stages._sides()` (Task 6).
- Produces: the keyword `reversal_sides: ReversalSides | None = None` on `score_candidates`, `legacy_link_events` and `run_link_stage`. When `_Stages._link` has sides, it calls `sides.absorb(rep_of, tags_before)`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.link import run_link_stage, score_candidates
from dnt.refine.primitives import occlusion_flags
from dnt.refine.reversal import ReversalSides
from dnt.refine.verify import Band, route_without_vlm

from ._fixtures import random_tracks
from ._reversal import FPS, rows, var_height, work_of


def _cfg(mode="scored"):
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.mode = mode
    return cfg


def _cands(work, sides=None, cfg=None):
    cfg = cfg or _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, _ = score_candidates(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                                occluded=occ, reversal_sides=sides)
    return {(c.i, c.j): c for c in cands}


def _link(work, sides=None, cfg=None):
    cfg = cfg or _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    band = Band.of(cfg.link)
    return run_link_stage(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                          occluded=occ, route=lambda e: route_without_vlm(e, band),
                          reversal_sides=sides)


def _split_by_hand(cut, h_of=lambda f: 10.0, n=60):
    """Raw 1 out and back, split at ``cut`` into work tracks 1 and 2 (both raw 1)."""
    return work_of(rows(1, range(cut), h_of=h_of), rows(2, range(cut, n), h_of=h_of), raw={2: 1})


def test_the_isolated_regression_relinks_at_23_without_the_guard_and_not_with_it():
    work = _split_by_hand(23, var_height)  # the cut the script's argmin made
    res = _link(work)
    (ev,) = res.events
    assert ev.tracks == [1, 2] and ev.applied and ev.algo_score == pytest.approx(0.975)
    assert ev.signals["heading"] is None
    assert (1, 2) not in _cands(work, ReversalSides(cuts={1: [23]}))
    assert _link(work, ReversalSides(cuts={1: [23]})).events == []


def test_legacy_mode_relinks_a_clean_cut_without_the_guard_and_not_with_it():
    work = _split_by_hand(30)
    assert [e.tracks for e in _link(work, cfg=_cfg("legacy")).events] == [[1, 2]]
    assert _link(work, ReversalSides(cuts={1: [30]}), cfg=_cfg("legacy")).events == []


def test_the_sides_stay_apart_after_rows_next_to_the_cut_are_dropped():
    work = work_of(rows(1, range(0, 27)), rows(2, range(33, 60)), raw={2: 1})
    assert (1, 2) not in _cands(work, ReversalSides(cuts={1: [30]}))


def test_a_re_split_side_may_rejoin_itself_but_not_the_other_side():
    work = work_of(rows(1, range(0, 30)), rows(2, range(30, 45)), rows(3, range(45, 60)),
                   raw={2: 1, 3: 1})
    cands = _cands(work, ReversalSides(cuts={1: [30]}))
    assert (2, 3) in cands
    assert (1, 2) not in cands and (1, 3) not in cands


def test_no_part_of_a_twice_reversed_track_links_to_another():
    def x(f):
        return 100.0 + (2.0 * f if f <= 30 else 120.0 - 2.0 * f if f <= 70 else 2.0 * f - 160.0)

    work = work_of(rows(1, range(0, 30), x), rows(2, range(30, 70), x), rows(3, range(70, 110), x),
                   raw={2: 1, 3: 1})
    cands = _cands(work, ReversalSides(cuts={1: [30, 70]}))
    assert not {(1, 2), (2, 3), (1, 3)} & set(cands)


@pytest.mark.parametrize("sides", [None, ReversalSides()])
def test_link_is_unchanged_without_cuts(sides):
    work = io.to_work(random_tracks(seed=0)).work
    base = _cands(work)
    got = _cands(work, sides)
    assert sorted(base) == sorted(got)
    assert all(base[k].score == got[k].score for k in base)
```

In the twice-reversed case, check that `x` is continuous at 30 and 70: `120 - 60 = 60` and `2·30 = 60`; `120 - 140 = -20` and `140 - 160 = -20`. Both match.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_link.py -v`
Expected: FAIL with `TypeError: score_candidates() got an unexpected keyword argument 'reversal_sides'`.

- [ ] **Step 3: Implement the pair guard in `link.py`**

Add the import: `from .reversal import ReversalSides, side_conflict`. `reversal.py` does not import `link.py`, so this creates no cycle. (Task 8 adds `resolve_reversal_chains` to this import; adding it now would fail lint as unused.)

Change `_legacy_matches` so that it takes a `forbid` keyword and skips forbidden pairs before gating:

```python
def _legacy_matches(
    stitchable: list[dict], *, forbid: Callable[[int, int], bool] | None = None, **gate_kw
) -> list[tuple[int, int, float]]:
    """Return ``link_tracklets``'s Hungarian matches as ``(end_track, start_track, cost)``.

    ``forbid(end_track, start_track)`` marks a pair infeasible (reversal spec 3.5).
    """
    ends = sorted(stitchable, key=lambda d: (d["t_end"], d["track"]))
    starts = sorted(stitchable, key=lambda d: (d["t_start"], d["track"]))
    inf = 1e9
    cost = np.full((len(ends), len(starts)), inf, dtype=float)
    for i, a in enumerate(ends):
        for j, b in enumerate(starts):
            if forbid is not None and forbid(int(a["track"]), int(b["track"])):
                continue
            c = _legacy_gate_cost(a, b, **gate_kw)
            if c is not None:
                cost[i, j] = c
    # ... the rest of the function is unchanged
```

In `legacy_link_events`, change the signature to `legacy_link_events(work, cfg, fps, excluded=None, *, reversal_sides: ReversalSides | None = None)`. Just before `matches = _legacy_matches(`, add:

```python
    forbid = None
    if reversal_sides is not None and reversal_sides.cuts:
        tags = reversal_sides.table_tags(work, excluded)

        def forbid(a: int, b: int) -> bool:
            return side_conflict(tags[a] | tags[b]) is not None
```

Then pass `forbid=forbid` to `_legacy_matches(...)`.

In `score_candidates`, add `reversal_sides: ReversalSides | None = None` after `excluded=None`, and document it in the docstring: "``reversal_sides``: no candidate joins two sides of a reversal cut (reversal spec 3.5)." After `spans = _split_spans(descs) if lc.keep_splits else {}`, add:

```python
    tags = (
        {t: reversal_sides.tags(t, d.lineage) for t, d in descs.items()}
        if reversal_sides is not None and reversal_sides.cuts
        else None
    )
```

In the pair loop, right after the `keep_splits` / `_resplit` check, add:

```python
            if tags is not None and side_conflict(tags[di.track] | tags[dj.track]) is not None:
                continue  # the two sides of a reversal cut are two vehicles (reversal spec 3.5)
```

In `run_link_stage`, add the keyword `reversal_sides: ReversalSides | None = None`. In the legacy branch, call `legacy_link_events(work, cfg, fps, excluded, reversal_sides=reversal_sides)`. In the scored branch, pass `reversal_sides=reversal_sides` to `score_candidates`. (Task 8 adds the chain check to both branches.)

- [ ] **Step 4: Pass the sides from `_Stages._link`**

```python
    def _link(self, work, occluded):
        cfg = self.cfg
        if not cfg.link.enabled or work.empty:
            return work, {}, set(), set(), {}
        band = Band.of(cfg.link)
        sides = self._sides()
        tags_before = sides.table_tags(work, self.excluded) if sides is not None else None
        res = run_link_stage(
            work,
            cfg,
            self.fps,
            appearance=self.link_app,
            context=self.link_ctx,
            frame_size=self.frame_size,
            occluded=occluded,
            route=lambda evs: self._route(evs, band, "link"),
            excluded=self.excluded,
            reversal_sides=sides,
        )
        work, rep_of = merge_chains(work, res.accepted)
        if sides is not None:
            sides.absorb(rep_of, tags_before)
        # ... the protected / pending code below is unchanged
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_link.py tests/refine/test_link_guards.py tests/refine/test_link_legacy.py tests/refine/test_link_scoring.py tests/refine/test_link_passes.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/dnt/refine tests/refine/test_reversal_link.py
git add src/dnt/refine/link.py src/dnt/refine/refiner.py tests/refine/test_reversal_link.py
git commit -m "feat(refine): link never pairs the two sides of a reversal cut

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Link never chains the two sides together

**Files:**
- Modify: `src/dnt/refine/link.py` (`run_link_stage`, both branches)
- Test: `tests/refine/test_reversal_link_chains.py`

**Interfaces:**
- Consumes: `resolve_reversal_chains` (Task 5), which this task adds to `link.py`'s `from .reversal import ...` line; the `reversal_sides` keyword (Task 7).
- Produces: `LinkStageResult.skipped`, which now also lists the LINK events the chain check dropped.

- [ ] **Step 1: Write the failing tests**

```python
from dnt.refine.config import RefineConfig
from dnt.refine.events import Event, EventKind
from dnt.refine.link import resolve_chains, run_link_stage, score_candidates
from dnt.refine.primitives import occlusion_flags
from dnt.refine.reversal import ReversalSides, resolve_reversal_chains
from dnt.refine.verify import Band, route_without_vlm

from ._reversal import FPS, link_event, rows, work_of


def _cfg(mode="scored"):
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.mode = mode
    return cfg


def _link(work, sides=None, cfg=None):
    cfg = cfg or _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    band = Band.of(cfg.link)
    return run_link_stage(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                          occluded=occ, route=lambda e: route_without_vlm(e, band),
                          reversal_sides=sides)


def _connected(pairs, a, b):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(j)] = find(i)
    return find(a) == find(b)


def _chain_work():
    # A = raw 1 frames 0-29, X = raw 2 frames 28-31, B = raw 1 frames 30-59 (review 21:16)
    return work_of(rows(1, range(0, 30)), rows(2, range(28, 32)), rows(3, range(30, 60)),
                   raw={3: 1})


def test_without_the_check_the_overlap_chain_merges_both_sides():
    res = _link(_chain_work())
    assert sorted(res.accepted) == [(1, 2), (2, 3)]


def test_the_chain_check_drops_one_link_of_the_overlap_chain():
    res = _link(_chain_work(), ReversalSides(cuts={1: [30]}))
    applied = [e for e in res.events if e.applied]
    dropped = [e for e in res.events if e.signals.get("skipped_reason") == "reversal"]
    assert len(applied) == 1 and len(dropped) == 1
    assert dropped[0].signals["conflicts_with"] == {"raw": 1, "sides": [0, 1], "tracks": [1, 3]}
    assert dropped[0] in res.skipped
    assert not _connected(res.accepted, 1, 3)


def test_the_path_follows_the_links_not_the_first_frames():
    # review 21:42: A 0-28 raw 1, X 30-31 raw 10, Y 29-32 raw 3, B 33-60 raw 10; cut raw 10 at 32
    work = work_of(rows(1, range(0, 29), lambda f: 100.0 + 3.0 * f),
                   rows(2, range(30, 32), lambda f: 100.0 + 3.0 * f),
                   rows(3, range(29, 33), lambda f: 100.0 + 3.0 * f),
                   rows(4, range(33, 61), lambda f: 100.0 + 3.0 * f), raw={2: 10, 4: 10})
    cfg = _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, descs = score_candidates(work, cfg, FPS, appearance=None, context=None,
                                    frame_size=None, occluded=occ)
    assert {(1, 2), (2, 3), (3, 4)} <= {(c.i, c.j) for c in cands}
    evs = [link_event(i, j, s, descs[i].lineage, descs[j].lineage)
           for (i, j), s in (((1, 2), 0.9), ((2, 3), 0.8), ((3, 4), 0.7))]
    _, skipped = resolve_chains(evs, descs, cfg.link.overlap_frames)
    assert skipped == []  # the frame-overlap check sees no problem
    sides = ReversalSides(cuts={10: [32]})
    tags = {t: sides.tags(t, d.lineage) for t, d in descs.items()}
    assert tags == {1: set(), 2: {(10, 0)}, 3: set(), 4: {(10, 1)}}
    (dropped,) = resolve_reversal_chains(evs, tags)
    assert dropped.tracks == [3, 4]  # the weakest on the path 2 -> 3 -> 4
    assert dropped.signals["conflicts_with"] == {"raw": 10, "sides": [0, 1], "tracks": [2, 4]}
    assert evs[0].applied  # A -> X is not on the path


def test_legacy_mode_runs_the_chain_check_too(monkeypatch):
    work = _chain_work()

    def fake_legacy(work, cfg, fps, excluded=None, *, reversal_sides=None):
        def lin(t):
            g = work[work["track"] == t]
            return [[int(g["raw_id"].iloc[0]), int(g["frame"].min()), int(g["frame"].max())]]

        return [
            Event.propose(stage="link", kind=EventKind.LINK, tracks=[a, b],
                          lineage=[lin(a), lin(b)], frames=(0, 1),
                          params={"gate": "legacy", "gap": [a, b]}, algo_score=1.0)
            for a, b in ((1, 2), (2, 3))
        ]

    monkeypatch.setattr("dnt.refine.link.legacy_link_events", fake_legacy)
    res = _link(work, ReversalSides(cuts={1: [30]}), cfg=_cfg("legacy"))
    assert len(res.accepted) == 1 and not _connected(res.accepted, 1, 3)
    assert [e.signals["skipped_reason"] for e in res.skipped] == ["reversal"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_link_chains.py -v`
Expected: `test_without_the_check…` and `test_the_path_follows…` PASS. The other two FAIL, because nothing drops a chain link yet.

- [ ] **Step 3: Implement in `run_link_stage`**

Replace the legacy branch with:

```python
    if cfg.link.mode == "legacy":
        evs = legacy_link_events(work, cfg, fps, excluded, reversal_sides=reversal_sides)
        route(evs)
        acc = [e for e in evs if e.decision in ACCEPTED]
        for e in acc:
            e.applied = True
        dropped: list[Event] = []
        if reversal_sides is not None and reversal_sides.cuts:
            dropped = resolve_reversal_chains(acc, reversal_sides.table_tags(work, excluded))
        gone = {id(e) for e in dropped}
        pairs = [(int(e.tracks[0]), int(e.tracks[1])) for e in acc if id(e) not in gone]
        return LinkStageResult(evs, pairs, set(), dropped)
```

At the end of the scored branch, replace the code from `pairs, skipped = resolve_chains(...)` to the `return` with:

```python
    pairs, skipped = resolve_chains(acc, descs, cfg.link.overlap_frames)
    skipped_ids = {id(e) for e in skipped}
    for e in acc:
        e.applied = id(e) not in skipped_ids
    if reversal_sides is not None and reversal_sides.cuts:
        active = [e for e in acc if e.applied]
        tags_of = {t: reversal_sides.tags(t, d.lineage) for t, d in descs.items()}
        dropped = resolve_reversal_chains(active, tags_of)
        if dropped:
            gone = {id(e) for e in dropped}
            pairs = [(int(e.tracks[0]), int(e.tracks[1])) for e in active if id(e) not in gone]
            skipped = [*skipped, *dropped]
    pending = {t for e in events if e.decision is Decision.HUMAN_PENDING for t in e.tracks}
    return LinkStageResult(events, pairs, pending, skipped)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_link_chains.py tests/refine/test_reversal_link.py tests/refine/test_link_passes.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/dnt/refine/link.py tests/refine/test_reversal_link_chains.py
git add src/dnt/refine/link.py tests/refine/test_reversal_link_chains.py
git commit -m "feat(refine): drop the weakest link that would chain the two sides of a cut

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Dedup never merges the two sides, and sides survive a merge

**Files:**
- Modify: `src/dnt/refine/dedup.py` (`apply_merges`)
- Modify: `src/dnt/refine/refiner.py` (`_Stages._dedup`)
- Test: `tests/refine/test_reversal_dedup.py`

**Interfaces:**
- Consumes: `ReversalSides`, `side_conflict` (Task 5); `_Stages._sides()` (Task 6).
- Produces: `apply_merges(work, events, cfg, *, reversal_sides=None)`. After it applies merges, `_Stages._dedup` calls `sides.absorb(out.absorbed, tags_before)`.

- [ ] **Step 1: Write the failing tests**

```python
from dnt.refine.apply import lineage_of_rows
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.link import run_link_stage
from dnt.refine.primitives import occlusion_flags
from dnt.refine.refiner import _Stages
from dnt.refine.reversal import ReversalSides
from dnt.refine.verify import Band, decide, route_without_vlm

from ._reversal import FPS, rows, work_of


def _merge(work, a, b, score):
    ga, gb = work[work["track"] == a], work[work["track"] == b]
    lo, hi = max(ga["frame"].min(), gb["frame"].min()), min(ga["frame"].max(), gb["frame"].max())
    ev = Event.propose(stage="dedup", kind=EventKind.MERGE, tracks=[a, b],
                       lineage=[lineage_of_rows(ga), lineage_of_rows(gb)], frames=(lo, hi),
                       params={"span": [int(lo), int(hi)]}, algo_score=score)
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    return ev


def _component_work():
    # A raw 1 (frames 0-10, then even to 28), X raw 2 (odd 11-49), B raw 1 (even 30-48, then 50-59)
    return work_of(rows(1, [*range(0, 11), *range(12, 30, 2)]), rows(2, range(11, 50, 2)),
                   rows(3, [*range(30, 50, 2), *range(50, 60)]), raw={3: 1})


def test_without_sides_dedup_joins_both_sides_through_a_component():
    work = _component_work()
    out = apply_merges(work, [_merge(work, 1, 2, 0.9), _merge(work, 2, 3, 0.8)],
                       RefineConfig.defaults("vehicle"))
    assert sorted(out.work["track"].unique()) == [1]


def test_dedup_refuses_the_join_that_would_put_both_sides_in_one_component():
    work = _component_work()
    evs = [_merge(work, 1, 2, 0.9), _merge(work, 2, 3, 0.8)]
    out = apply_merges(work, evs, RefineConfig.defaults("vehicle"),
                       reversal_sides=ReversalSides(cuts={1: [30]}))
    assert sorted(out.work["track"].unique()) == [1, 3]
    assert evs[0].applied and not evs[1].applied
    assert evs[1].signals["skipped_reason"] == "conflict"
    assert evs[1].signals["conflicts_with"] == {"tracks": [1, 3], "why": "reversal",
                                                "proposal_key": None}
    assert out.counts["conflict"] == 1


def _replacement_work():
    # review 21:42: A raw 1 frames 0-29 (0.8), B raw 1 frames 30-59 (0.9), a 0.95 copy of A
    return work_of(rows(1, range(0, 30), score=0.8), rows(2, range(30, 60), score=0.9),
                   rows(3, range(0, 30), score=0.95), raw={2: 1})


def _no_heading_cfg():
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.normal_heading_check = False  # only the cut protection may refuse the join
    return cfg


def _link(work, cfg, sides, excluded):
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    band = Band.of(cfg.link)
    return run_link_stage(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                          occluded=occ, route=lambda e: route_without_vlm(e, band),
                          excluded=excluded, reversal_sides=sides)


def test_a_replaced_side_is_relinked_without_carried_tags():
    cfg, work = _no_heading_cfg(), _replacement_work()
    evs = propose_merges(work, cfg, FPS, None)
    route_without_vlm(evs, Band.of(cfg.dedup))
    out = apply_merges(work, evs, cfg)
    assert set(out.work.loc[out.work["track"] == 1, "raw_id"]) == {3}  # all of A's rows lost
    (ev,) = _link(out.work, cfg, None, out.excluded).events
    assert ev.tracks == [1, 2] and ev.applied


def test_a_replaced_side_keeps_its_tag_and_is_not_relinked():
    cfg, work = _no_heading_cfg(), _replacement_work()
    sides = ReversalSides(cuts={1: [30]})
    evs = propose_merges(work, cfg, FPS, None)
    route_without_vlm(evs, Band.of(cfg.dedup))
    tags_before = sides.table_tags(work)
    out = apply_merges(work, evs, cfg, reversal_sides=sides)
    sides.absorb(out.absorbed, tags_before)
    assert (1, 0) in sides.carried[1]
    assert _link(out.work, cfg, sides, out.excluded).events == []


def test_the_pipeline_keeps_a_replaced_side_apart():
    cfg = _no_heading_cfg()
    cfg.switch.enabled = False
    cfg.screen.enabled = False  # screen could drop the copy before dedup sees it
    raw = work_of(rows(1, range(0, 30), score=0.8) + rows(1, range(30, 60), score=0.9),
                  rows(3, range(0, 30), score=0.95))
    stages = _Stages(cfg, FPS, None, None, None, None, {})
    out, events = stages.run(raw)
    (cut,) = [e for e in events if e.stage == "reversal"]
    assert cut.params["cut_frame"] == 30
    assert [e.tracks for e in events if e.stage == "dedup" and e.applied] == [[1, 3]]
    assert not [e for e in events if e.kind is EventKind.LINK and e.applied]
    assert out["track"].nunique() == 2
    assert (1, 0) in stages.reversal_sides.carried[1]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_dedup.py -v`
Expected: the two "without" tests PASS, because they pin today's behaviour. The others FAIL with `TypeError: apply_merges() got an unexpected keyword argument 'reversal_sides'`.

- [ ] **Step 3: Implement in `dedup.py`**

Add the import `from .reversal import ReversalSides, side_conflict`. Change the signature to:

```python
def apply_merges(
    work: pd.DataFrame,
    events: list[Event],
    cfg: RefineConfig,
    *,
    reversal_sides: ReversalSides | None = None,
) -> MergeOutcome:
```

Add a sentence to the docstring: "With ``reversal_sides``, a join whose components hold two sides of a reversal cut is a cannot-link pair too (reversal spec 3.5)."

After `tracks = describe(work)`, add:

```python
    sides = reversal_sides if reversal_sides is not None and reversal_sides.cuts else None
    side_tags = (
        {t: sides.tags(t, lineage_of_rows(tr.rows)) for t, tr in tracks.items()} if sides else {}
    )
```

In `blocker`, after the existing `for x, y in ...` loop and before `return None`, add:

```python
        if sides is not None:
            group = members[ra] + members[rb]
            conflict = side_conflict(set().union(*(side_tags[t] for t in group)))
            if conflict is not None:
                r, k1, k2 = conflict
                u = min(t for t in group if (r, k1) in side_tags[t])
                v = min(t for t in group if (r, k2) in side_tags[t])
                return u, v, "reversal", None
```

- [ ] **Step 4: Wire `_Stages._dedup`**

```python
    def _dedup(self, work):
        cfg = self.cfg
        if not cfg.dedup.enabled or work.empty:
            return work
        if self.vlm_runner is not None:
            log.info("dedup does not use the VLM; uncertain merges stay pending for review")
        evs = propose_merges(work, cfg, self.fps, self.link_app)
        self._route(evs, Band.of(cfg.dedup), "dedup", use_vlm=False)
        sides = self._sides()
        tags_before = sides.table_tags(work) if sides is not None else None
        out = apply_merges(work, evs, cfg, reversal_sides=sides)
        if sides is not None:
            sides.absorb(out.absorbed, tags_before)
        self.merged_reps, self.merge_pending = out.merged_reps, out.pending_endpoints
        self.excluded = out.excluded
        self.merge_counts = out.counts
        self._absorb(out.absorbed)
        return out.work
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_dedup.py tests/refine/test_dedup_apply.py tests/refine/test_dedup_stage.py tests/refine/test_dedup.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/dnt/refine tests/refine/test_reversal_dedup.py
git add src/dnt/refine/dedup.py src/dnt/refine/refiner.py tests/refine/test_reversal_dedup.py
git commit -m "feat(refine): dedup never joins the sides of a cut, and merges carry side tags

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Disabled-stage regression, existing tests, docs

**Files:**
- Test: `tests/refine/test_reversal_baseline.py`
- Modify: existing tests under `tests/refine/` that fail because vehicle output now changes (Step 3)
- Create: `docs/api/refine/reversal.md`
- Modify: `mkdocs.yml`, `docs/api/refine/index.md`, `docs/changelog.md`, `src/dnt/refine/__init__.py` (docstring)

**Interfaces:**
- Consumes: Task 1's baseline files; everything above.

- [ ] **Step 1: Write the baseline test**

```python
import copy
import json
import shutil

from dnt.refine import RefineConfig, TrackRefiner
from dnt.refine.events import Ledger

from ._fixtures import DATA

BASE = DATA / "reversal_baseline"


def _strip_paths(obj):
    if isinstance(obj, dict):
        return {k: _strip_paths(v) for k, v in obj.items() if k not in ("path", "abs_path")}
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    return obj


def _header(h):
    h = copy.deepcopy(h)
    h.pop("dnt_version")
    h["config"].pop("reversal", None)
    h["inputs"] = _strip_paths(h["inputs"])
    return h


def test_the_disabled_stage_reproduces_the_pre_change_output(tmp_path):
    raw = tmp_path / "raw.csv"
    shutil.copy(BASE / "raw.csv", raw)
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    cfg.reversal.enabled = False
    refiner = TrackRefiner(cfg)
    refiner.refine(raw, tmp_path / "tracks.csv", fps=10, verbose=False)
    assert (tmp_path / "tracks.csv").read_bytes() == (BASE / "tracks.csv").read_bytes()
    new, old = Ledger.read(refiner.last_result.ledger_path), Ledger.read(BASE / "ledger.jsonl")
    assert [e.to_dict() for e in new.events] == [e.to_dict() for e in old.events]
    assert _header(new.header) == _header(old.header)
    assert json.dumps(new.header["config"]["reversal"])  # the new block is the only addition


def test_the_enabled_stage_splits_the_scene_vehicle(tmp_path):
    raw = tmp_path / "raw.csv"
    shutil.copy(BASE / "raw.csv", raw)
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    refiner = TrackRefiner(cfg)
    refiner.refine(raw, tmp_path / "tracks.csv", fps=10, verbose=False)
    (cut,) = [e for e in refiner.last_result.events if e.stage == "reversal"]
    assert cut.tracks == [1] and cut.params["cut_frame"] == 30 and cut.applied
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/refine/test_reversal_baseline.py -v`
Expected: PASS. If the header comparison fails on a field other than `config.reversal`, `dnt_version` or input paths, that is a regression to fix in code. Do not normalise it away.

- [ ] **Step 3: Run the whole refine suite and fix tests that rely on unsplit vehicle tracks**

Run: `.venv/bin/python -m pytest tests/refine -q`

For each failure:
1. Read the test. If its fixture is a vehicle track that reverses, but the test is about something else (link, dedup, screen, switch), add `cfg.reversal.enabled = False` right after the config is built, with the comment `# this test is about <x>, not reversal splitting`.
2. If the failure is in a stage that the reversal stage now feeds (for example, a link count changes because a reversal was split), the new behaviour is the intended one. Update the expected value, and add a comment naming the reversal split.
3. Any other failure is a bug. Fix the code.

List each test you changed in the commit message body.

Then run the default suite: `.venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 4: Docs**

Create `docs/api/refine/reversal.md`:

```markdown
# Reversal splitting

::: dnt.refine.reversal
```

In `mkdocs.yml`, add `- Reversal splitting: api/refine/reversal.md` after `- Events and Ledger: api/refine/events.md`.

In `docs/api/refine/index.md`, change "it splits ID switches" in the first sentence to "it splits vehicle tracks that reverse, splits ID switches". Add this section before `## Interleaved duplicate tracks`:

````markdown
## Vehicle reversals

A tracker sometimes carries one vehicle's ID over to another that comes back along the same
path, usually far down the road where boxes are small. Stage `reversal` (first, before switch)
splits a vehicle track wherever its direction reverses: the velocities over `window` seconds
before and after a row must both reach `min_speed` box heights per second and differ by at
least `min_angle` degrees. The cut goes where the velocity changes most, and each part keeps at
least `min_rows` rows. Every split is applied (`AUTO_ACCEPT`) and recorded as a `SPLIT` event
with stage `reversal`. Link and dedup never join the two sides of a cut again, also through a
third track; a link or merge they refuse for this is kept in the ledger with
`skipped_reason: reversal` (links) or `conflicts_with.why: reversal` (merges).

```yaml
reversal:
  enabled: true      # default for the vehicle target; false for person
  window: 2.0        # seconds before / after a row
  min_speed: 0.5     # box heights per second, each side (0.2 splits queued cars)
  min_angle: 135.0   # degrees between the two velocities
  min_rows: 5        # rows each part must keep
  min_side_rows: 3   # rows each side needs to measure a velocity
```

Turn it off with `reversal: {enabled: false}` to get the output of earlier versions.
````

In `docs/changelog.md`, under `## Unreleased`, add to `### New`:

```markdown
- Stage `reversal` in `dnt.refine` splits a vehicle track where its direction reverses (two
  vehicles that share one tracker ID). Link and dedup never join the two parts again, also
  through a third track or after a merge replaces one part's rows. See "Vehicle reversals" in
  the refine docs.
```

Under `### Changed`, add:

```markdown
- `dnt.refine` output for the vehicle target changes by default: tracks that reverse are split
  by the new `reversal` stage. Set `reversal: {enabled: false}` for the previous output.
```

In `src/dnt/refine/__init__.py`, change the module docstring's first line to:

```python
"""Track refinement: reversal and switch splitting, false-track screening, linking, gap filling.
```

- [ ] **Step 5: Lint, build the docs, commit**

Run: `.venv/bin/ruff check src tests tools && .venv/bin/mkdocs build --strict 2>&1 | tail -5`
Expected: ruff reports `All checks passed!` and mkdocs reports no errors. (If `--strict` fails on warnings that already failed before this change, compare with `git stash` and report them; don't fix unrelated pages.)

```bash
git add tests/refine docs mkdocs.yml src/dnt/refine/__init__.py
git commit -m "docs(refine): document the reversal stage; pin the disabled-stage baseline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage

| Spec | Task |
|---|---|
| §2 success criterion 1 (cut at the turn, clean and noisy) | 3 |
| §2 success criterion 2 (same hit runs as the script) | 4 |
| §2 success criterion 3 (queue, 90° turn, parked car) | 3 |
| §2 success criterion 4 (no join directly, by chain, or after dedup) | 7, 8, 9 |
| §2 success criterion 5 (disabled reproduces the old output) | 1, 10 |
| §2 success criterion 6 (stable key) | 6 |
| §3.1 order, progress count | 6 |
| §3.2 rule, steps 1–7, tie tolerance | 3 |
| §3.3 event fields, key, IDs | 6 |
| §3.4 applying latest first, `split_raw`, `reversal_cuts` | 6 |
| §3.5 screen hand-off | 6 |
| §3.5 side tags, `absorb` | 5, 7, 9 |
| §3.5 link pairs (scored and legacy) | 7 |
| §3.5 link chains (graph path) | 5, 8 |
| §3.5 dedup blocker | 9 |
| §5 config and validation | 2 |
| §6.4 existing tests | 10 |
| §7 docs, changelog, lint | 10 |
| §8 degenerate side, short tables | 3, 6 |
