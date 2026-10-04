# Interleaved-Duplicate Merge (`dedup` stage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `dedup` stage to `dnt.refine` that merges interleaved duplicate tracks (one person under alternating IDs) before link and fill, with a new `MERGE` event kind, ledger and review support.

**Architecture:** A new module `dnt/refine/dedup.py` proposes `MERGE` events from pairwise co-motion and observation-occupancy signals, and applies the accepted ones with a deterministic union-find that refuses to join tracks that are densely co-observed or explicitly rejected. `_Stages` runs it between screen and link, carries its representatives and pending endpoints through link to the orphan pass, composes an `absorbed` map for output IDs, and passes an `excluded` map so later lineage never reads a dropped observation.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest, ruff. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-10-03-track-dedup-design.md](../specs/2026-10-03-track-dedup-design.md) (rev. 4; read §3.2, §3.5 and §12 first). Parent spec: [2026-09-27-track-refinement-design.md](../specs/2026-09-27-track-refinement-design.md).

**Prototype:** the signals and the apply procedure were run on the three raw pedestrian files in `/mnt/e/videos/miami/dets` before this plan was written (results in spec §12). The code in Tasks 4 and 5 is derived from that prototype; the tests, the wiring and the review changes have **not** been run.

## Global Constraints

- Run Python with the project venv: `.venv/bin/python -m pytest ...`. Baseline before this change: `tests/refine` has **1149 passed**.
- `ruff check src tests tools` must stay clean; line length 100; rules `E,F,I,UP,B,SIM,RUF,D`. The per-file baseline in `pyproject.toml` may shrink but must not grow. New public functions and classes in `src/` need numpy-style docstrings.
- Imports are package-relative (`from .events import ...`). `dnt.refine` must not import `Labeler` (`tests/test_refine_independence.py`).
- Event IDs are `{stage}-r0-{seq:06d}`; the new stage name is exactly `"dedup"` and the kind is exactly `"MERGE"`.
- The work table keeps its **index labels** through dedup: later stages align an `occluded` Series to the original index. Never `reset_index` the table in `dedup.py` or `apply.py`.
- `dedup` never uses the VLM, even when a backend is configured. Uncertain merges become `HUMAN_PENDING`.
- Starting config defaults (spec §6, calibrated): `enabled=True`, `accept_above=0.75`, `reject_below=0.40`, `min_overlap_seconds=1.0`, `min_observed=8`, `comotion_lo=0.25`, `comotion_hi=0.50`, `cooccur_lo=0.10`, `cooccur_hi=0.50`, `conflict_min_shared=3`, `appearance_floor=0.60`, `app_lo=0.40`, `app_hi=0.70`.
- Work on a branch: `git switch -c feature/refine-dedup` before Task 1.
- Commit trailer: end every commit message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Failure modes the spec implies but a straight reading of the tasks would not test. Each has a test in the task named in brackets.

1. **Vehicle target.** Cars (class 2) and trucks (class 7) are one class group, so interleaved tracks of those classes merge; a car and a motorcycle (class 3) do not. [Task 4, `test_vehicle_class_groups_merge_and_other_classes_do_not`]
2. **Empty and single-track tables.** `propose_merges` and `apply_merges` return nothing and leave the table alone. [Tasks 4 and 5]
3. **Realistic messy tracks.** On `random_tracks` (gaps, restarts, mixed classes) no `(track, frame)` is duplicated afterwards, and every input observation is either in the output or listed in a `dropped` run, never both. [Task 5, `test_random_tracks_keep_every_observation_accounted_for`]
4. **Appearance provider with missing tracks.** When the provider has no embeddings for one track, `appearance` is `None` and the score is motion-only; it must not raise. [Task 4]
5. **Stage disabled.** `dedup.enabled: false` gives no `dedup` events, an empty `absorbed` map and zero `summary["dedup"]` counts, so existing configs behave as before. [Task 7]

## File Structure

| File | Responsibility |
|---|---|
| `src/dnt/refine/config.py` (modify) | `DedupConfig` and its validation. |
| `src/dnt/refine/events.py` (modify) | `EventKind.MERGE`, its defining params, `Event.propose(key_lineage=...)`. |
| `src/dnt/refine/apply.py` (modify) | `lineage_of_rows(excluded=...)`, `rows_to_drop`, `merge_tracks`. |
| `src/dnt/refine/dedup.py` (create) | Signals, `propose_merges`, `apply_merges`, `MergeOutcome`. No stage-loop logic. |
| `src/dnt/refine/link.py`, `screen.py`, `refiner.py` (modify) | Thread `excluded` into every lineage call after dedup. |
| `src/dnt/refine/refiner.py` (modify) | `_Stages._dedup`, carried sets, `absorbed`, header and summary, progress total. |
| `src/dnt/refine/evidence.py` (modify) | `MERGE` evidence plan. |
| `src/dnt/refine/review.py` (modify) | `absorbed` lookup, read-only "Skipped merges" section, page-selection rule. |
| `tests/refine/test_config.py`, `test_events.py`, `test_apply.py`, `test_evidence.py`, `test_review.py` (modify) | Tests for the matching modules. |
| `tests/refine/test_dedup.py`, `test_dedup_apply.py`, `test_dedup_lineage.py`, `test_dedup_stage.py`, `test_dedup_real.py` (create) | Tests for the new module and the wiring. |
| `docs/changelog.md`, `docs/api/refine/index.md`, `docs/api/refine/dedup.md`, `mkdocs.yml` (modify/create) | Documentation. |

---

### Task 1: `DedupConfig`

**Files:**
- Modify: `src/dnt/refine/config.py` (add the dataclass before `LinkConfig`, a field on `RefineConfig`, checks in `validate`)
- Test: `tests/refine/test_config.py`

**Interfaces:**
- Produces: `DedupConfig` (fields in Global Constraints) and `RefineConfig.dedup: DedupConfig`.

- [ ] **Step 0: Branch**

```bash
git switch -c feature/refine-dedup
```

- [ ] **Step 1: Write the failing tests** (append to `tests/refine/test_config.py`)

```python
def test_dedup_defaults_and_yaml_round_trip(tmp_path):
    cfg = RefineConfig.defaults()
    d = cfg.dedup
    assert d.enabled is True
    assert (d.accept_above, d.reject_below) == (0.75, 0.40)
    assert (d.min_overlap_seconds, d.min_observed) == (1.0, 8)
    assert (d.comotion_lo, d.comotion_hi) == (0.25, 0.50)
    assert (d.cooccur_lo, d.cooccur_hi, d.conflict_min_shared) == (0.10, 0.50, 3)
    assert (d.appearance_floor, d.app_lo, d.app_hi) == (0.60, 0.40, 0.70)
    cfg.to_yaml(tmp_path / "c.yaml")
    assert RefineConfig.from_yaml(tmp_path / "c.yaml").dedup == d


def test_dedup_overlay_and_unknown_key():
    cfg = RefineConfig.from_dict({"dedup": {"enabled": False, "min_observed": 12}})
    assert cfg.dedup.enabled is False and cfg.dedup.min_observed == 12
    assert cfg.dedup.accept_above == 0.75  # untouched keys keep their defaults
    with pytest.raises(ValueError, match="unknown config key 'dedup.nope'"):
        RefineConfig.from_dict({"dedup": {"nope": 1}})


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"accept_above": 0.3, "reject_below": 0.4}, "dedup: need 0 <= reject_below"),
        ({"cooccur_lo": 0.6, "cooccur_hi": 0.5}, "dedup.cooccur_lo must be below"),
        ({"comotion_lo": 0.5, "comotion_hi": 0.5}, "dedup.comotion_lo must be below"),
        ({"app_lo": 0.8, "app_hi": 0.7}, "dedup.app_lo must be below"),
        ({"conflict_min_shared": 0}, "dedup.conflict_min_shared"),
        ({"min_observed": 0}, "dedup.min_observed"),
        ({"min_observed": True}, "dedup.min_observed"),
        ({"min_overlap_seconds": 0}, "dedup.min_overlap_seconds"),
        ({"appearance_floor": 1.5}, "dedup.appearance_floor"),
    ],
)
def test_dedup_validation(patch, message):
    with pytest.raises(ValueError, match=message):
        RefineConfig.from_dict({"dedup": patch})
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py -q -k dedup`
Expected: FAIL (`AttributeError: ... has no attribute 'dedup'`, and `unknown config key 'dedup'`).

- [ ] **Step 3: Implement**

In `src/dnt/refine/config.py`, add before `class LinkConfig`:

```python
@dataclass(kw_only=True)
class DedupConfig:
    """Stage ``dedup`` settings: merging interleaved duplicate tracks (dedup spec 6)."""

    enabled: bool = True
    accept_above: float = 0.75
    reject_below: float = 0.40
    min_overlap_seconds: float = 1.0
    min_observed: int = 8
    comotion_lo: float = 0.25
    comotion_hi: float = 0.50
    cooccur_lo: float = 0.10
    cooccur_hi: float = 0.50
    conflict_min_shared: int = 3
    appearance_floor: float = 0.60
    app_lo: float = 0.40
    app_hi: float = 0.70
```

Add a module-level helper next to `_url_problems`:

```python
def _dedup_problems(dd: DedupConfig) -> list[str]:
    """Return the rules of the dedup block that its values break."""

    def num(x):
        return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)

    def whole(x):
        return isinstance(x, int) and not isinstance(x, bool)

    p: list[str] = []
    if not (whole(dd.min_observed) and dd.min_observed >= 1):
        p.append("dedup.min_observed must be an integer >= 1")
    if not (whole(dd.conflict_min_shared) and dd.conflict_min_shared >= 1):
        p.append("dedup.conflict_min_shared must be an integer >= 1")
    if not (num(dd.min_overlap_seconds) and dd.min_overlap_seconds > 0):
        p.append("dedup.min_overlap_seconds must be a number > 0")
    for lo, hi in (
        ("comotion_lo", "comotion_hi"),
        ("cooccur_lo", "cooccur_hi"),
        ("app_lo", "app_hi"),
    ):
        a, b = getattr(dd, lo), getattr(dd, hi)
        if not (num(a) and num(b) and a < b):
            p.append(f"dedup.{lo} must be below dedup.{hi}")
    if not (num(dd.appearance_floor) and 0.0 <= dd.appearance_floor <= 1.0):
        p.append("dedup.appearance_floor must be a number in [0, 1]")
    return p
```

In `RefineConfig`, add the field after `screen`:

```python
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
```

In `validate`, extend the band loop and call the helper right after it:

```python
        for name in ("switch", "screen", "dedup", "link", "orphan"):
            s = getattr(self, name)
            if not 0.0 <= s.reject_below < s.accept_above <= 1.0:
                p.append(f"{name}: need 0 <= reject_below < accept_above <= 1")
        p.extend(_dedup_problems(self.dedup))
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py -q`
Expected: all PASS (the existing config tests included).

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/config.py tests/refine/test_config.py
git commit -m "feat(refine): DedupConfig for the dedup stage

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `MERGE` event kind and `key_lineage`

**Files:**
- Modify: `src/dnt/refine/events.py` (`EventKind`, `DEFINING_PARAMS`, `Event.propose`)
- Test: `tests/refine/test_events.py`

**Interfaces:**
- Produces: `EventKind.MERGE`; `DEFINING_PARAMS[EventKind.MERGE] == ("span",)`; `Event.propose(..., key_lineage=None, round=0)` where, when `key_lineage` is given, `proposal_key` is computed from it and `event.lineage` keeps the lineage passed.

- [ ] **Step 1: Write the failing tests** (append to `tests/refine/test_events.py`)

```python
from dnt.refine.events import DEFINING_PARAMS


def _merge(tracks, lineage, span=(652, 834), key_lineage=None):
    return Event.propose(stage="dedup", kind=EventKind.MERGE, tracks=tracks, lineage=lineage,
                         key_lineage=key_lineage, frames=span, params={"span": list(span)},
                         algo_score=0.9, signals={"shared": 7})


def test_merge_key_uses_key_lineage_and_ignores_track_order():
    a, b = [[122, 652, 834]], [[125, 652, 834]]

    def canon(lin):
        return sorted(lin, key=lambda spans: tuple(spans[0]))

    e1 = _merge([3, 9], [a, b], key_lineage=canon([a, b]))
    e2 = _merge([9, 3], [b, a], key_lineage=canon([b, a]))
    assert e1.proposal_key == e2.proposal_key
    assert e1.lineage == [a, b] and e2.lineage == [b, a]  # stored in tracks order, never sorted
    assert DEFINING_PARAMS[EventKind.MERGE] == ("span",)
    assert _merge([3, 9], [a, b], span=(652, 835), key_lineage=canon([a, b])).proposal_key \
        != e1.proposal_key


def test_without_key_lineage_the_key_follows_the_lineage_order():
    a, b = [[122, 652, 834]], [[125, 652, 834]]
    assert _merge([3, 9], [a, b]).proposal_key != _merge([9, 3], [b, a]).proposal_key


def test_merge_round_trips_through_the_ledger(tmp_path):
    ev = _merge([3, 9], [[[122, 652, 834]], [[125, 652, 834]]])
    ev.id = "dedup-r0-000001"
    Ledger({"format": "x"}, [ev]).write(tmp_path / "l.jsonl")
    back = Ledger.read(tmp_path / "l.jsonl").events[0]
    assert back.kind is EventKind.MERGE and back.to_dict() == ev.to_dict()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_events.py -q -k merge`
Expected: FAIL (`AttributeError: MERGE` / `unexpected keyword argument 'key_lineage'`).

- [ ] **Step 3: Implement** in `src/dnt/refine/events.py`

Add the member and the defining params:

```python
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    LINK = "LINK"
```

```python
    EventKind.SPLIT: ("cut_frame",),
    EventKind.MERGE: ("span",),
    EventKind.DROP: ("reason", "spans", "of"),
```

In `Event.propose` add the keyword (before `round`) and use it:

```python
        params: dict,
        algo_score: float,
        signals: dict | None = None,
        key_lineage=None,
        round: int = 0,
    ) -> Event:
        """Create an undecided proposal and compute its key.

        ``key_lineage``, when given, replaces ``lineage`` in the key only, so an event can store
        its lineage in ``tracks`` order and still have a key that does not depend on that order.
        """
        params = clean_json(dict(params))
        lineage = clean_json(lineage)
        key_source = lineage if key_lineage is None else clean_json(key_lineage)
        return cls(
            id="",
            proposal_key=proposal_key(stage, kind, key_source, params),
```

(Leave the rest of the constructor call as it is.)

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine/test_events.py tests/refine/test_apply.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/events.py tests/refine/test_events.py
git commit -m "feat(refine): MERGE event kind and Event.propose(key_lineage)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `lineage_of_rows(excluded)`, `rows_to_drop`, `merge_tracks`

**Files:**
- Modify: `src/dnt/refine/apply.py` (replace `lineage_of_rows`; add two functions)
- Test: `tests/refine/test_apply.py`

**Interfaces:**
- Produces:
  - `lineage_of_rows(rows, excluded: dict[int, set[int]] | None = None) -> list[list[int]]`
  - `rows_to_drop(rows: pd.DataFrame) -> pd.Index`: index labels of the rows that lose on a frame where several rows exist (best row: higher `score`, then smaller `raw_id`, then smaller `track`).
  - `merge_tracks(work, rep_of: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]`: `rep_of` maps every member track to its representative (representatives map to themselves); returns the merged table (index labels kept, sorted by track then frame) and the dropped rows.

- [ ] **Step 1: Write the failing tests** (append to `tests/refine/test_apply.py`)

```python
def test_lineage_of_rows_splits_spans_at_excluded_frames():
    rows = _work(box_rows(122, [f for f in range(10, 31) if f != 20], 0.0, 0.0))
    assert A.lineage_of_rows(rows) == [[122, 10, 30]]
    assert A.lineage_of_rows(rows, {122: {20}}) == [[122, 10, 19], [122, 21, 30]]
    rows2 = _work(box_rows(122, [f for f in range(10, 31) if f not in (20, 21)], 0.0, 0.0))
    assert A.lineage_of_rows(rows2, {122: {20, 21}}) == [[122, 10, 19], [122, 22, 30]]


def test_excluded_frames_outside_or_at_the_ends_of_a_span_change_nothing():
    rows = _work(box_rows(122, range(10, 31), 0.0, 0.0))
    assert A.lineage_of_rows(rows, {122: {5, 10, 30, 31}}) == [[122, 10, 30]]
    assert A.lineage_of_rows(rows, {999: {20}}) == [[122, 10, 30]]


def test_lineage_of_rows_orders_split_spans_across_raw_ids_by_first_frame():
    rows = _work(box_rows(7, range(0, 6), 0.0, 0.0),
                 box_rows(122, [f for f in range(10, 31) if f != 20], 0.0, 0.0))
    assert A.lineage_of_rows(rows, {122: {20}}) == [[7, 0, 5], [122, 10, 19], [122, 21, 30]]
    assert A.lineage_of_rows(rows.iloc[0:0], {122: {20}}) == []


def test_rows_to_drop_keeps_the_best_row_per_frame():
    w = _work(box_rows(1, range(3), 0.0, 0.0, score=0.5),
              box_rows(2, range(3), 0.0, 0.0, score=0.9),
              box_rows(3, [5], 0.0, 0.0))
    assert sorted(w.loc[A.rows_to_drop(w), "track"].tolist()) == [1, 1, 1]


def test_rows_to_drop_breaks_a_score_tie_with_the_smaller_raw_id():
    w = _work(box_rows(5, range(2), 0.0, 0.0, score=0.7), box_rows(3, range(2), 0.0, 0.0, score=0.7))
    assert set(w.loc[A.rows_to_drop(w), "track"]) == {5}


def test_merge_tracks_relabels_drops_losers_and_keeps_the_index():
    w = _work(box_rows(1, range(0, 10), 0.0, 0.0, score=0.9),
              box_rows(2, range(5, 15), 0.0, 0.0, score=0.8),
              box_rows(3, range(0, 4), 0.0, 0.0))
    out, dropped = A.merge_tracks(w, {1: 1, 2: 1})
    assert sorted(out["track"].unique()) == [1, 3]
    assert out[out["track"] == 1]["frame"].tolist() == list(range(15))
    assert dropped["raw_id"].unique().tolist() == [2]
    assert sorted(dropped["frame"].tolist()) == list(range(5, 10))
    assert set(out.index) | set(dropped.index) == set(w.index)
    assert set(out.index) & set(dropped.index) == set()
    same, none = A.merge_tracks(w, {})
    assert same is w and none.empty
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_apply.py -q -k "excluded or rows_to_drop or merge_tracks"`
Expected: FAIL (`unexpected argument` / `no attribute 'rows_to_drop'`).

- [ ] **Step 3: Implement** in `src/dnt/refine/apply.py`

Replace `lineage_of_rows`:

```python
def lineage_of_rows(
    rows: pd.DataFrame, excluded: dict[int, set[int]] | None = None
) -> list[list[int]]:
    """Return ``[[raw_id, first_frame, last_frame], ...]`` for rows, ordered by first frame.

    ``excluded`` maps a raw ID to frames that stage ``dedup`` dropped (dedup spec 3.5). A raw
    ID's span is split at each excluded frame that lies strictly inside it, so a consumer that
    reads the raw observations inside the spans never sees a dropped row.
    """
    if rows.empty:
        return []
    g = rows.groupby("raw_id")["frame"].agg(["min", "max"]).reset_index()
    spans: list[list[int]] = []
    for raw, first, last in g[["raw_id", "min", "max"]].to_numpy():
        raw, first, last = int(raw), int(first), int(last)
        start = first
        for f in sorted(f for f in (excluded or {}).get(raw, ()) if first < f < last):
            if f > start:
                spans.append([raw, start, f - 1])
            start = f + 1
        spans.append([raw, start, last])
    return sorted(spans, key=lambda s: (s[1], s[0]))
```

Add after `merge_chains`:

```python
def rows_to_drop(rows: pd.DataFrame) -> pd.Index:
    """Return the index labels of rows that lose on a frame where several rows exist.

    The best row of a frame has the higher ``score``; ties go to the smaller ``raw_id``, then
    to the smaller ``track``. The order is total, so the kept set does not depend on the order
    the rows or the merges arrive in (dedup spec 3.5).
    """
    order = rows.sort_values(
        ["frame", "score", "raw_id", "track"], ascending=[True, False, True, True], kind="stable"
    )
    return order.index[order.duplicated("frame", keep="first")]


def merge_tracks(
    work: pd.DataFrame, rep_of: dict[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel every member of a merge group to its representative and keep one row per frame.

    ``rep_of`` maps each member track to its representative (representatives map to
    themselves). Returns the merged table, with its index labels kept and sorted by track and
    frame, and the dropped rows.
    """
    if not rep_of:
        return work, work.iloc[0:0]
    new_track = work["track"].map(lambda t: rep_of.get(int(t), int(t)))
    members = work[work["track"].isin(rep_of)]
    drop: list = []
    for _, g in members.assign(_rep=new_track.loc[members.index]).groupby("_rep"):
        drop.extend(rows_to_drop(g).tolist())
    dropped = work.loc[drop]
    out = work.drop(index=drop).copy()
    out["track"] = new_track.loc[out.index].astype(int)
    return out.sort_values(["track", "frame"]), dropped
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine/test_apply.py -q`
Expected: PASS (existing lineage tests included, since `excluded=None` keeps the old output).

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/apply.py tests/refine/test_apply.py
git commit -m "feat(refine): lineage_of_rows(excluded), rows_to_drop and merge_tracks

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `dedup.py`: signals and `propose_merges`

**Files:**
- Create: `src/dnt/refine/dedup.py`
- Test: `tests/refine/test_dedup.py`

**Interfaces:**
- Consumes: `lineage_of_rows` (Task 3), `Event.propose(key_lineage=...)` and `EventKind.MERGE` (Task 2), `RefineConfig.dedup` (Task 1), `track_embeddings`, `majority_class`, `ramp`.
- Produces: `STAGE = "dedup"`, `propose_merges(work, cfg, fps, appearance=None) -> list[Event]` (undecided `MERGE` events in a deterministic order). Each event has `tracks=[first, second]` (earlier first observed frame, ties by track ID), `lineage` in that order, `frames=(lo, hi)`, `params={"span": [lo, hi]}`, and signals `shared`, `n_a`, `n_b`, `co_occupancy`, `comotion`, `appearance` (`n_a` belongs to `tracks[0]`). Also the helpers `describe`, `overlap`, `comotion`, `class_ok` and the `Overlap` and `Track` dataclasses, used by Task 5.

- [ ] **Step 1: Write the failing tests** (create `tests/refine/test_dedup.py`)

```python
import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import propose_merges
from dnt.refine.events import EventKind
from dnt.refine.features import ArrayAppearance

from ._fixtures import table

FPS = 10.0


def _rows(track, frames, *, dx=0.0, cls=0, score=0.9, shift=None):
    """Rows of a 30 x 60 box on the path x = 100 + 3 * frame, plus ``dx``; ``shift`` per frame."""
    return [
        [f, track, 100.0 + 3.0 * f + dx + (shift or {}).get(f, 0.0), 100.0, 30.0, 60.0,
         score, cls, -1, -1]
        for f in frames
    ]


def _work(*row_lists):
    return io.to_work(table(*row_lists)).work


def _interleaved(n=120):
    return _work(_rows(1, range(0, n, 2)), _rows(2, range(1, n, 2)))


def _expected(fa, fb):
    lo, hi = max(min(fa), min(fb)), min(max(fa), max(fb))
    a = {f for f in fa if lo <= f <= hi}
    b = {f for f in fb if lo <= f <= hi}
    return len(a & b), len(a), len(b)


def test_an_interleaved_pair_becomes_one_merge_event_with_ordered_lineage():
    (ev,) = propose_merges(_interleaved(), RefineConfig.defaults(), FPS)
    assert ev.stage == "dedup" and ev.kind is EventKind.MERGE and ev.tracks == [1, 2]
    assert ev.lineage == [[[1, 0, 118]], [[2, 1, 119]]]
    assert ev.frames == (1, 118) and ev.params == {"span": [1, 118]}
    assert ev.algo_score == pytest.approx(1.0)
    s = ev.signals
    assert (s["shared"], s["n_a"], s["n_b"]) == (0, 59, 59)
    assert s["co_occupancy"] == 0.0 and s["comotion"] == pytest.approx(1.0)
    assert s["appearance"] is None


def test_the_evening_counts_score_above_accept():
    shared = [0, 30, 60, 90, 120, 150, 182]
    rest = [f for f in range(183) if f not in shared]
    a_only = [rest[i] for i in np.linspace(0, 175, 69).astype(int)]
    b_only = [f for f in rest if f not in set(a_only)][:100]
    fa, fb = sorted(shared + a_only), sorted(shared + b_only)
    work = _work(_rows(1, fa, shift={f: 300.0 for f in shared}), _rows(2, fb))
    cfg = RefineConfig.defaults()
    (ev,) = propose_merges(work, cfg, FPS)
    s = ev.signals
    assert (s["shared"], s["n_a"], s["n_b"]) == (7, 76, 107)
    assert s["co_occupancy"] == pytest.approx(7 / 76)
    assert ev.algo_score >= cfg.dedup.accept_above


@pytest.mark.parametrize("n_shared", [0, 1, 5, 12])
def test_co_occupancy_is_shared_over_the_sparser_track(n_shared):
    fa = list(range(0, 80, 2))
    fb = sorted(set(range(1, 80, 2)) | set(range(2, 2 + 2 * n_shared, 2)))
    (ev,) = propose_merges(_work(_rows(1, fa), _rows(2, fb)), RefineConfig.defaults(), FPS)
    shared, n_a, n_b = _expected(fa, fb)
    s = ev.signals
    assert shared == n_shared and (s["shared"], s["n_a"], s["n_b"]) == (shared, n_a, n_b)
    assert s["co_occupancy"] == pytest.approx(shared / min(n_a, n_b))


@pytest.mark.parametrize("dx", [200.0, 10.0, 1.6], ids=["apart", "overlapping", "duplicate"])
def test_densely_co_observed_pairs_are_never_candidates(dx):
    work = _work(_rows(1, range(60)), _rows(2, range(60), dx=dx))
    assert propose_merges(work, RefineConfig.defaults(), FPS) == []


def test_the_class_overlap_and_observation_gates():
    cfg = RefineConfig.defaults()
    mixed = _work(_rows(1, range(0, 120, 2), cls=0), _rows(2, range(1, 120, 2), cls=2))
    assert propose_merges(mixed, cfg, FPS) == []
    short = _work(_rows(1, range(0, 10, 2)), _rows(2, range(1, 10, 2)))
    cfg.dedup.min_observed = 1  # isolate the span gate: the overlap is 8 frames, under 1 s
    assert propose_merges(short, cfg, FPS) == []
    cfg.dedup.min_overlap_seconds = 0.5
    assert len(propose_merges(short, cfg, FPS)) == 1
    few = _work(_rows(1, range(0, 120, 2)), _rows(2, [1, 21, 41, 61, 81]))
    cfg = RefineConfig.defaults()  # the overlap is 81 frames; track 2 has 5 rows in it, under 8
    assert propose_merges(few, cfg, FPS) == []
    cfg.dedup.min_observed = 5
    assert len(propose_merges(few, cfg, FPS)) == 1


def test_vehicle_class_groups_merge_and_other_classes_do_not():
    cfg = RefineConfig.defaults("vehicle")
    car_truck = _work(_rows(1, range(0, 120, 2), cls=2), _rows(2, range(1, 120, 2), cls=7))
    assert len(propose_merges(car_truck, cfg, FPS)) == 1
    car_moto = _work(_rows(1, range(0, 120, 2), cls=2), _rows(2, range(1, 120, 2), cls=3))
    assert propose_merges(car_moto, cfg, FPS) == []


def test_disabled_empty_and_single_track_tables_give_no_events():
    cfg = RefineConfig.defaults()
    work = _interleaved()
    cfg.dedup.enabled = False
    assert propose_merges(work, cfg, FPS) == []
    cfg.dedup.enabled = True
    assert propose_merges(work.iloc[0:0], cfg, FPS) == []
    assert propose_merges(_work(_rows(1, range(40))), cfg, FPS) == []


def test_appearance_can_lower_a_score_but_not_veto_it():
    work = _interleaved()
    fa, fb = list(range(0, 120, 2)), list(range(1, 120, 2))
    same = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (60, 1))),
                            2: (fb, np.tile([1.0, 0.0], (60, 1)))})
    opposite = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (60, 1))),
                                2: (fb, np.tile([-1.0, 0.0], (60, 1)))})
    cfg = RefineConfig.defaults()
    (e_same,) = propose_merges(work, cfg, FPS, same)
    (e_opp,) = propose_merges(work, cfg, FPS, opposite)
    assert e_same.signals["appearance"] == pytest.approx(1.0)
    assert e_same.algo_score == pytest.approx(1.0)
    assert e_opp.signals["appearance"] == pytest.approx(-1.0)
    assert e_opp.algo_score == pytest.approx(cfg.dedup.appearance_floor)


def test_a_provider_without_one_tracks_embeddings_gives_a_motion_only_score():
    fa = list(range(0, 120, 2))
    only_one = ArrayAppearance({1: (fa, np.tile([1.0, 0.0], (60, 1)))})
    (ev,) = propose_merges(_interleaved(), RefineConfig.defaults(), FPS, only_one)
    assert ev.signals["appearance"] is None and ev.algo_score == pytest.approx(1.0)


def test_lineage_tracks_and_counts_correspond_and_the_key_ignores_work_ids():
    fa = list(range(0, 80, 2))  # 40 frames
    fb = [0, *range(1, 78, 2), 78]  # 41 frames, starts on the same frame as the other track

    def run(ids):
        work = _work(_rows(122, fa), _rows(125, fb))
        work["track"] = work["track"].map(ids)  # raw_id keeps 122 and 125
        (ev,) = propose_merges(work, RefineConfig.defaults(), FPS)
        return work, ev

    w1, e1 = run({122: 3, 125: 7})
    w2, e2 = run({122: 7, 125: 3})
    assert e1.proposal_key == e2.proposal_key
    assert e1.tracks == e2.tracks == [3, 7]  # the tie is broken by work ID, so the order flips
    assert e1.lineage != e2.lineage
    n = {122: 40, 125: 41}
    for work, ev in ((w1, e1), (w2, e2)):
        raws = [int(work.loc[work["track"] == t, "raw_id"].iloc[0]) for t in ev.tracks]
        assert [lin[0][0] for lin in ev.lineage] == raws
        assert [ev.signals["n_a"], ev.signals["n_b"]] == [n[r] for r in raws]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup.py -q`
Expected: FAIL (`ModuleNotFoundError: dnt.refine.dedup`).

- [ ] **Step 3: Implement** (create `src/dnt/refine/dedup.py`)

```python
"""Stage ``dedup``: merge interleaved duplicate tracks (dedup spec)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig
from .events import Event, EventKind
from .features import track_embeddings
from .primitives import majority_class, ramp

log = logging.getLogger(__name__)

STAGE = "dedup"


@dataclass
class Track:
    """One track's observed rows, sorted by frame; ``rows`` keeps the work-table rows."""

    track: int
    cls: int
    frames: np.ndarray
    boxes: np.ndarray
    rows: pd.DataFrame


@dataclass
class Overlap:
    """The overlap of two tracks' frame ranges and the observations inside it (spec 3.2)."""

    lo: int
    hi: int
    n_a: int
    n_b: int
    shared: int
    co_occupancy: float
    in_a: np.ndarray
    in_b: np.ndarray


def describe(work: pd.DataFrame) -> dict[int, Track]:
    """Return a ``Track`` per track ID of ``work``."""
    out: dict[int, Track] = {}
    for t, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        out[int(t)] = Track(
            int(t),
            majority_class(g["cls"]),
            g["frame"].to_numpy(int),
            g[["x", "y", "w", "h"]].to_numpy(float),
            g,
        )
    return out


def class_ok(a: int, b: int, groups) -> bool:
    """Return True when two classes are equal or share a class group."""
    return a == b or any(a in g and b in g for g in groups)


def overlap(a: Track, b: Track) -> Overlap | None:
    """Return the overlap of ``a`` and ``b`` (frame range and observation counts), or None.

    ``co_occupancy`` is ``shared / min(n_a, n_b)``, the share of the sparser track's observed
    frames on which the other track is also observed; it is 0.0 when nothing is shared.
    """
    lo = int(max(a.frames[0], b.frames[0]))
    hi = int(min(a.frames[-1], b.frames[-1]))
    if lo > hi:
        return None
    in_a = (a.frames >= lo) & (a.frames <= hi)
    in_b = (b.frames >= lo) & (b.frames <= hi)
    n_a, n_b = int(in_a.sum()), int(in_b.sum())
    shared = int(np.intersect1d(a.frames[in_a], b.frames[in_b], assume_unique=True).size)
    denom = min(n_a, n_b)
    return Overlap(lo, hi, n_a, n_b, shared, shared / denom if denom else 0.0, in_a, in_b)


def _interp(t: Track, frames: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(frames, t.frames, t.boxes[:, k]) for k in range(4)])


def _iou_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def comotion(a: Track, b: Track, ov: Overlap) -> float:
    """Return the mean over both directions of the IoU of one track's observed boxes against the
    other track's interpolated box on the same frames (spec 3.2)."""
    ab = _iou_rows(a.boxes[ov.in_a], _interp(b, a.frames[ov.in_a])).mean()
    ba = _iou_rows(b.boxes[ov.in_b], _interp(a, b.frames[ov.in_b])).mean()
    return float((ab + ba) / 2.0)


def _unit_mean(frames: np.ndarray, emb: np.ndarray, lo: int, hi: int):
    keep = (frames >= lo) & (frames <= hi)
    if not keep.any():
        return None
    v = emb[keep].mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def appearance_similarity(a: Track, b: Track, ov: Overlap, appearance) -> float | None:
    """Return the cosine similarity of the two tracks' mean clean embeddings in the overlap.

    ``None`` without an appearance provider, or when either track has no clean sample there.
    """
    if appearance is None:
        return None
    ua = _unit_mean(*track_embeddings(appearance, lineage_of_rows(a.rows)), ov.lo, ov.hi)
    ub = _unit_mean(*track_embeddings(appearance, lineage_of_rows(b.rows)), ov.lo, ov.hi)
    if ua is None or ub is None:
        return None
    return float(ua @ ub)


def propose_merges(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, appearance=None
) -> list[Event]:
    """Propose one undecided ``MERGE`` event per candidate pair of tracks (spec 3.2, 3.3).

    A pair is a candidate when its classes share a group, the overlap of its frame ranges is at
    least ``min_overlap_seconds``, each track has at least ``min_observed`` observed rows in it,
    and the pair is not densely co-observed (``co_occupancy < cooccur_hi``).
    """
    dc = cfg.dedup
    if not dc.enabled or work.empty:
        return []
    groups = cfg.link.class_groups
    order = sorted(describe(work).values(), key=lambda t: (int(t.frames[0]), t.track))
    events: list[Event] = []
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            if second.frames[0] > first.frames[-1]:
                break
            if not class_ok(first.cls, second.cls, groups):
                continue
            ov = overlap(first, second)
            if ov is None or ov.hi - ov.lo + 1 < dc.min_overlap_seconds * fps:
                continue
            if ov.n_a < dc.min_observed or ov.n_b < dc.min_observed:
                continue
            if ov.co_occupancy >= dc.cooccur_hi:
                continue
            cm = comotion(first, second, ov)
            app = appearance_similarity(first, second, ov, appearance)
            term = (
                1.0
                if app is None
                else dc.appearance_floor
                + (1.0 - dc.appearance_floor) * float(ramp(app, dc.app_lo, dc.app_hi))
            )
            score = (
                float(ramp(cm, dc.comotion_lo, dc.comotion_hi))
                * (1.0 - float(ramp(ov.co_occupancy, dc.cooccur_lo, dc.cooccur_hi)))
                * term
            )
            lineage = [lineage_of_rows(first.rows), lineage_of_rows(second.rows)]
            events.append(
                Event.propose(
                    stage=STAGE,
                    kind=EventKind.MERGE,
                    tracks=[first.track, second.track],
                    lineage=lineage,
                    key_lineage=sorted(lineage, key=lambda spans: tuple(spans[0])),
                    frames=(ov.lo, ov.hi),
                    params={"span": [ov.lo, ov.hi]},
                    algo_score=score,
                    signals={
                        "shared": ov.shared,
                        "n_a": ov.n_a,
                        "n_b": ov.n_b,
                        "co_occupancy": ov.co_occupancy,
                        "comotion": cm,
                        "appearance": app,
                    },
                )
            )
    if appearance is None and events:
        log.info("dedup: no appearance provider; %d merge(s) scored on motion alone", len(events))
    return events
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup.py -q && .venv/bin/ruff check src/dnt/refine/dedup.py tests/refine/test_dedup.py`
Expected: all PASS; ruff clean. (If the `comotion` docstring trips a `D` rule, make its first line a single sentence ending in a period and move the rest to a second paragraph.)

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/dedup.py tests/refine/test_dedup.py
git commit -m "feat(refine): propose MERGE events for interleaved duplicate tracks

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `dedup.py`: `apply_merges`

**Files:**
- Modify: `src/dnt/refine/dedup.py` (append)
- Test: `tests/refine/test_dedup_apply.py`

**Interfaces:**
- Consumes: Task 3 (`merge_tracks`, `rows_to_drop`), Task 4 (`describe`, `overlap`, `class_ok`), `ACCEPTED`, `Decision`, `frame_runs`.
- Produces: `MergeOutcome` (`work`, `merged_reps: set[int]`, `pending_endpoints: set[int]`, `absorbed: dict[int, int]` (member to representative), `excluded: dict[int, set[int]]` (raw ID to dropped frames), `counts: dict[str, int]` with keys `applied`, `redundant`, `conflict`, `dropped_rows`) and `apply_merges(work, events, cfg) -> MergeOutcome`. It sets on each event `applied`, and in `signals`: `skipped_reason` (`"redundant"` or `"conflict"`), `conflicts_with` (`{"tracks": [x, y], "why": "rejected" | "class" | "dense", "proposal_key": str | None}`), and for applied edges `dropped_rows`, `dropped` (`[[raw_id, f0, f1], ...]`) and `merged_into`.

- [ ] **Step 1: Write the failing tests** (create `tests/refine/test_dedup_apply.py`)

```python
import random

import pandas as pd
import pytest

from dnt.refine import apply as A
from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.events import Decision, Event, EventKind, Ledger
from dnt.refine.verify import Band, decide, route_without_vlm

from ._fixtures import random_tracks
from .test_dedup import FPS, _rows, _work

CFG = RefineConfig.defaults()


def _routed(work, cfg=CFG):
    evs = propose_merges(work, cfg, FPS)
    route_without_vlm(evs, Band.of(cfg.dedup))
    return evs


def _ev(work, a, b, score=0.9, decision=Decision.AUTO_ACCEPT, source="auto"):
    """A hand-made MERGE event between tracks ``a`` and ``b`` of ``work``."""
    la, lb = A.lineage(work, a), A.lineage(work, b)
    lo = max(la[0][1], lb[0][1])
    ev = Event.propose(
        stage="dedup", kind=EventKind.MERGE, tracks=[a, b], lineage=[la, lb],
        key_lineage=sorted([la, lb], key=lambda spans: tuple(spans[0])),
        frames=(lo, lo), params={"span": [lo, lo]}, algo_score=score,
    )
    ev.id = f"dedup-r0-{a}-{b}"
    decide(ev, decision, source=source)
    return ev


def test_an_applied_merge_drops_the_lower_score_rows_and_reports_them():
    w = _work(_rows(1, range(0, 20), score=0.9), _rows(2, range(10, 30), score=0.8))
    ev = _ev(w, 1, 2)
    out = apply_merges(w, [ev], CFG)
    assert len(out.work) == 30 and out.work["track"].unique().tolist() == [1]
    assert ev.applied is True
    assert ev.signals["dropped_rows"] == 10 and ev.signals["dropped"] == [[2, 10, 19]]
    assert ev.signals["merged_into"] == 1
    assert out.excluded == {2: set(range(10, 20))}
    assert out.absorbed == {2: 1} and out.merged_reps == {1}
    assert out.counts == {"applied": 1, "redundant": 0, "conflict": 0, "dropped_rows": 10}


def test_a_three_way_interleave_merges_into_the_earliest_track():
    w = _work(_rows(1, range(0, 120, 3)), _rows(2, range(1, 120, 3)), _rows(3, range(2, 120, 3)))
    evs = _routed(w)
    assert len(evs) == 3 and all(e.decision is Decision.AUTO_ACCEPT for e in evs)
    out = apply_merges(w, evs, CFG)
    assert len(out.work) == 120 and out.work["track"].unique().tolist() == [1]
    assert out.absorbed == {2: 1, 3: 1}
    assert out.counts["applied"] == 2 and out.counts["redundant"] == 1
    redundant = [e for e in evs if e.signals.get("skipped_reason") == "redundant"]
    assert len(redundant) == 1 and redundant[0].applied is False
    assert redundant[0].signals.get("dropped_rows", 0) == 0


def _cyclic():
    frames = {1: [*range(0, 120, 3), 60], 2: [*range(1, 120, 3), 60], 3: [*range(2, 120, 3), 60]}
    w = _work(*(_rows(t, sorted(set(fs))) for t, fs in frames.items()))
    return w, _routed(w)


def test_a_cyclic_group_is_attributed_once_whatever_the_proposal_order():
    results = []
    for seed in range(5):
        w, evs = _cyclic()
        random.Random(seed).shuffle(evs)
        out = apply_merges(w, evs, CFG)
        kept = out.work[out.work["frame"] == 60]
        assert len(kept) == 1 and int(kept["raw_id"].iloc[0]) == 1  # tie: the smaller raw ID wins
        assert sum(e.signals.get("dropped_rows", 0) for e in evs) == 2
        assert out.counts == {"applied": 2, "redundant": 1, "conflict": 0, "dropped_rows": 2}
        results.append((
            out.work.reset_index(drop=True),
            {e.proposal_key: (e.applied, e.signals.get("dropped_rows", 0)) for e in evs},
        ))
    first = results[0]
    for table_, per_event in results[1:]:
        pd.testing.assert_frame_equal(table_, first[0])
        assert per_event == first[1]


def _dense_pair():
    # tracks 1 and 3 are densely co-observed (a person and a ghost of the same person); track 2
    # interleaves with both, so A/B and B/C are both accepted and A/C has no event
    return _work(_rows(1, range(0, 60, 2)), _rows(2, range(1, 60, 2)), _rows(3, range(0, 60, 2)))


def test_a_densely_co_observed_pair_blocks_the_bridge_whatever_the_order():
    applied_keys = set()
    for seed in range(5):
        w = _dense_pair()
        evs = _routed(w)
        assert len(evs) == 2  # 1/3 is gated, so it has no event
        random.Random(seed).shuffle(evs)
        out = apply_merges(w, evs, CFG)
        (done,) = [e for e in evs if e.applied]
        (blocked,) = [e for e in evs if not e.applied]
        applied_keys.add(done.proposal_key)
        assert blocked.decision is Decision.AUTO_ACCEPT  # accepted, but not applied
        assert blocked.signals["skipped_reason"] == "conflict"
        assert blocked.signals["conflicts_with"] == {
            "tracks": [1, 3], "why": "dense", "proposal_key": None}
        track_of = out.work.groupby("raw_id")["track"].first()
        assert track_of[1] != track_of[3]
        assert out.counts["applied"] == 1 and out.counts["conflict"] == 1
    assert len(applied_keys) == 1  # the same edge wins under every proposal order


def _bridge(c_frames):
    # A = 0..49 (track 1), B = 50..99 (track 2), C = c_frames then 100..110 (track 3); the A/B and
    # B/C edges are accepted; A/C has no event but overlaps A on the first frames of c_frames
    w = _work(_rows(1, range(0, 50)), _rows(2, range(50, 100)),
              _rows(3, [*c_frames, *range(100, 111)], score=0.8))
    return w, [_ev(w, 1, 2, score=0.95), _ev(w, 2, 3, score=0.90)]


@pytest.mark.parametrize(
    "c_frames", [range(41, 50), range(43, 50)], ids=["below-span-gate", "below-min-observed"]
)
def test_dense_overlap_below_the_proposal_gates_still_blocks_a_bridge(c_frames):
    w, evs = _bridge(c_frames)
    out = apply_merges(w, evs, CFG)
    assert [e.applied for e in evs] == [True, False]
    assert evs[1].signals["conflicts_with"]["why"] == "dense"
    assert evs[1].signals["conflicts_with"]["tracks"] == [1, 3]
    assert out.absorbed == {2: 1}


def test_a_coincidence_on_a_couple_of_frames_does_not_block_a_bridge():
    w, evs = _bridge(range(48, 50))  # shared on 2 frames, fewer than conflict_min_shared (3)
    out = apply_merges(w, evs, CFG)
    assert [e.applied for e in evs] == [True, True]
    assert out.absorbed == {2: 1, 3: 1} and out.work["track"].unique().tolist() == [1]
    assert out.counts["dropped_rows"] == 2  # C's rows on 48 and 49 lose to A's higher score


def _chain(c_decision, c_source="auto"):
    w = _work(_rows(1, range(0, 20)), _rows(2, range(20, 40)), _rows(3, range(40, 60)))
    return w, [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.90),
               _ev(w, 1, 3, 0.50, decision=c_decision, source=c_source)]


@pytest.mark.parametrize(
    ("decision", "source"), [(Decision.VLM_REJECT, "vlm"), (Decision.HUMAN_REJECT, "human")]
)
def test_an_explicit_rejection_blocks_a_join(decision, source):
    w, evs = _chain(decision, source)
    apply_merges(w, evs, CFG)
    assert [e.applied for e in evs[:2]] == [True, False]
    why = evs[1].signals["conflicts_with"]
    assert why == {"tracks": [1, 3], "why": "rejected", "proposal_key": evs[2].proposal_key}


@pytest.mark.parametrize("decision", [Decision.AUTO_REJECT, Decision.HUMAN_PENDING])
def test_an_auto_reject_or_a_pending_pair_does_not_block_a_join(decision):
    w, evs = _chain(decision)
    out = apply_merges(w, evs, CFG)
    assert [e.applied for e in evs[:2]] == [True, True]
    assert out.absorbed == {2: 1, 3: 1}


def test_different_classes_block_a_join():
    w = _work(_rows(1, range(0, 20), cls=0), _rows(2, range(20, 40), cls=0),
              _rows(3, range(40, 60), cls=2))
    evs = [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.90)]
    apply_merges(w, evs, CFG)
    assert [e.applied for e in evs] == [True, False]
    assert evs[1].signals["conflicts_with"] == {"tracks": [1, 3], "why": "class",
                                                "proposal_key": None}
    CFG_GROUP = RefineConfig.defaults()
    CFG_GROUP.link.class_groups = [[0, 2]]
    evs = [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.90)]
    apply_merges(w, evs, CFG_GROUP)
    assert [e.applied for e in evs] == [True, True]


def test_pending_endpoints_and_merged_reps_use_the_representatives():
    w = _work(_rows(1, range(0, 20)), _rows(2, range(20, 40)), _rows(3, range(40, 60)))
    evs = [_ev(w, 1, 2, 0.95), _ev(w, 2, 3, 0.60, decision=Decision.HUMAN_PENDING)]
    out = apply_merges(w, evs, CFG)
    assert out.merged_reps == {1} and out.absorbed == {2: 1}
    assert out.pending_endpoints == {1, 3}  # track 2 is now track 1
    assert evs[1].applied is False and "skipped_reason" not in evs[1].signals


def test_nothing_to_do_leaves_the_table_alone():
    w = _work(_rows(1, range(0, 20)))
    out = apply_merges(w, [], CFG)
    assert out.work is w or out.work.equals(w)
    assert out.merged_reps == set() and out.absorbed == {} and out.excluded == {}
    empty = apply_merges(w.iloc[0:0], [], CFG)
    assert empty.work.empty and empty.counts["applied"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_tracks_keep_every_observation_accounted_for(seed):
    w = io.to_work(random_tracks(seed=seed)).work
    evs = _routed(w)
    out = apply_merges(w, evs, CFG)
    assert not out.work.duplicated(["track", "frame"]).any()
    original = set(zip(w["raw_id"], w["frame"], strict=True))
    kept = set(zip(out.work["raw_id"], out.work["frame"], strict=True))
    dropped = {
        (raw, f)
        for e in evs
        for raw, a, b in e.signals.get("dropped", [])
        for f in range(a, b + 1)
    }
    assert kept | dropped == original and not (kept & dropped)
    assert len(out.work) == len(w) - sum(e.signals.get("dropped_rows", 0) for e in evs)


def test_skipped_and_applied_events_round_trip_through_the_ledger(tmp_path):
    w = _dense_pair()
    evs = _routed(w)
    apply_merges(w, evs, CFG)
    Ledger({"format": "x"}, evs).write(tmp_path / "l.jsonl")
    back = Ledger.read(tmp_path / "l.jsonl").events
    assert [e.to_dict() for e in back] == [e.to_dict() for e in evs]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup_apply.py -q`
Expected: FAIL (`ImportError: cannot import name 'apply_merges'`).

- [ ] **Step 3: Implement** (append to `src/dnt/refine/dedup.py`; add `Decision`, `ACCEPTED` to the events import, `merge_tracks`, `rows_to_drop` to the apply import and `frame_runs` to the primitives import)

Imports at the top become:

```python
from .apply import lineage_of_rows, merge_tracks, rows_to_drop
from .config import RefineConfig
from .events import ACCEPTED, Decision, Event, EventKind
from .features import track_embeddings
from .primitives import frame_runs, majority_class, ramp
```

Append:

```python
@dataclass
class MergeOutcome:
    """What stage ``dedup`` hands to the rest of the run (spec 3.5, 3.6, 5)."""

    work: pd.DataFrame
    merged_reps: set[int]
    pending_endpoints: set[int]
    absorbed: dict[int, int]
    excluded: dict[int, set[int]]
    counts: dict[str, int]


def _dense(a: Track, b: Track, dc) -> bool:
    """Return True when ``a`` and ``b`` are densely co-observed (rule 2, spec 3.5).

    This does not use ``min_overlap_seconds`` or ``min_observed``: those decide which pairs are
    worth proposing, not which pairs are safe to put in one track.
    """
    ov = overlap(a, b)
    return ov is not None and ov.shared >= dc.conflict_min_shared and ov.co_occupancy >= dc.cooccur_hi


def apply_merges(work: pd.DataFrame, events: list[Event], cfg: RefineConfig) -> MergeOutcome:
    """Apply the accepted ``MERGE`` events with the conflict rule of spec 3.5.

    Accepted edges are walked in the total order ``(-algo_score, proposal_key)`` with a
    union-find. An edge inside one component is ``redundant``; an edge that would join two
    components holding a cannot-link pair is a ``conflict``; both stay accepted with
    ``applied=False`` and a ``skipped_reason``. A cannot-link pair is a pair decided
    ``VLM_REJECT`` or ``HUMAN_REJECT``, a pair of classes in different groups, or a pair that
    is densely co-observed. Rows are dropped when an applied edge joins two components and are
    attributed to that edge.
    """
    dc = cfg.dedup
    counts = {"applied": 0, "redundant": 0, "conflict": 0, "dropped_rows": 0}
    if work.empty or not events:
        return MergeOutcome(work, set(), set(), {}, {}, counts)
    tracks = describe(work)
    groups = cfg.link.class_groups
    rejected = {
        frozenset(e.tracks): e.proposal_key
        for e in events
        if e.decision in (Decision.VLM_REJECT, Decision.HUMAN_REJECT)
    }
    accepted = sorted(
        (e for e in events if e.decision in ACCEPTED), key=lambda e: (-e.algo_score, e.proposal_key)
    )
    parent = {t: t for t in tracks}
    members = {t: [t] for t in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def blocker(ra: int, rb: int):
        for x, y in sorted((min(x, y), max(x, y)) for x in members[ra] for y in members[rb]):
            key = rejected.get(frozenset((x, y)))
            if key is not None:
                return x, y, "rejected", key
            if not class_ok(tracks[x].cls, tracks[y].cls, groups):
                return x, y, "class", None
            if _dense(tracks[x], tracks[y], dc):
                return x, y, "dense", None
        return None

    kept = pd.Series(True, index=work.index)
    applied: list[Event] = []
    for ev in accepted:
        a, b = ev.tracks
        ra, rb = find(a), find(b)
        if ra == rb:
            ev.applied = False
            ev.signals["skipped_reason"] = "redundant"
            counts["redundant"] += 1
            continue
        hit = blocker(ra, rb)
        if hit is not None:
            ev.applied = False
            ev.signals["skipped_reason"] = "conflict"
            ev.signals["conflicts_with"] = {
                "tracks": [hit[0], hit[1]],
                "why": hit[2],
                "proposal_key": hit[3],
            }
            counts["conflict"] += 1
            continue
        both = work[kept & work["track"].isin(members[ra] + members[rb])]
        lost = rows_to_drop(both)
        kept.loc[lost] = False
        gone = work.loc[lost]
        ev.signals["dropped_rows"] = int(len(gone))
        ev.signals["dropped"] = [
            [int(raw), int(f0), int(f1)]
            for raw, g in gone.groupby("raw_id")
            for f0, f1 in frame_runs(g["frame"])
        ]
        parent[rb] = ra
        members[ra] += members.pop(rb)
        ev.applied = True
        applied.append(ev)
        counts["applied"] += 1
    first = {t: int(d.frames[0]) for t, d in tracks.items()}
    rep_of: dict[int, int] = {}
    for group in (m for m in members.values() if len(m) > 1):
        rep = min(group, key=lambda t: (first[t], t))
        rep_of.update({m: rep for m in group})
    merged, dropped = merge_tracks(work, rep_of)
    excluded: dict[int, set[int]] = {}
    for raw, f in zip(dropped["raw_id"], dropped["frame"], strict=True):
        excluded.setdefault(int(raw), set()).add(int(f))
    for ev in applied:
        ev.signals["merged_into"] = rep_of[ev.tracks[0]]
    pending = {
        rep_of.get(t, t) for e in events if e.decision is Decision.HUMAN_PENDING for t in e.tracks
    }
    counts["dropped_rows"] = int(len(dropped))
    return MergeOutcome(
        work=merged,
        merged_reps={rep_of[e.tracks[0]] for e in applied},
        pending_endpoints=pending,
        absorbed={m: r for m, r in rep_of.items() if m != r},
        excluded=excluded,
        counts=counts,
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup.py tests/refine/test_dedup_apply.py -q && .venv/bin/ruff check src/dnt/refine/dedup.py tests/refine/test_dedup_apply.py && .venv/bin/ruff format --check src/dnt/refine/dedup.py`
Expected: PASS; ruff clean (run `ruff format src/dnt/refine/dedup.py` if the format check fails; the `_dense` return line may need wrapping).

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/dedup.py tests/refine/test_dedup_apply.py
git commit -m "feat(refine): apply accepted merges with the cannot-link conflict rule

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Thread `excluded` into every lineage call after dedup

**Files:**
- Modify: `src/dnt/refine/link.py` (`describe_tracks`, `score_candidates`, `legacy_link_events`, `run_link_stage`), `src/dnt/refine/screen.py` (`propose_orphans`), `src/dnt/refine/refiner.py` (`fill_stage`)
- Test: `tests/refine/test_dedup_lineage.py`

**Interfaces:**
- Consumes: `lineage_of_rows(rows, excluded)` (Task 3).
- Produces: an optional `excluded: dict[int, set[int]] | None = None` keyword on `describe_tracks` (4th positional), `score_candidates`, `legacy_link_events` (4th positional), `run_link_stage`, `propose_orphans`, and `fill_stage` (5th positional). With `excluded=None` every function behaves exactly as before.

- [ ] **Step 1: Write the failing tests** (create `tests/refine/test_dedup_lineage.py`)

```python
import numpy as np
import pandas as pd

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import EventKind
from dnt.refine.link import describe_tracks, legacy_link_events, score_candidates
from dnt.refine.refiner import fill_stage
from dnt.refine.screen import propose_orphans

from ._fixtures import table
from .test_dedup import _rows, _work

FRAMES = [f for f in range(10, 31) if f != 20]  # frame 20 was dropped by a merge
EXCLUDED = {122: {20}}


class _Recording:
    """An appearance provider that records the spans it is asked for."""

    def __init__(self):
        self.calls = []

    def clean_embeddings(self, raw_id, f0, f1):
        self.calls.append((int(raw_id), int(f0), int(f1)))
        frames = np.arange(int(f0), int(f1) + 1)
        return frames, np.tile([1.0, 0.0], (len(frames), 1))


def _link_work():
    return _work(_rows(122, FRAMES), _rows(7, range(33, 50)))


def test_link_descriptors_and_embedding_requests_skip_excluded_frames():
    work, cfg = _link_work(), RefineConfig.defaults()
    occluded = pd.Series(False, index=work.index)
    control = _Recording()
    score_candidates(work, cfg, 10.0, appearance=control, context=None, frame_size=None,
                     occluded=occluded)
    assert (122, 10, 30) in control.calls  # the span reads the dropped frame 20
    rec = _Recording()
    cands, descs = score_candidates(work, cfg, 10.0, appearance=rec, context=None,
                                    frame_size=None, occluded=occluded, excluded=EXCLUDED)
    assert cands and descs[122].lineage == [[122, 10, 19], [122, 21, 30]]
    assert (122, 10, 30) not in rec.calls
    assert (122, 10, 19) in rec.calls and (122, 21, 30) in rec.calls
    assert describe_tracks(work, cfg, 10.0, occluded, EXCLUDED)[122].lineage == descs[122].lineage


def test_legacy_link_events_use_the_excluded_frames():
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    (ev,) = legacy_link_events(_link_work(), cfg, 10.0, EXCLUDED)
    assert ev.lineage == [[[122, 10, 19], [122, 21, 30]], [[7, 33, 49]]]
    (plain,) = legacy_link_events(_link_work(), cfg, 10.0)
    assert plain.lineage == [[[122, 10, 30]], [[7, 33, 49]]]


def test_orphan_events_use_the_excluded_frames():
    work = _work(_rows(1, [0, 2, 3]))
    kw = dict(linked_tracks=set(), pending_endpoints=set())
    (ev,) = propose_orphans(work, RefineConfig.defaults(), 10.0, excluded={1: {1}}, **kw)[0]
    assert ev.lineage == [[[1, 0, 0], [1, 2, 3]]]
    (plain,) = propose_orphans(work, RefineConfig.defaults(), 10.0, **kw)[0]
    assert plain.lineage == [[[1, 0, 3]]]


def test_fill_events_use_the_excluded_frames():
    # frame 4 of raw 1 was dropped and the merged track holds raw 2's row there
    work = io.to_work(table(_rows(1, [f for f in range(10) if f != 4]), _rows(1, range(15, 25)),
                            _rows(1, [4]))).work
    work.loc[work["frame"] == 4, "raw_id"] = 2
    _, events = fill_stage(work, RefineConfig.defaults(), 10.0, {}, {1: {4}})
    fills = [e for e in events if e.kind is EventKind.FILL]
    assert len(fills) == 1
    assert fills[0].lineage == [[[1, 0, 3], [2, 4, 4], [1, 5, 24]]]
    _, plain = fill_stage(work, RefineConfig.defaults(), 10.0, {})
    assert [e for e in plain if e.kind is EventKind.FILL][0].lineage == [[[1, 0, 24], [2, 4, 4]]]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup_lineage.py -q`
Expected: FAIL (`unexpected keyword argument 'excluded'` / `takes ... positional arguments`).

- [ ] **Step 3: Implement**

`src/dnt/refine/link.py`:

```python
def describe_tracks(
    work, cfg: RefineConfig, fps: float, occluded, excluded=None
) -> dict[int, TrackDesc]:
    """Build a ``TrackDesc`` per track; ``occluded`` is the row-aligned occlusion mask.

    ``excluded`` maps a raw ID to frames dedup dropped; the descriptors' lineage skips them.
    """
```

and in its body replace `lineage=lineage_of_rows(g),` with `lineage=lineage_of_rows(g, excluded),`.

`score_candidates`: add `excluded=None` after `min_score: float | None = None,` in the signature, and change `descs = describe_tracks(work, cfg, fps, occluded)` to `descs = describe_tracks(work, cfg, fps, occluded, excluded)`.

`legacy_link_events`: change the signature to `def legacy_link_events(work, cfg: RefineConfig, fps: float, excluded=None) -> list[Event]:` and the two lineage lines to

```python
                lineage=[
                    lineage_of_rows(work[work["track"] == a], excluded),
                    lineage_of_rows(work[work["track"] == b], excluded),
                ],
```

`run_link_stage`: add `excluded=None,` after `route: Callable[[list[Event]], None],`; change `legacy_link_events(work, cfg, fps)` to `legacy_link_events(work, cfg, fps, excluded)`, and add `excluded=excluded,` to the `score_candidates(...)` call (after `min_score=cfg.link.reject_below,`).

`src/dnt/refine/screen.py`, `propose_orphans`: add `excluded=None,` after `pending_endpoints: set[int],` and change `lineage=[lineage_of_rows(g)],` (the orphan one, inside `propose_orphans`) to `lineage=[lineage_of_rows(g, excluded)],`. Leave `lin = lineage_of_rows(g)` in `propose_screen` alone (it runs before dedup).

`src/dnt/refine/refiner.py`, `fill_stage`: change the signature to

```python
def fill_stage(
    work: pd.DataFrame,
    cfg: RefineConfig,
    fps: float,
    protected: dict[int, list[tuple[int, int]]],
    excluded: dict[int, set[int]] | None = None,
) -> tuple[pd.DataFrame, list[Event]]:
```

and change `lin = lineage_of_rows(obs)` to `lin = lineage_of_rows(obs, excluded)` and `lineage=[lineage_of_rows(before[before["track"] == t])],` to `lineage=[lineage_of_rows(before[before["track"] == t], excluded)],`.

(`switch.py` runs before dedup and needs no change.)

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine -q`
Expected: PASS (1149 existing + the new tests; `excluded=None` keeps old behavior everywhere).

- [ ] **Step 5: Commit**

```bash
git add src/dnt/refine/link.py src/dnt/refine/screen.py src/dnt/refine/refiner.py tests/refine/test_dedup_lineage.py
git commit -m "feat(refine): lineage consumers skip frames dedup dropped

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Wire `dedup` into `_Stages` and `refine()`

**Files:**
- Modify: `src/dnt/refine/refiner.py`
- Test: `tests/refine/test_dedup_stage.py`; fix-ups in existing tests as Step 6 describes.

**Interfaces:**
- Consumes: Tasks 1 to 6.
- Produces: `_Stages` attributes `merged_reps`, `merge_pending`, `absorbed` (work ID to the work ID of the track that finally absorbed it), `excluded`, `merge_counts`; `_Stages._link` returns a 5-tuple `(work, protected, linked, pending, rep_of)`; ledger header gains `"absorbed": {str(work_id): survivor_work_id}`; `summary["dedup"]` holds `applied`, `redundant`, `conflict`, `dropped_rows`; `write_review(..., absorbed=...)` is passed (Task 8 adds the parameter; until then the call is left without it).

- [ ] **Step 1: Write the failing tests** (create `tests/refine/test_dedup_stage.py`)

```python
import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.dedup import propose_merges
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.refiner import TrackRefiner, _Stages
from dnt.refine.verify import Band

from ._fixtures import table
from .test_dedup import _rows, _work


def _stages(cfg, fps=20.0):
    return _Stages(cfg, fps, None, None, None, None, {})


def _short_pair_cfg(link_on):
    cfg = RefineConfig.defaults()
    cfg.link.enabled = link_on
    cfg.dedup.min_observed = 1
    cfg.dedup.min_overlap_seconds = 0.1
    return cfg


@pytest.mark.parametrize("link_on", [True, False])
def test_an_accepted_merge_protects_its_representative_from_the_orphan_drop(link_on):
    work = _work(_rows(1, [0, 20]), _rows(2, [10, 30]))  # 4 rows in all: 0.2 s at 20 fps
    off = _short_pair_cfg(link_on)
    off.dedup.enabled = False
    out, events = _stages(off).run(work)
    assert out.empty and {e.stage for e in events if e.kind is EventKind.DROP} == {"orphan"}
    on = _short_pair_cfg(link_on)
    out, events = _stages(on).run(work)
    assert out["track"].unique().tolist() == [1] and len(out) == 4
    (merge,) = [e for e in events if e.stage == "dedup"]
    assert merge.decision is Decision.AUTO_ACCEPT and merge.applied
    assert not [e for e in events if e.stage == "orphan"]


@pytest.mark.parametrize("link_on", [True, False])
def test_a_pending_merge_defers_the_orphan_drop_of_both_endpoints(link_on):
    work = _work(_rows(1, [0, 20]), _rows(2, [10, 30], dx=14.0))
    stages = _stages(_short_pair_cfg(link_on))
    out, events = stages.run(work)
    (merge,) = [e for e in events if e.stage == "dedup"]
    assert merge.decision is Decision.HUMAN_PENDING
    assert not [e for e in events if e.stage == "orphan"]
    assert sorted(stages.orphan_deferred) == [1, 2]
    assert sorted(out["track"].unique()) == [1, 2]


def test_dedup_routing_never_goes_through_the_vlm():
    cfg = _short_pair_cfg(True)
    stages = _stages(cfg)
    stages.vlm = object()  # route_with_vlm would fail on it: it has no runner or evidence
    events = propose_merges(_work(_rows(1, [0, 20]), _rows(2, [10, 30], dx=14.0)), cfg, 20.0)
    stages._route(events, Band.of(cfg.dedup), "dedup", use_vlm=False)
    assert [e.decision for e in events] == [Decision.HUMAN_PENDING]
    assert events[0].id == "dedup-r0-000001"


def test_absorbed_ids_compose_across_dedup_and_link():
    work = _work(_rows(1, range(0, 60, 2)), _rows(2, range(1, 60, 2)), _rows(3, range(62, 111)))
    stages = _stages(RefineConfig.defaults(), fps=10.0)
    out, events = stages.run(work)
    assert out["track"].unique().tolist() == [1]
    assert stages.absorbed == {2: 1, 3: 1}  # 2 into 1 by the merge, 3 into 1 by the link
    kinds = {(e.stage, str(e.kind), str(e.decision)) for e in events if e.stage != "fill"}
    assert ("dedup", "MERGE", "AUTO_ACCEPT") in kinds and ("link", "LINK", "AUTO_ACCEPT") in kinds
    assert stages.merge_counts["applied"] == 1


def test_a_disabled_stage_leaves_no_trace():
    cfg = RefineConfig.defaults()
    cfg.dedup.enabled = False
    work = _work(_rows(1, range(0, 60, 2)), _rows(2, range(1, 60, 2)))
    stages = _stages(cfg, fps=10.0)
    out, events = stages.run(work)
    assert not [e for e in events if e.stage == "dedup"]
    assert stages.absorbed == {} and stages.excluded == {}
    assert stages.merge_counts == {"applied": 0, "redundant": 0, "conflict": 0, "dropped_rows": 0}
    assert sorted(out["track"].unique()) == [1, 2]


def _write(tmp_path, df):
    p = tmp_path / "t.txt"
    df.to_csv(p, index=False, header=False)
    return p


def test_the_ledger_header_and_summary_describe_the_merges(tmp_path):
    src = _write(tmp_path, table(_rows(1, range(0, 120, 2)), _rows(2, range(1, 120, 2))))
    refiner = TrackRefiner(RefineConfig.defaults())
    tracks = refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    assert tracks["track"].nunique() == 1 and len(tracks) == 120
    ledger = Ledger.read(refiner.last_result.ledger_path)
    assert ledger.header["absorbed"] == {"2": 1}
    summary = refiner.last_result.summary
    assert summary["dedup"] == {"applied": 1, "redundant": 0, "conflict": 0, "dropped_rows": 0}
    assert "dedup/MERGE/AUTO_ACCEPT" in summary["events"]
    off = RefineConfig.defaults()
    off.dedup.enabled = False
    refiner = TrackRefiner(off)
    tracks = refiner.refine(src, tmp_path / "off.txt", fps=10, verbose=False)
    assert tracks["track"].nunique() == 2  # fill gives each ID the other's frames
    assert Ledger.read(refiner.last_result.ledger_path).header["absorbed"] == {}
    assert refiner.last_result.summary["dedup"]["applied"] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup_stage.py -q`
Expected: FAIL (no `dedup` events; `AttributeError: ... 'absorbed'`).

- [ ] **Step 3: Implement** in `src/dnt/refine/refiner.py`

Add to the imports: `from .dedup import apply_merges, propose_merges`.

In `_Stages.__init__`, after `self.orphan_deferred: list[int] = []`:

```python
        self.merged_reps: set[int] = set()
        self.merge_pending: set[int] = set()
        self.absorbed: dict[int, int] = {}
        self.excluded: dict[int, set[int]] = {}
        self.merge_counts = {"applied": 0, "redundant": 0, "conflict": 0, "dropped_rows": 0}
```

Replace `_route`:

```python
    def _route(self, evs: list[Event], band: Band, stage: str, *, use_vlm: bool = True) -> None:
        for e in evs:  # ids first: the VLM question tag uses them
            self.seq[stage] += 1
            e.id = f"{stage}-r0-{self.seq[stage]:06d}"
        if self.vlm is not None and use_vlm:
            route_with_vlm(evs, band, vlm=self.vlm)
        else:
            route_without_vlm(evs, band)
        self.events.extend(evs)
```

Add the two helpers after `_route`:

```python
    def _absorb(self, mapping: dict[int, int]) -> None:
        """Compose ``mapping`` (an absorbed work ID to its absorber) into ``self.absorbed``."""
        mapping = {int(k): int(v) for k, v in mapping.items() if int(k) != int(v)}
        if not mapping:
            return
        for k, v in self.absorbed.items():
            self.absorbed[k] = mapping.get(v, v)
        for k, v in mapping.items():
            self.absorbed.setdefault(k, v)

    def _dedup(self, work):
        cfg = self.cfg
        if not cfg.dedup.enabled or work.empty:
            return work
        if self.vlm_runner is not None:
            log.info("dedup does not use the VLM; uncertain merges stay pending for review")
        evs = propose_merges(work, cfg, self.fps, self.link_app)
        self._route(evs, Band.of(cfg.dedup), "dedup", use_vlm=False)
        out = apply_merges(work, evs, cfg)
        self.merged_reps, self.merge_pending = out.merged_reps, out.pending_endpoints
        self.excluded = out.excluded
        self.merge_counts = out.counts
        self._absorb(out.absorbed)
        return out.work
```

In `run`, replace the block from `work = self._screen(...)` through `work = self._orphans(...)`:

```python
        work = self._screen(work, split_raw, cuts)
        tick("screen")
        work = self._dedup(work)
        tick("dedup")
        work, protected, linked, pending, rep_of = self._link(work, occluded)
        tick("link")
        self._absorb(rep_of)
        linked = linked | {rep_of.get(t, t) for t in self.merged_reps}
        pending = pending | {rep_of.get(t, t) for t in self.merge_pending}
        work = self._orphans(work, linked, pending)
        tick("orphan")
```

In `_link`, change the early return to `return work, {}, set(), set(), {}`, add `excluded=self.excluded,` to the `run_link_stage(...)` call (after `route=...`), and change the final return to `return work, protected, set(rep_of.values()), pending, rep_of`.

In `_orphans`, add `excluded=self.excluded,` to the `propose_orphans(...)` call. In `_fill`, change the call to `fill_stage(work, self.cfg, self.fps, protected, self.excluded)`.

In `TrackRefiner.refine`: change `tqdm(total=5,` to `tqdm(total=6,`; add `"dedup": dict(stages.merge_counts),` to the `summary` dict (after `"events": ...`); add `"absorbed": {str(k): int(v) for k, v in stages.absorbed.items()},` to the ledger `header` (after `"id_map": ...`).

- [ ] **Step 4: Run to verify the new tests pass**

Run: `.venv/bin/python -m pytest tests/refine/test_dedup_stage.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole refine suite and triage**

Run: `.venv/bin/python -m pytest tests/refine -q -x -p no:cacheprovider 2>&1 | tail -30`
Then, without `-x`, list every failure: `.venv/bin/python -m pytest tests/refine -q 2>&1 | grep FAILED`.

For each failing existing test decide, by reading it:
- It asserts behavior of the pipeline *before* dedup existed (event counts, track counts, stage lists, progress ticks, baselines on `random_tracks`): set `cfg.dedup.enabled = False` on that test's config. Do not change its expected values.
- It exposes a defect in this task's code: fix the code, not the test.
- It shows a real, intended effect of dedup that the test should now assert: update the expectation and say why in the commit message.

Re-run until `tests/refine` is green. Record the list of tests you touched and the rule applied to each in the commit message body.

- [ ] **Step 6: Commit**

```bash
git add src/dnt/refine/refiner.py tests/refine
git commit -m "feat(refine): run dedup between screen and link

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Evidence and review for `MERGE`

**Files:**
- Modify: `src/dnt/refine/evidence.py` (`EvidenceBuilder.plan`), `src/dnt/refine/review.py`, `src/dnt/refine/refiner.py` (pass `absorbed` to `write_review`)
- Test: `tests/refine/test_evidence.py`, `tests/refine/test_review.py`

**Interfaces:**
- Consumes: `signals.skipped_reason == "conflict"` and `decision in ACCEPTED` identify a read-only card; the header-style `absorbed` map from Task 7.
- Produces: `write_review(..., absorbed: dict | None = None)`; `_output_ids(ev, id_map, absorbed=None)`; a `MERGE` branch in `EvidenceBuilder.plan` (rows `("A", tiles)` from `lineage[0]` and `("B", tiles)` from `lineage[1]`, at most 6 tiles each, plus one context frame); the "Skipped merges" section with `class="skipcard"` cards (no radio buttons, no `data-` attributes the page script reads).

- [ ] **Step 1: Write the failing evidence test** (append to `tests/refine/test_evidence.py`)

```python
def test_merge_plan_shows_each_track_in_its_own_row(tmp_path):
    w = _work(_walker(1, range(0, 60, 2)), _walker(2, range(1, 60, 2)))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    ev = Event.propose(stage="dedup", kind=EventKind.MERGE, tracks=[1, 2],
                       lineage=[[[1, 0, 58]], [[2, 1, 59]]], frames=(1, 58),
                       params={"span": [1, 58]}, algo_score=0.9, signals={})
    ev.id = "dedup-r0-000001"
    plan = b.plan(ev)
    (label_a, a), (label_b, bb) = plan.rows
    assert (label_a, label_b) == ("A", "B")
    assert 0 < len(a) <= 6 and 0 < len(bb) <= 6
    assert {raw for raw, _ in a} == {1} and {raw for raw, _ in bb} == {2}
    assert all(1 <= f <= 58 for _, f in a + bb)  # only frames inside the span
    assert len(plan.contexts) == 1
    assert b.build(ev) is not None
```

- [ ] **Step 2: Write the failing review tests** (append to `tests/refine/test_review.py`)

```python
from dnt.refine.events import ACCEPTED


def skipped_merge(idx=1, reason="conflict"):
    ev = pending(EventKind.MERGE, "dedup", idx, tracks=(1, 2), span=[10, 90],
                 signals={"co_occupancy": 0.9, "shared": 20})
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert ev.decision in ACCEPTED
    ev.applied = False
    ev.signals["skipped_reason"] = reason
    ev.signals["conflicts_with"] = {"tracks": [1, 3], "why": "dense", "proposal_key": None}
    return ev


def test_a_conflicting_accepted_merge_gets_a_read_only_card_and_keeps_the_page(tmp_path):
    page = write(tmp_path, [skipped_merge()])
    assert page is not None
    html = page.read_text()
    assert "Skipped merges" in html and 'class="skipcard"' in html
    assert "conflict" in html and "dense" in html
    assert 'type="radio"' not in html and 'class="card"' not in html


def test_a_redundant_merge_has_no_card_and_a_page_with_neither_group_is_removed(tmp_path):
    assert write(tmp_path, [skipped_merge(reason="redundant")]) is None
    write(tmp_path, [skipped_merge()])
    assert (tmp_path / "o.review.html").exists()
    assert write(tmp_path, []) is None
    assert not (tmp_path / "o.review.html").exists()
    assert not review_image_dir(tmp_path / "o.review.html").exists()


def test_pending_and_skipped_cards_are_separate_and_only_pending_ones_have_controls(tmp_path):
    pend = pending(EventKind.MERGE, "dedup", 1, tracks=(1, 2), span=[10, 90])
    html = write(tmp_path, [pend, skipped_merge(idx=2)]).read_text()
    assert html.count('class="card"') == 1 and html.count('class="skipcard"') == 1
    assert html.count('type="radio"') == 2  # accept and reject of the pending card only


def test_a_pending_merge_resolves_absorbed_endpoints_to_the_survivors_output_id(tmp_path):
    # track 4 was absorbed by a merge (into 2) and then by a link (into 1): the header's map
    # holds the resolved survivor for every absorbed ID
    ev = pending(EventKind.MERGE, "dedup", 1, tracks=(4, 9), span=[10, 90])
    html = write(tmp_path, [ev], id_map={1: 1, 9: 2}, absorbed={"4": 1, "2": 1}).read_text()
    assert "output id(s): 1, 2" in html and "track_ids=[1, 2]" in html
    # without the map the absorbed endpoint has no output ID, as before
    plain = write(tmp_path, [ev], id_map={1: 1, 9: 2}).read_text()
    assert "track_ids=[2]" in plain
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_evidence.py tests/refine/test_review.py -q -k "merge or absorbed or skipped"`
Expected: FAIL (empty plan rows; `unexpected keyword argument 'absorbed'`; no skipped section).

- [ ] **Step 4: Implement the evidence plan** in `src/dnt/refine/evidence.py`, inside `plan`, after the `LINK` branch (before `return plan`):

```python
        elif event.kind is EventKind.MERGE:
            b_spans = event.lineage[1]
            lo, hi = (int(v) for v in event.params["span"])
            a_obs = self._clean(a_spans, lo, hi)
            b_obs = self._clean(b_spans, lo, hi)
            plan.rows += [("A", _spread(a_obs, 6)), ("B", _spread(b_obs, 6))]
            f = a_obs[len(a_obs) // 2][1] if a_obs else (lo + hi) // 2
            plan.contexts += self._ctx(
                f,
                "A and B",
                [
                    ContextBox("A", GREEN, False, self._box_at(a_spans, f)),
                    ContextBox("B", RED, False, self._box_at(b_spans, f)),
                ],
            )
```

- [ ] **Step 5: Implement the review changes** in `src/dnt/refine/review.py`

Imports: `from .events import ACCEPTED, Decision, Event, EventKind`.

CSS: change the two rules to `.card,.skipcard{background:#fff;border:1px solid #ccc;border-radius:6px;margin:12px 0;padding:10px}` and `.card img,.skipcard img{max-width:100%;border:1px solid #ddd}`.

Replace `_output_ids`:

```python
def _output_ids(ev: Event, id_map: dict, absorbed: dict | None = None) -> list:
    """Return the output ids of the event's tracks; a track gone from the output has none.

    ``absorbed`` maps an absorbed work id to the work id that finally absorbed it (the ledger
    header's map); a track that merging or linking absorbed is shown as its survivor.
    """
    absorbed = absorbed or {}
    out = []
    for t in ev.tracks:
        s = absorbed.get(t, absorbed.get(str(t), t))
        i = id_map.get(s, id_map.get(str(s)))
        if i is not None and i not in out:
            out.append(i)
    return out
```

Add `absorbed=None` as the last parameter of `_snippet` and `_card`, and pass it on: `ids = _output_ids(ev, id_map, absorbed)` in `_snippet`, `out_ids = ", ".join(str(i) for i in _output_ids(ev, id_map, absorbed)) or "none"` in `_card`.

Add the read-only card after `_card`:

```python
def _skipcard(ev: Event, img_rel: str | None, id_map: dict, absorbed=None) -> str:
    image = (
        f'<img src="{_e(img_rel)}" alt="evidence" loading="lazy">'
        if img_rel
        else "<div>no image</div>"
    )
    out_ids = ", ".join(str(i) for i in _output_ids(ev, id_map, absorbed)) or "none"
    why = ev.signals.get("conflicts_with") or {}
    return (
        f'<div class="skipcard" data-skipped="{_e(ev.id)}">{image}'
        f'<div class="meta"><span><b>{_e(ev.kind)}</b> accepted, not applied: '
        f'{_e(ev.signals.get("skipped_reason"))}</span>'
        f"<span>tracks {_e(ev.tracks)}</span><span>output id(s): {_e(out_ids)}</span>"
        f"<span>frames {_e(ev.frames[0])}-{_e(ev.frames[1])}</span>"
        f"<span>score {ev.algo_score:.3f}</span></div>"
        f'<div class="sig">{_e(" ".join(_top_signals(ev.signals)))}</div>'
        f"<div>blocked by tracks {_e(why.get('tracks'))}: {_e(why.get('why'))}</div></div>"
    )
```

In `write_review`: add `absorbed: dict | None = None,` to the signature (after `run_key: str,`). After `pend = [...]` add

```python
    skipped = [
        e
        for e in events
        if e.kind is EventKind.MERGE
        and e.decision in ACCEPTED
        and e.signals.get("skipped_reason") == "conflict"
    ]
    shown = pend + skipped
```

and change the early exit to `if not shown:`. Replace the remaining uses of `pend` after that point: `evidence.build_many(pend)` becomes `evidence.build_many(shown)`; `stages = sorted({e.stage for e in pend})` becomes `sorted({e.stage for e in shown})`; both `for ev in pend:` loops become `for ev in shown:`; and inside the second loop build the card by group:

```python
        snippet = _snippet(ev, id_map, fps, video_file, track_file, absorbed)
        if ev in skipped:
            cards_skipped.append(_skipcard(ev, None if rel is None else quote(rel), id_map, absorbed))
        else:
            cards.append(
                _card(ev, None if rel is None else quote(rel), snippet, reclass_map, id_map, absorbed)
            )
```

with `cards_skipped = []` declared beside `cards`. In the page assembly, after the `<div id="cards">...</div>` element add the skipped section:

```python
    skipped_html = (
        f"<h3>Skipped merges ({len(skipped)})</h3><div id=\"skipped\">"
        + "".join(cards_skipped)
        + "</div>"
        if skipped
        else ""
    )
```

and place `+ skipped_html` between the `cards` div and `<script>` in the `page` string. (`ev in skipped` compares dataclasses by value; use `id(ev) in {id(e) for e in skipped}` computed once as `skipped_ids` if ruff or a profiler flags it.)

In `src/dnt/refine/refiner.py`, add `absorbed=stages.absorbed,` to the `write_review(...)` call in `refine()`.

- [ ] **Step 6: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/refine/test_evidence.py tests/refine/test_review.py tests/refine/test_dedup_stage.py -q`
Expected: PASS. Then `.venv/bin/python -m pytest tests/refine -q` for the whole package.

- [ ] **Step 7: Commit**

```bash
git add src/dnt/refine/evidence.py src/dnt/refine/review.py src/dnt/refine/refiner.py tests/refine/test_evidence.py tests/refine/test_review.py
git commit -m "feat(refine): MERGE evidence, absorbed-ID lookup and a read-only skipped-merges section

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Real-data regression, documentation and final checks

**Files:**
- Create: `tests/refine/test_dedup_real.py`, `docs/api/refine/dedup.md`
- Modify: `docs/changelog.md`, `docs/api/refine/index.md`, `mkdocs.yml`

- [ ] **Step 1: Write the real-data test** (create `tests/refine/test_dedup_real.py`)

```python
from pathlib import Path

import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.events import Ledger
from dnt.refine.verify import Band, route_without_vlm

DATA = Path("/mnt/e/videos/miami/dets")
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="the Miami clips are not mounted")


def _run(clip):
    track_file = next(DATA.glob(f"*{clip}_ped_track.txt"))
    ledger = next(DATA.glob(f"*{clip}_ped_track_refined_p3.ledger.jsonl"))
    fps = Ledger.read(ledger).header["fps"]
    work = io.read_tracks(track_file, fmt="dnt", class_id=0).work
    cfg = RefineConfig.defaults()
    events = propose_merges(work, cfg, fps)
    route_without_vlm(events, Band.of(cfg.dedup))
    return work, events, apply_merges(work, events, cfg)


def test_the_evening_group_is_one_person_and_becomes_one_track():
    work, events, out = _run("evening_1900_190000.00")
    assert {out.absorbed.get(t, t) for t in (122, 125, 134)} == {122}
    merged = out.work[out.work["raw_id"].isin([122, 125, 134])]
    assert merged["track"].nunique() == 1 and not merged.duplicated("frame").any()
    ours = [e for e in events if set(e.tracks) <= {122, 125, 134} and e.applied]
    assert sum(e.signals["dropped_rows"] for e in ours) <= 13
    assert len(out.work) < len(work) and out.work["track"].nunique() < work["track"].nunique()


def test_side_by_side_pairs_in_the_middle_clip_stay_apart():
    _, _, out = _run("middle_120000.00")
    track_of = out.work.groupby("raw_id")["track"].first()
    assert track_of[266] != track_of[270]
    assert track_of[15] != track_of[19]
```

Run: `.venv/bin/python -m pytest tests/refine/test_dedup_real.py -q`
Expected: PASS when `/mnt/e/videos/miami/dets` is mounted (it is on the author's machine), otherwise SKIPPED. If an assertion fails, do **not** loosen it: print the signals and scores of the pairs involved and compare with spec §12 (the prototype numbers) before deciding whether the code or a default is wrong.

- [ ] **Step 2: Calibration check** (no code change expected)

Run this scratch check from the repo root and compare with the table in spec §12 (tracks before and after, merges applied, rows dropped: beginning 63 to 62 (1, 1 row); middle 89 to 83 (6, 10 rows); evening 134 to 129 (5, 19 rows)):

```bash
.venv/bin/python - <<'EOF'
import glob
from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.verify import Band, route_without_vlm

D = "/mnt/e/videos/miami/dets/"
cfg = RefineConfig.defaults()
for clip in ("beginning_000000.00", "middle_120000.00", "evening_1900_190000.00"):
    work = io.read_tracks(glob.glob(D + f"*{clip}_ped_track.txt")[0], fmt="dnt", class_id=0).work
    evs = propose_merges(work, cfg, 10.0)
    route_without_vlm(evs, Band.of(cfg.dedup))
    out = apply_merges(work, evs, cfg)
    print(clip, work.track.nunique(), "->", out.work.track.nunique(), out.counts)
EOF
```

If a number differs by more than the borderline pairs (15/18 scores exactly 0.75, and 122/134 is 0.79), report the difference in the commit message and in the handoff; do not tune defaults silently.

- [ ] **Step 3: Documentation**

`docs/changelog.md`: under `## Unreleased` / `### New`, add:

```markdown
- `dnt.refine` merges interleaved duplicate tracks. A tracker sometimes alternates IDs on one
  person from frame to frame; the fill stage then interpolated each ID into the other's frames,
  so the output showed overlapping boxes on one person. The new `dedup` stage runs between
  screen and link: it finds pairs of tracks that move together but are almost never observed on
  the same frame, scores them (`MERGE` events), and merges the confident ones into the earlier
  ID, keeping the higher-score row where both have one. Pairs observed together often (two
  people walking side by side) are never merged. Uncertain merges go to the review page;
  merges that were accepted but blocked by a conflicting pair are shown there read-only. Set
  `dedup.enabled: false` for the old behavior. New config block `dedup`; the ledger header gains
  `absorbed` and the run summary gains `dedup`.
```

`docs/api/refine/dedup.md`:

```markdown
# Duplicate merging

::: dnt.refine.dedup
```

`mkdocs.yml`: add `          - Duplicate merging: api/refine/dedup.md` after the `Linking` entry.

`docs/api/refine/index.md`: in the first paragraph change "it splits ID switches, screens false tracks, links fragments" to "it splits ID switches, screens false tracks, merges interleaved duplicate tracks, links fragments", and add this section before the `Appearance` section (find it with `grep -n "^## Appearance" docs/api/refine/index.md`):

```markdown
## Interleaved duplicate tracks

A tracker can alternate IDs on one person from frame to frame. Each ID then covers part of the
trajectory over the same span, and gap filling makes the overlap visible. Stage `dedup` (between
screen and link) proposes a `MERGE` for two tracks whose boxes follow each other but which are
observed together on few frames (`co_occupancy`, the share of the sparser track's frames on
which the other is also observed, is low). Two people walking side by side are observed
together on most frames, so they are never merged.

```yaml
dedup:
  enabled: true
  accept_above: 0.75   # auto-merge at or above this score
  reject_below: 0.40   # auto-reject below it; in between the merge goes to review
  cooccur_hi: 0.50     # observed together on this share of frames or more: never a candidate
```

Merged tracks keep the earlier track's ID and, on a frame where both had a row, the row with
the higher score; the dropped rows are listed in the event's `dropped` signal. Merges are never
sent to the VLM in this release.
```

- [ ] **Step 4: Final checks**

Run each and fix what fails:

```bash
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src/dnt/refine
.venv/bin/python -m pytest -q
.venv/bin/mkdocs build --strict -d "$(mktemp -d)"
```

Expected: ruff clean (no growth of the per-file baseline in `pyproject.toml`); the default pytest suite passes (record the counts); `mkdocs build --strict` succeeds. Build to a temporary directory: `site/` is committed and must not change. If `mkdocs build` needs the PDF plugin's system packages and fails for that reason alone, say so in the handoff rather than skipping silently.

- [ ] **Step 5: Commit**

```bash
git add tests/refine/test_dedup_real.py docs/changelog.md docs/api/refine/dedup.md docs/api/refine/index.md mkdocs.yml
git commit -m "docs(refine): document the dedup stage and add the real-data regression

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (spec rev. 4)

| Spec | Task |
|---|---|
| §2 criteria 1 to 5 | 9 (criterion 1, negatives), 5 (criteria 3, 4), 4 (criterion 2, key), 6 (downstream exclusion) |
| §3.1 order, progress total, no new IDs | 7 |
| §3.2 signals, gate, score | 4 |
| §3.3 event, `key_lineage`, `n_a`/`n_b` correspondence | 2, 4 |
| §3.4 no-VLM routing | 7 (`use_vlm=False`) |
| §3.5 conflict rule, attribution, `excluded` | 3, 5, 6 |
| §3.6 hand-off to orphan and link | 7 |
| §4 replay compatibility | 2, 4 (keys); `apply` itself is out of scope |
| §5 absorbed map, review sections, summary | 7, 8 |
| §6 config | 1 |
| §7 compatibility (`enabled: false`, old ledgers) | 7 (disabled test), 8 (`absorbed=None`) |
| §9 tests 1 to 18 | 1 (17), 4 (1 to 6, 13), 5 (7 to 9, bands in 7), 6 (18), 7 (11, 12), 8 (14, 15), 9 (16) |

Spec test 9 (bands, no VLM) is covered in Task 7: `test_dedup_routing_never_goes_through_the_vlm` sets a `stages.vlm` that would fail if used and checks that the uncertain merge is still `HUMAN_PENDING`. The auto-accept and auto-reject bands are covered by the Task 5 tests (hand-made events) and the Task 7 orphan tests.

**Placeholder scan:** no TBD or "similar to" steps. The two places that say "if X, adjust" (ruff docstring wording in Task 4; the `ev in skipped` identity check in Task 8) are lint outcomes, not missing design.

**Type consistency:** `propose_merges(work, cfg, fps, appearance=None)`, `apply_merges(work, events, cfg) -> MergeOutcome`, `MergeOutcome.absorbed` (member to representative), `_Stages.absorbed` (composed, to survivor), `excluded: dict[int, set[int]]` everywhere, `rows_to_drop(rows) -> pd.Index`, `merge_tracks(work, rep_of) -> (merged, dropped)`, `lineage_of_rows(rows, excluded=None)`. The 5-tuple from `_link` is consumed only in `run`.

**Clarification of the spec made in this plan:** `propose_merges` proposes an event for every pair that passes the gates, including pairs scoring 0 (they are auto-rejected). The spec does not say; recording them keeps the ledger a complete account of what was considered. The prototype produced 23, 40 and 160 such events on the three clips (most auto-rejected).

**Not run:** the tests and code of Tasks 1 to 3 and 6 to 9, and the integration in Task 7 and 8. The prototype validated the signals, the conflict procedure and the three-way merge on real data, and the fixtures of Tasks 6 and 7 were run against the current (pre-change) code, which confirmed their baseline behavior.
