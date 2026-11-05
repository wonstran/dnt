# dnt.refine: Splitting Vehicle Tracks at a Reversal (stage `reversal`)

- **Status:** draft for review
- **Date:** 2026-11-04
- **Parent spec:** [2026-09-27-track-refinement-design.md](2026-09-27-track-refinement-design.md). This spec adds one stage to `dnt.refine`. Anything it does not state follows the parent spec (events §4.1, ledger §4.2, shared primitives §5, config §9). The dedup spec ([2026-10-03-track-dedup-design.md](2026-10-03-track-dedup-design.md)) set the pattern for adding a stage.
- **Origin:** the driver script `3a1-split_reversals.py` from the Miami 20-minute pipeline (`OUT_ROOT/tracks/{stem}_veh_tracks_raw.txt` → `{stem}_veh_tracks_split.txt`). This stage replaces it.

## 1. Problem

Sometimes a vehicle leaves the view and another vehicle comes back along the same path, and the tracker carries the first vehicle's ID over to the second. This happens mostly at the far end of the road, where the boxes are small. The result is one track ID that goes out and comes back: two vehicles.

The switch stage already scores a reversed heading for vehicles (`switch.heading_reversal`, [switch.py](../../../src/dnt/refine/switch.py) `_heading_reversal`). But that score only feeds the motion term `mot`. Switch cuts only when an *opportunity gate* also fires near the row: contact with another box, a gap, a size jump or a class change. Here none of them fires. The ID moves from a box that left to a box that arrives, with no overlap, often no gap, and similar small sizes. So the switch stage never cuts these tracks.

A road vehicle cannot turn around in one or two seconds. For the vehicle target, a sharp reversal of direction between two moving stretches is enough on its own to say the track holds two vehicles.

## 2. Goal and scope

**Goal.** Before any other refine stage runs, split every vehicle track whose direction reverses sharply between two moving stretches. The part after the reversal gets a new track ID. Each split is recorded in the refine ledger.

**In scope**
- A new stage, `reversal`, which runs first, before switch. It writes `SPLIT` events.
- The decision is a hard rule (§3.2): every split that meets the rule is applied, with `AUTO_ACCEPT`. There is no band, no VLM, and no review card.
- A new `ReversalConfig` block. It is on by default for the vehicle target and off for the person target.
- The hand-offs to screen and link (§3.5), plus summary, docs and changelog updates.

**Out of scope**
- Changes to the external Miami driver (`run.py`, `3a1-split_reversals.py`, `RAW_TRACK_SUFFIX`). Once this stage lands, refine reads the raw vehicle tracks directly and the `3a1` step can be removed.
- Exporting a labelled video. The `Labeler` already does this from the refined track file.
- Scoring with a band, or routing to the VLM or review (§8).
- A new cannot-link mechanism in link (§3.5 explains why link's heading check is enough).
- Running the stage on pedestrians: people do turn around.

**Success criteria**
1. A synthetic vehicle that goes out and back along one line is split into two IDs at the turn. The cut is within one row of the frame the script would choose.
2. Within the scope of §6.3, the stage cuts at the same rows as the script's `find_reversals` on the same tracks. The exceptions are the cases the middle-part rule (§3.2 step 5) changes on purpose.
3. A queue of stop-and-go cars, a 90° turn, and a parked car with jittering boxes are never split.
4. Link never joins the two parts of a reversal split again under the vehicle defaults.
5. With `reversal.enabled: false`, the refined output is byte-identical to the output before this change.
6. A `SPLIT` event from this stage has a stable `proposal_key` that does not depend on work-table IDs.

## 3. The stage

### 3.1 Order and contract

```
raw tracks
  reversal → switch → screen → dedup → link → orphan → fill
```

- **First:** switch, screen, dedup and link should all see two separate vehicles, not one track that holds both. Screen judges each part on its own motion, and link judges the part ends, which is what makes its heading check apply (§3.5).
- **No inputs beyond the table:** the stage reads only the work table and `fps`. It needs no video, encoder, context boxes or VLM, so it runs the same with or without them.
- **Observed rows only:** filled rows are already removed when the table is read (parent §2.5).
- **New IDs:** the stage creates track IDs with `apply.next_track_id`, as switch does. It runs before switch, and every later stage that creates IDs takes the next free one, so IDs never collide.

`_Stages.run` gains a `reversal` step before switch, with a `tick("reversal")`. The stage count goes from 7 to 8, or from 8 to 9 when Prepare is shown. The hard-coded `- 7` in `stage()` (`i + rep.n_stages - 7`) is replaced by a constant for the number of stages, so the next stage added does not need another edit there. `Reporter` (`n_stages=7 if store is None else 8`) changes to match.

### 3.2 The rule (new module `refine/reversal.py`)

For one track whose rows are sorted by frame (`frames`, the box centres `cx` and `cy`, and the heights `h`), with `w = to_frames(reversal.window, fps)`:

1. **Too short:** a track with fewer than `2 * min_rows` rows is never split.
2. **Per-row velocities:** for each row `i ≥ 1` at frame `t`, the rows *before* it are those with frames in `[t − w, t)`, and the rows *after* it are those with frames in `[t, t + w)`. Both ranges are found with `searchsorted` on the frame numbers, so gaps are handled. If either side has fewer than `min_side_rows` rows, the row has no value. Each side's velocity is `primitives.mean_velocity(frames, cx, cy)`. This is the same least-squares slope as the script's `_slope`, in pixels per frame.
3. **Speed:** each side's speed is `|v| * fps / max(median(h on that side), 1)`, in box heights per second. If either side is below `min_speed`, the row has no value. A box that barely moves has no reliable heading.
4. **Reversal rows:** a row is a hit when `cos(angle) = (v_b · v_a) / (|v_b| |v_a|) ≤ cos(min_angle)`. Rows without a value are never hits.
5. **One cut per run:** hit rows with consecutive indices form a run. Each run proposes one cut, at the row with the smallest cosine (the sharpest turn). On ties, the earliest row wins (`argmin`). Runs are taken in frame order. A cut at row `i` is kept only if `i − prev ≥ min_rows` and `n − i ≥ min_rows`, where `prev` is the row index of the previous kept cut (0 for the first). **This differs from the script on purpose:** the script measures `i ≥ MIN_ROWS` from the start of the track, so two runs close together could leave a middle part of only a few rows.

The cut row is the first row of the new part. All later rows, up to the next kept cut, get the new ID.

`find_reversals(frames, boxes, fps, rc) -> list[Reversal]` is a pure function. `Reversal` is a small dataclass: `row`, `frame`, `angle` (degrees), `speed_before`, `speed_after`, `rows_before`, `rows_after`. `propose_reversal_splits(work, cfg, fps) -> list[Event]` calls it for each track. Like the other stages, it advances the progress bar once per track.

### 3.3 Event and key

There is one `EventKind.SPLIT` event per kept cut. Its fields match switch's split events, so `apply_edit` and the ledger need no changes:

| Field | Value |
|---|---|
| `stage` | `"reversal"` |
| `tracks` | `[tid]`, the work ID of the track before the cut |
| `lineage` | `[lineage_of_rows(g)]` |
| `frames` | `(cut_frame, cut_frame)` |
| `params` | `{"cut_frame": cut_frame}` |
| `algo_score` | `1.0` |
| `signals` | `angle`, `speed_before`, `speed_after` (box heights per second), `rows_before`, `rows_after`; after it is applied, `new_track` (the work ID of the new part, which the header's `id_map` maps to an output ID, as for switch) |
| `decision` | `AUTO_ACCEPT` |

The event is built with `Event.propose` and decided right away, the same way fill decides its events: it is not routed through `Band`, and no VLM is asked. Its ID has the form `reversal-r0-000001`.

**Key.** `DEFINING_PARAMS[SPLIT] = ("cut_frame",)`. So the key is the SHA-256 of `("reversal", SPLIT, lineage, cut_frame)`. The lineage is made of raw IDs and frame spans, so the key does not change when work IDs are renumbered (parent §4.2). The stage name is part of the key, so a reversal split and a switch split at the same frame never have the same key.

### 3.4 Applying

The kept cuts of a track are applied **latest first**, with `apply_edit(work, ev, new_id=next_track_id(work))`. Each edit moves the rows at or after `cut_frame` to the new ID. Applying latest first gives each part its own ID: for a track with cuts at `c1 < c2`, the rows in `[c1, c2)` get one new ID and the rows `≥ c2` get another. This is the order `_Stages._switch` already uses. Each applied event gets `applied = True` and `signals["new_track"]`.

`_Stages._reversal(work)` returns `(work, split_raw)`. `split_raw` is the set of raw IDs of every track the stage split.

### 3.5 Hand-off to screen and link

- **Screen.** Switch already passes `split_raw` (the raw IDs of tracks it split) to screen as `ScreenContext.split_raw_ids`. Screen uses it to decide whether a reclass hint is *localized*. The reversal stage's `split_raw` is added to the same set before screen runs, so a track split here is treated like a track split by switch.
- **Link.** The two parts of a split end and start one frame apart, so link will consider joining them. Link's existing reversed-heading check refuses that join. It is on for vehicles (`link.normal_heading_check`), it rejects joins with a heading change above `link.max_heading_change = 120°` once both ends move faster than `link.heading_min_speed = 0.2` box heights per second, and it measures heading over `link.heading_seconds = 2.0`. A reversal split has an angle of at least `min_angle = 135°` and speeds of at least `0.5` box heights per second over the same 2 s window, so it is expected to fail link's check. The two measurements are not identical, though. Link measures the end of one track and the start of the next and divides by the end and start box heights (`h_end`, `h_start`), while this stage centres its windows on the cut row and divides by the median height. So the guarantee comes from the margins (135° against 120°, and 0.5 against 0.2 box heights per second), and the pipeline test in §6.2 checks it. No new cannot-link mechanism is added (`excluded` in link means frames that dedup dropped, not pairs that must stay apart).

  To keep that true when the settings change, `RefineConfig.validate()` **logs a warning**, but does not raise, when `reversal.enabled` is true and any of these holds: `link.normal_heading_check` is false; `link.max_heading_change ≥ reversal.min_angle`; or `link.heading_min_speed > reversal.min_speed`. Each of these lets link join what this stage split. A test checks the invariant at the defaults (§6).

- **Switch, dedup, orphan, fill:** nothing changes. Each part is an ordinary track. A short part that is too short for orphan is handled the same way as any other short track.

## 4. Replay

A reversal `SPLIT` has the stable key of §3.3. A recorded decision on that key, for example a `HUMAN_REJECT` from a later `apply` command, is applied by key when the stage runs again, as for every other stage (parent §4.2). The `apply` command itself is still out of scope (P4, on hold). This spec only guarantees that the events are compatible with it. The stage is deterministic, so the same inputs give the same keys.

## 5. Configuration

A new `ReversalConfig` in `RefineConfig` (`cfg.reversal`). It is parsed, overlaid and validated like the other stage blocks: unknown keys raise. The defaults are the values tuned in the script:

| Field | Default | Rule | Meaning |
|---|---|---|---|
| `enabled` | `False`; `True` in `RefineConfig.defaults("vehicle")` | bool | Run the stage. Off for the person target. |
| `window` | `2.0` | seconds, > 0 | Length of the before and after windows of a row. |
| `min_speed` | `0.5` | box heights per second, finite, ≥ 0 | Minimum speed of each side. A value of 0.2 split queued cars. |
| `min_angle` | `135.0` | degrees, 90 < x ≤ 180 | Minimum angle between the two velocities. 180 is a full reversal. |
| `min_rows` | `5` | int ≥ 1 | Rows each part must keep (0.5 s at 10 fps). Counted in rows, as tuned in the script. |
| `min_side_rows` | `3` | int ≥ 2 | Rows each side of a row needs before it has a velocity. A slope needs two frames. |

`window` joins the `durations` check in `validate()`. The other rules are added with messages that name the field, in the style of `switch.heading_min_speed`. The warnings of §3.5 are logged after the checks pass.

**Speed units.** `switch.heading_min_speed` is in pixels per second, while this block, like `link.heading_min_speed`, uses box heights per second. The docs say this explicitly, because the switch field has a similar name.

**Compatibility.** A vehicle config with no `reversal` block gets the defaults, with the stage on, so **vehicle output changes by default**. Setting `reversal: {enabled: false}` gives exactly the output from before this change (success criterion 5). Person configs do not change.

## 6. Testing (`tests/refine/test_reversal.py`, synthetic tracks built in numpy)

### 6.1 Rule (`find_reversals`)
- **Out and back:** a box moves right at 2 box heights per second for 3 s, then left for 3 s (10 fps). One cut, at the turn row, with an angle close to 180°.
- **Two reversals:** out, back, out again. Two cuts, which become three IDs once applied.
- **Gap:** a straight track with a 15-frame gap is not split. An out-and-back track with a gap at the turn is split at the first row after the gap.
- **Must not split:**
  - a stop-and-go queue, where the speed stays below `min_speed` around each stop
  - a 90° turn
  - a parked box with ±2 px jitter
  - a reversal whose tail has fewer than `min_rows` rows
  - a track shorter than `2 * min_rows`
- **Middle part:** two runs fewer than `min_rows` rows apart keep only the first cut (the deliberate difference in §3.2 step 5).
- **Ties:** a run where several rows have the same cosine cuts at the earliest of them.

### 6.2 Stage and pipeline
- `propose_reversal_splits` events have the fields of §3.3. Their decision is `AUTO_ACCEPT`, `algo_score` is 1.0, the IDs are `reversal-r0-…`, and `signals.new_track` is set after the split is applied.
- **Key stability:** renumbering the input track IDs leaves every `proposal_key` the same.
- **Link hand-off:** an out-and-back vehicle run through `TrackRefiner` with the vehicle defaults gives two output tracks, and link does not join them.
- **Screen hand-off:** the raw IDs of a reversal-split track are in `ScreenContext.split_raw_ids`.
- **Disabled:** with `reversal.enabled: false`, the output file and ledger are identical to a run without the stage.
- **Targets:** `RefineConfig.defaults("person").reversal.enabled` is false, and the vehicle default is true.
- **Config:** each rule in §5 rejects a bad value with a message that names the field. The three settings of §3.5 each log a warning.
- **Progress:** the reporter shows `Stage 1/8 reversal` (`2/9` when Prepare is shown).

### 6.3 Parity with the script
The script's `find_reversals` (from `3a1-split_reversals.py`) is copied into the test module as a reference oracle. On about 200 seeded random tracks (piecewise-linear paths with random turns, speeds, gaps and noise), the stage's cut frames must equal the oracle's. The exception is any track where the oracle returns two cuts fewer than `min_rows` rows apart: there the stage must keep a subset of the oracle's cuts, which is the rule of §3.2 step 5.

### 6.4 Existing tests
Refine tests that run the vehicle target and compare exact outputs either set `reversal.enabled: false` or have their expected values updated in the same change. The plan lists which ones. The golden tests (`-m golden`) compare against 0.3.2.4 detection and tracking and are not affected.

## 7. Packaging and documentation
- `docs/api/`: the refine page lists the `reversal` module, and `mkdocs.yml` gets a `nav` entry for it if a page is added.
- The changelog entry under the next version says that vehicle refine output changes by default and how to turn the stage off.
- The `dnt.refine` package docstring lists the stage.
- `ruff check src tests tools` passes with no new baseline entries.

## 8. Errors and degraded modes
- **Empty table, or the stage disabled:** the stage is skipped and the reporter marks it as skipped, like the other stages.
- **Unknown fps:** `fps` is resolved before any stage runs (`resolve_fps`), so the stage always has a positive fps.
- **Degenerate side:** if all the frames on one side are the same (a track with duplicate rows on one frame), `mean_velocity` would divide by zero. The stage checks for this and gives the row no value, so it is never a hit. `min_side_rows ≥ 2` makes the case rare, but the check does not depend on that.
- **No video, encoder or VLM:** nothing changes. The stage uses none of them.
- **A future banded version:** if hard-rule splits turn out to be wrong in practice (real U-turns, small boxes whose detections drift), a later version can turn the angle and speed margins into a score and route the uncertain band through `Band.of(cfg.reversal)`. The event fields of §3.3 already carry everything such a score would need.
