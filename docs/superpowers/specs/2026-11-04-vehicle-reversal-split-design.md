# dnt.refine: Splitting Vehicle Tracks at a Reversal (stage `reversal`)

- **Status:** draft for review (rev. 2: addresses [review 2026-11-04 21:04](../../review_2026-11-04-21-04-32.md) / [response](../../response_2026-11-04_21-40-00.md); see §9)
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
- The hand-offs to screen and link (§3.5), including a new `forbidden_cuts` input to link that stops it from joining the two sides of a cut directly. Plus summary, docs and changelog updates.

**Out of scope**
- Changes to the external Miami driver (`run.py`, `3a1-split_reversals.py`, `RAW_TRACK_SUFFIX`). Once this stage lands, refine reads the raw vehicle tracks directly and the `3a1` step can be removed.
- Exporting a labelled video. The `Labeler` already does this from the refined track file.
- Scoring with a band, or routing to the VLM or review (§8).
- Blocking chains of links across a cut (§3.5, last point).
- Running the stage on pedestrians: people do turn around.

**Success criteria**
1. A synthetic vehicle that goes out and back along one line is split into two IDs **at the turn row**, both without noise and with noise on the box centres (§6.1). The script cuts early on such a track (§3.2 step 5), so here the stage deliberately does not match it.
2. Within the scope of §6.3, the stage finds the same reversals as the script's `find_reversals` on the same tracks: one cut in each of the script's runs. Where the cut falls inside a run can differ (§3.2 step 5), and so can the cuts the middle-part rule removes.
3. A queue of stop-and-go cars, a 90° turn, and a parked car with jittering boxes are never split.
4. Link never joins the two sides of a reversal cut directly, whatever the box heights, speeds or link settings, and also after screen or dedup has removed rows near the cut (§3.5). A *chain* through a third track that lies between the two sides is not covered (§3.5, last point).
5. With `reversal.enabled: false`, the refined track file is byte-identical to the one the code wrote before this change. The ledger has identical events, and its header differs only by the new `config.reversal` block and `dnt_version` (§6.2).
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
5. **One cut per run, at the turn:** hit rows with consecutive indices form a run. Each run proposes one cut, at the row with the **largest velocity change** `|v_a − v_b|` (pixels per frame). On ties, the earliest row wins. Runs are taken in frame order.

   **The cosine cannot locate the cut.** When a vehicle moves along one line, every window that reaches across the turn still gives two velocities pointing in exactly opposite directions, so the cosine is −1 on many rows of the run. On an out-and-back line at 10 fps with a 2 s window and the turn at frame 30, rows 23 to 38 all have cosine −1. The script's `argmin` then cuts at row 23, seven frames before the turn, and the new part starts with seven rows of the first vehicle. With 0.5 px of noise on `cy`, the smallest cosine falls at row 33 instead, which depends only on the noise. `|v_a − v_b|` is largest at row 30, in both cases, because only there does each window hold one direction only. Switch already breaks ties between saturated heading rows the same way (`turn` in `_heading_reversal`). **This is the first deliberate difference from the script.** A cut at row `i` is kept only if `i − prev ≥ min_rows` and `n − i ≥ min_rows`, where `prev` is the row index of the previous kept cut (0 for the first). **This is the second deliberate difference from the script:** the script measures `i ≥ MIN_ROWS` from the start of the track, so two runs close together could leave a middle part of only a few rows.

The cut row is the first row of the new part. All later rows, up to the next kept cut, get the new ID.

`find_reversals(frames, boxes, fps, rc) -> list[Reversal]` is a pure function. `Reversal` is a small dataclass: `row`, `frame`, `angle` (degrees, at the cut row), `change` (`|v_a − v_b|` in pixels per frame), `run` (the first and last frames of the run), `speed_before`, `speed_after`, `rows_before`, `rows_after`. `propose_reversal_splits(work, cfg, fps) -> list[Event]` calls it for each track. Like the other stages, it advances the progress bar once per track.

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
| `signals` | `angle`, `change`, `run`, `speed_before`, `speed_after` (box heights per second), `rows_before`, `rows_after`; after it is applied, `new_track` (the work ID of the new part, which the header's `id_map` maps to an output ID, as for switch) |
| `decision` | `AUTO_ACCEPT` |

The event is built with `Event.propose` and decided right away, the same way fill decides its events: it is not routed through `Band`, and no VLM is asked. Its ID has the form `reversal-r0-000001`.

**Key.** `DEFINING_PARAMS[SPLIT] = ("cut_frame",)`. So the key is the SHA-256 of `("reversal", SPLIT, lineage, cut_frame)`. The lineage is made of raw IDs and frame spans, so the key does not change when work IDs are renumbered (parent §4.2). The stage name is part of the key, so a reversal split and a switch split at the same frame never have the same key.

### 3.4 Applying

The kept cuts of a track are applied **latest first**, with `apply_edit(work, ev, new_id=next_track_id(work))`. Each edit moves the rows at or after `cut_frame` to the new ID. Applying latest first gives each part its own ID: for a track with cuts at `c1 < c2`, the rows in `[c1, c2)` get one new ID and the rows `≥ c2` get another. This is the order `_Stages._switch` already uses. Each applied event gets `applied = True` and `signals["new_track"]`.

`_Stages._reversal(work)` returns `(work, split_raw, reversal_cuts)`:
- `split_raw` is the set of raw IDs of every track the stage split.
- `reversal_cuts: dict[int, list[int]]` maps each raw ID to the cut frames of its applied reversal splits. It is keyed by raw ID and frame, not by work ID, so it stays valid after switch splits, screen drops rows, dedup merges and link renumbers within the stage run. `_Stages` keeps it for link (§3.5).

### 3.5 Hand-off to screen and link

- **Screen.** Switch already passes `split_raw` (the raw IDs of tracks it split) to screen as `ScreenContext.split_raw_ids`. Screen uses it to decide whether a reclass hint is *localized*. The reversal stage's `split_raw` is added to the same set before screen runs, so a track split here is treated like a track split by switch.
- **Link: forbidden joins.** The two sides of a cut end and start one frame apart, so link will consider joining them. Link's reversed-heading check (`link.normal_heading_check`) **cannot be relied on** to refuse the join. It measures something different: the last and first `link.heading_seconds` of the two tracks, normalised by `h_end` and `h_start`, which are medians over `motion.height_window` rows. It also returns no heading at all (`None`, so the join is allowed) when either end is slower than `link.heading_min_speed`. The review's counterexample shows it: a box whose height is 10 px for most of the track and 50 px around the turn passes this stage's rule at 2.0 and 0.74 box heights per second (window median 10 px). After the cut, link sees 50 px endpoint heights, gets no heading, and auto-accepts the re-link with score 0.975.

  So link receives `reversal_cuts` and **never forms a candidate that joins the two sides of a reversal cut**. A pair `(di, dj)` is forbidden when, for some raw ID `r` and one of its cut frames `c`, `di.lineage` has a span of `r` that ends before `c` and `dj.lineage` has a span of `r` that starts at or after `c`. The check uses the full lineage, not only the last and first spans. So it holds after screen has dropped rows next to the cut, after switch has cut either side again, and after dedup has merged another track into either side. The rule is applied:
  - in `score_candidates` (scored mode), next to the `keep_splits` / `_resplit` check, before gating, so a forbidden pair never becomes a candidate or an event;
  - in `legacy_link_events` (legacy mode), by marking the pair infeasible in the cost matrix given to `_legacy_matches`, so the assignment can still pick another start for that end.

  `run_link_stage` and `score_candidates` gain a keyword `forbidden_cuts: dict[int, list[int]] | None = None`. When it is `None`, link behaves exactly as before.

  **Why not `link.keep_splits`?** It already refuses to re-join neighbouring halves of one raw ID, but it applies to every switch split as well. Turning it on for vehicles would change link's behaviour for all switch splits, which is outside the scope of this spec (§2). It also lets the halves join when a third track holds a span of the raw ID between them. That is right for an intruder that switch cut out, but wrong across a reversal, where both sides are different vehicles whatever lies between.

  **Not covered: chains.** If link accepts `A → X` and `X → B`, where `A` and `B` are the two sides of a cut and `X` is a third track, `resolve_chains` puts all three in one track. The pairwise rule does not see this. For it to happen, `X` must start after `A` ends and end before `B` starts, within `link.overlap_frames` (2). Right after the cut, `A` and `B` are one frame apart, so `X` can only exist if screen dropped rows at the cut, and it would be a track of a few rows. The risk is accepted for this version, and the limit is written in success criterion 4. A component-level check like dedup's cannot-link pairs is the follow-up if it occurs (§8).

  The `validate()` warnings from rev. 1 are removed: link no longer depends on its heading settings to keep these splits.

- **Dedup.** Dedup only proposes pairs whose frame ranges overlap by `dedup.min_overlap_seconds`. The two sides of a cut have disjoint frame ranges, so dedup never proposes them, and no change is needed.
- **Switch, orphan, fill:** nothing changes. Each part is an ordinary track. A part that is too short for orphan is handled the same way as any other short track.

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

`window` joins the `durations` check in `validate()`. The other rules are added with messages that name the field, in the style of `switch.heading_min_speed`.

**Speed units.** `switch.heading_min_speed` is in pixels per second, while this block, like `link.heading_min_speed`, uses box heights per second. The docs say this explicitly, because the switch field has a similar name.

**Compatibility.** A vehicle config with no `reversal` block gets the defaults, with the stage on, so **vehicle output changes by default**. Setting `reversal: {enabled: false}` gives exactly the output from before this change (success criterion 5). Person configs do not change.

## 6. Testing (`tests/refine/test_reversal.py`, synthetic tracks built in numpy)

### 6.1 Rule (`find_reversals`)
- **Out and back, clean (the review's fixture):** frames `0..59` at 10 fps, centre `(100 + x(t), 100)` with `x = 2t` for `t < 30` and `120 − 2t` after, height 10 px. The hit run is rows 23 to 38, all with cosine −1. The cut must be at **row 30** (frame 30), not row 23, which is the earliest smallest cosine and the script's cut.
- **Out and back, noisy:** the same track with seeded Gaussian noise (σ = 0.5 px) on `cy`. The cut must still be at row 30. (The smallest cosine falls at row 33 with seed 0.)
- **Two reversals:** out, back, out again. Two cuts, which become three IDs once applied.
- **Gap:** a straight track with a 15-frame gap is not split. An out-and-back track with a gap at the turn is split at the first row after the gap.
- **Must not split:**
  - a stop-and-go queue, where the speed stays below `min_speed` around each stop
  - a 90° turn
  - a parked box with ±2 px jitter
  - a reversal whose tail has fewer than `min_rows` rows
  - a track shorter than `2 * min_rows`
- **Middle part:** two runs fewer than `min_rows` rows apart keep only the first cut (the deliberate difference in §3.2 step 5).
- **Ties:** in a run where two rows have the same largest `|v_a − v_b|`, the cut is at the earlier of them.

### 6.2 Stage and pipeline
- `propose_reversal_splits` events have the fields of §3.3. Their decision is `AUTO_ACCEPT`, `algo_score` is 1.0, the IDs are `reversal-r0-…`, and `signals.new_track` is set after the split is applied.
- **Key stability (work IDs):** build the work table with `io.to_work`, keep `raw_id` and the frame spans, and renumber only `work.track` (for example `t → t + 1000`) before calling `propose_reversal_splits`. Every `proposal_key` must be the same as before renumbering.
- **Key identity (raw IDs):** changing the raw ID in the lineage (that is, the input track ID) must change the key. The key is meant to identify the raw track and cut, not to survive changes to it.
- **Link hand-off, variable height (the review's counterexample):** the out-and-back centre path with height 50 px on frames 15 to 30 and 10 px elsewhere (top-left set from the centre so the centre does not move), width 10 px, class 2, score 0.9, no appearance. First, without `forbidden_cuts`, `run_link_stage` must re-link the two sides. This pins the failure, so the test shows the fix is needed. Then, with `forbidden_cuts`, no LINK candidate or event joins the two sides, and `TrackRefiner` with the vehicle defaults writes two output tracks.
- **Link hand-off, shortened sides:** drop the last 3 rows before the cut and the first 3 rows after it (as screen would), then run link with `forbidden_cuts`. The sides are still never joined.
- **Link hand-off, re-split side:** cut the part after the reversal again at a later frame (as switch would). Link may join the two pieces of that part, but never either of them to the part before the reversal.
- **Link hand-off, legacy mode:** with `link.mode: legacy`, the forbidden pair gets no LINK event.
- **Link unchanged without cuts:** with `forbidden_cuts=None` or `{}`, `score_candidates` returns the same candidates as before the change.
- **Screen hand-off:** the raw IDs of a reversal-split track are in `ScreenContext.split_raw_ids`.
- **Disabled:** with `reversal.enabled: false`, the run is compared with a baseline that the **pre-change** code produced on a fixed synthetic vehicle fixture. The plan's first task records it in `tests/refine/data/reversal_baseline/` (the track file and the ledger, which holds the events and the header) before any code changes. The comparison is:
  - the track file: byte-identical;
  - ledger events: identical, compared field by field (IDs, keys, decisions, params, signals, `applied`);
  - ledger header: identical after removing `config.reversal` (which the new config always serializes, since the header stores `cfg.to_dict()`) and `dnt_version` (which can change with a release). No other header field may differ.

  The ledger keeps serializing the full config: a header that records every setting used is worth more than byte-identical old ledgers, so no legacy serialization mode is added.
- **Targets:** `RefineConfig.defaults("person").reversal.enabled` is false, and the vehicle default is true.
- **Config:** each rule in §5 rejects a bad value with a message that names the field.
- **Progress:** the reporter shows `Stage 1/8 reversal` (`2/9` when Prepare is shown).

### 6.3 Parity with the script
The script's `find_reversals` (from `3a1-split_reversals.py`) is copied into the test module as a reference oracle. The oracle is extended to also return each cut's run (its first and last hit row), which needs no change to its logic. On about 200 seeded random tracks (piecewise-linear paths with random turns, speeds, gaps and noise):
- **Same runs:** every stage cut lies in a run that the oracle also cut. Where the oracle's cuts are all at least `min_rows` rows apart, the two sets of runs are equal.
- **Within a run:** the stage's cut is the row with the largest `|v_a − v_b|` in the run. It is not compared with the oracle's frame, because the oracle cuts at the earliest smallest cosine, which is the first deliberate difference (§3.2 step 5).
- **Middle-part rule:** where the oracle returns two cuts fewer than `min_rows` rows apart, the stage keeps a subset of the oracle's runs (the second deliberate difference).

The script's file was not in the workspace when the review ran. Before this test is written, the plan copies `find_reversals` and `_slope` from the pasted script into the test module word for word, with a comment naming the source.

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
- **Link chains across a cut:** not blocked in this version (§3.5). If one is ever seen in the ledger (a LINK chain whose merged lineage holds raw `r` on both sides of one of its cuts), the follow-up is a component check after `resolve_chains` that skips the weakest accepted link on the path, like dedup's cannot-link pairs.
- **Degenerate side:** if all the frames on one side are the same (a track with duplicate rows on one frame), `mean_velocity` would divide by zero. The stage checks for this and gives the row no value, so it is never a hit. `min_side_rows ≥ 2` makes the case rare, but the check does not depend on that.
- **No video, encoder or VLM:** nothing changes. The stage uses none of them.
- **A future banded version:** if hard-rule splits turn out to be wrong in practice (real U-turns, small boxes whose detections drift), a later version can turn the angle and speed margins into a score and route the uncertain band through `Band.of(cfg.reversal)`. The event fields of §3.3 already carry everything such a score would need.

## 9. Revision 2: changes from the review of 2026-11-04 21:04

| Finding | Change |
|---|---|
| **[P1]** Link's heading check does not keep the two sides apart | Reproduced: with the review's variable-height fixture, `run_link_stage` (vehicle defaults, no appearance) gets heading `None` and applies the link `[1, 2]` with `AUTO_ACCEPT` at 0.975. Link now receives `forbidden_cuts`, keyed by raw ID and cut frame, and never forms a candidate that joins the two sides of a cut. The check uses the full lineage, so it holds after screen, switch or dedup edits (§3.5). The rev. 1 `validate()` warnings are removed. Chains through a third track are explicitly out of scope, and success criterion 4 says so. New tests: the counterexample (failing first without the fix), shortened sides, a re-split side, legacy mode, and no change without cuts (§6.2). |
| **[P2]** The earliest smallest cosine does not find the turn | Reproduced: rows 23 to 38 all have cosine −1, so the earliest smallest cosine is row 23. With noise it moves to row 33. The cut is now at the largest `|v_a − v_b|` in the run, which is row 30 in both cases. This is a declared difference from the script (§3.2 step 5). Parity is now about which runs get cut, not the exact frame (§6.3). The clean and noisy fixtures are added (§6.1). |
| **[P2]** A byte-identical ledger is impossible once the config gains a block | The disabled test compares against a baseline recorded with the pre-change code. The track file must be byte-identical and the ledger events equal, and the header may differ only by `config.reversal` and `dnt_version` (§6.2, success criterion 5). |
| **[P2]** Renumbering input IDs changes the lineage and so the key | The key test renumbers only `work.track` after `io.to_work`, keeping `raw_id`. A second test checks that a changed raw ID gives a different key (§6.2). |
