# dnt.refine: Splitting Vehicle Tracks at a Reversal (stage `reversal`)

- **Status:** draft for review (rev. 5: addresses the [review 2026-11-04 21:56](../../review_2026-11-04-21-56-20.md) / [response](../../response_2026-11-04_22-20-00.md), see §12; rev. 4: addresses the [review 2026-11-04 21:42](../../review_2026-11-04-21-42-35.md) / [response](../../response_2026-11-04_22-10-00.md), see §11; rev. 3: addresses the [follow-up review 2026-11-04 21:16](../../review_2026-11-04-21-16-09.md) / [response](../../response_2026-11-04_21-55-00.md), see §10; rev. 2: addresses [review 2026-11-04 21:04](../../review_2026-11-04-21-04-32.md) / [response](../../response_2026-11-04_21-40-00.md); see §9)
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
- The hand-offs to screen, link and dedup (§3.5). Link and dedup receive the cuts as side tags carried by each track (§3.5) and never put the two sides of a cut into one track, directly or through a chain. Plus summary, docs and changelog updates.

**Out of scope**
- Changes to the external Miami driver (`run.py`, `3a1-split_reversals.py`, `RAW_TRACK_SUFFIX`). Once this stage lands, refine reads the raw vehicle tracks directly and the `3a1` step can be removed.
- Exporting a labelled video. The `Labeler` already does this from the refined track file.
- Scoring with a band, or routing to the VLM or review (§8).
- Running the stage on pedestrians: people do turn around.

**Success criteria**
1. A synthetic vehicle that goes out and back along one line is split into two IDs **at the turn row**, both without noise and with noise on the box centres (§6.1). The script cuts early on such a track (§3.2 step 6), so here the stage deliberately does not match it.
2. Within the scope of §6.3, the stage finds exactly the same **hit runs** as the script's `find_reversals` on the same tracks (§3.2 steps 1–5). Which row of a run is cut (step 6), and which cuts survive the length checks (step 7), can differ on purpose.
3. A queue of stop-and-go cars, a 90° turn, and a parked car with jittering boxes are never split.
4. Neither link nor dedup puts the two sides of a reversal cut into one track: not directly, not through a chain of links or merges via a third track, and also after screen, switch or dedup have edited either side, including when dedup replaces every row of one side with a duplicate's rows (§3.5). This holds whatever the box heights, speeds or link settings.
5. With `reversal.enabled: false`, the refined track file is byte-identical to the one the code wrote before this change. The ledger has identical events, and its header differs only by the new `config.reversal` block and `dnt_version` (§6.2).
6. A `SPLIT` event from this stage has a stable `proposal_key` that does not depend on work-table IDs.

## 3. The stage

### 3.1 Order and contract

```
raw tracks
  reversal → switch → screen → dedup → link → orphan → fill
```

- **First:** switch, screen, dedup and link should all see two separate vehicles, not one track that holds both. Screen judges each part on its own motion, and link and dedup are told where the cuts are, so they never join the parts again (§3.5).
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
5. **Runs:** hit rows with consecutive indices form a run. Steps 2 to 5 are identical to the script. The runs they produce, before any cut is chosen or any length check is applied, are what §6.3 compares with the script.
6. **One cut per run, at the turn:** each run proposes one cut, at the row with the **largest velocity change** `|v_a − v_b|` (pixels per frame). On ties, the earliest row wins.

   **The cosine cannot locate the cut.** When a vehicle moves along one line, every window that reaches across the turn still gives two velocities pointing in exactly opposite directions, so the cosine is −1 on many rows of the run. On an out-and-back line at 10 fps with a 2 s window and the turn at frame 30, rows 23 to 38 all have cosine −1. The script's `argmin` then cuts at row 23, seven frames before the turn, and the new part starts with seven rows of the first vehicle. With 0.5 px of noise on `cy`, the smallest cosine falls at row 33 instead, which depends only on the noise. `|v_a − v_b|` is largest at row 30, in both cases, because only there does each window hold one direction only. Switch already breaks ties between saturated heading rows the same way (`turn` in `_heading_reversal`). **This is the first deliberate difference from the script.**
7. **Part lengths:** the proposed cuts are taken in frame order. A cut at row `i` is kept only if `i − prev ≥ min_rows` and `n − i ≥ min_rows`, where `prev` is the row index of the previous kept cut (0 for the first). The check uses the rows chosen in step 6. **This is the second deliberate difference from the script:** the script measures `i ≥ MIN_ROWS` from the start of the track, so two runs close together could leave a middle part of only a few rows.

   Because steps 6 and 7 both differ from the script, the set of runs that end up cut can differ too, even for a track with one reversal. On frames `0..29` with `x = 2t` for `t < 5` and `20 − 2t` after (height 10 px), the run is rows 3 to 9. The script picks row 3 and drops it (3 head rows < 5). The stage picks row 5, which leaves 5 head rows and 25 tail rows, and keeps it.

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
- `reversal_cuts: dict[int, list[int]]` maps each raw ID to the cut frames of its applied reversal splits. It is keyed by raw ID and frame, not by work ID. `_Stages` builds `ReversalSides(cuts=reversal_cuts)` from it and passes that to dedup and link (§3.5).

### 3.5 Hand-off to screen and link

- **Screen.** Switch already passes `split_raw` (the raw IDs of tracks it split) to screen as `ScreenContext.split_raw_ids`. Screen uses it to decide whether a reclass hint is *localized*. The reversal stage's `split_raw` is added to the same set before screen runs, so a track split here is treated like a track split by switch.
- **Link: forbidden joins.** The two sides of a cut end and start one frame apart, so link will consider joining them. Link's reversed-heading check (`link.normal_heading_check`) **cannot be relied on** to refuse the join. It measures something different: the last and first `link.heading_seconds` of the two tracks, normalised by `h_end` and `h_start`, which are medians over `motion.height_window` rows. It also returns no heading at all (`None`, so the join is allowed) when either end is slower than `link.heading_min_speed`. The review's counterexample shows it: a box whose height is 10 px for most of the track and 50 px around the turn passes this stage's rule at 2.0 and 0.74 box heights per second (window median 10 px). After the cut, link sees 50 px endpoint heights, gets no heading, and auto-accepts the re-link with score 0.975.

  **Side tags, not lineage.** Rev. 3 read each track's side of a cut from the raw lineage of the rows it still holds. Dedup breaks that. A nested merge can replace every row of one side with a duplicate's rows, so the merged track no longer holds any row of the cut raw ID. The review of 21:42 reproduces this:
  - `A` is raw 1 on frames 0–29 (score 0.8) and `B` is raw 1 on frames 30–59 (score 0.9), cut at 30.
  - A duplicate, raw 3, has `A`'s boxes and score 0.95.
  - Dedup auto-accepts `[1, 3]` at 1.0, and track 1 is left holding only raw 3.
  - With `link.normal_heading_check = False`, link auto-accepts `[1, 2]` at 0.975, and the lineages `[[3, 0, 29]]` and `[[1, 30, 59]]` show no conflict.

  So which side a track is on is tracked separately from its rows, as a **side tag** that every merge passes on.

  - **Tag.** For a raw ID `r` with sorted cut frames `c_1 < … < c_m`, side `k` is the frames `[c_k, c_{k+1})`, with `c_0 = −∞` and `c_{m+1} = +∞`. A tag is the pair `(r, k)`.
  - **Tags of a track.** `tags(t) = own(t) ∪ carried[t]`.
    - `own(t)` comes from the track's current lineage. Each span `(r, f0, f1)` of a cut raw ID gives `(r, k)` with `k = bisect_right(cuts[r], f0)`. A span never crosses a cut, because the reversal stage split there and nothing later puts rows of both sides back into one span.
    - `carried[t]` holds the tags of every track merged into `t`, even when none of their rows survive.
  - **Conflict.** A set of tags conflicts when it holds `(r, k1)` and `(r, k2)` with `k1 ≠ k2`. The function `side_conflict(tags) -> (r, k1, k2) | None` returns the smallest such `r`, and the two smallest differing `k` for it.
  - **Owner.** `_Stages` keeps a `ReversalSides` object holding `cuts` (raw ID → sorted cut frames) and `carried` (work ID → set of tags). It provides `tags(track, lineage)`, `side_conflict`, and `absorb(mapping, tags_before)`.

    **`absorb` contract.** `tags_before` is a snapshot of `tags(t)` (own ∪ carried) for **every** track on the table before the merges are applied, the representatives included. `mapping` is a flattened absorption map (absorbed work ID → surviving work ID), such as `MergeOutcome.absorbed` or link's `rep_of`. It has no entry for a representative itself. The transfer is:

    ```python
    for track, tags in tags_before.items():
        representative = mapping.get(track, track)
        carried[representative] |= tags
    ```

    A representative keeps its own pre-merge tags this way too. That matters because its own rows can all lose to an absorbed duplicate: in the replacement case below, `mapping = {3: 1}`, `tags_before[1] = {(1, 0)}` and `tags_before[3] = ∅`, so carrying only the absorbed tracks' tags would leave track 1 untagged. Because `tags_before[t]` already includes `carried[t]`, tags carried by an earlier merge (dedup) are kept when a later map (link) is applied. Tags are only ever added. Entries for IDs that no longer exist are harmless and never read.
  - **Effect on keys and evidence.** Tags are never written into lineage. Proposal keys, evidence crops and the ledger lineage keep the observed-row meaning they have today. Each applied split event records the tags of its two parts in `signals.sides`, so a reader can follow which part is which side.

  **How tags pass through the stages.**
  - **Reversal** creates `cuts`. `carried` starts empty: right after the split, every track's tags are its own.
  - **Switch and screen** only split tracks or drop rows. Each piece's tags come from its own lineage. Tags only need carrying where tracks are merged.
  - **Dedup** (§ *Dedup* below) computes the tags of every track on the table *before* it applies merges. `blocker` refuses a join whose union of component tags conflicts, and after applying, `ReversalSides.absorb(out.absorbed, tags_before)` adds each absorbed track's tags to its representative. In the review's case, track 1 gets `carried = {(1, 0)}` from `A`'s tags, computed before `A`'s rows are dropped. Raw 3 has no cuts, so it adds no tags of its own, and `tags(1) = {(1, 0)}`. With `tags(2) = {(1, 1)}`, the pair conflicts and is never a link candidate, whatever the heading settings.
  - **Link** reads `tags` for its pair check and its chain check (below). After link, `_absorb(rep_of)` already composes link's merges into `absorbed`. `ReversalSides.absorb(rep_of, tags_before_link)` is called with a snapshot taken from the work table link received, for every track. So a later joining step would see the tags too. (No stage after link joins tracks today.)

  **Link, pairs.** Link receives the `ReversalSides` and never forms a candidate `(di, dj)` for which `side_conflict(tags(di) ∪ tags(dj))` is not `None`:
  - in `score_candidates` (scored mode), next to the `keep_splits` / `_resplit` check, before gating, so a forbidden pair never becomes a candidate or an event;
  - in `legacy_link_events` (legacy mode), by marking the pair infeasible in the cost matrix given to `_legacy_matches`, so the assignment can still pick another start for that end.

  **Link, chains.** The pair rule cannot see a chain. The review of 21:16 shows one that needs no removed rows. `A` holds frames 0–29 and `B` holds frames 30–59 of one raw ID, cut at 30, and `X`, a different raw ID, holds matching boxes on frames 28–31. Production link, with vehicle defaults and no appearance, auto-accepts `A → X` and `X → B` through the overlap gate, both at 1.0, and `resolve_chains` merges all three tracks into one.

  So the accepted links also go through `resolve_reversal_chains(active, tags_of)`, in both modes: after `resolve_chains` in scored mode, and on the accepted events in legacy mode. `tags_of` maps each work ID that appears in `active` to its tags, computed once before the call. The check repeats until there is no conflict:
  1. **Graph.** Build an undirected graph whose nodes are track IDs and whose edges are the active links, with each edge storing its event. Its connected components are the chains.
  2. **Order of components.** Visit components in increasing order of their smallest member track ID. For each one, compute `side_conflict` of the union of its members' tags. Stop at the first component that conflicts, with conflict `(r, k1, k2)`.
  3. **The two members.** `u` is the smallest track ID whose tags contain `(r, k1)`, and `v` is the smallest whose tags contain `(r, k2)`. `u ≠ v`, because one track never holds two sides: it would conflict on its own, and that cannot happen. The reversal stage gives each part one side, and dedup and link refuse any join that conflicts.
  4. **The path.** Find the path from `u` to `v` by breadth-first search. Neighbours are visited in increasing track ID, so the path is deterministic. Each step follows an actual accepted link, whatever the tracks' first frames are. If two links join the same pair of tracks (one in each direction), the step uses the weaker of them.
  5. **Drop.** Drop the weakest event on that path, by `(algo_score, proposal_key)`. It stays accepted but gets `applied = False`, `skipped_reason = "reversal"`, and `conflicts_with = {"raw": r, "sides": [k1, k2], "tracks": [u, v]}`. Then repeat from step 1.

  Each round removes one active link, so the loop ends. Only links on the `u`–`v` path can be dropped, so a link elsewhere in the component, unrelated to the cut, is never removed for it.

  The review of 21:42 showed why first-frame order was wrong in rev. 3: in the chain `A → X → Y → B` (frames 0–28, 30–31, 29–32, 33–60), `Y` starts before `X`. So the rev. 3 path for the conflicting members `X` and `B` was empty, and the resolver could not progress. Breadth-first search finds `X → Y → B`.

  `run_link_stage` and `score_candidates` gain a keyword `reversal_sides: ReversalSides | None = None`. When it is `None` or has no cuts, link behaves exactly as before, and the chain check is not run.

  **Why not `link.keep_splits`?** It already refuses to re-join neighbouring halves of one raw ID, but it applies to every switch split as well. Turning it on for vehicles would change link's behaviour for all switch splits, which is outside the scope of this spec (§2). It also lets the halves join when a third track holds a span of the raw ID between them. That is right for an intruder that switch cut out, but wrong across a reversal, where both sides are different vehicles whatever lies between.

  The `validate()` warnings from rev. 1 are removed: link no longer depends on its heading settings to keep these splits.

- **Dedup.** Dedup never proposes the two sides as a pair, because it needs frame ranges that overlap by `dedup.min_overlap_seconds` and the two sides have disjoint ranges. A *component* can still join them. A long duplicate track `X` that co-moves with `A` before the cut and with `B` after it can produce two accepted `MERGE` events, `A–X` and `X–B`. `apply_merges` gains a keyword `reversal_sides`. Before it walks the edges, it computes `tags_before[t]` for every track. `blocker(ra, rb)` gains a fourth reason, `"reversal"`, which fires when `side_conflict` of the union of the tags of all members of both components is not `None`. The check is on the components, not on pairs, so the tags of tracks already absorbed count too. The edge is then skipped with `skipped_reason = "conflict"` and `conflicts_with = {"tracks": [u, v], "why": "reversal", "proposal_key": None}`, where `u` and `v` are the smallest member IDs holding the two sides. After applying, `_Stages` calls `reversal_sides.absorb(out.absorbed, tags_before)`, so a side survives even when all of its rows were dropped (the replacement case above). With `reversal_sides` set to `None` or without cuts, nothing changes.
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
- **Middle part:** two cuts fewer than `min_rows` rows apart keep only the first cut (the deliberate difference in §3.2 step 7).
- **Near the start (the follow-up review's fixture):** frames `0..29`, height 10 px, constant `cy`, `x = 2t` for `t < 5` and `20 − 2t` after. The hit run must be rows 3 to 9, and the cut must be at **row 5** and kept. The script would pick row 3 and drop it.
- **Near the end:** frames `0..29`, `x = 2t` for `t < 25` and `100 − 2t` after. The hit run must be rows 23 to 27, and the cut must be at **row 25** and kept (5 tail rows).
- **Ties:** in a run where two rows have the same largest `|v_a − v_b|`, the cut is at the earlier of them.

### 6.2 Stage and pipeline
- `propose_reversal_splits` events have the fields of §3.3. Their decision is `AUTO_ACCEPT`, `algo_score` is 1.0, the IDs are `reversal-r0-…`, and `signals.new_track` is set after the split is applied.
- **Key stability (work IDs):** build the work table with `io.to_work`, keep `raw_id` and the frame spans, and renumber only `work.track` (for example `t → t + 1000`) before calling `propose_reversal_splits`. Every `proposal_key` must be the same as before renumbering.
- **Key identity (raw IDs):** changing the raw ID in the lineage (that is, the input track ID) must change the key. The key is meant to identify the raw track and cut, not to survive changes to it.
- **Link hand-off, variable height (the first review's counterexample):** the out-and-back centre path with height 50 px on frames 15 to 30 and 10 px elsewhere (top-left set from the centre so the centre does not move), width 10 px, class 2, score 0.9, no appearance. The test has two parts:
  - *Isolated link regression.* The test splits the track **by hand at frame 23**, the cut that rule's earliest smallest cosine produced and the one the review re-linked. It does not take the cut from the stage. Without `reversal_sides`, `run_link_stage` must re-link the two sides (`AUTO_ACCEPT`, 0.975, heading `None`), which shows the guard is needed. With `reversal_sides` with cuts `{1: [23]}`, no LINK candidate or event joins them.
  - *Rule and pipeline.* On the same fixture, `find_reversals` must find the hit run rows 23 to 25 and cut at **row 25** (the largest velocity change). `TrackRefiner` with the vehicle defaults must write two output tracks, split at frame 25. With that cut, link proposes no event even without the guard: its heading check already refuses the join. So this part checks the stage's own output, not the guard.
- **Link hand-off, shortened sides:** drop the last 3 rows before the cut and the first 3 rows after it (as screen would), then run link with `reversal_sides`. The sides are still never joined.
- **Link hand-off, re-split side:** cut the part after the reversal again at a later frame (as switch would). Link may join the two pieces of that part, but never either of them to the part before the reversal.
- **Link hand-off, legacy mode:** with `link.mode: legacy`, the forbidden pair gets no LINK event.
- **Link chain (the follow-up review's fixture):** the clean out-and-back path with height and width 10 px, class 2, score 0.9. `A` is raw 1 on frames 0 to 29 and `B` is raw 1 on frames 30 to 59, with cuts `{1: [30]}` in `reversal_sides`. `X` is raw 2, with the same boxes on frames 28 to 31. No rows are removed and there is no appearance.
  - Without `reversal_sides`, both `A → X` and `X → B` are applied and merging leaves one track (the failure, pinned).
  - With `reversal_sides`, exactly one of the two is applied. The other has `skipped_reason = "reversal"`, and merging leaves `A` and `B` in different tracks.
  - Each link is reported against the proposal key that decided the tie.
- **Link chain, path not in first-frame order (the review of 21:42):** constant-velocity boxes, 10 × 10 px, class 2, vehicle defaults. Tracks: `A` (work 1, frames 0–28, raw 1), `X` (work 2, frames 30–31, raw 10), `Y` (work 3, frames 29–32, raw 3) and `B` (work 4, frames 33–60, raw 10). `reversal_sides` has cuts `{10: [32]}`, so `X` has tag `(10, 0)` and `B` has `(10, 1)`. The test passes the accepted events `A → X`, `X → Y` and `Y → B` straight to `resolve_reversal_chains`. It checks three things: `resolve_chains` skips none of them; the conflict is `(10, 0, 1)` between `u = 2` and `v = 4`; and the path found is `X → Y → B`. The dropped event is the weaker of `X → Y` and `Y → B` (the test gives them distinct scores), and `A → X` stays applied.
- **Link chain, unrelated link kept:** the same fixture plus a track `Y` linked after `B`. Only a link on the `A … B` path is dropped, and `B → Y` stays applied.
- **Link chain, legacy mode:** the chain fixture with `link.mode: legacy` (built so that `_legacy_matches` pairs `A → X` and `X → B`) leaves `A` and `B` apart.
- **Dedup replacement, then link (the review of 21:42):** the clean out-and-back path, 10 × 10 px, class 2. `A` is work 1, raw 1, frames 0–29, score 0.8. `B` is work 2, raw 1, frames 30–59, score 0.9. Cut at 30. Work 3 is raw 3, with `A`'s boxes and frames and score 0.95. The config is the vehicle defaults **with `link.normal_heading_check = False`**, so only the cut protection can refuse the join. Dedup must still auto-accept and apply `[1, 3]`, after which track 1 holds only raw 3, and `ReversalSides.carried[1]` must contain `(1, 0)`. With `reversal_sides`, link must form no candidate `[1, 2]`. Without it, link must accept `[1, 2]` at 0.975 (the failure, pinned).
- **Side tags (unit):** `own` tags from lineage, including several cuts of one raw ID; `side_conflict` ordering (smallest `r`, then the two smallest `k`); and `absorb` composing over a dedup map followed by a link map (a tag carried by dedup survives the link map). Also the minimal case `mapping = {3: 1}`, `tags_before = {1: {(1, 0)}, 3: ∅}`: after `absorb`, `tags(1)` must contain `(1, 0)` while track 1's own lineage gives no tag.
- **Dedup component:** a track `X` that co-moves with `A` for 1.5 s before the cut and with `B` for 1.5 s after it (interleaved observations, so both `MERGE` events are accepted). With `reversal_sides`, one edge is skipped with `why = "reversal"`, and `A` and `B` stay apart. Without it, all three merge (the failure, pinned).
- **Link unchanged without cuts:** with `reversal_sides=None` or with no cuts, `score_candidates` returns the same candidates as before the change.
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
The script's `find_reversals` (from `3a1-split_reversals.py`) is copied into the test module as a reference oracle. Parity is checked on the **raw hit runs**, before either side chooses a cut or applies a length check. Kept cuts are not comparable: moving a cut inside a run (step 6) changes which head, tail and middle parts pass `min_rows` (step 7), even for a track with one reversal (the near-start example in §3.2).

The oracle is split at the point where the script computes `hit = np.flatnonzero(cos <= cos_max)`. A helper `oracle_hit_rows(g, fps)` runs the script's code up to that line, unchanged, and returns `hit`. The stage exposes the same intermediate as `reversal.hit_rows(frames, boxes, fps, rc)`, which `find_reversals` calls. On about 200 seeded random tracks (piecewise-linear paths with random turns, speeds, gaps and noise):
- **Same hit rows:** `hit_rows` equals `oracle_hit_rows` exactly, and so the runs are equal too.
- **Cut inside each run:** for each run, the proposed cut is the row with the largest `|v_a − v_b|` in it (ties: earliest). This is computed from the stage's own per-row values. It is not compared with the oracle's choice.
- **Length checks:** the kept cuts are exactly the proposed cuts that pass the step 7 check, evaluated in frame order on the *stage's* rows, with `prev` updated only by kept cuts. The test checks this with an independent ten-line reference implementation of step 7.
- **Kept runs may differ:** the test does not compare kept cuts with the oracle. It counts the tracks where they differ and prints the count for the record.

**A spacing fixture.** The review also asks for a case where localization changes the spacing between adjacent cuts. A search for one ([response 2026-11-04 21:55](../../response_2026-11-04_21-55-00.md)) covered 40,000 random two-turn tracks (2-D paths with random speeds, headings and turn spacing) and a sweep of clean out-back-out lines with turns 4 to 29 frames apart. It found no track where both selections pass the head and tail checks on their own and the spacing check then disagrees. On clean lines, both selections move each cut by the same amount, so the spacing is unchanged (turns at 20 and 32 give old cuts 13 and 33 and new cuts 16 and 36). The step 7 reference check above covers spacing whatever cuts the stage picks, so no fixture is pinned for it. If the plan's random tracks find one, the plan pins it.

The script's file was not in the workspace when the review ran. Before this test is written, the plan copies `find_reversals` and `_slope` from the pasted script into the test module word for word, with a comment naming the source.

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
- **Links or merges that a cut blocks:** these are recorded, not lost. A LINK skipped by the chain check has `skipped_reason = "reversal"`, and a MERGE skipped by dedup has `conflicts_with.why = "reversal"`. Both stay in the ledger with `applied = False`.
- **Two reversals closer together than `window`:** each before and after window then holds both turns, so the velocities mix and the cuts move away from the turns. On a clean line with turns at frames 20 and 30 (2 s window), the runs are rows 13 to 18 and 33 to 38, and the cuts land at rows 15 and 35. A vehicle ID that reverses twice within two seconds is not the case this stage targets (§1), so this is documented, not handled.
- **Degenerate side:** if all the frames on one side are the same (a track with duplicate rows on one frame), `mean_velocity` would divide by zero. The stage checks for this and gives the row no value, so it is never a hit. `min_side_rows ≥ 2` makes the case rare, but the check does not depend on that.
- **No video, encoder or VLM:** nothing changes. The stage uses none of them.
- **A future banded version:** if hard-rule splits turn out to be wrong in practice (real U-turns, small boxes whose detections drift), a later version can turn the angle and speed margins into a score and route the uncertain band through `Band.of(cfg.reversal)`. The event fields of §3.3 already carry everything such a score would need.

## 9. Revision 2: changes from the review of 2026-11-04 21:04

| Finding | Change |
|---|---|
| **[P1]** Link's heading check does not keep the two sides apart | Reproduced: with the review's variable-height fixture, `run_link_stage` (vehicle defaults, no appearance) gets heading `None` and applies the link `[1, 2]` with `AUTO_ACCEPT` at 0.975. Link now receives `reversal_sides`, keyed by raw ID and cut frame, and never forms a candidate that joins the two sides of a cut. The check uses the full lineage, so it holds after screen, switch or dedup edits (§3.5). The rev. 1 `validate()` warnings are removed. Chains through a third track are explicitly out of scope, and success criterion 4 says so. New tests: the counterexample (failing first without the fix), shortened sides, a re-split side, legacy mode, and no change without cuts (§6.2). |
| **[P2]** The earliest smallest cosine does not find the turn | Reproduced: rows 23 to 38 all have cosine −1, so the earliest smallest cosine is row 23. With noise it moves to row 33. The cut is now at the largest `|v_a − v_b|` in the run, which is row 30 in both cases. This is a declared difference from the script (§3.2 step 5). Parity is now about which runs get cut, not the exact frame (§6.3). The clean and noisy fixtures are added (§6.1). |
| **[P2]** A byte-identical ledger is impossible once the config gains a block | The disabled test compares against a baseline recorded with the pre-change code. The track file must be byte-identical and the ledger events equal, and the header may differ only by `config.reversal` and `dnt_version` (§6.2, success criterion 5). |
| **[P2]** Renumbering input IDs changes the lineage and so the key | The key test renumbers only `work.track` after `io.to_work`, keeping `raw_id`. A second test checks that a changed raw ID gives a different key (§6.2). |

## 10. Revision 3: changes from the follow-up review of 2026-11-04 21:16

| Finding | Change |
|---|---|
| **[P2]** Pin the historical cut in the variable-height regression | Reproduced: on that fixture the revised rule's hit run is rows 23 to 25 and it cuts at 25. With a cut at 25, production link proposes no event; with the hand-made cut at 23 it re-links at 0.975. The test now has two parts: an isolated link regression that splits **by hand at 23** and tests the guard, and a rule-and-pipeline part that asserts the cut at **25** and two output tracks (§6.2). |
| **[P2]** Check parity before cut-dependent length filtering | Reproduced: the near-start fixture has hit run 3 to 9; the old choice (3) is dropped and the new choice (5) is kept. Parity is now equality of the raw hit rows (`hit_rows` against the script's `hit` array), checked before any cut is chosen. The cut choice and the length check are tested on the stage's own rows against an independent reference (§6.3). §3.2 now separates runs (step 5), choosing the cut (step 6) and the length checks (step 7), and success criterion 2 is restated. The near-start fixture and a near-end one are pinned (§6.1). No fixture where localization changes the spacing between adjacent cuts was found; the search and why no fixture is pinned are recorded in §6.3. |
| **[P2]** The chain limitation does not need dropped rows | Reproduced: with untouched sides, `X` on frames 28 to 31 gets `A → X` and `X → B` auto-accepted through the overlap gate at 1.0, and merging leaves one track. Because the premise for leaving chains out was wrong, chains are **brought into scope** instead of re-documented. A shared `straddles` predicate drives three checks: link pairs, a link chain check after `resolve_chains` that drops the weakest link on the path between the two sides (`skipped_reason = "reversal"`), and a fourth reason in dedup's `blocker` for merge components. Success criterion 4 now covers chains. New tests: the review's chain fixture, an unrelated link that must survive, legacy mode, and a dedup component (§6.2). §8 also records a limitation found while checking this: reversals closer together than `window` are cut away from the turns. |

## 11. Revision 4: changes from the review of 2026-11-04 21:42

| Finding | Change |
|---|---|
| **[P1]** Dedup can remove every row of one side, and the side goes with them | Reproduced with production `propose_merges`, `apply_merges` and `run_link_stage`: dedup accepts `[1, 3]` at 1.0, track 1 is left holding only raw 3, and with `link.normal_heading_check = False` link accepts `[1, 2]` at 0.975. A track's side of a cut is now a **side tag** `(r, k)`. It comes from the track's own lineage plus the tags `carried` from every track merged into it, kept in `ReversalSides`. Dedup unions the tags into the representative after applying merges, link's merges do the same, and tags never enter lineage, keys or evidence. `straddles` is replaced by `side_conflict` over tags (§3.5). New tests: the replacement case run through dedup and then link, with the heading check off, plus unit tests for the tags (§6.2). |
| **[P2]** Find the chain path by graph traversal, not first-frame intervals | Accepted. `resolve_reversal_chains` builds an undirected graph of the active links, visits components in order of their smallest member, picks the conflicting members `u` and `v` deterministically, finds the `u`–`v` path by breadth-first search with neighbours in ID order, and drops the weakest event on that path (§3.5). The review's chain whose path is not in first-frame order is a test (§6.2). |

## 12. Revision 5: changes from the review of 2026-11-04 21:56

| Finding | Change |
|---|---|
| **[P2]** `absorb` must also keep the representative's own pre-merge tags | Accepted. The `absorb` contract (§3.5) now takes a snapshot of every track's tags before the merges, the representatives included, and carries each one into `mapping.get(t, t)`. So a representative whose own rows all lost to a duplicate keeps its side, and tags carried by dedup survive the link map. The minimal case (`{3: 1}`, `tags_before[1] = {(1, 0)}`, `tags_before[3] = ∅`) is a unit test (§6.2). The design is otherwise unchanged. |
