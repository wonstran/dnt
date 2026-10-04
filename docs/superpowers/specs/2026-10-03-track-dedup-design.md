# dnt.refine: Merging Interleaved Duplicate Tracks (stage `dedup`)

- **Status:** draft for review (rev. 4: measured counts, calibrated defaults and one rule amendment found while prototyping the plan, see §12; rev. 3: addresses [review 2026-10-03 21:24](../../review_2026-10-03_21-24-58.md) / [response](../../response_2026-10-03_22-10-00.md); rev. 2: addresses [review 2026-10-03 21:13](../../review_2026-10-03_21-13-21.md) / [response](../../response_2026-10-03_21-45-00.md))
- **Date:** 2026-10-03
- **Parent spec:** [2026-09-27-track-refinement-design.md](2026-09-27-track-refinement-design.md). This adds one stage to `dnt.refine`. Everything not stated here follows the parent spec (events §4.1, ledger §4.2, bands §4.3, shared primitives §5, review §8.1, config §9).
- **Evidence:** `A1A_Collins Ave & 17th St` clips, pedestrian target (§1).

## 1. Problem

The tracker sometimes alternates IDs on one person from frame to frame, so two or three IDs each cover part of the same trajectory over the same span. Stages 1–3 never see it as an error: link treats track ends and starts as sequential fragments, switch looks inside one track, and screen compares tracks with context. Stage 4 then makes it visible. Fill interpolates each ID across its own gaps, and those gaps are the frames where another ID had the observation. The output holds overlapping duplicate boxes on one person.

Measured on the evening clip (`..._evening_1900_190000.00_ped_track.txt`), P3 output IDs 21, 22 and 24, from raw tracks 122, 125 and 134. The span is the overlap of the two tracks' frame ranges; `n_x` counts a track's observed rows inside it:

| Pair (raw IDs) | Span (frames) | `n` of each (earlier track first) | Shared observed frames | Mean IoU on the shared frames | Shared frames after fill | Mean IoU after fill |
|---|---|---|---|---|---|---|
| 122 / 125 | 183 | 76 / 107 | 7 | 0.00 | 87 | 0.49 |
| 122 / 134 | 136 | 13 / 89 | 0 | not defined | 54 | 0.48 |
| 125 / 134 | 102 | 30 / 13 | 6 | 0.13 | 41 | 0.41 |

The counts were measured with a prototype of the signals in §3.2 on the raw track file (fps 10). A viewer of frames 700 to 820 confirmed that the three tracks are one person. The plan re-measures them with the final code and records them in the test fixtures (§9).

The shared frames of 122/125 are worth noticing: on those 7 frames both IDs are observed with boxes that do not overlap at all (IoU 0.00), and the pair is still one person's trajectory. The stage must therefore tolerate a *small* number of simultaneous observations (detector noise, or flicker that lands on one frame twice), and must refuse pairs that are observed together *often*.

No link or split event touched these tracks. The only events on them are `FILL`.

A crude scan of the raw files (two tracks overlapping at least 30 frames, one's observed boxes against the other's interpolated path, mean IoU above 0.3) found 3 candidate pairs in `beginning`, 12 in `evening` and 7 in `middle`. That scan also flags real people walking side by side (for example evening 257/259 share 36 observed frames, middle 266/270 share 50), which is why the stage needs an occupancy veto (§3.2).

## 2. Goal and scope

**Goal.** Consolidate each group of interleaved duplicate tracks into one track before fill, keep every real observation, and leave people who walk side by side alone.

**In scope**
- A new stage, `dedup`, between screen and link, with a new event kind `MERGE`.
- Detection from motion, with appearance as a secondary signal when a video is given.
- Banded routing like the other stages. The uncertain band becomes `HUMAN_PENDING` (no VLM in this version).
- Ledger, review-page, config, docs and changelog support.
- Composed output-ID mapping for merged and linked tracks (§5), because the review needs it.

**Out of scope**
- **Simultaneous duplicate detections.** Two tracks that are observed on the same frames with overlapping boxes (one person detected twice at once) are not interleaved duplicates, and the stage does not handle them: dense co-observation always vetoes a merge (§3.2). That is a detector or NMS problem.
- A VLM prompt and evidence image for `MERGE` (§8).
- Changes to link, fill or the existing stages, apart from the hand-off in §3.6.
- Any tuning for vehicles beyond running the same stage with the same config (§7).

**Success criteria**
1. On the evening clip, raw tracks 122, 125 and 134 (one person, confirmed on video) become one track under ID 122, and the P3-style output has no overlapping boxes among them.
2. Two people walking in parallel are never merged, whether their boxes are disjoint or overlapping (for example IoU 0.5 on every frame), as long as both are observed on most of the same frames.
3. No real observation is lost: every observed row of the merged tracks is in the output, except the rows listed in the ledger as dropped (§3.5), one per frame where several tracks had a row. A dropped row is also excluded from everything computed downstream of the merge (appearance samples, evidence images), not only from the table (§3.5).
4. A component of merged tracks never contains a pair that is densely co-observed or rejected, **whether or not that pair was eligible for a proposal** (§3.5).
5. Replay compatibility: a `MERGE` event has a stable `proposal_key` that does not depend on work-table IDs (§3.3, §4).

## 3. The stage

### 3.1 Order

```
raw tracks
  1 switch → 2 screen → dedup → 3 link → orphan → 4 fill
```

- **After screen:** false tracks are dropped before anything is merged into them.
- **Before link:** link sees one consolidated track instead of interleaved fragments that it could link to each other.
- **Before fill:** fill interpolates over final identities, which is where the duplicates came from.

`dedup` follows the parent spec's stage contract: it proposes events from the previous stage's applied table, routes them (§4.3 of the parent), applies the accepted ones, and passes the table on. `_Stages.run` gains a `dedup` step and a `tick("dedup")`; the progress bar total changes from 5 to 6. Dedup and link create no new track IDs (only switch does, before dedup), so an absorbed ID is never reused.

### 3.2 Candidates and signals

Descriptors are built from **observed rows only** (filled rows were already removed on read, parent §2.5). For each track: sorted observed frames, boxes, class, and linear interpolation of the box over the track's own span.

For a pair `(a, b)`, let `[lo, hi]` be the intersection of the two tracks' frame ranges. Let `O_a` and `O_b` be the sets of frames inside `[lo, hi]` on which each track has an observed row, `n_a = |O_a|`, `n_b = |O_b|`.

A pair is a candidate when:
- its classes are in the same class group (`link.class_groups`; for a person target, the same class);
- `hi - lo + 1 >= dedup.min_overlap_seconds * fps`;
- `n_a >= dedup.min_observed` and `n_b >= dedup.min_observed` (so `min(n_a, n_b) > 0` always holds below);
- it passes the **occupancy gate** below.

Signals, all over `[lo, hi]`:

| Signal | Definition |
|---|---|
| `shared` | `|O_a ∩ O_b|`: frames observed in both tracks. |
| `co_occupancy` | `shared / min(n_a, n_b)`: the share of the sparser track's observed frames on which the other track is also observed. It is `0.0` when `shared == 0`. It does not depend on box IoU. Example from §1: 122/125 is `7 / 76 = 0.09` (both counts measured). 125/134 is `6 / min(30, 13) = 0.46` (measured): above the original `cooccur_hi` of 0.40, below the calibrated 0.50 (§6). |
| `comotion` | The mean over both directions of the IoU between one track's observed box and the other track's interpolated box on that frame (frames in `O_a` against `b`'s interpolation, frames in `O_b` against `a`'s). Both directions, so one dense track cannot hide a mismatch. |
| `appearance` | Mean cosine similarity between the two tracks' embeddings (the same embeddings link uses), or `None` without a video or encoder. |

**Occupancy gate.** A pair with `co_occupancy >= dedup.cooccur_hi` is **not a candidate**: the tracks are observed together too often to be one flickering identity. This holds whatever the boxes look like: disjoint boxes (two people apart), overlapping boxes at IoU 0.5 (two people close together), and boxes at IoU 0.9 (a simultaneous duplicate detection, out of scope, §2). The gate depends only on observation counts, so overlapping parallel boxes cannot pass it by having IoU above `distinct_iou`, as they could under an IoU-based test.

Score, for pairs that pass the gate:

```
S_dedup = comotion_ramp(comotion) * (1 - cooccur_ramp(co_occupancy)) * appearance_term
```

- `comotion_ramp` rises from 0 at `dedup.comotion_lo` to 1 at `dedup.comotion_hi` (the parent's ramp primitive, §5.4).
- `cooccur_ramp` rises from 0 at `dedup.cooccur_lo` to 1 at `dedup.cooccur_hi`. A pair with `co_occupancy <= cooccur_lo` is not penalized; between the two it is penalized in proportion. So 122/125 (0.09) is not penalized. 125/134 (0.46) is a candidate at the calibrated defaults, penalized almost fully, and its score (about 0.02) auto-rejects it; because an auto-reject does not block a join (§3.5, rule 1), 125 and 134 still end in one track through 122.
- `appearance_term` is 1.0 when `appearance` is `None`. Otherwise it is a ramp from `dedup.appearance_floor` (at similarity `app_lo`) to 1.0 (at `app_hi`). Appearance can lower a score but, with `appearance_floor` above 0, cannot veto a pair that motion strongly supports. This follows the way link treats appearance as a secondary signal.

The defaults for the `comotion_*`, `cooccur_*` and `app_*` knobs are chosen in the plan against the evening tracks (positive) and the side-by-side pairs (negative), and recorded in the config table in §6.

### 3.3 Event and key

`EventKind.MERGE` is added. One event per candidate pair.

| Field | Value |
|---|---|
| `tracks` | `[a, b]` with `a` the **representative**: the track whose first observed frame is earlier, ties broken by work ID. This order only chooses the surviving ID (§3.5). It plays no part in the key. |
| `lineage` | The raw spans of both tracks, **in `tracks` order**: `lineage[0]` belongs to `tracks[0]` and `lineage[1]` to `tracks[1]`, the convention link events and `EvidenceBuilder` already use (parent §4.1). It is never reordered. |
| `frames` | The overlap span `[lo, hi]`. |
| `params` | `{"span": [lo, hi]}` |
| `algo_score` | `S_dedup` |
| `signals` | `shared`, `n_a`, `n_b`, `co_occupancy`, `comotion`, `appearance`: numerators and denominators included, so a reader can recompute `co_occupancy`. `n_a` and `n_b` correspond to `tracks[0]` and `tracks[1]`. After application: `applied`-related fields (§3.5). |
| `edit` | `{"kind": "MERGE", "params": {"span": [lo, hi]}}`, filled when accepted. |

**Key lineage.** `proposal_key` hashes the lineage it is given, in that order, and does not sort. The key of a `MERGE` must not depend on work IDs or on which track is the representative, so it is computed from a **canonical copy**: the two lineage entries sorted by their first span `(raw_id, f0, f1)`. Two tracks cannot share a first raw span, so the order is total. `Event.propose` gains an optional keyword `key_lineage`: when given, the key is computed from it and the event keeps the `lineage` it was given. `dedup.py` passes the track-order lineage as `lineage` and the canonical copy as `key_lineage`. So the stored `tracks`, `lineage`, `n_a` and `n_b` always correspond position by position, and consumers need no extra permutation.

`DEFINING_PARAMS[MERGE] = ("span",)`, so the key is the SHA-256 of `(stage, MERGE, key lineage, span)`. It survives renumbering of work IDs, as parent §4.2 requires, including for two tracks that start on the same frame, where the representative order follows work IDs and may reverse.

### 3.4 Routing

The stage's band is `Band.of(cfg.dedup)`. `route_without_vlm` already maps the uncertain band to `HUMAN_PENDING`, and `route_with_vlm` would ask the VLM; `dedup` always uses the no-VLM router in this version, even when a VLM backend is configured. The review page gets a card with controls for each pending `MERGE`, and a read-only card for each accepted merge that conflicted (§5).

### 3.5 Applying accepted merges

Accepted `MERGE` events are applied by a deterministic procedure that keeps every component free of vetoed or rejected pairs.

**Cannot-link pairs.** A pair `(x, y)` of tracks may never end up in one component when any of these holds:
1. a `MERGE` event for `(x, y)` is decided `VLM_REJECT` or `HUMAN_REJECT`: an explicit judgment (the human one comes from a recorded decision when the stage re-runs under `apply`). `AUTO_REJECT` and `HUMAN_PENDING` events constrain nothing: a low score for a pair of sparse tracks is weak evidence when each of them matches a third track (125/134 scores about 0.02 yet both match 122), and the dense co-observation in rule 2 is the hard evidence against identity;
2. `(x, y)` are **densely co-observed**: over the overlap of their frame ranges (at least one frame), `shared >= dedup.conflict_min_shared` and `co_occupancy >= cooccur_hi`. This is computed for any pair of tracks in two components being joined, **whether or not an event exists for it**, and it does not use `min_overlap_seconds` or `min_observed`: those gates decide which pairs are worth proposing, not which pairs are safe to put in one component. A pair co-observed on fewer than `conflict_min_shared` frames (default 3) is treated as noise and does not block a join, so a lone coincident frame from flicker cannot veto a merge, while a short overlap that is densely co-observed (20 frames at 30 fps, below the one-second span gate; or 7 frames, below `min_observed`) does;
3. their classes are in different class groups.

**Procedure.**
1. Sort the accepted events by `(-algo_score, proposal_key)`: a total order that does not depend on proposal order or on work IDs.
2. Walk them with a union-find. For an edge `(a, b)`:
   - if `a` and `b` are already in one component, the edge is **redundant**: `applied: false`, `signals.skipped_reason: "redundant"`;
   - otherwise, if any member `x` of `a`'s component and any member `y` of `b`'s component form a cannot-link pair, the edge is **conflicting**: `applied: false`, `signals.skipped_reason: "conflict"`, and `signals.conflicts_with` names the first such pair, in sorted order, as `{"tracks": [x, y], "why": "rejected" | "dense" | "class", "proposal_key": <key of the rejecting event, or null>}`; `rejected` means a `VLM_REJECT` or `HUMAN_REJECT` event;
   - otherwise the two components are joined and the edge is **applied**: `applied: true`, and its rows are resolved as below.
3. An edge that is skipped keeps its accepted decision. The ledger shows the decision and `applied: false`, so an accepted but unapplied event is explicit, as for link's `skipped_reason: "overlap"`. A person reviewing the result sees both the accepted edge and the cannot-link evidence that stopped it, and the run summary counts skipped edges per reason.

For the A/B, B/C accepted and A/C vetoed case: whichever of A/B and B/C scores higher is applied; the other is skipped as `conflict` because joining it would put A and C together. The outcome depends only on scores and keys, not on the order the pairs were proposed in. The same holds when A/C was never proposed because its overlap is below the span or observation gate: the cannot-link check still sees it (rule 2), and the join is skipped as `conflict` with `why: "dense"`.

**Rows and attribution.** Each component keeps the earliest member's track ID (the representative; ties by work ID). All members' observed rows are kept, except that on a frame where two or more rows exist, only the best row stays. "Best" is a total order on rows: higher `score`; on a tie, the smaller `raw_id`; on a further tie, the smaller `track` ID. Rows are dropped **when an applied edge joins two components**: the rows on frames where both components have one, losing under that order, are dropped and attributed to **that edge**. Each edge's signals hold:
- `dropped_rows`: the count dropped at this edge;
- `dropped`: the dropped rows as `[raw_id, f0, f1]` runs of consecutive frames, so the ledger identifies exactly which raw observations were discarded. Every observed row not listed in some event's `dropped` is in the output;
- `merged_into`: the representative's work ID.

Because every edge is applied in the total order of step 1, each dropped row has exactly one owning edge, redundant and conflicting edges own none (`dropped_rows: 0`), and the sum of `dropped_rows` over a component's applied edges is the number of rows the component lost. The final kept set is the best row per frame over the whole component, whatever order the edges were applied in, so output and counts are the same under any proposal order, and with score ties.

**Downstream lineage excludes dropped rows.** A discarded observation must not influence anything computed for the merged track afterwards. Lineage spans are `[raw_id, f0, f1]` and consumers read the raw observations of that raw ID inside the span: `track_embeddings` and `dense_track_embeddings` request the raw ID's embeddings across it, and `EvidenceBuilder` indexes the raw rows in it. Today `lineage_of_rows` returns one span per raw ID from its minimum to its maximum surviving frame, so a dropped *interior* row would still be read. The rule is:
- `_Stages` keeps `excluded`, a map `raw_id -> set of frames` that dedup dropped;
- `lineage_of_rows(rows, excluded=None)` splits a raw ID's span at every excluded frame that falls strictly inside it, so `[122, 10, 30]` with frame 20 excluded becomes `[122, 10, 19]` and `[122, 21, 30]`. The span format is unchanged, so consumers need no change, and the spans cover exactly the retained rows' frames plus never-observed gaps, as before;
- every `lineage_of_rows` call after dedup receives `excluded` (link's descriptors and pair events, orphan, and fill), so link scoring and the evidence of later events see only retained observations;
- the `MERGE` events' own `lineage` is not changed: it describes the tracks as proposed, before the drop.

Cyclic case: A/B, B/C and A/C all accepted, all three with a row on frame `f`. The two highest-ranked edges join the three tracks and drop two rows between them on `f` (the loser of each join); the third is `redundant` with `dropped_rows: 0`. No edge reports six.

`merge_tracks(work, edges)` in `apply.py` implements the join and the row choice; the procedure above (sorting, cannot-link checks, attribution) lives in `dedup.py`.

### 3.6 Hand-off to orphan and link

The stage returns, besides the merged table:
- `merged_reps`: the representative IDs of components with at least one applied edge;
- `pending_endpoints`: the IDs (mapped to their representatives) of both tracks of every `HUMAN_PENDING` `MERGE`;
- the absorbed map for this stage (§5).

`_Stages` carries `merged_reps` and `pending_endpoints` through link and into the orphan pass:
- link maps both sets through its representative map (`rep_of`), the way it already maps its own pending endpoints, so a merged track that link then absorbs is still found by its final representative;
- with `link.enabled: false`, `_link` returns the carried sets unchanged;
- `_orphans` receives `linked | merged_reps` as `linked_tracks` and `link_pending | pending_endpoints` as `pending_endpoints`.

So an accepted merge protects its representative from the orphan drop, exactly as a linked track is protected, and a track waiting on a pending merge is deferred (it appears in `orphan_deferred`) instead of being dropped before review. This matters because a sparse track can meet dedup's span-overlap gate and still be too short in observed seconds (`propose_orphans` counts observed rows divided by `fps`, not the span).

## 4. Replay and keys

`MERGE` events carry the stable `proposal_key` of §3.3. A recorded decision on a `MERGE` event is applied by key when the stage re-runs, the same way as for the other stages (parent §4.2), and a recorded rejection becomes a cannot-link pair (§3.5). The stage proposes deterministically from the same inputs, so unchanged inputs reproduce the same keys. The `apply` command itself is out of scope here (it belongs to P4, which is on hold); this spec only guarantees that the events are compatible with it.

## 5. Output IDs, review page and ledger

**Absorbed map.** Merging and linking remove track IDs from the table. `_Stages` keeps `absorbed`, a map from each absorbed work ID to the work ID that absorbed it, updated after dedup (from its representative map) and after link (from `rep_of`). Resolution follows the chain to the survivor, `resolve(t)`, so a track absorbed by a merge and then by a link resolves to the link's representative. The final output ID of any work ID is `id_map[resolve(t)]`, and is absent only when the survivor was dropped (orphan or screen).

The ledger header stores the fully resolved map as `absorbed`: `{work_id: survivor_work_id}` for every absorbed ID. A consumer that has `id_map` and `absorbed` can map any event's `tracks` to output IDs, for any stage.

**Review page.** `review._output_ids` takes `absorbed` and applies `id_map[resolve(t)]`, so a pending `MERGE` whose endpoint was absorbed by an accepted merge, and then by a link, still shows the output trajectory and its clip snippet. LINK cards use the same lookup, which also fixes link endpoints absorbed by other links, an existing gap in the same function. `signals.merged_into` on applied events is informational only and is not what the page relies on.

**Ledger.** `MERGE` events are written like all others, including `HUMAN_PENDING` ones and accepted-but-skipped ones (§3.5), with `stage: "dedup"` and IDs of the form `dedup-r0-000001`.

**Which events the report shows.** The inherited `write_review` selects only `HUMAN_PENDING` events and removes the page when there are none. For dedup it selects two groups:
- **Pending `MERGE`** (and every other kind's pending events, as before): cards with accept and reject controls, exported in `decisions.json` as usual.
- **Skipped `MERGE` with `skipped_reason: "conflict"`**: a **read-only** card in a separate "Skipped merges" section, with no accept or reject controls and no entry in the exported decisions. It shows why the edge was accepted yet not applied (`conflicts_with`). A reviewer cannot override a cannot-link pair in this version, so there is nothing to decide on these cards. `redundant` edges are not shown (they appear only in the ledger).

The page is written when there is at least one card of either group, and removed (with its listed images and manifest, parent §8.1) only when there is none of either. The page's stage filter lists `dedup`.

**Review card.** A `MERGE` card shows the kind, both tracks (stage-time and output IDs), the overlap frames, `algo_score` and its signals (`co_occupancy` with its `shared` and `min(n_a, n_b)`, `comotion`, `appearance`), and, for a skipped edge, its `skipped_reason` and `conflicts_with`. Its evidence image is a pair of crops of the two tracks at alternating observed frames across the span, built by the existing `EvidenceBuilder` (a new method that reuses its crop code); crops are labeled by `tracks[0]` with `lineage[0]` and `tracks[1]` with `lineage[1]`. With no video, the card has no image, as for other events. A pending card has the usual `Labeler.draw_track_clips(...)` snippet for both tracks.

**Summary.** `summary["events"]` counts `(dedup, MERGE, decision)`, and `summary["dedup"]` counts applied, redundant and conflicting edges and the total `dropped_rows`. The run summary also records tracks before and after, which already captures the drop in unique IDs.

## 6. Configuration

A new `DedupConfig` in `RefineConfig`, parsed and validated like the other stage configs (unknown keys fail; `accept_above` must exceed `reject_below`; `cooccur_lo < cooccur_hi`; `conflict_min_shared >= 1`):

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Run the stage. |
| `accept_above` | `0.75` | Auto-accept at or above. |
| `reject_below` | `0.40` | Auto-reject below. |
| `min_overlap_seconds` | `1.0` | Minimum span overlap for a candidate. |
| `min_observed` | `8` | Minimum observed rows of each track in the overlap. |
| `comotion_lo`, `comotion_hi` | `0.25`, `0.50` | Ramp for co-motion. |
| `cooccur_lo`, `cooccur_hi` | `0.10`, `0.50` | Co-occupancy: no penalty up to `lo`, a proportional penalty to `hi`, and not a candidate at or above `hi`. |
| `conflict_min_shared` | `3` | Fewest shared frames for a dense co-observation to block a join (§3.5). Independent of `min_overlap_seconds` and `min_observed`. |
| `appearance_floor` | `0.60` | Lowest `appearance_term`. |
| `app_lo`, `app_hi` | `0.40`, `0.70` | Ramp for appearance similarity. |

(`distinct_iou` from rev. 1 is removed: occupancy no longer uses box IoU.)

The numeric defaults above were calibrated by hand on the three clips (§12). The plan's calibration task re-checks them to the evening clip's raw tracks (positives: 122/125/134, and the other pairs with few shared frames that look like one person in the video; negatives: 257/259 and 266/270), re-measures the §1 table, and writes the final values into this table and `config.py` together. `enabled: false` reproduces the old pipeline exactly, so existing configs and tests keep their behavior; with the flag on, outputs change by design.

## 7. Targets and compatibility

- **Person and vehicle.** The stage runs for both targets with the same config. Vehicle class grouping uses `link.class_groups`. The reference case is pedestrians; vehicles are not tuned in this version.
- **Existing configs.** A config without a `dedup` block gets the defaults (the stage on). A ledger written before this change has no `dedup` events and no `absorbed` map; reading it is unchanged, and the review treats a missing `absorbed` as empty.
- **Golden tests.** Pipelines that compare against stored outputs either set `dedup.enabled: false` or have their goldens regenerated in the same change, as the plan specifies.
- **API surface.** `EventKind.MERGE`, `dnt.refine.dedup.propose_merges`, `dnt.refine.apply.merge_tracks`, `DedupConfig`, and the `absorbed` header field. The docs page for `dnt.refine` lists the new config block and the stage.

## 8. Errors and degraded modes

- **No video or encoder:** motion-only scoring (`appearance` is `None`), logged at INFO, like the other stages.
- **Too few observed rows:** a track below `min_observed` in the overlap is not a candidate for that pair.
- **Class mismatch:** never a candidate and a cannot-link pair.
- **Config:** invalid values fail at construction with a `ValueError` that names the field.
- **VLM:** a configured backend is not used for `MERGE`. This is logged once at INFO so the behavior is not a surprise. A later version can add a prompt and an evidence image and route the uncertain band through `route_with_vlm`.

## 9. Testing (synthetic tracks in numpy, `tests/refine/test_dedup.py`)

Fixtures state their observation counts and the IoU on the shared frames, and the tests assert the signals, not only the outcome.

1. **Interleaved pair merges, with the real counts.** `n_a = 76`, `n_b = 107`, 7 shared frames whose boxes have IoU 0.00, all on one path: `co_occupancy = 7/76`, below `cooccur_lo`; `MERGE` is auto-accepted; one track remains. The fixture holds exactly the span's rows (76 + 107 = 183 observed rows), the 7 shared frames each lose one row, so the output has **176** rows, `dropped_rows` is 7, and the `dropped` runs name exactly those 7 `(raw_id, frame)` rows.
2. **Occupancy denominator.** `co_occupancy` equals `shared / min(n_a, n_b)` for several `(shared, n_a, n_b)` cases, including `shared = 0` (value `0.0`, no division error), and the signals carry `shared`, `n_a` and `n_b`.
3. **Side by side, disjoint boxes.** Two people walk in parallel, both observed on every frame, boxes apart: not a candidate (`co_occupancy = 1`).
4. **Side by side, overlapping boxes.** The same, with equal-size boxes offset by a third of their width, so IoU is 0.5 on every frame, both fully observed: not a candidate. This case fails under an IoU-based test and passes under the occupancy gate.
5. **Simultaneous duplicate.** Two tracks on the same frames at IoU 0.9: not merged, as the spec states (out of scope).
6. **Three-way group.** Three IDs interleave: one track under the earliest ID.
7. **Conflict rule.** A/B and B/C are both accepted, A/C is dense (co-occupancy above the gate and `shared >= conflict_min_shared`): the higher-ranked of A/B and B/C is applied, the other is accepted but `applied: false` with `skipped_reason: "conflict"` and `conflicts_with` naming A/C; A and C end in different tracks. A second case replaces the dense pair with a rejected A/C `MERGE` (`VLM_REJECT` and `HUMAN_REJECT`, each) with the same result. **An auto-reject does not block:** with A/C decided `AUTO_REJECT` (score near 0, not dense, `shared < cooccur_hi`-level co-occupancy), A/B and B/C both applied, A, B and C end in one track (the 125/134 case); a `HUMAN_PENDING` A/C does not block either. Repeat with shuffled proposal order: same output and ledger. **Dense A/C below the proposal gates still blocks the bridge:** (a) A/C overlap just below the span gate (29 frames at 30 fps, all co-observed, distinct boxes); (b) A/C overlap with `n` below `min_observed` (7 rows each, all shared). In both, A/C has no `MERGE` event, A/B and B/C are accepted, the lower-ranked bridge is skipped as `conflict` with `why: "dense"`, and A and C end in different tracks. **Noise does not block:** A/C co-observed on only 2 frames (`shared < conflict_min_shared`) joins normally.
8. **Dropped rows, cyclic.** A/B, B/C and A/C all accepted, all with a row on one frame, with tied scores, repeated over shuffled proposal orders: identical output tracks, identical `dropped` runs, the third edge `redundant` with `dropped_rows: 0`, and the sum of `dropped_rows` equals the rows lost.
9. **Bands.** A score in the uncertain band becomes `HUMAN_PENDING` with no VLM call even when a backend is configured; a score below `reject_below` is `AUTO_REJECT` and changes nothing.
10. **Gates.** Different classes, too short an overlap, and too few observed rows each produce no candidate.
11. **Orphan hand-off.** A sparse merged representative (few observed seconds, long span) is not dropped as an orphan; a track that is an endpoint of a pending `MERGE` is deferred, not dropped, and appears in `orphan_deferred`. Both with `link.enabled: true` (with a link that absorbs the representative in one case) and `link.enabled: false`.
12. **Order.** With `dedup.enabled: false` the output equals the previous pipeline's; with it on, link receives the merged table (a link event that would have joined two interleaved IDs does not appear).
13. **Keys.** The `proposal_key` is unchanged when the same input is run again. A second test **renumbers the work-table track IDs while keeping raw IDs**, including a pair that starts on the same frame whose work-ID order reverses (so the representative changes): the key is the same in both runs, and the key lineage is the canonical copy in both. In both runs the event's `lineage[i]`, `tracks[i]` and `signals` `n_a`/`n_b` correspond position by position (checked against the raw IDs: for a representative with raw ID 125 and a later member with raw ID 122, `lineage[0]` starts with raw 125), and the evidence crops are labeled with the track ID that matches the lineage they were cut from.
14. **Absorbed IDs in review.** A pending `MERGE` A/C whose endpoint A is absorbed by an accepted merge (A into B) and then B by a link: the header's `absorbed` resolves A to the final survivor, and the card and its clip snippet list the survivor's output ID.
15. **Ledger and review.** `MERGE` events round-trip through `Ledger.write`/`read`; a pending `MERGE` produces a card with controls and its signals; the review stage filter lists `dedup`. **Skipped merges are visible:** a run with only accepted edges, one skipped as `conflict`, and **no pending events** still writes the page, with a read-only card in "Skipped merges" that shows `skipped_reason` and `conflicts_with`, has no accept or reject controls, and adds no entry to the exported decisions; a `redundant` edge produces no card; with neither pending nor conflicting events the page and its listed images are removed.
16. **Real-data regression (skipped when the data is absent).** Evening clip raw tracks 122, 125 and 134 become one track under ID 122, with the counts of §1 and 13 dropped rows or fewer in all, and no pair among the merged IDs shares a frame in the output. Another assertion keeps the negatives apart: middle 266/270 and 15/19 are not merged (gated). It is skipped when `/mnt/e/videos/miami` is absent.
17. **Config.** Defaults, YAML round trip, unknown-key, band-order, `cooccur` order and `conflict_min_shared` errors.
18. **Dropped rows leave downstream lineage.** Raw 122 has frames 10 to 30 and an interior row at frame 20 loses to the other track's row. After the merge, the track's lineage after dedup is `[122, 10, 19]` and `[122, 21, 30]`. With a recording fake appearance provider, link's embedding request for the merged track never covers `(122, 20)`; `EvidenceBuilder` for a later link event never draws it; and `lineage_of_rows` without `excluded` is unchanged (so existing callers and tests keep their behavior).

## 10. Packaging and documentation

- New module `src/dnt/refine/dedup.py`; `merge_tracks` and the `excluded` parameter of `lineage_of_rows` in `apply.py`; `EventKind.MERGE`, its `DEFINING_PARAMS` entry and the `key_lineage` keyword of `Event.propose` in `events.py`; `DedupConfig` in `config.py`; the step, the carried sets, the `absorbed` map and the `excluded` map (passed to every later `lineage_of_rows` call in `link.py`, `refiner.py` and the orphan pass) in `_Stages` in `refiner.py`; the header field; the cards, the "Skipped merges" section, the page-selection rule, the filter entry and the `absorbed` lookup in `review.py`; a `MERGE` evidence method in `evidence.py`.
- Numpy-style docstrings (ruff `D` rules apply); no new entries in the per-file lint baseline.
- `docs/api` page for `dnt.refine` already covers the package; the changelog and the refine docs describe the stage, its config and the `dedup.enabled` switch.
- No version bump in this change; the release process in `CLAUDE.md` applies when it ships.

## 11. Open questions deferred to the plan

1. Re-checking the numeric defaults of §6 against more clips than the three used for the hand calibration in §12.
2. Whether the pair crops for the evidence image use all alternating frames or a fixed sample of at most 6.

## 12. Revision 4: findings from prototyping the plan

While preparing the implementation plan, I ran a prototype of the signals and the apply procedure on the three raw pedestrian files (fps 10) and looked at the video. It changed three things, each marked above.

1. **Measured counts (§1, §3.2).** The earlier table mixed up which track the `n` values belonged to. The measured values: 122/134 has `n = 13 / 89` and 125/134 has `n = 30 / 13`, so 125/134 is `6/13 = 0.46`, not the lower bound 0.20 of rev. 3. The question left open in rev. 3 is closed.
2. **Rule 1 amended (§3.5).** With the rev. 3 rule, 125/134 (co-motion 0.29, auto-rejected) would have blocked the join of 122/125/134, so the merge that the evidence case (§1) is about could not happen at any workable threshold. Only `VLM_REJECT` and `HUMAN_REJECT` now block; dense co-observation (rule 2) remains the hard veto. Decided with the project owner after viewing the frames, which show one person.
3. **Calibrated defaults (§6).** `accept_above` 0.80 to 0.75 (122/134 scores 0.79) and `cooccur_hi` 0.40 to 0.50 (so 125/134 at 0.46 is not densely co-observed, while the side-by-side pairs middle 266/270 at 0.53 and 15/19 at 0.62 stay gated). With these and the amended rule, the prototype merges 122, 125 and 134 into 122 (13 rows dropped in all) and leaves 266/270 and 15/19 alone. The margin on `cooccur_hi` is thin (0.46 against 0.53) and rests on three clips; §11 keeps the re-check open.

Prototype counts per clip at these settings (tracks before and after, merges applied, rows dropped): beginning 63 to 62 (1, 1 row); middle 89 to 83 (6, 10 rows); evening 134 to 129 (5, 19 rows). The prototype is not the implementation, and these numbers come from it, not from the final code.
