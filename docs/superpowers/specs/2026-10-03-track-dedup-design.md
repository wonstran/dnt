# dnt.refine: Merging Interleaved Duplicate Tracks (stage `dedup`)

- **Status:** draft for review
- **Date:** 2026-10-03
- **Parent spec:** [2026-09-27-track-refinement-design.md](2026-09-27-track-refinement-design.md). This adds one stage to `dnt.refine`. Everything not stated here follows the parent spec (events §4.1, ledger §4.2, bands §4.3, shared primitives §5, review §8.1, config §9).
- **Evidence:** `A1A_Collins Ave & 17th St` clips, pedestrian target (§1).

## 1. Problem

The tracker sometimes alternates IDs on one person from frame to frame, so two or three IDs each cover part of the same trajectory over the same span. Stages 1–3 never see it as an error: link treats track ends and starts as sequential fragments, switch looks inside one track, and screen compares tracks with context. Stage 4 then makes it visible. Fill interpolates each ID across its own gaps, and those gaps are the frames where another ID had the observation. The output holds overlapping duplicate boxes on one person.

Measured on the evening clip (`..._evening_1900_190000.00_ped_track.txt`), P3 output IDs 21, 22 and 24, from raw tracks 122, 125 and 134:

| Pair (raw IDs) | Shared observed frames | Mean IoU, observed rows | Shared frames after fill | Mean IoU after fill |
|---|---|---|---|---|
| 122 / 125 | 7 | 0.00 | 87 | 0.49 |
| 122 / 134 | 0 | not defined | 54 | 0.48 |
| 125 / 134 | 6 | 0.13 | 41 | 0.41 |

Each track has about 107 to 294 observed rows over 182 to 456 frames, and the tracks almost never have a row on the same frame. No link or split event touched them. The only events on them are `FILL`.

A crude scan of the raw files (two tracks overlapping at least 30 frames, one's observed boxes against the other's interpolated path, mean IoU above 0.3) found 3 candidate pairs in `beginning`, 12 in `evening` and 7 in `middle`. That scan also flags real people walking side by side (for example evening 257/259 share 36 observed frames, middle 266/270 share 50), which is why the stage needs a co-occurrence veto (§3.2).

## 2. Goal and scope

**Goal.** Consolidate each group of interleaved duplicate tracks into one track before fill, keep every real observation, and leave people who walk side by side alone.

**In scope**
- A new stage, `dedup`, between screen and link, with a new event kind `MERGE`.
- Detection from motion, with appearance as a secondary signal when a video is given.
- Banded routing like the other stages. The uncertain band becomes `HUMAN_PENDING` (no VLM in this version).
- Ledger, review-page, config, docs and changelog support.

**Out of scope**
- A VLM prompt and evidence image for `MERGE` (§8).
- Changes to link, fill or the existing stages.
- Any change to the vehicle defaults beyond running the same stage with the same config (§7).

**Success criteria**
1. On the evening clip, raw tracks 122, 125 and 134 become one track, and the P3-style output has no overlapping boxes among them.
2. A synthetic pair of people walking side by side, each with a full observed trajectory, is never merged.
3. The unique ID count drops and no real observation is lost: every observed row from the merged tracks is in the output, except the lower-score row on a frame where both tracks have one.
4. Replay compatibility: a `MERGE` event has a stable `proposal_key` (§4).

## 3. The stage

### 3.1 Order

```
raw tracks
  1 switch → 2 screen → dedup → 3 link → orphan → 4 fill
```

- **After screen:** false tracks are dropped before anything is merged into them.
- **Before link:** link sees one consolidated track instead of interleaved fragments that it could link to each other.
- **Before fill:** fill interpolates over final identities, which is where the duplicates came from.

`dedup` follows the parent spec's stage contract: it proposes events from the previous stage's applied table, routes them (§4.3 of the parent), applies the accepted ones, and passes the table on. `_Stages.run` gains a `dedup` step and a `tick("dedup")`; the progress bar total changes from 5 to 6.

### 3.2 Candidates and signals

Descriptors are built from **observed rows only** (filled rows were already removed on read, parent §2.5). For each track: sorted observed frames, boxes, class, and linear interpolation of the box over the track's own span.

A pair `(a, b)` is a candidate when:
- their classes are in the same class group (`link.class_groups`; for a person target, the same class);
- their spans overlap by at least `dedup.min_overlap_seconds` (converted to frames with `fps`);
- each has at least `dedup.min_observed` observed rows inside the overlap.

For a candidate, all signals are computed over the span overlap:

| Signal | Definition |
|---|---|
| `comotion` | The mean over both directions of the IoU between a track's observed box and the other track's interpolated box on that frame. Both directions, so one dense track cannot hide a mismatch. |
| `cooccur` | The share of frames observed in **both** tracks whose boxes have IoU below `dedup.distinct_iou`. A high share means two separate boxes at the same time: two people. |
| `shared` | The count of frames observed in both tracks, logged for review. |
| `appearance` | Mean cosine similarity between the two tracks' embeddings (the same embeddings link uses), or `None` without a video or encoder. |

Score:

```
S_dedup = comotion_ramp(comotion) * (1 - cooccur_ramp(cooccur)) * appearance_term
```

- `comotion_ramp` rises from 0 at `dedup.comotion_lo` to 1 at `dedup.comotion_hi` (the parent's ramp primitive, §5.4).
- `cooccur_ramp` rises from 0 at `dedup.cooccur_lo` to 1 at `dedup.cooccur_hi`. A few co-observed frames from the flicker itself (7 shared frames for 122/125) fall below `cooccur_lo` and do not block a merge. Many distinct co-observed boxes drive the score to 0, so side-by-side pedestrians are never candidates for acceptance.
- `appearance_term` is 1.0 when `appearance` is `None`. Otherwise it is a ramp from `dedup.appearance_floor` (at similarity `app_lo`) to 1.0 (at `app_hi`). Appearance can lower a score but, with `appearance_floor` above 0, cannot veto a pair that motion strongly supports. This follows the way link treats appearance as a secondary signal.

The defaults for the `comotion_*`, `cooccur_*` and `app_*` knobs are chosen in the plan against the three evening tracks (positive) and the side-by-side pairs (negative), and recorded in the config table in §6.

### 3.3 Event

`EventKind.MERGE` is added. One event per candidate pair.

| Field | Value |
|---|---|
| `tracks` | `[a, b]`, with `a` the track whose first observed frame is earlier (ties broken by ID). |
| `lineage` | The raw spans of both tracks (parent §4.2). |
| `frames` | The overlap span `[lo, hi]`. |
| `params` | `{"span": [lo, hi]}` |
| `algo_score` | `S_dedup` |
| `signals` | `comotion`, `cooccur`, `shared`, `appearance`, `n_a`, `n_b` (observed rows in the overlap), and, once applied, `merged_into` and `dropped_rows`. |
| `edit` | `{"kind": "MERGE", "params": {"span": [lo, hi]}}`, filled when accepted. |

`DEFINING_PARAMS[MERGE] = ("span",)`, so the `proposal_key` is the SHA-256 of `(stage, MERGE, lineage, span)`. It does not depend on the work-table track IDs, so a decision keeps its key across reruns, as parent §4.2 requires.

### 3.4 Routing

The stage's band is `Band.of(cfg.dedup)`. `route_without_vlm` already maps the uncertain band to `HUMAN_PENDING`, and `route_with_vlm` would ask the VLM; `dedup` always uses the no-VLM router in this version, even when a VLM backend is configured. The review page gets a card for each pending `MERGE` (§5).

### 3.5 Applying accepted merges

1. Accepted pairs are grouped with union-find. A group of three or more IDs (122/125/134) merges into one track.
2. `merge_tracks(work, pairs)` in `apply.py` sets every member's `track` to the earliest member's ID. Where two members have a row on the same frame, it keeps the row with the higher `score` (ties keep the earlier member's row) and drops the other. All other observed rows are kept.
3. The number of dropped rows is recorded per event in `signals.dropped_rows`, and the representative ID in `signals.merged_into`.
4. A merge is never applied if it would make the merged track have two overlapping boxes on a frame after the row choice; that cannot happen with one row kept per frame, so no further overlap check is needed (unlike link's `resolve_chains`).

The stage returns the merged table and the set of merged representative IDs. Link receives the table only; it needs no knowledge of the merge.

## 4. Replay and keys

`MERGE` events carry the stable `proposal_key` of §3.3. A recorded decision on a `MERGE` event is applied by key when the stage re-runs, the same way as for the other stages (parent §4.2). The stage proposes deterministically from the same inputs, so unchanged inputs reproduce the same keys. The `apply` command itself is out of scope here (it belongs to P4, which is on hold); this spec only guarantees that the events are compatible with it.

## 5. Review page and ledger

- **Ledger.** `MERGE` events are written like all others, including `HUMAN_PENDING` ones, with `stage: "dedup"` and IDs of the form `dedup-r0-000001`.
- **Review page.** A `MERGE` card shows the kind, both tracks (stage-time and output IDs), the overlap frames, `algo_score` and its signals (`comotion`, `cooccur`, `shared`, `appearance`). Its evidence image is a pair of crops of the two tracks at alternating observed frames across the span, built by the existing `EvidenceBuilder` (a new method that reuses its crop code). With no video, the card has no image, as for other events. It has accept and reject controls and the usual `Labeler.draw_track_clips(...)` snippet for both tracks. The page's stage filter gains `dedup`.
- **Summary.** `summary["events"]` counts `(dedup, MERGE, decision)`. The run summary also records tracks before and after, which already captures the drop in unique IDs.

## 6. Configuration

A new `DedupConfig` in `RefineConfig`, parsed and validated like the other stage configs (unknown keys fail; `accept_above` must exceed `reject_below`):

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Run the stage. |
| `accept_above` | `0.80` | Auto-accept at or above. |
| `reject_below` | `0.40` | Auto-reject below. |
| `min_overlap_seconds` | `1.0` | Minimum span overlap for a candidate. |
| `min_observed` | `8` | Minimum observed rows of each track in the overlap. |
| `distinct_iou` | `0.30` | A co-observed pair of boxes below this IoU counts as distinct. |
| `comotion_lo`, `comotion_hi` | `0.25`, `0.50` | Ramp for co-motion. |
| `cooccur_lo`, `cooccur_hi` | `0.10`, `0.40` | Ramp for the co-occurrence veto. |
| `appearance_floor` | `0.60` | Lowest `appearance_term`. |
| `app_lo`, `app_hi` | `0.40`, `0.70` | Ramp for appearance similarity. |

The numeric defaults above are starting values. The plan's first task fits them to the evening clip's raw tracks (positives: 122/125/134, and the other pairs with at most 10 shared frames that look like one person in the video; negatives: 257/259 and 266/270) and writes the final values into this table and `config.py` together. `enabled: false` reproduces the old pipeline exactly, so existing configs and tests keep their behavior; with the flag on, outputs change by design.

## 7. Targets and compatibility

- **Person and vehicle.** The stage runs for both targets with the same config. Vehicle class grouping uses `link.class_groups`. The reference case is pedestrians; vehicles are not tuned in this version.
- **Existing configs.** A config without a `dedup` block gets the defaults (the stage on). A ledger written before this change has no `dedup` events; reading it is unchanged.
- **Golden tests.** Pipelines that compare against stored outputs either set `dedup.enabled: false` or have their goldens regenerated in the same change, as the plan specifies.
- **API surface.** `EventKind.MERGE`, `dnt.refine.dedup.propose_merges`, `dnt.refine.apply.merge_tracks`, `DedupConfig`. The docs page for `dnt.refine` lists the new config block and the stage.

## 8. Errors and degraded modes

- **No video or encoder:** motion-only scoring (`appearance` is `None`), logged at INFO, like the other stages.
- **Too few observed rows:** a track below `min_observed` in the overlap is not a candidate for that pair.
- **Class mismatch:** never a candidate.
- **Config:** invalid values fail at construction with a `ValueError` that names the field.
- **VLM:** a configured backend is not used for `MERGE`. This is logged once at INFO so the behavior is not a surprise. A later version can add a prompt and an evidence image and route the uncertain band through `route_with_vlm`.

## 9. Testing (synthetic tracks in numpy, `tests/refine/test_dedup.py`)

1. **Interleaved pair merges.** Two IDs alternate frames along the same path; `MERGE` is auto-accepted and one track remains with all observed rows.
2. **Three-way group.** Three IDs interleave; one track remains under the earliest ID.
3. **Side by side is never merged.** Two people walk in parallel with distinct boxes on every frame (`cooccur` high); no `MERGE` is accepted.
4. **A few co-observed flicker frames still merge.** An interleaved pair with 7 shared frames (as in the evening data) still scores above `accept_above`.
5. **Same-frame rows.** Where both tracks have a row on one frame, the higher-score row is kept and the count is in `signals.dropped_rows`.
6. **Bands.** A score in the uncertain band becomes `HUMAN_PENDING` with no VLM call even when a backend is configured; a score below `reject_below` is `AUTO_REJECT` and changes nothing.
7. **Gates.** Different classes, too short an overlap, and too few observed rows each produce no candidate.
8. **Order.** With `dedup.enabled: false` the output equals the previous pipeline's; with it on, link receives the merged table (a link event that would have joined two interleaved IDs does not appear).
9. **Keys.** The `proposal_key` is unchanged when the same input is run again and when track IDs are renumbered.
10. **Ledger and review.** `MERGE` events round-trip through `Ledger.write`/`read`; a pending `MERGE` produces a card with its signals; the review stage filter lists `dedup`.
11. **Real-data regression (marked as slow or data-dependent).** Evening clip raw tracks 122, 125 and 134 become one track, and no pair among the merged IDs shares a frame in the output. It is skipped when `/mnt/e/videos/miami` is absent.
12. **Config.** Defaults, YAML round trip, unknown-key and band-order errors.

## 10. Packaging and documentation

- New module `src/dnt/refine/dedup.py`; `merge_tracks` in `apply.py`; `EventKind.MERGE` and its `DEFINING_PARAMS` entry in `events.py`; `DedupConfig` in `config.py`; the step in `_Stages` in `refiner.py`; a card and filter entry in `review.py`; a `MERGE` evidence method in `evidence.py`.
- Numpy-style docstrings (ruff `D` rules apply); no new entries in the per-file lint baseline.
- `docs/api` page for `dnt.refine` already covers the package; the changelog and the refine docs describe the stage, its config and the `dedup.enabled` switch.
- No version bump in this change; the release process in `CLAUDE.md` applies when it ships.

## 11. Open questions deferred to the plan

1. The final numeric defaults of §6 (fitted to the evening clip as described there).
2. Whether the pair crops for the evidence image use all alternating frames or a fixed sample of at most 6.
