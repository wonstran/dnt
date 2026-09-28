# dnt.post: Track Refinement with Algorithmic Screening and VLM Verification

- **Status:** written spec, awaiting review
- **Date:** 2026-09-27
- **Baseline:** dnt 0.3.3 (`a290821`)
- **Roadmap:** [`design/dnt-0.4-upgrade.md`](../../../design/dnt-0.4-upgrade.md). This work belongs to sub-project F (new capabilities). It adopts the `dnt/post/` location from §3 of the roadmap. It does not depend on sub-projects B–E, because its only inputs are a track file and a video.

## 0. Context and agreed constraints

Pedestrian trajectories come from StrongSORT and vehicle trajectories from BoTSORT, in separate track files. They show four kinds of error:

1. **Gaps.** A trajectory breaks for some frames because detections are missing.
2. **ID fragmentation.** One object gets several IDs over time.
3. **ID switch.** One ID moves from one object to another.
4. **False tracks.** The tracked thing is not a target (a person or a vehicle).

Constraints agreed during brainstorming:

1. **Independent of tracking.** The procedure is standalone post-processing. It reads a track file and its video. It does not import `dnt.track`, `dnt.detect`, or BoxMOT, and it makes no assumption about which tracker produced the file.
2. **Algorithms screen, the VLM verifies.** Trajectory algorithms examine every track and flag short suspicious events (seconds to minutes). A VLM reviews only the events whose algorithmic confidence is in an uncertain band.
3. **Auto-apply if sure.** Confident verdicts, algorithmic or VLM, are applied automatically. Uncertain or failed verdicts go to a human review report. Until a person decides, a pending event leaves the tracks unchanged.
4. **Pluggable VLM.** One backend interface, with a local implementation (any OpenAI-compatible server, such as vLLM or Ollama) and a cloud implementation (Anthropic).
5. **Pluggable appearance.** One `AppearanceEncoder` interface, shipping two implementations: a generic DINOv2 encoder and a torchreid ReID encoder. The config selects one per target class.
6. **Downstream uses.** Counts and volumes, pedestrian–vehicle conflict metrics (PET, TTC), and speed/behavior analysis. No error type can be traded away. ID switches and interpolated positions matter most for conflicts.
7. **No ground truth.** Success is measured by spot-checking applied edits (§8.2).
8. **False-track definitions (pedestrian file).**
   - **Drop:** static non-objects (poles, signs, mannequins, posters, reflections, shadows), and people inside vehicles (drivers, passengers, bus riders).
   - **Reclass:** cyclists, motorcycle riders, and scooter riders become class *cyclist*, *motorcycle*, or *scooter*.
   - People outside the study area are **not** handled here. `dnt.filter.Filter` zones already do that.
9. **False-track definitions (vehicle file).**
   - **Drop:** non-vehicles (shadows, reflections, structure edges), and duplicate boxes on one vehicle (a trailer, or a bus split in two).
   - Parked vehicles are real and are kept.

## 1. Goal, scope, and success criteria

**Goal:** given a track file and its video, produce a corrected track file in the same format, a ledger of every edit with its evidence, and a review report of the edits that need a person.

**In scope**
1. New subpackage `dnt.post`, with the event model, ledger, four stages plus an orphan pass, evidence builder, VLM backends, review and audit reports, `TrackRefiner`, `RefineConfig`, and a `dnt-refine` CLI.
2. Moving `interpolate_tracks_rts` and `link_tracklets` into `dnt.post`, leaving a re-export shim at `dnt.track.post_process`.
3. Two appearance encoders behind optional extras.
4. Tests, API docs page, and changelog entry.

**Out of scope**
- Learned linkers (such as StrongSORT's AFLink) and global graph optimization. The stage 3 scorer sits behind an interface so one can be added later.
- Detector re-runs during screening (what `ReClass` does). `ReClass` stays where it is, unchanged.
- Camera calibration or world coordinates. All motion is measured in box heights per second.
- Study-area filtering. That stays in `Filter`.
- An interactive review server. The review page is static HTML.

**Success criteria**
1. With `vlm.backend: none` and no video, `refine` runs to completion on a 10-column track file, and every uncertain event ends up `HUMAN_PENDING`.
2. On synthetic fixtures with injected faults (§11.1), each stage proposes the expected event, and the score lands in the expected band.
3. Replaying a ledger with a decisions file gives byte-identical output across repeated runs and makes no VLM calls.
4. `dnt.post` imports nothing from `dnt.track`, `dnt.detect`, or `boxmot`. A test enforces this.
5. Existing `tests/test_post_process.py` passes unchanged through the shim.
6. On the user's own clips, the audit (§8.2) reports per-stage precision of applied edits with confidence intervals. The precision targets are set by the user after the first audit, not by this spec.

## 2. Architecture

### 2.1 Modules

```
src/dnt/post/
  __init__.py          # public API: TrackRefiner, RefineConfig, interpolate_tracks_rts, link_tracklets
  io.py                # read/write track CSV (dnt 10-col, MOTChallenge adapter); column constants
  events.py            # Event, Decision, EventKind; Ledger (JSONL read/write, replay)
  primitives.py        # scale-free speed, Kalman residual (NIS), occlusion mask, ramp()
  features.py          # TrackFeatures: per-track primitives + clean embeddings; .npz cache
  encoders/
    __init__.py        # AppearanceEncoder protocol, make_encoder(cfg)
    dino.py            # DINOv2 encoder (extra: dnt[post-dino])
    reid.py            # torchreid encoder (extra: dnt[post-reid])
  screen.py            # stage 1: DROP / RECLASS proposals; orphan pass
  switch.py            # stage 2: SPLIT proposals
  link.py              # stage 3: LINK proposals; contains the moved link_tracklets
  interpolate.py       # stage 4: contains the moved interpolate_tracks_rts
  apply.py             # applies decided events to a track table (pure function)
  evidence.py          # frame fetch, crops, composite grid image
  vlm/
    __init__.py        # VLMBackend protocol, VLMAnswer, make_backend(cfg)
    prompts.py         # per-event-kind prompt and option lists
    openai_compat.py   # vLLM, Ollama, any OpenAI-compatible endpoint
    anthropic.py       # Anthropic Messages API
    fake.py            # scripted backend for tests
    cache.py           # on-disk answer cache
  verify.py            # banding, VLM routing, budget, votes, decision assignment
  review.py            # review report and audit report (static HTML)
  config.py            # RefineConfig and nested dataclasses; YAML round-trip
  refiner.py           # TrackRefiner: orchestrates stages, verify, apply, outputs
  cli.py               # dnt-refine entry point
```

Each module has one job. `screen.py`, `switch.py`, and `link.py` only **propose** events. They never edit tracks. `apply.py` is the only code that changes a track table.

### 2.2 Dependency rule

`dnt.post` may import `dnt.shared` and third-party libraries only. It must not import `dnt.track`, `dnt.detect`, `dnt.label`, `dnt.filter`, or `boxmot`. `tests/test_post_independence.py` enforces this by importing `dnt.post` in a subprocess and checking `sys.modules`.

Evidence rendering uses cv2 directly rather than `Labeler`, so the rule holds.

Encoders and VLM backends import their heavy dependencies (`transformers`, `torchreid`, `openai`, `anthropic`) lazily, inside the constructor.

### 2.3 The move of existing functions

- `interpolate_tracks_rts` moves to `dnt/post/interpolate.py`.
- `link_tracklets` moves to `dnt/post/link.py`.

Their signatures and behavior stay the same. `dnt/track/post_process.py` becomes a shim: `from ..post.interpolate import interpolate_tracks_rts` and `from ..post.link import link_tracklets`, with `__all__`. That keeps `from dnt.track import link_tracklets` and existing scripts working. The shim direction (track → post) does not violate §2.2, which only restricts what `dnt.post` imports.

Stage 3 does not call `link_tracklets`. It reuses that function's gating helpers (`_iou_xywh`, `_estimate_velocity`), which move to `primitives.py`, and adds the new scoring described in §6.3.

### 2.4 Entry points

```python
from dnt.post import TrackRefiner, RefineConfig

cfg = RefineConfig.from_yaml("ped.yaml")
result = TrackRefiner(cfg).refine(
    "ped_tracks.csv",
    video="cam1.mp4",                 # optional; without it, motion-only mode (§10)
    context_tracks="veh_tracks.csv",  # optional; enables in-vehicle, rider-overlap, and duplicate cues
    out="ped_clean.csv",
)
# result.tracks: DataFrame; result.ledger_path; result.review_path; result.summary
```

```
dnt-refine run    TRACKS --video V [--context C] --config CFG --out OUT
dnt-refine apply  TRACKS --ledger L [--decisions D.json] --out OUT
dnt-refine audit  --ledger L --video V --n 50 [--seed S]
dnt-refine audit-score --ledger L --marks M.json
```

The CLI is a thin `argparse` wrapper over `TrackRefiner`, registered as `[project.scripts] dnt-refine = "dnt.post.cli:main"`.

### 2.5 Input and output contract

**Input**
- A headerless dnt track CSV with 10 columns: `frame, track, x, y, w, h, score, cls, r3, r4`.
- Column 8 may already be `interp` from an earlier interpolation. A value of `1` means filled, and `0` or the legacy `-1` means observed. Filled rows are treated as unobserved in every stage.
- MOTChallenge files (`frame, id, x, y, w, h, conf, x3d, y3d, z3d`) are read with `io.read_tracks(path, fmt="mot")`. That format has no class, so `cls` is set to `RefineConfig.class_ids[0]`.

**Output**
- `OUT`: headerless 10-column CSV in the dnt layout, with column 8 as `interp` (`0` observed, `1` filled). This is the same layout `interpolate_tracks_rts` writes today. Sorted by `frame, track`. Track IDs are renumbered contiguously from 1, and `ledger.id_map` records the renumbering.
- `OUT.ledger.jsonl`: header line, then one line per event (§4.2).
- `OUT.review.html` plus `OUT.review/` (images), written only when at least one event is `HUMAN_PENDING`.
- `OUT.features.npz`: the embedding cache (§5.3). Reused when the track file hash and encoder settings match.

## 3. Processing order

```
raw tracks
  └─ 1 screen  ── DROP / RECLASS ──┐
  └─ 2 switch  ── SPLIT ───────────┤  each stage: propose → verify (§7) → apply
  └─ 3 link    ── LINK ────────────┤  before the next stage runs
  └─ orphan    ── DROP ────────────┤
  └─ 4 fill    ── (deterministic RTS, no events)
clean tracks
```

The order stops errors from compounding.
- **Screen first.** A pole's static track must not be linked to a real pedestrian.
- **Split before link.** A track containing two identities must be cut before its pieces can be re-paired.
- **Orphans after link.** Short fragments get the chance to be linked before being judged too short.
- **Fill last.** Interpolation runs over the final identities and never across a rejected link.

Each stage sees the tracks produced by the previous stage's applied events. VLM calls for one stage finish before the next stage starts.

## 4. Event model, bands, and ledger

### 4.1 Event

```python
class EventKind(StrEnum): DROP, RECLASS, SPLIT, LINK

class Decision(StrEnum):
    AUTO_ACCEPT, AUTO_REJECT,        # decided by algo_score alone
    VLM_ACCEPT, VLM_REJECT,          # decided by a sure VLM verdict
    HUMAN_PENDING,                   # waiting for a person; not applied
    HUMAN_ACCEPT, HUMAN_REJECT       # decided by a person via decisions.json

@dataclass
class Event:
    id: str                  # f"{stage}-{seq:06d}", stable across replays
    stage: str               # "screen" | "switch" | "link" | "orphan"
    kind: EventKind
    tracks: list[int]        # IDs as they were when the stage ran
    frames: tuple[int, int]  # the frame span the event concerns
    params: dict             # DROP: {reason}; RECLASS: {new_cls | None}; SPLIT: {cut_frame}; LINK: {}
    algo_score: float        # [0, 1]
    signals: dict            # every named cue value and ramp output behind algo_score
    decision: Decision
    vlm: dict | None         # {backend, model, answer, confidence, votes, reason, evidence, cached}
    applied: bool
```

`EventKind` and `Decision` subclass `StrEnum`.

### 4.2 Ledger

A JSONL file.
- **Line 1** is the header: dnt version, config (full YAML as a dict), input file SHA-256, video path and SHA-256 of its first 64 MiB, fps, `id_map`, and the before/after summary (§8.3).
- **Each later line** is one `Event`, written when its decision is final.

`Ledger.replay(raw_tracks, decisions=None)`:
1. Re-applies events in ledger order, applying those whose decision is an accept.
2. Overrides decisions from a `decisions.json` (`{event_id: "accept" | "reject" | {"accept": true, "new_cls": 3}}`).
3. Re-runs stage 4.

It makes no VLM or encoder calls. Replay is how human decisions take effect and how edits are undone.

Replay does not re-run proposals, so a human decision cannot create events that were never proposed. This limitation is accepted. If a person rejects a `SPLIT`, the stage 3 events that involved its tail become invalid and are skipped, with `applied: false` and a `skipped_reason`.

### 4.3 Bands

For each stage, `accept_above` and `reject_below`, with `reject_below < accept_above`:

| algo_score | Route |
|---|---|
| `≥ accept_above` | `AUTO_ACCEPT` |
| `< reject_below` | `AUTO_REJECT` (still written to the ledger, so audits can sample rejections) |
| otherwise | VLM (§7). With `vlm.backend: none`, `HUMAN_PENDING` |

**Exceptions**
- **Static screen.** The static-object score is capped at `screen.static_score_cap` (default 0.80). Keep the cap below `screen.accept_above`, so static tracks never auto-drop (§6.1).
- **Rider subtype.** A `RECLASS` whose rider score is `AUTO_ACCEPT` still needs one VLM call to choose the subtype. That call only picks the subtype and cannot overturn the rider decision. If the VLM answers with a non-rider option (for example `pedestrian`), the algorithm and the VLM disagree, and the event becomes `HUMAN_PENDING`. It also becomes `HUMAN_PENDING` if `params.new_cls` is still `None` after verification.
- **Link margin.** For stage 3, the routed value is `algo_score × ramp(margin)` (§6.3), not the raw score.

## 5. Shared primitives

### 5.1 Scale-free motion

- `h̃_t` is the rolling median of box height over `motion.height_window` frames (default 15).
- Speed is `v_t = ‖c_t − c_{t−Δ}‖ / (h̃_t · Δ/fps)`, in **box heights per second** (h/s), where `c` is the box center and `Δ` is the step to the previous observed frame.
- Taking a person as about 1.7 m tall, walking at about 1.0–1.6 m/s gives about 0.6–1.0 h/s, and cycling at 4–7 m/s gives about 2.5–4 h/s.
- `fps` comes from the video, or from `RefineConfig.fps` when no video is given. If neither is available, that is an error.

### 5.2 Kalman residual

- A constant-velocity Kalman filter runs over observed frames, with state `(cx, cy, w, h, vx, vy, vw, vh)` and the same noise parameters as `interpolate_tracks_rts`.
- Each observed frame gives a normalized innovation squared, `NIS_t = yᵀ S⁻¹ y`, with 4 degrees of freedom.
- A value above `switch.nis_hi` (default 18.47, the χ²₄ 99.9% quantile) is a motion break.

### 5.3 Clean embeddings

- The encoder embeds a crop every `encoder.sample_every` observed frames (default 5). Stage 2 finds candidates on these coarse samples, then embeds every observed frame within ±`switch.window` of each candidate and recomputes that candidate's score on the dense samples.
- A crop is used only if the box's maximum IoU with every other track's box in that frame (in the same file and in the context file) is below `encoder.occlusion_iou` (default 0.3). Crops that fail are marked `occluded` and excluded from appearance statistics.
- Embeddings are L2-normalized.
- The cache (`OUT.features.npz`) is keyed by input SHA-256, encoder kind, weights, `sample_every`, and `occlusion_iou`.

### 5.4 Ramp

`ramp(x; lo, hi) = clip((x − lo) / (hi − lo), 0, 1)`. If `lo > hi`, the ramp decreases. Every cue in §6 passes through a ramp whose `lo` and `hi` come from config. Each raw value and each ramp output is recorded in `signals`.

### 5.5 Appearance encoders

- **Protocol:** `AppearanceEncoder.encode(crops: list[np.ndarray]) -> np.ndarray` (N×D, float32, L2-normalized), with `name` and `dim` properties.
- **`dino`:** DINOv2 ViT-S/14 via `transformers` (`facebook/dinov2-small` by default, configurable). Uses the CLS token. Crops are resized to 224 on the long side and padded. Extra: `dnt[post-dino] = ["transformers>=4.40"]`.
- **`reid`:** `torchreid` feature extractor. The default for `person` is `osnet_x1_0` with MSMT17 weights. The `vehicle` target has no default and needs `encoder.weights` (for example, VeRi-776 weights). Extra: `dnt[post-reid] = ["torchreid"]`.
- `RefineConfig.encoder.kind` selects the encoder. Validation runs when the config loads: a missing extra raises `ImportError` with the install command, and `reid` for a vehicle without `weights` raises `ValueError`.

## 6. Stage algorithms

All thresholds below are **starting defaults**, drawn from typical walking and cycling speeds and box geometry. They are not fitted values. Each one is a config field. Tuning happens through the audit (§8.2).

### 6.1 Stage 1: screen (per track)

Each hypothesis is scored separately. If the highest score reaches `reject_below`, it becomes one event. Otherwise no event is written.

**Static non-object → `DROP{reason: "static"}`** (both targets)

Cues:

| Cue | Meaning | Ramp | Default range |
|---|---|---|---|
| `R` | 95th-percentile distance of the center from its median, divided by the median `h̃` | decreasing | 0.3 → 0.05 |
| `J` | median frame-to-frame center jitter, divided by `h̃` | decreasing | 0.03 → 0.005 |
| `C` | mean detection score | decreasing | 0.6 → 0.3 |
| `T` | track duration | increasing | 2 s → 10 s |
| `H` | hotspot count: static tracks (`ramp_R ≥ 0.5`) anywhere in the video whose median centers are within `0.5·h̃` of this track's | increasing | 1 → 4 |

Score:

```
S_static = min(static_score_cap, ramp_R · ramp_T · mean(ramp_J, ramp_C, ramp_H))
```

For vehicles, the cap is `vehicle_static_score_cap` (default 0.6), so a static vehicle only becomes `DROP` when the VLM answers "not a vehicle".

**Person in a vehicle → `DROP{reason: "in_vehicle"}`** (target `person`, requires `context_tracks`)

For each observed frame, `inside_t` is true if some context track whose class is in `context.vehicle_classes` (default `[2, 5, 7]`, COCO car, bus, truck) has `IoB(ped, veh) ≥ 0.8` and `‖v_ped − v_veh‖ < 0.3` h/s.

```
S_inveh = ramp(mean_t inside_t; 0.5 → 0.9)
```

Someone getting on or off a bus is inside for only part of the track, so they score low and are kept.

**Rider → `RECLASS`** (target `person`)

Cues:

| Cue | Meaning | Ramp |
|---|---|---|
| `F` | fraction of observed frames with `v_t > 1.8` h/s | 0.3 → 0.7 |
| `S` | smoothness, `1 − circular variance of heading` over moving frames | 0.5 → 0.9 |
| `K` | with context: fraction of frames where a context track whose class is in `context.twowheeler_classes` (default `[1, 3]`) has `IoU ≥ 0.3` and moves with the person | 0.2 → 0.6 |

```
S_rider = ramp_F · ramp_S                  (no context)
S_rider = max(ramp_F · ramp_S, ramp_K)     (with context)
```

`params.new_cls` starts as `None`. The VLM chooses cyclist, motorcycle, or scooter, which map through `reclass_map` (default `{cyclist: 1, motorcycle: 3, scooter: 36}`).

**Vehicle duplicate → `DROP{reason: "duplicate", of: <track>}`** (target `vehicle`)

For pairs that exist together for at least 10 frames:

```
D = fraction of shared frames with IoB(smaller, larger) ≥ 0.7 and ‖Δv‖ < 0.3 h/s
S_dup = ramp(D; 0.5 → 0.9)
```

The smaller track (by median area) is the one proposed for dropping.

**Orphan pass → `DROP{reason: "orphan"}`** (after stage 3)

A track with no accepted link and fewer than `orphan.min_seconds` (default 0.5 s) of observed frames:

```
S_orphan = ramp(observed_seconds; 0.5 → 0.1)
```

The orphan pass has its own bands, defaulting to accept 0.7 and reject 0.3.

### 6.2 Stage 2: switch (per track)

Tracks with fewer than 2 × `switch.min_side_seconds` of clean embeddings are skipped. In motion-only mode (§10), observed frames are counted instead of clean embeddings, here and in the per-candidate side rule below.

For each observed frame t, three components are computed.

**Appearance change**
- `A_t = 1 − cos(mean(e[t−W, t)), mean(e[t, t+W)))` over clean embeddings, where `W = switch.window` (default 1.0 s).
- The baseline is the median and MAD of `A` over all t in the track. The z-score is `z_A = (A_t − median) / (1.4826·MAD + ε)`, with `ramp(z_A; 2 → 5)`.
- **Bimodality check:** run 2-means on the track's clean embeddings. If the clusters split in time (at least 90% of cluster-1 samples come before t and at least 90% of cluster-2 samples after, for the t that best separates them) and the silhouette is at least 0.25, then `bimodal_t = 1` at that separating t. Otherwise it is 0.
- `app_t = max(ramp(z_A), bimodal_t · ramp(silhouette; 0.25 → 0.5))`

**Motion break**
- `mot_t = max(ramp(NIS_t; nis_hi/2 → nis_hi), ramp(|log(h_t / h_prev)|; 0.15 → 0.4))`

**Opportunity gate**
- `gate_t = 1` if, within ±δ frames (default `δ = 0.5 s`), another track in the same file has `IoU > 0.1` with this one, **or** the gap to the previous observed frame is greater than 1 frame. Otherwise `gate_t = 0`.

**Score**

```
S_switch(t) = gate_t · (w_app · app_t + w_mot · mot_t)   # defaults w_app = 0.65, w_mot = 0.35
```

- Candidates are local maxima of `S_switch`, with non-maximum suppression at spacing ≥ 1 s.
- Each side of a candidate needs at least `min_side_seconds` (default 0.5 s) of clean embeddings.
- One `SPLIT{cut_frame: t}` event is written per surviving candidate that reaches `reject_below`.
- When applied, frames ≥ t get a new track ID.

**Swap confirmation**

If tracks i and j both have candidates within ±δ of each other, and the IoU gate fired between them, compute:

```
cross = [cos(before_i, after_j) − cos(before_i, after_i)] + [cos(before_j, after_i) − cos(before_j, after_j)]
```

If `cross > 0`, both events get `S ← min(1, S + 0.2·ramp(cross; 0 → 0.4))`, and `signals.swap_with` is set. Stage 3 re-pairs the four pieces.

In motion-only mode (no video), `app_t = 0` and `w_mot` is renormalized to 1. With motion alone, no split is ever `AUTO_ACCEPT`, because the score is capped at `switch.motion_only_cap` (default 0.7).

### 6.3 Stage 3: link (pairs of track end → track start)

**Candidates** are pairs (i, j) of the same class, where i ends at `t_e`, j starts at `t_s`, and `g = t_s − t_e`.

**Hard gates**
1. **Gap:**
   - `1 ≤ g ≤ link.max_gap` (default 1.0 s × fps), or
   - `g ≤ link.max_gap_static` (default 10 s × fps) when i's speed over its last 0.5 s is below 0.2 h/s and j's start is within `0.5·h̃` of i's end.
2. **Overlap:** a small negative gap, `−2 ≤ g ≤ 0`, is allowed only if the boxes have `IoU ≥ 0.5` on every overlapping frame. The overlapping rows from j are dropped on merge.
3. **Size:** `h_j / h_i ∈ [1/size_ratio_max, size_ratio_max]` (default 2.0).
4. **Position:** the prediction error `d = ‖c_j(t_s) − (c_i(t_e) + v_i·g)‖ / h̃_i` must satisfy `d ≤ dist_mult · (1 + α · g/fps)` (defaults `dist_mult = 2.5`, `α = 0.5`). `v_i` is estimated with a first-order fit on i's last `vel_frames` observed frames, as `link_tracklets` does.

**Costs** (each in [0, 1])

| Cost | Definition |
|---|---|
| `c_mot` | `d / gate_radius` |
| `c_app` | `(1 − cos(mean of i's last K clean embeddings, mean of j's first K clean embeddings)) / 2`, with `K = 5`. `0.5` when either side has no clean embeddings |
| `c_size` | `ramp(|log(h_j / h_i)|; 0 → log(size_ratio_max))` |
| `c_gap` | `g / max_gap`, using `max_gap_static` for static-gated pairs |

**Birth/death prior**
- `b = 1` if i ends and j starts inside the scene: more than `0.5·h̃` from the image border, and outside any polygon in `link.entry_exit_zones` (optional, same polygon format as `Filter` zones, read with shapely). Otherwise `b = 0`.

**Score**

```
S_link = (1 − (w_mot·c_mot + w_app·c_app + w_size·c_size + w_gap·c_gap)) · (0.8 + 0.2·b)
```

Defaults are `w_mot = 0.35, w_app = 0.40, w_size = 0.10, w_gap = 0.15`. In motion-only mode `w_app = 0` and the others are renormalized.

**Assignment**
- Build the bipartite graph of gated pairs whose `S_link ≥ reject_below`.
- Split it into connected components and run `scipy.optimize.linear_sum_assignment` on each component to maximize the total `S_link`.
- **Margin:** for each chosen pair, `margin = S_link − max(second-best S for i, second-best S for j)`, or `S_link` when there is no alternative.
- The **routed score** is `S_link · ramp(margin; 0 → 0.2)`, and bands apply to it. The raw `S_link` and the margin are stored in `signals`.
- Each chosen pair becomes one `LINK` event.
- After verification, union-find merges the accepted links into chains. A chain that would put two boxes of the same object on the same frame is rejected: its lowest-scoring link is set to `applied: false` with `skipped_reason: "overlap"`.

### 6.4 Stage 4: fill

`interpolate_tracks_rts(fill_gaps_only=True, max_gap=fill.max_gap)` runs on the final tracks. `fill.max_gap` defaults to `link.max_gap`, so a link across a long static gap is joined but not filled. This keeps a waiting pedestrian's invented positions out of conflict and speed analysis, since those analyses already exclude rows with `interp == 1`.

## 7. VLM verification

### 7.1 Evidence packet

Each routed event gets **one composite JPEG**, built by `evidence.py`:
- A grid of labeled tiles on a neutral background.
- Crops are padded to 1.5× the box and upscaled so their height is at least 160 px.
- Context frames are downscaled to 768 px wide, with the event's boxes drawn and labeled "A" or "B".
- Frames are fetched in one pass per stage, sorted by frame index, with sequential reads and seeking only across large jumps.

| Event | Tiles |
|---|---|
| DROP / RECLASS | 6 crops spread evenly across the track's observed frames, plus 1 context frame at mid-life |
| SPLIT at t | Row A: 3 clean crops before t. Row B: 3 clean crops after t. Plus the context frame at t, with nearby tracks drawn |
| LINK i→j | Row A: i's last 3 clean crops. Row B: j's first 3 clean crops. Plus context frames at `t_e` and `t_s` |

If `vlm.send_context_frames: false`, context frames are left out.

### 7.2 Prompts and answers

`prompts.py` holds one prompt template per (kind, target). The VLM must choose one option and return JSON:

```json
{"answer": "<one of the options>", "confidence": 0.0, "reason": "<one sentence>"}
```

| Event | Options |
|---|---|
| person screen (DROP/RECLASS) | `pedestrian`, `cyclist`, `motorcycle_rider`, `scooter_rider`, `person_in_vehicle`, `not_a_person`, `unsure` |
| vehicle screen (DROP) | `vehicle`, `part_or_duplicate_of_another_vehicle`, `not_a_vehicle`, `unsure` |
| SPLIT, LINK | `same_individual`, `different`, `unsure` |

**How answers map to decisions** (in `verify.py`):
- **Person screen**
  - `pedestrian` → `VLM_REJECT`.
  - `person_in_vehicle` → `VLM_ACCEPT` as `DROP{in_vehicle}`.
  - `not_a_person` → `VLM_ACCEPT` as `DROP{static}`.
  - A rider answer → `VLM_ACCEPT` as `RECLASS` with `new_cls` taken from `reclass_map`.
  - The VLM's answer can therefore **change** the event's kind and reason. The originally proposed kind and reason stay in `signals.proposed`.
- **Vehicle screen**
  - `vehicle` → `VLM_REJECT`.
  - `part_or_duplicate_of_another_vehicle` → accepts only a `duplicate` event. For a `static` event it is treated as `unsure`.
  - `not_a_vehicle` → `VLM_ACCEPT` as `DROP{static}`.
- **SPLIT**
  - `different` → accept.
  - `same_individual` → reject.
- **LINK**
  - `same_individual` → accept.
  - `different` → reject.
- `unsure`, or `confidence < vlm.min_conf` (default 0.7) → `HUMAN_PENDING`.

**Votes**
- With `vlm.votes = n > 1`, the same question is asked n times at temperature `vlm.vote_temperature` (default 0.7).
- The answer is the majority option, and `confidence = majority count / n`. The model's self-reported confidence is ignored.
- A tie → `HUMAN_PENDING`.

**Invalid output**
- Non-JSON output, or an answer outside the options, is retried once with a reminder of the required format. If it fails again, the event becomes `HUMAN_PENDING` with `vlm.error`.

### 7.3 Backends

```python
@dataclass
class VLMAnswer:
    answer: str
    confidence: float
    reason: str
    raw: str

class VLMBackend(Protocol):
    name: str
    model: str
    async def ask(self, image_jpeg: bytes, prompt: str, options: list[str], temperature: float) -> VLMAnswer: ...
```

- **`openai_compat`**
  - Settings: `base_url` and `model`.
  - API key from `vlm.api_key_env` (default `OPENAI_API_KEY`, optional for local servers).
  - Sends the image as a base64 data URL.
  - Requests `response_format={"type": "json_object"}` when `vlm.json_mode: true` (the default), and falls back to parsing JSON from the text.
  - Extra: `dnt[post-vlm] = ["openai>=1.40", "anthropic>=0.40"]`.
- **`anthropic`**
  - Anthropic Messages API, sending the image as a base64 image block.
  - The key comes from `ANTHROPIC_API_KEY`.
  - The default model ID is pinned at implementation time, using the current Anthropic model reference. Config can override it.
- **`fake`**: scripted answers keyed by event ID or kind. Used in tests.
- **`none`**: no backend. Events in the uncertain band become `HUMAN_PENDING`.

### 7.4 Budget, concurrency, caching, failures

- **Budget.** `vlm.max_calls` per `refine` run (default 500; votes count individually). Events in the uncertain band are sent in order of `|algo_score − band midpoint|`, closest first. Once the budget is spent, the rest become `HUMAN_PENDING` with `vlm.error: "budget"`. Rider-subtype calls come out of the same budget.
- **Concurrency.** An `asyncio` semaphore enforces `vlm.max_concurrency` (default 4). The public API stays synchronous and runs its own event loop.
- **Cache.** `vlm.cache_dir` (default `~/.cache/dnt/vlm`) stores one JSON file per SHA-256 of (image bytes, prompt, options, backend, model, temperature, vote index). A cache hit costs nothing against the budget and sets `vlm.cached: true`.
- **Failures.**
  - Timeouts (`vlm.timeout_s`, default 60), HTTP 429, and HTTP 5xx are retried with exponential backoff: 3 attempts, starting at 2 s.
  - When the retries run out, or on any other error, the event becomes `HUMAN_PENDING` with `vlm.error`.
  - VLM errors never abort the run and never apply an edit.

## 8. Human review, audit, and summary

### 8.1 Review report

`OUT.review.html` is a static page, with images in `OUT.review/`. It has one card per `HUMAN_PENDING` event, showing:
- the evidence image,
- the kind, reason, tracks, and frames,
- `algo_score` and the top signals,
- the VLM's answer and reason, if there was one,
- accept and reject controls, plus a class picker for `RECLASS` and for screen events that the VLM redirected.

An **Export decisions** button downloads `decisions.json` in the §4.2 format.

The page makes no network requests. Its choices are saved in `localStorage` while the person works, as a convenience only. The exported file is what counts.

Cards can be filtered by stage and sorted by score.

### 8.2 Audit

`dnt-refine audit`:
1. Samples `n` events whose decision is final: `AUTO_*`, `VLM_*`, or `HUMAN_*`.
2. Stratifies the sample by (stage, decision), sampling each stratum in proportion to its size but with at least 3 per non-empty stratum, using a fixed `--seed`.
3. Builds evidence images for them, reusing `evidence.py`.
4. Writes `audit.html`, which has the same card layout with **correct** and **incorrect** controls, and exports `marks.json`.

`dnt-refine audit-score` reads the ledger and marks. For each stratum it prints and saves (`audit-score.json`) the count, the number marked correct, precision, and a 95% Wilson interval.

That precision is how success is measured (§1, criterion 6) and how bands are tuned. For example, if `link` `AUTO_ACCEPT` precision is high, `link.accept_above` can be lowered, and fewer events go to the VLM.

### 8.3 Run summary

The summary is written to the ledger header, printed at the end of the run, and returned as `result.summary`. It contains:
- tracks per class, before and after,
- observed and interpolated row counts,
- median observed track duration,
- event counts per (stage, kind, decision),
- VLM calls, cache hits, and failures.

## 9. Configuration

`RefineConfig` is a dataclass tree with `to_yaml` / `from_yaml`. Unknown keys raise `ValueError`, following `MOTBaseConfig`'s strictness. One config applies to one target, so the pedestrian and vehicle files each have their own YAML. `RefineConfig.defaults("person")` and `RefineConfig.defaults("vehicle")` return the defaults for each target.

```yaml
target: person               # person | vehicle
class_ids: [0]               # classes in the file that belong to the target
fps: null                    # null → from video
reclass_map: {cyclist: 1, motorcycle: 3, scooter: 36}
context:
  vehicle_classes: [2, 5, 7]
  twowheeler_classes: [1, 3]
motion: {height_window: 15}
encoder:
  kind: dino                 # dino | reid
  model: facebook/dinov2-small
  weights: null
  device: auto               # cuda → xpu → mps → cpu, same order as Detector
  sample_every: 5
  occlusion_iou: 0.3
  batch_size: 64
screen:
  enabled: true
  accept_above: 0.85
  reject_below: 0.40
  static_score_cap: 0.80
  vehicle_static_score_cap: 0.60
  # ramp ranges for R, J, C, T, H, inside, F, S, K, D as in §6.1
switch:
  enabled: true
  accept_above: 0.90
  reject_below: 0.50
  window: 1.0                # seconds
  min_side_seconds: 0.5
  nis_hi: 18.47
  w_app: 0.65
  w_mot: 0.35
  motion_only_cap: 0.70
link:
  enabled: true
  accept_above: 0.80
  reject_below: 0.40
  max_gap: 1.0               # seconds
  max_gap_static: 10.0       # seconds
  size_ratio_max: 2.0
  dist_mult: 2.5
  alpha: 0.5
  vel_frames: 5
  weights: {mot: 0.35, app: 0.40, size: 0.10, gap: 0.15}
  entry_exit_zones: null     # optional list of polygons [[x, y], ...]
orphan:
  enabled: true
  min_seconds: 0.5
  accept_above: 0.70
  reject_below: 0.30
fill:
  enabled: true
  max_gap: null              # null → link.max_gap
vlm:
  backend: none              # none | openai_compat | anthropic
  base_url: null
  model: null
  api_key_env: null
  json_mode: true
  min_conf: 0.7
  votes: 1
  vote_temperature: 0.7
  max_calls: 500
  max_concurrency: 4
  timeout_s: 60
  send_context_frames: true
  cache_dir: ~/.cache/dnt/vlm
```

**Validation, run when the config loads**
- `0 ≤ reject_below < accept_above ≤ 1` for every stage.
- `static_score_cap < screen.accept_above`, so static tracks can never auto-drop (§6.1).
- The link weights sum to 1.
- The chosen encoder's extra is installed.
- `reid` with the `vehicle` target has `weights` set.
- A non-`none` VLM backend has `model` set (except `anthropic`, which has a default), and its extra is installed.

Durations in config are in seconds and are converted to frames with `fps`.

## 10. Error handling and degraded modes

| Situation | Behavior |
|---|---|
| No `video` | Motion-only mode: no embeddings, and appearance weights set to 0 (§6.2, §6.3). The VLM backend is forced to `none` with a warning. Everything in the uncertain band becomes `HUMAN_PENDING`, and no review images are made (cards show signals only). |
| No `context_tracks` | In-vehicle, two-wheeler-overlap, and vehicle-duplicate cues are skipped and recorded as `null` in `signals`. |
| Max track frame exceeds the video's frame count | `ValueError` naming both numbers, since the track file does not belong to this video. |
| Malformed CSV (fewer than 6 columns, non-numeric) | `ValueError` naming the file and first bad line. |
| Empty track file | Writes an empty output, a ledger with header only, and no review. |
| Encoder or VLM extra missing, or invalid config | Fails at config load, before any processing (§9). |
| VLM errors | §7.4: `HUMAN_PENDING`, never an abort. |
| Decisions file references an unknown event ID | `ValueError` listing the unknown IDs. Nothing is applied. |

## 11. Testing

All tests below run in the default suite (CPU, no network), except where a marker is named.

### 11.1 Stage detectors, with synthetic tracks built in numpy (`tests/post/`)

- `test_screen.py`:
  - a static box with low confidence repeated at one spot → static score at the cap, routed to the VLM band, never `AUTO_ACCEPT`.
  - a person box moving inside a car box → `DROP{in_vehicle}` `AUTO_ACCEPT`.
  - a person boarding a bus (inside for 20% of frames) → no event.
  - a "person" at 3 h/s on a smooth path → `RECLASS`.
  - a walker at 0.8 h/s → no event.
  - two vehicles moving together with IoB 0.9 → `DROP{duplicate}` on the smaller.
  - an orphan of 0.2 s after linking → `DROP{orphan}`.
- `test_switch.py`, with embeddings injected through a stub encoder:
  - swapped embedding sequences at t with IoU contact → `SPLIT` at t (±1 sample), and swap confirmation raises the score.
  - an appearance drift with no gate → no event.
  - a position jump after a 5-frame gap → motion-only candidate, capped below accept.
  - a side shorter than 0.5 s → no event.
- `test_link.py`:
  - collinear fragments with a 10-frame gap and similar embeddings → `LINK` `AUTO_ACCEPT`.
  - two equally good candidates → low margin, routed to the VLM.
  - a stationary pedestrian with a 6 s gap at the same spot → linked through the static gate. With `max_gap_static` = 5 s → not linked.
  - a start near the border → prior lowers the score.
  - a chain producing overlap → lowest link skipped with `skipped_reason: "overlap"`.

### 11.2 Events, verify, apply, replay

- `test_events.py`: ledger round-trip, header contents, stable event IDs.
- `test_verify.py` with `FakeVLMBackend`:
  - band routing, including the static cap and the margin rule.
  - redirected answers (a DROP answered as a rider → RECLASS).
  - votes (majority, tie → pending), `min_conf`.
  - budget exhaustion ordered by distance to the band midpoint.
  - invalid JSON then valid on retry.
  - repeated invalid output → pending.
  - an exception → pending.
  - a cache hit makes no backend call.
- `test_apply.py`: each event kind on a small table; renumbering and `id_map`; the SPLIT-rejected-then-LINK-skipped case.
- `test_replay.py`: `apply` with a decisions file gives byte-identical output across two runs and makes zero backend calls (the fake backend raises if called). Unknown event IDs → `ValueError`.

### 11.3 Evidence and reports

- `test_evidence.py`: a cv2-generated synthetic video with moving rectangles (extends `tests/support/synthetic.py`'s `make_synthetic_video`). Checks the grid size for each event kind, crop padding and minimum height, and the effect of `send_context_frames: false`.
- `test_review.py`: the HTML contains one card per pending event, references existing image files, and contains no external URLs. Audit sampling is deterministic for a seed. `audit-score` Wilson intervals match known values.

### 11.4 Integration and compatibility

- `test_refiner.py`: end-to-end `refine` on the synthetic video with the stub encoder and the fake VLM; motion-only mode; the empty file; the CLI `run` / `apply` via `subprocess`.
- `test_post_independence.py`: the §2.2 import rule.
- The existing `tests/test_post_process.py` passes unchanged through the shim.
- `test_config.py`: YAML round-trip, unknown keys, every validation rule in §9.
- `@pytest.mark.model`:
  - `test_encoders_real.py` (DINOv2 and torchreid OSNet on sample crops: output shape and norm).
  - `test_vlm_real.py` (skipped unless `DNT_VLM_BASE_URL` / `ANTHROPIC_API_KEY` is set; one screen question on a fixture image returns a valid option).

## 12. Packaging and documentation

- **`pyproject.toml`**
  - extras: `post-dino`, `post-reid`, `post-vlm`, and `post` (the union of those three).
  - `[project.scripts] dnt-refine`.
  - No new required dependencies.
- **Docs**
  - `docs/api/post.md`, pointing mkdocstrings at `dnt.post`, `dnt.post.config`, and `dnt.post.vlm`, plus a `mkdocs.yml` `nav` entry.
  - A "Refining tracks" section in `docs/quickstart.md`, with the pedestrian and vehicle YAML examples and the review/audit loop.
- **`docs/changelog.md`:** a new-feature entry, and a note that `dnt.track.post_process` is now a shim.
- **Lint:** all new code is ruff-clean under the repo rules and numpy-style docstrings. Nothing is added to the legacy per-file baseline.
- **Version:** this adds public API, so it ships in a minor release and not in a 0.3.x patch. The version number is picked when the release is cut, following the "bump both files" rule in CLAUDE.md.

## 13. Open questions deferred to implementation

None block the plan. Each has a default chosen above, and the default holds unless evidence from the user's clips argues otherwise.

1. Whether DINOv2 or OSNet is the better default for pedestrians. The default is `dino` for both targets. Revisit after the first audit.
2. The pinned default model ID for `anthropic` (§7.3).
