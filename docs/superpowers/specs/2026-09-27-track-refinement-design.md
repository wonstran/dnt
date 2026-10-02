# dnt.refine: Track Refinement with Algorithmic Screening and VLM Verification

- **Status:** approved 2026-09-28 (rev. 7: addresses [review 2026-09-28 11:24](../../review_2026-09-28_11-24-44.md) / [response](../../response_2026-09-28_11-24-44.md); rev. 6: addresses [review 2026-09-28 11:05](../../review_2026-09-28_11-05-56.md) / [response](../../response_2026-09-28_11-05-56.md); rev. 5: addresses [review 2026-09-28 10:31](../../review_2026-09-28_10-31-56.md) / [response](../../response_2026-09-28_10-31-56.md); rev. 4: addresses [review 2026-09-28 09:53](../../review_2026-09-28_09-53-41.md) / [response](../../response_2026-09-28_09-53-41.md); rev. 3: occlusion-witnessed linking, takeover gates, class groups; rev. 2: reuse of existing dnt post-processing, no location-based filtering)
- **Date:** 2026-09-27
- **Baseline:** dnt 0.3.3 (`a290821`)
- **Roadmap:** [`design/dnt-0.4-upgrade.md`](../../../design/dnt-0.4-upgrade.md). This work belongs to sub-project F (new capabilities). It lives in `dnt/refine/`, a verb-named package like `dnt.detect`, `dnt.track`, `dnt.label` and `dnt.filter`, and supersedes the roadmap's `dnt/post/` name for this work. It does not depend on sub-projects B–E, because its only inputs are a track file and a video.

## 0. Context and agreed constraints

Pedestrian trajectories come from StrongSORT and vehicle trajectories from BoTSORT, in separate track files. They show four kinds of error:

1. **Gaps.** A trajectory breaks for some frames because detections are missing.
2. **ID fragmentation.** One object gets several IDs over time.
3. **ID switch.** One ID moves from one object to another.
4. **False tracks.** The tracked thing is not a target (a person or a vehicle).

Constraints agreed during brainstorming:

1. **Independent of tracking.** The procedure is standalone post-processing. It reads a track file and its video, plus two optional files: a context file and a hints file (§2.5). It does not import `dnt.track`, `dnt.detect`, or BoxMOT, and it makes no assumption about which tracker produced the file.
2. **Algorithms screen, the VLM verifies.** Trajectory algorithms examine every track and flag short suspicious events (seconds to minutes). A VLM reviews only the events whose algorithmic confidence is in an uncertain band.
3. **Auto-apply if sure.** Confident verdicts, algorithmic or VLM, are applied automatically. Uncertain or failed verdicts go to a human review report. Until a person decides, a pending event leaves the tracks unchanged.
4. **Pluggable VLM.** One backend interface, with a local implementation (any OpenAI-compatible server, such as vLLM or Ollama) and a cloud implementation (Anthropic).
5. **Pluggable appearance.** One `AppearanceEncoder` interface, shipping two implementations: a generic DINOv2 encoder and a torchreid ReID encoder. The config selects one per target class.
6. **Downstream uses.** Counts and volumes, pedestrian–vehicle conflict metrics (PET, TTC), and speed/behavior analysis. No error type can be traded away. ID switches and interpolated positions matter most for conflicts.
7. **No ground truth.** Success is measured by spot-checking applied edits (§8.2).
8. **False-track definitions (pedestrian file).**
   - **Drop:** static non-objects (poles, signs, mannequins, posters, reflections, shadows), and people inside vehicles (drivers, passengers, bus riders).
   - **Reclass:** cyclists, motorcycle riders, and scooter riders become class *cyclist*, *motorcycle*, or *scooter*.
   - People outside the study area are **not** handled here. Location-based filtering (study-area zones, line crossings) is the **next procedure** in the pipeline, and it runs on refinement's output.
9. **False-track definitions (vehicle file).**
   - **Drop:** non-vehicles (shadows, reflections, structure edges), and duplicate boxes on one vehicle (a trailer, or a bus split in two).
   - Parked vehicles are real and are kept.

**Reference hard case.** The Miami left turn, #12 → #81 (`/home/wonstran/repos/miami/vehicle_tracking_analysis.md` §5), contains both of the hardest errors:
- **A switch with no ID change.** A car tracked as #12 is covered by a truck that never had a track of its own. From frame 821, #12 follows the truck. Around that frame the car's label flickers 2 → 7 → 5, and the box width jumps from 56 to 103 px while the height barely changes.
- **A long occluded fragment.** The car comes back 63 frames later (6.3 s) as #81, partway through its turn.

The correct result: split #12 at about frame 821, give the truck part its own ID, link #12's car part to #81, and leave the 63-frame gap unfilled. §6.1 and §6.3 are designed so that this case works. §11 tests it with a synthetic replica and, optionally, on the real file.

## 1. Goal, scope, and success criteria

**Goal:** given a track file and its video, produce three things:
- a corrected track file in the same format;
- a ledger of every edit. Identity and class edits (`SPLIT`, `DROP`, `RECLASS`, `LINK`) are verified events with their evidence. Position edits are deterministic records: `FILL` for each filled gap and `SMOOTH` for each smoothed track (§4.1, §6.4);
- a review report of the edits that need a person.

**In scope**
1. New subpackage `dnt.refine`, with the event model, ledger, four stages plus an orphan pass, evidence builder, VLM backends, review and audit reports, `TrackRefiner`, `RefineConfig`, and a `dnt-refine` CLI.
2. Moving `interpolate_tracks_rts` and `link_tracklets` into `dnt.refine`, leaving a re-export shim at `dnt.track.post_process`, and pointing the `Filter.interpolate_tracks_rts` wrapper at the new location.
3. Two appearance encoders behind optional extras.
4. Tests, API docs page, and changelog entry.
5. Reuse of dnt's existing post-processing code (§2.6):
   - `link_tracklets`'s gates and cost become stage 3's motion part, checked by a parity test.
   - `interpolate_tracks_rts`'s Kalman model is shared with stage 1.
   - `ReClass` output is accepted as an optional hints file.
   - `dnt.engine` supplies the geometry helpers.

**Out of scope**
- Learned linkers (such as StrongSORT's AFLink) and global graph optimization. The stage 3 scorer sits behind an interface so one can be added later.
- Detector re-runs inside `refine`. `ReClass` stays in `dnt.track`, unchanged. Its output file can be passed in as hints (§2.5, §6.2).
- Camera calibration or world coordinates. All motion is measured in box heights per second.
- Location-based filtering of any kind: study-area zones, line crossings, and user-drawn polygons. That is the next procedure after refinement. `refine` takes no zone, line, or polygon input.
- An interactive review server. The review page is static HTML.

**Success criteria**
1. On an installation with none of the `refine-*` extras, `refine` runs to completion on a 10-column track file, using the default config, with no video and the frame rate passed as `fps=` (`--fps`). All three proposal stages score the tracks, and every uncertain event ends up `HUMAN_PENDING`.
2. On synthetic fixtures with injected faults (§11.1), each stage proposes the expected event, and the score lands in the expected band.
3. Replay is deterministic.
   - Re-applying a ledger with a decisions file that changes no applied edit reproduces the output byte for byte, with no re-proposals and no VLM calls.
   - When a decision changes, `apply` re-proposes from the stage that owns the changed event, or from the first stage whose input changed, whichever comes first. It takes all inputs from the ledger header and verifies them. Unchanged events keep their earlier decisions and edits (§4.2).
   - Given the same inputs, ledger, decisions, and VLM cache, the output is byte-identical.
4. `dnt.refine` imports nothing from `dnt.track`, `dnt.detect`, `dnt.label`, `dnt.filter`, or `boxmot`. A test enforces this.
5. Existing `tests/test_post_process.py` passes unchanged through the shim.
6. On the user's own clips, the audit (§8.2) reports per-stage precision of applied edits with confidence intervals. The precision targets are set by the user after the first audit, not by this spec.
7. Stage 3 with `link.mode: legacy` reproduces `link_tracklets`'s track-ID mapping on the same input (§6.3).

## 2. Architecture

### 2.1 Modules

```
src/dnt/refine/
  __init__.py          # public API: TrackRefiner, RefineConfig, interpolate_tracks_rts, link_tracklets
  io.py                # read/write track CSV (dnt 10-col, MOTChallenge adapter); read detection CSV (context); column constants
  events.py            # Event, Decision, EventKind; Ledger (JSONL read/write, replay)
  primitives.py        # scale-free speed, cv_kalman() shared with interpolate.py, NIS, occlusion mask, ramp()
  hints.py             # optional external cue files (ReClass output)
  features.py          # TrackFeatures: per-track primitives + clean embeddings; .npz cache
  encoders/
    __init__.py        # AppearanceEncoder protocol, make_encoder(cfg)
    dino.py            # DINOv2 encoder (extra: dnt[refine-dino])
    reid.py            # torchreid encoder (extra: dnt[refine-reid])
  switch.py            # stage 1: SPLIT proposals
  screen.py            # stage 2: DROP / RECLASS proposals; orphan pass
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

`dnt.refine` may import `dnt.shared`, `dnt.engine`, and third-party libraries only. `dnt.engine` is pure numeric code. Its one native dependency, `cython_bbox`, is already a required dependency of dnt. It must not import `dnt.track`, `dnt.detect`, `dnt.label`, `dnt.filter`, or `boxmot`. `tests/test_refine_independence.py` enforces this by importing `dnt.refine` in a subprocess and checking `sys.modules`.

Evidence rendering uses cv2 directly rather than `Labeler`, so the rule holds.

Encoders and VLM backends import their heavy dependencies (`transformers`, `torchreid`, `openai`, `anthropic`) lazily, inside the constructor.

### 2.3 The move of existing functions

- `interpolate_tracks_rts` moves to `dnt/refine/interpolate.py`.
- `link_tracklets` moves to `dnt/refine/link.py`.

Their signatures stay the same. `interpolate_tracks_rts` gets two backward-compatible changes (§6.4): rows flagged as filled are no longer used as measurements, and a new `protected_gaps` keyword marks gaps that must not be filled. For raw tracker output, which never contains filled rows, the output is unchanged. `dnt/track/post_process.py` becomes a shim: `from ..refine.interpolate import interpolate_tracks_rts` and `from ..refine.link import link_tracklets`, with `__all__`. That keeps `from dnt.track import link_tracklets` and existing scripts working. The shim direction (track → refine) does not violate §2.2, which only restricts what `dnt.refine` imports.

Two pieces of existing code are **factored out and shared**, rather than copied:

- **Link gates.** The gate and cost logic inside `link_tracklets` becomes module-level helpers in `link.py`: `_iou_xywh`, `_estimate_velocity`, and a new `_legacy_gate_cost(end, start, …) -> float | None`, which returns the legacy cost or `None` when a gate fails. Both `link_tracklets` and stage 3 call them. `link_tracklets`'s output does not change.
- **Kalman model.** The Kalman setup inside `interpolate_tracks_rts` becomes `primitives.cv_kalman(process_var, meas_var_pos, meas_var_size)`. Stage 4 (through `interpolate_tracks_rts`) and stage 1's motion test (§5.2) both use it.

`Filter.interpolate_tracks_rts` is a backward-compatible wrapper. It is changed to import from `dnt.refine.interpolate`.

### 2.4 Entry points

`TrackRefiner` follows the shape of `Detector` and `Tracker`:
- a verb-named package, `dnt.refine`, like `dnt.detect` and `dnt.track`;
- a class that is configured once, with `config=` or `config_yaml=`. `device=` overrides `encoder.device`, the way `Tracker(device=...)` does;
- a per-file method that takes input and output paths and returns a DataFrame;
- a batch method.

Common keywords match `Tracker.track`: `video_file`, `video_index`, `video_tot`, `message`, and `verbose` (a progress bar).

```python
from dnt.refine import RefineConfig, TrackRefiner

refiner = TrackRefiner(config=RefineConfig.defaults("person"))  # or TrackRefiner(config_yaml="ped.yaml")
tracks = refiner.refine(
    "ped_track.txt", "ped_refined.txt",
    video_file="cam1.mp4",         # optional; without it, motion-only mode (§10)
    context_file="veh_track.txt",  # optional; a track file or a detection file (§2.5)
    reclass_file="reclass.csv",    # optional; ReClass output (§2.5, §6.2)
    fps=None,                      # required when there is no video (§5.1)
)                                  # -> DataFrame, like Tracker.track()
refiner.last_result                # RefineResult: tracks, ledger_path, review_path, summary, events

refiner.refine_batch(track_files, video_files=video_files, output_path="refined/")  # -> list[str]

tracks = refiner.apply(            # replay after review (§4.2); inputs come from the ledger header
    "ped_refined.ledger.jsonl", "ped_refined_v2.txt", decisions_file="decisions.json",
)
```

**`refine_batch`.** Its signature is `refine_batch(track_files, video_files=None, output_path=None, context_files=None, reclass_files=None, fps=None, is_overwrite=False, is_report=True, message="", verbose=True) -> list[str]`.
- **Pairing.** Files are matched by position, as in `track_batch`.
- **Naming.** Each output is `<output_path>/<base>_refined.txt`, where `<base>` is the track file's stem with any trailing `_track` removed.
- **Existing outputs** are skipped unless `is_overwrite`. With `is_report`, skipped outputs are still listed in the return value.
- **`output_path` is required**, because every run writes a ledger next to its output.

```
dnt-refine run    TRACKS [--video V] [--fps F] [--context C] [--reclass-hints R] --config CFG --out OUT
dnt-refine apply  --ledger L [--decisions D.json] [--tracks T] [--video V] [--context C] [--reclass-hints R] [--features F] [--no-vlm] [--no-fill] --out OUT
dnt-refine audit  --ledger L [--video V] --n 50 [--seed S]
dnt-refine audit-score --ledger L --marks M.json
```

The CLI is a thin `argparse` wrapper over `TrackRefiner`, registered as `[project.scripts] dnt-refine = "dnt.refine.cli:main"`.

### 2.5 Input and output contract

**Input**
- A headerless dnt track CSV with 10 columns: `frame, track, x, y, w, h, score, cls, r3, r4`.
- Column 8 may already be `interp` from an earlier interpolation. A value of `1` means filled, and `0` or the legacy `-1` means observed. dnt's tracker always writes `-1` here, so a `1` can only come from an earlier interpolation.
- **Filled input rows are removed on input** and never reach the output. Every stage works on observed rows only, and stage 4 fills gaps again from observed rows. The old fills were computed for identities that refinement may change, so they cannot be trusted. The ledger header records how many rows were removed.
- MOTChallenge files (`frame, id, x, y, w, h, conf, x3d, y3d, z3d`) are read with `io.read_tracks(path, fmt="mot")`. That format has no class, so `cls` is set to `RefineConfig.class_ids[0]`.
- **Context** (optional) is used by the stage 2 cues that need other objects' boxes (in-vehicle, two-wheeler overlap, vehicle duplicate). It is either:
  - a dnt track file (10 columns), or
  - a dnt detection file (8 columns: `frame, res, x, y, w, h, conf, class`, as `Detector` writes it).

  The format is detected from the column count, or can be set with `context.format`. Only boxes and classes are used. For the vehicle target, the vehicle file itself is the context for the duplicate cue.
- **Reclass hints** (optional): the CSV that `ReClass.re_classify(out_file=...)` writes, with header `track, cls, avg_score`.
  - It is keyed by the input file's raw track IDs. `ReClass` judges a whole raw track from a sample of its frames, so the hint cannot say which part of a split track it came from.
  - A hint is **localized** only when the scored unit covers its whole raw track: the raw track has no applied `SPLIT`, and screening is scoring the whole track rather than a segment.
  - After a split, or during segment scoring (§6.2), the hint is **unlocalized**, and it only informs review (§6.2).
  - Rows with unknown track IDs are ignored, with a warning.

**Output**
- `OUT`: headerless 10-column CSV in the dnt layout, with column 8 as `interp` (`0` observed, `1` filled). This is the same layout `interpolate_tracks_rts` writes today. Sorted by `frame, track`. Track IDs are renumbered contiguously from 1, and `ledger.id_map` records the renumbering.
- `OUT.ledger.jsonl`: header line, then one line per event (§4.2).
- `OUT.review.html` plus `OUT.review/` (images), written only when at least one event is `HUMAN_PENDING`.
- `OUT.features.npz`: the embedding cache (§5.3). Reused only when its full key matches.

### 2.6 Reuse of existing dnt post-processing

| Existing code | Role in refinement |
|---|---|
| `link_tracklets` | Its gates and cost are stage 3's motion part (§6.3). `link.mode: legacy` reproduces it exactly. |
| `interpolate_tracks_rts` | Stage 4, with two backward-compatible changes (§6.4). Its Kalman model is shared with stage 1 (§5.2). |
| `ReClass.re_classify` | Not imported. Its output file is an optional rider cue and subtype source (§6.2). |
| `dnt.engine.ious`, `iobs`, `cluster_by_gap` | Geometry and frame-run helpers (§5.6). |
| `Labeler.draw_track_clips` | Not imported. The review report prints a ready-to-run snippet for each pending event (§8.1). |
| `Filter.deduplicate_boxes` | Not part of refinement. It is a detection-stage step. The docs recommend it before tracking, because it stops duplicate and nested boxes from becoming tracks at all. |

Not reused:
- `engine.interpolate_bboxes`: cubic splines overshoot, which creates false speed spikes.
- `shared/files.read_track`: it forces columns 7–9 to int, which fails on the NaN values in interpolated rows.

## 3. Processing order

```
raw tracks (filled rows removed)
  └─ 1 switch  ── SPLIT ───────────┐
  └─ 2 screen  ── DROP / RECLASS ──┤  each stage: propose → verify (§7) → apply
  └─ 3 link    ── LINK ────────────┤  before the next stage runs
  └─ orphan    ── DROP ────────────┤
  └─ 4 fill    ── FILL / SMOOTH records (deterministic, never verified)
clean tracks
```

The order stops errors from compounding.
- **Split first.** A track that follows two objects must be cut before either part is judged. Otherwise a whole-track `DROP` of a false second part (for example, a passenger inside a vehicle) would also remove a real first part. Splits that are still unresolved are handled by segment-aware screening (§6.2).
- **Screen before link.** A pole's static track must not be linked to a real pedestrian.
- **Orphans after link.** Short fragments get the chance to be linked before being judged too short.
- **Fill last.** Interpolation runs over the final identities, only from observed rows, and never across a rejected link or a protected gap.

When review changes a decision, `apply` re-runs everything from the earliest stage whose applied edits changed (§4.2).

Each stage sees the tracks produced by the previous stage's applied events. VLM calls for one stage finish before the next stage starts.

## 4. Event model, bands, and ledger

### 4.1 Event

```python
class EventKind(StrEnum): DROP, RECLASS, SPLIT, LINK, FILL, SMOOTH

class Decision(StrEnum):
    AUTO_ACCEPT, AUTO_REJECT,        # decided by algo_score alone
    VLM_ACCEPT, VLM_REJECT,          # decided by a sure VLM verdict
    HUMAN_PENDING,                   # waiting for a person; not applied
    HUMAN_ACCEPT, HUMAN_REJECT       # decided by a person via decisions.json

@dataclass
class Event:
    id: str                  # f"{stage}-r{round}-{seq:06d}"; kept by matched events in later rounds
    proposal_key: str        # immutable key of the proposal, fixed before verification (§4.2)
    round: int               # review round that proposed it (0 = first run)
    stage: str               # "switch" | "screen" | "link" | "orphan" | "fill"
    kind: EventKind          # the proposed kind; never changed by verification
    tracks: list[int]        # IDs as they were when the stage ran
    lineage: list[list]      # per track: [[raw_id, f0, f1], ...], the raw rows it is made of
    frames: tuple[int, int]  # the frame span the event concerns
    params: dict             # the proposed params; never changed by verification
                             # DROP: {reason, spans | None}; RECLASS: {new_cls | None, spans | None};
                             # SPLIT: {cut_frame}; LINK: {gate, gap: [t_e, t_s]};
                             # FILL: {gap: [f_before, f_after], n_rows}; SMOOTH: {n_rows}
    edit: dict | None        # the final edit {kind, params} when accepted; None otherwise
    algo_score: float        # [0, 1]
    signals: dict            # every named cue value and ramp output behind algo_score
    decision: Decision       # current decision; HUMAN_PENDING events are written too
    decision_history: list[dict]  # [{decision, round, source: auto | vlm | human}, ...]
    vlm: dict | None         # {backend, model, answer, confidence, votes, reason, evidence, cached}
    applied: bool
```

`EventKind` and `Decision` subclass `StrEnum`.

**Proposal and edit are separate.** `kind` and `params` record what the stage proposed, and they never change. `edit` records what is actually applied when the event is accepted.
- It equals the proposal unless verification changed it: the VLM redirected it (§7.2), or a person picked another class.
- `apply.py` applies `edit`, never `kind`/`params` directly.

**Position records.** `FILL` and `SMOOTH` are written by stage 4.
- They are always `AUTO_ACCEPT`, never routed to the VLM, and never sent to the review report.
- They exist so that the ledger accounts for every position edit, and so the audit can sample them (§8.2).
- Original positions are not copied into the ledger. They can be reconstructed exactly: `dnt-refine apply --no-fill` replays the ledger without stage 4 and reproduces the table as it was before filling and smoothing.

### 4.2 Ledger

A JSONL file.
- **Line 1** is the header:
  - the dnt version and the config (the full YAML, as a dict);
  - `inputs`, a record of every input the run read (below);
  - `fps` with its source, and `frame_size`;
  - `id_map`, the count of filled input rows removed (§2.5), and the before/after summary (§8.3).

  `inputs` holds:
  - `tracks`: `{path, sha256, format}`;
  - `video`: `{path, fingerprint, frame_count}` or `null` (§5.3 defines the fingerprint);
  - `context`: `{path, sha256, format}` or `null`;
  - `hints`: `{reclass: {path, sha256}}` or `null`;
  - `features`: `{path, sha256, cache_key}` or `null`, the embedding cache the run wrote (§5.3).

  Paths are stored as given and also resolved to absolute paths.
- **Each later line** is one `Event`. **Every proposed event is written, including `HUMAN_PENDING` ones**, with its current decision. The review page and `apply` refer to pending events by their ledger `id`. A stage writes its events once its verification is done (so after its last pass, in stage 3).
- **Ledgers are never edited in place.** `apply` writes a new ledger for its output (`OUT.ledger.jsonl`) that contains the complete, current state:
  - Its header records `parent: {path, sha256}` (the ledger it replayed) and `round` (the parent's round + 1).
  - Every event keeps its `id` across rounds when it is matched by `proposal_key`, so a reviewer's IDs stay valid. Only new events get new IDs, with the new round in them.
  - Each event carries `decision` (current) and `decision_history`, a list of `{decision, round, source}`, where `source` is `auto`, `vlm`, or `human`.

**Lineage.** Every event records, for each track it involves, the raw rows that track is made of: `[raw_id, f0, f1]` spans. The header's `id_map` and the events' lineage let any output row be traced back to its input row.

**Proposal key.** `proposal_key` is the SHA-256 of `(stage, proposed kind, lineage of the involved tracks, the proposal's defining params)`, where the defining params are the cut frame, the gap span, the proposed reason, or the spans.
- It is computed when the stage proposes the event, before any verification, and it never changes.
- A VLM redirection, or a person's class choice, changes `edit` but not the key. On re-proposal, the same proposal gets the same key and takes over the recorded `edit`.
- A decision about "cut raw track 12 at frame 821", or "link the raw-12 part ending at 820 to raw 81 starting at 883", keeps the same key however the tracks happen to be numbered.

**Inputs for replay.** `TrackRefiner.apply(ledger_file, out_file, decisions_file=None, *, track_file=None, video_file=None, context_file=None, reclass_file=None, features_file=None, vlm=True, fill=True, video_index=None, video_tot=None, message="", verbose=True) -> DataFrame` takes its inputs from the ledger header. Each `*_file` keyword overrides only the *location* of a recorded input, for example after files have moved. It never changes which input is used. Before any processing:
- **Resolution.** Each recorded input is taken from its override if one is given, otherwise from its recorded path.
- **Verification of defining inputs.** The **defining inputs** are `tracks`, `video`, `context`, and `hints`, because they determine the proposals. Each one's SHA-256 (the fingerprint, for the video) must equal the recorded value. A mismatch raises `ValueError`, naming the input and both hashes. The fix is to run `dnt-refine run` again: `apply` never re-proposes with different inputs.
- **The feature cache is a derived input.** It is computed entirely from the defining inputs and the encoder settings, so it does not follow the rule above. A bad cache is a *cache miss*, never a different input. The rule for it is below.
- **Missing files.** A recorded `tracks`, `context`, or `hints` file that cannot be found raises `ValueError`.
- **Appearance for re-proposal.** Stages 1 and 3 use embeddings, so they need appearance if they will re-run and `encoder.kind` is not `none`. Which stages re-run is known before processing, from the overrides. The cache is read and checked only in that case; otherwise it is not touched. When it is needed, it is in one of three states:
  - **valid:** the file exists, its SHA-256 matches `inputs.features.sha256`, and its stored key matches `cache_key`. A cache written by `refine` is **complete for any replay**: the set of embedded samples depends only on the raw inputs and on stage 1's proposals, and stage 1's proposals never change (§5.3);
  - **missing:** the file is not found at its override or recorded path;
  - **invalid:** the file exists, but its SHA-256 or key differs, or it cannot be read.

  | Cache | Verified video available | Behavior |
  |---|---|---|
  | valid | either way | Use the cache. |
  | missing | yes | Cache miss (logged at INFO): recompute from the video. |
  | invalid | yes | Cache miss (logged at **WARNING**, naming the recorded and found hashes): recompute from the video. The invalid file is left untouched. |
  | missing | no | `ValueError` before processing: "feature cache not found at <path>; restore it or the video at <path>, or pass `--features` / `--video`". |
  | invalid | no | `ValueError` before processing: "feature cache at <path> does not match the ledger (recorded <sha>, found <sha>); restore the original cache or the video at <path>". |

  After a recomputation:
  - `apply` writes the new cache next to its output (`OUT.features.npz`), never over the file it rejected.
  - The new ledger's header records the new cache in `inputs.features`, together with `features_recomputed: true`.
  - With `encoder.device: cpu`, recomputation reproduces the original embeddings, and so the original output. On a GPU, floating-point nondeterminism can shift embeddings slightly, and a score that sits right on a threshold can then fall on the other side. That is why the header flag is recorded.
- **Missing video with a valid cache.** Allowed, with a warning. New events that need evidence images become `HUMAN_PENDING` without images.
- **No new inputs.** An input that the original run did not have (for example `context` when the header says `null`) is rejected, because it would change the proposals.
- **Settings.** `fps` and `frame_size` come from the header.

`apply` runs `Ledger.replay`:
1. **Overrides.** Decisions come from `decisions.json` (`{event_id: "accept" | "reject" | {"accept": true, "new_cls": 3}}`). Each event ID is resolved to its `proposal_key`. An override that differs from the recorded decision marks that event as **changed**.
2. **Unchanged stages.** For each stage in order: if its input tracks are identical to those recorded, **and none of its own events changed**, the recorded events are applied as they are, with no proposals and no VLM calls.
3. **Re-proposed stages.** The first stage whose input differs, **or which owns a changed event**, re-runs its proposals, and so does every later stage.
   - Proposals are deterministic, so on an unchanged input they come out with the same keys.
   - The overridden decisions are then applied by key, and the stage continues from there. In stage 3 this means re-assignment runs without a newly rejected edge (§6.3), even though the stage's input table is unchanged.
   - A re-proposed event whose `proposal_key` matches an earlier event takes over that event's decision, VLM answer, and final `edit`.
   - Only events with no match are routed (§4.3). With `vlm=False` (`--no-vlm`), or with no video for evidence, they become `HUMAN_PENDING`.
   - New events get `round = previous round + 1`, and a new review report is written for any that are pending.
4. **Fill.** Stage 4 runs last, unless `fill=False` (`--no-fill`).

Re-proposal is how a human decision takes effect. For example, accepting a pending `SPLIT` gives its tail the chance to be linked in stage 3. Rejecting a pending `LINK` re-runs stage 3 itself, so its endpoints can be re-assigned (§6.3).

Stage 1 has no stage upstream of it, so its proposals never change. Only its decisions can.

**Equivalence.** Accepting a pending decision through `apply` gives the same output tracks as a fresh `refine` in which that decision was made during the run.

Embeddings are read from the verified cache (§5.3). Its key does not depend on decisions, so re-proposal never re-encodes frames that were already encoded.

### 4.3 Bands

For each stage, `accept_above` and `reject_below`, with `reject_below < accept_above`:

| algo_score | Route |
|---|---|
| `≥ accept_above` | `AUTO_ACCEPT` |
| `< reject_below` | `AUTO_REJECT` (still written to the ledger, so audits can sample rejections) |
| otherwise | VLM (§7). With `vlm.backend: none`, `HUMAN_PENDING` |

**Exceptions**
- **Static screen.** The static-object score is capped at `screen.static_score_cap` (default 0.80). Keep the cap below `screen.accept_above`, so static tracks never auto-drop (§6.2).
- **Rider subtype.** A `RECLASS` whose rider score is `AUTO_ACCEPT` still needs one VLM call to choose the subtype, unless a ReClass hint has already settled it (§6.2). That call only picks the subtype and cannot overturn the rider decision. If the VLM answers with a non-rider option (for example `pedestrian`), the algorithm and the VLM disagree, and the event becomes `HUMAN_PENDING`. It also becomes `HUMAN_PENDING` if `params.new_cls` is still `None` after verification.
- **Link ambiguity.** In stage 3, a pair whose margin over the next-best alternative is below `link.margin_min` (default 0.1) is capped at `link.ambiguous_cap` (default 0.75, below `link.accept_above`). An ambiguous pair is never auto-accepted. Ambiguity alone also never auto-rejects a pair: if its score reaches `reject_below`, it goes to the VLM (§6.3).
- **Mixed screen.** A partial `DROP` or `RECLASS` (§6.2) is capped at `screen.mixed_score_cap` (default 0.75), so it is never auto-applied.
- **Occluded link.** A link through the occlusion-witness gate (§6.3) is capped at `link.occluded_score_cap` (default 0.75). Keep the cap below `link.accept_above`, so these links are never auto-accepted. Waiting in a queue also produces occlusion, so a witness makes a link plausible, not certain.

## 5. Shared primitives

### 5.1 Scale-free motion

- `h̃_t` is the rolling median of box height over `motion.height_window` frames (default 15).
- Speed is `v_t = ‖c_t − c_{t−Δ}‖ / (h̃_t · Δ/fps)`, in **box heights per second** (h/s), where `c` is the box center and `Δ` is the step to the previous observed frame.
- Taking a person as about 1.7 m tall, walking at about 1.0–1.6 m/s gives about 0.6–1.0 h/s, and cycling at 4–7 m/s gives about 2.5–4 h/s.
- **Frame rate.** `fps` is resolved in this order: the `fps=` argument (`--fps`), then `RefineConfig.fps`, then the video.
  - If an explicit value differs from the video's rate by more than 1%, a warning is logged and the explicit value is used.
  - With no video and neither setting, `refine` raises `ValueError` before any processing, and the message asks for `fps=`. There is no default frame rate: every threshold in seconds depends on it, so a guess would silently change them all.
  - The ledger header records the value and where it came from, and `apply` reuses it.

### 5.2 Kalman residual

- Stage 1 uses the same constant-velocity model as stage 4: `primitives.cv_kalman(...)`, factored out of `interpolate_tracks_rts` (§2.3). As a result, both stages agree on what normal motion is. The model has:
  - state `[cx, vx, cy, vy, w, vw, h, vh]`, one step per frame;
  - `Q` from `Q_discrete_white_noise(dim=2, var=process_var)` for each (value, rate) pair;
  - `R = diag(meas_var_pos, meas_var_pos, meas_var_size, meas_var_size)`;
  - defaults `process_var = 10.0`, `meas_var_pos = 25.0`, `meas_var_size = 16.0`, set in config as `motion.*`.
- The forward pass predicts through missed frames and updates on observed ones. Each observed frame gives a normalized innovation squared, `NIS_t = yᵀ S⁻¹ y`, with 4 degrees of freedom.
- A value above `switch.nis_hi` (default 18.47, the χ²₄ 99.9% quantile) is a motion break.

### 5.3 Clean embeddings

- The encoder embeds a crop every `encoder.sample_every` observed frames (default 5). Stage 1 finds candidates on these coarse samples, then embeds every observed frame within ±`switch.window` of each candidate and recomputes that candidate's score on the dense samples.
- A crop is used only if the box's maximum IoU with every other box in that frame (in the input file and in the context file) is below `encoder.occlusion_iou` (default 0.3). Crops that fail are marked `occluded` and excluded from appearance statistics.
- **Minimum box size, per stage.** Stage 1 ignores the appearance samples of boxes whose longer side, `max(w, h)` in pixels as in the track file, is below `switch.min_crop_px` (default 40). Stage 3 does the same with `link.min_crop_px` (default 0: every clean crop). `0` turns a stage's filter off. In that stage a filtered row is treated like an occluded one, and the coarse ordinals still count every observed row. Stage 1 skips a track or side without enough samples, so a track of small boxes gets no `SPLIT` event at all (section 6.1). Stage 3 uses `c_app = 0.5` for a side without samples (section 6.3). Rows below the smaller of the two values are never embedded. The other rows are embedded whichever stage asks for them, so the cache does not depend on the stages' filters beyond that smallest value. On three 640x480 pedestrian clips, the ID-switch splits found from small crops cut single pedestrians, while most audited links scored from small crops were correct; hence the two defaults. `refine` logs at INFO how many coarse samples each stage uses and how many its filter leaves out.
- The mask is computed from the **raw** input boxes and the context boxes. It does not change when decisions change, so embeddings stay valid across re-proposals.
- Embeddings are L2-normalized.
- **The embedded samples do not depend on decisions.** They are the coarse samples, which depend only on the raw tracks and `sample_every`, plus the dense samples around every stage 1 candidate, whatever its decision. Stage 1's proposals are deterministic and never change on replay. So the cache that `refine` writes covers every embedding any replay can request, including the first and last clean samples of tails created by splits that are accepted later.
- **Cache.** `OUT.features.npz` stores embeddings per (raw track ID, frame), under a key made of:
  - the input track file's SHA-256;
  - the video's **fingerprint**: the SHA-256 of the **entire file**, read in 8 MiB chunks, plus the file size and frame count;
  - the context file's SHA-256, or `none`;
  - encoder kind, model name, and a digest of the weights actually loaded (the SHA-256 of the weights file; for a Hub model, which has no file path, the SHA-256 of the loaded parameters, so a model name that later resolves to different weights misses the cache);
  - `sample_every`, `occlusion_iou`, and the smallest embedded box size, `min(switch.min_crop_px, link.min_crop_px)`;
  - crop preprocessing: padding factor, resize target, and normalization;
  - `FEATURES_VERSION`, a constant bumped whenever the crop or embedding code changes.

  In `refine`, if any part of the key differs, the cache is discarded (logged at INFO) and embeddings are recomputed. In `apply`, the cache is a derived replay input, handled as a cache miss when the video is available and as an error otherwise (§4.2).
- **Fingerprint.** It is computed once per run and shared by the cache key, the ledger header, and `apply`'s input check. Hashing the whole file costs about 1 s per 1–2 GB, which is small next to decoding. A partial hash, or a hash of size and modification time, is not used, because frames changed later in the file would go undetected.

### 5.4 Ramp

`ramp(x; lo, hi) = clip((x − lo) / (hi − lo), 0, 1)`. If `lo > hi`, the ramp decreases. Every cue in §6 passes through a ramp whose `lo` and `hi` come from config. Each raw value and each ramp output is recorded in `signals`.

### 5.5 Appearance encoders

- **Protocol:** `AppearanceEncoder.encode(crops: list[np.ndarray]) -> np.ndarray` (N×D, float32, L2-normalized), with `name` and `dim` properties.
- **`dino`:** DINOv2 ViT-S/14 via `transformers` (`facebook/dinov2-small` by default, configurable). Uses the CLS token. Crops are resized to 224 on the long side and padded. Extra: `dnt[refine-dino] = ["transformers>=4.40"]`.
- **`reid`:** `torchreid` feature extractor. The default for `person` is `osnet_x1_0` with MSMT17 weights. The `vehicle` target has no default and needs `encoder.weights` (for example, VeRi-776 weights). Extra: `dnt[refine-reid] = ["torchreid"]`.
- **`none`:** no appearance. Stages 1 and 3 run in motion-only mode (§10). The video, if given, is still used for VLM evidence.
- `RefineConfig.encoder.kind` selects the encoder. A config that is structurally wrong is rejected when it loads: an unknown kind, or `reid` for a vehicle without `weights`, raises `ValueError`.
- **Dependency checks are deferred.** They run at the start of `refine` and `apply`, before any processing, and only for the components that run will use. The encoder's extra is required only when a video is given and `kind` is not `none`. A missing extra raises `ImportError` with the install command, and suggests `encoder.kind: none` as an alternative.

### 5.6 Geometry

- IoU and IoB come from `dnt.engine.ious` and `dnt.engine.iobs`. `dnt.refine` has no second implementation.
- Runs of consecutive observed frames (gaps) come from `dnt.engine.cluster_by_gap`.
- `ious` uses cython_bbox's pixel-inclusive convention, where a box's area is `(w+1)(h+1)`. The thresholds in §6 are applied to those values as they are. For boxes wider than about 20 px the difference is negligible.

## 6. Stage algorithms

All thresholds below are **starting defaults**, drawn from typical walking and cycling speeds and box geometry. They are not fitted values. Each one is a config field. Tuning happens through the audit (§8.2).

### 6.1 Stage 1: switch (per track)

Tracks with fewer than 2 × `switch.min_side_seconds` of clean embeddings are skipped. In motion-only mode (§10), observed frames are counted instead of clean embeddings, here and in the per-candidate side rule below.

For each observed frame t, three components are computed.

**Appearance change**
- `A_t = 1 − cos(mean(e[t−W, t)), mean(e[t, t+W)))` over clean embeddings, where `W = switch.window` (default 1.0 s).
- The baseline is the median and MAD of `A` over all t in the track. The z-score is `z_A = (A_t − median) / (1.4826·MAD + ε)`, with `ramp(z_A; 2 → 5)`.
- **Bimodality check:** run 2-means on the track's clean embeddings. If the clusters split in time (at least 90% of cluster-1 samples come before t and at least 90% of cluster-2 samples after, for the t that best separates them) and the silhouette is at least 0.25, then `bimodal_t = 1` at that separating t. Otherwise it is 0.
- `app_t = max(ramp(z_A), bimodal_t · ramp(silhouette; 0.25 → 0.5))`

**Motion break**
- `mot_t = max(ramp(NIS_t; nis_hi/2 → nis_hi), ramp(jump_t; 0.15 → 0.4))`
- `jump_t = max(|log(w_t / w_prev)|, |log(h_t / h_prev)|)`. It uses both dimensions because a takeover can change mostly the width. In the reference case the width goes from 56 to 103 px (jump 0.61), while the height goes from 59 to 68 px (0.14).

**Opportunity gate**
`gate_t = 1` if any of the following holds within ±δ frames (default `δ = 0.5 s`). Otherwise `gate_t = 0`.
  1. **Contact:** another track in the same file has `IoU > 0.1` with this one.
  2. **Gap:** the gap to the previous observed frame is greater than 1 frame.
  3. **Class change:** the per-row `cls` changes at least once (`switch.class_change_gate`, default on). Class groups (§6.3) don't apply here: a car ↔ truck flicker opens the gate too.
  4. **Size jump:** `jump_t ≥ log(switch.size_gate)` for some observed frame, with default `size_gate = 1.5`. In the reference case the jump is 1.84× (width), so a 2× gate would miss it.

Conditions 3 and 4 cover an object that takes over a track without ever having had a track of its own, like the truck in the reference case. In that situation, condition 1 never fires.

A gate condition only makes a switch *possible*. The score still needs appearance or motion evidence, so a class flicker alone, with no change in appearance or motion, produces no event. `signals.gate` records which conditions fired.

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

### 6.2 Stage 2: screen (per track)

Each hypothesis is scored separately. If the highest score reaches `reject_below`, it becomes one event. Otherwise no event is written.

**Segment-aware scoring.** Screening runs on the tracks produced by stage 1, but a track can still follow two objects. That happens when a `SPLIT` is `HUMAN_PENDING`, or when a stage 1 candidate scored below `switch.reject_below` but at least `screen.segment_at` (default 0.3). The cut frames of those candidates divide the track into *segments*, and each hypothesis is scored on every segment as well as on the whole track.
- **Every segment reaches `reject_below` for the hypothesis:** the event covers the whole track as usual.
- **Only some segments do:** the event is **partial**.
  - `params.spans` lists the frame spans of the supported segments.
  - The score is capped at `screen.mixed_score_cap` (§4.3), so a person or the VLM decides.
  - The evidence image shows crops from the supported and unsupported segments in separate rows.
  - If accepted, only rows inside `spans` are dropped or reclassed.
- A whole-track `DROP` or `RECLASS` is therefore never auto-applied to a track whose segments disagree.

In the case that motivated this rule, an ID follows a pedestrian and then a passenger inside a vehicle, with the switch still pending. The in-vehicle cue holds only on the second segment, so the result is a partial `DROP` for review, and the pedestrian's frames are kept.

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

**Person in a vehicle → `DROP{reason: "in_vehicle"}`** (target `person`, requires `context`)

For each observed frame, `inside_t` is true if some context box whose class is in `context.vehicle_classes` (default `[2, 5, 7]`, COCO car, bus, truck) has `IoB(ped, veh) ≥ 0.8`, and the two move together. How "move together" is tested depends on the context format:
- **Track context:** `‖v_ped − v_veh‖ < 0.3` h/s.
- **Detection context:** detections carry no velocity, so persistence is used instead. The person is moving (`v_t ≥ 0.3` h/s), a vehicle detection contains it at both t and the previous observed frame, and those two vehicle boxes have `IoU ≥ 0.5`.

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
| `K` | with context: fraction of frames where a context box whose class is in `context.twowheeler_classes` (default `[1, 3]`) has `IoU ≥ 0.3` and moves with the person (same move-together test as the in-vehicle cue, per context format) | 0.2 → 0.6 |
| `P` | with a **localized** ReClass hint (§2.5): the hint's `avg_score` when its `cls` is a key of `hints.reclass_class_map`, otherwise 0 | 0.75 → 0.9 |

```
S_rider = max(ramp_F · ramp_S, ramp_K, ramp_P)    # ramp_K and ramp_P count as 0 when their input is absent
```

**Subtype.** `params.new_cls` starts as `None`, and one of two sources settles it:
1. **Hint.** Only a localized hint can settle the subtype. If `ramp_P ≥ hints.subtype_min` (default 1.0, which means `avg_score ≥ 0.9`), the subtype comes from `hints.reclass_class_map` (default `{1: cyclist, 3: motorcycle, 36: scooter}`). No subtype VLM call is made, and `signals.subtype_source` is `"reclass"`.
2. **VLM.** Otherwise the VLM chooses cyclist, motorcycle, or scooter.

Either way, the subtype maps to a class ID through `reclass_map` (default `{cyclist: 1, motorcycle: 3, scooter: 36}`).

If the event is in the VLM band anyway and the VLM names a different rider subtype than the hint, the event becomes `HUMAN_PENDING`.

**Unlocalized hints.** A hint whose raw track has been split, or whose track is being scored in segments, is not used as a cue: `ramp_P` counts as 0, and the hint cannot settle the subtype. It is recorded in `signals.hint_unlocalized` and shown on the review card, but it is not included in the VLM prompt. This stops a strong rider hint on a raw track from making every segment a rider, including a pedestrian segment that a correct split separated from the rider segment.

ReClass's default `match_class=[1, 36]` leaves out motorcycles. The docs recommend `match_class=[1, 3, 36]` when producing hints.

**Vehicle duplicate → `DROP{reason: "duplicate", of: <track>}`** (target `vehicle`)

For pairs that exist together for at least 10 frames:

```
D = fraction of shared frames with IoB(smaller, larger) ≥ 0.7 and ‖Δv‖ < 0.3 h/s
S_dup = ramp(D; 0.5 → 0.9)
```

The smaller track (by median area) is the one proposed for dropping.

**Orphan pass → `DROP{reason: "orphan"}`** (after stage 3)

A track with no accepted link and fewer than `orphan.min_seconds` (default 0.5 s) of observed frames is scored as below.

**Pending links defer the orphan decision.** A track that is an endpoint of a `HUMAN_PENDING` `LINK` is **skipped**, not scored. The ledger header's summary lists it under `orphan_deferred`. When the reviewer decides the link, stage 3 re-runs because it owns a changed event (§4.2), and the orphan pass re-runs after it:
- a rejected link makes the fragment an orphan candidate again;
- an accepted link makes it part of a longer track.

So a fragment is never deleted while the decision about linking it is still open.

Score:

```
S_orphan = ramp(observed_seconds; 0.5 → 0.1)
```

The orphan pass has its own bands, defaulting to accept 0.7 and reject 0.3.

### 6.3 Stage 3: link (pairs of track end → track start)

**Candidates** are pairs (i, j) that pass the class gate (gate 3), where i ends at `t_e`, j starts at `t_s`, and `g = t_s − t_e`.

**Majority class.** A track's class for linking is the mode of `cls` over its observed rows. On a tie, the class of the latest observed row wins. Per-row `cls` values in the output are not rewritten.

**Hard gates.** Gates 3–6 are `link_tracklets`'s gates, computed by the shared `_legacy_gate_cost` helper (§2.3).
1. **Gap:**
   - `1 ≤ g ≤ link.max_gap` (default 1.0 s × fps), or
   - **static gate:** `g ≤ link.max_gap_static` (default 10 s × fps) when i's speed over its last 0.5 s is below 0.2 h/s. For static-gated pairs, gates 5 and 6 are replaced by `‖c_j(t_s) − c_i(t_e)‖ ≤ 0.5·h̃_i`. Otherwise the growth term in gate 5 would open a huge gate over a 10 s gap.
   - **occlusion-witness gate:** `link.max_gap < g ≤ link.max_gap_occluded` (default 8 s × fps), for pairs that are not static-gated. Gates 5 and 6 are replaced by gates 7–9. Straight-line prediction over several seconds fails for turning objects, so these gates check feasibility instead.
2. **Overlap:** a small negative gap, `−2 ≤ g ≤ 0`, is allowed only if the boxes have `IoU ≥ 0.5` on every overlapping frame. The overlapping rows from j are dropped on merge.
3. **Class:** i's and j's majority classes, taken after stage 2's reclassing, are equal or share a group in `link.class_groups`. The default is `[]` for `person` and `[[2, 7]]` (car, truck) for `vehicle`, because detectors often label pickups and SUVs as trucks. Bus (5) stays in its own group.
4. **Size:** `w_j / w_i` and `h_j / h_i` are each within `[1/size_ratio_max, size_ratio_max]` (default 2.0). For occlusion-witnessed pairs, the check compares i's last and j's first *unoccluded* boxes (§5.3), falling back to the raw end and start boxes. Boxes at the edge of an occlusion are often partial.
5. **Position:** `dist = ‖c_j(t_s) − (c_i(t_e) + v_i·g)‖ < dist_mult · √area_i · (1 + dist_growth · g)`, with defaults `dist_mult = 2.5` and `dist_growth = 0.03` per frame. `v_i` is a first-order fit on i's last `vel_frames` observed frames.
6. **Predicted-box IoU:** `IoU(i's end box shifted by v_i·g, j's start box) ≥ iou_min` (default 0.05).

Gates 7–9 apply only to occlusion-witnessed pairs.

7. **Witness:**
   - For each gap frame, a *hidden box* `B_t` is linearly interpolated (center and size) between i's end box and j's start box.
   - *Occluders* are the boxes, in that frame, of every other track in the file (after stages 1 and 2 have been applied) and of the context file, excluding i and j.
   - `witness = fraction of gap frames in which some occluder has IoB(B_t, occluder) ≥ link.witness_iob`, where the default `witness_iob` is 0.5, and IoB is measured relative to `B_t`. Gap frames with no occluder boxes count as uncovered.
   - Gate: `witness ≥ link.witness_min` (default 0.7).
8. **Heading:** the angle between i's exit velocity `v_i` and the chord `c_j(t_s) − c_i(t_e)` is at most `link.max_heading_change` (default 120°). This rules out linking to an object moving the other way. If i was stopped (`‖v_i‖ < 0.2` h/s), the check is skipped.
9. **Speed feasibility:** `v_need = ‖c_j(t_s) − c_i(t_e)‖ / (h̃_i · g/fps)` in h/s. Gate: `v_need ≤ link.speed_factor · v_ref` (default `speed_factor = 1.5`), where `v_ref = max(i's speed over its last 1 s, j's speed over its first 1 s, link.min_feasible_speed)` and `min_feasible_speed` defaults to 0.5 h/s.

**Costs** (each in [0, 1])

| Cost | Definition |
|---|---|
| `c_mot` | `ramp(legacy; 0 → link.legacy_cost_hi)` (default 3.0). `legacy` is `link_tracklets`'s own cost: `w_d·dist/√area_i + w_iou·(1 − IoU_pred) + w_s·(|log w_j/w_i| + |log h_j/h_i|)`, with `link.legacy_weights` defaults `{d: 1.0, iou: 1.0, s: 0.3}`. For static-gated pairs, `c_mot = ‖c_j − c_i‖ / (0.5·h̃_i)` |
| `c_app` | `(1 − cos(mean of i's last K clean embeddings, mean of j's first K clean embeddings)) / 2`, with `K = 5`. `0.5` when either side has no clean embeddings |
| `c_gap` | `g / max_gap`, using `max_gap_static` for static-gated pairs and `max_gap_occluded` for occlusion-witnessed pairs |

For occlusion-witnessed pairs, `c_mot = 0.5 · v_need / (speed_factor · v_ref) + 0.5 · heading / max_heading_change`, with the heading term 0 when the check was skipped. The pair is scored with `link.weights_occluded` (default `{mot: 0.25, app: 0.60, gap: 0.15}`), because after a long occlusion the motion evidence is weak and appearance carries the decision. `signals` records `witness`, the occluder track IDs, `v_need`, `v_ref`, and `heading`.

**Birth/death prior**
- `b = 1` if i ends and j starts inside the image: more than `0.5·h̃` from the image border. Otherwise `b = 0`.
- The image size comes from the video, or from `RefineConfig.frame_size` when there is no video. If neither is known, the prior is not applied (`b = 1`).
- No user-supplied zones are used. Location-based logic is out of scope (§1).

**Score**

```
S_link = (1 − (w_mot·c_mot + w_app·c_app + w_gap·c_gap)) · (0.8 + 0.2·b)
```

Defaults are `w_mot = 0.45, w_app = 0.40, w_gap = 0.15` (`weights_occluded` for occlusion-witnessed pairs). Size consistency is already part of `c_mot` through the legacy cost. In motion-only mode `w_app = 0` and the others are renormalized. For occlusion-witnessed pairs, `S_link` is then capped at `occluded_score_cap` (§4.3).

**Assignment**
- Build the bipartite graph of gated pairs whose `S_link ≥ reject_below`.
- Split it into connected components and run `scipy.optimize.linear_sum_assignment` on each component to maximize the total `S_link`.
- **Margin:** for each chosen pair, `margin = S_link − max(second-best S for i, second-best S for j)`, or `S_link` when there is no alternative.
- The **routed score** is `S_link` when `margin ≥ link.margin_min`, and `min(S_link, link.ambiguous_cap)` otherwise (§4.3). Bands apply to the routed score. `S_link` and the margin are stored in `signals`.
- Each chosen pair becomes one `LINK` event. It records its next-best alternatives, up to two, in `signals.alternatives`, and the review card shows them.
- **Re-assignment after rejection.** Assignment runs in **passes**. Pass 1 is the assignment above. Once a pass's events are decided, the next pass re-runs assignment on the affected components, with margins recomputed, over a reduced graph:
  - **Rejected edges** (`AUTO_REJECT`, `VLM_REJECT`, `HUMAN_REJECT`) are removed.
  - **Accepted pairs are frozen.** An accepted pair (`AUTO_ACCEPT`, `VLM_ACCEPT`, `HUMAN_ACCEPT`) is kept as it is, and both of its endpoints are removed from the graph. A later pass can never re-assign a verified link's endpoint to a different partner, so the ledger cannot hold two accepted links that conflict.
  - **Pending endpoints are reserved.** The endpoints of `HUMAN_PENDING` edges are removed too, so a reviewer never sees two competing links for one track end.

  Each newly chosen pair becomes a new `LINK` event, with `signals.pass` and with `signals.replaces` set to the rejected event. Passes repeat up to `link.max_passes` times (default 3). The pass number is not part of the `proposal_key`, so a pair has the same key in whichever pass it is proposed.

  **On replay.** When stage 3 re-runs (§4.2), it follows the same rules. Recorded decisions are applied by key, pass by pass: accepted pairs stay frozen, and endpoints freed by a newly rejected edge become available for the next pass. A person who rejects a previously *accepted* link frees its endpoints in the same way.
- After verification, union-find merges the accepted links into chains. A chain that would put two boxes of the same object on the same frame is rejected: its lowest-scoring link is set to `applied: false` with `skipped_reason: "overlap"`.

**Legacy mode (`link.mode: legacy`).** In this mode stage 3 behaves exactly like `link_tracklets`:
- gate 1 without the static and occlusion-witness gates, then gates 3–6. Gate 3 uses `link_tracklets`'s rule: the `cls` of i's last row equals the `cls` of j's last row, with no groups. That is what `link_tracklets`'s per-track descriptors compare;
- Hungarian assignment on the raw legacy cost over the full end × start matrix;
- every assigned pair recorded as `AUTO_ACCEPT` (`algo_score` = 1, legacy cost in `signals`).

It must produce the same track-ID mapping as `link_tracklets` on the same input (§1, criterion 7). It exists as a regression anchor and for users who want the old behavior with a ledger. The default is `mode: scored`.

Legacy mode also shows the weakness that scored mode fixes: `link_tracklets` merges every pair that passes the gates, however high its cost.

### 6.4 Stage 4: fill

`interpolate_tracks_rts(fill_gaps_only=True, max_gap=fill.max_gap, protected_gaps=…)` runs on the final tracks, which contain observed rows only (§2.5). `fill.max_gap` defaults to `link.max_gap`, so a link across a long static gap is joined but not filled. This keeps a waiting pedestrian's invented positions out of conflict and speed analysis, since those analyses already exclude rows with `interp == 1`.

**Gaps bridged by an occlusion-witnessed link are never filled, whatever `fill.max_gap` is.** Over a long occlusion, often during a turn, constant-velocity motion would draw a straight chord through the occluder and create false conflicts. **How protected gaps reach the function.** Stage 4 builds `protected_gaps` from the applied `LINK` events whose `params.gate == "occluded"`:
- Each event contributes its `params.gap` (`[t_e, t_s]`), under the final ID of the chain it belongs to.
- A chain with several such links contributes several gaps.
- Replay rebuilds the map from the same events, so it gives the same result.

**Changes to `interpolate_tracks_rts`.** Both are backward-compatible.
1. **Filled rows are not measurements.** Rows whose flag column (`interp`, or `r3` in the positional layout) equals 1 are excluded from the Kalman updates. They are estimated again like any other missing frame: filled, with flag 1, where the gap rule allows, and dropped otherwise.
   - Raw tracker output never contains a 1 (§2.5), so its output is unchanged, and the existing tests still pass.
   - Refinement removes filled rows before stage 4 anyway. This change fixes the public function for callers who use it on their own.
2. **New keyword `protected_gaps: Mapping[int, Iterable[tuple[int, int]]] | None = None`.**
   - For each track ID, it lists gaps as `(last observed frame before, first observed frame after)`.
   - The track is cut into independent filter-and-smoother segments at each protected gap, so neither filling nor `smooth_existing` crosses one.
   - `None` keeps the current behavior.

**Position records.**
- **`FILL`:** one record per filled gap, with `params.gap`, `n_rows`, and `signals` = `{gap_seconds, chord_h, max_fill_speed_h_s}`.
- **`SMOOTH`:** when `smooth_existing` is on, one record per smoothed track, with `n_rows` and `signals` = `{mean_shift_px, max_shift_px, max_shift_frame}`.
- **Protected gaps:** no `FILL`. The applied occluded `LINK` already records the gap.

`fill.smooth_existing` (default `false`) passes `smooth_existing=True`. Observed rows are then replaced by their RTS-smoothed boxes, which gives steadier speed estimates. Those rows keep `interp = 0`, and the ledger header records that smoothing was applied, since observed positions were changed.

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
| LINK i→j | Row A: i's last 3 clean crops. Row B: j's first 3 clean crops. Plus context frames at `t_e` and `t_s`. For occlusion-witnessed links, also a context frame at the middle of the gap, with the hidden box `B_t` drawn dashed and the occluder labeled |

| FILL (audit only, never sent to a VLM) | The last observed crop before the gap and the first after it, plus the context frame at mid-gap with the filled box drawn |
| SMOOTH (audit only) | The context frame at `max_shift_frame`, with the original box and the smoothed box drawn |

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
  - The VLM's answer can therefore change the final edit: `edit` gets the answered kind and params (for example `RECLASS{new_cls: 1}` for a proposed `DROP{static}`). The event's `kind`, `params`, and `proposal_key` keep the proposal (§4.1).
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
  - Extra: `dnt[refine-vlm] = ["openai>=1.40", "anthropic>=0.40"]`.
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
- for `LINK` events, the recorded next-best alternatives,
- for partial screen events, the supported and unsupported segments,
- accept and reject controls, plus a class picker for `RECLASS` and for screen events that the VLM redirected,
- a copy-to-clipboard `Labeler.draw_track_clips(...)` snippet covering the event's tracks and frame span ±2 s, for events that need motion to judge. `dnt.refine` does not import `Labeler`; it only prints the snippet.

An **Export decisions** button downloads `decisions.json` in the §4.2 format.

The page makes no network requests. Its choices are saved in `localStorage` while the person works, as a convenience only. The exported file is what counts.

Cards can be filtered by stage and sorted by score.

### 8.2 Audit

`dnt-refine audit`:
1. Samples `n` events whose decision is final: `AUTO_*`, `VLM_*`, or `HUMAN_*`. This includes `FILL` and `SMOOTH` records, so the precision of interpolation and smoothing is measured too (`--kinds` restricts the sample).
2. Stratifies the sample by (stage, decision), sampling each stratum in proportion to its size but with at least 3 per non-empty stratum, using a fixed `--seed`.
3. Builds evidence images for them, reusing `evidence.py`.
4. Writes `audit.html`, which has the same card layout with **correct** and **incorrect** controls, and exports `marks.json`.

`dnt-refine audit-score` reads the ledger and marks. For each stratum it prints and saves (`audit-score.json`) the count, the number marked correct, precision, and a 95% Wilson interval.

That precision is how success is measured (§1, criterion 6) and how bands are tuned. For example, if `link` `AUTO_ACCEPT` precision is high, `link.accept_above` can be lowered, and fewer events go to the VLM.

### 8.3 Run summary

The summary is written to the ledger header, printed at the end of the run, and available as `refiner.last_result.summary`. It contains:
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
fps: null                    # null → the fps= argument, else the video (§5.1)
reclass_map: {cyclist: 1, motorcycle: 3, scooter: 36}
frame_size: null             # [width, height]; null → from video
context:
  format: auto               # auto | tracks | dets
  vehicle_classes: [2, 5, 7]
  twowheeler_classes: [1, 3]
hints:
  reclass_class_map: {1: cyclist, 3: motorcycle, 36: scooter}
  reclass_ramp: [0.75, 0.9]
  subtype_min: 1.0
motion:
  height_window: 15
  process_var: 10.0          # shared Kalman model (§5.2), same defaults as interpolate_tracks_rts
  meas_var_pos: 25.0
  meas_var_size: 16.0
encoder:
  kind: dino                 # dino | reid | none
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
  mixed_score_cap: 0.75
  segment_at: 0.30
  # ramp ranges for R, J, C, T, H, inside, F, S, K, D as in §6.2
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
  class_change_gate: true
  size_gate: 1.5
  min_crop_px: 40            # appearance of boxes with a smaller longer side (px) is ignored
link:
  enabled: true
  mode: scored               # scored | legacy (§6.3)
  accept_above: 0.80
  reject_below: 0.40
  max_gap: 1.0               # seconds
  max_gap_static: 10.0       # seconds
  max_gap_occluded: 8.0      # seconds (§6.3 gates 7–9)
  witness_iob: 0.5
  witness_min: 0.7
  max_heading_change: 120    # degrees
  speed_factor: 1.5
  min_feasible_speed: 0.5    # h/s
  occluded_score_cap: 0.75
  margin_min: 0.10
  ambiguous_cap: 0.75
  max_passes: 3
  weights_occluded: {mot: 0.25, app: 0.60, gap: 0.15}
  class_groups: []           # vehicle default: [[2, 7]]
  size_ratio_max: 2.0        # legacy gates (link_tracklets defaults)
  dist_mult: 2.5
  dist_growth: 0.03          # per frame
  iou_min: 0.05
  vel_frames: 5
  legacy_weights: {d: 1.0, iou: 1.0, s: 0.3}
  legacy_cost_hi: 3.0
  weights: {mot: 0.45, app: 0.40, gap: 0.15}
  min_crop_px: 0             # as switch.min_crop_px, for stage 3; 0: every clean crop
orphan:
  enabled: true
  min_seconds: 0.5
  accept_above: 0.70
  reject_below: 0.30
fill:
  enabled: true
  max_gap: null              # null → link.max_gap
  smooth_existing: false
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
- `static_score_cap < screen.accept_above`, so static tracks can never auto-drop (§6.2).
- `link.weights` and `link.weights_occluded` each sum to 1.
- `link.occluded_score_cap < link.accept_above`, so occluded links can never auto-accept.
- `link.max_gap < link.max_gap_occluded`.
- No class appears in more than one `link.class_groups` group.
- `screen.mixed_score_cap < screen.accept_above`, `link.ambiguous_cap < link.accept_above`, and `screen.segment_at < switch.reject_below`.
- `reid` with the `vehicle` target has `weights` set.
- `switch.min_crop_px` and `link.min_crop_px` are integers `>= 0`.
- A non-`none` VLM backend has `model` set (except `anthropic`, which has a default).

**Deferred until `refine` or `apply` starts, and only for components the run will use** (§5.5): the encoder's extra (needed only with a video and `encoder.kind` other than `none`) and the VLM backend's extra (needed only with a non-`none` backend and a video).

Durations in config are in seconds and are converted to frames with `fps`.

## 10. Error handling and degraded modes

| Situation | Behavior |
|---|---|
| No `video` | Motion-only mode: no embeddings, and appearance weights set to 0 (§6.1, §6.3). The VLM backend is forced to `none` with a warning. Everything in the uncertain band becomes `HUMAN_PENDING`, and no review images are made (cards show signals only). No encoder or VLM extra is needed. |
| `encoder.kind: none` with a video | Motion-only scoring, as above, but the VLM and the evidence images are still available. |
| No `context` | In-vehicle and two-wheeler-overlap cues are skipped and recorded as `null` in `signals`. The vehicle-duplicate cue still runs, because it uses the vehicle file itself. |
| Context file with neither 8 nor 10 columns, when `context.format: auto` | `ValueError` naming the file and the column count. |
| Hints file missing the `track, cls, avg_score` header | `ValueError`. Rows with unknown track IDs are ignored, with a warning. |
| Max track frame exceeds the video's frame count | `ValueError` naming both numbers, since the track file does not belong to this video. |
| Malformed CSV (fewer than 6 columns, non-numeric) | `ValueError` naming the file and first bad line. |
| Empty track file | Writes an empty output, a ledger with header only, and no review. |
| Invalid config | Fails at config load (§9). |
| Encoder or VLM extra missing for a component the run will use | Fails at the start of `refine` or `apply`, before any processing (§5.5). |
| Input contains filled rows (`interp == 1`) | They are removed on input and counted in the ledger header (§2.5). |
| VLM errors | §7.4: `HUMAN_PENDING`, never an abort. |
| Decisions file references an unknown event ID | `ValueError` listing the unknown IDs. Nothing is applied. |
| No video and no `fps` (argument or config) | `ValueError` before any processing, asking for `fps=` (§5.1). |
| `apply`: a recorded input's hash or fingerprint differs, or a recorded `tracks`/`context`/`hints` file is missing | `ValueError` before any processing, naming the input (§4.2). |
| `apply`: an input the original run did not have | `ValueError`: re-run `dnt-refine run` instead (§4.2). |
| `apply`: the recorded video is missing, and the feature cache is valid | Warning. New events that need evidence images become `HUMAN_PENDING` without images (§4.2). |
| `apply`: stage 1 or 3 will re-run with appearance, and the feature cache is missing or invalid while the verified video is available | Cache miss: recompute from the video. INFO for missing, WARNING with both hashes for invalid. The new cache is written as `OUT.features.npz` (§4.2). |
| `apply`: stage 1 or 3 will re-run with appearance, and neither the video nor a valid feature cache is available | `ValueError` before any processing. The message says whether the cache is missing or mismatched, names both paths, and says how to supply either (§4.2). |

## 11. Testing

All tests below run in the default suite (CPU, no network), except where a marker is named.

### 11.1 Stage detectors, with synthetic tracks built in numpy (`tests/refine/`)

- `test_screen.py`:
  - a static box with low confidence repeated at one spot → static score at the cap, routed to the VLM band, never `AUTO_ACCEPT`.
  - a person box moving inside a car box → `DROP{in_vehicle}` `AUTO_ACCEPT`.
  - a person boarding a bus (inside for 20% of frames) → no event.
  - a "person" at 3 h/s on a smooth path → `RECLASS`.
  - a walker at 0.8 h/s → no event.
  - the same in-vehicle case with a **detection** context → `DROP{in_vehicle}`, using the persistence test.
  - a fast "person" with a ReClass hint `(cls 3, avg_score 0.95)` → `RECLASS` to motorcycle with no subtype VLM call (the fake backend raises if called).
  - a hint that disagrees with an in-band VLM subtype answer → `HUMAN_PENDING`.
  - **mixed pedestrian/rider track with a strong raw hint** `(cls 3, avg_score 0.95)`, split at the change:
    - The pedestrian segment gets no `RECLASS`: its motion cues are low, and the hint is unlocalized.
    - The rider segment gets a `RECLASS` from motion cues, with its subtype chosen by the VLM rather than the hint. `signals.hint_unlocalized` is set.
    - An unsplit fast track with the same hint still gets its subtype from the hint (regression).
  - two vehicles moving together with IoB 0.9 → `DROP{duplicate}` on the smaller.
  - an orphan of 0.2 s after linking → `DROP{orphan}`.
  - a 0.1 s fragment that is an endpoint of a `HUMAN_PENDING` `LINK` → no orphan event, listed in `orphan_deferred`, and present in the output (end-to-end in `test_replay.py`).
  - **mixed track:** a pedestrian segment followed by an in-vehicle segment.
    - With the `SPLIT` applied → only the second track is dropped.
    - With the `SPLIT` pending → a partial `DROP` with `spans` covering only the second segment, capped and routed, never `AUTO_ACCEPT`. Accepting it keeps the pedestrian rows.
    - With a stage 1 candidate at S = 0.35, below `switch.reject_below` → segmentation still applies, and there is no whole-track auto-drop.
- `test_switch.py`, with embeddings injected through a stub encoder:
  - swapped embedding sequences at t with IoU contact → `SPLIT` at t (±1 sample), and swap confirmation raises the score.
  - an appearance drift with no gate → no event.
  - a position jump after a 5-frame gap → motion-only candidate, capped below accept.
  - a side shorter than 0.5 s → no event.
  - **takeover by an untracked object** (reference case replica): a car box that, at t, widens 1.8× with little height change, whose row classes flicker 2 → 7 → 5, and whose embeddings change, with no other track nearby → the gate opens through class change and size jump, and `SPLIT` lands at t. With `class_change_gate: false`, the size jump alone still opens the gate.
  - a class flicker alone, with unchanged embeddings and smooth motion → the gate opens but no event is written.
- `test_link.py`:
  - **parity:** `link.mode: legacy` gives the same track-ID mapping as `link_tracklets` on the existing `test_post_process.py` fixtures and on a randomized set of 200 synthetic tracklets (fixed seed).
  - a pair that `link_tracklets` merges despite a high cost → `scored` mode sends it to `AUTO_REJECT` or the VLM band, not to `AUTO_ACCEPT`.
  - collinear fragments with a 10-frame gap and similar embeddings → `LINK` `AUTO_ACCEPT`.
  - two equally good candidates → margin below `margin_min`, capped at `ambiguous_cap`, routed to the VLM (neither auto-accepted nor auto-rejected), with `signals.alternatives` filled in.
  - **re-assignment:** three tracklets where i→j scores highest but the fake VLM rejects it, and i→k is correct → pass 2 proposes i→k with `signals.replaces` set. It is accepted, and i and k share an ID.
  - a pending edge keeps its endpoints reserved: no competing event in later passes.
  - **frozen accepted pairs:** a component where a→x is accepted in pass 1 and b→y is rejected. Without x, the unconstrained optimum for pass 2 would be b→x plus a→z. Instead a→x stays applied, x is not offered to b, and pass 2 proposes b's best partner other than x (or nothing). No two accepted links share an endpoint. The same holds on replay when b→y is rejected through `apply --decisions`.
  - a stationary pedestrian with a 6 s gap at the same spot → linked through the static gate. With `max_gap_static` = 5 s → not linked.
  - a start near the border → prior lowers the score.
  - a chain producing overlap → lowest link skipped with `skipped_reason: "overlap"`.
  - **occluded turn** (reference case replica): i ends heading north, a large box covers the linearly interpolated path for 63 frames at 10 fps, and j starts heading west with matching embeddings → an occlusion-witnessed `LINK` routed to the VLM band (never `AUTO_ACCEPT`).
  - the same pair with no occluder → not linked. With the occluder covering only 50% of the gap → not linked. With j moving opposite to i's heading → not linked. With `v_need` above `speed_factor · v_ref` → not linked.
  - car ↔ truck majority classes → linked with the vehicle default `class_groups`, not linked with `class_groups: []`.

### 11.2 Events, verify, apply, replay

- `test_events.py`: ledger round-trip, header contents, stable event IDs.
- `test_verify.py` with `FakeVLMBackend`:
  - band routing, including the static cap and the margin rule.
  - redirected answers (a DROP answered as a rider → `edit` = RECLASS; `kind`, `params`, and `proposal_key` unchanged).
  - votes (majority, tie → pending), `min_conf`.
  - budget exhaustion ordered by distance to the band midpoint.
  - invalid JSON then valid on retry.
  - repeated invalid output → pending.
  - an exception → pending.
  - a cache hit makes no backend call.
- `test_apply.py`: each event kind on a small table, including partial `DROP`/`RECLASS` with `spans`; `edit` applied instead of the proposal; renumbering, `id_map`, and lineage; proposal keys stay the same when track IDs are renumbered.
- `test_replay.py`:
  - A decisions file that changes nothing → byte-identical output, zero re-proposals, and zero backend calls (the fake backend raises if called).
  - **Accepting a pending `SPLIT` →** `apply` re-proposes stages 2–4. The tail gets a `LINK` event (round 1), which is routed and applied. Every other event keeps its decision through key matching, so the fake backend sees only the new event.
  - **Rejecting a pending `LINK` via `apply --decisions`** (the stage 3 input table is unchanged) → stage 3 re-runs because it owns a changed event, and re-assignment proposes the alternative as a new event.
  - **Pending event round trip:**
    1. `refine` leaves a `LINK` pending.
    2. The ledger contains it with `decision: HUMAN_PENDING` and an `id`, and it deserializes to an equal `Event`.
    3. A `decisions.json` exported for that `id` is passed to `apply`.
    4. The new ledger keeps the same `id`, has `decision: HUMAN_ACCEPT`, and its `decision_history` has two entries. Its header names the parent ledger and `round: 1`. The parent ledger file is unchanged, byte for byte.
  - **Orphan with a pending link, end to end:** a 0.1 s fragment whose link is pending stays in the output.
    - When `apply` rejects the link → the orphan pass proposes `DROP{orphan}`, which is auto-accepted.
    - When `apply` accepts the link → the fragment is linked, and no orphan event is proposed.
  - **Missing video and cache:**
    - With the recorded video and the feature cache both removed, `apply` accepting a pending `SPLIT` (so stages 2–4 re-run, and stage 3 needs appearance) → `ValueError` before processing, naming both paths.
    - With only the video removed → `apply` succeeds, using the cache.
    - With the video removed and the cache modified → `ValueError` whose message says "does not match" and gives both hashes.
    - With the video removed and the cache removed → `ValueError` whose message says "not found".
    - With the cache removed and the video present (CPU) → embeddings are recomputed (logged at INFO). The output equals the run that used the cache, and the new header has `features_recomputed: true`.
    - **With the cache modified and the verified video present (CPU)** → no error: a WARNING naming both hashes, then recomputation. The output equals the run that used the original cache. The modified file is unchanged on disk, and the new `OUT.features.npz` has the header's new `sha256`.
    - With the cache modified but neither stage 1 nor stage 3 re-running (the only change overrides an orphan-pass decision, so only the orphan pass and stage 4 re-run) → the cache is not read, and there is no warning or error.
  - **Redirected screen event survives re-proposal:**
    1. `refine` with the fake VLM redirecting a proposed `DROP{static}` to a cyclist `RECLASS`.
    2. `apply` with a decision change on an unrelated stage 1 split, which forces stage 2 to re-run.
    3. The redirected event is matched by `proposal_key` and keeps its `RECLASS` edit. The fake backend is not called for it.
  - **Replay with context and hints:**
    1. `refine` with a context file and a hints file, where a `SPLIT` ends up pending.
    2. `apply` accepting it.
    3. The output tracks equal those of a fresh `refine` whose fake VLM accepts that split directly, and the set of applied `proposal_key`s is the same.
  - `apply` with an overridden context path whose content differs from the recorded hash → `ValueError` naming `context`.
  - `apply` with the recorded hints file missing → `ValueError`.
  - `apply` supplying a context file the original run did not have → `ValueError`.
  - `apply` with the video missing → a warning, and no error when no new event needs evidence.
  - The same `apply` run twice → byte-identical output.
  - `--no-fill` reproduces the table as it was before filling, including the original observed positions of smoothed rows.
  - Unknown event IDs → `ValueError`.

### 11.3 Evidence and reports

- `test_evidence.py`: a cv2-generated synthetic video with moving rectangles (extends `tests/support/synthetic.py`'s `make_synthetic_video`). Checks the grid size for each event kind, crop padding and minimum height, and the effect of `send_context_frames: false`.
- `test_review.py`: the HTML contains one card per pending event, references existing image files, and contains no external URLs. Audit sampling is deterministic for a seed. `audit-score` Wilson intervals match known values.

### 11.4 Integration and compatibility

- `test_refiner.py`: end-to-end `refine` on the synthetic video with the stub encoder and the fake VLM; motion-only mode; the empty file; the CLI `run` / `apply` via `subprocess`.
- `test_reference_case.py`: the synthetic replica of #12 → #81 end to end. The fake VLM answers `different` for the split and `same_individual` for the link. Expected result:
  - the car part of #12 and #81 share one ID;
  - the truck part has its own ID;
  - no rows are filled in the 63-frame gap, even with `fill.max_gap` raised to 100 frames.
- `@pytest.mark.realdata` (new marker, excluded by default): the same checks on the real Miami track file and video, found through `DNT_REFINE_CASE_DIR`. That directory holds `tracks.txt`, `video.mp4`, and an `expected.yaml` listing the expected `SPLIT` and `LINK` events, with frame tolerance ±3.
- `test_refine_independence.py`: the §2.2 import rule.
- The existing `tests/test_post_process.py` passes unchanged through the shim. The existing `tests/test_filter.py` passes with the retargeted `Filter.interpolate_tracks_rts` wrapper.
- `test_interpolate.py`:
  - An input with `interp == 1` rows → those rows are not used as measurements, and are either filled again with flag 1 or dropped. A raw tracker file → output identical to the pre-change function.
  - `protected_gaps`: a 63-frame gap stays empty with `max_gap=100`. Two protected gaps in one chain both stay empty. With `smooth_existing=True`, smoothing does not cross a protected gap (the observed rows on each side match smoothing each segment alone).
- `test_refiner.py` (additions): an input that already has `interp == 1` rows → they are removed, the count is in the header, and the output fills come only from stage 4. One `FILL` record per filled gap, and one `SMOOTH` record per smoothed track.
- `test_features_cache.py`: the cache is reused when nothing changes, and invalidated when any of these change: the video (same track file), the context file, the encoder weights, a crop preprocessing setting, or `FEATURES_VERSION`.
- `test_fingerprint.py`: two 65 MiB files that are identical except for one byte after the 64 MiB mark → different fingerprints, with the same size. The cache built with one is not reused with the other.
- `test_minimal_install.py`: with `transformers`, `torchreid`, `openai`, and `anthropic` imports made to fail (via monkeypatching `sys.modules`):
  - the default config loads;
  - `refine` without a video and with `fps=10`, on a fixture with an injected fragment, an injected takeover, and a static box → it completes, the ledger holds proposals from stages 1, 2, and 3, and every uncertain one is `HUMAN_PENDING`;
  - `refine` without a video and without `fps` → `ValueError` before processing, asking for `fps=`;
  - `refine` with a video and `encoder.kind: dino` raises `ImportError` before any processing, naming the extra;
  - `refine` with a video and `encoder.kind: none` succeeds.
- `test_primitives.py`: `cv_kalman` gives the same smoothed boxes as the pre-refactor `interpolate_tracks_rts` on the fixtures (the refactor is behavior-preserving), and NIS is χ²₄-distributed on simulated constant-velocity tracks (mean ≈ 4 within tolerance).
- `test_config.py`: YAML round-trip, unknown keys, every validation rule in §9.
- `@pytest.mark.model`:
  - `test_encoders_real.py` (DINOv2 and torchreid OSNet on sample crops: output shape and norm).
  - `test_vlm_real.py` (skipped unless `DNT_VLM_BASE_URL` / `ANTHROPIC_API_KEY` is set; one screen question on a fixture image returns a valid option).

## 12. Packaging and documentation

- **`pyproject.toml`**
  - extras: `refine-dino`, `refine-reid`, `refine-vlm`, and `refine` (the union of those three).
  - `[project.scripts] dnt-refine`.
  - a `realdata` pytest marker, added to the default `-m` exclusions in `addopts`.
  - No new required dependencies.
- **Docs**
  - `docs/api/refine.md`, pointing mkdocstrings at `dnt.refine`, `dnt.refine.config`, and `dnt.refine.vlm`, plus a `mkdocs.yml` `nav` entry.
  - A "Refining tracks" section in `docs/quickstart.md`, covering:
    - the pedestrian and vehicle YAML examples;
    - the review/audit loop;
    - producing ReClass hints with `match_class=[1, 3, 36]`;
    - the recommendation to run `Filter.deduplicate_boxes` on detections before tracking;
    - a note that location-based filtering is the next procedure and runs on the refined output.
- **`docs/changelog.md`:** a new-feature entry, and a note that `dnt.track.post_process` is now a shim and that `Filter.interpolate_tracks_rts` points at `dnt.refine`.
- **Lint:** all new code is ruff-clean under the repo rules and numpy-style docstrings. Nothing is added to the legacy per-file baseline.
- **Version:** this adds public API, so it ships in a minor release and not in a 0.3.x patch. The version number is picked when the release is cut, following the "bump both files" rule in CLAUDE.md.

## 13. Open questions deferred to implementation

None block the plan. Each has a default chosen above, and the default holds unless evidence from the user's clips argues otherwise.

1. Whether DINOv2 or OSNet is the better default for pedestrians. The default is `dino` for both targets. Revisit after the first audit.
2. The pinned default model ID for `anthropic` (§7.3).
