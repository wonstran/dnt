# dnt.track: Classifying Vehicle Tracks as Car, Bus or Truck (`VehReClass`)

- **Status:** implemented (rev. 3, 2026-10-16: car means any light vehicle, so re-detection may promote to bus or truck only for a heavy-flagged track, see §5.2; rev. 2: addresses [review 2026-10-15 22:04](../../review_2026-10-15-22-04-16.md) / [response](../../response_2026-10-15_22-40-00.md))
- **Date:** 2026-10-15
- **Related:** `dnt.track.re_class.ReClass` (the rider re-classifier for person tracks). The review of that
  module on 2026-10-15 found the issues cited below as "review issue N".

## 1. Problem

Vehicle track files come out of tracking with no usable class. The `cls` column is `-1`
(`Tracker(output_score_cls=False)`), or the detector used a single generic vehicle class. The downstream
stages need each vehicle track labeled **car (2), bus (5) or truck (7)**, the COCO classes.

`ReClass` cannot do this. It asks whether a small object (a bicycle or skateboard) sits *inside* a person box.
Here the question is which class the *same box* belongs to. Its scoring also lets one matching frame decide a
whole track (review issue 1).

## 2. Goal and scope

**Goal.** Give every vehicle track one class for the whole track, using the cheapest evidence that settles it:

1. the track's size (a big track is a heavy vehicle, so bus or truck);
2. re-detection with a strong detector on spaced, large frames;
3. a vision-language model (VLM) on crops, for tracks that re-detection cannot settle;
4. a fixed fallback.

**In scope**

- A new class `VehReClass` in `src/dnt/track/re_class.py`, exported from `dnt.track`.
- Private helpers in the same file for spaced frame sampling and one-pass batched detection, shared with
  `ReClass`.
- `ReClass` moves onto those helpers: one detection pass instead of one per track, and no nested progress bars
  (review issues 5 and 6). Its scoring and output stay as they are.
- Removing the stale `try/except ModuleNotFoundError` around the re_class import in `track/__init__.py`
  (review issue 7). It would hide a broken `VehReClass` import as `None`.

**Out of scope**

- Fixing `ReClass`'s rider scoring (review issues 1 to 4). That is a separate change.
- Separating large trucks from pickups. COCO class 7 covers both, and this design uses COCO classes as they are.
- Per-zone or perspective-aware size thresholds. The threshold is one pixel value per camera or video.
- Feeding vehicle results to `refine` as hints. `refine`'s hints stay rider cues.
- Writing a per-track summary file.

## 3. API

```python
from dnt.track import VehReClass

vr = VehReClass(
    hv_size=None,              # px; layer 1 off when None
    hv_min_count=3,            # layer 1: real rows that must reach hv_size
    num_frames=10,             # layer 2: frames sampled per track
    min_gap=5,                 # layer 2 and 3: minimum frame distance between samples
    iou_threshold=0.5,         # layer 2: a re-detection must overlap the track box this much
    min_support=0.3,           # layer 2: share of sampled frames the winning class needs
    vlm=None,                  # layer 3 off when None; a dnt.refine VLMConfig otherwise
    vlm_crops=3,               # layer 3: crops tiled into the one image per track
    model=DetectorModel.RTDETRx,
    weights=None,
    device="auto",
)
tracks = vr.re_classify(
    tracks,                    # track file path (headerless, Tracker.TRACK_FIELDS) or a DataFrame
    input_video,
    track_ids=None,            # None: every track; else only these tracks are classified
    out_file=None,             # headerless CSV, same layout as the input
    verbose=True,
)
vr.decisions                   # DataFrame of the last run; see §6
```

Class constants: `CAR = 2`, `BUS = 5`, `TRUCK = 7`. They are not constructor parameters.

The layer 3 confidence threshold is the config's own `vlm.min_conf` (default 0.7), as in `refine`. There is no
separate `VehReClass` parameter for it.

**Constructor validation.** Every rule below raises `ValueError`, and all broken rules are listed in one
message:

- `hv_size` is None or a finite number > 0.
- `hv_min_count`, `num_frames` and `vlm_crops` are integers >= 1, and `min_gap` is an integer >= 0. `bool` is
  not accepted as an integer, and neither is a float such as `10.0`.
- `iou_threshold` and `min_support` are finite numbers in (0, 1].

**VLM preparation.** When `vlm` is given, the constructor works on a deep copy, in this order, before it builds
the detector:

1. **Validate and resolve.** The VLM rules now inside `RefineConfig.validate` (blank strings to None,
   endpoint keys, `use` naming a known endpoint, `resolve()`, backend name, model required for the backend,
   votes, `min_conf`, `max_calls`, concurrency, timeouts, `extra_body`, `max_tokens`, cache dir, key variable
   and key file, base URL) move into one function in `refine/config.py`,
   `vlm_problems(vlm) -> tuple[list[str], VLMConfig]`. It returns the problems and the resolved copy.
   `RefineConfig.validate` calls it, so `refine`'s behavior and messages stay the same. `VehReClass` calls it
   too and raises `ValueError` with the problems.
2. **Reject a config that does nothing.** If the resolved backend is `none`, the constructor raises
   `ValueError`. `vlm=None` is the one way to turn layer 3 off, so a config that resolves to no backend is
   treated as a mistake rather than silently skipped.
3. **Preflight.** `check_vlm_dependencies` and `resolve_api_key` run on the **resolved** copy, so a missing
   package or key fails before any detection runs.

The resolved copy is kept, and the caller's object is never changed. The detector is built last, as in
`ReClass`.

## 4. Data flow

0. **Check the video.** A missing `input_video` raises `FileNotFoundError` first, even when the track input
   is empty. A wrong path is a caller error, and in a batch it should not pass silently just because one file
   happens to have no tracks.
1. **Read.** A path is read with `shared.files.read_track` and given the `Tracker.TRACK_FIELDS` names. A
   zero-byte file makes `read_csv` raise `pandas.errors.EmptyDataError`. `VehReClass` catches that one
   exception at its own boundary and uses an empty table with the `TRACK_FIELDS` columns. `read_track` itself
   is not changed. A DataFrame must hold at least `frame, track, x, y, w, h, cls`. Extra columns such as
   `interp` are kept, and the columns of the returned or written table are exactly those of the input.
2. **Real rows.** A row is *filled* when its interpolation flag is 1, and *real* otherwise. The flag is column 8.
   It is named `r3` after a headerless file is read with `TRACK_FIELDS`, and `interp` in the named table that
   `interpolate_tracks_rts` returns. A row is filled when **either** `r3 == 1` or `interp == 1`, so a table that
   holds both names is handled, and a saved file gives the same evidence as the named table it came from. Values
   are compared after `pd.to_numeric(errors="coerce")`, so a missing or non-numeric flag counts as real. This is
   the rule `refine.io.to_work` already applies; it moves into one helper in `refine/io.py`,
   `filled_mask(df) -> numpy.ndarray`, which both call. Sizes, samples and crops use real rows only. A track
   with no real rows goes straight to the fallback.
3. **Layer 1, size** (§5.1) for every selected track.
4. **Layer 2, re-detection** (§5.2). The sampled frames of all selected tracks are merged, sorted and
   detected in **one pass** over the video.
5. **Layer 3, VLM** (§5.3) for the tracks layer 2 left undecided, as one `VLMRunner.ask_many` batch.
6. **Fallback** (§5.4) for the rest.
7. **Write.** `cls` is set on **every row** of each classified track, interpolated rows included. No other
   column, row or row order changes. When `out_file` is given, the table is written with
   `shared.files.write_track` (headerless).

## 5. The cascade

Each track goes through the layers in order and stops at the first one that decides.

### 5.1 Layer 1: size

- `size` = the `hv_min_count`-th largest value of `max(w, h)` over the track's real rows (the 3rd largest by
  default). It is NaN when the track has fewer real rows than `hv_min_count`.
- `size >= hv_size` flags the track **heavy**: at least `hv_min_count` real rows reached the threshold. From then
  on only bus and truck are allowed classes.
- **Robustness.** Up to `hv_min_count - 1` glitched or merged boxes cannot flag a track, whatever their size and
  however short the track. A percentile would not give this: with nine boxes of 100 px and one of 1,000 px,
  NumPy's default 95th percentile is 595 px. The guarantee matters because the flag is hard: it removes car
  from both later layers and turns the fallback into truck.
- **Short tracks.** A track with fewer than `hv_min_count` real rows is never flagged. It still allows car, bus
  and truck, so layers 2 and 3 decide it. Missing a flag only loses the narrowing, while a false flag forces a
  wrong class, so erring toward "not flagged" is the safer side.
- An unflagged track allows car, bus and truck, because a truck far from the camera can look small.
- Layer 1 never decides a class on its own. With `hv_size=None` no track is flagged.

### 5.2 Layer 2: re-detection

**Sampling.** The track's real rows are ranked by area `w * h`, largest first. Walking that order, a row is
taken when its frame is at least `min_gap` frames from every frame already taken, until `num_frames` are taken
or the rows run out. Ties in area go to the earlier frame, so the result is deterministic. With `min_gap=0` this
is the plain top-N by area.

**Detection.** The union of all tracks' sampled frames is read in increasing order with
`refine.crops.FrameReader`, and each frame is detected once with the constructor's detector.

**Reading frames tolerantly.** `FrameReader.frames()` raises `ValueError` on a frame it cannot read, and the
generator is then finished, so one `for` loop over the whole union would lose every later frame. The shared
helper therefore asks for **one frame per call** on the same open reader:

```python
for f in sorted(union):
    try:
        (_, img), = reader.frames([f])
    except ValueError:
        unread.add(f)          # logged at debug level
        continue
    ...
```

`FrameReader` keeps its position (`_next`) between calls, so this still decodes forward over short gaps and
seeks over long ones. After a failed read it sets `_next` to None, so the next call seeks. `FrameReader` itself
does not change, and its existing callers keep their strict behavior. A frame in `unread` is removed from the
sample of **each track that sampled it**, and only from those tracks, so it leaves their *n*. Other tracks are
unaffected. Layer 3's crop reads use the same helper and policy: an unread crop frame is skipped, and the tile
is built from the crops that were read.

Detection needs an image-level entry point on `Detector` (the body of `detect_frames`'s per-frame predict and
box conversion, factored out), so `detect_frames` and both re-classifiers share it.

**Voting.** With *n* the number of sampled frames that were read:

1. In each frame, the candidate detections are those whose class is allowed for the track (§5.1) and whose
   IoU with the track box is at least `iou_threshold`.
2. If there are candidates, the one with the **highest detector confidence** casts that frame's single vote,
   weighted by its confidence. Confidence, not IoU, picks it, because the detector often puts both a bus and a
   truck box on one vehicle and its confidence says which it believes. A frame with no candidate casts no vote
   but still counts in *n*.
3. For each allowed class: `score = sum of its vote weights / n` and `support = its vote count / n`.
4. The class with the highest `score` wins (ties go to the class with more votes, then the lower class ID). It
   **decides** the track if its `support >= min_support` **and** it is car or the track is heavy-flagged.
   Otherwise, or when *n* is 0, the track goes to layer 3.

**Why a bus or truck vote needs the size flag (rev. 3).** In this project car means any light vehicle,
including pickups and passenger or cargo vans (Ford Transit, Mercedes Sprinter), and truck means a heavy goods
vehicle. COCO's `truck` covers all of these, and a large van close to the camera is routinely detected as truck
with high confidence (found on the afternoon Collins Ave clip, tracks 20 and 37: truck in 8 of 10 frames,
support 0.8). So a bus or truck winner on a track that size has not flagged heavy does not decide: the track
goes to the VLM, which sees all four options, or falls back to car. The winner is still recorded in
`decisions.redetect_cls`.

Dividing by *n* rather than by the number of matched frames keeps one stray detection from deciding a track
(the flaw of review issue 1).

Example: *n* = 10. Bus wins 4 frames at confidence 0.8, truck wins 2 at 0.6, 4 frames have no match. Bus has
score 0.32 and support 0.4, so a heavy-flagged track is **bus** (an unflagged one goes to layer 3). Had bus
won 2 frames, its support would be 0.2 and the
track would go on to layer 3.

### 5.3 Layer 3: VLM

Runs only when `vlm` is set and the track is still undecided.

- **Image.** Pick `vlm_crops` real rows with the §5.2 sampling rule (largest first, `min_gap` apart). Crop
  each with `refine.crops.crop_box` and tile the crops side by side into one JPEG, each scaled to a common
  height. Frames are read in the same sorted way as layer 2. A track with no usable crop goes to the fallback.
- **Question.** One `vlm.runner.Question` per track: the crops show the same vehicle, and the model picks one
  option. The options are `car, bus, truck, unclear`, or `bus, truck, unclear` for a heavy-flagged track. The
  prompt text lives next to the other prompts in `refine/vlm/prompts.py`. Its glossary (rev. 3) defines car
  as a light vehicle (passenger car, SUV, minivan, pickup, passenger or cargo van), bus as a bus or coach that
  carries many passengers (not a passenger van), and truck as a heavy goods vehicle (box truck, dump truck,
  tanker, semi-trailer; not a pickup or van).
- **Asking.** All questions go to one `VLMRunner.ask_many` call. That gives the configured backend
  (`anthropic` or `openai_compat`), its retries on transient errors, voting (`vlm.votes`) and the answer cache.
- **Runner lifecycle.** Each `re_classify` call builds its own backend and `VLMRunner` from the resolved config
  (§3), and closes the runner in a `finally` block, as `TrackRefiner` does. The runner owns a loop thread and
  the backend's client, and its call budget and fatal-error state persist across `ask_many` calls. A fresh runner
  per call means `vlm.max_calls` is a budget **per `re_classify` call**, and a fatal error in one call does not
  carry into the next. Nothing is built when no track reaches layer 3.
- **Decision.** An answer other than `unclear` with `confidence >= vlm.min_conf` decides the track. An
  `unclear` answer, a lower confidence, or a verdict with an `error` (including `budget`) sends the track to the
  fallback. An error is logged as a warning naming the track.

**Cost.** Every track layer 2 leaves undecided costs one VLM question per vote. That includes ordinary cars
seen at poor angles. With `vlm=None` the cascade stops after layer 2 and costs nothing.

### 5.4 Fallback

A heavy-flagged track becomes **truck** (7), the likelier heavy class on most roads. Any other track becomes
**car** (2).

## 6. `decisions`

After `re_classify`, `vr.decisions` is a DataFrame with one row per classified track:

| column | meaning |
|---|---|
| `track` | track ID |
| `cls` | the class written |
| `layer` | `redetect`, `vlm` or `fallback` (layer 1 only narrows, so it is never the deciding layer) |
| `heavy` | whether layer 1 flagged the track |
| `size` | the §5.1 size in px (NaN with fewer than `hv_min_count` real rows) |
| `n` | sampled frames read in layer 2 |
| `support` | the layer 2 winner's support (NaN when *n* is 0) |
| `redetect_cls` | the layer 2 winner, also when it did not decide (nullable integer; `<NA>` with no votes) |
| `vlm_answer`, `vlm_conf` | the layer 3 answer and confidence (None and NaN when layer 3 did not run) |

It is not written to disk. It exists to tune `hv_size`, `hv_min_count`, `min_support` and `vlm.min_conf`.

## 7. Errors and edge cases

| Case | Behavior |
|---|---|
| Missing video | `FileNotFoundError`, checked first, even for empty input (§4 step 0). |
| Empty track table, or a zero-byte track file | Returns an empty table (the input's columns, or `TRACK_FIELDS` for a file). An `out_file` is still written, as a zero-byte file. `decisions` is empty. No error. |
| Video that cannot be opened | `ValueError` from `FrameReader`. |
| `track_ids` with IDs not in the table | Ignored, with one warning giving the count. |
| Undecodable sampled frames | Dropped from *n* of the tracks that sampled them, and reading goes on (§5.2). |
| Track with no real rows | Fallback, with `heavy` False. |
| Invalid VLM settings, unknown `vlm.use` endpoint, or a config resolving to backend `none` | `ValueError` in the constructor (§3). |
| VLM dependencies or key missing | Error in the constructor, checked on the resolved config (§3). |
| VLM transient failure | Retried by `VLMRunner`. A track still without an answer falls back, with a warning. |
| VLM call budget used up | The remaining layer 3 tracks fall back, each with a warning. The budget is per `re_classify` call. |
| Input `cls` already set | Overwritten for classified tracks. Rows of other tracks are untouched. |

## 8. Changes to existing code

- `src/dnt/track/re_class.py`: add `VehReClass` and the private sampling and batched-detection helpers.
  `ReClass.re_classify` uses the batched-detection helper only. **Its sample selection is not shared**: it keeps
  its current code (`sort_values(by="area", ascending=False)` then `head(num_frames)`) unchanged, so ties in area
  resolve exactly as they do today. The new spaced, earlier-frame-wins rule of §5.2 is used by `VehReClass`
  alone.
- `src/dnt/refine/config.py`: move the VLM checks out of `RefineConfig.validate` into
  `vlm_problems(vlm)` (§3). `RefineConfig.validate` calls it, with the same messages.
- `src/dnt/refine/io.py`: move the filled-row rule out of `to_work` into `filled_mask(df)` (§4).
  `to_work` calls it.
- `src/dnt/detect/yolo/detector.py`: factor the per-frame predict and box conversion out of `detect_frames`
  into an image-level method that both `detect_frames` and the helpers call. `detect_frames` output stays
  byte-identical.
- `src/dnt/track/__init__.py`: import and export `VehReClass`, and drop the `try/except ModuleNotFoundError`
  fallback.
- `src/dnt/refine/vlm/prompts.py`: add the vehicle-class prompt.
- Docs: `docs/api/track/re_class.md` already renders the whole module, so `VehReClass` appears without a nav
  change. Fix the `ReClass` docstring errors from review issue 8 while touching the file. Update CLAUDE.md's
  note that `ReClass` may be `None`.

`dnt.track` importing `dnt.refine` follows an existing direction (`track/post_process.py` already imports
`refine.interpolate`). The VLM backends are imported only when `vlm` is set.

## 9. Testing

Unit tests in `tests/test_veh_reclass.py`. They need no GPU and no network: a fake detector (monkeypatched as
the existing `ReClass` tests do) and `refine.vlm.fake.FakeBackend`, with small synthetic videos or a faked
`FrameReader`.

- **Filled rows:** a named DataFrame with `interp` and the headerless file written from it give the same
  `cls` and the same `decisions`. A table with both `r3` and `interp` treats a row as filled when either is 1.
  An all-filled track goes to the fallback with `heavy` False. The output keeps the input's columns.
- **Size:** a track just below, at and above `hv_size`, with filled rows ignored. `hv_size=None` flags nothing.
  For 1, 2, 5, 10 and 20 real rows, one outlier far above `hv_size` (the rest well below) never flags the
  track, and `hv_min_count` rows at the threshold do flag it when the track has that many rows.
- **Sampling:** largest first, `min_gap` respected, ties to the earlier frame, short tracks, `min_gap=0` equals
  top-N.
- **Voting:** the IoU gate, highest confidence wins within a frame, classes outside the allowed set are
  ignored for heavy tracks, `support` at and just below `min_support`, one stray detection in 10 frames does
  not decide.
- **Tolerant reading:** with two tracks sampling overlapping frames, a reader that fails on a **middle**
  requested frame still reads the frames after it. Only the tracks that sampled the failed frame lose it from
  *n*. A failed VLM crop frame leaves a tile of the remaining crops.
- **Single detection pass:** frames shared by two tracks are detected once, in increasing order.
- **VLM:** a confident answer decides, `unclear` and low confidence fall back, an error and a `budget` verdict
  fall back with a warning, heavy tracks get the two-class options, `vlm=None` skips the layer. The runner is
  closed after the call, also when `ask_many` raises, and a second `re_classify` gets a fresh budget.
- **VLM config, no network:** a config whose `use` selects a named `openai_compat` endpoint (top-level backend
  `none`) is accepted and the preflight checks that endpoint. An unknown `use`, invalid VLM settings (for example
  `votes=0`), and a config resolving to backend `none` each raise `ValueError` in the constructor. The caller's
  config object is unchanged. `RefineConfig.validate` gives the same messages as before the move.
- **Fallback:** heavy becomes truck, other becomes car, no real rows becomes car.
- **Output:** `cls` on every row of a classified track, other columns and row order unchanged, `track_ids`
  leaves other tracks alone, the headerless `out_file` reads back with `read_track`.
- **Empty input:** an empty DataFrame and a zero-byte file, each with `out_file` set, return an empty table,
  write a zero-byte file and leave `decisions` empty. A missing video raises `FileNotFoundError` even with empty
  input. The test checks the written file's size, not `read_track` on it, since `read_track` still raises on an
  empty file.
- **`decisions`:** one row per classified track with the deciding layer.
- **Regression:** `ReClass.re_classify` gives the same result before and after the helper change, including an
  area tie (equal-area rows in frame order `[9, 1, 5]` with `num_frames=1` still sample frame 9), and
  `detect_frames` output is unchanged. `refine.io.to_work` drops the same rows after the `filled_mask` move.
- **Constructor:** each rule in §3 has a failing case, including `vlm_crops=0`, `num_frames=10.0`,
  `hv_min_count=True`, and `hv_size=float("inf")`.
