# Quickstart

## Detect and track a video

```python
from dnt.detect import Detector, DetectorModel
from dnt.track import Tracker, BoTSORTConfig

det = Detector(model=DetectorModel.YOLO26x, conf=0.25, device="auto")
det.detect("cam1.mp4", iou_file="cam1_det.txt")        # detections, 8 columns

tracker = Tracker(config=BoTSORTConfig())              # the config picks the BoxMOT backend
tracks = tracker.track("cam1_det.txt", "cam1_track.txt", video_file="cam1.mp4")
```

Weights download on first use into `src/dnt/detect/yolo/models/`. A custom `weights=` path is
relative to that directory unless it is absolute.

### Choosing methods

- **Detector model**: pick a `DetectorModel` member (`YOLO26x` is the default; `n` and `s`
  sizes are faster). `classes=[...]` keeps only the classes you need.
- **Tracking backend**: pass one of `BoTSORTConfig`, `ByteTrackConfig`, `OCSORTConfig`,
  `StrongSORTConfig`, `DeepOCSORTConfig`, `HybridSORTConfig`, `BoostTrackConfig` or
  `SFSORTConfig`. Motion-only backends (ByteTrack, OC-SORT) need no ReID weights.
- **Several videos**: `Detector.detect_batch` and `Tracker.track_batch` take lists of files.
- **Saved settings**: `config.export_yaml(path)` writes a config and
  `Tracker(config_yaml=path)` reads it back.

### Reading the output

| Columns | Meaning |
|---|---|
| `frame` | Frame number in the source video |
| `track` | Track ID (detections have `res` instead, the detector's resolution field) |
| `x, y, w, h` | Box top-left corner, width and height, in pixels |
| `conf` / `score` | Detector confidence / tracker score |
| `class` / `cls` | Class index from the detector's class list |
| `r3, r4` | Reserved |

## Refine the tracks

```python
from dnt.refine import RefineConfig, TrackRefiner

cfg = RefineConfig.defaults("vehicle")                 # or "person"
cfg.encoder.kind = "none"                              # motion only; no extra install
refiner = TrackRefiner(config=cfg)
refiner.refine("cam1_track.txt", "cam1_refined.txt", video_file="cam1.mp4")
```

The refiner splits tracks that reverse or switch IDs, drops false tracks, merges duplicates,
links broken tracks and fills gaps. Every edit lands in a JSONL ledger next to the output.
From a shell: `dnt-refine run cam1_track.txt --fps 10 --config refine.yaml --out clean.csv`.
Appearance encoders and VLM verification are described under
[Track Refinement](api/refine/index.md).

## Re-classify, filter and label

```python
from dnt.track import ReClass
from dnt.filter import Filter
from dnt.label import Labeler

# cars, buses and trucks; kind="ped" and kind="twowheel" are the other two
tracks = ReClass(kind="veh").re_classify("cam1_refined.txt", "cam1.mp4",
                                         out_file="cam1_veh.txt")

# keep the tracks that cross a zone (shapely polygons in pixel coordinates)
inside = Filter.filter_tracks_by_zones(tracks, zones=[zone_polygon])

Labeler().draw_tracks("cam1.mp4", "cam1_labeled.mp4", tracks=inside, tail_length=30)
```

- `Filter` also filters detections (`filter_iou`, `deduplicate_boxes`) and tracks by lines
  (`filter_tracks_by_lines`).
- `Labeler` also cuts clips (`clip`, `clip_by_time`) and exports frames
  (`export_frames`, `export_track_frames`).
- `VehReClass` and `PedReClass` can ask a vision-language model about hard cases when you give
  them a `vlm=` setting; see [Re-classification](api/track/re_class.md).

## Working with the files directly

The stage files are plain headerless CSV. `dnt.shared.files` holds the readers and writers, so
you can load any stage into pandas without the stage classes, and `dnt.engine` has the box
helpers (IoU, IoB, interpolation) used throughout. Browse the
[API Reference](api/index.md) for every class and function.
