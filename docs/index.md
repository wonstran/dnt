# dnt API Manual

**Version 0.3.4.2** — see the [changelog](changelog.md).

`dnt` (Detection and Tracking) turns roadside and intersection video into clean road-user
tracks: it detects objects, tracks them across frames, refines and re-classifies the tracks,
filters them by zones and lines, and draws the result back onto video. Every stage reads the
previous stage's file and writes its own, so stages run on their own or chained.

## Install

```bash
pip install dnt                 # detection, tracking, filtering, labeling
pip install "dnt[refine]"       # track refinement: DINO/ReID encoders and VLM backends
pip install -e ".[dev]"         # from a checkout: tests and linting
pip install -e ".[docs]"        # from a checkout: build this manual
```

Python 3.11 or 3.12. Detection and deep ReID trackers use a GPU when one is available
(`device="auto"` tries cuda, then xpu, mps and cpu). `ffmpeg` is needed by `Labeler` for
the ffmpeg and `chrome_safe` methods. The `refine` extra splits into `refine-dino`,
`refine-reid` and `refine-vlm` if you need only one encoder or backend.

## Packages

| Package | What it does | Main entry point |
|---|---|---|
| [`dnt.detect`](api/detect/index.md) | Object detection with Ultralytics YOLO (v8, 11, 26) or RT-DETR; pedestrian-signal and timestamp readers | `Detector` |
| [`dnt.track`](api/track/index.md) | Multi-object tracking on BoxMOT; gap filling; track re-classification | `Tracker`, `ReClass` |
| [`dnt.refine`](api/refine/index.md) | Splits reversals and ID switches, screens false tracks, merges duplicates, links fragments, fills gaps, with an audit ledger | `TrackRefiner`, `dnt-refine` |
| [`dnt.filter`](api/filter/filter.md) | Filters detections and tracks by zones and lines | `Filter` |
| [`dnt.label`](api/label/labeler.md) | Draws detections, tracks and shapes on video; cuts clips; exports frames | `Labeler` |
| [`dnt.engine`](api/engine/index.md) | Box IoU, IoB, interpolation and gap clustering | `ious`, `iob` |
| [`dnt.shared`](api/shared/index.md) | CSV readers and writers, class-name lists, downloads, multi-video synchronization | `Synchronizer` |

## Pipeline

```text
video ─Detector─▶ detections ─Tracker─▶ tracks ─TrackRefiner─▶ refined tracks
                                                                    │
labeled video ◀─Labeler─ filtered tracks ◀─Filter─ re-classed ◀─ReClass
```

Files are headerless CSV text. Detections have 8 columns
(`frame, res, x, y, w, h, conf, class`); tracks have 10
(`frame, track, x, y, w, h, score, cls, r3, r4`). Boxes are top-left `x, y` with width and
height. The [Quickstart](quickstart.md) runs the whole chain.

## Units

- Box coordinates, widths and heights are in pixels of the source video frame.
- `Tracker` and `TrackRefiner` need the frame rate (from the video, or `--fps`) wherever
  they reason about time, such as how long a gap may be before a link is refused.
