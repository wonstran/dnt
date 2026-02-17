# DNT

Python package for video-based traffic analysis: detection, tracking, labeling, and post-processing.

## Features

- Object detection (`dnt.detect.Detector`, YOLO/RT-DETR backend).
- Multi-object tracking (`dnt.track.Tracker`, BoxMOT backend).
- Video labeling/visualization (`dnt.label.Labeler`).
- Track post-processing:
  - RTS interpolation for trajectory gaps.
  - Tracklet linking (stitching broken IDs).

## Requirements

- OS: Ubuntu 20.04+ (or compatible Linux).
- Python: 3.9+.
- CUDA GPU recommended for detection/tracking speed.

Install dependencies from:
- `requirements.txt`
- `pyproject.toml`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install dnt
```

## Quick Workflow

### 1) Detection

```python
from dnt.detect import Detector

detector = Detector(device="auto")
dets = detector.detect(
    input_video="/path/to/video.mp4",
    iou_file="/path/to/dets.txt",
    verbose=True,
)
```

### 2) Tracking

```python
from dnt.track import ByteTrackConfig, Tracker

cfg = ByteTrackConfig()
tracker = Tracker(cfg=cfg, device="auto")
tracks = tracker.track(
    input_video="/path/to/video.mp4",
    det_file="/path/to/dets.txt",
    output_file="/path/to/tracks.txt",
)
```

### 3) RTS interpolation (post-process)

```python
from dnt.track.post_process import interpolate_tracks_rts

tracks_interp = interpolate_tracks_rts(
    track_file="/path/to/tracks.txt",
    output_file="/path/to/tracks_interp.txt",
    max_gap=30,        # max consecutive missing frames to fill
    interp_col="interp",
    verbose=True,
)
```

Notes:
- `interp == 1` means interpolated frame.
- Real detections are treated as `interp != 1` (supports legacy `interp=-1` files).
- Output file is written in track-file format (no CSV header), compatible with `Labeler.draw_tracks`.

### 4) Tracklet linking (ID stitching)

```python
from dnt.track.post_process import link_tracklets

tracks_linked = link_tracklets(
    track_file="/path/to/tracks_interp.txt",
    output_file="/path/to/tracks_linked.txt",
    max_gap=20,        # candidate end-start frame gap
    verbose=True,
)
```

`link_tracklets` uses:
- hard gates (time/class/size/motion/IoU),
- cost matrix scoring,
- global 1-to-1 assignment (Hungarian),
- union-find chain merge and ID remap.

### 5) Labeling

```python
from dnt.label import Labeler

labeler = Labeler()
labeler.draw_tracks(
    input_video="/path/to/video.mp4",
    output_video="/path/to/output_labeled.mp4",
    track_file="/path/to/tracks_linked.txt",
    verbose=True,
)
```

## Modules

- `dnt.detect`
- `dnt.track`
- `dnt.label`
- `dnt.track.post_process`

## Author

Zhenyu Wang ([wonstran@hotmail.com](mailto:wonstran@hotmail.com))

## License

MIT License. See `LICENSE.md`.
