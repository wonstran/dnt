"""Full detect -> track (BoT-SORT) -> label pipeline using DNT."""

import sys
import time
from pathlib import Path

# Allow running this script directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dnt.detect import Detector  # ruff: ignore[module-import-not-at-top-of-file]
from dnt.label import Labeler  # ruff: ignore[module-import-not-at-top-of-file]
from dnt.track import BoTSORTConfig, ReIDWeights, Tracker  # ruff: ignore[module-import-not-at-top-of-file]

input_video = "/mnt/e/videos/sample/traffic.mp4"
det_file = "/mnt/e/videos/sample/dets/traffic_det.txt"
track_file = "/mnt/e/videos/sample/tracks/traffic_track.txt"
label_file = "/mnt/e/videos/sample/labels/traffic_track.mp4"

tic = time.time()

# 1) Detection
detector = Detector(device="auto", class_names=["car", "truck", "bus", "motorcycle"])
detector.detect(input_video, iou_file=det_file, message="vehicle")

# 2) Tracking (BoT-SORT: motion + camera-motion compensation + ReID, on by default)
cfg = BoTSORTConfig(reid_weights=ReIDWeights.CLIP_VEHICLEID)
print(cfg)
tracker = Tracker(cfg, device="auto")
tracker.track(det_file, track_file, input_video, message="vehicle")

toc = time.time()
print("Time:", int(toc - tic))

# 3) Labeling
labeler = Labeler()
labeler.draw_tracks(track_file=track_file, input_video=input_video, output_video=label_file)

print("ok")
