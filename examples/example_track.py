"""Example script for object tracking using DNT library."""

import sys
import time
from pathlib import Path

# Allow running this script directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dnt.label.labeler import Labeler
from dnt.track import ByteTrackConfig, Tracker

"""
input_video = "/mnt/d/videos/samples/ped_veh.mp4"
det_file = "/mnt/d/videos/samples/dets/ped_veh_iou.txt"
track_file = "/mnt/d/videos/samples/tracks/ped_veh_track.txt"
label_video = "/mnt/d/videos/samples/labels/ped_veh_track.mp4"

input_video = "/mnt/d/videos/samples/traffic.mp4"
det_file = "/mnt/d/videos/samples/dets/traffic_iou.txt"
track_file = "/mnt/d/videos/samples/tracks/traffic_track_dsort.txt"
label_video = "/mnt/d/videos/samples/labels/traffic_track.mp4"
"""
input_video = "/mnt/d/videos/sample/traffic.mp4"
det_file = "/mnt/d/videos/sample/dets/traffic_det.txt"
track_file = "/mnt/d/videos/sample/tracks/traffic_track.txt"
label_file = "/mnt/d/videos/sample/labels/traffic_track.mp4"

tic = time.time()
cfg = ByteTrackConfig()

print(cfg)
tracker = Tracker(cfg, device="auto")
tracker.track(det_file, track_file, input_video)

toc = time.time()
print("Time:", int(toc - tic))

labeler = Labeler()
labeler.draw_tracks(track_file=track_file, input_video=input_video, output_video=label_file, tail=100)

print("ok")
