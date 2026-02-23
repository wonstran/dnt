"""Example script for object tracking using DNT library."""

import os
import sys
from pathlib import Path

# Allow running this script directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dnt.detect import Detector, DetectorModel
from dnt.label import Labeler, LabelMethod
from dnt.label.labeler import Labeler, LabelMethod
from dnt.track import BoTSORTConfig, ReIDWeights, Tracker, interpolate_tracks_rts, link_tracklets

base_dir = "/mnt/e/videos/tmc/before/009/View_South/videos/4-19-23"
input_video = f"{base_dir}/4-19-23-hiv00307.mp4"
det_file = f"{base_dir}/dets/4-19-23-hiv00307_iou.txt"
track_file = f"{base_dir}/tracks/4-19-23-hiv00307_ped_track_bot.txt"
track_interp_file = f"{base_dir}/tracks/4-19-23-hiv00307_ped_track_bot_interp.txt"
tracks_linked_file = f"{base_dir}/tracks/4-19-23-hiv00307_ped_track_bot_linked.txt"
label_dir = f"{base_dir}/labels/track_clips"
os.makedirs(label_dir, exist_ok=True)

detector = Detector(model=DetectorModel.RTDETRx, device="auto")
# detector.detect(input_video, det_file)

cfg = BoTSORTConfig(ReIDWeights.CLIP_VEHICLEID)
tracker = Tracker(cfg, device="auto")
# tracks = tracker.track(det_file, track_file, input_video)

labeler = Labeler(LabelMethod.CHROME_SAFE)

labeler.draw_track_clips(
    track_file=track_file, input_video=input_video, output_path=label_dir, label_prefix=False, fill=True, alpha=0.2
)
tracks_linked = link_tracklets(track_file=track_file, output_file=tracks_linked_file, max_gap=30)
tracks_interp = interpolate_tracks_rts(track_file=track_file, output_file=track_interp_file, verbose=True)

labeler.draw_track_clips(
    track_file=tracks_linked_file,
    input_video=input_video,
    output_path=label_dir,
    label_prefix=True,
    fill=True,
    alpha=0.2,
)

print("ok")
