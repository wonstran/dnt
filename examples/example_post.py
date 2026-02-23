"""Example script for object tracking using DNT library."""

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
from dnt.track import BoTSORTConfig, ReIDWeights, Tracker, interpolate_tracks_rts

input_video = "/mnt/d/videos/sample/short-off-ramp.mp4"
det_file = "/mnt/d/videos/sample/dets/short-off-ramp_det.txt"
track_file = "/mnt/d/videos/sample/tracks/short-off-ramp_track.txt"
track_interp_file = "/mnt/d/videos/sample/tracks/short-off-ramp_track_interp.txt"
label_file = "/mnt/d/videos/sample/labels/short-off-ramp_track.mp4"
label_interp_file = "/mnt/d/videos/sample/labels/short-off-ramp_track_interp.mp4"
label_dir = "/mnt/d/videos/sample/labels/track_clips"

detector = Detector(model=DetectorModel.RTDETRx, device="auto")
# detector.detect(input_video, det_file)

cfg = BoTSORTConfig(ReIDWeights.CLIP_VEHICLEID)
tracker = Tracker(cfg, device="auto")
tracks = tracker.track(det_file, track_file, input_video)

labeler = Labeler(LabelMethod.CHROME_SAFE)
labeler.draw_tracks(track_file=track_file, input_video=input_video, output_video=label_file, fill=True, alpha=0.2)

tracks_interp = interpolate_tracks_rts(track_file=track_file, output_file=track_interp_file, verbose=True)
labeler.draw_tracks(
    track_file=track_interp_file,
    input_video=input_video,
    output_video=label_interp_file,
    fill=True,
    alpha=0.2,
)

print("ok")
