import sys
from pathlib import Path

# add project src/ to import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dnt.detect import Detector  # noqa: E402

video_path = Path("/mnt/d/videos/sample")
det_path = Path("/mnt/d/videos/samples/dets")
track_path = Path("/mnt/d/videos/samples/tracks")
label_path = Path("/mnt/d/videos/samples/labels")

# collect mp4s
input_videos = list(video_path.glob("*.mp4"))

# create output folder if it doesn't exist
det_path.mkdir(parents=True, exist_ok=True)

detector = Detector(half=True, weights="n")
dets = detector.detect_batch(
    [str(p) for p in input_videos],
    output_path=str(det_path),
    is_overwrite=True,
)

'''
print('Tracking ...')
cfg = Tracker.export_default_cfg("dsort")
cfg['max_age'] = 5
cfg['nn_budget'] = 30
cfg['nms_max_overlap'] = 1

tracker = Tracker(method='dsort', deepsort_cfg=cfg)
tracks = tracker.track_batch(dets, input_videos, track_path, is_overwrite=True)

print('Labeling ...')
labeler = Labeler()
labeler.draw_batch(tracks, input_videos, label_path, is_overwrite=True)
'''
print("ok")