from dnt.detect import Detector
from dnt.shared.filter import filter_class_iou
from dnt.track import Tracker
from dnt.label import Labeler
from datetime import datetime
import pandas as pd
import glob, os

video_path = "/mnt/d/videos/samples"
det_path = "/mnt/d/videos/samples/dets"
track_path = "/mnt/d/videos/samples/tracks"
label_path = "/mnt/d/videos/samples/labels"

input_videos = glob.glob(os.path.join(video_path, '*.mp4'))

print('Detecting ...')
detector = Detector()
dets = detector.detect_batch(input_videos, det_path, is_overwrite=True)

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

print("ok")