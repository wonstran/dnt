from dnt.label.labeler2 import Labeler
from dnt.track import Tracker
from datetime import datetime
import pandas as pd
import glob, os
from dnt.shared.util import load_class_dict

input_video = "/mnt/d/videos/ped2stage/videos/gh1306.mp4"
track_file = "/mnt/d/videos/ped2stage/tracks/gh1306_track_botsort.txt"
track_video = "/mnt/d/videos/ped2stage/labels/gh1306_track_botsort.mp4"

'''
input_video = "/mnt/d/videos/samples/traffic.mp4"
det_file = "/mnt/d/videos/samples/dets/traffic_iou.txt"
track_file = "/mnt/d/videos/samples/tracks/traffic_track.txt"
track_label = "/mnt/d/videos/samples/tracks/traffic_label.txt"
det_video = "/mnt/d/videos/samples/labels/traffic_iou.mp4"
track_video = "/mnt/d/videos/samples/labels/traffic_track.mp4"

cfg = Tracker.export_default_cfg('dsort')
cfg['max_age'] = 5
cfg['nn_budget'] = 30
cfg['nms_max_overlap'] = 0.5
print(cfg)
tracker = Tracker('dsort', deepsort_cfg=cfg)
tracker.track(det_file, track_file, input_video)
tracks = pd.read_csv(track_file, header=None)
'''
labeler = Labeler()
labeler.draw_tracks(input_video=input_video, output_video=track_video, track_file=track_file, class_name=True)
#dets = pd.read_csv(det_file, header=None)
#labeler.draw_dets(input_video, det_video, dets = dets, class_name=True)

#frames = range(800,1000)
#labeler.export_frames(input_video, frames, "/mnt/d/videos/ped2stage/frames")

print("ok")