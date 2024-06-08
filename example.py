from dnt.detect import Detector
from dnt.shared.filter import filter_class_iou
from dnt.track import Tracker, Config
from dnt.label.labeler2 import Labeler
from datetime import datetime
import pandas as pd

video_file = "/mnt/d/videos/samples/traffic_short.mp4"
mot_file = "/mnt/d/videos/samples/traffic_short_mot.txt"
iou_file = "/mnt/d/videos/samples/dets/traffic_short_iou.txt"
iou_file_filtered = "/mnt/d/videos/samples/dets/traffic_short_iou_filtered.txt"
track_file = "/mnt/d/videos/samples/tracks/traffic_short_track.txt"
label_file = "/mnt/d/videos/samples/labels/traffic_short_track.mp4"
label_file_2 = "/mnt/d/videos/samples/labels/traffic_short_det.mp4"

detector = Detector(model='rtdetr')
ious = detector.detect(video_file, iou_file=iou_file)

'''
class_list = [0, 1, 2, 3, 5, 7]   
filter_class_iou(iou_file, iou_file_filtered, class_list)

tracker = Tracker(cfg=Config.get_cfg_botsort())
tracker.track(iou_file_filtered, track_file, video_file, video_index=1, total_videos=1)
'''
labeler = Labeler()
labeler.draw_dets(input_video=video_file, output_video=label_file_2, det_file=iou_file)
#labeler.draw_tracks(track_file=track_file, input_video=video_file, output_video=label_file, tail = 100, class_name=True)

print("ok")