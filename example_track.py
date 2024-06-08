import os, time
from dnt.track import Tracker
from dnt.label.labeler2 import Labeler
from dnt.track import Config
'''
input_video = "/mnt/d/videos/samples/ped_veh.mp4"
det_file = "/mnt/d/videos/samples/dets/ped_veh_iou.txt"
track_file = "/mnt/d/videos/samples/tracks/ped_veh_track.txt"
label_video = "/mnt/d/videos/samples/labels/ped_veh_track.mp4"
'''
input_video = "/mnt/d/videos/samples/traffic.mp4"
det_file = "/mnt/d/videos/samples/dets/traffic_iou.txt"
track_file = "/mnt/d/videos/samples/tracks/traffic_track_dsort.txt"
label_video = "/mnt/d/videos/samples/labels/traffic_track.mp4"

tic = time.time()
cfg = Config.get_cfg_botsort()
cfg['with_reid'] = False
print(cfg)
tracker = Tracker(cfg)
tracker.track(det_file, track_file, input_video)
toc = time.time()
print('Time:', toc-tic)

labeler = Labeler()
#labeler.draw_tracks(track_file=track_file, input_video=input_video, output_video=label_video, tail = 100, cls = True)

print('ok')


