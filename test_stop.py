from dnt.track import Tracker
from dnt.label.labeler2 import Labeler
from dnt.filter import Filter
from dnt.analysis.stop3 import StopAnalyzer
import pandas as pd
from tqdm import tqdm
import os

input_video = '/mnt/d/videos/ped2stage/05032024/wb_20240417.mp4'

# Generate paths
basename = os.path.splitext(os.path.basename(input_video))[0]
pathname = os.path.dirname(input_video)
veh_th_track_file = os.path.join(pathname, 'tracks', basename+'_veh_th_track.txt')
veh_stop_file = os.path.join(pathname, 'stops', basename+"_veh_th_stop.csv")
veh_stop_track_file = os.path.join(pathname, 'stops', basename+"_veh_th_stop_track.csv")
veh_stop_label_file = os.path.join(pathname, 'stops', basename+"_veh_th_stop_label.csv")
label_video = os.path.join(pathname, 'labels', basename+"_veh_th_stop_label.mp4")
track_video = os.path.join(pathname, 'labels', basename+"_veh_th_track.mp4")
label_path = os.path.join(pathname, "labels")
veh_stop_test_file = os.path.join(pathname, 'stops', basename+"_veh_th_stop_test.csv")

print('Loading tracks ...')
tracks = pd.read_csv(veh_th_track_file)

stop_zones = [[(0, 144), (110, 122), (234, 107), (401, 96), (345, 480), (0, 480)],                     # pre-stop
                [(368, 98), (219, 478), (345, 480), (401, 96)],                                         # after-stop
                [[498, 94], (671, 104), (854, 122), (854, 480), (802, 480), (665, 283), (551, 146)],    # after-crosswalk
                [(345, 480), (401, 96), (498, 94), (551, 146), (665, 283), (802, 479)]]                 # crosswalk  
lane_zones = [[(0, 211), (332, 191), (351, 141), (112, 160), (0, 178)],    # t1
                  [(351, 141), (112, 160), (0, 178), (0, 158), (114, 136), (264, 120), (360, 115)],    # t2
                  [(0, 158), (114, 136), (264, 120), (360, 115), (368, 97), (232, 107), (111, 123), (0, 144)]] # t3
stop_zones = StopAnalyzer.gen_zones(stop_zones)
lane_zones = StopAnalyzer.gen_zones(lane_zones)


event_dicts = [
    {'zone': 0, 'code': 0, 'desc': 'before', 'color':(0, 255, 0)},
    {'zone': 1, 'code': 1, 'desc': 'pass', 'color':(0, 165, 255)},
    {'zone': 2, 'code': 2, 'desc': 'inside', 'color':(255, 0, 0)},
    {'zone': 3, 'code': 3, 'desc': 'incursion', 'color':(0, 0, 255)}
]

'''
event_dicts = [
    {'zone': 0, 'code': 1, 'desc': 'before stop bar'},
    {'zone': 1, 'code': 2, 'desc': 'pass stop bar'},
    {'zone': 3, 'code': 3, 'desc': 'pass crosswalk'},
    {'zone': 2, 'code': 4, 'desc': 'crosswalk incursion'}
]
'''

analyzer = StopAnalyzer(stop_zones=stop_zones, lane_zones=lane_zones, event_dicts=event_dicts, lane_zone_ref='bc', 
                        ref_offset=(0, -7), stop_iou=0.98, iou_mode='mean', bbox_iob=0.03, lane_adjust=True)
results = analyzer.scan_stop(tracks)
#results = pd.read_csv(veh_stop_track_file)
results = analyzer.scan_zones(results)
results = analyzer.scan_leading_at_first_stop(results)
results = analyzer.scan_first_stop_event(results)
events = analyzer.count_event(results)
results.to_csv(veh_stop_test_file, index=False)
events.to_csv(veh_stop_file, index=False)

#results = pd.read_csv(veh_stop_track_file)
#events = pd.read_csv(veh_stop_file)

labels = analyzer.generate_labels(results, events, show_desc=True, show_track=True, size=0.5, thick=2)
labeler = Labeler()
start_min = 7*60
end_min = 8*60
labeler.draw(input_video, output_video=label_video, draws = labels)
#labeler.draw(input_video, output_video=label_video, draws = labels, start_frame=start_min*60*25, end_frame=end_min*60*25)

#labeler.draw_tracks(input_video, output_video=track_video, tracks=tracks, 
#                    start_frame=(8*60*60+7*60+7)*25-(25*60)*25, end_frame=(8*60*60+8*60+0)*25-(25*60)*25)
print('ok')