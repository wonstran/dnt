from dnt.track import Tracker
from dnt.label.labeler2 import Labeler
from dnt.filter import Filter
from dnt.analysis import YieldAnalyzer
import pandas as pd
from shapely import Polygon
import os

input_video = '/mnt/d/videos/ped2stage/05032024/wb_20240417.mp4'

cross_zone = Polygon([(345, 480), (401, 96), (498, 94), (551, 146), (665, 283), (802, 479)])
rt_zone = Polygon([(0, 237), (316, 228), (259, 377), (0, 344)])

# Generate paths
basename = os.path.splitext(os.path.basename(input_video))[0]
pathname = os.path.dirname(input_video)
ped_track_file = os.path.join(pathname, 'tracks', basename+'_cross_ped_track.txt')
rt_track_file = os.path.join(pathname, 'tracks', basename+'_rt_veh_track.txt')
yield_file = os.path.join(pathname, 'yield', basename+"_yield.csv")
label_path = os.path.join(pathname, 'labels_yield')

ped_tracks = pd.read_csv(ped_track_file, header=None)
rt_tracks = pd.read_csv(rt_track_file, header=None)
ped_tracks = Tracker.infill_frames(ped_tracks)
rt_tracks = Tracker.infill_frames(rt_tracks)

analyzer = YieldAnalyzer(waiting_dist_p=230, waiting_dist_y=600, yield_gap=50, p_zone=cross_zone, y_zone=rt_zone)
events = analyzer.analyze(ped_tracks, rt_tracks, name_p='peds', name_y='rt_vehs')
events.to_csv(yield_file, index=False)

#events = pd.read_csv(yield_file)

YieldAnalyzer.draw_event_clips(input_video=input_video, out_path=label_path, yields=events, tracks_p=ped_tracks, tracks_y=rt_tracks, 
                               show_track=True, size=0.5, method='yield')
print('ok')

