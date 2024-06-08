from dnt.track import Tracker
from dnt.label.labeler2 import Labeler
from dnt.filter import Filter
from dnt.analysis.stop3 import StopAnalyzer
import pandas as pd
from tqdm import tqdm
import os

input_video = '/mnt/d/videos/ped2stage/videos/gh1291.mp4'

# Generate paths
basename = os.path.splitext(os.path.basename(input_video))[0]
pathname = os.path.dirname(input_video)
veh_th_track_file = os.path.join(pathname, 'tracks', basename+'_track_veh_th.txt')
veh_stop_file = os.path.join(pathname, 'stops', basename+"_veh_th_stop.csv")
veh_stop_track_file = os.path.join(pathname, 'stops', basename+"_veh_th_stop_track.csv")
label_path = os.path.join(pathname, "labels")

print('Loading tracks ...')
tracks = pd.read_csv(veh_th_track_file)
tracks = tracks[tracks.iloc[:,0]>=(50*30)].copy()

rt_zones = [(0, 620), (667, 634), (1567, 642), (1830, 958), (653, 817), (0, 763)]
th_zones = [(0, 620), (667, 634), (1567, 642), (1440, 493), (761, 492), (0, 507)]

stop_zones = [ [(0, 765), (269, 502), (761, 492), (550, 810)],              # pre-stop
                [(761, 492), (550, 810), (652, 817), (814, 492)],           # after-stop
                [(959, 490), (1089, 870), (1830, 957), (1440, 493)],        # after-crosswalk
                [(652, 817), (814, 492), (959, 490), (1089, 870)]        # crosswalk
                ]          

lane_zones = [[(0, 620), (667, 634), (1567, 642), (1506, 569), (709, 567), (0, 565)],    # t1
                  [(0, 565), (709, 567), (1506, 569), (1475, 532), (739, 524), (0, 534)],    # t2
                  [(0, 534), (739, 524), (1475, 532), (1440, 493), (761, 492), (0, 507)]] # t3

stop_zones = StopAnalyzer.gen_zones(stop_zones)
lane_zones = StopAnalyzer.gen_zones(lane_zones)

labeler = Labeler()
labeler.draw_track_clips(input_video=input_video, output_path=label_path, tracks=tracks, method='random', thick=2)

print('ok')