from dnt.filter import Filter
import os
from matplotlib import pyplot as plt
from shapely import Polygon, LineString
import pandas as pd

input_videos = ["/mnt/d/videos/hfst_nb/Standard_SCU7VM_2023-01-30_0630.001.mp4",
                "/mnt/d/videos/hfst_nb/Standard_SCU7VM_2023-01-31_1600.002.mp4"]
track_path = '/mnt/d/videos/hfst_nb/tracks'

rt_zones = [Polygon([(74, 270), (80, 184), (128, 171), (148, 254)])]
th_zones = [Polygon([(169, 97), (289, 169), (540, 103), (337, 70)])]

for video in input_videos:
    track_file = os.path.join(track_path, os.path.basename(video).replace('.mp4', '_veh_track.txt'))
    tracks = pd.read_csv(track_file)
    print(len(tracks))
    filter_tracks = Filter.filter_tracks_by_zones_agg(tracks=tracks, zones=th_zones, ref_point='bc', offset=(0, 5))
    print(len(filter_tracks))
    th_tracks = Filter.filter_tracks_by_zones_agg(tracks=filter_tracks, zones = rt_zones, method='exclude', ref_point='bc', offset=(0, 5))
    print(len(th_tracks))
    rt_tracks = Filter.filter_tracks_by_zones_agg(tracks=filter_tracks, zones = rt_zones, method='include', ref_point='bc', offset=(0, 5))
    print(len(rt_tracks))
    
    input('...')
    th_tracks.to_csv(os.path.join(track_path,  os.path.basename(video).replace('.mp4', '_th_veh_track.txt')), index=False, header=None)
    rt_tracks.to_csv(os.path.join(track_path,  os.path.basename(video).replace('.mp4', '_rt_veh_track.txt')), index=False, header=None)   
    
print('ok')