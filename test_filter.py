from dnt.filter import Filter
from dnt.label.labeler2 import Labeler
from shapely import LineString, Polygon
import os, pandas as pd

input_video = '/mnt/d/videos/ped2stage/05032024/wb_20240417.mp4'

# Generate paths
basename = os.path.splitext(os.path.basename(input_video))[0]
pathname = os.path.dirname(input_video)
veh_track_file = os.path.join(pathname, 'tracks', basename+'_veh_track.txt')
veh_th_track_file = os.path.join(pathname, 'tracks', basename+'_veh_th_track.txt')
veh_rt_track_file = os.path.join(pathname, 'tracks', basename+'_veh_rt_track.txt')

print('Loading tracks ...')
tracks = pd.read_csv(veh_track_file)

rt_zones = [Polygon([(0, 237), (316, 228), (259, 377), (0, 344)])]
#th_zones = [Polygon([(0, 237), (316, 228), (368, 97), (232, 107), (111, 123), (0, 144)])]

th_zones = [Polygon([(0, 237), (316, 228), (864, 229), (854, 122), (559, 95), (368, 97), (232, 107), (111, 123), (0, 144)])]
#th_zones = [Polygon([(0, 211), (332, 191), (351, 141), (112, 160), (0, 178)]),    # t1
#            Polygon([(351, 141), (112, 160), (0, 178), (0, 158), (114, 136), (264, 120), (360, 115)]),    # t2
#            Polygon([(0, 158), (114, 136), (264, 120), (360, 115), (368, 97), (232, 107), (111, 123), (0, 144)])] # t3

#th_tracks = Filter.filter_frames_by_zones_agg(tracks=tracks, zones=th_zones)
rt_tracks = Filter.filter_tracks_by_zones_agg(tracks=tracks, zones=rt_zones, method='exclude')
print(len(rt_tracks['track'].unique()))

#th_tracks.to_csv(veh_th_track_file, index=False, header=None)
#rt_tracks.to_csv(veh_rt_track_file, index=False, header=None)
#print(len(rt_tracks['track'].unique()))