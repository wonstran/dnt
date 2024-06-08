from dnt.label.labeler2 import Labeler
from shapely import LineString
import os, pandas as pd

input_video = '/mnt/d/videos/ped2stage/05032024/wb_20240417.mp4'

# Generate paths
basename = os.path.splitext(os.path.basename(input_video))[0]
pathname = os.path.dirname(input_video)
ped_track_file = os.path.join(pathname, 'tracks', basename+'_ped_track.txt')
cross_ped_track_file = os.path.join(pathname, 'tracks', basename+'_cross_ped_track.txt')

tracks = pd.read_csv(ped_track_file, header=None)
selected = tracks[tracks[1]==902]
Labeler.export_track_frames(input_video=input_video, tracks=selected, output_path=os.path.join(pathname, 'frames2'))

