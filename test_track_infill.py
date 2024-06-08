from dnt.track import Tracker
from dnt.label.labeler2 import Labeler
from dnt.filter import Filter
import pandas as pd
from tqdm import tqdm
import os

input_video = '/mnt/d/videos/ped2stage/05032024/wb_20240417.mp4'

# Generate paths
basename = os.path.splitext(os.path.basename(input_video))[0]
pathname = os.path.dirname(input_video)
veh_th_track_file = os.path.join(pathname, 'tracks', basename+'_veh_th_track.txt')
ped_track_file = os.path.join(pathname, 'tracks', basename+'_ped_track.txt')
label_path = os.path.join(pathname, "labels_veh")
track_video1 = os.path.join(pathname, 'labels', basename+"_track1.mp4")
track_video2 = os.path.join(pathname, 'labels', basename+"_track2.mp4")

id = 560

tracks = pd.read_csv(ped_track_file)
'''
tracks.columns = Tracker.export_track_header()
tracks_grouped = tracks.groupby('track')
ids = tracks['track'].unique().tolist()
results = []
for id in ids:
    frames = tracks_grouped.get_group(id)
    ma = max(frames['frame'].values)
    mi = min(frames['frame'].values)

    if (ma-mi+1) > len(frames):
        results.append(id)

pd.DataFrame(results).to_csv(os.path.join(label_path, 'ids.csv'), index=False)
input('...')
'''

tracks_infilled = Tracker.infill_frames(tracks, ids=[id], inplace=False)
#tracks_selected = tracks[tracks['track']==id]
tracks_infilled.to_csv(os.path.join(label_path, 'a.csv'), index=False)

#tracks_a = Tracker.cluster_frames(tracks, inplace=False)

labler = Labeler()
labler.draw_track_clips(input_video, label_path, tracks_infilled, thick=2)
#Labeler.export_track_frames(input_video=input_video, tracks=tracks_infilled, output_path=label_path, bbox=True)
#Labeler.export_track_frames(input_video=input_video, tracks=tracks_infilled, output_path=label_path, bbox=True, prefix='infill')

