from dnt.analysis import StopAnalyzer
from dnt.label import Labeler
from dnt.detect import Detector
from dnt.filter import Filter
import pandas as pd
from shapely import geometry
from geopandas import gpd
import datetime, os, tqdm


lines = pd.read_csv("/mnt/d/videos/hfst_nb/frame/before_lines.csv")
lines_v = lines.loc[lines['h']==0][['x1', 'y1', 'x2', 'y2']].values.tolist()
lines_h = lines.loc[lines['h']==1][['x1', 'y1', 'x2', 'y2']].values.tolist()

v_coords = []
for line in lines_v:
    v_coords.append([(line[0], line[1]), (line[2], line[3])])

h_coords = []
for line in lines_h:
    h_coords.append([(line[0], line[1]), (line[2], line[3])])

event_dict = {
        0: [0],     # Stop
        1: [1],     # Pass
        2: [2]      # Invade
    }

rt_zones = [[(116, 209), (175, 204), (188, 332), (113, 340)]]

video_file = "/mnt/d/videos/hfst_nb/labels/Standard_SCU7VM_2022-09-13_1600.003_lines.mp4"
track_file = "/mnt/d/videos/hfst_nb/dets/Standard_SCU7VM_2022-09-13_1600.003_track.txt"
track_file_filtered = "/mnt/d/videos/hfst_nb/dets/Standard_SCU7VM_2022-09-13_1600.003_track_filtered.txt"
track_file2 = "/mnt/d/videos/hfst_nb/stops/Standard_SCU7VM_2022-09-13_1600.003_track2.txt"
result_file = "/mnt/d/videos/hfst_nb/stops/Standard_SCU7VM_2022-09-13_1600.003_result.txt"
label_file = "/mnt/d/videos/hfst_nb/stops/Standard_SCU7VM_2022-09-13_1600.003_label.txt"
label_video = "/mnt/d/videos/hfst_nb/stops/Standard_SCU7VM_2022-09-13_1600.003_stop.mp4"

init_time_str = "2022-09-13 6:00:01 PM"

tracks = pd.read_csv(track_file, header=None)

z = []
for zone in rt_zones:
    z.append(geometry.Polygon(zone))
geo_zones = geometry.MultiPolygon(z)

tracks = Filter.filter_tracks(tracks, exclude_zones=geo_zones)
tracks.to_csv(track_file_filtered, header=None, index=False)

analyzer = StopAnalyzer(h_coords=h_coords, v_coords=v_coords, event_dict=event_dict)
analyzer.analysis(track_file=track_file_filtered, result_file=result_file, output_file=track_file2)
results = pd.read_csv(result_file)
results = results.loc[results['LANE']!=-1]
results.to_csv(result_file, index=False)

if init_time_str:
    init_time=datetime.datetime.strptime(init_time_str, '%Y-%m-%d %I:%M:%S %p')
    fps = Detector.get_fps(video_file)
    StopAnalyzer.add_timestamp(result_file, result_file, init_time, fps)

StopAnalyzer.export_label(track_file2, result_file, label_file, label_field=6, event_label=['STOP', 'PASS', 'INVADE'], vid_disp=False)

labeler = Labeler(compress_message=False, label_fields=[6], zoom_factor=0.5)
labeler.draw(label_file, video_file, label_video, video_index=1, video_tot=1)