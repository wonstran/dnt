from dnt.analysis.stop import StopAnalyzer
from dnt.label import Labeler
from dnt.detect import Detector
import pandas as pd
import datetime    

v_coords = [[(256, 228),(408, 337)], 
          [(334, 212),(515, 297)],
          [(403, 197),(596, 268)],
          [(455, 187),(651, 247)],
          [(507, 174),(693, 231)]]

h_coords = [[(256, 228),(506, 175)], 
          [(290, 256),(557, 188)],
          [(345, 291),(635, 211)],
          [(380, 313),(672, 225)],
          [(408, 337),(691, 232)]]
    
event_dict = {
        0: [0, 1, 2],    # Stop
        1: [3]         # Pass
    }

track_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_track.txt"
label_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_label.txt"
result_file = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_result.txt"
result_file2 = "/mnt/d/videos/hfst/tracks/Standard_SCU7WH_2022-09-16_0630.02.001_result2.txt"
label_video = "/mnt/d/videos/hfst/labels/Standard_SCU7WH_2022-09-16_0630.02.001_label.mp4"
video_file = "/mnt/d/videos/hfst/labels/Standard_SCU7WH_2022-09-16_0630.02.001_lines_3.mp4"

analyzer = StopAnalyzer(h_coords=h_coords, v_coords=v_coords, event_dict=event_dict)
analyzer.analysis(track_file=track_file, result_file=result_file, video_index=1, video_tot=1)

results = pd.read_csv(result_file)
results = results.loc[results['LANE']!=-1]
results.to_csv(result_file, index=False)

fps = Detector.get_fps(video_file)
init_time = datetime.datetime.strptime('2022-09-16 10:47:08 AM', '%Y-%m-%d %I:%M:%S %p')
StopAnalyzer.add_timestamp(result_file, result_file2, init_time=init_time, fps=fps)

results = pd.read_csv(result_file2)
results = results.loc[results['DURATION']>=1]
results.to_csv(result_file2, index=False)

StopAnalyzer.export_label(track_file, result_file2, label_file, label_field=6, event_label=['STOP', 'PASS'], vid_disp=False)

labeler = Labeler(compress_message=False, label_fields=[6], nodraw_empty=True, zoom_factor=0.5)
labeler.draw(label_file, video_file, label_video)

