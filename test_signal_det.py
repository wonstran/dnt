from dnt.detect.signal.detector import SignalDetector
from dnt.label.labeler2 import Labeler
import os, pandas as pd

input_video = '/mnt/d/videos/ped2stage/05032024/wb_20240417.mp4'
det_file = os.path.join(os.path.dirname(input_video), 'signal', os.path.splitext(os.path.basename(input_video))[0]+'_sig_det.csv')
sig_file = os.path.join(os.path.dirname(input_video), 'signal', os.path.splitext(os.path.basename(input_video))[0]+'_sig.csv')
label_file = os.path.join(os.path.dirname(input_video), 'signal', os.path.splitext(os.path.basename(input_video))[0]+'_sig_label.csv')
label_video = os.path.join(os.path.dirname(input_video), 'signal', os.path.splitext(os.path.basename(input_video))[0]+'_sig_label.mp4')

detector = SignalDetector(det_zones=[(400, 10, 25, 25)], batchsz=1024, weights='/mnt/d/videos/ped2stage/05032024/signal/wb_ped_signal.pt')
#detector = SignalDetector(det_zones=[(330, 55, 30, 30)], batchsz=1024)
#dets = detector.detect(input_video=input_video, det_file=det_file)
dets = pd.read_csv(det_file)
#sigs = detector.gen_ped_interval(dets=dets, input_video=input_video, walk_interval=7, countdown_interval=14, out_file=sig_file)
#labels = detector.generate_labels(signals=sigs, input_video=input_video, label_file=label_file, size_factor=1.2, thick=2)

labeler = Labeler()
labeler.draw(input_video=input_video, output_video=label_video, draw_file=label_file, start_frame=20*60*60*25, end_frame=21*60*60*25)



