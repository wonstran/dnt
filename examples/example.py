import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from dnt.detect import Detector  # noqa: E402
from dnt.label.labeler import Labeler  # noqa: E402
from dnt.track import Config, Tracker  # noqa: E402

video_file = "/mnt/d/videos/sample/traffic.mp4"
det_file = "/mnt/d/videos/sample/dets/traffic_det.txt"
track_file = "/mnt/d/videos/sample/tracks/traffic_track.txt"
label_file = "/mnt/d/videos/sample/labels/traffic.mp4"

"""
video_file = "/mnt/d/videos/samples/traffic_short.mp4"
mot_file = "/mnt/d/videos/samples/traffic_short_mot.txt"
iou_file = "/mnt/d/videos/samples/dets/traffic_short_iou.txt"
iou_file_filtered = "/mnt/d/videos/samples/dets/traffic_short_iou_filtered.txt"
track_file = "/mnt/d/videos/samples/tracks/traffic_short_track.txt"
label_file = "/mnt/d/videos/samples/labels/traffic_short_track.mp4"
label_file_2 = "/mnt/d/videos/samples/labels/traffic_short_det.mp4"
"""
detector = Detector()
ious = detector.detect(video_file, iou_file=det_file, show_filename=True)
# ious = detector.detect_frames(input_video=video_file, frames=list(range(1, 100)))

# tracker = Tracker(cfg=Config.get_cfg_dsort())
tracker = Tracker(cfg=Config.get_cfg_botsort())
tracker.track(det_file, track_file, video_file)
"""
class_list = [0, 1, 2, 3, 5, 7]   
filter_class_iou(iou_file, iou_file_filtered, class_list)

tracker = Tracker(cfg=Config.get_cfg_botsort())
tracker.track(iou_file_filtered, track_file, video_file, video_index=1, video_tot=1)
"""
labeler = Labeler()
# labeler.draw_dets(input_video=video_file, output_video=label_file, det_file=det_file)
labeler.draw_tracks(track_file=track_file, input_video=video_file, output_video=label_file, tail=100, class_name=True)

print("ok")
