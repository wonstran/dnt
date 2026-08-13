import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from dnt.detect import Detector, DetectorModel  # noqa: E402

video_file = "/mnt/d/videos/sample/traffic.mp4"
det_file = "/mnt/d/videos/sample/dets/traffic_det.txt"
track_file = "/mnt/d/videos/sample/tracks/traffic_track.txt"
label_file = "/mnt/d/videos/sample/labels/traffic.mp4"

detector = Detector(model=DetectorModel.YOLO26m)
ious = detector.detect(video_file, iou_file=det_file)

# labeler = Labeler()
# labeler.draw_dets(input_video=video_file, output_video=label_file, det_file=det_file)

print("ok")
