"""Example script for object detection using DNT library."""

import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from dnt.detect import Detector, DetectorModel  # ruff: ignore[module-import-not-at-top-of-file]
from dnt.label import Encoder, Labeler, LabelMethod  # ruff: ignore[module-import-not-at-top-of-file]

video_file = "/mnt/e/videos/sample/traffic.mp4"
det_file = "/mnt/e/videos/sample/dets/traffic_det.txt"
label_file = "/mnt/e/videos/sample/labels/traffic.mp4"

detector = Detector(model=DetectorModel.YOLO26x, agnostic_nms=True, half=True, fast=True, device="auto")
ious = detector.detect(video_file, iou_file=det_file)

labeler = Labeler(method=LabelMethod.CHROME_SAFE, encoder=Encoder.H264_NVENC)
labeler.draw_dets(input_video=video_file, output_video=label_file, det_file=det_file)

print("ok")
