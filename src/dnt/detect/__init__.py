"""Detection module for object detection using YOLOv8."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from .yolov8.detector import Detector, DetectorModel

__all__ = ["Detector", "DetectorModel"]
