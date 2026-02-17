"""Detection module for object detection using YOLO."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from .yolo.detector import Detector, DetectorModel

__all__ = ["Detector", "DetectorModel"]
