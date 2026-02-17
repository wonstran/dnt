"""Engine module for detection tracking.

This module provides utilities for bounding box interpolation, IoU calculations,
clustering, and IoB (Intersection over Background) operations.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .bbox_interp import interpolate_bbox, interpolate_bboxes
from .bbox_iou import ious
from .cluster import cluster_by_gap
from .iob import iob, iobs

__all__ = ["cluster_by_gap", "interpolate_bbox", "interpolate_bboxes", "iob", "iobs", "ious"]
