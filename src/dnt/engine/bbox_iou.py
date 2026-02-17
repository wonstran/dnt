"""Compute IoU (Intersection over Union) between bounding boxes.

This module provides functions to calculate the IoU metric for bounding boxes.
"""

import numpy as np
from cython_bbox import bbox_overlaps


def ious(atlbrs, btlbrs):
    """Compute cost based on IoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_overlaps(np.ascontiguousarray(atlbrs, dtype=np.float64), np.ascontiguousarray(btlbrs, dtype=np.float64))

    return ious
