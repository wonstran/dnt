"""Bounding box interpolation module for object tracking."""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


# Define the function signature
def interpolate_bbox(boxes: np.ndarray, frames: np.ndarray, target_frame: int, method="cubic") -> np.ndarray:
    """Interpolates bounding boxes using cubic splines.

    Args:
        boxes: a (n, 4) array of bounding box coordinates (x, y, width, height).
        frames: a (n,) array of frame indices corresponding to the bounding boxes.
        target_frame: the frame index at which to interpolate the bounding box.
        method: the spline function, default is 'cubic', 'nearest', 'linear'

    Returns:
        A 1d array (x, y, width, height) representing the interpolated bounding box.

    """
    n = boxes.shape[0]

    if n != frames.shape[0]:
        raise ValueError("Length of boxes and frames must be equal.")

    if n == 0:
        raise ValueError("Input arrays must not be empty.")

    if target_frame < frames[0] or target_frame > frames[-1]:
        raise ValueError("Target frame is out of bounds.")

    # Unpack the boxes into separate arrays for x, y, width, and height
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    # Create the cubic splines for each parameter
    if method == "cubic":
        spline_x = CubicSpline(frames, x)
        spline_y = CubicSpline(frames, y)
        spline_w = CubicSpline(frames, w)
        spline_h = CubicSpline(frames, h)
    elif method == "nearest":
        spline_x = interp1d(frames, x, kind="nearest")
        spline_y = interp1d(frames, y, kind="nearest")
        spline_w = interp1d(frames, w, kind="nearest")
        spline_h = interp1d(frames, h, kind="nearest")
    else:
        spline_x = interp1d(frames, x, kind="linear")
        spline_y = interp1d(frames, y, kind="linear")
        spline_w = interp1d(frames, w, kind="linear")
        spline_h = interp1d(frames, h, kind="linear")

    # Evaluate the splines at the target frame
    x_t = int(spline_x(target_frame))
    y_t = int(spline_y(target_frame))
    w_t = int(spline_w(target_frame))
    h_t = int(spline_h(target_frame))

    return np.array([x_t, y_t, w_t, h_t])


def interpolate_bboxes(boxes: np.ndarray, frames: np.ndarray, target_frames: np.ndarray, method="cubic") -> np.ndarray:
    """Interpolates bounding boxes using cubic splines.

    Args:
        boxes: a (n, 4) array of bounding box coordinates (x, y, width, height).
        frames: a (n,) array of frame indices corresponding to the bounding boxes.
        target_frames: the frame indexes at which to interpolate the bounding boxes.
        method: the spline function, default is 'cubic', 'nearest', 'linear'

    Returns:
        A (m, 4) array of interpolated bounding boxes for the target frames.

    """
    n_frames = target_frames.shape[0]
    results = []

    for i in range(n_frames):
        target_frame = target_frames[i]
        results.append(interpolate_bbox(boxes, frames, target_frame, method))

    return np.vstack(results)
