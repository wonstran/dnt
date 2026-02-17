"""Module for calculating Intersection over Box (IoB) metrics.

This module provides functions to compute IoB values between single and multiple
bounding boxes, useful for evaluating object detection and tracking results.
"""

import numpy as np


def iob(box_a: np.ndarray, box_b: np.ndarray) -> tuple[float, float]:
    """Calculate the Intersection over Box (IoB) between two bounding boxes.

    Parameters
    ----------
    box_a : np.ndarray
        Bounding box of shape (4,) with format [left, top, width, height].
    box_b : np.ndarray
        Bounding box of shape (4,) with format [left, top, width, height].

    Returns
    -------
    tuple[float, float]
        A tuple (iob_a, iob_b) where:
        - iob_a: ratio of intersection area to box_a area
        - iob_b: ratio of intersection area to box_b area

    Raises
    ------
    ValueError
        If either box does not have shape (4,) or contains negative width/height.

    Notes
    -----
    Bounding box format is [left, top, width, height] where:
    - left: x-coordinate of top-left corner
    - top: y-coordinate of top-left corner
    - width: box width (must be non-negative)
    - height: box height (must be non-negative)

    Examples
    --------
    >>> box1 = np.array([0, 0, 10, 10])
    >>> box2 = np.array([5, 5, 10, 10])
    >>> iob(box1, box2)
    (0.25, 0.25)

    """
    if box_a.shape != (4,) or box_b.shape != (4,):
        raise ValueError("Both boxes must have shape (4,) with format [left, top, width, height].")
    if box_a[2] < 0 or box_a[3] < 0 or box_b[2] < 0 or box_b[3] < 0:
        raise ValueError("Box width and height must be non-negative.")

    # Determine the (x, y)-coordinates of the intersection rectangle
    x_l = max(box_a[0], box_b[0])
    y_t = max(box_a[1], box_b[1])
    x_r = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, x_r - x_l)
    inter_height = max(0, y_b - y_t)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    box_a_area = box_a[2] * box_a[3]
    box_b_area = box_b[2] * box_b[3]

    # Compute the intersection over box by dividing intersection area by each box's area
    iob_a = inter_area / box_a_area if box_a_area != 0 else 0.0
    iob_b = inter_area / box_b_area if box_b_area != 0 else 0.0

    return iob_a, iob_b


def iobs(alrbs: np.ndarray, blrbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the IoB matrix for multiple bounding boxes.

    Parameters
    ----------
    alrbs : np.ndarray
        Array of shape (N, 4) containing N bounding boxes with format
        [left, top, width, height].
    blrbs : np.ndarray
        Array of shape (M, 4) containing M bounding boxes with format
        [left, top, width, height].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple (iobs_a, iobs_b) where:
        - iobs_a: array of shape (N, M) with IoB values relative to alrbs
        - iobs_b: array of shape (N, M) with IoB values relative to blrbs

    Raises
    ------
    ValueError
        If arrays do not have shape (N, 4) and (M, 4) respectively, or contain
        negative width/height values.

    Notes
    -----
    Bounding box format is [left, top, width, height] where:
    - left: x-coordinate of top-left corner
    - top: y-coordinate of top-left corner
    - width: box width (must be non-negative)
    - height: box height (must be non-negative)

    Examples
    --------
    >>> boxes1 = np.array([[0, 0, 10, 10], [5, 5, 10, 10]])
    >>> boxes2 = np.array([[0, 0, 5, 5], [10, 10, 5, 5]])
    >>> iobs_a, iobs_b = iobs(boxes1, boxes2)
    >>> iobs_a.shape
    (2, 2)

    """
    if alrbs.ndim != 2 or alrbs.shape[1] != 4:
        raise ValueError("alrbs must have shape (N, 4) with format [left, top, width, height].")
    if blrbs.ndim != 2 or blrbs.shape[1] != 4:
        raise ValueError("blrbs must have shape (M, 4) with format [left, top, width, height].")
    if np.any(alrbs[:, 2:] < 0) or np.any(blrbs[:, 2:] < 0):
        raise ValueError("Box width and height must be non-negative.")

    num_a = alrbs.shape[0]
    num_b = blrbs.shape[0]
    iobs_a = np.zeros((num_a, num_b))
    iobs_b = np.zeros((num_a, num_b))

    for i in range(num_a):
        for j in range(num_b):
            iobs_a[i, j], iobs_b[i, j] = iob(alrbs[i, :], blrbs[j, :])

    return iobs_a, iobs_b


if __name__ == "__main__":
    # Example usage
    boxes = np.array([[0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5], [1, 1, 2, 2]])

    matrix_a, matrix_b = iobs(boxes, boxes)
    print(matrix_a)
    print(matrix_b)
