"""Clustering utilities for grouping array elements by gap threshold.

This module provides:
    - cluster_by_gap: Groups array elements into clusters based on a gap threshold.
"""

import numpy as np


def cluster_by_gap(arr: np.ndarray, gap: int) -> list[list]:
    """Group array elements into clusters based on a gap threshold.

    Parameters
    ----------
    arr : np.ndarray
        A 1D numpy array of numbers to be clustered.
    gap : int
        The maximum allowed difference between consecutive elements in a cluster.
        Must be non-negative.

    Returns
    -------
    list[list]
        A list of clusters, where each cluster is a list of numbers from the input array.

    Raises
    ------
    ValueError
        If the input array is empty, not 1-dimensional, or gap is negative.

    Examples
    --------
    >>> cluster_by_gap(np.array([1, 2, 5, 6, 10]), gap=2)
    [[1, 2], [5, 6], [10]]
    >>> cluster_by_gap(np.array([1, 2, 5, 6, 10]), gap=1)
    [[1, 2], [5, 6], [10]]
    >>> cluster_by_gap(np.array([1, 2, 5, 6, 10]), gap=4)
    [[1, 2, 5, 6, 10]]

    """
    if arr.size == 0:
        raise ValueError("Input array cannot be empty.")
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")
    if gap < 0:
        raise ValueError("Gap threshold must be non-negative.")

    arr = np.sort(arr)
    clusters = []
    current_cluster = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] <= gap:
            current_cluster.append(arr[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [arr[i]]

    clusters.append(current_cluster)
    return clusters
