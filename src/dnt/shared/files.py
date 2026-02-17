"""File I/O utilities for reading and writing IOU and tracking data.

This module provides functions to read and write CSV files containing
IOU (Intersection over Union) metrics and tracking information.
"""

import pandas as pd


def read_iou(iou_file):
    """Read IOU metrics from a CSV file.

    Parameters
    ----------
    iou_file : str
        Path to the CSV file containing IOU metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame containing IOU metrics with columns for video ID, detection ID,
        and IOU values.

    """
    results = pd.read_csv(
        iou_file, header=None, dtype={0: int, 1: int, 2: float, 3: float, 4: float, 5: float, 6: float, 7: int}
    )
    return results


def write_iou(ious, iou_file):
    """Write IOU metrics to a CSV file.

    Parameters
    ----------
    ious : pd.DataFrame
        DataFrame containing IOU metrics to write.
    iou_file : str
        Path to the output CSV file.

    """
    ious.to_csv(iou_file, header=False, index=False)


def read_track(track_file):
    """Read tracking data from a CSV file.

    Parameters
    ----------
    track_file : str
        Path to the CSV file containing tracking information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing tracking data with video ID, track ID, coordinates,
        and additional tracking metrics.

    """
    results = pd.read_csv(
        track_file,
        header=None,
        dtype={0: int, 1: int, 2: float, 3: float, 4: float, 5: float, 6: float, 7: int, 8: int, 9: int},
    )
    return results


def write_track(tracks, track_file):
    """Write tracking data to a CSV file.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame containing tracking data to write.
    track_file : str
        Path to the output CSV file.

    """
    tracks.to_csv(track_file, header=False, index=False)


def read_spd(spd_file):
    """Read speed data from a CSV file.

    Parameters
    ----------
    spd_file : str
        Path to the CSV file containing speed information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing speed data with columns for video ID, reference, and frame.

    """
    results = pd.read_csv(spd_file, header=None, dtype={0: int, 1: int, 2: int})
    return results
