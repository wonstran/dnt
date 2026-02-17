"""Provide the ReClass class for re-classifying object tracks in video frames.

Use detection results, with support for configurable models, thresholds, and classes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..detect import Detector
from ..engine.iob import iobs


class ReClass:
    """ReClass is responsible for re-classifying object tracks in video frames using detection results.

    Attributes
    ----------
    detector : Detector
        The detection model used for re-classification.
    num_frames : int
        Number of frames to consider for re-classification.
    threshold : float
        Threshold for matching detections to tracks.
    default_class : int
        Default class to assign if no match is found.
    match_class : list
        List of classes to match during re-classification.

    Methods
    -------
    match_mmv(track: pd.DataFrame, dets: pd.DataFrame) -> tuple
        Matches a track to detections and computes the average score.
    re_classify(
        tracks: pd.DataFrame,
        input_video: str,
        track_ids: list = None,
        out_file: str = None,
        verbose: bool = True
    ) -> pd.DataFrame
        Re-classifies tracks and returns a DataFrame with results.

    """

    def __init__(
        self,
        num_frames: int = 25,
        threshold: float = 0.75,
        model: str = "rtdetr",
        weights: str = "x",
        device: str = "auto",
        default_class: int = 0,
        match_class: list | None = None,
    ) -> None:
        """Re-classify tracks based on detection results.

        Parameters
        ----------
        num_frames : int
            Number of frames to consider for re-classification, default 25
        threshold : float
            Threshold for matching, default 0.75
        model : str
            Detection model to use, default 'rtdetr'
        weights : str
            Weights for the detection model, default 'x'
        device : str
            Device to use for detection, default 'auto'
        default_class : int
            Default class to assign if no match found, default 0 (pedestrian)
        match_class : list
            List of classes to match, default [1, 36] (bicycle, skateboard/scooter)

        """
        self.detector = Detector(model=model, device=device)
        self.num_frames = num_frames
        self.threshold = threshold
        self.default_class = default_class
        self.match_class = match_class if match_class is not None else [1, 36]

    def match_mmv(self, track: pd.DataFrame, dets: pd.DataFrame) -> tuple[bool, float]:
        """Match track bboxes to detection bboxes and compute average overlap score.

        Parameters
        ----------
        track : pd.DataFrame
            DataFrame containing track data with columns [x, y, w, h, frame].
        dets : pd.DataFrame
            DataFrame containing detection data with columns [x, y, w, h, frame, class].

        Returns
        -------
        tuple[bool, float]
            A tuple (hit, avg_score) where:
            - hit : bool
                True if average overlap score meets threshold, False otherwise.
            - avg_score : float
                Average Intersection over Box (IoB) score across all matched detections.

        Notes
        -----
        Only frames present in both track and detection datasets are considered.
        The matching uses IoB metric from the engine.iob module.

        """
        if track.empty or dets.empty:
            return False, 0.0

        score = 0.0
        cnt = 0
        for _, row in track.iterrows():
            bboxes = row[["x", "y", "w", "h"]].values.reshape(1, -1)
            det = dets[dets["frame"] == row["frame"]]
            if len(det) > 0:
                match_bboxes = det[["x", "y", "w", "h"]].values
                _, overlaps_mmv = iobs(bboxes, match_bboxes)
                if overlaps_mmv.size > 0:
                    max_overlap = np.max(overlaps_mmv)
                    if max_overlap >= self.threshold:
                        score += max_overlap
                        cnt += 1

        avg_score = score / cnt if cnt > 0 else 0.0
        hit = avg_score >= self.threshold

        return hit, avg_score

    def re_classify(
        self,
        tracks: pd.DataFrame,
        input_video: str,
        track_ids: list | None = None,
        out_file: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Re-classify tracks using detection matching against reference image frame samples.

        For each track, extracts the top N largest frames (by area), runs detection on
        those frames, and matches detections against the track bboxes using IoB metric.
        Assigns the highest-scoring match class if confidence exceeds self.threshold.

        Parameters
        ----------
        tracks : pd.DataFrame
            Input tracks DataFrame with required columns: track, x, y, w, h, frame.
            Additional columns are preserved in output.
        input_video : str
            Path to source video file from which to extract frame samples.
        track_ids : list | None, optional
            List of track IDs to re-classify. If None (default), all tracks in
            the input are re-classified.
        out_file : str | None, optional
            Path to save re-classified results as CSV. If None (default), results
            are not saved to file.
        verbose : bool, optional
            If True (default), display progress bar during re-classification.

        Returns
        -------
        pd.DataFrame
            Output DataFrame with columns [track, cls, avg_score] where:
            - track : int
                Track ID from input tracks.
            - cls : int
                Re-classified class ID. Set to default_class if no match found.
            - avg_score : float
                Maximum IoB score among matched detections, rounded to 2 decimals.

        Raises
        ------
        ValueError
            If input tracks DataFrame is empty.
        FileNotFoundError
            If input_video does not exist.

        Notes
        -----
        The method considers only the top N frames (self.num_frames) by bounding
        box area for computational efficiency. It matches detections from match_class
        list against track bboxes and selects the class with highest average score.

        Examples
        --------
        >>> import pandas as pd
        >>> from .re_class import ReClass
        >>> tracks = pd.DataFrame({
        ...     'frame': [0, 1, 2],
        ...     'track': [1, 1, 1],
        ...     'x': [100, 102, 104],
        ...     'y': [50, 52, 54],
        ...     'w': [50, 50, 50],
        ...     'h': [100, 100, 100],
        ... })
        >>> rc = ReClass(num_frames=2, threshold=0.75, match_class=[1, 36])
        >>> result = rc.re_classify(tracks, 'video.mp4')
        >>> print(result)  # DataFrame with [track, cls, avg_score]

        """
        if tracks.empty:
            raise ValueError("Input tracks DataFrame is empty.")
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Video file not found: {input_video}")

        if track_ids is None:
            track_ids = tracks["track"].unique().tolist()

        results = []
        if verbose:
            pbar = tqdm(total=len(track_ids), unit="track", desc="Re-classifying tracks")
        for track_id in track_ids:
            target_track = tracks[tracks["track"] == track_id].copy()
            target_track["area"] = target_track["w"] * target_track["h"]
            target_track.sort_values(by="area", inplace=True, ascending=False)

            top_frames = target_track.head(self.num_frames) if len(target_track) >= self.num_frames else target_track

            dets = self.detector.detect_frames(input_video, top_frames["frame"].values.tolist())

            matched = []
            for cls in self.match_class:
                match_dets = dets[dets["class"] == cls]
                hit, avg_score = self.match_mmv(top_frames, match_dets)
                if hit:
                    matched.append((cls, avg_score))

            if len(matched) > 0:
                cls, avg_score = max(matched, key=lambda x: x[1])
            else:
                cls = self.default_class
                avg_score = 0

            results.append([track_id, cls, round(avg_score, 2)])
            if verbose:
                pbar.update()
        if verbose:
            pbar.close()

        df = pd.DataFrame(results, columns=["track", "cls", "avg_score"])
        if out_file:
            df.to_csv(out_file, index=False)

        return df
