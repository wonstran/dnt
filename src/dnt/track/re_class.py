"""Provide the ReClass class for re-classifying object tracks in video frames.

Use detection results, with support for configurable models, thresholds, and classes.
"""

import copy
import logging
import math
import numbers
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..detect import Detector, DetectorModel
from ..engine.iob import iobs
from ..refine.crops import FrameReader, crop_box

log = logging.getLogger(__name__)

#: Detection columns, copied at import so tests may replace ``Detector``.
_DET_FIELDS = list(Detector.DET_FIELDS)


def _read_frames(reader, frames: Iterable[int]) -> Iterator[tuple[int, np.ndarray | None]]:
    """Yield ``(frame, BGR image or None)`` for each unique frame, in increasing order.

    Each frame is requested on its own, so a frame the reader cannot decode yields ``None``
    and reading goes on: ``FrameReader`` seeks on the next request after a failure.
    """
    for f in sorted({int(v) for v in frames}):
        try:
            ((_, img),) = reader.frames([f])
        except ValueError:
            log.debug("cannot read frame %d of %s; it is left out", f, getattr(reader, "path", "?"))
            yield f, None
            continue
        yield f, img


def _detect_frames(
    detector,
    reader,
    frames: Iterable[int],
    *,
    verbose: bool = False,
    desc: str = "Detecting frames",
) -> tuple[dict[int, pd.DataFrame], set[int]]:
    """Detect each unique frame once, in increasing order.

    Returns the detections of every frame that was read, and the frames that were not.
    """
    wanted = sorted({int(v) for v in frames})
    by_frame: dict[int, pd.DataFrame] = {}
    unread: set[int] = set()
    pbar = tqdm(total=len(wanted), unit="frame", desc=desc) if verbose else None
    for f, img in _read_frames(reader, wanted):
        if img is None:
            unread.add(f)
        else:
            by_frame[f] = detector.detect_image(img, frame=f)
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()
    return by_frame, unread


def _dets_for(by_frame: dict[int, pd.DataFrame], frames: Iterable[int]) -> pd.DataFrame:
    """Return the detections of ``frames`` as one table (empty, with columns, when none)."""
    parts = [by_frame[int(f)] for f in frames if int(f) in by_frame]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=_DET_FIELDS)
    return pd.concat(parts, ignore_index=True)


#: COCO class IDs used by the vehicle re-classifier.
CAR, BUS, TRUCK = 2, 5, 7


def _size(rows: pd.DataFrame, k: int) -> float:
    """Return the ``k``-th largest ``max(w, h)`` of ``rows``, or NaN with fewer than ``k`` rows."""
    if len(rows) < k:
        return math.nan
    sides = np.maximum(rows["w"].to_numpy(float), rows["h"].to_numpy(float))
    return float(np.sort(sides)[-k])


def _spaced_sample(
    rows: pd.DataFrame, n: int, min_gap: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Pick up to ``n`` rows, largest box first, with frames at least ``min_gap`` apart.

    Ties in area go to the earlier frame. A frame is never taken twice.
    """
    if rows.empty:
        return []
    order = rows.assign(_area=rows["w"] * rows["h"], _frame=rows["frame"].astype(int))
    order = order.sort_values(["_area", "_frame"], ascending=[False, True], kind="mergesort")
    taken: list[tuple[int, tuple[float, float, float, float]]] = []
    for f, x, y, w, h in order[["_frame", "x", "y", "w", "h"]].itertuples(index=False):
        if any(abs(f - t) < min_gap or f == t for t, _ in taken):
            continue
        taken.append((int(f), (float(x), float(y), float(w), float(h))))
        if len(taken) == n:
            break
    return taken


def _iou(box: tuple, boxes) -> np.ndarray:
    """IoU of one ``(x, y, w, h)`` box with each row of ``boxes``; 0 where the union is empty."""
    x, y, w, h = (float(v) for v in box)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    iw = np.clip(np.minimum(x + w, b[:, 0] + b[:, 2]) - np.maximum(x, b[:, 0]), 0, None)
    ih = np.clip(np.minimum(y + h, b[:, 1] + b[:, 3]) - np.maximum(y, b[:, 1]), 0, None)
    inter = iw * ih
    union = w * h + b[:, 2] * b[:, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class _Vote(NamedTuple):
    """Outcome of voting over a track's sampled frames."""

    cls: int | None
    support: float
    n: int


def _vote(
    boxes: dict[int, tuple],
    by_frame: dict[int, pd.DataFrame],
    allowed: tuple[int, ...],
    iou_threshold: float,
) -> _Vote:
    """Vote for a class over the sampled frames (spec 5.2).

    Each frame's single vote goes to the allowed-class detection with the highest confidence
    among those overlapping the track box by at least ``iou_threshold``; it is weighted by that
    confidence. ``support`` is the winner's votes divided by the number of frames ``n``.
    """
    n = len(boxes)
    weight: dict[int, float] = {}
    count: dict[int, int] = {}
    for f, box in boxes.items():
        d = by_frame.get(int(f))
        if d is None or d.empty:
            continue
        d = d[d["class"].isin(allowed)]
        if d.empty:
            continue
        d = d[_iou(box, d[["x", "y", "w", "h"]].to_numpy(float)) >= iou_threshold]
        if d.empty:
            continue
        best = d.loc[d["conf"].astype(float).idxmax()]
        c = int(best["class"])
        weight[c] = weight.get(c, 0.0) + float(best["conf"])
        count[c] = count.get(c, 0) + 1
    if n == 0:
        return _Vote(None, math.nan, 0)
    if not count:
        return _Vote(None, 0.0, n)
    winner = min(count, key=lambda c: (-weight[c], -count[c], c))
    return _Vote(winner, count[winner] / n, n)


_TILE_HEIGHT = 160
_TILE_GAP = 6
_JPEG_QUALITY = 85
_ANSWER_CLASS = {"car": CAR, "bus": BUS, "truck": TRUCK}


def _tile(crops: list[np.ndarray]) -> bytes | None:
    """Scale RGB crops to one height, place them side by side, and return JPEG bytes."""
    import cv2

    parts = []
    for c in crops:
        h, w = c.shape[:2]
        width = max(1, round(w * _TILE_HEIGHT / h))
        parts.append(cv2.resize(c, (width, _TILE_HEIGHT), interpolation=cv2.INTER_AREA))
    if not parts:
        return None
    width = sum(p.shape[1] for p in parts) + _TILE_GAP * (len(parts) + 1)
    canvas = np.full((_TILE_HEIGHT + 2 * _TILE_GAP, width, 3), 128, np.uint8)
    x = _TILE_GAP
    for p in parts:
        canvas[_TILE_GAP : _TILE_GAP + _TILE_HEIGHT, x : x + p.shape[1]] = p
        x += p.shape[1] + _TILE_GAP
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _prepare_vlm(vlm, api_key: str | None, *, check: bool):
    """Validate and resolve a copy of ``vlm``; with ``check``, preflight it.

    Raises ``ValueError`` for invalid settings, a config resolving to backend ``none``, or a
    missing key; ``ImportError`` when the backend's package is missing.
    """
    from ..refine.config import vlm_problems

    problems, cfg = vlm_problems(copy.deepcopy(vlm))
    if problems:
        raise ValueError("invalid vlm config: " + "; ".join(problems))
    if cfg.backend == "none":
        raise ValueError("vlm resolves to backend 'none'; pass vlm=None to turn layer 3 off")
    if check:
        from ..refine import vlm as vlm_mod

        vlm_mod.check_vlm_dependencies(cfg)
        _check_vlm_key(cfg, api_key)
    return cfg


def _check_vlm_key(cfg, api_key: str | None) -> None:
    """Raise ``ValueError`` when the backend needs a key and none is found (no network)."""
    from ..refine.vlm import missing_key_message, resolve_api_key

    if cfg.backend == "anthropic":
        if resolve_api_key(cfg, "ANTHROPIC_API_KEY", api_key) is None:
            raise ValueError(
                missing_key_message("anthropic", cfg.api_key_env or "ANTHROPIC_API_KEY")
            )
    elif cfg.backend == "openai_compat":
        key = resolve_api_key(cfg, "OPENAI_API_KEY", api_key)
        if key is None and not (cfg.base_url or os.environ.get("OPENAI_BASE_URL")):
            raise ValueError(
                missing_key_message("openai_compat", cfg.api_key_env or "OPENAI_API_KEY")
                + "; a local server needs vlm.base_url instead"
            )


class _VLMResult(NamedTuple):
    answer: str | None
    confidence: float
    error: str | None


def _ask_vlm(cfg, backend_factory, input_video: str, requests) -> dict:
    """Ask one question per track about a tile of its crops.

    ``requests`` maps a track ID to ``(samples, heavy)``. A track with no usable crop is left
    out. The runner is built only when there is a question, and it is always closed.
    """
    from ..refine.vlm.cache import AnswerCache
    from ..refine.vlm.prompts import HEAVY_CLASS, VEHICLE_CLASS, vehicle_class_prompt
    from ..refine.vlm.runner import Question, VLMRunner

    want: dict[int, list[tuple[object, tuple]]] = {}
    for t, (samples, _) in requests.items():
        for f, box in samples:
            want.setdefault(int(f), []).append((t, box))
    crops: dict[object, list[tuple[int, np.ndarray]]] = {t: [] for t in requests}
    with FrameReader(str(input_video)) as reader:
        for f, img in _read_frames(reader, want):
            if img is None:
                continue
            for t, box in want[f]:
                c = crop_box(img, box)
                if c is not None:
                    crops[t].append((f, c))
    questions, asked = [], []
    for t, (_, heavy) in requests.items():
        image = _tile([c for _, c in sorted(crops[t], key=lambda fc: fc[0])])
        if image is None:
            log.debug("track %s has no usable crop for the VLM", t)
            continue
        options = list(HEAVY_CLASS if heavy else VEHICLE_CLASS)
        questions.append(
            Question(
                tag=f"vehclass:{_tag_id(t)}",
                image=image,
                prompt=vehicle_class_prompt(options),
                options=options,
            )
        )
        asked.append(t)  # the original ID, kept as is; the tag is only diagnostic text
    if not questions:
        return {}
    runner = VLMRunner(cfg, backend_factory(cfg), AnswerCache(cfg.cache_dir))
    try:
        verdicts = runner.ask_many(questions)
    finally:
        runner.close()
    out = {}
    for t, v in zip(asked, verdicts, strict=True):
        if v.error:
            log.warning("track %s: no VLM answer (%s); it falls back", t, v.error)
        out[t] = _VLMResult(v.answer, float(v.confidence), v.error)
    return out


def _tag_id(t) -> str:
    """Display a track ID for a question tag: an integral float such as ``1.0`` shows as ``1``."""
    if isinstance(t, numbers.Real) and not isinstance(t, bool) and float(t).is_integer():
        return str(int(t))
    return str(t)


class ReClass:
    """ReClass re-classifies object tracks in video frames using detection results.

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
        model: DetectorModel | str = DetectorModel.RTDETRx,
        weights: str | None = None,
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
        model : DetectorModel or str
            Detection model (DetectorModel or its name), default RT-DETR-x
        weights : str or None
            Optional custom weights path passed to Detector, default None
        device : str
            Device to use for detection, default 'auto'
        default_class : int
            Default class to assign if no match found, default 0 (pedestrian)
        match_class : list
            List of classes to match, default [1, 36] (bicycle, skateboard/scooter)

        """
        self.detector = Detector(model=model, weights=weights, device=device)
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

        samples = {}
        for track_id in track_ids:
            target_track = tracks[tracks["track"] == track_id].copy()
            target_track["area"] = target_track["w"] * target_track["h"]
            target_track.sort_values(by="area", inplace=True, ascending=False)
            samples[track_id] = target_track.head(self.num_frames)

        frames = {int(f) for s in samples.values() for f in s["frame"]}
        with FrameReader(str(input_video)) as reader:
            by_frame, _ = _detect_frames(
                self.detector, reader, frames, verbose=verbose, desc="Re-classifying tracks"
            )

        results = []
        for track_id in track_ids:
            top_frames = samples[track_id]
            dets = _dets_for(by_frame, top_frames["frame"])

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

        df = pd.DataFrame(results, columns=["track", "cls", "avg_score"])
        if out_file:
            df.to_csv(out_file, index=False)

        return df
