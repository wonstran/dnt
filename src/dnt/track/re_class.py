"""Provide ReClass and VehReClass, which re-classify object tracks from video frames.

``ReClass`` matches a track's boxes against detections of chosen classes (for example
bicycles and scooters) and assigns the best-matching class. ``VehReClass`` classifies vehicle
tracks as car, bus or truck with a cascade of a size rule, re-detection voting, an optional
vision-language model, and a fallback.
"""

import copy
import logging
import math
import numbers
import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..detect import Detector, DetectorModel
from ..engine.iob import iobs
from ..refine.crops import FrameReader, crop_box
from ..refine.io import filled_mask
from ..shared.files import read_track, write_track
from .tracker import Tracker

if TYPE_CHECKING:
    from ..refine.config import VLMConfig

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
    if unread:
        log.warning(
            "%d of %d sampled frames of %s could not be read; tracks that sampled them "
            "have less evidence",
            len(unread),
            len(wanted),
            getattr(reader, "path", "?"),
        )
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


def _fit_height(crop: np.ndarray) -> np.ndarray:
    """Scale a crop to the tile height, keeping its aspect ratio; a fitted crop is kept."""
    import cv2

    h, w = crop.shape[:2]
    if h == _TILE_HEIGHT:
        return crop
    width = max(1, round(w * _TILE_HEIGHT / h))
    return cv2.resize(crop, (width, _TILE_HEIGHT), interpolation=cv2.INTER_AREA)


def _tile(crops: list[np.ndarray]) -> bytes | None:
    """Scale RGB crops to one height, place them side by side, and return JPEG bytes."""
    import cv2

    parts = [_fit_height(c) for c in crops]
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


def _missing_key_message(backend: str, env: str) -> str:
    """Return the error for a backend with no API key; it names the ways VehReClass takes one."""
    return (
        f"VehReClass: vlm.backend {backend} needs an API key and none was found. Pass it as "
        f"VehReClass(..., vlm_api_key=...), set vlm.api_key_file to a file that holds it, "
        f"or set the {env} environment variable (vlm.api_key_env names another variable)"
    )


def _check_vlm_key(cfg, api_key: str | None) -> None:
    """Raise ``ValueError`` when the backend needs a key and none is found (no network)."""
    from ..refine.vlm import resolve_api_key

    if cfg.backend == "anthropic":
        if resolve_api_key(cfg, "ANTHROPIC_API_KEY", api_key) is None:
            raise ValueError(
                _missing_key_message("anthropic", cfg.api_key_env or "ANTHROPIC_API_KEY")
            )
    elif cfg.backend == "openai_compat":
        key = resolve_api_key(cfg, "OPENAI_API_KEY", api_key)
        if key is None and not (cfg.base_url or os.environ.get("OPENAI_BASE_URL")):
            raise ValueError(
                _missing_key_message("openai_compat", cfg.api_key_env or "OPENAI_API_KEY")
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
    unread = 0
    with FrameReader(str(input_video)) as reader:
        for f, img in _read_frames(reader, want):
            if img is None:
                unread += 1
                continue
            for t, box in want[f]:
                c = crop_box(img, box)
                if c is not None:
                    crops[t].append((f, _fit_height(c)))  # keep only tile-sized crops in memory
    if unread:
        log.warning(
            "%d of %d crop frames of %s could not be read; tracks that sampled them have "
            "fewer crops for the VLM",
            unread,
            len(want),
            input_video,
        )
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
    cache = AnswerCache(cfg.cache_dir)  # built first: a failure here must not leak a backend
    runner = VLMRunner(cfg, backend_factory(cfg), cache)
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
                Mean IoB over the frames whose best overlap reached the threshold,
                rounded to 2 decimals (0 when no class matched).

        Raises
        ------
        ValueError
            If input tracks DataFrame is empty, or the video cannot be opened by OpenCV.
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
        >>> from dnt.track import ReClass
        >>> tracks = pd.DataFrame({
        ...     'frame': [0, 1, 2],
        ...     'track': [1, 1, 1],
        ...     'x': [100, 102, 104],
        ...     'y': [50, 52, 54],
        ...     'w': [50, 50, 50],
        ...     'h': [100, 100, 100],
        ... })
        >>> rc = ReClass(num_frames=2, threshold=0.75, match_class=[1, 36])
        >>> result = rc.re_classify(tracks, 'video.mp4')  # doctest: +SKIP
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


_DECISION_COLUMNS = [
    "track",
    "cls",
    "layer",
    "heavy",
    "size",
    "n",
    "support",
    "redetect_cls",
    "vlm_answer",
    "vlm_conf",
]
_TRACK_COLUMNS = ("frame", "track", "x", "y", "w", "h", "cls")


def _setting_problems(
    hv_size, hv_min_count, num_frames, min_gap, iou_threshold, min_support, vlm_crops
) -> list[str]:
    def is_int(v):
        return isinstance(v, numbers.Integral) and not isinstance(v, bool)

    def is_num(v):
        return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)

    p = []
    if hv_size is not None and not (is_num(hv_size) and hv_size > 0):
        p.append("hv_size must be None or a finite number > 0")
    for name, v in (
        ("hv_min_count", hv_min_count),
        ("num_frames", num_frames),
        ("vlm_crops", vlm_crops),
    ):
        if not (is_int(v) and v >= 1):
            p.append(f"{name} must be an integer >= 1")
    if not (is_int(min_gap) and min_gap >= 0):
        p.append("min_gap must be an integer >= 0")
    for name, v in (("iou_threshold", iou_threshold), ("min_support", min_support)):
        if not (is_num(v) and 0 < v <= 1):
            p.append(f"{name} must be a number in (0, 1]")
    return p


def _read_tracks(tracks) -> pd.DataFrame:
    """Return a copy of a track table, or read a headerless track file into one."""
    if isinstance(tracks, pd.DataFrame):
        if tracks.empty:
            return tracks.copy()
        missing = [c for c in _TRACK_COLUMNS if c not in tracks.columns]
        if missing:
            raise ValueError(f"tracks lacks columns {missing}")
        return tracks.copy()
    try:
        df = read_track(tracks)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=Tracker.TRACK_FIELDS)
    if df.shape[1] != len(Tracker.TRACK_FIELDS):
        raise ValueError(
            f"{tracks}: expected {len(Tracker.TRACK_FIELDS)} columns, got {df.shape[1]}"
        )
    df.columns = Tracker.TRACK_FIELDS
    return df


class VehReClass:
    """Classify vehicle tracks as car, bus or truck with a size, re-detection and VLM cascade.

    Each track goes through the layers in order and stops at the first that decides:

    1. size: at least ``hv_min_count`` real rows with ``max(w, h) >= hv_size`` flag the track
       heavy, which allows only bus and truck from then on;
    2. re-detection: spaced, large frames are detected once, and each frame votes for the
       most confident allowed class overlapping the track box; a car vote decides, but a bus
       or truck vote decides only for a heavy track, because COCO's truck (and bus) also covers
       vans and pickups, which count as car here;
    3. VLM: a tile of crops is shown to the configured model, whose glossary defines car as
       any light vehicle (vans and pickups included) and truck as a heavy goods vehicle;
    4. fallback: truck for a heavy track, car otherwise.

    Parameters
    ----------
    hv_size : float or None
        Heavy-vehicle size in pixels; None turns layer 1 off. Default None.
    hv_min_count : int
        Real rows that must reach ``hv_size``. Default 3.
    num_frames : int
        Frames sampled per track for re-detection. Default 10.
    min_gap : int
        Minimum distance in frames between samples (layers 2 and 3). Default 5.
    iou_threshold : float
        IoU a re-detection needs with the track box, in (0, 1]. Default 0.5.
    min_support : float
        Share of sampled frames the winning class needs, in (0, 1]. Default 0.3.
    vlm : dnt.refine.config.VLMConfig or None
        Layer 3 settings; None turns layer 3 off. Its ``min_conf`` is the answer threshold.
    vlm_crops : int
        Crops tiled into the one image per track. Default 3.
    model : DetectorModel or str
        Re-detection model. Default RT-DETR-x.
    weights : str or None
        Custom weights passed to ``Detector``.
    device : str
        Detector device. Default ``"auto"``.
    vlm_api_key : str or None
        API key; takes precedence over ``vlm.api_key_file`` and the environment. Not allowed
        together with ``vlm_backend_factory``, which builds the backend itself.
    vlm_backend_factory : callable or None
        Builds the backend from the resolved ``VLMConfig`` (tests); skips the preflight.

    Raises
    ------
    ValueError
        For a broken setting, an invalid VLM config, one resolving to backend ``none``, a
        missing API key, or ``vlm_api_key`` given together with ``vlm_backend_factory``.
    ImportError
        When the VLM backend's package is not installed.

    """

    CAR, BUS, TRUCK = CAR, BUS, TRUCK

    def __init__(
        self,
        hv_size: float | None = None,
        hv_min_count: int = 3,
        num_frames: int = 10,
        min_gap: int = 5,
        iou_threshold: float = 0.5,
        min_support: float = 0.3,
        vlm: "VLMConfig | None" = None,
        vlm_crops: int = 3,
        model: DetectorModel | str = DetectorModel.RTDETRx,
        weights: str | None = None,
        device: str = "auto",
        *,
        vlm_api_key: str | None = None,
        vlm_backend_factory: Callable | None = None,
    ) -> None:
        """Validate the settings, prepare the VLM, then build the detector."""
        if vlm_api_key is not None and vlm_backend_factory is not None:
            raise ValueError(
                "vlm_api_key is ignored when vlm_backend_factory builds the backend; pass one"
            )
        problems = _setting_problems(
            hv_size, hv_min_count, num_frames, min_gap, iou_threshold, min_support, vlm_crops
        )
        if problems:
            raise ValueError("invalid VehReClass settings: " + "; ".join(problems))
        self.hv_size = hv_size
        self.hv_min_count = hv_min_count
        self.num_frames = num_frames
        self.min_gap = min_gap
        self.iou_threshold = iou_threshold
        self.min_support = min_support
        self.vlm_crops = vlm_crops
        self._vlm_api_key = vlm_api_key
        self._vlm_backend_factory = vlm_backend_factory
        self.vlm = (
            None
            if vlm is None
            else _prepare_vlm(vlm, vlm_api_key, check=vlm_backend_factory is None)
        )
        self.detector = Detector(model=model, weights=weights, device=device)
        self.decisions = pd.DataFrame(columns=_DECISION_COLUMNS)

    def _backend_factory(self):
        if self._vlm_backend_factory is not None:
            return self._vlm_backend_factory
        from ..refine.vlm import make_backend

        return lambda cfg: make_backend(cfg, api_key=self._vlm_api_key)

    @staticmethod
    def _select(table: pd.DataFrame, track_ids) -> list:
        present = pd.unique(table["track"]).tolist()
        if track_ids is None:
            return present
        known = set(present)
        wanted = list(track_ids)
        ids = list(dict.fromkeys(t for t in wanted if t in known))
        unknown = sum(1 for t in wanted if t not in known)
        if unknown:
            log.warning("%d track ID(s) in track_ids are not in the tracks; ignored", unknown)
        return ids

    def re_classify(
        self,
        tracks: pd.DataFrame | str,
        input_video: str,
        track_ids: list | None = None,
        out_file: str | None = None,
        verbose: bool = True,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Classify vehicle tracks and write the class into ``cls``.

        Parameters
        ----------
        tracks : pandas.DataFrame or str
            A track table (at least ``frame, track, x, y, w, h, cls``) or a headerless track
            file. Filled rows (``r3 == 1`` or ``interp == 1``) are never used as evidence.
        input_video : str
            The source video.
        track_ids : list or None
            Tracks to classify; None classifies all. Unknown IDs are ignored with a warning.
        out_file : str or None
            Where to write the result, headerless, in the input's column layout.
        verbose : bool
            Show a progress bar while detecting.
        video_index : int, optional
            Index of this video in a batch, used only for progress display.
        video_tot : int, optional
            Total number of videos in the batch, used only for progress display.

        Returns
        -------
        pandas.DataFrame
            The input table with ``cls`` set on every row of each classified track; the
            per-track reasons are in ``self.decisions``.

        Raises
        ------
        FileNotFoundError
            If ``input_video`` does not exist (checked first, even for empty input).
        ValueError
            If the video cannot be opened or the track input is malformed.

        """
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Video file not found: {input_video}")
        table = _read_tracks(tracks)
        self.decisions = pd.DataFrame(columns=_DECISION_COLUMNS)
        if table.empty:
            if out_file:
                write_track(table, out_file)
            return table

        ids = self._select(table, track_ids)
        real = table.loc[~filled_mask(table)]
        groups = dict(tuple(real.groupby("track", sort=False)))
        none = real.iloc[0:0]
        rows: dict = {}
        samples: dict = {}
        for t in ids:
            g = groups.get(t, none)
            size = _size(g, self.hv_min_count)
            heavy = self.hv_size is not None and size >= self.hv_size
            rows[t] = {
                "track": t,
                "cls": None,
                "layer": None,
                "heavy": bool(heavy),
                "size": size,
                "n": 0,
                "support": math.nan,
                "redetect_cls": None,
                "vlm_answer": None,
                "vlm_conf": math.nan,
            }
            samples[t] = _spaced_sample(g, self.num_frames, self.min_gap)

        frames = {f for s in samples.values() for f, _ in s}
        desc = "Re-classifying vehicles"
        if video_index is not None and video_tot is not None:
            desc += f" {video_index} of {video_tot}"
        with FrameReader(str(input_video)) as reader:
            by_frame, unread = _detect_frames(
                self.detector, reader, frames, verbose=verbose, desc=desc
            )
        for t in ids:
            r = rows[t]
            allowed = (BUS, TRUCK) if r["heavy"] else (CAR, BUS, TRUCK)
            boxes = {f: b for f, b in samples[t] if f not in unread}
            v = _vote(boxes, by_frame, allowed, self.iou_threshold)
            r["n"], r["support"], r["redetect_cls"] = v.n, v.support, v.cls
            # COCO's "truck" (and "bus") also covers vans and pickups, which count as car here,
            # so a bus or truck vote decides only when size has flagged the track heavy;
            # otherwise the track goes on to the VLM (or falls back to car)
            confirmed = v.cls == CAR or r["heavy"]
            if v.cls is not None and v.support >= self.min_support and confirmed:
                r["cls"], r["layer"] = v.cls, "redetect"

        pending = {
            t: (_spaced_sample(groups[t], self.vlm_crops, self.min_gap), rows[t]["heavy"])
            for t in ids
            if rows[t]["layer"] is None and t in groups
        }
        if self.vlm is not None and pending:
            answers = _ask_vlm(self.vlm, self._backend_factory(), str(input_video), pending)
            for t, a in answers.items():
                r = rows[t]
                if a.error:  # no answer was given; the warning already names the error
                    r["vlm_answer"], r["vlm_conf"] = None, math.nan
                else:
                    r["vlm_answer"], r["vlm_conf"] = a.answer, a.confidence
                ok = (
                    a.error is None
                    and a.answer in _ANSWER_CLASS
                    and math.isfinite(a.confidence)
                    and a.confidence >= self.vlm.min_conf
                )
                if ok:
                    r["cls"], r["layer"] = _ANSWER_CLASS[a.answer], "vlm"

        for r in rows.values():
            if r["layer"] is None:
                r["cls"], r["layer"] = (TRUCK if r["heavy"] else CAR), "fallback"

        out = table.copy()
        cls_map = {t: r["cls"] for t, r in rows.items()}
        hit = out["track"].isin(list(cls_map))
        out.loc[hit, "cls"] = out.loc[hit, "track"].map(cls_map).astype(int)
        if out_file:
            write_track(out, out_file)
        decisions = pd.DataFrame(list(rows.values()), columns=_DECISION_COLUMNS)
        decisions["redetect_cls"] = decisions["redetect_cls"].astype("Int64")  # <NA>: no votes
        self.decisions = decisions
        return out
