"""Provide ReClass and VehReClass, which re-classify object tracks from video frames.

``ReClass`` matches a person track's boxes against detections of bicycles, motorcycles and
skateboards or scooters, optionally asks a vision-language model about ambiguous tracks, and
assigns the rider class. ``VehReClass`` classifies vehicle
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


def _prepare_vlm(vlm, api_key: str | None, *, check: bool, owner: str = "VehReClass"):
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
        _check_vlm_key(cfg, api_key, owner)
    return cfg


def _missing_key_message(backend: str, env: str, owner: str = "VehReClass") -> str:
    """Return the error for a backend with no API key; it names the ways ``owner`` takes one."""
    return (
        f"{owner}: vlm.backend {backend} needs an API key and none was found. Pass it as "
        f"{owner}(..., vlm_api_key=...), set vlm.api_key_file to a file that holds it, "
        f"or set the {env} environment variable (vlm.api_key_env names another variable)"
    )


def _check_vlm_key(cfg, api_key: str | None, owner: str = "VehReClass") -> None:
    """Raise ``ValueError`` when the backend needs a key and none is found (no network)."""
    from ..refine.vlm import resolve_api_key

    if cfg.backend == "anthropic":
        if resolve_api_key(cfg, "ANTHROPIC_API_KEY", api_key) is None:
            raise ValueError(
                _missing_key_message("anthropic", cfg.api_key_env or "ANTHROPIC_API_KEY", owner)
            )
    elif cfg.backend == "openai_compat":
        key = resolve_api_key(cfg, "OPENAI_API_KEY", api_key)
        if key is None and not (cfg.base_url or os.environ.get("OPENAI_BASE_URL")):
            raise ValueError(
                _missing_key_message("openai_compat", cfg.api_key_env or "OPENAI_API_KEY", owner)
                + "; a local server needs vlm.base_url instead"
            )


class _VLMResult(NamedTuple):
    answer: str | None
    confidence: float
    error: str | None


def _ask_vlm(
    cfg, backend_factory, input_video: str, requests, *, kind: str = "vehclass", options=None
) -> dict:
    """Ask one question per track about a tile of its crops.

    ``requests`` maps a track ID to its crop samples ``[(frame, box), ...]``. By default every
    question is the vehicle question with the options car, bus, truck and unclear; with
    ``kind="riderclass"`` it is the rider question, whose ``options`` the caller gives. A track
    with no usable crop is left out. The runner is built only when there is a question, and it
    is always closed.
    """
    from ..refine.vlm.cache import AnswerCache
    from ..refine.vlm.prompts import (
        RIDER_CLASS,
        VEHICLE_CLASS,
        rider_class_prompt,
        vehicle_class_prompt,
    )
    from ..refine.vlm.runner import Question, VLMRunner

    rider = kind == "riderclass"
    ask_options = list(options or (RIDER_CLASS if rider else VEHICLE_CLASS))
    prompt = (rider_class_prompt if rider else vehicle_class_prompt)(ask_options)

    want: dict[int, list[tuple[object, tuple]]] = {}
    for t, samples in requests.items():
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
    for t in requests:
        image = _tile([c for _, c in sorted(crops[t], key=lambda fc: fc[0])])
        if image is None:
            log.debug("track %s has no usable crop for the VLM", t)
            continue
        questions.append(
            Question(
                tag=f"{kind}:{_tag_id(t)}",
                image=image,
                prompt=prompt,
                options=list(ask_options),
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


#: Rider answers and the COCO class each stands for (scooter shares skateboard's class).
_RIDER_ANSWER = {"cyclist": 1, "motorcycle_rider": 3, "scooter_rider": 36}


def _tag_id(t) -> str:
    """Display a track ID for a question tag: an integral float such as ``1.0`` shows as ``1``."""
    if isinstance(t, numbers.Real) and not isinstance(t, bool) and float(t).is_integer():
        return str(int(t))
    return str(t)


class ReClass:
    """ReClass re-classifies person tracks as riders using detection results and a VLM.

    A track's largest boxes are matched against detections of ``match_class`` (bicycle,
    motorcycle, skateboard/scooter) by overlap. A class that overlaps on at least
    ``min_support`` of the sampled frames, and is the only one to overlap, is accepted. With a
    ``vlm``, any other overlap (weak, or several classes at once) is put to the model, which
    answers pedestrian, cyclist, motorcycle_rider, scooter_rider or unclear; a track with no
    overlap at all stays ``default_class`` and is never sent to it. Without a ``vlm``, the
    class with the best overlap wins whatever its support.

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
    min_support : float
        Share of the sampled frames a class must overlap on to be accepted without the VLM.
    vlm : dnt.refine.config.VLMConfig or None
        VLM settings, resolved; None turns the VLM off. Its ``min_conf`` is the answer threshold.
    decisions : pd.DataFrame
        Per-track reasons of the last ``re_classify`` call: ``track, cls, layer, support,
        vlm_answer, vlm_conf`` with ``layer`` one of ``overlap``, ``vlm`` and ``default``.

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
        *,
        min_support: float = 0.5,
        vlm: "VLMConfig | None" = None,
        vlm_crops: int = 3,
        vlm_api_key: str | None = None,
        vlm_backend_factory: Callable | None = None,
        rider_map: dict[str, int] | None = None,
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
            List of classes to match, default [1, 3, 36] (bicycle, motorcycle,
            skateboard/scooter)
        min_support : float
            Share of the sampled frames a class must overlap on to be accepted without the
            VLM, in (0, 1], default 0.5. It only matters when ``vlm`` is given.
        vlm : dnt.refine.config.VLMConfig or None
            Settings of the model that decides ambiguous tracks; None (default) turns it off.
        vlm_crops : int
            Crops tiled into the one image per track, default 3
        vlm_api_key : str or None
            API key; takes precedence over ``vlm.api_key_file`` and the environment. Not allowed
            together with ``vlm_backend_factory``.
        vlm_backend_factory : callable or None
            Builds the backend from the resolved ``VLMConfig`` (tests); skips the preflight.
        rider_map : dict or None
            The class each VLM rider answer (``cyclist``, ``motorcycle_rider``,
            ``scooter_rider``) stands for; default ``{"cyclist": 1, "motorcycle_rider": 3,
            "scooter_rider": 36}``. An answer left out is not offered to the model. An answer
            mapped to ``default_class`` leaves the track a pedestrian, which suits a pipeline
            where another stage owns bicycles and motorcycles, e.g.
            ``{"cyclist": 0, "motorcycle_rider": 0, "scooter_rider": 36}``. An answer mapped to
            another class outside ``match_class`` is not offered.

        Raises
        ------
        ValueError
            For an invalid ``min_support``, ``vlm_crops`` or ``rider_map``, an invalid VLM
            config, one resolving to backend ``none``, a missing API key, or ``vlm_api_key``
            given together with ``vlm_backend_factory``.
        ImportError
            When the VLM backend's package is not installed.

        """
        if vlm_api_key is not None and vlm_backend_factory is not None:
            raise ValueError(
                "vlm_api_key is ignored when vlm_backend_factory builds the backend; pass one"
            )
        if not (isinstance(min_support, numbers.Real) and 0 < min_support <= 1):
            raise ValueError(f"min_support must be in (0, 1], not {min_support!r}")
        if not (isinstance(vlm_crops, numbers.Integral) and vlm_crops >= 1):
            raise ValueError(f"vlm_crops must be an integer >= 1, not {vlm_crops!r}")
        self.num_frames = num_frames
        self.threshold = threshold
        self.default_class = default_class
        self.match_class = match_class if match_class is not None else [1, 3, 36]
        self.min_support = min_support
        rider_map = dict(_RIDER_ANSWER if rider_map is None else rider_map)
        bad = [a for a in rider_map if a not in _RIDER_ANSWER]
        if bad:
            raise ValueError(f"rider_map keys must be among {list(_RIDER_ANSWER)}, not {bad}")
        self.rider_map = rider_map
        self.vlm_crops = vlm_crops
        self._vlm_api_key = vlm_api_key
        self._vlm_backend_factory = vlm_backend_factory
        self.vlm = (
            None
            if vlm is None
            else _prepare_vlm(vlm, vlm_api_key, check=vlm_backend_factory is None, owner="ReClass")
        )
        self.detector = Detector(model=model, weights=weights, device=device)
        self.decisions = pd.DataFrame(columns=_RIDER_DECISION_COLUMNS)

    def _backend_factory(self):
        if self._vlm_backend_factory is not None:
            return self._vlm_backend_factory
        from ..refine.vlm import make_backend

        return lambda cfg: make_backend(cfg, api_key=self._vlm_api_key)

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
        _, avg_score = self._overlap(track, dets)
        return avg_score >= self.threshold, avg_score

    def _overlap(self, track: pd.DataFrame, dets: pd.DataFrame) -> tuple[int, float]:
        """Return the frames whose best IoB reaches the threshold, and their mean IoB."""
        if track.empty or dets.empty:
            return 0, 0.0

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

        return cnt, (score / cnt if cnt > 0 else 0.0)

    def _ask(self, rows: dict, pending: list, tracks: pd.DataFrame, input_video: str) -> None:
        """Put the ambiguous tracks to the VLM and apply the answers it is sure of."""
        by_answer = {
            a: c
            for a, c in self.rider_map.items()
            if c in self.match_class or c == self.default_class
        }
        options = ["pedestrian", *by_answer, "unclear"]
        requests = {
            t: _spaced_sample(tracks[tracks["track"] == t], self.vlm_crops, 0) for t in pending
        }
        answers = _ask_vlm(
            self.vlm, self._backend_factory(), input_video, requests,
            kind="riderclass", options=options,
        )
        for t, a in answers.items():
            r = rows[t]
            if a.error:  # no answer was given; the warning already names the error
                continue
            r["vlm_answer"], r["vlm_conf"] = a.answer, a.confidence
            if not (math.isfinite(a.confidence) and a.confidence >= self.vlm.min_conf):
                continue
            if a.answer == "pedestrian" or by_answer.get(a.answer) == self.default_class:
                r.update(cls=self.default_class, avg_score=0.0, layer="vlm")
            elif a.answer in by_answer:
                cls = by_answer[a.answer]
                r.update(
                    cls=cls, layer="vlm", avg_score=max(r["overlap"].get(cls, 0.0), a.confidence)
                )

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

        rows: dict = {}
        pending: list = []
        for track_id in track_ids:
            top_frames = samples[track_id]
            dets = _dets_for(by_frame, top_frames["frame"])

            matched = []
            for cls in self.match_class:
                count, avg_score = self._overlap(top_frames, dets[dets["class"] == cls])
                if count and avg_score >= self.threshold:
                    matched.append((cls, avg_score, count / len(top_frames)))

            r = {
                "track": track_id,
                "cls": self.default_class,
                "avg_score": 0.0,
                "layer": "default",
                "support": 0.0,
                "vlm_answer": None,
                "vlm_conf": math.nan,
            }
            if matched:
                cls, avg_score, support = max(matched, key=lambda x: x[1])
                r.update(cls=cls, avg_score=avg_score, layer="overlap", support=support)
                if self.vlm is not None and (len(matched) > 1 or support < self.min_support):
                    pending.append(track_id)
            r["overlap"] = {cls: avg for cls, avg, _ in matched}
            rows[track_id] = r

        if pending:
            self._ask(rows, pending, tracks, str(input_video))

        results = [[r["track"], r["cls"], round(r["avg_score"], 2)] for r in rows.values()]
        self.decisions = pd.DataFrame(list(rows.values()), columns=_RIDER_DECISION_COLUMNS)

        df = pd.DataFrame(results, columns=["track", "cls", "avg_score"])
        if out_file:
            df.to_csv(out_file, index=False)

        return df


_RIDER_DECISION_COLUMNS = ["track", "cls", "layer", "support", "vlm_answer", "vlm_conf"]
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
       heavy, which makes a bus or truck vote from re-detection trustworthy (a big box is
       evidence of a big vehicle) and a car vote suspect (a big box near the camera is also
       what an SUV or pickup looks like);
    2. re-detection: spaced, large frames are detected once, and each frame votes for the
       most confident class (car, bus or truck) overlapping the track box. On a track that is
       not heavy, a car vote decides and a bus or truck vote does not, because COCO's truck
       (and bus) also covers vans and pickups, which count as car here. On a heavy track it
       is the other way round;
    3. VLM: a tile of crops is shown to the configured model with the options car, bus, truck
       and unclear, whose glossary defines car as any light vehicle (vans and pickups
       included) and truck as a heavy goods vehicle;
    4. fallback: car for a track whose re-detection leans car, otherwise truck for a heavy
       track and car for the rest.

    Parameters
    ----------
    hv_size : float or None
        Heavy-vehicle size in pixels; None turns layer 1 off. Default None. Set it above the
        largest SUV or pickup the camera sees (see ``tools/hv_size_crops.py``): a vehicle
        close to the camera is large in pixels whatever it is.
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
            boxes = {f: b for f, b in samples[t] if f not in unread}
            v = _vote(boxes, by_frame, (CAR, BUS, TRUCK), self.iou_threshold)
            r["n"], r["support"], r["redetect_cls"] = v.n, v.support, v.cls
            strong = v.cls is not None and v.support >= self.min_support
            # Size and the vote must agree. COCO's "truck" (and "bus") also covers vans and
            # pickups, which count as car here, so a bus or truck vote decides only on a heavy
            # track; and a big box close to the camera is also what an SUV or pickup looks
            # like, so a car vote decides only on a track that is not heavy. A disagreement goes
            # on to the VLM, which sees the vehicle.
            if strong and (v.cls == CAR) != r["heavy"]:
                r["cls"], r["layer"] = v.cls, "redetect"
            r["leans_car"] = strong and v.cls == CAR

        pending = {
            t: _spaced_sample(groups[t], self.vlm_crops, self.min_gap)
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
                # a heavy track the detector calls car stays car: size alone does not overrule
                # it; a heavy track with no usable vote is a truck, the likelier heavy class
                heavy_truck = r["heavy"] and not r["leans_car"]
                r["cls"], r["layer"] = (TRUCK if heavy_truck else CAR), "fallback"

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


_TWOWHEEL_COLUMNS = [
    "track",
    "cls",
    "layer",
    "n",
    "mc_ratio",
    "bike_ratio",
    "vlm_answer",
    "vlm_conf",
]
#: COCO class IDs a two-wheeler track can take (the skateboard class stands for a scooter).
BICYCLE, MOTORCYCLE, SKATEBOARD = 1, 3, 36


def _read_dets(dets) -> pd.DataFrame:
    """Return detections as a table with ``frame, x, y, w, h``; None or an empty file is empty."""
    cols = ["frame", "x", "y", "w", "h"]
    if dets is None:
        return pd.DataFrame(columns=cols)
    if not isinstance(dets, pd.DataFrame):
        try:
            dets = pd.read_csv(dets, header=None, names=_DET_FIELDS)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=cols)
    missing = [c for c in cols if c not in dets.columns]
    if missing:
        raise ValueError(f"detections lack columns {missing}")
    return dets[cols]


def _iob_frames(rows: pd.DataFrame, dets: pd.DataFrame, threshold: float) -> int:
    """Count the track rows whose box has an IoB with some detection of its frame >= threshold.

    The IoB is the intersection over the *track* box, so a detection inside the track box
    counts as fully overlapping.
    """
    if rows.empty or dets.empty:
        return 0
    by_frame = {int(f): g[["x", "y", "w", "h"]].to_numpy(float) for f, g in dets.groupby("frame")}
    hits = 0
    for f, x, y, w, h in rows[["frame", "x", "y", "w", "h"]].itertuples(index=False):
        b = by_frame.get(int(f))
        if b is None or w <= 0 or h <= 0:
            continue
        iw = np.clip(np.minimum(x + w, b[:, 0] + b[:, 2]) - np.maximum(x, b[:, 0]), 0, None)
        ih = np.clip(np.minimum(y + h, b[:, 1] + b[:, 3]) - np.maximum(y, b[:, 1]), 0, None)
        if float((iw * ih).max()) / (w * h) >= threshold:
            hits += 1
    return hits


def _even_sample(rows: pd.DataFrame, n: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Pick up to ``n`` rows spread evenly over the track, as ``(frame, box)``."""
    if rows.empty:
        return []
    rows = rows.sort_values("frame")
    idx = np.unique(np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int))
    picked = rows.iloc[idx]
    return [
        (int(f), (float(x), float(y), float(w), float(h)))
        for f, x, y, w, h in picked[["frame", "x", "y", "w", "h"]].itertuples(index=False)
    ]


class TwoWheelReClass:
    """Classify two-wheeler tracks as bicycle (1) or motorcycle (3), optionally dropping scooters.

    The input tracks are already known to be two-wheelers (for example a refined bicycle and
    motorcycle track file); this decides which kind each is. It re-detects nothing: it matches
    each track's boxes against detections of motorcycles and bicycles that are already
    computed, and against the class the track already carries. A track stops at the first
    layer that decides:

    1. scooter: a track whose class is mostly skateboard/scooter (36) is a stand-up scooter,
       not a bicycle or motorcycle; it is dropped, or with ``scooter="keep"`` kept as 36;
    2. prior: a clear majority decides. A class wins when the track overlaps its detections
       on at least ``min_ratio`` of the frames and ``dominance`` times as many as the other
       class, or when it carries that class on at least ``vote_share`` of its rows and none
       of the other;
    3. VLM: a tile of crops is shown to the configured model, which answers pedestrian,
       cyclist, motorcycle_rider, scooter_rider or unclear. Cyclist is a bicycle and
       motorcycle_rider a motorcycle. Pedestrian and scooter_rider are not two-wheelers and
       drop the track (scooter_rider becomes 36 with ``scooter="keep"``). An answer below
       ``vlm.min_conf``, or unclear, decides nothing;
    4. fallback: the class with the larger overlap, then the larger class vote, motorcycle on
       a tie.

    Every row of a kept track gets the one class, so a track never flickers. Filled rows
    (``r3 == 1`` or ``interp == 1``) are not used as evidence.

    Parameters
    ----------
    scooter : {"drop", "keep"}
        What to do with stand-up scooters and skateboards. Default ``"drop"``.
    iob_threshold : float
        Share of the track box a detection must cover to count as overlapping, in (0, 1].
        Default 0.2.
    min_ratio : float
        Share of frames a class must overlap on for layer 2, in (0, 1]. Default 0.7.
    dominance : float
        How many times the other class's overlap the winner needs, at least 1. Default 1.5.
    vote_share : float
        Share of rows carrying a class that decides layer 2 by votes, in (0, 1]. Default 0.85.
    scooter_share : float
        Share of rows carrying class 36 that flags a scooter in layer 1, in (0, 1]. Default
        0.4. A track with at least as many 36 rows as 1 and 3 rows is flagged too.
    vlm : dnt.refine.config.VLMConfig or None
        Layer 3 settings; None turns layer 3 off. Its ``min_conf`` is the answer threshold.
    vlm_crops : int
        Crops tiled into the one image per track. Default 3.
    vlm_api_key : str or None
        API key; takes precedence over ``vlm.api_key_file`` and the environment. Not allowed
        together with ``vlm_backend_factory``, which builds the backend itself.
    vlm_backend_factory : callable or None
        Builds the backend from the resolved ``VLMConfig`` (tests); skips the preflight.

    Attributes
    ----------
    decisions : pandas.DataFrame
        Per-track reasons of the last ``re_classify`` call: ``track, cls, layer, n,
        mc_ratio, bike_ratio, vlm_answer, vlm_conf``, with ``cls`` missing for a dropped
        track and ``layer`` one of ``scooter``, ``prior``, ``vlm`` and ``fallback``.

    Raises
    ------
    ValueError
        For a broken setting, an invalid VLM config, one resolving to backend ``none``, a
        missing API key, or ``vlm_api_key`` given together with ``vlm_backend_factory``.
    ImportError
        When the VLM backend's package is not installed.

    """

    BICYCLE, MOTORCYCLE, SKATEBOARD = BICYCLE, MOTORCYCLE, SKATEBOARD

    def __init__(
        self,
        scooter: str = "drop",
        iob_threshold: float = 0.2,
        min_ratio: float = 0.7,
        dominance: float = 1.5,
        vote_share: float = 0.85,
        scooter_share: float = 0.4,
        vlm: "VLMConfig | None" = None,
        vlm_crops: int = 3,
        *,
        vlm_api_key: str | None = None,
        vlm_backend_factory: Callable | None = None,
    ) -> None:
        """Validate the settings, then prepare the VLM."""
        if vlm_api_key is not None and vlm_backend_factory is not None:
            raise ValueError(
                "vlm_api_key is ignored when vlm_backend_factory builds the backend; pass one"
            )

        def share(v):
            return (
                isinstance(v, numbers.Real)
                and not isinstance(v, bool)
                and math.isfinite(v)
                and 0 < v <= 1
            )

        problems = []
        if scooter not in ("drop", "keep"):
            problems.append("scooter must be 'drop' or 'keep'")
        for name, v in (
            ("iob_threshold", iob_threshold),
            ("min_ratio", min_ratio),
            ("vote_share", vote_share),
            ("scooter_share", scooter_share),
        ):
            if not share(v):
                problems.append(f"{name} must be a number in (0, 1]")
        if not (
            isinstance(dominance, numbers.Real)
            and not isinstance(dominance, bool)
            and math.isfinite(dominance)
            and dominance >= 1
        ):
            problems.append("dominance must be a number >= 1")
        if not (
            isinstance(vlm_crops, numbers.Integral)
            and not isinstance(vlm_crops, bool)
            and vlm_crops >= 1
        ):
            problems.append("vlm_crops must be an integer >= 1")
        if problems:
            raise ValueError("invalid TwoWheelReClass settings: " + "; ".join(problems))
        self.scooter = scooter
        self.iob_threshold = iob_threshold
        self.min_ratio = min_ratio
        self.dominance = dominance
        self.vote_share = vote_share
        self.scooter_share = scooter_share
        self.vlm_crops = vlm_crops
        self._vlm_api_key = vlm_api_key
        self._vlm_backend_factory = vlm_backend_factory
        self.vlm = (
            None
            if vlm is None
            else _prepare_vlm(
                vlm, vlm_api_key, check=vlm_backend_factory is None, owner="TwoWheelReClass"
            )
        )
        self.decisions = pd.DataFrame(columns=_TWOWHEEL_COLUMNS)

    def _backend_factory(self):
        if self._vlm_backend_factory is not None:
            return self._vlm_backend_factory
        from ..refine.vlm import make_backend

        return lambda cfg: make_backend(cfg, api_key=self._vlm_api_key)

    def _answer_class(self, answer: str) -> int | None:
        """Return the class a VLM answer stands for, or None when it is not a two-wheeler."""
        if answer == "cyclist":
            return BICYCLE
        if answer == "motorcycle_rider":
            return MOTORCYCLE
        if answer == "scooter_rider" and self.scooter == "keep":
            return SKATEBOARD
        return None

    def re_classify(
        self,
        tracks: pd.DataFrame | str,
        input_video: str | None = None,
        mc_dets: pd.DataFrame | str | None = None,
        bike_dets: pd.DataFrame | str | None = None,
        track_ids: list | None = None,
        out_file: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Classify two-wheeler tracks and write the class into ``cls``.

        Parameters
        ----------
        tracks : pandas.DataFrame or str
            A track table (at least ``frame, track, x, y, w, h, cls``) or a headerless track
            file. Filled rows are never used as evidence.
        input_video : str or None
            The source video; read only to crop frames for the VLM, so it may be None without
            a ``vlm``.
        mc_dets : pandas.DataFrame or str or None
            Motorcycle detections: a table with ``frame, x, y, w, h``, or a headerless
            detection file (``Detector.DET_FIELDS`` layout). None means none.
        bike_dets : pandas.DataFrame or str or None
            Bicycle detections, in the same forms.
        track_ids : list or None
            Tracks to classify; None classifies all. Unknown IDs are ignored.
        out_file : str or None
            Where to write the result, headerless, in the input's column layout.
        verbose : bool
            Unused; kept for symmetry with the other re-classifiers.

        Returns
        -------
        pandas.DataFrame
            The input table with ``cls`` set on every row of each classified track, and the
            rows of dropped tracks removed; the per-track reasons are in ``self.decisions``.

        Raises
        ------
        FileNotFoundError
            If a ``vlm`` is set, ``input_video`` is needed and does not exist.
        ValueError
            If the track or detection input is malformed.

        """
        table = _read_tracks(tracks)
        self.decisions = pd.DataFrame(columns=_TWOWHEEL_COLUMNS)
        if table.empty:
            if out_file:
                write_track(table, out_file)
            return table

        mc = _read_dets(mc_dets)
        bike = _read_dets(bike_dets)
        ids = VehReClass._select(table, track_ids)
        real = table.loc[~filled_mask(table)]
        groups = dict(tuple(real.groupby("track", sort=False)))
        rows: dict = {}
        pending: list = []
        votes: dict = {}
        for t in ids:
            g = groups.get(t)
            if g is None or g.empty:  # nothing but filled rows: use them rather than nothing
                g = table[table["track"] == t]
            n = len(g)
            cls_votes = g["cls"].value_counts()
            v1, v3, v36 = (int(cls_votes.get(c, 0)) for c in (BICYCLE, MOTORCYCLE, SKATEBOARD))
            votes[t] = (v1, v3)
            mc_ratio = _iob_frames(g, mc, self.iob_threshold) / n
            bike_ratio = _iob_frames(g, bike, self.iob_threshold) / n
            r = {
                "track": t,
                "cls": None,
                "layer": None,
                "n": n,
                "mc_ratio": mc_ratio,
                "bike_ratio": bike_ratio,
                "vlm_answer": None,
                "vlm_conf": math.nan,
            }
            rows[t] = r
            if v36 / n >= self.scooter_share or (v36 > 0 and v36 >= max(v1, v3)):
                r["cls"] = SKATEBOARD if self.scooter == "keep" else None
                r["layer"] = "scooter"
            elif (mc_ratio >= self.min_ratio and mc_ratio >= bike_ratio * self.dominance) or (
                v3 / n >= self.vote_share and v1 == 0
            ):
                r["cls"], r["layer"] = MOTORCYCLE, "prior"
            elif (bike_ratio >= self.min_ratio and bike_ratio >= mc_ratio * self.dominance) or (
                v1 / n >= self.vote_share and v3 == 0
            ):
                r["cls"], r["layer"] = BICYCLE, "prior"
            else:
                pending.append(t)

        if self.vlm is not None and pending:
            if input_video is None or not Path(input_video).exists():
                raise FileNotFoundError(f"Video file not found: {input_video}")
            requests = {
                t: _even_sample(groups.get(t, table.iloc[0:0]), self.vlm_crops) for t in pending
            }
            answers = _ask_vlm(
                self.vlm, self._backend_factory(), str(input_video), requests, kind="riderclass"
            )
            for t, a in answers.items():
                r = rows[t]
                if a.error:  # no answer was given; the warning already names the error
                    continue
                r["vlm_answer"], r["vlm_conf"] = a.answer, a.confidence
                if not (math.isfinite(a.confidence) and a.confidence >= self.vlm.min_conf):
                    continue
                if a.answer in ("pedestrian", "cyclist", "motorcycle_rider", "scooter_rider"):
                    r["cls"], r["layer"] = self._answer_class(a.answer), "vlm"

        for t in pending:
            r = rows[t]
            if r["layer"] is not None:
                continue
            v1, v3 = votes[t]
            if r["mc_ratio"] != r["bike_ratio"]:
                r["cls"] = MOTORCYCLE if r["mc_ratio"] > r["bike_ratio"] else BICYCLE
            else:
                r["cls"] = MOTORCYCLE if v3 >= v1 else BICYCLE
            r["layer"] = "fallback"

        dropped = [t for t, r in rows.items() if r["cls"] is None]
        cls_map = {t: r["cls"] for t, r in rows.items() if r["cls"] is not None}
        out = table.copy()
        hit = out["track"].isin(list(cls_map))
        out.loc[hit, "cls"] = out.loc[hit, "track"].map(cls_map).astype(int)
        out = out[~out["track"].isin(dropped)].reset_index(drop=True)
        if dropped:
            log.info("dropped %d track(s) that are not bicycles or motorcycles", len(dropped))
        if out_file:
            write_track(out, out_file)
        decisions = pd.DataFrame(list(rows.values()), columns=_TWOWHEEL_COLUMNS)
        decisions["cls"] = decisions["cls"].astype("Int64")  # <NA>: dropped
        self.decisions = decisions
        return out
