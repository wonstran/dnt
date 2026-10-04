"""Stage ``dedup``: merge interleaved duplicate tracks (dedup spec)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig
from .events import Event, EventKind
from .features import track_embeddings
from .primitives import majority_class, ramp

log = logging.getLogger(__name__)

STAGE = "dedup"


@dataclass
class Track:
    """One track's observed rows, sorted by frame; ``rows`` keeps the work-table rows."""

    track: int
    cls: int
    frames: np.ndarray
    boxes: np.ndarray
    rows: pd.DataFrame


@dataclass
class Overlap:
    """The overlap of two tracks' frame ranges and the observations inside it (spec 3.2)."""

    lo: int
    hi: int
    n_a: int
    n_b: int
    shared: int
    co_occupancy: float
    in_a: np.ndarray
    in_b: np.ndarray


def describe(work: pd.DataFrame) -> dict[int, Track]:
    """Return a ``Track`` per track ID of ``work``."""
    out: dict[int, Track] = {}
    for t, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        out[int(t)] = Track(
            int(t),
            majority_class(g["cls"]),
            g["frame"].to_numpy(int),
            g[["x", "y", "w", "h"]].to_numpy(float),
            g,
        )
    return out


def class_ok(a: int, b: int, groups) -> bool:
    """Return True when two classes are equal or share a class group."""
    return a == b or any(a in g and b in g for g in groups)


def overlap(a: Track, b: Track) -> Overlap | None:
    """Return the overlap of ``a`` and ``b`` (frame range and observation counts), or None.

    ``co_occupancy`` is ``shared / min(n_a, n_b)``, the share of the sparser track's observed
    frames on which the other track is also observed; it is 0.0 when nothing is shared.
    """
    lo = int(max(a.frames[0], b.frames[0]))
    hi = int(min(a.frames[-1], b.frames[-1]))
    if lo > hi:
        return None
    in_a = (a.frames >= lo) & (a.frames <= hi)
    in_b = (b.frames >= lo) & (b.frames <= hi)
    n_a, n_b = int(in_a.sum()), int(in_b.sum())
    shared = int(np.intersect1d(a.frames[in_a], b.frames[in_b], assume_unique=True).size)
    denom = min(n_a, n_b)
    return Overlap(lo, hi, n_a, n_b, shared, shared / denom if denom else 0.0, in_a, in_b)


def _interp(t: Track, frames: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(frames, t.frames, t.boxes[:, k]) for k in range(4)])


def _iou_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def comotion(a: Track, b: Track, ov: Overlap) -> float:
    """Return the mean two-way IoU of observed boxes against the other track's interpolation.

    For each direction, one track's observed boxes are compared with the other track's
    interpolated box on the same frames (spec 3.2); the two directions are averaged.
    """
    ab = _iou_rows(a.boxes[ov.in_a], _interp(b, a.frames[ov.in_a])).mean()
    ba = _iou_rows(b.boxes[ov.in_b], _interp(a, b.frames[ov.in_b])).mean()
    return float((ab + ba) / 2.0)


def _unit_mean(frames: np.ndarray, emb: np.ndarray, lo: int, hi: int):
    keep = (frames >= lo) & (frames <= hi)
    if not keep.any():
        return None
    v = emb[keep].mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def appearance_similarity(a: Track, b: Track, ov: Overlap, appearance) -> float | None:
    """Return the cosine similarity of the two tracks' mean clean embeddings in the overlap.

    ``None`` without an appearance provider, or when either track has no clean sample there.
    """
    if appearance is None:
        return None
    ua = _unit_mean(*track_embeddings(appearance, lineage_of_rows(a.rows)), ov.lo, ov.hi)
    ub = _unit_mean(*track_embeddings(appearance, lineage_of_rows(b.rows)), ov.lo, ov.hi)
    if ua is None or ub is None:
        return None
    return float(ua @ ub)


def propose_merges(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, appearance=None
) -> list[Event]:
    """Propose one undecided ``MERGE`` event per candidate pair of tracks (spec 3.2, 3.3).

    A pair is a candidate when its classes share a group, the overlap of its frame ranges is at
    least ``min_overlap_seconds``, each track has at least ``min_observed`` observed rows in it,
    and the pair is not densely co-observed (``co_occupancy < cooccur_hi``).
    """
    dc = cfg.dedup
    if not dc.enabled or work.empty:
        return []
    groups = cfg.link.class_groups
    order = sorted(describe(work).values(), key=lambda t: (int(t.frames[0]), t.track))
    events: list[Event] = []
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            if second.frames[0] > first.frames[-1]:
                break
            if not class_ok(first.cls, second.cls, groups):
                continue
            ov = overlap(first, second)
            if ov is None or ov.hi - ov.lo + 1 < dc.min_overlap_seconds * fps:
                continue
            if ov.n_a < dc.min_observed or ov.n_b < dc.min_observed:
                continue
            if ov.co_occupancy >= dc.cooccur_hi:
                continue
            cm = comotion(first, second, ov)
            app = appearance_similarity(first, second, ov, appearance)
            term = (
                1.0
                if app is None
                else dc.appearance_floor
                + (1.0 - dc.appearance_floor) * float(ramp(app, dc.app_lo, dc.app_hi))
            )
            score = (
                float(ramp(cm, dc.comotion_lo, dc.comotion_hi))
                * (1.0 - float(ramp(ov.co_occupancy, dc.cooccur_lo, dc.cooccur_hi)))
                * term
            )
            lineage = [lineage_of_rows(first.rows), lineage_of_rows(second.rows)]
            events.append(
                Event.propose(
                    stage=STAGE,
                    kind=EventKind.MERGE,
                    tracks=[first.track, second.track],
                    lineage=lineage,
                    key_lineage=sorted(lineage, key=lambda spans: tuple(spans[0])),
                    frames=(ov.lo, ov.hi),
                    params={"span": [ov.lo, ov.hi]},
                    algo_score=score,
                    signals={
                        "shared": ov.shared,
                        "n_a": ov.n_a,
                        "n_b": ov.n_b,
                        "co_occupancy": ov.co_occupancy,
                        "comotion": cm,
                        "appearance": app,
                    },
                )
            )
    if appearance is None and events:
        log.info("dedup: no appearance provider; %d merge(s) scored on motion alone", len(events))
    return events
