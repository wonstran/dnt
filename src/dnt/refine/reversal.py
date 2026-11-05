"""Stage ``reversal``: split vehicle tracks where the direction reverses (reversal spec).

Design: docs/superpowers/specs/2026-11-04-vehicle-reversal-split-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ReversalConfig, to_frames
from .primitives import mean_velocity

STAGE = "reversal"
_TIE_RTOL = 1e-9


@dataclass(frozen=True)
class Reversal:
    """One kept cut of a track (spec 3.2); ``run`` is the first and last frame of its run."""

    row: int
    frame: int
    angle: float
    change: float
    run: tuple[int, int]
    speed_before: float
    speed_after: float
    rows_before: int
    rows_after: int


@dataclass(frozen=True)
class RowValues:
    """Per-row values of the rule; ``cos`` and ``change`` are NaN where a row has no value."""

    cos: np.ndarray
    change: np.ndarray
    speed_before: np.ndarray
    speed_after: np.ndarray


def row_values(frames, boxes, fps: float, rc: ReversalConfig) -> RowValues:
    """Return each row's cosine, velocity change and side speeds (spec 3.2 steps 2-3).

    Parameters
    ----------
    frames : array_like of int
        The track's frames, sorted ascending.
    boxes : array_like
        ``(n, 4)`` boxes ``x, y, w, h`` in the same order.
    fps : float
        Frame rate; converts ``rc.window`` to frames and speeds to per second.
    rc : ReversalConfig
        The stage settings.

    Returns
    -------
    RowValues
        Speeds are in box heights per second, ``change`` in pixels per frame.

    """
    frames = np.asarray(frames, dtype=int)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(frames)
    cx = b[:, 0] + b[:, 2] / 2.0
    cy = b[:, 1] + b[:, 3] / 2.0
    h = b[:, 3]
    w = to_frames(rc.window, fps)
    cos = np.full(n, np.nan)
    change = np.full(n, np.nan)
    sb = np.zeros(n)
    sa = np.zeros(n)
    for i in range(1, n):
        t = frames[i]
        lo = int(np.searchsorted(frames, t - w, side="left"))
        hi = int(np.searchsorted(frames, t + w, side="left"))
        if i - lo < rc.min_side_rows or hi - i < rc.min_side_rows:
            continue
        if frames[lo] == frames[i - 1] or frames[i] == frames[hi - 1]:
            continue  # one distinct frame on a side has no slope (spec 8)
        vb = mean_velocity(frames[lo:i], cx[lo:i], cy[lo:i])
        va = mean_velocity(frames[i:hi], cx[i:hi], cy[i:hi])
        nb, na = float(np.hypot(*vb)), float(np.hypot(*va))
        if nb == 0.0 or na == 0.0:
            continue
        s_b = nb * fps / max(float(np.median(h[lo:i])), 1.0)
        s_a = na * fps / max(float(np.median(h[i:hi])), 1.0)
        if s_b < rc.min_speed or s_a < rc.min_speed:
            continue
        cos[i] = float(vb @ va) / (nb * na)
        change[i] = float(np.hypot(*(va - vb)))
        sb[i], sa[i] = s_b, s_a
    return RowValues(cos, change, sb, sa)


def _hits(v: RowValues, rc: ReversalConfig) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.flatnonzero(v.cos <= np.cos(np.radians(rc.min_angle)))


def hit_rows(frames, boxes, fps: float, rc: ReversalConfig) -> np.ndarray:
    """Return the rows that turn by at least ``rc.min_angle`` (spec 3.2 steps 1-4).

    These are the values the script's ``hit`` array holds; a track shorter than
    ``2 * rc.min_rows`` rows has none.
    """
    if len(frames) < 2 * rc.min_rows:
        return np.empty(0, dtype=int)
    return _hits(row_values(frames, boxes, fps, rc), rc)


def runs(hit: np.ndarray) -> list[np.ndarray]:
    """Split sorted hit rows into runs of consecutive rows (spec 3.2 step 5)."""
    hit = np.asarray(hit, dtype=int)
    if len(hit) == 0:
        return []
    return np.split(hit, np.flatnonzero(np.diff(hit) > 1) + 1)


def cut_in_run(run: np.ndarray, change: np.ndarray) -> int:
    """Return the run's row with the largest velocity change, the earliest of ties (step 6)."""
    c = np.asarray(change, dtype=float)[run]
    best = float(np.max(c))
    tied = np.flatnonzero(c >= best - _TIE_RTOL * max(1.0, abs(best)))
    return int(run[int(tied[0])])


def keep_cuts(proposed, n: int, min_rows: int) -> list[int]:
    """Keep, in row order, the cuts whose parts each keep ``min_rows`` rows (step 7)."""
    kept: list[int] = []
    prev = 0
    for i in sorted(int(x) for x in proposed):
        if i - prev >= min_rows and n - i >= min_rows:
            kept.append(i)
            prev = i
    return kept


def find_reversals(frames, boxes, fps: float, rc: ReversalConfig) -> list[Reversal]:
    """Return the kept cuts of one track, in frame order (spec 3.2).

    Parameters
    ----------
    frames, boxes, fps, rc
        As for ``row_values``.

    Returns
    -------
    list of Reversal
        The row of each cut is the first row of the part after it.

    """
    frames = np.asarray(frames, dtype=int)
    n = len(frames)
    if n < 2 * rc.min_rows:
        return []
    v = row_values(frames, boxes, fps, rc)
    proposed = {cut_in_run(r, v.change): r for r in runs(_hits(v, rc))}
    kept = keep_cuts(list(proposed), n, rc.min_rows)
    out: list[Reversal] = []
    for k, i in enumerate(kept):
        run = proposed[i]
        prev = kept[k - 1] if k else 0
        nxt = kept[k + 1] if k + 1 < len(kept) else n
        out.append(
            Reversal(
                row=i,
                frame=int(frames[i]),
                angle=float(np.degrees(np.arccos(np.clip(v.cos[i], -1.0, 1.0)))),
                change=float(v.change[i]),
                run=(int(frames[run[0]]), int(frames[run[-1]])),
                speed_before=float(v.speed_before[i]),
                speed_after=float(v.speed_after[i]),
                rows_before=i - prev,
                rows_after=nxt - i,
            )
        )
    return out
