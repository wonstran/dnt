"""Stage ``reversal``: split vehicle tracks where the direction reverses (reversal spec).

Design: docs/superpowers/specs/2026-11-04-vehicle-reversal-split-design.md.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig, ReversalConfig, to_frames
from .events import Event, EventKind
from .primitives import mean_velocity
from .reporter import advance

STAGE = "reversal"
log = logging.getLogger(__name__)
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


Tag = tuple[int, int]


def side_conflict(tags) -> tuple[int, int, int] | None:
    """Return ``(r, k1, k2)`` for the smallest raw ID with two sides in ``tags``, else None.

    ``k1 < k2`` are that raw ID's two smallest sides (spec 3.5).
    """
    by: dict[int, set[int]] = {}
    for r, k in tags:
        by.setdefault(int(r), set()).add(int(k))
    for r in sorted(by):
        ks = sorted(by[r])
        if len(ks) > 1:
            return r, ks[0], ks[1]
    return None


@dataclass
class ReversalSides:
    """Reversal cuts and the side tags each work track carries (spec 3.5).

    A tag ``(r, k)`` is side ``k`` of raw ID ``r``: the frames from its ``k``-th cut to the next.
    A track's tags are its own (read from its lineage) plus those it ``carried`` from tracks
    merged into it, so a side survives a merge that drops all of its rows.
    """

    cuts: dict[int, list[int]] = field(default_factory=dict)
    carried: dict[int, set[Tag]] = field(default_factory=dict)

    def own(self, lineage) -> set[Tag]:
        """Return the tags of a lineage's spans of cut raw IDs.

        Each span of a raw ID is assumed to lie on one side of its cuts (the reversal stage split
        there), so a span's first frame decides its side.
        """
        out: set[Tag] = set()
        for raw, first, _last in lineage:
            cs = self.cuts.get(int(raw))
            if cs:
                out.add((int(raw), bisect_right(cs, int(first))))
        return out

    def tags(self, track: int, lineage) -> set[Tag]:
        """Return ``own(lineage)`` plus the tags ``track`` carries."""
        return self.own(lineage) | self.carried.get(int(track), set())

    def table_tags(self, work: pd.DataFrame, excluded=None) -> dict[int, set[Tag]]:
        """Return the tags of every track of ``work``."""
        return {
            int(t): self.tags(int(t), lineage_of_rows(g, excluded))
            for t, g in work.groupby("track", sort=True)
        }

    def absorb(self, mapping, tags_before) -> None:
        """Carry every pre-merge track's tags into the track it now belongs to.

        ``tags_before`` holds the tags of every track before the merges, representatives
        included; ``mapping`` maps absorbed work IDs to survivors (``rep_of``, ``absorbed``).
        """
        for track, tags in tags_before.items():
            rep = int(mapping.get(track, track))
            self.carried.setdefault(rep, set()).update(tags)


def _weaker_key(e: Event) -> tuple[float, str]:
    return (float(e.algo_score), e.proposal_key)


def _graph(active: list[Event]) -> dict[int, dict[int, Event]]:
    adj: dict[int, dict[int, Event]] = {}
    for e in active:
        a, b = int(e.tracks[0]), int(e.tracks[1])
        for x, y in ((a, b), (b, a)):
            nb = adj.setdefault(x, {})
            if y not in nb or _weaker_key(e) < _weaker_key(nb[y]):
                nb[y] = e  # two links between one pair: the path step uses the weaker
    return adj


def _bfs_path(adj, u: int, v: int) -> list[int]:
    prev: dict[int, int | None] = {u: None}
    queue = deque([u])
    while queue:
        x = queue.popleft()
        if x == v:
            break
        for y in sorted(adj[x]):
            if y not in prev:
                prev[y] = x
                queue.append(y)
    out = [v]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return out[::-1]


def _first_side_conflict(active: list[Event], tags_of):
    adj = _graph(active)
    seen: set[int] = set()
    for start in sorted(adj):  # the first unseen node is its component's smallest member
        if start in seen:
            continue
        comp, queue = [], deque([start])
        seen.add(start)
        while queue:
            x = queue.popleft()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        conflict = side_conflict(set().union(*(tags_of.get(t, set()) for t in comp)))
        if conflict is None:
            continue
        r, k1, k2 = conflict
        u = min(t for t in comp if (r, k1) in tags_of.get(t, set()))
        v = min(t for t in comp if (r, k2) in tags_of.get(t, set()))
        if u == v:  # one track holds both sides; no link can undo that (spec 3.5)
            log.warning("track %d holds both sides of raw %d's reversal cut", u, r)
            continue
        steps = _bfs_path(adj, u, v)
        on_path = [adj[a][b] for a, b in pairwise(steps)]
        return min(on_path, key=_weaker_key), r, k1, k2, u, v
    return None


def resolve_reversal_chains(active: list[Event], tags_of) -> list[Event]:
    """Drop links until no chain of ``active`` links holds both sides of a cut (spec 3.5).

    Each round drops the weakest link, by ``(algo_score, proposal_key)``, on the path
    between the two smallest conflicting members of the first conflicting component.

    Parameters
    ----------
    active : list of Event
        Accepted LINK events that will be applied.
    tags_of : dict
        Work track ID to its tags (``ReversalSides.tags``), for every track in ``active``.

    Returns
    -------
    list of Event
        The dropped events, each with ``applied=False``, ``skipped_reason="reversal"`` and
        ``conflicts_with``.

    """
    active = list(active)
    dropped: list[Event] = []
    while (hit := _first_side_conflict(active, tags_of)) is not None:
        ev, r, k1, k2, u, v = hit
        ev.applied = False
        ev.signals["skipped_reason"] = "reversal"
        ev.signals["conflicts_with"] = {"raw": r, "sides": [k1, k2], "tracks": [u, v]}
        active = [e for e in active if e is not ev]
        dropped.append(ev)
    return dropped


def propose_reversal_splits(work: pd.DataFrame, cfg: RefineConfig, fps: float) -> list[Event]:
    """Propose one undecided ``SPLIT`` event per kept cut of every track (spec 3.3).

    Parameters
    ----------
    work : pandas.DataFrame
        Work table from ``io.to_work``.
    cfg : RefineConfig
        ``cfg.reversal`` holds the thresholds.
    fps : float
        Video frame rate.

    Returns
    -------
    list of Event
        In track order, then frame order. The caller decides and applies them.

    """
    events: list[Event] = []
    if work.empty:
        return events
    for tid, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        found = find_reversals(
            g["frame"].to_numpy(int), g[["x", "y", "w", "h"]].to_numpy(float), fps, cfg.reversal
        )
        advance()
        lin = lineage_of_rows(g)
        for r in found:
            events.append(
                Event.propose(
                    stage=STAGE,
                    kind=EventKind.SPLIT,
                    tracks=[int(tid)],
                    lineage=[lin],
                    frames=(r.frame, r.frame),
                    params={"cut_frame": r.frame},
                    algo_score=1.0,
                    signals={
                        "angle": r.angle,
                        "change": r.change,
                        "run": list(r.run),
                        "speed_before": r.speed_before,
                        "speed_after": r.speed_after,
                        "rows_before": r.rows_before,
                        "rows_after": r.rows_after,
                    },
                )
            )
    return events
