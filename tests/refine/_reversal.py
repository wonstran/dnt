"""Synthetic vehicle tracks and events for the reversal-stage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnt.refine import io
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.verify import decide

from ._fixtures import table

FPS = 10.0


def out_and_back(n: int = 60, turn: int = 30, speed: float = 2.0):
    """Frames and centre x of a box that moves right, then back left from frame ``turn``."""
    t = np.arange(n)
    return t, 100.0 + np.where(t < turn, speed * t, 2.0 * speed * turn - speed * t)


def path(vels) -> np.ndarray:
    """Positions starting at 100 that move by ``vels[k]`` into frame ``k`` (k >= 1)."""
    v = np.asarray(vels, dtype=float)
    return 100.0 + np.concatenate([[0.0], np.cumsum(v[1:])])


def boxes_at(cx, cy=100.0, h=10.0, w=10.0) -> np.ndarray:
    """``(n, 4)`` boxes centred on ``(cx, cy)``; ``cy`` and ``h`` broadcast."""
    cx = np.asarray(cx, dtype=float)
    cy = np.broadcast_to(np.asarray(cy, dtype=float), cx.shape)
    h = np.broadcast_to(np.asarray(h, dtype=float), cx.shape)
    return np.column_stack([cx - w / 2.0, cy - h / 2.0, np.full(cx.shape, w), h])


def cx_out_and_back(f, turn: int = 30, speed: float = 2.0) -> float:
    """Centre x at frame ``f`` of ``out_and_back``."""
    return 100.0 + (speed * f if f < turn else 2.0 * speed * turn - speed * f)


def var_height(f) -> float:
    """The review's height: 50 px on frames 15..30, 10 px elsewhere."""
    return 50.0 if 15 <= f <= 30 else 10.0


def rows(track, frames, cx_of=cx_out_and_back, *, h_of=lambda f: 10.0, w=10.0, cy=100.0,
         score=0.9, cls=2):
    """Raw rows of a box centred on ``(cx_of(f), cy)`` with height ``h_of(f)``."""
    return [
        [int(f), track, float(cx_of(f)) - w / 2.0, cy - h_of(f) / 2.0, w, float(h_of(f)),
         score, cls, -1, -1]
        for f in frames
    ]


def work_of(*row_lists, raw=None) -> pd.DataFrame:
    """Work table of the rows; ``raw`` maps a track ID to the raw ID its rows carry."""
    work = io.to_work(table(*row_lists)).work
    for t, r in (raw or {}).items():
        work.loc[work["track"] == t, "raw_id"] = r
    return work


def link_event(i, j, score, lineage_i=None, lineage_j=None) -> Event:
    """An accepted, applied LINK event ``i -> j``."""
    ev = Event.propose(
        stage="link",
        kind=EventKind.LINK,
        tracks=[i, j],
        lineage=[lineage_i or [[i, 0, 0]], lineage_j or [[j, 1, 1]]],
        frames=(0, 1),
        params={"gate": "normal", "gap": [0, 1]},
        algo_score=score,
    )
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    ev.applied = True
    return ev
