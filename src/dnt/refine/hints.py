# src/dnt/refine/hints.py
"""Optional external cue files: ReClass hints (spec 2.5, 6.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)
_REQUIRED = ("track", "cls", "avg_score")


@dataclass(frozen=True)
class ReclassHint:
    """One ``ReClass.re_classify`` result for a raw track."""

    raw_id: int
    cls: int
    avg_score: float


def read_reclass_hints(path, known_raw_ids) -> dict[int, ReclassHint]:
    """Read ReClass output (header ``track, cls, avg_score``); ignore unknown track IDs."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"{path}: empty hints file; expected header track, cls, avg_score"
        ) from exc
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: hints file lacks {missing}; expected header track, cls, avg_score"
        )
    known = {int(k) for k in known_raw_ids}
    out: dict[int, ReclassHint] = {}
    unknown = 0
    for t, c, s in df[list(_REQUIRED)].itertuples(index=False):
        if int(t) not in known:
            unknown += 1
            continue
        out[int(t)] = ReclassHint(int(t), int(c), float(s))
    if unknown:
        log.warning("%s: ignored %d hint row(s) for unknown track IDs", path, unknown)
    return out
