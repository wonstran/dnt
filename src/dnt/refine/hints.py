# src/dnt/refine/hints.py
"""Optional external cue files: ReClass hints (spec 2.5, 6.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
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
    repeated = 0
    for i, (t, c, s) in enumerate(df[list(_REQUIRED)].itertuples(index=False)):
        line = i + 2  # the header is line 1
        track = _as_int(t, "track", path, line)
        cls = _as_int(c, "cls", path, line)
        score = _as_score(s, path, line, track)
        if track not in known:
            unknown += 1
            continue
        if track in out:
            repeated += 1
        out[track] = ReclassHint(track, cls, score)
    if unknown:
        log.warning("%s: ignored %d hint row(s) for unknown track IDs", path, unknown)
    if repeated:
        log.warning(
            "%s: %d repeated hint row(s) for the same track; the last one is kept", path, repeated
        )
    return out


def _as_int(value, column: str, path, line: int) -> int:
    """Return ``value`` as an int, or raise ``ValueError`` naming the file and row."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{path}: row {line}: {column} {value!r} is not an integer")
    return int(number)


def _as_score(value, path, line: int, track: int) -> float:
    """Return ``value`` as a score within [0, 1], or raise ``ValueError`` naming file and row."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float("nan")
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{path}: row {line} (track {track}): avg_score {value!r} must be a number in [0, 1]"
        )
    return score
