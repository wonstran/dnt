"""Track, context, and video I/O for dnt.refine (spec 2.5)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRACK_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]
OUT_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]
WORK_COLUMNS = [*OUT_COLUMNS, "raw_id"]
CONTEXT_COLUMNS = ["frame", "track", "x", "y", "w", "h", "cls"]
_INT_OUT = ["frame", "track", "x", "y", "w", "h", "cls", "interp", "r4"]
_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class TrackInput:
    """A work table plus counts of the input rows that were removed."""

    work: pd.DataFrame
    n_filled_removed: int
    n_duplicates_removed: int


def empty_work() -> pd.DataFrame:
    """Return an empty work table with the standard columns."""
    floats = {"x", "y", "w", "h", "score"}
    return pd.DataFrame({c: pd.Series(dtype=float if c in floats else int) for c in WORK_COLUMNS})


def _read_numeric_csv(
    path, min_cols: int, check_cols: int | None = None, int_cols: tuple[int, ...] = ()
) -> pd.DataFrame:
    """Read a headerless numeric CSV; name the file and line of the first bad value.

    The first ``check_cols`` columns (default ``min_cols``) must be finite numbers, and the
    columns in ``int_cols`` (such as frame and track) must hold whole numbers.
    """
    path = Path(path)
    rows_list = []
    true_line_map = []
    first_field_count = None
    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            n_fields = line.count(",") + 1
            if first_field_count is None:
                first_field_count = n_fields
            elif n_fields != first_field_count:
                msg = (
                    f"{path}: expected {first_field_count} fields, found {n_fields} on "
                    f"line {line_num}: {line}"
                )
                raise ValueError(msg)
            rows_list.append(line)
            true_line_map.append(line_num)
    if not rows_list:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(
            StringIO("\n".join(rows_list)),
            header=None,
            dtype=str,
        )
    except pd.errors.ParserError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if raw.shape[1] < min_cols:
        raise ValueError(f"{path}: expected at least {min_cols} columns, found {raw.shape[1]}")
    num = raw.apply(pd.to_numeric, errors="coerce")
    n_check = min(raw.shape[1], check_cols or min_cols)
    checked = num.iloc[:, :n_check].to_numpy(float)
    ints = num.iloc[:, [c for c in int_cols if c < raw.shape[1]]].to_numpy(float)
    for what, bad in (
        ("non-numeric", np.isnan(checked).any(axis=1)),
        ("non-finite", ~np.isfinite(checked).all(axis=1)),
        ("non-integer frame or track", (ints != np.round(ints)).any(axis=1)),
    ):
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            text = ",".join(raw.iloc[i].fillna("").tolist())
            raise ValueError(f"{path}: {what} value on line {true_line_map[i]}: {text}")
    num = num.reset_index(drop=True)
    return num


def to_work(df: pd.DataFrame, *, source: str = "tracks") -> TrackInput:
    """Build a work table from a raw 6-10 column track table (positional or named).

    A row is a filled (interpolated) row, and is removed, when its flag is 1. The flag is
    column 8, named ``r3`` in the tracker layout and ``interp`` in interpolated output; a
    named table may use either name (or both).
    """
    df = df.copy()
    if not all(c in df.columns for c in ("frame", "track", "x", "y", "w", "h")):
        df = df.iloc[:, : len(TRACK_COLUMNS)]
        df.columns = TRACK_COLUMNS[: df.shape[1]]
    for c, default in (("score", -1.0), ("cls", -1), ("r4", -1)):
        if c not in df.columns:
            df[c] = default
    filled = np.zeros(len(df), dtype=bool)
    for flag in ("r3", "interp"):
        if flag in df.columns:
            filled |= (pd.to_numeric(df[flag], errors="coerce").fillna(-1) == 1).to_numpy()
    n_filled = int(filled.sum())
    df = df.loc[~filled]
    dup = df.duplicated(["track", "frame"], keep="first")
    n_dup = int(dup.sum())
    if n_dup:
        log.warning("%s: %d duplicate (track, frame) rows; kept the first of each", source, n_dup)
    df = df.loc[~dup]
    work = pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "track": df["track"].astype(int),
            "x": df["x"].astype(float),
            "y": df["y"].astype(float),
            "w": df["w"].astype(float),
            "h": df["h"].astype(float),
            "score": pd.to_numeric(df["score"], errors="coerce").fillna(-1.0).astype(float),
            "cls": pd.to_numeric(df["cls"], errors="coerce").fillna(-1).astype(int),
            "interp": 0,
            "r4": pd.to_numeric(df["r4"], errors="coerce").fillna(-1).astype(int),
            "raw_id": df["track"].astype(int),
        }
    )
    work = work.sort_values(["track", "frame"]).reset_index(drop=True)
    return TrackInput(work=work, n_filled_removed=n_filled, n_duplicates_removed=n_dup)


def read_tracks(path, *, fmt: str = "dnt", class_id: int = 0) -> TrackInput:
    """Read a dnt (10-column) or MOTChallenge track file into a work table (spec 2.5).

    Every column the format uses must be numeric: all ten for dnt (so a bad score, class, or
    filled-row flag is an error, not a silent -1), the first seven for MOT.
    """
    raw = _read_numeric_csv(path, min_cols=6, check_cols=10 if fmt == "dnt" else 7, int_cols=(0, 1))
    if raw.empty:
        return TrackInput(work=empty_work(), n_filled_removed=0, n_duplicates_removed=0)
    if fmt == "dnt":
        df = raw.iloc[:, : len(TRACK_COLUMNS)].copy()
    elif fmt == "mot":
        df = pd.DataFrame(
            {
                "frame": raw[0],
                "track": raw[1],
                "x": raw[2],
                "y": raw[3],
                "w": raw[4],
                "h": raw[5],
                "score": raw[6] if raw.shape[1] > 6 else -1.0,
                "cls": class_id,
                "r3": -1,
                "r4": -1,
            }
        )
    else:
        raise ValueError(f"unknown track format {fmt!r}; expected 'dnt' or 'mot'")
    return to_work(df, source=str(path))


def read_context(path, fmt: str = "auto") -> tuple[pd.DataFrame, str]:
    """Read a context file (dnt tracks or detections) as boxes with classes (spec 2.5)."""
    raw = _read_numeric_csv(path, min_cols=6, check_cols=8, int_cols=(0,))
    if raw.empty:
        return pd.DataFrame(columns=CONTEXT_COLUMNS), ("tracks" if fmt == "auto" else fmt)
    ncol = raw.shape[1]
    if fmt == "auto":
        if ncol == 10:
            fmt = "tracks"
        elif ncol == 8:
            fmt = "dets"
        else:
            raise ValueError(
                f"{path}: context file has {ncol} columns; expected 8 (detections) or 10 (tracks)"
            )
    if fmt not in ("tracks", "dets"):
        raise ValueError(f"unknown context format {fmt!r}; expected 'auto', 'tracks' or 'dets'")
    if ncol < 8:
        raise ValueError(f"{path}: context file has {ncol} columns; the class is column 8")
    ctx = pd.DataFrame(
        {
            "frame": raw[0].astype(int),
            "track": raw[1].astype(int) if fmt == "tracks" else -1,
            "x": raw[2].astype(float),
            "y": raw[3].astype(float),
            "w": raw[4].astype(float),
            "h": raw[5].astype(float),
            "cls": raw[7].astype(int),
        }
    )
    return ctx, fmt


def write_tracks(work: pd.DataFrame, path) -> None:
    """Write a work table as a headerless 10-column dnt track file sorted by frame, track."""
    out = work.reindex(columns=OUT_COLUMNS).copy()
    for c in _INT_OUT:
        out[c] = pd.to_numeric(out[c]).fillna(-1).round().astype(int)
    out["score"] = pd.to_numeric(out["score"]).fillna(-1.0).astype(float)
    out = out.sort_values(["frame", "track"], kind="mergesort")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, header=False)


def sha256_file(path) -> str:
    """Return the SHA-256 of a whole file, read in 8 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def video_info(path) -> dict:
    """Return ``fps``, ``frame_count``, ``width`` and ``height`` of a video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def video_fingerprint(path, frame_count: int) -> dict:
    """Return the video fingerprint: whole-file SHA-256, size, and frame count (spec 5.3)."""
    p = Path(path)
    return {"sha256": sha256_file(p), "size": p.stat().st_size, "frame_count": int(frame_count)}
