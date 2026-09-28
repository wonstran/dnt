"""Deterministic synthetic track tables for the dnt.refine tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]
DATA = Path(__file__).resolve().parent / "data"


def box_rows(track, frames, x0, y0, *, vx=0.0, vy=0.0, w=30.0, h=60.0, cls=0, score=0.9):
    """Rows of one constant-velocity box; position is relative to the first listed frame."""
    frames = [int(f) for f in frames]
    if not frames:
        return []
    f0 = frames[0]
    return [
        [f, track, x0 + vx * (f - f0), y0 + vy * (f - f0), w, h, score, cls, -1, -1]
        for f in frames
    ]


def table(*row_lists) -> pd.DataFrame:
    """Concatenate row lists into a 10-column raw track table."""
    rows = [r for rl in row_lists for r in rl]
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def random_tracks(seed: int = 0, n_objects: int = 40, n_frames: int = 300) -> pd.DataFrame:
    """Linear movers with random gaps; long gaps often restart the object under a new ID."""
    rng = np.random.default_rng(seed)
    rows = []
    next_id = 1
    for _ in range(n_objects):
        start = int(rng.integers(0, n_frames - 60))
        length = int(rng.integers(40, n_frames - start))
        x0, y0 = rng.uniform(0, 1500), rng.uniform(0, 900)
        vx, vy = rng.uniform(-6, 6), rng.uniform(-4, 4)
        w, h = rng.uniform(20, 80), rng.uniform(40, 120)
        cls = int(rng.choice([0, 2]))
        tid = next_id
        next_id += 1
        f = start
        while f < start + length:
            if rng.random() < 0.03:
                gap = int(rng.integers(2, 25))
                f += gap
                if gap > 8 and rng.random() < 0.6:
                    tid = next_id
                    next_id += 1
                continue
            j = rng.normal(0, 1.0, size=4)
            k = f - start
            rows.append([
                f, tid, round(x0 + vx * k + j[0], 1), round(y0 + vy * k + j[1], 1),
                round(w + j[2], 1), round(h + j[3], 1), round(float(rng.uniform(0.3, 0.95)), 2),
                cls, -1, -1,
            ])
            f += 1
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    return df.sort_values(["frame", "track"]).reset_index(drop=True)


def load_raw(seed: int) -> pd.DataFrame:
    """Read a saved random track table back with named columns (the baseline input form)."""
    return pd.read_csv(DATA / f"raw_{seed}.csv", header=None, names=TRACK_COLUMNS)
