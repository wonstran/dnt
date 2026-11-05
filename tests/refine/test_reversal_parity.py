"""Parity with the Miami driver script ``3a1-split_reversals.py`` (reversal spec 6.3).

``_slope`` and the body of ``oracle_hit_rows`` are copied word for word from the script's
``_slope`` and ``find_reversals``, cut at ``hit = np.flatnonzero(cos <= cos_max)``.
"""

import numpy as np
import pandas as pd

from dnt.refine.config import RefineConfig
from dnt.refine.reversal import find_reversals, hit_rows, row_values, runs

WINDOW_S = 2.0
MIN_SPEED = 0.5
MIN_ANGLE = 135.0
MIN_ROWS = 5
MIN_SIDE_ROWS = 3
RC = RefineConfig.defaults("vehicle").reversal


def _slope(frames: np.ndarray, vals: np.ndarray) -> float:
    """Least-squares slope of ``vals`` over ``frames`` (units per frame)."""
    t = frames - frames.mean()
    den = float((t * t).sum())
    return float((t * (vals - vals.mean())).sum() / den) if den > 0 else 0.0


def oracle_hit_rows(g: pd.DataFrame, fps: float) -> np.ndarray:
    frames = g["frame"].to_numpy()
    n = len(frames)
    if n < 2 * MIN_ROWS:
        return np.empty(0, dtype=int)
    cx = (g["x"] + g["w"] / 2.0).to_numpy(float)
    cy = (g["y"] + g["h"] / 2.0).to_numpy(float)
    h = g["h"].to_numpy(float)
    w = max(1, round(WINDOW_S * fps))
    cos_max = np.cos(np.radians(MIN_ANGLE))
    cos = np.full(n, np.nan)
    for i in range(1, n):
        t = frames[i]
        lo = int(np.searchsorted(frames, t - w, side="left"))
        hi = int(np.searchsorted(frames, t + w, side="left"))
        if i - lo < MIN_SIDE_ROWS or hi - i < MIN_SIDE_ROWS:
            continue
        vb = np.array([_slope(frames[lo:i], cx[lo:i]), _slope(frames[lo:i], cy[lo:i])])
        va = np.array([_slope(frames[i:hi], cx[i:hi]), _slope(frames[i:hi], cy[i:hi])])
        nb, na = float(np.hypot(*vb)), float(np.hypot(*va))
        sb = nb * fps / max(float(np.median(h[lo:i])), 1.0)
        sa = na * fps / max(float(np.median(h[i:hi])), 1.0)
        if sb < MIN_SPEED or sa < MIN_SPEED:
            continue
        cos[i] = float(vb @ va) / (nb * na)
    return np.flatnonzero(cos <= cos_max)


def random_track(seed: int) -> pd.DataFrame:
    """A piecewise-linear 2-D path: 1-3 legs, random turns (often reversals), gaps and noise."""
    rng = np.random.default_rng(seed)
    h = float(rng.uniform(8, 40))
    w = 0.8 * h
    f, px, py = 0, 500.0, 500.0
    head = float(rng.uniform(0, 2 * np.pi))
    fr, xs, ys = [], [], []
    for leg in range(int(rng.integers(1, 4))):
        if leg:
            head += np.pi if rng.random() < 0.6 else float(rng.uniform(-np.pi, np.pi))
        sp = float(rng.uniform(0.1, 3.0)) * h / 10.0
        for _ in range(int(rng.integers(8, 50))):
            if rng.random() < 0.05:
                f += int(rng.integers(2, 12))
            px += sp * np.cos(head)
            py += sp * np.sin(head)
            fr.append(f)
            xs.append(px + rng.normal(0, 0.5))
            ys.append(py + rng.normal(0, 0.5))
            f += 1
    cx, cy = np.array(xs), np.array(ys)
    return pd.DataFrame({"frame": fr, "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h})


def _ref_keep(proposed, n, min_rows):
    """Independent reference of spec 3.2 step 7."""
    out, prev = [], 0
    for i in sorted(proposed):
        if i - prev < min_rows or n - i < min_rows:
            continue
        out.append(i)
        prev = i
    return out


def test_the_hit_rows_equal_the_scripts_on_random_tracks():
    with_hits = 0
    for seed in range(200):
        g = random_track(seed)
        boxes = g[["x", "y", "w", "h"]].to_numpy()
        expected = oracle_hit_rows(g, 10.0)
        assert hit_rows(g["frame"].to_numpy(), boxes, 10.0, RC).tolist() == expected.tolist(), seed
        with_hits += bool(len(expected))
    assert with_hits >= 60  # 80 of the 200 seeds hold a reversal; the test must exercise them


def test_cuts_follow_the_turn_rule_and_the_length_rule_on_the_stages_own_rows():
    for seed in range(200):
        g = random_track(seed)
        frames, boxes = g["frame"].to_numpy(), g[["x", "y", "w", "h"]].to_numpy()
        v = row_values(frames, boxes, 10.0, RC)
        proposed = []
        for run in runs(hit_rows(frames, boxes, 10.0, RC)):
            c = v.change[run]
            proposed.append(int(run[np.flatnonzero(c >= c.max() - 1e-9 * max(1.0, c.max()))[0]]))
        expected = _ref_keep(proposed, len(frames), RC.min_rows)
        assert [r.row for r in find_reversals(frames, boxes, 10.0, RC)] == expected, seed
