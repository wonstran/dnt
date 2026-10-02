"""Numeric primitives shared by the dnt.refine stages (spec section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import cluster_by_gap, iobs, ious


def ramp(x, lo: float, hi: float):
    """Map ``x`` linearly from ``[lo, hi]`` onto ``[0, 1]`` and clip; ``lo > hi`` decreases."""
    arr = np.asarray(x, dtype=float)
    out = (arr >= lo).astype(float) if hi == lo else np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def cv_kalman(process_var: float = 10.0, meas_var_pos: float = 25.0, meas_var_size: float = 16.0):
    """Return the constant-velocity Kalman filter shared by stage 1 and stage 4 (spec 5.2).

    State is ``[cx, vx, cy, vy, w, vw, h, vh]`` with one step per frame. The caller sets ``x``.
    """
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import KalmanFilter

    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8)
    for i in range(4):
        kf.F[2 * i, 2 * i + 1] = 1.0
    kf.H = np.zeros((4, 8))
    for i in range(4):
        kf.H[i, 2 * i] = 1.0
    q2 = Q_discrete_white_noise(dim=2, dt=1.0, var=process_var)
    kf.Q = np.zeros((8, 8))
    for i in range(4):
        kf.Q[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = q2
    kf.R = np.diag([meas_var_pos, meas_var_pos, meas_var_size, meas_var_size]).astype(float)
    kf.P = np.eye(8) * 100.0
    return kf


def xywh_to_z(boxes) -> np.ndarray:
    """Convert (N, 4) top-left ``x, y, w, h`` boxes to Kalman measurements ``cx, cy, w, h``."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0, b[:, 2], b[:, 3]])


def kalman_nis(
    frames,
    boxes,
    *,
    process_var: float = 10.0,
    meas_var_pos: float = 25.0,
    meas_var_size: float = 16.0,
) -> np.ndarray:
    """Return the normalized innovation squared at each observed row; NaN for the first row."""
    frames = np.asarray(frames, dtype=int)
    out = np.full(len(frames), np.nan)
    if len(frames) == 0:
        return out
    z = xywh_to_z(boxes)
    kf = cv_kalman(process_var, meas_var_pos, meas_var_size)
    kf.x = np.array([z[0, 0], 0.0, z[0, 1], 0.0, z[0, 2], 0.0, z[0, 3], 0.0])
    row_of = {int(f): i for i, f in enumerate(frames)}
    for f in range(int(frames[0]) + 1, int(frames[-1]) + 1):
        kf.predict()
        i = row_of.get(f)
        if i is None:
            continue
        y = z[i] - kf.H @ kf.x
        s = kf.H @ kf.P @ kf.H.T + kf.R
        out[i] = float(y @ np.linalg.solve(s, y))
        kf.update(z[i])
    return out


def box_centers(boxes) -> np.ndarray:
    """Return the (N, 2) centers of ``x, y, w, h`` boxes."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return b[:, :2] + b[:, 2:4] / 2.0


def rolling_height(h, window: int) -> np.ndarray:
    """Return the centered rolling median of box heights (spec 5.1)."""
    s = pd.Series(np.asarray(h, dtype=float))
    return s.rolling(max(int(window), 1), center=True, min_periods=1).median().to_numpy()


def speeds_hps(frames, boxes, fps: float, window: int = 15) -> np.ndarray:
    """Return per-row speed in box heights per second; NaN for the first row (spec 5.1)."""
    fr = np.asarray(frames, dtype=float)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = np.full(len(fr), np.nan)
    if len(fr) < 2:
        return out
    c = box_centers(b)
    ht = np.maximum(rolling_height(b[:, 3], window), 1.0)
    dist = np.linalg.norm(np.diff(c, axis=0), axis=1)
    out[1:] = dist / (ht[1:] * np.maximum(np.diff(fr), 1.0) / fps)
    return out


def span_speed(frames, boxes, fps: float, seconds: float, *, at: str) -> float:
    """Return the net speed (h/s) over the first or last ``seconds`` of rows; 0.0 if undefined."""
    fr = np.asarray(frames, dtype=int)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(fr) < 2:
        return 0.0
    span = seconds * fps
    if at == "end":
        sel = fr >= fr[-1] - span
    elif at == "start":
        sel = fr <= fr[0] + span
    else:
        raise ValueError(f"at must be 'start' or 'end', not {at!r}")
    f, bb = fr[sel], b[sel]
    if len(f) < 2 or f[-1] == f[0]:
        return 0.0
    c = box_centers(bb[[0, -1]])
    h = max(float(np.median(bb[:, 3])), 1.0)
    return float(np.linalg.norm(c[1] - c[0]) / (h * (f[-1] - f[0]) / fps))


def heading_smoothness(
    frames, boxes, fps: float, *, window: int = 15, moving_min: float = 0.3
) -> float:
    """Return ``1 - circular variance`` of heading over moving steps; NaN if under two steps."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(b) < 3:
        return float("nan")
    v = speeds_hps(frames, b, fps, window)
    d = np.diff(box_centers(b), axis=0)[v[1:] >= moving_min]
    if len(d) < 2:
        return float("nan")
    u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return float(np.linalg.norm(u.mean(axis=0)))


def majority_class(values) -> int:
    """Return the most frequent class; ties go to the class seen latest; -1 when empty."""
    vals = [int(v) for v in values]
    if not vals:
        return -1
    counts: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, c in enumerate(vals):
        counts[c] = counts.get(c, 0) + 1
        last[c] = i
    best = max(counts.values())
    return max((c for c, n in counts.items() if n == best), key=lambda c: last[c])


def _tlbr(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])


def iou_matrix(a, b) -> np.ndarray:
    """Return the (N, M) IoU of ``x, y, w, h`` boxes via ``dnt.engine.ious`` (spec 5.6)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return ious(_tlbr(a), _tlbr(b))


def iob_matrix(a, b) -> np.ndarray:
    """Return the (N, M) intersection over the area of each box in ``a`` (spec 5.6)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return iobs(a, b)[0]


def frame_runs(frames) -> list[tuple[int, int]]:
    """Return runs of consecutive frames as ``(first, last)`` pairs."""
    f = np.unique(np.asarray(frames, dtype=int))
    if f.size == 0:
        return []
    return [(int(r[0]), int(r[-1])) for r in cluster_by_gap(f, 1)]


#: Per frame, the work boxes and the context boxes are matched one to one (the assignment with
#: the largest total IoU). A context box matched to a work box with IoU >= this is taken to be
#: that row's own detection (for example, a detection file from the same run passed as context),
#: not another object. Such boxes are left out of the occlusion mask and of stage 3's witness
#: occluders (final-review rulings R22 and R6); the stage 2 context cues still see them.
CONTEXT_MATCH_IOU = 0.5


def context_duplicates(
    work: pd.DataFrame, context: pd.DataFrame | None, thr: float = CONTEXT_MATCH_IOU
) -> np.ndarray:
    """Return, per context row, whether it is a work row's own detection.

    In each frame the work boxes and the context boxes are matched one to one, maximizing the
    total IoU (``scipy.optimize.linear_sum_assignment``). A context box is a row's own detection
    when it is matched and the pair's IoU is >= ``thr``. One to one means a second box over the
    same row stays an occluder: only the row's best match is its own detection.

    Parameters
    ----------
    work : pandas.DataFrame
        Work rows (``frame, x, y, w, h``).
    context : pandas.DataFrame or None
        Context boxes (``frame, x, y, w, h``).
    thr : float
        Lowest IoU of a matched pair that counts as the row's own detection.

    Returns
    -------
    numpy.ndarray
        Boolean array, one entry per context row.

    """
    if context is None or not len(context):
        return np.zeros(0 if context is None else len(context), dtype=bool)
    from scipy.optimize import linear_sum_assignment

    dup = np.zeros(len(context), dtype=bool)
    frames = context["frame"].to_numpy(int)
    ctx_boxes = context[["x", "y", "w", "h"]].to_numpy(float)
    by_frame = {int(f): g[["x", "y", "w", "h"]].to_numpy(float) for f, g in work.groupby("frame")}
    order = np.argsort(frames, kind="stable")
    bounds = np.flatnonzero(np.diff(frames[order])) + 1
    for idx in np.split(order, bounds):
        own = by_frame.get(int(frames[idx[0]]))
        if own is None:
            continue
        m = iou_matrix(ctx_boxes[idx], own)
        rows, cols = linear_sum_assignment(m, maximize=True)
        matched = m[rows, cols] >= thr
        dup[idx[rows[matched]]] = True
    return dup


def drop_context_duplicates(
    work: pd.DataFrame, context: pd.DataFrame | None, thr: float = CONTEXT_MATCH_IOU
) -> pd.DataFrame | None:
    """Return ``context`` without the work rows' own detections (``context_duplicates``)."""
    if context is None or not len(context):
        return context
    return context.loc[~context_duplicates(work, context, thr)]


def occlusion_flags(work: pd.DataFrame, context: pd.DataFrame | None, thr: float) -> pd.Series:
    """Return True where a row's box has IoU >= ``thr`` with another box in its frame (spec 5.3).

    The other boxes are the frame's other work rows and its context boxes, except the rows' own
    detections (``context_duplicates``, the same rule as ``drop_context_duplicates``), so a
    detection file of the same run does not flag every row.
    """
    flags = pd.Series(False, index=work.index)
    ctx: dict[int, np.ndarray] = {}
    others = drop_context_duplicates(work, context)
    if others is not None and len(others):
        ctx = {int(f): g[["x", "y", "w", "h"]].to_numpy(float) for f, g in others.groupby("frame")}
    for f, g in work.groupby("frame"):
        boxes = g[["x", "y", "w", "h"]].to_numpy(float)
        best = np.zeros(len(boxes))
        if len(boxes) > 1:
            m = iou_matrix(boxes, boxes)
            np.fill_diagonal(m, 0.0)
            best = m.max(axis=1)
        occluders = ctx.get(int(f))
        if occluders is not None and len(occluders):
            best = np.maximum(best, iou_matrix(boxes, occluders).max(axis=1))
        flags.loc[g.index] = best >= thr
    return flags
