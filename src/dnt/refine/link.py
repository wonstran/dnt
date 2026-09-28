"""Stage 3: tracklet linking (spec 6.3), including the legacy ``link_tracklets``.

``link_tracklets`` moved here from ``dnt.track.post_process`` (which re-exports it) in dnt 0.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .apply import lineage_of_rows
from .config import RefineConfig, to_frames
from .events import Event, EventKind
from .features import Appearance, track_embeddings
from .primitives import box_centers, iou_matrix, majority_class, ramp, span_speed

LEGACY_COL_NAMES = ["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]


def _iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    a : tuple[float, float, float, float]
        Bounding box A as (x, y, width, height).
    b : tuple[float, float, float, float]
        Bounding box B as (x, y, width, height).

    Returns
    -------
    float
        IoU value in range [0.0, 1.0].

    Notes
    -----
    Uses standard IoU formula: intersection / union.
    Coordinates are in (x, y, width, height) format where (x, y) is top-left.

    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw * ah + bw * bh - inter, 1e-6)
    return inter / union


def _estimate_velocity(
    frames: np.ndarray, cx: np.ndarray, cy: np.ndarray, k: int
) -> tuple[float, float]:
    """Estimate velocity using recent observations via polynomial fitting.

    Parameters
    ----------
    frames : np.ndarray
        Array of frame numbers (timestamps) where observations occur.
    cx : np.ndarray
        Array of center x-coordinates corresponding to frames.
    cy : np.ndarray
        Array of center y-coordinates corresponding to frames.
    k : int
        Number of recent frames to use for velocity estimation.
        Uses last k points if available, otherwise uses all points.

    Returns
    -------
    tuple[float, float]
        Velocity (vx, vy) as pixels per frame.
        Returns (0.0, 0.0) if fewer than 2 observations available.

    Notes
    -----
    Uses 1st-order polynomial (linear) fit via np.polyfit for robust
    velocity estimation. Falls back to simple difference (cx[-1]-cx[-2])/dt
    if fitting fails or insufficient unique frame times.

    """
    n = len(frames)
    if n < 2:
        return 0.0, 0.0
    s = max(0, n - k)
    t = frames[s:].astype(float)
    x = cx[s:].astype(float)
    y = cy[s:].astype(float)
    if len(t) < 2 or np.allclose(t, t[0]):
        dt = float(max(frames[-1] - frames[-2], 1))
        return float((cx[-1] - cx[-2]) / dt), float((cy[-1] - cy[-2]) / dt)
    vx = float(np.polyfit(t, x, 1)[0])
    vy = float(np.polyfit(t, y, 1)[0])
    return vx, vy


class _DSU:
    """Disjoint Set Union (Union-Find) data structure for tracklet merging.

    Efficiently tracks which tracklet IDs belong to the same connected component
    using path compression and union by root heuristics.

    Attributes
    ----------
    parent : dict[int, int]
        Parent map where parent[x] points to parent node. If parent[x] == x,
        then x is a root (representative) of its component.

    Methods
    -------
    find(x: int) -> int
        Find the root representative of x's component with path compression.
    union(a: int, b: int) -> None
        Merge components containing a and b under a's root representative.

    Examples
    --------
    >>> dsu = _DSU([1, 2, 3, 4])
    >>> dsu.union(1, 2)  # Merge components
    >>> dsu.union(2, 3)  # Also connects 1 and 3
    >>> dsu.find(1) == dsu.find(3)  # Both have same root
    True

    """

    def __init__(self, elems: list[int]) -> None:
        """Initialize DSU with elements in separate components.

        Parameters
        ----------
        elems : list[int]
            List of element IDs to initialize. Each starts in its own component.

        """
        self.parent = {e: e for e in elems}

    def find(self, x: int) -> int:
        """Find root representative of x's component with path compression.

        Parameters
        ----------
        x : int
            Element ID to find.

        Returns
        -------
        int
            Root representative (parent[root] == root).

        """
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        """Merge components containing a and b under a's root representative.

        Parameters
        ----------
        a : int
            Element in first component.
        b : int
            Element in second component.

        Notes
        -----
        Updates parent[root_b] = root_a so all members of b's component
        now point to a's root as their ultimate parent.

        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _prepare_legacy(tracks: pd.DataFrame, col_names: list[str]) -> pd.DataFrame:
    df = tracks.copy()
    required = ["frame", "track", "x", "y", "w", "h"]
    if not all(c in df.columns for c in required):
        if len(df.columns) < 6:
            raise ValueError("tracks must include at least frame/track/x/y/w/h columns.")
        df.columns = col_names[: len(df.columns)]
    if "cls" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "cls"})
    if "interp" not in df.columns:
        df["interp"] = 0
    else:
        df["interp"] = pd.to_numeric(df["interp"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["frame", "track"]).reset_index(drop=True)
    df["cx"] = df["x"].astype(float) + (df["w"].astype(float) / 2.0)
    df["cy"] = df["y"].astype(float) + (df["h"].astype(float) / 2.0)
    df["area"] = df["w"].astype(float) * df["h"].astype(float)
    return df


def _legacy_descriptors(df: pd.DataFrame, vel_frames: int, pbar=None) -> dict[int, dict]:
    descriptors: dict[int, dict] = {}
    for tid, g in df.groupby("track", sort=False):
        g_real = g[g["interp"] != 1].sort_values("frame")
        if pbar is not None:
            pbar.update(1)
        if len(g_real) < 2:
            descriptors[int(tid)] = {"stitchable": False}
            continue
        start_row, end_row = g_real.iloc[0], g_real.iloc[-1]
        vx, vy = _estimate_velocity(
            g_real["frame"].to_numpy(), g_real["cx"].to_numpy(), g_real["cy"].to_numpy(), vel_frames
        )
        descriptors[int(tid)] = {
            "stitchable": True,
            "track": int(tid),
            "cls": int(end_row["cls"]) if "cls" in g_real.columns else -1,
            "t_start": int(g_real["frame"].iloc[0]),
            "t_end": int(g_real["frame"].iloc[-1]),
            "start_c": (float(start_row["cx"]), float(start_row["cy"])),
            "end_c": (float(end_row["cx"]), float(end_row["cy"])),
            "start_box": (float(start_row["x"]), float(start_row["y"]),
                          float(start_row["w"]), float(start_row["h"])),
            "end_box": (float(end_row["x"]), float(end_row["y"]),
                        float(end_row["w"]), float(end_row["h"])),
            "area_end": max(float(end_row["area"]), 1.0),
            "vx": vx,
            "vy": vy,
        }
    return descriptors


def _legacy_gate_cost(
    a: dict, b: dict, *, max_gap: int, size_ratio_max: float, dist_mult: float, iou_min: float,
    w_d: float, w_iou: float, w_s: float, dist_growth: float = 0.03, check_class: bool = True,
    detail: bool = False,
):
    """Return ``link_tracklets``'s cost for linking end ``a`` to start ``b``, or None if gated."""
    if a["track"] == b["track"]:
        return None
    dt = b["t_start"] - a["t_end"]
    if dt < 1 or dt > max_gap:
        return None
    if check_class and a["cls"] != b["cls"]:
        return None
    wi, hi = max(a["end_box"][2], 1.0), max(a["end_box"][3], 1.0)
    wj, hj = max(b["start_box"][2], 1.0), max(b["start_box"][3], 1.0)
    w_ratio, h_ratio = wj / wi, hj / hi
    if not (1.0 / size_ratio_max <= w_ratio <= size_ratio_max):
        return None
    if not (1.0 / size_ratio_max <= h_ratio <= size_ratio_max):
        return None
    pred_cx = a["end_c"][0] + a["vx"] * dt
    pred_cy = a["end_c"][1] + a["vy"] * dt
    sx, sy = b["start_c"]
    dist = float(np.hypot(pred_cx - sx, pred_cy - sy))
    if dist >= dist_mult * np.sqrt(a["area_end"]) * (1.0 + (dist_growth * dt)):
        return None
    iou = _iou_xywh((pred_cx - (wi / 2.0), pred_cy - (hi / 2.0), wi, hi), b["start_box"])
    if iou < iou_min:
        return None
    dist_norm = dist / (np.sqrt(a["area_end"]) + 1e-6)
    size_cost = abs(np.log(max(w_ratio, 1e-6))) + abs(np.log(max(h_ratio, 1e-6)))
    cost = (w_d * dist_norm) + (w_iou * (1.0 - iou)) + (w_s * size_cost)
    if detail:
        return cost, {"dist": dist, "iou_pred": iou, "w_ratio": w_ratio, "h_ratio": h_ratio}
    return cost


def _legacy_matches(stitchable: list[dict], **gate_kw) -> list[tuple[int, int, float]]:
    """Return ``link_tracklets``'s Hungarian matches as ``(end_track, start_track, cost)``."""
    ends = sorted(stitchable, key=lambda d: (d["t_end"], d["track"]))
    starts = sorted(stitchable, key=lambda d: (d["t_start"], d["track"]))
    inf = 1e9
    cost = np.full((len(ends), len(starts)), inf, dtype=float)
    for i, a in enumerate(ends):
        for j, b in enumerate(starts):
            c = _legacy_gate_cost(a, b, **gate_kw)
            if c is not None:
                cost[i, j] = c
    matches: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(ri, ci, strict=True) if cost[r, c] < inf]
    except Exception:
        used_r: set[int] = set()
        used_c: set[int] = set()
        pairs = sorted(np.argwhere(cost < inf), key=lambda rc: float(cost[rc[0], rc[1]]))
        for r, c in pairs:
            if int(r) in used_r or int(c) in used_c:
                continue
            used_r.add(int(r))
            used_c.add(int(c))
            matches.append((int(r), int(c)))
    return [(int(ends[r]["track"]), int(starts[c]["track"]), float(cost[r, c])) for r, c in matches]


def link_tracklets(
    tracks: pd.DataFrame | None = None,
    track_file: str | None = None,
    output_file: str | None = None,
    col_names: list[str] | None = None,
    max_gap: int = 20,
    vel_frames: int = 5,
    size_ratio_max: float = 2.0,
    dist_mult: float = 2.5,
    iou_min: float = 0.05,
    w_d: float = 1.0,
    w_iou: float = 1.0,
    w_s: float = 0.3,
    verbose: bool = True,
    video_index: int | None = None,
    video_tot: int | None = None,
) -> pd.DataFrame:
    """Reconnect broken tracklets using global optimal 1-to-1 matching.

    Links tracklets (short track segments) by computing a cost matrix based on
    spatial proximity, appearance similarity (IoU), and size consistency.
    Uses linear sum assignment (Hungarian algorithm) to find optimal matches,
    then merges tracklets via union-find to handle transitive connections.

    Parameters
    ----------
    tracks : pd.DataFrame | None, optional
        Input track data with columns: frame, track, x, y, w, h, and optionally
        score, cls, interp, r4. If None (default), ``track_file`` is used.
    track_file : str | None, optional
        CSV file path to read tracks from when ``tracks`` is None.
    output_file : str | None, optional
        CSV file path to write linked results. If None (default), results
        are not saved to file.
    col_names : list[str] | None, optional
        Column names to apply when input has positional integer columns.
        Default: ["frame","track","x","y","w","h","score","cls","interp","r4"].
    max_gap : int, optional
        Maximum frame gap between tracklet end and start to attempt linking.
        Default is 20.
    vel_frames : int, optional
        Number of recent frames to use for velocity estimation (polynomial fit).
        Default is 5.
    size_ratio_max : float, optional
        Maximum allowed width/height ratio between tracklet end and start.
        Default is 2.0. Values outside [1/ratio_max, ratio_max] are rejected.
    dist_mult : float, optional
        Distance threshold multiplier: distance_threshold = dist_mult * sqrt(area).
        Default is 2.5. Larger values allow more spatial flexibility.
    iou_min : float, optional
        Minimum Intersection over Union (IoU) between predicted and actual start box.
        Default is 0.05. Range [0.0, 1.0].
    w_d : float, optional
        Weight for normalized distance cost in weighted sum. Default is 1.0.
    w_iou : float, optional
        Weight for (1 - IoU) cost in weighted sum. Default is 1.0.
    w_s : float, optional
        Weight for size inconsistency cost (log ratio) in weighted sum.
        Default is 0.3 (smaller weight for size).
    verbose : bool, optional
        If True (default), display tqdm progress bar over tracklets.
    video_index : int | None, optional
        Current video index for progress description. Default is None.
    video_tot : int | None, optional
        Total number of videos for progress description. Default is None.

    Returns
    -------
    pd.DataFrame
        Output tracks with linked IDs. Same columns as input. Track IDs are
        remapped so that all frames belonging to a logical track share the same ID.
        Frame and track are sorted in output.

    Raises
    ------
    ValueError
        If tracks has fewer than 6 columns and no named columns provided.
    FileNotFoundError
        If track_file path does not exist.

    Notes
    -----
    **Algorithm Overview:**

    1. Extract descriptor for each track: endpoints, velocity, bounding boxes, class
    2. Build cost matrix using spatial (distance, IoU), appearance (class), and
       size (width/height ratio) metrics with weighted combination
    3. Solve linear sum assignment problem (Hungarian algorithm) to find optimal
       1-to-1 tracklet pairings with minimum total cost
    4. Use Union-Find (Disjoint Set Union) to handle transitive merges:
       if tracklet A links to B and B links to C, they all get merged to same group
    5. Remap all track IDs according to merged components

    **Cost Function Details:**

    - Velocity is estimated using polynomial fit (1st order) on recent observed frames
    - Predicted next tracklet start = end_position + velocity * temporal_gap
    - Distance is normalized by sqrt(bounding_box_area) for scale invariance
    - Only considers tracklets from same class (if class info available)
    - Skips linking if temporal gap, size ratio, or distance threshold exceeded

    **Input Requirements:**

    - Requires "frame", "track", "x", "y", "w", "h" columns minimum
    - If "interp" column exists, uses only rows with interp==0 for velocity estimation
    - If "cls" column exists, only links tracklets with same class

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample tracklets
    >>> tracks = pd.DataFrame({
    ...     'frame': [0, 1, 10, 11, 20, 21],
    ...     'track': [1, 1, 2, 2, 3, 3],
    ...     'x': [10, 12, 25, 27, 40, 42],
    ...     'y': [20, 22, 35, 37, 50, 52],
    ...     'w': [50, 50, 50, 50, 50, 50],
    ...     'h': [100, 100, 100, 100, 100, 100],
    ...     'cls': [1, 1, 1, 1, 1, 1],
    ... })
    >>> linked = link_tracklets(tracks, max_gap=15, verbose=False)
    >>> # Track IDs may now be remapped: e.g., [1, 1, 1, 1, 1, 1]
    >>> print(linked['track'].unique())  # All in same track if linked

    """
    if col_names is None:
        col_names = list(LEGACY_COL_NAMES)
    if tracks is None:
        if not track_file:
            raise ValueError("Either `tracks` or `track_file` must be provided.")
        tracks = pd.read_csv(track_file, header=None)
    if len(tracks) == 0:
        out = tracks.copy()
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out
    df = _prepare_legacy(tracks, col_names)
    n_tracks = df["track"].nunique()
    pbar = tqdm(total=n_tracks, unit=" tracklets", disable=not verbose)
    if verbose:
        if video_index is not None and video_tot is not None:
            pbar.set_description_str(f"Link tracklets {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Link tracklets")
    descriptors = _legacy_descriptors(df, vel_frames, pbar)
    pbar.close()
    stitchable = [d for d in descriptors.values() if d.get("stitchable", False)]
    if len(stitchable) <= 1:
        out = df.drop(columns=["cx", "cy", "area"])
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out
    matches = _legacy_matches(
        stitchable, max_gap=max_gap, size_ratio_max=size_ratio_max, dist_mult=dist_mult,
        iou_min=iou_min, w_d=w_d, w_iou=w_iou, w_s=w_s,
    )
    dsu = _DSU([int(d["track"]) for d in stitchable])
    for a_tid, b_tid, _ in matches:
        dsu.union(a_tid, b_tid)
    comps: dict[int, list[int]] = {}
    for d in stitchable:
        comps.setdefault(dsu.find(int(d["track"])), []).append(int(d["track"]))
    tstart_by_tid = {int(d["track"]): int(d["t_start"]) for d in stitchable}
    rep_map: dict[int, int] = {}
    for members in comps.values():
        rep = min(members, key=lambda t: (tstart_by_tid.get(t, 10**9), t))
        for t in members:
            rep_map[t] = rep
    for tid in df["track"].astype(int).unique().tolist():
        rep_map.setdefault(int(tid), int(tid))
    out = df.copy()
    out["track"] = out["track"].astype(int).map(rep_map).astype(int)
    out = out.drop(columns=["cx", "cy", "area"]).sort_values(["frame", "track"])
    out = out.reset_index(drop=True)
    if output_file:
        out.to_csv(output_file, index=False, header=False)
    return out


STAGE = "link"


@dataclass
class TrackDesc:
    """What stage 3 needs to know about one track (spec 6.3)."""

    track: int
    frames: np.ndarray
    boxes: np.ndarray
    cls_major: int
    cls_last: int
    h_end: float
    h_start: float
    vel: np.ndarray
    speed_static: float
    speed_end: float
    speed_start: float
    end_clean: np.ndarray
    start_clean: np.ndarray
    lineage: list

    @property
    def t_s(self) -> int:
        """First observed frame."""
        return int(self.frames[0])

    @property
    def t_e(self) -> int:
        """Last observed frame."""
        return int(self.frames[-1])

    @property
    def start_box(self) -> np.ndarray:
        """First box."""
        return self.boxes[0]

    @property
    def end_box(self) -> np.ndarray:
        """Last box."""
        return self.boxes[-1]

    @property
    def start_c(self) -> np.ndarray:
        """Center of the first box."""
        return box_centers(self.boxes[0])[0]

    @property
    def end_c(self) -> np.ndarray:
        """Center of the last box."""
        return box_centers(self.boxes[-1])[0]

    def legacy(self) -> dict:
        """Return the descriptor dict ``_legacy_gate_cost`` expects."""
        return {
            "track": self.track, "cls": self.cls_last, "t_start": self.t_s, "t_end": self.t_e,
            "start_c": tuple(map(float, self.start_c)), "end_c": tuple(map(float, self.end_c)),
            "start_box": tuple(map(float, self.start_box)),
            "end_box": tuple(map(float, self.end_box)),
            "area_end": max(float(self.end_box[2] * self.end_box[3]), 1.0),
            "vx": float(self.vel[0]), "vy": float(self.vel[1]),
        }


def describe_tracks(work, cfg: RefineConfig, fps: float, occluded) -> dict[int, TrackDesc]:
    """Build a ``TrackDesc`` per track; ``occluded`` is the row-aligned occlusion mask."""
    lc, hw = cfg.link, cfg.motion.height_window
    out: dict[int, TrackDesc] = {}
    for t, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy(int)
        boxes = g[["x", "y", "w", "h"]].to_numpy(float)
        c = box_centers(boxes)
        vx, vy = _estimate_velocity(frames, c[:, 0], c[:, 1], lc.vel_frames)
        occ = occluded.reindex(g.index, fill_value=False).to_numpy(bool)
        clean = np.flatnonzero(~occ)
        out[int(t)] = TrackDesc(
            track=int(t), frames=frames, boxes=boxes, cls_major=majority_class(g["cls"]),
            cls_last=int(g["cls"].iloc[-1]),
            h_end=max(float(np.median(boxes[-hw:, 3])), 1.0),
            h_start=max(float(np.median(boxes[:hw, 3])), 1.0),
            vel=np.array([vx, vy], dtype=float),
            speed_static=span_speed(frames, boxes, fps, lc.static_seconds, at="end"),
            speed_end=span_speed(frames, boxes, fps, lc.speed_seconds, at="end"),
            speed_start=span_speed(frames, boxes, fps, lc.speed_seconds, at="start"),
            end_clean=boxes[clean[-1]] if len(clean) else boxes[-1],
            start_clean=boxes[clean[0]] if len(clean) else boxes[0],
            lineage=lineage_of_rows(g),
        )
    return out


class _Occluders:
    """Boxes from the work table (owner = track) and the context (owner = -1), sorted by frame."""

    def __init__(self, work, context):
        """Index boxes by frame."""
        parts = [work[["frame", "x", "y", "w", "h"]].assign(owner=work["track"].astype(int))]
        if context is not None and len(context):
            parts.append(context[["frame", "x", "y", "w", "h"]].assign(owner=-1))
        allb = pd.concat(parts, ignore_index=True).sort_values("frame", kind="stable")
        self.frames = allb["frame"].to_numpy(int)
        self.boxes = allb[["x", "y", "w", "h"]].to_numpy(float)
        self.owners = allb["owner"].to_numpy(int)

    def witness(self, di: TrackDesc, dj: TrackDesc, iob_thr: float) -> tuple[float, list[int]]:
        """Return the fraction of gap frames whose hidden box is covered, and the occluders.

        The hidden box of each gap frame is interpolated between ``di``'s last box and ``dj``'s
        first box; a frame is covered when another box has ``IoB >= iob_thr`` with it (the
        intersection over the hidden box's area). All gap frames are scanned in one vectorized
        pass because ``dnt.engine.iobs`` loops in Python over every pair.
        """
        n = dj.t_s - di.t_e - 1
        if n <= 0:
            return 0.0, []
        lo = int(np.searchsorted(self.frames, di.t_e + 1, side="left"))
        hi = int(np.searchsorted(self.frames, dj.t_s, side="left"))
        owners = self.owners[lo:hi]
        keep = (owners != di.track) & (owners != dj.track)
        if not keep.any():
            return 0.0, []
        boxes = self.boxes[lo:hi][keep]
        owners = owners[keep]
        step = self.frames[lo:hi][keep] - di.t_e  # 1..n
        a = (step / (dj.t_s - di.t_e))[:, None]
        hidden = (1.0 - a) * di.end_box + a * dj.start_box
        hit = _iob_rows(hidden, boxes) >= iob_thr
        return len(np.unique(step[hit])) / n, sorted(int(o) for o in np.unique(owners[hit]))


def _iob_rows(hidden: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return ``iob_matrix(hidden[k], boxes[k])`` for every row ``k`` (both ``(N, 4)`` xywh)."""
    iw = np.minimum(hidden[:, 0] + hidden[:, 2], boxes[:, 0] + boxes[:, 2]) - np.maximum(
        hidden[:, 0], boxes[:, 0])
    ih = np.minimum(hidden[:, 1] + hidden[:, 3], boxes[:, 1] + boxes[:, 3]) - np.maximum(
        hidden[:, 1], boxes[:, 1])
    inter = np.maximum(0.0, iw) * np.maximum(0.0, ih)
    area = hidden[:, 2] * hidden[:, 3]
    return np.divide(inter, area, out=np.zeros_like(inter), where=area != 0)


@dataclass
class Candidate:
    """A gated end->start pair and its score (spec 6.3)."""

    i: int
    j: int
    gate: str
    g: int
    score: float
    signals: dict = field(default_factory=dict)


def _class_ok(a: int, b: int, groups) -> bool:
    return a == b or any(a in grp and b in grp for grp in groups)


def _size_ok(a, b, r: float) -> bool:
    wr = max(float(b[2]), 1.0) / max(float(a[2]), 1.0)
    hr = max(float(b[3]), 1.0) / max(float(a[3]), 1.0)
    return 1.0 / r <= wr <= r and 1.0 / r <= hr <= r


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _c_app(di: TrackDesc, dj: TrackDesc, appearance: Appearance, k: int, cache: dict) -> float:
    def emb(d):
        if d.track not in cache:
            cache[d.track] = track_embeddings(appearance, d.lineage)
        return cache[d.track]

    fi, ei = emb(di)
    fj, ej = emb(dj)
    if not len(fi) or not len(fj):
        return 0.5
    ei_k = ei[fi <= di.t_e][-k:]
    ej_k = ej[fj >= dj.t_s][:k]
    if not len(ei_k) or not len(ej_k):
        return 0.5
    return float((1.0 - _unit(ei_k.mean(axis=0)) @ _unit(ej_k.mean(axis=0))) / 2.0)


def _prior(di: TrackDesc, dj: TrackDesc, frame_size, margin: float) -> int:
    if frame_size is None:
        return 1
    width, height = frame_size

    def inside(box, m):
        x, y, w, h = box
        return x >= m and y >= m and x + w <= width - m and y + h <= height - m

    return int(inside(di.end_box, margin * di.h_end) and inside(dj.start_box, margin * dj.h_start))


def _overlap_gate(di: TrackDesc, dj: TrackDesc, lc):
    shared = np.intersect1d(di.frames, dj.frames)
    if not len(shared) or not _size_ok(di.end_box, dj.start_box, lc.size_ratio_max):
        return None
    bi = di.boxes[np.searchsorted(di.frames, shared)]
    bj = dj.boxes[np.searchsorted(dj.frames, shared)]
    vals = np.array([iou_matrix(bi[k : k + 1], bj[k : k + 1])[0, 0] for k in range(len(shared))])
    if (vals < lc.overlap_iou).any():
        return None
    return "overlap", float(1.0 - vals.mean()), {"overlap_iou": float(vals.mean())}


def _gate(di: TrackDesc, dj: TrackDesc, g: int, cfg: RefineConfig, fps: float, gaps, occl):
    lc = cfg.link
    mg, mgs, mgo = gaps
    if -lc.overlap_frames <= g <= 0:
        return _overlap_gate(di, dj, lc)
    if g < 1:
        return None
    if g <= mg:
        res = _legacy_gate_cost(
            di.legacy(), dj.legacy(), max_gap=mg, size_ratio_max=lc.size_ratio_max,
            dist_mult=lc.dist_mult, iou_min=lc.iou_min, w_d=lc.legacy_weights["d"],
            w_iou=lc.legacy_weights["iou"], w_s=lc.legacy_weights["s"],
            dist_growth=lc.dist_growth, check_class=False, detail=True,
        )
        if res is None:
            return None
        cost, terms = res
        return "normal", float(ramp(cost, 0.0, lc.legacy_cost_hi)), {"legacy_cost": cost, **terms}
    if di.speed_static < lc.static_speed and g <= mgs:
        if not _size_ok(di.end_box, dj.start_box, lc.size_ratio_max):
            return None
        dist = float(np.linalg.norm(dj.start_c - di.end_c))
        radius = lc.static_radius * di.h_end
        if dist > radius:
            return None
        return "static", dist / radius, {"static_dist": dist}
    if g <= mgo:
        if not _size_ok(di.end_clean, dj.start_clean, lc.size_ratio_max):
            return None
        chord = dj.start_c - di.end_c
        clen = float(np.linalg.norm(chord))
        speed_i = float(np.linalg.norm(di.vel)) * fps / di.h_end
        heading = None
        if speed_i >= lc.heading_min_speed and clen > 0:
            cosang = float(di.vel @ chord / (np.linalg.norm(di.vel) * clen))
            heading = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            if heading > lc.max_heading_change:
                return None
        v_need = clen / (di.h_end * g / fps)
        v_ref = max(di.speed_end, dj.speed_start, lc.min_feasible_speed)
        if v_need > lc.speed_factor * v_ref:
            return None
        # the witness scan is the costliest gate, so it runs after the cheap ones
        witness, ids = occl.witness(di, dj, lc.witness_iob)
        if witness < lc.witness_min:
            return None
        c_mot = (0.5 * v_need / (lc.speed_factor * v_ref)
                 + 0.5 * ((heading or 0.0) / lc.max_heading_change))
        return "occluded", float(c_mot), {"witness": witness, "occluders": ids,
                                          "v_need": v_need, "v_ref": v_ref, "heading": heading}
    return None


def score_candidates(
    work, cfg: RefineConfig, fps: float, *, appearance: Appearance | None, context,
    frame_size, occluded,
) -> tuple[list[Candidate], dict[int, TrackDesc]]:
    """Gate and score every end->start pair (spec 6.3)."""
    lc = cfg.link
    motion_only = appearance is None
    if motion_only:
        for name in ("weights", "weights_occluded"):
            wts = getattr(lc, name)
            if wts["mot"] + wts["gap"] <= 0:
                raise ValueError(
                    f"link.{name}: motion-only scoring (no appearance provider) needs a "
                    "positive mot + gap weight"
                )
    descs = describe_tracks(work, cfg, fps, occluded)
    if len(descs) < 2:
        return [], descs
    occl = _Occluders(work, context)
    gaps = (to_frames(lc.max_gap, fps), to_frames(lc.max_gap_static, fps),
            to_frames(lc.max_gap_occluded, fps))
    reach = max(gaps)  # each gate applies its own limit; the window only bounds the search
    order = sorted(descs.values(), key=lambda d: (d.t_s, d.track))
    starts = np.array([d.t_s for d in order])
    cache: dict = {}
    cands: list[Candidate] = []
    for di in sorted(descs.values(), key=lambda d: d.track):
        lo = int(np.searchsorted(starts, di.t_e - lc.overlap_frames, side="left"))
        hi = int(np.searchsorted(starts, di.t_e + reach, side="right"))
        for dj in order[lo:hi]:
            if dj.track == di.track or not _class_ok(di.cls_major, dj.cls_major, lc.class_groups):
                continue
            g = dj.t_s - di.t_e
            res = _gate(di, dj, g, cfg, fps, gaps, occl)
            if res is None:
                continue
            gate, c_mot, sig = res
            w = dict(lc.weights_occluded if gate == "occluded" else lc.weights)
            c_app = None
            if motion_only:
                total = w["mot"] + w["gap"]
                w = {"mot": w["mot"] / total, "gap": w["gap"] / total, "app": 0.0}
            else:
                c_app = _c_app(di, dj, appearance, lc.k_embed, cache)
            limit = {"normal": gaps[0], "overlap": gaps[0], "static": gaps[1],
                     "occluded": gaps[2]}[gate]
            c_gap = max(g, 0) / limit
            b = _prior(di, dj, frame_size, lc.border_margin)
            cost = w["mot"] * c_mot + w["gap"] * c_gap + w["app"] * (c_app or 0.0)
            s = float(np.clip((1.0 - cost) * (0.8 + 0.2 * b), 0.0, 1.0))
            if gate == "occluded":
                s = min(s, lc.occluded_score_cap)
            cands.append(Candidate(di.track, dj.track, gate, int(g), s, {
                **sig, "gate": gate, "g": int(g), "c_mot": c_mot, "c_app": c_app,
                "c_gap": c_gap, "b": b, "motion_only": motion_only,
            }))
    return cands, descs


def legacy_link_events(work, cfg: RefineConfig, fps: float) -> list[Event]:
    """Return ``link_tracklets``'s matches as LINK proposals (``link.mode: legacy``)."""
    lc = cfg.link
    if work.empty:
        return []
    df = _prepare_legacy(work[LEGACY_COL_NAMES], LEGACY_COL_NAMES)
    stitchable = [d for d in _legacy_descriptors(df, lc.vel_frames).values()
                  if d.get("stitchable")]
    if len(stitchable) <= 1:
        return []
    matches = _legacy_matches(
        stitchable, max_gap=to_frames(lc.max_gap, fps), size_ratio_max=lc.size_ratio_max,
        dist_mult=lc.dist_mult, iou_min=lc.iou_min, w_d=lc.legacy_weights["d"],
        w_iou=lc.legacy_weights["iou"], w_s=lc.legacy_weights["s"], dist_growth=lc.dist_growth,
    )
    by = {d["track"]: d for d in stitchable}
    events = []
    for a, b, cost in matches:
        t_e, t_s = by[a]["t_end"], by[b]["t_start"]
        events.append(Event.propose(
            stage=STAGE, kind=EventKind.LINK, tracks=[a, b],
            lineage=[lineage_of_rows(work[work["track"] == a]),
                     lineage_of_rows(work[work["track"] == b])],
            frames=(t_e, t_s), params={"gate": "legacy", "gap": [t_e, t_s]}, algo_score=1.0,
            signals={"legacy_cost": cost, "pass": 1},
        ))
    return events
