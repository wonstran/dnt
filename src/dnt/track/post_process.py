"""Backward-compatible home of the track post-processing functions (moved to ``dnt.refine``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..refine.interpolate import interpolate_tracks_rts

__all__ = ["interpolate_tracks_rts", "link_tracklets"]


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

    def _estimate_velocity(frames: np.ndarray, cx: np.ndarray, cy: np.ndarray, k: int) -> tuple[float, float]:
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

    if col_names is None:
        col_names = ["frame", "track", "x", "y", "w", "h", "score", "cls", "interp", "r4"]

    if tracks is None:
        if not track_file:
            raise ValueError("Either `tracks` or `track_file` must be provided.")
        tracks = pd.read_csv(track_file, header=None)

    if len(tracks) == 0:
        out = tracks.copy()
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out

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

    descriptors: dict[int, dict] = {}
    grouped = list(df.groupby("track", sort=False))
    pbar = tqdm(total=len(grouped), unit=" tracklets", disable=not verbose)
    if verbose:
        if video_index is not None and video_tot is not None:
            pbar.set_description_str(f"Link tracklets {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Link tracklets")

    for tid, g in grouped:
        # Treat only interp==1 as synthesized points; 0/-1 are real detections.
        g_real = g[g["interp"] != 1].sort_values("frame")
        if len(g_real) < 2:
            descriptors[int(tid)] = {"stitchable": False}
            pbar.update(1)
            continue
        t_start = int(g_real["frame"].iloc[0])
        t_end = int(g_real["frame"].iloc[-1])
        start_row = g_real.iloc[0]
        end_row = g_real.iloc[-1]
        vx, vy = _estimate_velocity(
            g_real["frame"].to_numpy(),
            g_real["cx"].to_numpy(),
            g_real["cy"].to_numpy(),
            vel_frames,
        )
        descriptors[int(tid)] = {
            "stitchable": True,
            "track": int(tid),
            "cls": int(end_row["cls"]) if "cls" in g_real.columns else -1,
            "t_start": t_start,
            "t_end": t_end,
            "start_c": (float(start_row["cx"]), float(start_row["cy"])),
            "end_c": (float(end_row["cx"]), float(end_row["cy"])),
            "start_box": (
                float(start_row["x"]),
                float(start_row["y"]),
                float(start_row["w"]),
                float(start_row["h"]),
            ),
            "end_box": (
                float(end_row["x"]),
                float(end_row["y"]),
                float(end_row["w"]),
                float(end_row["h"]),
            ),
            "area_end": max(float(end_row["area"]), 1.0),
            "vx": vx,
            "vy": vy,
        }
        pbar.update(1)
    pbar.close()

    stitchable = [d for d in descriptors.values() if d.get("stitchable", False)]
    if len(stitchable) <= 1:
        out = df.drop(columns=["cx", "cy", "area"])
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out

    ends = sorted(stitchable, key=lambda d: (d["t_end"], d["track"]))
    starts = sorted(stitchable, key=lambda d: (d["t_start"], d["track"]))
    n_end, n_start = len(ends), len(starts)
    inf = 1e9
    cost = np.full((n_end, n_start), inf, dtype=float)

    for i, a in enumerate(ends):
        for j, b in enumerate(starts):
            if a["track"] == b["track"]:
                continue
            dt = b["t_start"] - a["t_end"]
            if dt < 1 or dt > max_gap:
                continue
            if a["cls"] != b["cls"]:
                continue
            wi, hi = max(a["end_box"][2], 1.0), max(a["end_box"][3], 1.0)
            wj, hj = max(b["start_box"][2], 1.0), max(b["start_box"][3], 1.0)
            w_ratio, h_ratio = wj / wi, hj / hi
            if not (1.0 / size_ratio_max <= w_ratio <= size_ratio_max):
                continue
            if not (1.0 / size_ratio_max <= h_ratio <= size_ratio_max):
                continue

            pred_cx = a["end_c"][0] + a["vx"] * dt
            pred_cy = a["end_c"][1] + a["vy"] * dt
            sx, sy = b["start_c"]
            dist = float(np.hypot(pred_cx - sx, pred_cy - sy))
            dist_thr = dist_mult * np.sqrt(a["area_end"]) * (1.0 + (0.03 * dt))
            if dist >= dist_thr:
                continue

            pred_box = (pred_cx - (wi / 2.0), pred_cy - (hi / 2.0), wi, hi)
            iou = _iou_xywh(pred_box, b["start_box"])
            if iou < iou_min:
                continue

            dist_norm = dist / (np.sqrt(a["area_end"]) + 1e-6)
            iou_cost = 1.0 - iou
            size_cost = abs(np.log(max(w_ratio, 1e-6))) + abs(np.log(max(h_ratio, 1e-6)))
            c = (w_d * dist_norm) + (w_iou * iou_cost) + (w_s * size_cost)
            cost[i, j] = c

    matches: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci, strict=True):
            if cost[r, c] < inf:
                matches.append((r, c))
    except Exception:
        used_r: set[int] = set()
        used_c: set[int] = set()
        finite_pairs = np.argwhere(cost < inf)
        finite_pairs = sorted(finite_pairs, key=lambda rc: float(cost[rc[0], rc[1]]))
        for r, c in finite_pairs:
            r_i, c_i = int(r), int(c)
            if r_i in used_r or c_i in used_c:
                continue
            used_r.add(r_i)
            used_c.add(c_i)
            matches.append((r_i, c_i))

    dsu = _DSU([int(d["track"]) for d in stitchable])
    for r, c in matches:
        a_tid = int(ends[r]["track"])
        b_tid = int(starts[c]["track"])
        dsu.union(a_tid, b_tid)

    comps: dict[int, list[int]] = {}
    for d in stitchable:
        tid = int(d["track"])
        root = dsu.find(tid)
        comps.setdefault(root, []).append(tid)

    tstart_by_tid = {int(d["track"]): int(d["t_start"]) for d in stitchable}
    rep_map: dict[int, int] = {}
    for members in comps.values():
        rep = min(members, key=lambda t: (tstart_by_tid.get(t, 10**9), t))
        for t in members:
            rep_map[t] = rep

    all_tids = df["track"].astype(int).unique().tolist()
    for tid in all_tids:
        rep_map.setdefault(int(tid), int(tid))

    out = df.copy()
    out["track"] = out["track"].astype(int).map(rep_map).astype(int)
    out = out.drop(columns=["cx", "cy", "area"]).sort_values(["frame", "track"]).reset_index(drop=True)

    if output_file:
        out.to_csv(output_file, index=False, header=False)
    return out
