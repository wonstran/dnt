"""Post-process utilities for tracked trajectories."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from tqdm import tqdm


def interpolate_tracks_rts(
    tracks: pd.DataFrame | None = None,
    track_file: str | None = None,
    output_file: str | None = None,
    col_names: list[str] | None = None,
    fill_gaps_only: bool = True,
    smooth_existing: bool = False,
    process_var: float = 10.0,
    meas_var_pos: float = 25.0,
    meas_var_size: float = 16.0,
    min_track_len: int = 2,
    max_gap: int = 30,
    add_interp_flag: bool = True,
    interp_col: str = "interp",
    verbose: bool = True,
    video_index: int | None = None,
    video_tot: int | None = None,
) -> pd.DataFrame:
    """Interpolate trajectory gaps in each track chain using RTS smoothing.

    Applies a constant-velocity Kalman filter per track on bounding box center
    and size states, then runs Rauch-Tung-Striebel (RTS) smoothing from FilterPy
    to produce smooth, continuous trajectories. Missing frames are interpolated
    with velocity estimates.

    Parameters
    ----------
    tracks : pd.DataFrame, optional
        Input track data with columns at minimum: frame, track, x, y, w, h.
        May also contain cls, score, and other columns which are preserved.
        If None, ``track_file`` is used.
    track_file : str, optional
        CSV file path to read tracks from when ``tracks`` is None.
    output_file : str, optional
        CSV file path to write the interpolated results.
    col_names : list[str], optional
        Column names to apply when input columns are positional integers.
        Default is ["frame","track","x","y","w","h","score","cls","r3","r4"].
    fill_gaps_only : bool, optional
        If True (default), only interpolate frames without observations.
        If False, also smooth observed frames.
    smooth_existing : bool, optional
        If True, apply smoothed state to observed frames. Only used when
        fill_gaps_only is True. Default is False.
    process_var : float, optional
        Process noise variance for Kalman filter. Controls model uncertainty.
        Default is 10.0.
    meas_var_pos : float, optional
        Measurement noise variance for position (cx, cy). Default is 25.0.
    meas_var_size : float, optional
        Measurement noise variance for size (w, h). Default is 16.0.
    min_track_len : int, optional
        Minimum track length to apply interpolation. Tracks shorter than this
        are returned as-is. Default is 2.
    max_gap : int, optional
        Maximum number of consecutive missing frames allowed to interpolate
        within a track chain. Gaps larger than this value are not filled.
        Default is 30.
    add_interp_flag : bool, optional
        If True (default), add column with interpolation flags (0=observed, 1=interpolated).
    interp_col : str, optional
        Name of the interpolation flag column. Default is "interp".
    verbose : bool, optional
        If True, show tqdm progress bar over tracks. Default is True.
    video_index : int, optional
        Current video index for progress description. Default is None.
    video_tot : int, optional
        Total videos for progress description. Default is None.

    Returns
    -------
    pd.DataFrame
        Output tracks with interpolated frames. Columns include all input
        columns plus interp_col if add_interp_flag is True. Frame indices are
        continuous within each track after interpolation.

    Raises
    ------
    ValueError
        If tracks has fewer than 6 columns (when columns are not named).

    Notes
    -----
    The Kalman filter uses an 8-state constant-velocity model:
        [cx, vx, cy, vy, w, vw, h, vh]
    where (cx, cy) is bounding box center, (w, h) is size, and
    (vx, vy, vw, vh) are their velocities.

    Input coordinates assume [x, y, w, h] format where x, y is top-left corner.
    These are converted to center coordinates for Kalman processing.

    Frame gaps within tracks are filled by interpolation. If a track has
    missing frames between observations, the filter predicts values for those
    frames based on velocity estimates from nearby observations.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample track with gaps
    >>> tracks = pd.DataFrame({
    ...     'frame': [0, 1, 5, 6],
    ...     'track': [1, 1, 1, 1],
    ...     'x': [10.0, 12.0, 20.0, 22.0],
    ...     'y': [20.0, 22.0, 30.0, 32.0],
    ...     'w': [100.0, 100.0, 100.0, 100.0],
    ...     'h': [50.0, 50.0, 50.0, 50.0],
    ... })
    >>> result = interpolate_tracks_rts(tracks, fill_gaps_only=True)
    >>> print(result[['frame', 'track', 'interp']])  # Shows interpolated frames

    """
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import KalmanFilter, rts_smoother

    if col_names is None:
        col_names = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]

    if tracks is None:
        if not track_file:
            raise ValueError("Either `tracks` or `track_file` must be provided.")
        tracks = pd.read_csv(track_file)

    if len(tracks) == 0:
        out = tracks.copy()
        if output_file:
            out.to_csv(output_file, index=False)
        return out

    df = tracks.copy()

    # Support both positional and named-column track tables.
    required = ["frame", "track", "x", "y", "w", "h"]
    if all(c in df.columns for c in required):
        work = df.copy()
    else:
        if len(df.columns) < len(required):
            raise ValueError("tracks must include at least frame/track/x/y/w/h columns.")
        renamed = col_names[: len(df.columns)]
        work = df.copy()
        work.columns = renamed

    work = work.sort_values(["track", "frame"]).reset_index(drop=True)
    output_rows: list[dict] = []
    grouped = list(work.groupby("track", sort=False))

    pbar = tqdm(total=len(grouped), unit=" tracks", disable=not verbose)
    if verbose:
        if video_index is not None and video_tot is not None:
            pbar.set_description_str(f"RTS interpolate {video_index} of {video_tot}")
        else:
            pbar.set_description_str("RTS interpolate")

    for track_id, g in grouped:
        g = g.sort_values("frame").drop_duplicates("frame", keep="first").reset_index(drop=True)
        if len(g) < min_track_len:
            rows = g.to_dict("records")
            if add_interp_flag:
                for r in rows:
                    if "r3" in g.columns:
                        r["r3"] = 0
                    else:
                        r[interp_col] = 0
            output_rows.extend(rows)
            pbar.update(1)
            continue

        frames_obs = g["frame"].astype(int).to_numpy()
        frame_start = int(frames_obs.min())
        frame_end = int(frames_obs.max())
        frames_full = np.arange(frame_start, frame_end + 1, dtype=int)
        observed_set = set(frames_obs.tolist())
        fillable_missing: set[int] = set()
        for f0, f1 in pairwise(frames_obs):
            gap = int(f1 - f0 - 1)
            if 0 < gap <= max_gap:
                fillable_missing.update(range(int(f0) + 1, int(f1)))

        cx = (g["x"].astype(float) + (g["w"].astype(float) / 2.0)).to_numpy()
        cy = (g["y"].astype(float) + (g["h"].astype(float) / 2.0)).to_numpy()
        ww = g["w"].astype(float).to_numpy()
        hh = g["h"].astype(float).to_numpy()
        z_map = {int(f): np.array([cx[i], cy[i], ww[i], hh[i]], dtype=float) for i, f in enumerate(frames_obs)}
        row_map = {int(row["frame"]): row for row in g.to_dict("records")}

        # State: [cx, vx, cy, vy, w, vw, h, vh]
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        q2 = Q_discrete_white_noise(dim=2, dt=1.0, var=process_var)
        kf.Q = np.zeros((8, 8), dtype=float)
        for i in range(4):
            i0 = i * 2
            kf.Q[i0 : i0 + 2, i0 : i0 + 2] = q2
        kf.R = np.diag([meas_var_pos, meas_var_pos, meas_var_size, meas_var_size]).astype(float)
        kf.P = np.eye(8, dtype=float) * 100.0
        z0 = z_map[frame_start]
        kf.x = np.array([z0[0], 0.0, z0[1], 0.0, z0[2], 0.0, z0[3], 0.0], dtype=float)

        xs, ps, fs, qs = [], [], [], []
        for f in frames_full:
            kf.predict()
            z = z_map.get(int(f))
            if z is not None:
                kf.update(z)
            xs.append(kf.x.copy())
            ps.append(kf.P.copy())
            fs.append(kf.F.copy())
            qs.append(kf.Q.copy())

        xs_s, _, _, _ = rts_smoother(np.asarray(xs), np.asarray(ps), np.asarray(fs), np.asarray(qs))

        if "cls" in g.columns and len(g["cls"].dropna()) > 0:
            cls_mode = g["cls"].mode()
            cls_fill = float(cls_mode.iloc[0]) if len(cls_mode) > 0 else -1
        else:
            cls_fill = -1
        score_fill = float(g["score"].mean()) if "score" in g.columns and len(g["score"].dropna()) > 0 else -1.0

        for i, frame in enumerate(frames_full.tolist()):
            sm_cx = float(xs_s[i, 0])
            sm_cy = float(xs_s[i, 2])
            sm_w = max(1.0, float(xs_s[i, 4]))
            sm_h = max(1.0, float(xs_s[i, 6]))
            sm_x = sm_cx - (sm_w / 2.0)
            sm_y = sm_cy - (sm_h / 2.0)

            if frame in observed_set:
                row = dict(row_map[frame])
                if smooth_existing or (not fill_gaps_only):
                    row["x"] = sm_x
                    row["y"] = sm_y
                    row["w"] = sm_w
                    row["h"] = sm_h
                if add_interp_flag:
                    if "r3" in g.columns:
                        row["r3"] = 0
                    else:
                        row[interp_col] = 0
                output_rows.append(row)
            else:
                if frame not in fillable_missing:
                    continue
                row = {c: np.nan for c in g.columns}
                row["frame"] = frame
                row["track"] = track_id
                row["x"] = sm_x
                row["y"] = sm_y
                row["w"] = sm_w
                row["h"] = sm_h
                if "cls" in g.columns:
                    row["cls"] = cls_fill
                if "score" in g.columns:
                    row["score"] = score_fill
                if add_interp_flag:
                    if "r3" in g.columns:
                        row["r3"] = 1
                    else:
                        row[interp_col] = 1
                output_rows.append(row)
        pbar.update(1)

    pbar.close()

    out = pd.DataFrame(output_rows)
    if "r3" in out.columns:
        cols = list(out.columns)
        idx = cols.index("r3")
        out.rename(columns={"r3": interp_col}, inplace=True)
        cols[idx] = interp_col
        out = out[cols]

    # Keep compatibility with legacy track file readers that enforce integer dtypes.
    int_cols = ["frame", "track", "x", "y", "w", "h", "cls", "r4", interp_col]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].fillna(-1).round().astype(int)
    if "score" in out.columns:
        out["score"] = out["score"].fillna(-1).astype(float)

    out = out.sort_values(["frame", "track"]).reset_index(drop=True)
    if output_file:
        out.to_csv(output_file, index=False, header=False)
    return out


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
