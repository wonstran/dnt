"""Kalman RTS gap filling for track tables (spec 6.4).

Moved from ``dnt.track.post_process`` (which re-exports it) in dnt 0.4.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from tqdm import tqdm

from .primitives import cv_kalman

DEFAULT_COL_NAMES = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]


def _set_flag(row: dict, columns, add_interp_flag: bool, interp_col: str, value: int) -> None:
    if not add_interp_flag:
        return
    if "r3" in columns:
        row["r3"] = value
    else:
        row[interp_col] = value


def _rts_rows(
    g: pd.DataFrame,
    track_id,
    *,
    fill_gaps_only: bool,
    smooth_existing: bool,
    process_var: float,
    meas_var_pos: float,
    meas_var_size: float,
    max_gap: int,
    add_interp_flag: bool,
    interp_col: str,
) -> list[dict]:
    """Smooth one run of observed rows (sorted, unique frames) and return output rows."""
    from filterpy.kalman import rts_smoother

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
    z_map = {
        int(f): np.array([cx[i], cy[i], ww[i], hh[i]], dtype=float)
        for i, f in enumerate(frames_obs)
    }
    row_map = {int(row["frame"]): row for row in g.to_dict("records")}

    kf = cv_kalman(process_var, meas_var_pos, meas_var_size)
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
    has_score = "score" in g.columns and len(g["score"].dropna()) > 0
    score_fill = float(g["score"].mean()) if has_score else -1.0

    rows: list[dict] = []
    for i, frame in enumerate(frames_full.tolist()):
        sm_w = max(1.0, float(xs_s[i, 4]))
        sm_h = max(1.0, float(xs_s[i, 6]))
        sm_x = float(xs_s[i, 0]) - (sm_w / 2.0)
        sm_y = float(xs_s[i, 2]) - (sm_h / 2.0)
        if frame in observed_set:
            row = dict(row_map[frame])
            if smooth_existing or (not fill_gaps_only):
                row["x"], row["y"], row["w"], row["h"] = sm_x, sm_y, sm_w, sm_h
            _set_flag(row, g.columns, add_interp_flag, interp_col, 0)
            rows.append(row)
        elif frame in fillable_missing:
            row = {c: np.nan for c in g.columns}
            row["frame"] = frame
            row["track"] = track_id
            row["x"], row["y"], row["w"], row["h"] = sm_x, sm_y, sm_w, sm_h
            if "cls" in g.columns:
                row["cls"] = cls_fill
            if "score" in g.columns:
                row["score"] = score_fill
            _set_flag(row, g.columns, add_interp_flag, interp_col, 1)
            rows.append(row)
    return rows


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
    if col_names is None:
        col_names = list(DEFAULT_COL_NAMES)
    if tracks is None:
        if not track_file:
            raise ValueError("Either `tracks` or `track_file` must be provided.")
        try:
            tracks = pd.read_csv(track_file, header=None)
        except pd.errors.EmptyDataError:
            tracks = pd.DataFrame(columns=col_names)
    if len(tracks) == 0:
        out = tracks.copy()
        if output_file:
            out.to_csv(output_file, index=False, header=False)
        return out

    df = tracks.copy()
    required = ["frame", "track", "x", "y", "w", "h"]
    if all(c in df.columns for c in required):
        work = df.copy()
    else:
        if len(df.columns) < len(required):
            raise ValueError("tracks must include at least frame/track/x/y/w/h columns.")
        work = df.copy()
        work.columns = col_names[: len(df.columns)]
    work = work.sort_values(["track", "frame"]).reset_index(drop=True)

    output_rows: list[dict] = []
    grouped = list(work.groupby("track", sort=False))
    pbar = tqdm(total=len(grouped), unit=" tracks", disable=not verbose)
    if verbose:
        if video_index is not None and video_tot is not None:
            pbar.set_description_str(f"RTS interpolate {video_index} of {video_tot}")
        else:
            pbar.set_description_str("RTS interpolate")
    kw = {
        "fill_gaps_only": fill_gaps_only,
        "smooth_existing": smooth_existing,
        "process_var": process_var,
        "meas_var_pos": meas_var_pos,
        "meas_var_size": meas_var_size,
        "max_gap": max_gap,
        "add_interp_flag": add_interp_flag,
        "interp_col": interp_col,
    }
    for track_id, g in grouped:
        g = g.sort_values("frame").drop_duplicates("frame", keep="first").reset_index(drop=True)
        if len(g) < min_track_len:
            rows = g.to_dict("records")
            for r in rows:
                _set_flag(r, g.columns, add_interp_flag, interp_col, 0)
            output_rows.extend(rows)
        else:
            output_rows.extend(_rts_rows(g, track_id, **kw))
        pbar.update(1)
    pbar.close()

    out = pd.DataFrame(output_rows)
    if "r3" in out.columns:
        cols = list(out.columns)
        idx = cols.index("r3")
        out = out.rename(columns={"r3": interp_col})
        cols[idx] = interp_col
        out = out[cols]
    # Keep compatibility with legacy track file readers that enforce integer dtypes.
    for c in ["frame", "track", "x", "y", "w", "h", "cls", "r4", interp_col]:
        if c in out.columns:
            out[c] = out[c].fillna(-1).round().astype(int)
    if "score" in out.columns:
        out["score"] = out["score"].fillna(-1).astype(float)
    out = out.sort_values(["frame", "track"]).reset_index(drop=True)
    if output_file:
        out.to_csv(output_file, index=False, header=False)
    return out
