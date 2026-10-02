"""Stage 1: ID-switch proposals (spec 6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig, to_frames
from .events import Event, EventKind
from .features import Appearance, dense_track_embeddings, track_embeddings
from .primitives import iou_matrix, kalman_nis, ramp

STAGE = "switch"


@dataclass
class SwitchResult:
    """Stage 1 proposals plus the weaker cut points screening uses (spec 6.2)."""

    events: list[Event]
    weak_cuts: dict[int, list[int]] = field(default_factory=dict)
    candidates: dict[int, list[int]] = field(default_factory=dict)


def contact_flags(work: pd.DataFrame, thr: float) -> pd.Series:
    """Return True where a row's box has IoU > ``thr`` with another track's box that frame."""
    flags = pd.Series(False, index=work.index)
    for _, g in work.groupby("frame"):
        if len(g) < 2:
            continue
        boxes = g[["x", "y", "w", "h"]].to_numpy(float)
        m = iou_matrix(boxes, boxes)
        np.fill_diagonal(m, 0.0)
        flags.loc[g.index] = m.max(axis=1) > thr
    return flags


def _window_any(frames: np.ndarray, flags: np.ndarray, delta: int) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(flags.astype(int))])
    lo = np.searchsorted(frames, frames - delta, side="left")
    hi = np.searchsorted(frames, frames + delta, side="right")
    return (cs[hi] - cs[lo]) > 0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _side_means(ef, emb, t, w):
    b = emb[(ef >= t - w) & (ef < t)]
    a = emb[(ef >= t) & (ef < t + w)]
    if not len(b) or not len(a):
        return None
    return _unit(b.mean(axis=0)), _unit(a.mean(axis=0))


def _appearance_change(frames, ef, emb, w) -> np.ndarray:
    out = np.full(len(frames), np.nan)
    for i, t in enumerate(frames):
        m = _side_means(ef, emb, t, w)
        if m is not None:
            out[i] = 1.0 - float(m[0] @ m[1])
    return out


def _two_means(emb: np.ndarray, iters: int = 20) -> np.ndarray:
    centers = np.stack([emb[0], emb[-1]]).astype(float)
    labels = np.full(len(emb), -1)
    for _ in range(iters):
        new = np.argmax(emb @ centers.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                centers[k] = _unit(emb[labels == k].mean(axis=0))
    return labels


_SILHOUETTE_MAX = 600


def _silhouette(emb: np.ndarray, labels: np.ndarray) -> float:
    if len(emb) > _SILHOUETTE_MAX:  # bounds the n x n distance matrix; deterministic subsample
        pick = np.linspace(0, len(emb) - 1, _SILHOUETTE_MAX).astype(int)
        emb, labels = emb[pick], labels[pick]
    d = 1.0 - emb @ emb.T
    vals = []
    for i in range(len(emb)):
        same = labels == labels[i]
        same[i] = False
        other = labels != labels[i]
        if not same.any() or not other.any():
            continue
        a, b = d[i, same].mean(), d[i, other].mean()
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def _bimodal_split(ef, emb, purity: float):
    """Return ``(k, silhouette)`` where sample ``k`` starts the second cluster, or None.

    Two-means clusters must split in time (spec 6.1): for the best cut ``k`` and orientation,
    at least ``purity`` of the first cluster's samples lie before ``k`` and at least ``purity``
    of the second cluster's samples lie at or after it (cluster recall).
    """
    n = len(ef)
    if n < 4:
        return None
    labels = _two_means(emb)
    if labels.min() == labels.max():
        return None
    ones = np.concatenate([[0], np.cumsum(labels == 1)])
    zeros = np.concatenate([[0], np.cumsum(labels == 0)])
    tot = {0: int(zeros[-1]), 1: int(ones[-1])}
    cum = {0: zeros, 1: ones}
    ks = np.arange(1, n)
    best = (-1.0, 0)
    for o in (0, 1):
        other = 1 - o
        r_first = cum[o][ks] / tot[o]
        r_second = (tot[other] - cum[other][ks]) / tot[other]
        score = np.minimum(r_first, r_second)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), int(ks[j]))
    if best[0] < purity:
        return None
    return best[1], _silhouette(emb, labels)


def _covered_seconds(samples: np.ndarray, fps: float, motion_only: bool) -> float:
    """Amount of clean data in seconds: observed frames (motion-only) or samples x stride."""
    n = len(samples)
    if n == 0:
        return 0.0
    if motion_only:
        return n / fps
    stride = 1.0 if n == 1 else max(1.0, float(np.median(np.diff(samples))))
    return n * stride / fps


def _score_track(g, contact, cfg: RefineConfig, fps, appearance, lin, delta, w):
    sc, mc = cfg.switch, cfg.motion
    frames = g["frame"].to_numpy(int)
    boxes = g[["x", "y", "w", "h"]].to_numpy(float)
    cls = g["cls"].to_numpy(int)
    n = len(frames)
    motion_only = appearance is None
    if motion_only:
        ef, emb = np.empty(0, dtype=int), np.empty((0, 0))
    else:
        ef, emb = track_embeddings(appearance, lin)
    samples = frames if motion_only else ef
    if n < 3 or len(samples) == 0:
        return None
    if _covered_seconds(samples, fps, motion_only) < 2 * sc.min_side_seconds:
        return None
    nis = kalman_nis(
        frames,
        boxes,
        process_var=mc.process_var,
        meas_var_pos=mc.meas_var_pos,
        meas_var_size=mc.meas_var_size,
    )
    jump = np.zeros(n)
    ratio = np.maximum(boxes[1:, 2:4], 1e-6) / np.maximum(boxes[:-1, 2:4], 1e-6)
    jump[1:] = np.abs(np.log(ratio)).max(axis=1)
    mot = np.maximum(
        np.nan_to_num(ramp(nis, sc.nis_hi / 2.0, sc.nis_hi)), ramp(jump, *sc.ramps["jump"])
    )
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(frames) > 1
    change = np.zeros(n, dtype=bool)
    change[1:] = cls[1:] != cls[:-1]
    fired = {
        "contact": _window_any(frames, contact, delta),
        "gap": _window_any(frames, gap, delta),
        "size_jump": _window_any(frames, jump >= np.log(sc.size_gate), delta),
    }
    if sc.class_change_gate:
        fired["class_change"] = _window_any(frames, change, delta)
    gate = np.zeros(n, dtype=bool)
    for v in fired.values():
        gate |= v
    a_raw = np.zeros(n)
    app = np.zeros(n)
    z = np.full(n, np.nan)
    bim = np.zeros(n)
    sil = None
    med = mad = float("nan")
    if not motion_only:
        change_a = _appearance_change(frames, ef, emb, w)
        fin = np.isfinite(change_a)
        if fin.any():
            med = float(np.median(change_a[fin]))
            mad = float(np.median(np.abs(change_a[fin] - med)))
            z = (change_a - med) / (1.4826 * mad + sc.mad_floor)
            app = np.nan_to_num(ramp(z, *sc.ramps["z_app"]))
        a_raw = np.nan_to_num(change_a)
        split = _bimodal_split(ef, emb, sc.bimodal_purity)
        if split is not None and split[1] >= sc.bimodal_silhouette_min:
            k, sil = split
            after = np.flatnonzero(frames > ef[k - 1])
            if len(after):
                i = int(after[0])
                bim[i] = ramp(sil, *sc.ramps["silhouette"])
                app[i] = max(app[i], bim[i])
        score = gate * (sc.w_app * app + sc.w_mot * mot)
    else:
        score = np.minimum(gate * mot, sc.motion_only_cap)
    score[0] = 0.0
    return {
        "frames": frames,
        "S": score,
        "A": a_raw,
        "app": app,
        "z": z,
        "bim": bim,
        "sil": sil,
        "mot": mot,
        "nis": nis,
        "jump": jump,
        "fired": fired,
        "samples": samples,
        "motion_only": motion_only,
        "ef": ef,
        "emb": emb,
        "med": med,
        "mad": mad,
    }


def _candidates(info, fps, cfg: RefineConfig, nms: int) -> list[int]:
    sc = cfg.switch
    s, frames = info["S"], info["frames"]
    n = len(s)
    left = np.concatenate([[-np.inf], s[:-1]])
    right = np.concatenate([s[1:], [-np.inf]])
    peaks = [i for i in range(1, n) if s[i] > 0 and s[i] >= left[i] and s[i] >= right[i]]
    peaks.sort(key=lambda i: (-s[i], -info["A"][i], -info["mot"][i], frames[i]))
    kept: list[int] = []
    for i in peaks:
        if all(abs(int(frames[i]) - int(frames[j])) >= nms for j in kept):
            kept.append(i)
    samples = info["samples"]
    out = []
    for i in kept:
        t = frames[i]
        before, after = samples[samples < t], samples[samples >= t]
        if _covered_seconds(before, fps, info["motion_only"]) < sc.min_side_seconds:
            continue
        if _covered_seconds(after, fps, info["motion_only"]) < sc.min_side_seconds:
            continue
        out.append(i)
    return sorted(out)


def _pair_contact(work, ti, tj, t, delta, thr) -> bool:
    win = work[work["frame"].between(t - delta, t + delta)]
    a = win[win["track"] == ti]
    b = win[win["track"] == tj]
    m = a.merge(b, on="frame", suffixes=("_a", "_b"))
    for row in m.itertuples(index=False):
        box_a = [[row.x_a, row.y_a, row.w_a, row.h_a]]
        box_b = [[row.x_b, row.y_b, row.w_b, row.h_b]]
        if iou_matrix(box_a, box_b)[0, 0] > thr:
            return True
    return False


def _lineage_windows(lin, f0: int, f1: int) -> list[tuple[int, int, int]]:
    """Return the ``(raw_id, lo, hi)`` pieces of ``[f0, f1]`` that lie inside the lineage."""
    out = []
    for r, a, b in lin:
        lo, hi = max(int(a), f0), min(int(b), f1)
        if lo <= hi:
            out.append((int(r), lo, hi))
    return out


def _span(info, i: int, radius: int, w: int) -> tuple[int, int, int, int]:
    """Rows ``lo..hi`` around candidate row ``i``, and the frame range their sides need."""
    fr = info["frames"]
    lo, hi = max(1, i - radius), min(len(fr) - 1, i + radius)
    return lo, hi, int(fr[lo]) - w, int(fr[hi]) + w - 1


def _relocate(info, i: int, sc, lo: int, hi: int, w: int, ef_d, emb_d, eligible) -> int:
    """Rescore rows ``lo..hi`` on dense samples, store the best eligible row's values, return it.

    The best row has the highest ``S``, then the highest raw appearance change (``S`` saturates
    for strong changes), then the smallest distance from ``i``, then the earliest frame.
    ``eligible(j)`` says whether a row may hold the cut; the original row ``i`` always may.
    """
    if not np.isfinite(info["med"]):
        return i
    best = None
    for j in range(lo, hi + 1):
        if not eligible(j):
            continue
        m = _side_means(ef_d, emb_d, int(info["frames"][j]), w)
        if m is None:
            continue
        a = 1.0 - float(m[0] @ m[1])
        z = (a - info["med"]) / (1.4826 * info["mad"] + sc.mad_floor)
        app = max(float(ramp(z, *sc.ramps["z_app"])), float(info["bim"][j]))
        gate = any(bool(v[j]) for v in info["fired"].values())
        s = (sc.w_app * app + sc.w_mot * float(info["mot"][j])) if gate else 0.0
        key = (s, a, -abs(j - i), -j)
        if best is None or key > best[0]:
            best = (key, j, a, z, app, s)
    if best is None:
        return i
    _, j, a, z, app, s = best
    info["A"][j], info["z"][j], info["app"][j], info["S"][j] = a, z, app, s
    return j


def _sides_ok(info, t: int, fps: float, min_side: float) -> bool:
    """Return whether both sides of a cut at frame ``t`` keep ``min_side`` seconds of samples."""
    s = info["samples"]
    return (
        _covered_seconds(s[s < t], fps, info["motion_only"]) >= min_side
        and _covered_seconds(s[s >= t], fps, info["motion_only"]) >= min_side
    )


def _merge_dense(ef, emb, parts):
    """Merge dense ``(frames, embeddings)`` parts into a track's samples: sorted, one per frame."""
    fs, es = [], []
    if len(ef):
        fs.append(ef)
        es.append(emb)
    for f, e in parts:
        if len(f):
            fs.append(f)
            es.append(e)
    if not any(len(f) for f, _ in parts):
        return ef, emb
    f = np.concatenate(fs)
    e = np.vstack(es)
    order = np.argsort(f, kind="stable")
    f, e = f[order], e[order]
    keep = np.concatenate([[True], np.diff(f) > 0])
    return f[keep], e[keep]


def _dense_rescore(appearance, infos, cands, cfg: RefineConfig, w: int, fps: float, nms: int):
    """Rescore each candidate on dense embeddings and move it to its best row (spec 5.3).

    A move must keep the rules that chose the candidate: ``switch.min_side_seconds`` of samples
    on each side, and ``nms`` frames from every other candidate of the track.
    """
    if appearance is None or not hasattr(appearance, "dense_embeddings"):
        return
    sc = cfg.switch
    radius = max(1, int(cfg.encoder.sample_every))
    wanted = []
    for tid, idx in cands.items():
        info, lin = infos[tid]
        for i in idx:
            _, _, f0, f1 = _span(info, i, radius, w)
            wanted += _lineage_windows(lin, f0, f1)
    prefetch = getattr(appearance, "prefetch_dense", None)
    if prefetch is not None and wanted:
        prefetch(wanted)
    for tid, idx in cands.items():
        info, lin = infos[tid]
        fr = info["frames"]
        order = sorted(
            idx,
            key=lambda i, info=info, fr=fr: (
                -float(info["S"][i]),
                -float(info["A"][i]),
                -float(info["mot"][i]),
                int(fr[i]),
            ),
        )
        pos = {i: int(fr[i]) for i in idx}  # candidate -> its current frame
        moved, parts = [], []
        for i in order:
            lo, hi, f0, f1 = _span(info, i, radius, w)
            ef_d, emb_d = dense_track_embeddings(appearance, lin, f0, f1)
            others = [f for k, f in pos.items() if k != i]

            def eligible(j, info=info, fr=fr, others=others):
                t = int(fr[j])
                return all(abs(t - o) >= nms for o in others) and _sides_ok(
                    info, t, fps, sc.min_side_seconds
                )

            j = _relocate(info, i, sc, lo, hi, w, ef_d, emb_d, eligible)
            pos[i] = int(fr[j])
            moved.append(j)
            parts.append((ef_d, emb_d))
        cands[tid] = sorted(set(moved))
        info["ef"], info["emb"] = _merge_dense(info["ef"], info["emb"], parts)


def propose_splits(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, appearance: Appearance | None = None
) -> SwitchResult:
    """Propose SPLIT events at likely ID switches (spec 6.1).

    Every observed frame of a track is scored from an appearance change, a motion break and an
    opportunity gate. Local maxima survive non-maximum suppression when both sides have enough
    samples. With an appearance provider that has ``dense_embeddings``, each candidate found on
    the coarse samples is rescored on dense samples and moved to its best frame (spec 5.3). Swap
    pairs (two contacting tracks whose tails cross) get a score boost. Without an appearance
    provider the score is motion only and capped. Tracks are never edited.

    Parameters
    ----------
    work : pandas.DataFrame
        Work table from ``io.to_work``.
    cfg : RefineConfig
        Refinement configuration; ``cfg.switch`` and ``cfg.motion`` hold the thresholds.
    fps : float
        Video frame rate, used to convert seconds to frames.
    appearance : Appearance, optional
        Embedding provider; ``None`` runs motion-only.

    Returns
    -------
    SwitchResult
        Proposed SPLIT events, weak cut frames and all surviving candidate frames per track.

    """
    sc = cfg.switch
    result = SwitchResult(events=[])
    if work.empty:
        return result
    contact = contact_flags(work, sc.contact_iou)
    delta = to_frames(sc.delta, fps)
    w = to_frames(sc.window, fps)
    nms = to_frames(sc.nms_seconds, fps)
    infos: dict[int, tuple[dict, list]] = {}
    cands: dict[int, list[int]] = {}
    for tid, g in work.groupby("track", sort=True):
        g = g.sort_values("frame")
        lin = lineage_of_rows(g)
        info = _score_track(
            g, contact.loc[g.index].to_numpy(bool), cfg, fps, appearance, lin, delta, w
        )
        if info is None:
            continue
        infos[int(tid)] = (info, lin)
        cands[int(tid)] = _candidates(info, fps, cfg, nms)
    _dense_rescore(appearance, infos, cands, cfg, w, fps, nms)
    for tid, idx in cands.items():
        if idx:
            result.candidates[tid] = [int(infos[tid][0]["frames"][i]) for i in idx]

    boost: dict[tuple[int, int], tuple[float, int, float]] = {}
    if appearance is not None:
        flat = [(t, i) for t, idx in cands.items() for i in idx]
        for a in range(len(flat)):
            for b in range(a + 1, len(flat)):
                (ti, ii), (tj, jj) = flat[a], flat[b]
                if ti == tj:
                    continue
                fi = int(infos[ti][0]["frames"][ii])
                fj = int(infos[tj][0]["frames"][jj])
                if abs(fi - fj) > delta:
                    continue
                if not _pair_contact(work, ti, tj, (fi + fj) // 2, delta, sc.contact_iou):
                    continue
                mi = _side_means(infos[ti][0]["ef"], infos[ti][0]["emb"], fi, w)
                mj = _side_means(infos[tj][0]["ef"], infos[tj][0]["emb"], fj, w)
                if mi is None or mj is None:
                    continue
                (bi, ai), (bj, aj) = mi, mj
                cross = float(bi @ aj - bi @ ai + bj @ ai - bj @ aj)
                if cross > 0:
                    inc = sc.swap_boost * ramp(cross, *sc.ramps["cross"])
                    for key, other in (((ti, ii), tj), ((tj, jj), ti)):
                        old = boost.get(key)
                        if old is None or (inc, cross, -other) > (old[0], old[2], -old[1]):
                            boost[key] = (inc, other, cross)

    for tid, idx in cands.items():
        info, lin = infos[tid]
        for i in idx:
            s = float(info["S"][i])
            extra = {}
            if (tid, i) in boost:
                inc, other, cross = boost[(tid, i)]
                s = min(1.0, s + inc)
                extra = {"swap_with": int(other), "swap_cross": cross, "swap_boost": inc}
            t = int(info["frames"][i])
            if s >= sc.reject_below:
                signals = {
                    "app": float(info["app"][i]),
                    "z_app": float(info["z"][i]),
                    "bimodal": float(info["bim"][i]),
                    "silhouette": info["sil"],
                    "mot": float(info["mot"][i]),
                    "nis": float(info["nis"][i]),
                    "jump": float(info["jump"][i]),
                    "gate": [k for k, v in info["fired"].items() if v[i]],
                    "motion_only": info["motion_only"],
                    **extra,
                }
                result.events.append(
                    Event.propose(
                        stage=STAGE,
                        kind=EventKind.SPLIT,
                        tracks=[tid],
                        lineage=[lin],
                        frames=(t, t),
                        params={"cut_frame": t},
                        algo_score=s,
                        signals=signals,
                    )
                )
            elif s >= cfg.screen.segment_at:
                result.weak_cuts.setdefault(tid, []).append(t)
    return result
