# src/dnt/refine/screen.py
"""Stage 2: false-track screening, and the orphan pass (spec 6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np
import pandas as pd

from .apply import lineage_of_rows
from .config import RefineConfig
from .events import Event, EventKind
from .hints import ReclassHint
from .primitives import box_centers, heading_smoothness, iou_matrix, ramp, speeds_hps

STAGE = "screen"
ORPHAN_STAGE = "orphan"


@dataclass
class ScreenContext:
    """Inputs screening needs besides the work table."""

    boxes: pd.DataFrame | None = None
    fmt: str | None = None
    hints: dict[int, ReclassHint] = field(default_factory=dict)
    split_raw_ids: set[int] = field(default_factory=set)


@dataclass
class _Unit:
    track: int
    frames: np.ndarray
    boxes: np.ndarray
    score: np.ndarray
    hmed: float
    v: np.ndarray
    vel: np.ndarray
    localized_hint: ReclassHint | None
    unlocalized_hint: ReclassHint | None
    area: float = 0.0
    unique_frames: bool = False


def _pixel_velocity(frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    c = box_centers(boxes)
    vel = np.zeros_like(c)
    if len(c) > 1:
        vel[1:] = np.diff(c, axis=0) / np.maximum(np.diff(frames), 1)[:, None]
        vel[0] = vel[1]
    return vel


def _unit(track, rows: pd.DataFrame, cfg, fps, localized=None, unlocalized=None) -> _Unit:
    frames = rows["frame"].to_numpy(int)
    boxes = rows[["x", "y", "w", "h"]].to_numpy(float)
    return _Unit(
        track=int(track), frames=frames, boxes=boxes, score=rows["score"].to_numpy(float),
        hmed=max(float(np.median(boxes[:, 3])), 1.0),
        v=speeds_hps(frames, boxes, fps, cfg.motion.height_window),
        vel=_pixel_velocity(frames, boxes), localized_hint=localized, unlocalized_hint=unlocalized,
        area=float(np.median(boxes[:, 2] * boxes[:, 3])),
        unique_frames=bool(np.all(np.diff(frames) > 0)),
    )


def _iob_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return IoB of each ``a`` box (its intersection over its own area) with each ``b`` box.

    ``a`` is ``(N, 4)`` and ``b`` is ``(M, 4)`` in ``[left, top, width, height]``; the result
    is ``(N, M)`` and equals ``primitives.iob_matrix(a, b)`` for boxes of non-negative size.
    """
    ax, ay = a[:, 0:1], a[:, 1:2]
    iw = np.minimum(ax + a[:, 2:3], b[:, 0] + b[:, 2]) - np.maximum(ax, b[:, 0])
    ih = np.minimum(ay + a[:, 3:4], b[:, 1] + b[:, 3]) - np.maximum(ay, b[:, 1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)
    area = (a[:, 2] * a[:, 3])[:, None]
    return np.divide(inter, area, out=np.zeros_like(inter), where=area != 0)


class _ContextIndex:
    """Context boxes per frame, with pixel velocities for track contexts."""

    def __init__(self, sctx: ScreenContext):
        """Index ``sctx.boxes`` by frame."""
        self.fmt = sctx.fmt
        self.present = sctx.boxes is not None
        self._by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._views: dict[frozenset, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        if not self.present or not len(sctx.boxes):
            return
        b = sctx.boxes.sort_values(["track", "frame"]).reset_index(drop=True)
        boxes = b[["x", "y", "w", "h"]].to_numpy(float)
        vel = np.zeros((len(b), 2))
        if self.fmt == "tracks":
            for _, g in b.groupby("track"):
                idx = g.index.to_numpy()
                vel[idx] = _pixel_velocity(g["frame"].to_numpy(int), boxes[idx])
        cls = b["cls"].to_numpy(int)
        for f, g in b.groupby("frame"):
            idx = g.index.to_numpy()
            self._by_frame[int(f)] = (boxes[idx], cls[idx], vel[idx])

    def view(self, classes) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return ``{frame: (boxes, velocities)}`` of context boxes of ``classes``.

        The per-class filtering happens once per class set; frames without a matching box are
        absent from the result.
        """
        key = frozenset(int(c) for c in classes)
        cached = self._views.get(key)
        if cached is None:
            wanted = list(key)
            cached = {}
            for f, (boxes, cls, vel) in self._by_frame.items():
                keep = np.isin(cls, wanted)
                if keep.any():
                    cached[f] = (boxes[keep], vel[keep])
            self._views[key] = cached
        return cached


_NO_BOXES = (np.empty((0, 4)), np.empty((0, 2)))


def _context_fraction(u: _Unit, ctx: _ContextIndex, classes, overlap: str, thr: float,
                      cfg: RefineConfig, fps: float) -> float:
    """Fraction of rows where a context box of ``classes`` overlaps ``u`` and moves with it."""
    overlap_fn = _iob_rows if overlap == "iob" else iou_matrix
    view = ctx.view(classes)
    hits = np.zeros(len(u.frames), dtype=bool)
    for i, f in enumerate(u.frames):
        entry = view.get(int(f))
        if entry is None:
            continue
        boxes, vel = entry
        ov = overlap_fn(u.boxes[i : i + 1], boxes)[0]
        ok = ov >= thr
        if not ok.any():
            continue
        if ctx.fmt == "tracks":
            dv = np.linalg.norm(vel[ok] - u.vel[i], axis=1) / u.hmed * fps
            hits[i] = bool((dv < cfg.screen.move_together).any())
        else:
            if i == 0 or not np.isfinite(u.v[i]) or u.v[i] < cfg.motion.moving_min:
                continue
            prev_boxes, _ = view.get(int(u.frames[i - 1]), _NO_BOXES)
            if not len(prev_boxes):
                continue
            prev_ok = prev_boxes[overlap_fn(u.boxes[i - 1 : i], prev_boxes)[0] >= thr]
            if len(prev_ok):
                persist = iou_matrix(boxes[ok], prev_ok) >= cfg.screen.persistence_iou
                hits[i] = bool(persist.any())
    return float(hits.mean()) if len(hits) else 0.0


def _static_center(u: _Unit, cfg: RefineConfig) -> tuple[float, float, np.ndarray]:
    c = box_centers(u.boxes)
    med = np.median(c, axis=0)
    r = float(np.percentile(np.linalg.norm(c - med, axis=1), 95) / u.hmed)
    return r, float(ramp(r, *cfg.screen.ramps["R"])), med


def _static(u: _Unit, cfg, fps, static_meds: tuple[np.ndarray, np.ndarray], cap: float):
    r = cfg.screen.ramps
    raw_r, r_ramp, med = _static_center(u, cfg)
    c = box_centers(u.boxes)
    if len(u.frames) > 1:
        step = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(u.frames), 1)
        jit = float(np.median(step) / u.hmed)
    else:
        jit = 0.0
    valid = u.score[u.score >= 0]
    conf = float(valid.mean()) if len(valid) else None
    dur = (u.frames[-1] - u.frames[0] + 1) / fps
    radius = cfg.screen.hotspot_radius * u.hmed
    ids, meds = static_meds
    near = int(np.count_nonzero((ids != u.track) & (np.linalg.norm(meds - med, axis=1) <= radius)))
    hot = near + (1 if r_ramp >= 0.5 else 0)
    parts = [ramp(jit, *r["J"]), ramp(hot, *r["H"])]
    if conf is not None:
        parts.append(ramp(conf, *r["C"]))
    score = min(cap, r_ramp * ramp(dur, *r["T"]) * float(np.mean(parts)))
    signals = {"R": raw_r, "J": jit, "C": conf, "T": dur, "H": hot}
    return score, signals, {"reason": "static"}


def _in_vehicle(u: _Unit, ctx: _ContextIndex, cfg, fps):
    if not ctx.present:
        return None
    frac = _context_fraction(u, ctx, cfg.context.vehicle_classes, "iob",
                             cfg.screen.in_vehicle_iob, cfg, fps)
    return ramp(frac, *cfg.screen.ramps["inside"]), {"inside_frac": frac}, {"reason": "in_vehicle"}


def _rider(u: _Unit, ctx: _ContextIndex, cfg: RefineConfig, fps):
    sc, r = cfg.screen, cfg.screen.ramps
    v = u.v[1:]
    v = v[np.isfinite(v)]
    frac_fast = float(np.mean(v > sc.rider_speed)) if len(v) else 0.0
    smooth = heading_smoothness(u.frames, u.boxes, fps, window=cfg.motion.height_window,
                                moving_min=cfg.motion.moving_min)
    smooth = 0.0 if not np.isfinite(smooth) else smooth
    k = (_context_fraction(u, ctx, cfg.context.twowheeler_classes, "iou", sc.twowheeler_iou,
                           cfg, fps) if ctx.present else None)
    hint = u.localized_hint
    p = hint.avg_score if hint is not None and hint.cls in cfg.hints.reclass_class_map else None
    rp = ramp(p, *cfg.hints.reclass_ramp) if p is not None else 0.0
    score = max(ramp(frac_fast, *r["F"]) * ramp(smooth, *r["S"]),
                ramp(k, *r["K"]) if k is not None else 0.0, rp)
    signals = {"F": frac_fast, "S_smooth": smooth, "K": k, "P": p}
    new_cls = None
    if p is not None and rp >= cfg.hints.subtype_min:
        subtype = cfg.hints.reclass_class_map[hint.cls]
        new_cls = int(cfg.reclass_map[subtype])
        signals["subtype_source"] = "reclass"
        signals["subtype"] = subtype
    if u.unlocalized_hint is not None:
        signals["hint_unlocalized"] = {"cls": u.unlocalized_hint.cls,
                                       "avg_score": u.unlocalized_hint.avg_score}
    return score, signals, {"new_cls": new_cls}


class _DupIndex:
    """First frame, last frame and median area of every whole track, for candidate pruning."""

    def __init__(self, units: dict[int, _Unit]):
        """Stack the per-track ranges of ``units`` (ascending track order)."""
        self.units = units
        self.ids = np.array(list(units), dtype=int)
        self.first = np.array([u.frames[0] for u in units.values()], dtype=int)
        self.last = np.array([u.frames[-1] for u in units.values()], dtype=int)
        self.area = np.array([u.area for u in units.values()], dtype=float)


def _duplicate(u: _Unit, index: _DupIndex, cfg: RefineConfig, fps):
    sc = cfg.screen
    # Cheap vectorized pruning first: enough frame range in common, and u must be the smaller.
    span = np.minimum(index.last, u.frames[-1]) - np.maximum(index.first, u.frames[0]) + 1
    smaller = (u.area < index.area) | ((u.area == index.area) & (u.track > index.ids))
    cand = np.flatnonzero((index.ids != u.track) & (span >= sc.duplicate_min_frames) & smaller)
    best = None
    for k in cand:
        tid = int(index.ids[k])
        o = index.units[tid]
        common, iu, io_ = np.intersect1d(u.frames, o.frames, assume_unique=(
            u.unique_frames and o.unique_frames), return_indices=True)
        if len(common) < sc.duplicate_min_frames:
            continue
        iob = _paired_iob(u.boxes[iu], o.boxes[io_])
        dv = np.linalg.norm(u.vel[iu] - o.vel[io_], axis=1) / u.hmed * fps
        frac = int(np.count_nonzero((iob >= sc.duplicate_iob) & (dv < sc.move_together))) / len(
            common)
        s = ramp(frac, *sc.ramps["D"])
        if best is None or s > best[0]:
            best = (s, {"D": frac}, {"reason": "duplicate", "of": tid})
    return best


def _paired_iob(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the IoB of ``a[i]`` (over its own area) with ``b[i]`` for each row ``i``."""
    iw = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    ih = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)
    area = a[:, 2] * a[:, 3]
    return np.divide(inter, area, out=np.zeros_like(inter), where=area != 0)


def _segments(rows: pd.DataFrame, cuts: list[int]) -> list[pd.DataFrame]:
    f = rows["frame"].to_numpy(int)
    cuts = sorted({c for c in cuts if f[0] < c <= f[-1]})
    if not cuts:
        return []
    bounds = [f[0], *cuts, f[-1] + 1]
    segs = [rows[(rows["frame"] >= a) & (rows["frame"] < b)] for a, b in pairwise(bounds)]
    return [s for s in segs if len(s)]


def propose_screen(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, sctx: ScreenContext,
    segment_cuts: dict[int, list[int]],
) -> list[Event]:
    """Propose DROP / RECLASS events for false tracks (spec 6.2)."""
    if work.empty:
        return []
    sc = cfg.screen
    ctx = _ContextIndex(sctx)
    cap = sc.vehicle_static_score_cap if cfg.target == "vehicle" else sc.static_score_cap
    tracks = {int(t): g.sort_values("frame") for t, g in work.groupby("track", sort=True)}
    wholes = {t: _unit(t, g, cfg, fps) for t, g in tracks.items()}
    meds = {}
    for t, u in wholes.items():
        _, r_ramp, med = _static_center(u, cfg)
        if r_ramp >= 0.5:
            meds[t] = med
    static_meds = (np.array(list(meds), dtype=int), np.array(list(meds.values())).reshape(-1, 2))
    dup_index = _DupIndex(wholes) if cfg.target != "person" else None

    def hyps(u):
        out = {"static": lambda: _static(u, cfg, fps, static_meds, cap)}
        if cfg.target == "person":
            out["in_vehicle"] = lambda: _in_vehicle(u, ctx, cfg, fps)
            out["rider"] = lambda: _rider(u, ctx, cfg, fps)
        else:
            out["duplicate"] = lambda: _duplicate(u, dup_index, cfg, fps)
        return out

    events: list[Event] = []
    for t, g in tracks.items():
        raw_ids = {int(r) for r in g["raw_id"].unique()}
        hint = sctx.hints.get(next(iter(raw_ids))) if len(raw_ids) == 1 else None
        segs = _segments(g, segment_cuts.get(t, []))
        localized = hint is not None and not segs and not (raw_ids & sctx.split_raw_ids)
        whole = replace(wholes[t], localized_hint=hint if localized else None,
                        unlocalized_hint=None if localized else hint)
        seg_units = [_unit(t, s, cfg, fps, unlocalized=hint) for s in segs]
        whole_h = hyps(whole)
        seg_h = [hyps(su) for su in seg_units]
        best = None
        for name, fn in whole_h.items():
            cand = None
            if not seg_units:
                w = fn()
                cand = None if w is None else (w[0], w[1], w[2], None)
            else:
                scores = [h[name]() for h in seg_h]
                sup = [(su, s) for su, s in zip(seg_units, scores, strict=True)
                       if s is not None and s[0] >= sc.reject_below]
                segments = [[int(su.frames[0]), int(su.frames[-1]),
                             None if s is None else float(s[0])]
                            for su, s in zip(seg_units, scores, strict=True)]
                if sup and len(sup) == len(seg_units):
                    # Every segment supports the hypothesis: the event covers the whole track,
                    # scored by its weakest segment (spec 6.2). The whole-track score is not
                    # used, because mixing segments can hide what each shows (e.g. two static
                    # spots far apart).
                    _, low = min(sup, key=lambda p: p[1][0])
                    signals = {**low[1], "all_segments": True, "segments": segments}
                    cand = (low[0], signals, low[2], None)
                elif sup:
                    _, top = max(sup, key=lambda p: p[1][0])
                    spans = [[int(su.frames[0]), int(su.frames[-1])] for su, _ in sup]
                    signals = {**top[1], "partial": True, "segments": segments}
                    cand = (min(top[0], sc.mixed_score_cap), signals, top[2], spans)
            if cand is not None and (best is None or cand[0] > best[1][0]):
                best = (name, cand)
        if best is None or best[1][0] < sc.reject_below:
            continue
        name, (score, signals, params, spans) = best
        lin = lineage_of_rows(g)
        kind = EventKind.RECLASS if name == "rider" else EventKind.DROP
        frames = ((spans[0][0], spans[-1][1]) if spans is not None
                  else (int(g["frame"].iloc[0]), int(g["frame"].iloc[-1])))
        events.append(Event.propose(
            stage=STAGE, kind=kind, tracks=[t], lineage=[lin], frames=frames,
            params={**params, "spans": spans}, algo_score=score,
            signals={**signals, "hypothesis": name},
        ))
    return events


def propose_orphans(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, *, linked_tracks: set[int],
    pending_endpoints: set[int],
) -> tuple[list[Event], list[int]]:
    """Propose dropping short unlinked tracks; defer those with a pending link (spec 6.2)."""
    oc = cfg.orphan
    events: list[Event] = []
    deferred: list[int] = []
    for t, g in work.groupby("track", sort=True):
        t = int(t)
        if t in linked_tracks:
            continue
        observed = len(g) / fps
        if observed >= oc.min_seconds:
            continue
        if t in pending_endpoints:
            deferred.append(t)
            continue
        score = float(ramp(observed, *oc.ramp))
        if score < oc.reject_below:
            continue
        g = g.sort_values("frame")
        events.append(Event.propose(
            stage=ORPHAN_STAGE, kind=EventKind.DROP, tracks=[t], lineage=[lineage_of_rows(g)],
            frames=(int(g["frame"].iloc[0]), int(g["frame"].iloc[-1])),
            params={"reason": "orphan", "spans": None}, algo_score=score,
            signals={"observed_seconds": observed},
        ))
    return events, deferred
