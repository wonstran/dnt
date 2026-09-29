"""``TrackRefiner``: runs the refinement stages in order and writes the outputs (spec 3)."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import __version__
from . import io
from .apply import apply_edit, lineage_of_rows, merge_chains, next_track_id, renumber
from .config import RefineConfig, to_frames
from .events import ACCEPTED, Decision, Event, EventKind, Ledger
from .features import Appearance
from .hints import read_reclass_hints
from .interpolate import interpolate_tracks_rts
from .link import run_link_stage
from .primitives import (
    box_centers,
    drop_context_duplicates,
    frame_runs,
    majority_class,
    occlusion_flags,
    speeds_hps,
)
from .screen import ScreenContext, propose_orphans, propose_screen
from .switch import propose_splits
from .verify import Band, decide, route_without_vlm

log = logging.getLogger(__name__)
LEDGER_FORMAT = "dnt.refine.ledger/1"


@dataclass
class RefineResult:
    """What ``TrackRefiner.refine`` produced."""

    tracks: pd.DataFrame
    ledger_path: Path
    review_path: Path | None
    summary: dict
    events: list[Event] = field(default_factory=list)


def output_paths(out) -> dict[str, Path]:
    """Return the ledger, review, and feature-cache paths that sit next to ``out`` (spec 2.5)."""
    out = Path(out)
    return {
        "ledger": out.with_suffix(".ledger.jsonl"),
        "review": out.with_suffix(".review.html"),
        "features": out.with_suffix(".features.npz"),
    }


def resolve_fps(arg, cfg_fps, video_fps) -> tuple[float, str]:
    """Resolve the frame rate: argument, then config, then video (spec 5.1)."""
    explicit, source = (arg, "argument") if arg is not None else (cfg_fps, "config")
    if explicit is not None:
        explicit = float(explicit)
        if explicit <= 0:
            raise ValueError(f"fps must be positive, got {explicit}")
        if video_fps and abs(explicit - video_fps) / video_fps > 0.01:
            log.warning(
                "fps %.3f (%s) differs from the video's %.3f by more than 1%%; using %.3f",
                explicit,
                source,
                video_fps,
                explicit,
            )
        return explicit, source
    if video_fps and video_fps > 0:
        return float(video_fps), "video"
    raise ValueError(
        "a frame rate is needed because every threshold is in seconds: pass fps=... "
        "(or --fps) when no video is given, or when the video reports no usable "
        "frame rate"
    )


def _file_record(path, sha256: str | None = None, **extra) -> dict:
    p = Path(path)
    return {
        "path": str(path),
        "abs_path": str(p.resolve()),
        "sha256": sha256 if sha256 is not None else io.sha256_file(p),
        **extra,
    }


def _is_blank(path) -> bool:
    """Whether a file has no content besides whitespace (checked without reading big files)."""
    size = Path(path).stat().st_size
    return size == 0 or (size <= 4096 and not Path(path).read_bytes().strip())


def table_summary(work: pd.DataFrame, fps: float) -> dict:
    """Summarize a work table for the ledger header (spec 8.3)."""
    if work.empty:
        return {
            "tracks": 0,
            "tracks_per_class": {},
            "observed_rows": 0,
            "interpolated_rows": 0,
            "median_track_seconds": 0.0,
        }
    per_track = work.groupby("track")["cls"].agg(majority_class)
    obs = work[work["interp"] == 0]
    dur = obs.groupby("track")["frame"].agg(lambda f: (f.max() - f.min() + 1) / fps)
    return {
        "tracks": int(work["track"].nunique()),
        "tracks_per_class": {
            str(k): int(v) for k, v in per_track.value_counts().sort_index().items()
        },
        "observed_rows": len(obs),
        "interpolated_rows": int((work["interp"] == 1).sum()),
        "median_track_seconds": float(dur.median()) if len(dur) else 0.0,
    }


def _event_counts(events: list[Event]) -> dict[str, int]:
    c = Counter(f"{e.stage}/{e.kind}/{e.decision}" for e in events)
    return dict(sorted(c.items()))


def fill_stage(
    work: pd.DataFrame, cfg: RefineConfig, fps: float, protected: dict[int, list[tuple[int, int]]]
) -> tuple[pd.DataFrame, list[Event]]:
    """Run stage 4; return the filled table and its FILL / SMOOTH records (spec 6.4)."""
    if work.empty:
        return work, []
    max_gap_s = cfg.fill.max_gap if cfg.fill.max_gap is not None else cfg.link.max_gap
    before = work[io.WORK_COLUMNS].reset_index(drop=True)
    out = interpolate_tracks_rts(
        tracks=before.copy(),
        fill_gaps_only=True,
        smooth_existing=cfg.fill.smooth_existing,
        process_var=cfg.motion.process_var,
        meas_var_pos=cfg.motion.meas_var_pos,
        meas_var_size=cfg.motion.meas_var_size,
        max_gap=to_frames(max_gap_s, fps),
        verbose=False,
        protected_gaps=protected,
    )
    out = out.sort_values(["track", "frame"]).reset_index(drop=True)
    out["raw_id"] = out.groupby("track")["raw_id"].ffill().astype(int)
    events: list[Event] = []
    gapped = out.loc[out["interp"] == 1, "track"].unique()
    for t, g in out[out["track"].isin(gapped)].groupby("track", sort=True):
        obs = g[g["interp"] == 0]
        lin = lineage_of_rows(obs)
        for a, b in frame_runs(g.loc[g["interp"] == 1, "frame"]):
            f_before = int(obs.loc[obs["frame"] < a, "frame"].iloc[-1])
            f_after = int(obs.loc[obs["frame"] > b, "frame"].iloc[0])
            seg = g[(g["frame"] >= f_before) & (g["frame"] <= f_after)]
            boxes = seg[["x", "y", "w", "h"]].to_numpy(float)
            v = speeds_hps(seg["frame"].to_numpy(), boxes, fps, cfg.motion.height_window)
            ends = box_centers(boxes[[0, -1]])
            h = max(float(np.median(boxes[:, 3])), 1.0)
            ev = Event.propose(
                stage="fill",
                kind=EventKind.FILL,
                tracks=[int(t)],
                lineage=[lin],
                frames=(f_before, f_after),
                params={"gap": [f_before, f_after], "n_rows": int(b - a + 1)},
                algo_score=1.0,
                signals={
                    "gap_seconds": (f_after - f_before - 1) / fps,
                    "chord_h": float(np.linalg.norm(ends[1] - ends[0]) / h),
                    "max_fill_speed_h_s": float(np.nanmax(v[1:])) if len(v) > 1 else 0.0,
                },
            )
            decide(ev, Decision.AUTO_ACCEPT, source="auto")
            ev.applied = True
            events.append(ev)
    if cfg.fill.smooth_existing:
        after = out.loc[out["interp"] == 0, ["track", "frame", "x", "y", "w", "h"]]
        m = before.merge(after, on=["track", "frame"], suffixes=("_0", "_1"))
        for t, g in m.groupby("track", sort=True):
            c0 = box_centers(g[["x_0", "y_0", "w_0", "h_0"]].to_numpy(float))
            c1 = box_centers(g[["x_1", "y_1", "w_1", "h_1"]].to_numpy(float))
            shift = np.linalg.norm(c1 - c0, axis=1)
            if not (shift > 0).any():
                continue
            k = int(np.argmax(shift))
            f = g["frame"].to_numpy(int)
            ev = Event.propose(
                stage="fill",
                kind=EventKind.SMOOTH,
                tracks=[int(t)],
                lineage=[lineage_of_rows(before[before["track"] == t])],
                frames=(int(f.min()), int(f.max())),
                params={"n_rows": int((shift > 0).sum())},
                algo_score=1.0,
                signals={
                    "mean_shift_px": float(shift.mean()),
                    "max_shift_px": float(shift[k]),
                    "max_shift_frame": int(f[k]),
                },
            )
            decide(ev, Decision.AUTO_ACCEPT, source="auto")
            ev.applied = True
            events.append(ev)
    return out, events


class _Stages:
    """Runs stages 1-4 on a work table and collects their events (spec 3)."""

    def __init__(
        self, cfg: RefineConfig, fps: float, frame_size, appearance, ctx_boxes, ctx_fmt, hints
    ):
        """Hold the per-run inputs."""
        self.cfg, self.fps, self.frame_size = cfg, fps, frame_size
        self.appearance, self.ctx_boxes, self.ctx_fmt, self.hints = (
            appearance,
            ctx_boxes,
            ctx_fmt,
            hints,
        )
        self.link_ctx = ctx_boxes
        self.seq: Counter = Counter()
        self.events: list[Event] = []
        self.orphan_deferred: list[int] = []

    def _route(self, evs: list[Event], band: Band, stage: str) -> None:
        route_without_vlm(evs, band)
        for e in evs:
            self.seq[stage] += 1
            e.id = f"{stage}-r0-{self.seq[stage]:06d}"
        self.events.extend(evs)

    def run(
        self, work: pd.DataFrame, tick: Callable[[str], None] | None = None
    ) -> tuple[pd.DataFrame, list[Event]]:
        """Run every enabled stage in the spec's order; ``tick(name)`` follows each stage."""
        tick = tick or (lambda _name: None)
        occluded = occlusion_flags(work, self.ctx_boxes, self.cfg.encoder.occlusion_iou)
        # own detections are judged against the input rows, before any stage edits them
        self.link_ctx = drop_context_duplicates(work, self.ctx_boxes)
        work, split_raw, cuts = self._switch(work)
        tick("switch")
        work = self._screen(work, split_raw, cuts)
        tick("screen")
        work, protected, linked, pending = self._link(work, occluded)
        tick("link")
        work = self._orphans(work, linked, pending)
        tick("orphan")
        work = self._fill(work, protected)
        tick("fill")
        return work, self.events

    def _switch(self, work):
        cfg = self.cfg
        if not cfg.switch.enabled or work.empty:
            return work, set(), {}
        res = propose_splits(work, cfg, self.fps, self.appearance)
        self._route(res.events, Band.of(cfg.switch), "switch")

        def raw_at(track, frame):
            rows = work.loc[(work["track"] == track) & (work["frame"] == frame), "raw_id"]
            return int(rows.iloc[0])

        cut_points = [(raw_at(t, c), c) for t, cs in res.weak_cuts.items() for c in cs]
        accepted = []
        for ev in res.events:
            t, c = ev.tracks[0], int(ev.params["cut_frame"])
            if ev.decision in ACCEPTED:
                accepted.append(ev)
            elif ev.decision is Decision.HUMAN_PENDING:
                cut_points.append((raw_at(t, c), c))
        split_raw: set[int] = set()
        for ev in sorted(accepted, key=lambda e: -int(e.params["cut_frame"])):
            t = ev.tracks[0]
            split_raw |= set(work.loc[work["track"] == t, "raw_id"].astype(int).tolist())
            work = apply_edit(work, ev, new_id=next_track_id(work))
            ev.applied = True
        cuts: dict[int, list[int]] = {}
        for raw, c in cut_points:
            rows = work.loc[(work["raw_id"] == raw) & (work["frame"] == c), "track"]
            if len(rows):
                cuts.setdefault(int(rows.iloc[0]), []).append(c)
        return work, split_raw, cuts

    def _screen(self, work, split_raw, cuts):
        cfg = self.cfg
        if not cfg.screen.enabled or work.empty:
            return work
        sctx = ScreenContext(
            boxes=self.ctx_boxes, fmt=self.ctx_fmt, hints=self.hints, split_raw_ids=split_raw
        )
        evs = propose_screen(work, cfg, self.fps, sctx, cuts)
        self._route(evs, Band.of(cfg.screen), "screen")
        for ev in evs:
            if ev.decision in ACCEPTED:
                work = apply_edit(work, ev)
                ev.applied = True
        return work

    def _link(self, work, occluded):
        cfg = self.cfg
        if not cfg.link.enabled or work.empty:
            return work, {}, set(), set()
        band = Band.of(cfg.link)
        res = run_link_stage(
            work,
            cfg,
            self.fps,
            appearance=self.appearance,
            context=self.link_ctx,
            frame_size=self.frame_size,
            occluded=occluded,
            route=lambda evs: self._route(evs, band, "link"),
        )
        work, rep_of = merge_chains(work, res.accepted)
        protected: dict[int, list[tuple[int, int]]] = {}
        for ev in res.events:
            if ev.applied and ev.params.get("gate") == "occluded":
                rep = rep_of.get(ev.tracks[0], ev.tracks[0])
                protected.setdefault(rep, []).append(tuple(ev.params["gap"]))
        pending = {rep_of.get(t, t) for t in res.pending_endpoints}
        return work, protected, set(rep_of.values()), pending

    def _orphans(self, work, linked, pending):
        cfg = self.cfg
        if not cfg.orphan.enabled or work.empty:
            return work
        evs, self.orphan_deferred = propose_orphans(
            work, cfg, self.fps, linked_tracks=linked, pending_endpoints=pending
        )
        self._route(evs, Band.of(cfg.orphan), "orphan")
        for ev in evs:
            if ev.decision in ACCEPTED:
                work = apply_edit(work, ev)
                ev.applied = True
        return work

    def _fill(self, work, protected):
        if not self.cfg.fill.enabled or work.empty:
            return work
        work, evs = fill_stage(work, self.cfg, self.fps, protected)
        for ev in evs:
            self.seq["fill"] += 1
            ev.id = f"fill-r0-{self.seq['fill']:06d}"
        self.events.extend(evs)
        return work


class TrackRefiner:
    """Refine track files after tracking; used like ``Detector`` and ``Tracker`` (spec 2.4)."""

    def __init__(
        self,
        config: RefineConfig | None = None,
        config_yaml: str | None = None,
        device: str | None = None,
        *,
        appearance_factory: Callable[..., Appearance | None] | None = None,
    ) -> None:
        """Configure once with ``config`` or ``config_yaml``; ``device`` sets ``encoder.device``."""
        if config is not None and config_yaml is not None:
            raise ValueError("pass config or config_yaml, not both")
        if config_yaml is not None:
            config = RefineConfig.from_yaml(config_yaml)
        self.config = config if config is not None else RefineConfig.defaults()
        if device is not None:
            self.config.encoder.device = device
        self.config.validate()
        self.appearance_factory = appearance_factory
        self.last_result: RefineResult | None = None

    def refine(
        self,
        track_file,
        out_file,
        video_file=None,
        context_file=None,
        reclass_file=None,
        fps: float | None = None,
        fmt: str = "dnt",
        video_index: int | None = None,
        video_tot: int | None = None,
        message: str | None = "",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Refine one track file into ``out_file`` and write its ledger (spec 2.4, 2.5).

        Returns the refined track table, like ``Tracker.track()``; ``self.last_result`` holds
        the ledger path, summary, and events.
        """
        cfg = self.config
        out = Path(out_file)
        paths = output_paths(out)
        track_sha = io.sha256_file(track_file)
        context_sha = None
        if context_file is not None:
            context_sha = io.sha256_file(context_file)
            if context_sha == track_sha and not _is_blank(track_file):
                raise ValueError(
                    "context_file has the same content as track_file; the occlusion mask "
                    "compares each track with the context boxes, so the same file would flag "
                    "every row as occluded. Pass the detections or tracks of a different "
                    "run as context."
                )
        vinfo = io.video_info(video_file) if video_file is not None else None
        fps_val, fps_src = resolve_fps(fps, cfg.fps, vinfo["fps"] if vinfo else None)
        if cfg.frame_size:
            frame_size = tuple(int(v) for v in cfg.frame_size)
        elif vinfo:
            frame_size = (vinfo["width"], vinfo["height"])
        else:
            frame_size = None
        tin = io.read_tracks(track_file, fmt=fmt, class_id=cfg.class_ids[0])
        work = tin.work
        if (
            vinfo
            and vinfo["frame_count"] > 0
            and len(work)
            and int(work["frame"].max()) > vinfo["frame_count"]
        ):
            raise ValueError(
                f"track frame {int(work['frame'].max())} exceeds the video's frame count "
                f"{vinfo['frame_count']}; the track file does not belong to this video"
            )
        ctx_boxes, ctx_fmt = (
            io.read_context(context_file, cfg.context.format)
            if context_file is not None
            else (None, None)
        )
        hint_map = (
            read_reclass_hints(reclass_file, set(work["raw_id"].unique().tolist()))
            if reclass_file is not None
            else {}
        )
        inputs = {
            "tracks": _file_record(track_file, track_sha, format=fmt),
            "video": None
            if video_file is None
            else {
                "path": str(video_file),
                "abs_path": str(Path(video_file).resolve()),
                "fingerprint": io.video_fingerprint(video_file, vinfo["frame_count"]),
                "frame_count": vinfo["frame_count"],
            },
            "context": None
            if context_file is None
            else _file_record(context_file, context_sha, format=ctx_fmt),
            "hints": None if reclass_file is None else {"reclass": _file_record(reclass_file)},
            "features": None,
        }
        before = table_summary(work, fps_val)
        appearance = self._appearance(work, video_file, ctx_boxes, fps_val)
        stages = _Stages(cfg, fps_val, frame_size, appearance, ctx_boxes, ctx_fmt, hint_map)
        desc = (
            "Refining"
            if video_index is None or video_tot is None
            else f"Refining {video_index} of {video_tot}"
        )
        if message:
            desc += f" {message}"
        with tqdm(total=5, desc=desc, unit=" stage", disable=not verbose) as pbar:
            work, events = stages.run(
                work, tick=lambda name: (pbar.set_postfix_str(name), pbar.update(1))
            )
        work, id_map = renumber(work)
        summary = {
            "before": before,
            "after": table_summary(work, fps_val),
            "events": _event_counts(events),
            "vlm": {"calls": 0, "cache_hits": 0, "failures": 0},
            "orphan_deferred": sorted(id_map[t] for t in stages.orphan_deferred if t in id_map),
            "filled_input_rows_removed": tin.n_filled_removed,
            "duplicate_input_rows_removed": tin.n_duplicates_removed,
        }
        header = {
            "format": LEDGER_FORMAT,
            "dnt_version": __version__,
            "round": 0,
            "parent": None,
            "config": cfg.to_dict(),
            "inputs": inputs,
            "fps": fps_val,
            "fps_source": fps_src,
            "frame_size": list(frame_size) if frame_size else None,
            "id_map": {str(k): v for k, v in id_map.items()},
            "n_filled_input_rows_removed": tin.n_filled_removed,
            "smoothing": cfg.fill.smooth_existing,
            "summary": summary,
        }
        # The ledger goes first and the track file last, so an existing output implies its ledger
        # (refine_batch skips existing outputs).
        Ledger(header, events).write(paths["ledger"])
        io.write_tracks(work, out)
        log.info("refined %s: %s", out, summary["events"])
        self.last_result = RefineResult(
            tracks=work,
            ledger_path=paths["ledger"],
            review_path=None,
            summary=summary,
            events=events,
        )
        return work

    def refine_batch(
        self,
        track_files: Sequence,
        video_files: Sequence | None = None,
        output_path=None,
        context_files: Sequence | None = None,
        reclass_files: Sequence | None = None,
        fps: float | None = None,
        is_overwrite: bool = False,
        is_report: bool = True,
        message: str | None = "",
        verbose: bool = True,
    ) -> list[str]:
        """Refine several track files, like ``Tracker.track_batch`` (spec 2.4).

        Files are paired by position. Each output is ``<output_path>/<base>_refined.txt``,
        where ``<base>`` is the track file's stem without a trailing ``_track``. Existing
        outputs are skipped unless ``is_overwrite``; with ``is_report`` they are still listed.

        Raises
        ------
        ValueError
            If ``output_path`` is missing, or if ``video_files``, ``context_files`` or
            ``reclass_files`` is given with a length different from ``track_files``.

        """
        if output_path is None:
            raise ValueError(
                "refine_batch needs output_path: every run writes a ledger next to its output"
            )
        for name, seq in (
            ("video_files", video_files),
            ("context_files", context_files),
            ("reclass_files", reclass_files),
        ):
            if seq is not None and len(seq) != len(track_files):
                raise ValueError(
                    f"{name} has {len(seq)} entries for {len(track_files)} "
                    "track files; files are paired by position"
                )
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        def pick(seq, i):
            return seq[i] if seq is not None else None

        results: list[str] = []
        total = len(track_files)
        for i, track_file in enumerate(track_files):
            base = re.sub(r"_track$", "", Path(track_file).stem)
            out = out_dir / f"{base}_refined.txt"
            if out.exists() and not is_overwrite:
                if is_report:
                    results.append(str(out))
                continue
            self.refine(
                track_file,
                out,
                video_file=pick(video_files, i),
                context_file=pick(context_files, i),
                reclass_file=pick(reclass_files, i),
                fps=fps,
                video_index=i + 1,
                video_tot=total,
                message=message,
                verbose=verbose,
            )
            results.append(str(out))
        return results

    def _appearance(self, work, video, context, fps) -> Appearance | None:
        if self.appearance_factory is not None:
            return self.appearance_factory(
                work=work, video=video, context=context, fps=fps, config=self.config
            )
        if video is not None and self.config.encoder.kind != "none":
            log.warning("appearance encoders arrive in dnt.refine Plan 2; running motion-only")
        return None
