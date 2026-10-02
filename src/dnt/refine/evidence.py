"""Composite evidence images for VLM questions and review cards (spec 7.1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crops import FrameReader, crop_box
from .events import Event, EventKind

log = logging.getLogger(__name__)

BACKGROUND = 128
EVIDENCE_PAD = 1.5
MIN_TILE_HEIGHT = 160
MAX_TILE_WIDTH = 360
CONTEXT_WIDTH = 768
GAP = 6
JPEG_QUALITY = 85
CHUNK = 64
GREEN, RED, YELLOW, GRAY = (0, 200, 0), (0, 0, 220), (0, 220, 220), (170, 170, 170)


@dataclass
class ContextBox:
    """A box drawn on a context frame."""

    label: str
    color: tuple
    dashed: bool
    xywh: tuple | None


@dataclass
class ContextTile:
    """A context frame, its caption, and the boxes drawn on it."""

    frame: int
    caption: str
    boxes: list[ContextBox] = field(default_factory=list)


@dataclass
class EvidencePlan:
    """What an evidence image shows: crop rows and context frames."""

    rows: list[tuple[str, list[tuple[int, int]]]] = field(default_factory=list)
    contexts: list[ContextTile] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to draw."""
        return not any(tiles for _, tiles in self.rows) and not self.contexts


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


class EvidenceBuilder:
    """Builds one composite JPEG per event from the video and the raw tracks."""

    def __init__(
        self,
        video_file,
        raw_work: pd.DataFrame,
        occluded: pd.Series,
        *,
        frame_count: int,
        send_context_frames: bool = True,
    ):
        """Index the raw boxes by ``(raw_id, frame)``; ``occluded`` aligns with ``raw_work``."""
        self.video_file = video_file
        self.frame_count = int(frame_count)
        self.send_context_frames = bool(send_context_frames)
        w = raw_work.assign(_occ=occluded.reindex(raw_work.index).fillna(False).to_numpy(bool))
        self._box = {
            (int(r), int(f)): (float(x), float(y), float(ww), float(hh))
            for r, f, x, y, ww, hh in zip(
                w["raw_id"], w["frame"], w["x"], w["y"], w["w"], w["h"], strict=True
            )
        }
        self._occ = {
            (int(r), int(f)): bool(o)
            for r, f, o in zip(w["raw_id"], w["frame"], w["_occ"], strict=True)
        }
        self._by_frame: dict[int, list[int]] = {}
        for r, f in self._box:
            self._by_frame.setdefault(f, []).append(r)

    # ---- planning (no video access) ----

    def _pairs(self, spans, lo: int | None, hi: int | None, *, include_occluded: bool):
        """``(raw_id, frame)`` pairs of the lineage ``spans`` with a drawable box, by frame."""
        out = []
        for raw, f0, f1 in spans:
            for f in range(int(f0), int(f1) + 1):
                if (lo is not None and f < lo) or (hi is not None and f > hi):
                    continue
                key = (int(raw), f)
                box = self._box.get(key)
                if box is None or box[2] <= 0 or box[3] <= 0 or f >= self.frame_count:
                    continue
                if self._occ[key] and not include_occluded:
                    continue
                out.append(key)
        return sorted(out, key=lambda k: k[1])

    def _clean(self, spans, lo: int | None = None, hi: int | None = None):
        """Clean ``(raw_id, frame)`` pairs of the lineage ``spans``, sorted by frame."""
        return self._pairs(spans, lo, hi, include_occluded=False)

    def _observed(self, spans, lo: int | None = None, hi: int | None = None):
        """Like ``_clean`` but keeps rows flagged occluded (screen evidence, spec 7.1)."""
        return self._pairs(spans, lo, hi, include_occluded=True)

    def _box_at(self, spans, frame: int):
        for raw, f0, f1 in spans:
            if int(f0) <= frame <= int(f1) and (int(raw), frame) in self._box:
                return self._box[(int(raw), frame)]
        return None

    def _ctx(self, frame, caption, boxes):
        if not self.send_context_frames or not 0 <= frame < self.frame_count:
            return []
        return [ContextTile(int(frame), caption, boxes)]

    def plan(self, event: Event) -> EvidencePlan:
        """Return the tiles the image of ``event`` will show (spec 7.1)."""
        plan = EvidencePlan()
        a_spans = event.lineage[0] if event.lineage else []
        if event.kind in (EventKind.DROP, EventKind.RECLASS):
            spans = event.params.get("spans")
            if spans:  # a partial edit: the supported segments (A) and the rest of the track (B)
                inside = sorted(
                    {k for lo, hi in spans for k in self._observed(a_spans, lo, hi)},
                    key=lambda k: k[1],
                )
                chosen = set(inside)
                outside = [k for k in self._observed(a_spans) if k not in chosen]
                plan.rows += [("A", _spread(inside, 3)), ("B", _spread(outside, 3))]
                anchor = inside
            else:
                anchor = self._observed(a_spans)
                plan.rows.append(("A", _spread(anchor, 6)))
            if anchor:
                f = anchor[len(anchor) // 2][1]
                box = self._box_at(a_spans, f)
                plan.contexts += self._ctx(f, "A", [ContextBox("A", GREEN, False, box)])
        elif event.kind is EventKind.SPLIT:
            t = int(event.params["cut_frame"])
            before = self._clean(a_spans, None, t - 1)[-3:]
            after = self._clean(a_spans, t, None)[:3]
            plan.rows += [("A", before), ("B", after)]
            boxes = [ContextBox("A", GREEN, False, self._box_at(a_spans, t))]
            own = {int(r) for r, _, _ in a_spans}
            mine = self._box_at(a_spans, t)
            for raw in self._by_frame.get(t, []):
                if raw in own or mine is None:
                    continue
                ob = self._box[(raw, t)]
                if abs(ob[0] - mine[0]) < 3 * mine[3] and abs(ob[1] - mine[1]) < 3 * mine[3]:
                    boxes.append(ContextBox("", GRAY, False, ob))
            plan.contexts += self._ctx(t, f"cut at {t}", boxes)
        elif event.kind is EventKind.LINK:
            b_spans = event.lineage[1]
            t_e, t_s = (int(v) for v in event.params["gap"])
            plan.rows += [
                ("A", self._clean(a_spans)[-3:]),
                ("B", self._clean(b_spans)[:3]),
            ]
            plan.contexts += self._ctx(
                t_e, "A ends", [ContextBox("A", GREEN, False, self._box_at(a_spans, t_e))]
            )
            if event.params.get("gate") == "occluded":
                a_last, b_first = self._box_at(a_spans, t_e), self._box_at(b_spans, t_s)
                mid = (t_e + t_s) // 2
                hidden = None
                if a_last is not None and b_first is not None and t_s > t_e:
                    k = (mid - t_e) / (t_s - t_e)
                    hidden = tuple(a + k * (b - a) for a, b in zip(a_last, b_first, strict=True))
                plan.contexts += self._ctx(
                    mid, "hidden path", [ContextBox("?", YELLOW, True, hidden)]
                )
            plan.contexts += self._ctx(
                t_s, "B starts", [ContextBox("B", RED, False, self._box_at(b_spans, t_s))]
            )
        return plan

    # ---- rendering ----

    def build(self, event: Event) -> bytes | None:
        """Return the composite JPEG for one event, or ``None``."""
        return self.build_many([event]).get(event.id)

    def build_many(self, events: list[Event]) -> dict[str, bytes | None]:
        """Return ``{event.id: JPEG or None}``; frames are read once per chunk, in order."""
        out: dict[str, bytes | None] = {}
        for i in range(0, len(events), CHUNK):
            chunk = events[i : i + CHUNK]
            plans = {e.id: self.plan(e) for e in chunk}
            parts: dict[tuple, np.ndarray] = {}
            requests: dict[int, list[tuple]] = {}
            for e in chunk:
                p = plans[e.id]
                for ri, (_, tiles) in enumerate(p.rows):
                    for ti, (raw, f) in enumerate(tiles):
                        requests.setdefault(f, []).append((e.id, "crop", ri, ti, raw))
                for ci, ctx in enumerate(p.contexts):
                    requests.setdefault(ctx.frame, []).append((e.id, "ctx", ci, 0, 0))
            if requests:
                try:
                    reader = FrameReader(self.video_file)
                except ValueError as exc:
                    log.warning("evidence: %s; no images for %d events", exc, len(chunk))
                    out.update({e.id: None for e in chunk})
                    continue
                with reader:
                    self._render(reader, requests, plans, parts)
            for e in chunk:
                out[e.id] = self._compose(plans[e.id], parts, e.id)
        return out

    def _render(self, reader, requests: dict, plans: dict, parts: dict) -> None:
        """Decode the wanted frames in order; a frame that cannot be read is skipped."""
        pending = sorted(requests)
        while pending:
            frames = reader.frames(pending)
            while pending:
                try:
                    f, img = next(frames)
                except StopIteration:
                    return
                except ValueError as exc:
                    log.warning("evidence: skipping frame %d: %s", pending.pop(0), exc)
                    break  # the reader seeks on its next call: restart over the rest
                pending.pop(0)  # frames come back in sorted order, one per wanted index
                for eid, kind, a, b, raw in requests[f]:
                    if kind == "crop":
                        tile = self._crop_tile(img, raw, f)
                    else:
                        tile = self._context_tile(img, plans[eid].contexts[a])
                    if tile is not None:
                        parts[(eid, kind, a, b)] = tile

    def _crop_tile(self, img, raw: int, frame: int):
        import cv2

        crop = crop_box(img, self._box[(raw, frame)], EVIDENCE_PAD)
        if crop is None:
            return None
        h, w = crop.shape[:2]
        scale = max(1.0, MIN_TILE_HEIGHT / h)
        scale = min(scale, MAX_TILE_WIDTH / w) if w * scale > MAX_TILE_WIDTH else scale
        tile = cv2.resize(
            np.ascontiguousarray(crop[..., ::-1]),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.putText(tile, f"f{frame}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return tile

    def _context_tile(self, img, ctx: ContextTile):
        import cv2

        _, w = img.shape[:2]
        scale = CONTEXT_WIDTH / w if w > CONTEXT_WIDTH else 1.0
        frame = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()
        for box in ctx.boxes:
            if box.xywh is None:
                continue
            x, y, bw, bh = (v * scale for v in box.xywh)
            p0, p1 = (int(x), int(y)), (int(x + bw), int(y + bh))
            if box.dashed:
                _dashed_rect(frame, p0, p1, box.color)
            else:
                cv2.rectangle(frame, p0, p1, box.color, 1)
            if box.label:
                cv2.putText(
                    frame,
                    box.label,
                    (p0[0], max(12, p0[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box.color,
                    1,
                )
        cv2.putText(
            frame,
            f"{ctx.caption} (f{ctx.frame})",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame

    def _compose(self, plan: EvidencePlan, parts: dict, eid: str) -> bytes | None:
        import cv2

        rows = []
        for ri, (label, tiles) in enumerate(plan.rows):
            imgs = [
                parts[(eid, "crop", ri, ti)]
                for ti in range(len(tiles))
                if (eid, "crop", ri, ti) in parts
            ]
            if imgs:
                rows.append(_hstack(imgs, label))
        for ci in range(len(plan.contexts)):
            if (eid, "ctx", ci, 0) in parts:
                rows.append(parts[(eid, "ctx", ci, 0)])
        if not rows:
            return None
        width = max(r.shape[1] for r in rows)
        canvas = np.full(
            (sum(r.shape[0] + GAP for r in rows) + GAP, width + 2 * GAP, 3), BACKGROUND, np.uint8
        )
        y = GAP
        for r in rows:
            canvas[y : y + r.shape[0], GAP : GAP + r.shape[1]] = r
            y += r.shape[0] + GAP
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None


def _hstack(imgs: list[np.ndarray], label: str) -> np.ndarray:
    import cv2

    height = max(i.shape[0] for i in imgs)
    cols = [np.full((height, 28, 3), BACKGROUND, np.uint8)]
    cv2.putText(cols[0], label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for im in imgs:
        pad = np.full((height - im.shape[0], im.shape[1], 3), BACKGROUND, np.uint8)
        cols += [np.vstack([im, pad]), np.full((height, GAP, 3), BACKGROUND, np.uint8)]
    return np.hstack(cols)


def _dashed_rect(img, p0, p1, color, dash: int = 6) -> None:
    import cv2

    (x0, y0), (x1, y1) = p0, p1
    for x in range(x0, x1, 2 * dash):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, 1)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, 1)
    for y in range(y0, y1, 2 * dash):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, 1)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, 1)
