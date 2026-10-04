"""The only code that edits the work table (spec 2.1). Every function preserves the index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .events import Event, EventKind


def lineage_of_rows(
    rows: pd.DataFrame, excluded: dict[int, set[int]] | None = None
) -> list[list[int]]:
    """Return ``[[raw_id, first_frame, last_frame], ...]`` for rows, ordered by first frame.

    ``excluded`` maps a raw ID to frames that stage ``dedup`` dropped (dedup spec 3.5). A raw
    ID's span is split at each excluded frame that lies strictly inside it, so a consumer that
    reads the raw observations inside the spans never sees a dropped row.
    """
    if rows.empty:
        return []
    g = rows.groupby("raw_id")["frame"].agg(["min", "max"]).reset_index()
    spans: list[list[int]] = []
    for raw, first, last in g[["raw_id", "min", "max"]].to_numpy():
        raw, first, last = int(raw), int(first), int(last)
        start = first
        for f in sorted(f for f in (excluded or {}).get(raw, ()) if first < f < last):
            if f > start:
                spans.append([raw, start, f - 1])
            start = f + 1
        spans.append([raw, start, last])
    return sorted(spans, key=lambda s: (s[1], s[0]))


def lineage(work: pd.DataFrame, track: int) -> list[list[int]]:
    """Return the lineage of one track (spec 4.2)."""
    return lineage_of_rows(work.loc[work["track"] == track])


def next_track_id(work: pd.DataFrame) -> int:
    """Return an ID larger than every track ID in the table."""
    return int(work["track"].max()) + 1 if len(work) else 1


def _in_spans(frames: pd.Series, spans) -> np.ndarray:
    """Return boolean mask of frames inside any (inclusive) span."""
    if isinstance(spans, np.ndarray):
        spans = spans.tolist()
    mask = np.zeros(len(frames), dtype=bool)
    for a, b in spans:
        mask |= ((frames >= a) & (frames <= b)).to_numpy()
    return mask


def split_track(work: pd.DataFrame, track: int, cut_frame: int, new_id: int) -> pd.DataFrame:
    """Give rows of ``track`` at or after ``cut_frame`` the ID ``new_id``."""
    new_id = int(new_id)
    if new_id in work["track"].values:
        raise ValueError(f"new_id {new_id} already exists in the work table")
    out = work.copy()
    out.loc[(out["track"] == track) & (out["frame"] >= cut_frame), "track"] = new_id
    return out


def drop_rows(work: pd.DataFrame, track: int, spans=None) -> pd.DataFrame:
    """Drop a track, or only its rows inside ``spans``. An empty ``spans`` list is no-op."""
    if spans is not None:
        if isinstance(spans, np.ndarray):
            spans = spans.tolist()
        if not spans:
            return work
    mask = (work["track"] == track).to_numpy()
    if spans is not None:
        mask &= _in_spans(work["frame"], spans)
    return work.loc[~mask]


def reclass_rows(work: pd.DataFrame, track: int, new_cls: int, spans=None) -> pd.DataFrame:
    """Set the class of a track, or of its rows inside ``spans``. An empty ``spans`` is no-op."""
    if spans is not None:
        if isinstance(spans, np.ndarray):
            spans = spans.tolist()
        if not spans:
            return work
    out = work.copy()
    mask = (out["track"] == track).to_numpy()
    if spans is not None:
        mask &= _in_spans(out["frame"], spans)
    out.loc[mask, "cls"] = int(new_cls)
    return out


def merge_chains(
    work: pd.DataFrame, pairs: list[tuple[int, int]]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Merge linked tracks into chains that keep the earliest member's ID.

    A later member's rows on frames the chain already has (the small overlap that stage 3's
    gate 2 allows) are dropped; its other rows are kept, even inside the overlap span.
    """
    if not pairs:
        return work, {}
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(int(j))] = find(int(i))
    groups: dict[int, list[int]] = {}
    for t in {int(t) for p in pairs for t in p}:
        groups.setdefault(find(t), []).append(t)
    first = work.groupby("track")["frame"].min()
    rep_of: dict[int, int] = {}
    drop_idx: list = []
    for members in groups.values():
        members.sort(key=lambda t: (int(first[t]), t))
        seen = set(work.loc[work["track"] == members[0], "frame"].tolist())
        for t in members[1:]:
            frames = work.loc[work["track"] == t, "frame"]
            drop_idx.extend(frames.index[frames.isin(seen)].tolist())
            seen |= set(frames.tolist())
        for t in members:
            rep_of[t] = members[0]
    out = work.drop(index=drop_idx).copy()
    out["track"] = out["track"].map(lambda t: rep_of.get(int(t), int(t))).astype(int)
    return out.sort_values(["track", "frame"]), rep_of


def rows_to_drop(rows: pd.DataFrame) -> pd.Index:
    """Return the index labels of rows that lose on a frame where several rows exist.

    The best row of a frame has the higher ``score``; ties go to the smaller ``raw_id``, then
    to the smaller ``track``. The order is total, so the kept set does not depend on the order
    the rows or the merges arrive in (dedup spec 3.5).
    """
    order = rows.sort_values(
        ["frame", "score", "raw_id", "track"], ascending=[True, False, True, True], kind="stable"
    )
    return order.index[order.duplicated("frame", keep="first")]


def merge_tracks(
    work: pd.DataFrame, rep_of: dict[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel every member of a merge group to its representative and keep one row per frame.

    ``rep_of`` maps each member track to its representative (representatives map to
    themselves). Returns the merged table, with its index labels kept and sorted by track and
    frame, and the dropped rows.
    """
    if not rep_of:
        return work, work.iloc[0:0]
    new_track = work["track"].map(lambda t: rep_of.get(int(t), int(t)))
    members = work[work["track"].isin(rep_of)]
    drop: list = []
    for _, g in members.assign(_rep=new_track.loc[members.index]).groupby("_rep"):
        drop.extend(rows_to_drop(g).tolist())
    dropped = work.loc[drop]
    out = work.drop(index=drop).copy()
    out["track"] = new_track.loc[out.index].astype(int)
    return out.sort_values(["track", "frame"]), dropped


def renumber(work: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """Renumber tracks 1..N by (first frame, old ID); return the table and old->new map."""
    if work.empty:
        return work.reset_index(drop=True), {}
    first = work.groupby("track")["frame"].min().reset_index().sort_values(["frame", "track"])
    id_map = {int(t): i + 1 for i, t in enumerate(first["track"])}
    out = work.copy()
    out["track"] = out["track"].map(id_map).astype(int)
    return out.sort_values(["frame", "track"]).reset_index(drop=True), id_map


def apply_edit(work: pd.DataFrame, event: Event, *, new_id: int | None = None) -> pd.DataFrame:
    """Apply an event's final ``edit`` (spec 4.1); LINK edits are applied by ``merge_chains``."""
    if event.edit is None:
        raise ValueError(f"event {event.id or event.proposal_key[:12]} has no edit to apply")
    kind = EventKind(event.edit["kind"])
    p = event.edit["params"]
    track = event.tracks[0]
    if kind is EventKind.SPLIT:
        return split_track(
            work, track, int(p["cut_frame"]), next_track_id(work) if new_id is None else new_id
        )
    if kind is EventKind.DROP:
        return drop_rows(work, track, p.get("spans"))
    if kind is EventKind.RECLASS:
        if p.get("new_cls") is None:
            raise ValueError("a RECLASS edit needs new_cls")
        return reclass_rows(work, track, int(p["new_cls"]), p.get("spans"))
    raise ValueError(f"{kind} edits are applied at stage level, not by apply_edit")
