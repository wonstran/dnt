import cv2
import numpy as np
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.events import Event, EventKind
from dnt.refine.evidence import EvidenceBuilder

from ._fixtures import box_rows, table
from ._video import BLUE, RED, make_color_video, video_rows

N = 120


def _work(*row_lists):
    work = io.to_work(table(*row_lists)).work
    return work


def _builder(tmp_path, work, colors, occluded=None, **kw):
    rows = []
    for raw, color in colors.items():
        rows += video_rows(work[work.raw_id == raw].assign(track=raw).pipe(_as_rows), color)
    video = make_color_video(tmp_path / "v.mp4", rows, N)
    occ = pd.Series(False, index=work.index) if occluded is None else occluded
    return EvidenceBuilder(video, work, occ, frame_count=N, **kw)


def _as_rows(df):
    return [[r.frame, r.track, r.x, r.y, r.w, r.h] for r in df.itertuples()]


def _decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _ev(kind, stage, tracks, lineage, frames, **params):
    ev = Event.propose(stage=stage, kind=kind, tracks=tracks, lineage=lineage, frames=frames,
                       params=params, algo_score=0.5, signals={})
    ev.id = f"{stage}-r0-000001"
    return ev


def _walker(raw, frames, x0=40.0, y0=60.0):
    return box_rows(raw, frames, x0, y0, vx=1.0, w=30.0, h=70.0)


def test_screen_event_plan_has_six_even_crops_and_one_context(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    plan = b.plan(ev)
    (_, tiles), = plan.rows
    assert len(tiles) == 6 and tiles[0] == (1, 0) and tiles[-1] == (1, 59)
    assert [f for _, f in tiles] == sorted(f for _, f in tiles)
    assert len(plan.contexts) == 1 and plan.contexts[0].frame in range(25, 35)  # mid-life


def test_a_partial_screen_event_shows_the_supported_and_the_unsupported_segments(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.RECLASS, "screen", [1], [[[1, 0, 59]]], (20, 40), new_cls=None,
             spans=[[20, 40]])
    plan = b.plan(ev)
    (la, a), (lb, bb) = plan.rows
    assert (la, lb) == ("A", "B")
    assert 1 <= len(a) <= 3 and all(20 <= f <= 40 for _, f in a)
    assert 1 <= len(bb) <= 3 and all(f < 20 or f > 40 for _, f in bb)
    assert 20 <= plan.contexts[0].frame <= 40  # the context frame is inside the supported part
    jpeg = b.build(ev)
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] >= 2 * 160 + 240  # two rows of crops (160 px+) plus the context frame (a single row plus context is only 418 px)


def test_a_partial_event_that_covers_the_whole_track_has_an_empty_second_row(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.RECLASS, "screen", [1], [[[1, 0, 59]]], (0, 59), new_cls=None,
             spans=[[0, 59]])
    (_, a), (_, bb) = b.plan(ev).rows
    assert len(a) == 3 and bb == []


def test_screen_evidence_keeps_overlapping_rows_that_identity_evidence_drops(tmp_path):
    w = _work(_walker(1, range(60)))
    occ = pd.Series(True, index=w.index)  # always overlapping another box
    b = _builder(tmp_path, w, {1: RED}, occluded=occ)
    drop = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="duplicate",
               spans=None, of=2)
    plan = b.plan(drop)
    (_, tiles), = plan.rows
    assert len(tiles) == 6 and len(plan.contexts) == 1 and not plan.is_empty
    assert b.build(drop)[:2] == b"\xff\xd8"
    split = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    assert b.plan(split).rows == [("A", []), ("B", [])]  # identity questions need clean crops


def test_split_plan_uses_three_clean_crops_each_side(tmp_path):
    w = _work(_walker(1, range(60)))
    occ = pd.Series(False, index=w.index)
    occ[w.frame.isin([28, 29])] = True  # the two frames before the cut are occluded
    b = _builder(tmp_path, w, {1: RED}, occluded=occ)
    ev = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    plan = b.plan(ev)
    (_, a), (_, bb) = plan.rows
    assert [f for _, f in a] == [25, 26, 27] and [f for _, f in bb] == [30, 31, 32]
    assert plan.contexts[0].frame == 30


def test_link_plan_rows_contexts_and_the_hidden_path(tmp_path):
    w = _work(_walker(1, range(0, 40)), _walker(2, range(60, 100), x0=100.0))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    base = dict(gap=[39, 60])
    ev = _ev(EventKind.LINK, "link", [1, 2], [[[1, 0, 39]], [[2, 60, 99]]], (39, 60),
             gate="normal", **base)
    plan = b.plan(ev)
    (_, a), (_, bb) = plan.rows
    assert [f for _, f in a] == [37, 38, 39] and [f for _, f in bb] == [60, 61, 62]
    assert [c.frame for c in plan.contexts] == [39, 60]
    occ = _ev(EventKind.LINK, "link", [1, 2], [[[1, 0, 39]], [[2, 60, 99]]], (39, 60),
              gate="occluded", **base)
    mid = b.plan(occ).contexts
    assert [c.frame for c in mid] == [39, 49, 60]
    hidden = [bx for bx in mid[1].boxes if bx.dashed]
    assert len(hidden) == 1 and hidden[0].xywh is not None
    # A's last box (frame 39) is at x = 79, B's first (frame 60) at x = 100; frame 49 is 10/21 of
    # the way, so the hidden box is at x = 89 with the same y, w, h
    assert hidden[0].xywh == pytest.approx((89.0, 60.0, 30.0, 70.0))


def test_fewer_clean_crops_than_wanted_and_no_context_option(tmp_path):
    w = _work(_walker(1, range(2)))
    b = _builder(tmp_path, w, {1: RED}, send_context_frames=False)
    ev = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 1]]], (1, 1), cut_frame=1)
    plan = b.plan(ev)
    (_, a), (_, bb) = plan.rows
    assert [f for _, f in a] == [0] and [f for _, f in bb] == [1]
    assert plan.contexts == []


def test_build_makes_a_jpeg_of_bounded_width(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    jpeg = b.build(ev)
    assert jpeg[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.ndim == 3 and 100 < img.shape[1] <= 800 and img.shape[0] >= 160
    # a crop tile is at least 160 px high: the first row's tallest tile
    no_ctx = _builder(tmp_path, w, {1: RED}, send_context_frames=False).build(ev)
    img2 = cv2.imdecode(np.frombuffer(no_ctx, np.uint8), cv2.IMREAD_COLOR)
    assert img2.shape[0] < img.shape[0]


def test_build_many_keys_by_event_id_and_skips_frames_past_the_video(tmp_path):
    w = _work(_walker(1, range(60)), _walker(2, [10, 500], x0=150.0))
    b = _builder(tmp_path, w, {1: RED})
    e1 = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    e2 = _ev(EventKind.DROP, "screen", [2], [[[2, 10, 500]]], (10, 500), reason="static",
             spans=None)
    e2.id = "screen-r0-000002"
    out = b.build_many([e1, e2])
    assert set(out) == {"screen-r0-000001", "screen-r0-000002"}
    assert out["screen-r0-000001"] is not None
    assert out["screen-r0-000002"] is not None  # frame 10 is drawn, frame 500 is skipped


def test_an_event_with_nothing_to_draw_gives_none(tmp_path):
    # identity evidence needs clean crops: an always-overlapping track has none, so a SPLIT on it
    # (no context frames either) has nothing to draw; a screen event on the same track does
    # (see test_screen_evidence_keeps_overlapping_rows_that_identity_evidence_drops)
    w = _work(_walker(1, range(60)))
    occ = pd.Series(True, index=w.index)
    b = _builder(tmp_path, w, {1: RED}, occluded=occ, send_context_frames=False)
    split = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    assert b.plan(split).is_empty and b.build(split) is None
    # a screen event whose lineage matches no observed row has nothing to draw either
    clean = _builder(tmp_path, w, {1: RED}, send_context_frames=False)
    ghost = _ev(EventKind.DROP, "screen", [1], [[[1, 500, 509]]], (500, 509), reason="static",
                spans=None)
    assert clean.plan(ghost).is_empty and clean.build(ghost) is None


def test_a_box_partly_outside_the_frame_still_gives_a_tile(tmp_path):
    w = _work(box_rows(1, range(10), -20.0, 200.0, w=40.0, h=60.0))  # cut at the left and bottom
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 9]]], (0, 9), reason="static", spans=None)
    assert len(b.plan(ev).rows[0][1]) == 6
    jpeg = b.build(ev)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    # the crop row is really drawn: without the context frame there is still an image of crop
    # height, and the image with it is taller
    no_ctx = _builder(tmp_path, w, {1: RED}, send_context_frames=False).build(ev)
    assert no_ctx is not None and _decode(no_ctx).shape[0] >= 160
    assert _decode(jpeg).shape[0] > _decode(no_ctx).shape[0]


def test_frames_are_read_in_one_sorted_pass_per_chunk(tmp_path, monkeypatch):
    from dnt.refine import evidence

    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    evs = []
    for i in range(3):
        e = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static",
                spans=None)
        e.id = f"screen-r0-{i:06d}"
        evs.append(e)
    opened = []
    real = evidence.FrameReader

    def counting(path):
        opened.append(path)
        return real(path)

    monkeypatch.setattr(evidence, "FrameReader", counting)
    b.build_many(evs)
    assert len(opened) == 1


def test_a_video_shorter_than_frame_count_skips_the_unreadable_frames(tmp_path, caplog):
    w = _work(_walker(1, range(60)), _walker(2, [10, 125], x0=150.0), _walker(3, [125, 126]))
    rows = []
    for raw, color in {1: RED, 2: BLUE}.items():
        rows += video_rows(w[w.raw_id == raw].assign(track=raw).pipe(_as_rows), color)
    video = make_color_video(tmp_path / "v.mp4", rows, N)  # 120 frames
    b = EvidenceBuilder(video, w, pd.Series(False, index=w.index), frame_count=130)
    e1 = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    e2 = _ev(
        EventKind.DROP, "screen", [2], [[[2, 10, 125]]], (10, 125), reason="static", spans=None
    )
    e2.id = "screen-r0-000002"
    e3 = _ev(
        EventKind.DROP, "screen", [3], [[[3, 125, 126]]], (125, 126), reason="static", spans=None
    )
    e3.id = "screen-r0-000003"
    with caplog.at_level("WARNING", logger="dnt.refine.evidence"):
        out = b.build_many([e1, e2, e3])
    assert set(out) == {e1.id, e2.id, e3.id}
    assert out[e1.id] is not None  # unaffected
    assert out[e2.id] is not None  # frame 10 is drawn, frame 125 cannot be read
    assert out[e3.id] is None  # nothing readable to draw
    assert any("skipping frame" in r.message for r in caplog.records)


def test_an_unopenable_video_gives_none_for_every_event(tmp_path, caplog):
    w = _work(_walker(1, range(60)))
    b = EvidenceBuilder(tmp_path / "missing.mp4", w, pd.Series(False, index=w.index), frame_count=N)
    evs = []
    for i in range(2):
        e = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
        e.id = f"screen-r0-{i:06d}"
        evs.append(e)
    with caplog.at_level("WARNING", logger="dnt.refine.evidence"):
        out = b.build_many(evs)
    assert out == {e.id: None for e in evs}
    assert any("cannot open video" in r.message for r in caplog.records)


@pytest.mark.parametrize("frame_count", [0, -1])
def test_an_unknown_frame_count_does_not_bound_the_evidence(tmp_path, frame_count):
    # raw .h264, some .ts/.mkv, and streams report 0 (or -1) frames: that is "unknown", not
    # "empty", so every box still qualifies and an unreadable frame is skipped when read
    w = _work(_walker(1, range(60)))
    rows = video_rows(w.assign(track=1).pipe(_as_rows), RED)
    video = make_color_video(tmp_path / "v.mp4", rows, N)
    b = EvidenceBuilder(video, w, pd.Series(False, index=w.index), frame_count=frame_count)
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    plan = b.plan(ev)
    assert len(plan.rows[0][1]) == 6 and len(plan.contexts) == 1
    jpeg = b.build(ev)
    assert jpeg is not None and _decode(jpeg).shape[0] > 160
    split = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    assert len(b.plan(split).contexts) == 1 and b.build(split) is not None


def test_the_index_is_built_lazily_for_the_raw_ids_the_events_reference(tmp_path):
    w = _work(_walker(1, range(60)), _walker(2, range(60), x0=255.0), _walker(3, range(60)))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    assert b._box == {} and b._occ == {} and b._loaded == set()  # nothing indexed yet
    drop = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static",
               spans=None)
    b.plan(drop)
    assert b._loaded == {1} and {r for r, _ in b._box} == {1} and len(b._box) == 60
    link = _ev(EventKind.LINK, "link", [1, 2], [[[1, 0, 29]], [[2, 40, 59]]], (29, 40),
               gap=[29, 40], gate="normal")
    b.build_many([link])
    assert b._loaded == {1, 2}  # raw 3 is never indexed
    # a split's context frame shows the other boxes near the cut: only that frame is read
    split = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    plan = b.plan(split)
    assert b._loaded == {1, 2} and (3, 30) in b._box and (3, 31) not in b._box
    assert len(plan.contexts[0].boxes) == 2  # its own box and raw 3 nearby (raw 2 is far)


def test_the_lazy_index_matches_the_rows_and_keeps_the_last_duplicate(tmp_path):
    w = _work(_walker(1, range(10)))
    dup = w[w.frame == 4].assign(x=99.0)
    w2 = pd.concat([w, dup], ignore_index=True)
    occ = pd.Series(False, index=w2.index)
    occ.iloc[-1] = True
    b = EvidenceBuilder(tmp_path / "v.mp4", w2, occ, frame_count=N)
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 9]]], (0, 9), reason="static", spans=None)
    b.plan(ev)
    assert b._box[(1, 4)][0] == 99.0 and b._occ[(1, 4)] is True  # the last row wins, as before
    assert b._box[(1, 3)] == tuple(float(v) for v in w.loc[w.frame == 3, ["x", "y", "w", "h"]]
                                   .iloc[0])
    # the builder keeps its own copy: later edits of the table do not change the evidence
    w2.loc[w2.frame == 5, "x"] = -1.0
    b2 = EvidenceBuilder(tmp_path / "v.mp4", w, pd.Series(False, index=w.index), frame_count=N)
    w.loc[w.frame == 5, "x"] = -1.0
    b2.plan(ev)
    assert b2._box[(1, 5)][0] != -1.0


def test_chunks_are_cut_after_sorting_the_events_by_their_first_frame(tmp_path, monkeypatch):
    from dnt.refine import evidence

    w = _work(_walker(1, range(0, 20)), _walker(2, range(90, 110), x0=150.0))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE}, send_context_frames=False)
    evs = []
    for i, raw in enumerate([2, 1, 2, 1]):  # late, early, late, early
        f0 = 0 if raw == 1 else 90
        e = _ev(EventKind.DROP, "screen", [raw], [[[raw, f0, f0 + 19]]], (f0, f0 + 19),
                reason="static", spans=None)
        e.id = f"screen-r0-{i:06d}"
        evs.append(e)
    ranges = []
    real = evidence.FrameReader

    class Recording(real):
        def frames(self, wanted):
            wanted = list(wanted)
            ranges.append((min(wanted), max(wanted)))
            return super().frames(wanted)

    monkeypatch.setattr(evidence, "FrameReader", Recording)
    monkeypatch.setattr(evidence, "CHUNK", 2)
    out = b.build_many(evs)
    assert set(out) == {e.id for e in evs} and all(v is not None for v in out.values())
    assert ranges == [(0, 19), (90, 109)]  # each chunk reads one compact range


def _merge_ev(lin_a, lin_b, span):
    ev = Event.propose(stage="dedup", kind=EventKind.MERGE, tracks=[1, 2],
                       lineage=[lin_a, lin_b], frames=tuple(span),
                       params={"span": list(span)}, algo_score=0.9, signals={})
    ev.id = "dedup-r0-000002"
    return ev


def test_merge_context_draws_an_interpolated_dashed_b_when_b_is_not_observed_there(tmp_path):
    w = _work(_walker(1, range(0, 60, 2)), _walker(2, range(1, 60, 2)))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    plan = b.plan(_merge_ev([[1, 0, 58]], [[2, 1, 59]], (1, 58)))
    (ctx,) = plan.contexts
    assert b._box_at([[2, 1, 59]], ctx.frame) is None
    (box_b,) = [c for c in ctx.boxes if c.label == "B"]
    assert box_b.xywh is not None and box_b.dashed
    (box_a,) = [c for c in ctx.boxes if c.label == "A"]
    assert not box_a.dashed
    assert abs(box_b.xywh[0] - box_a.xywh[0]) <= 2.0  # walkers move 1 px/frame


def test_merge_context_keeps_a_solid_b_when_b_is_observed_at_the_frame(tmp_path):
    w = _work(_walker(1, range(0, 60)), _walker(2, range(0, 60), x0=100.0))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    plan = b.plan(_merge_ev([[1, 0, 59]], [[2, 0, 59]], (0, 59)))
    (ctx,) = plan.contexts
    (box_b,) = [c for c in ctx.boxes if c.label == "B"]
    assert box_b.xywh is not None and not box_b.dashed


def test_merge_plan_shows_each_track_in_its_own_row(tmp_path):
    w = _work(_walker(1, range(0, 60, 2)), _walker(2, range(1, 60, 2)))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    ev = Event.propose(stage="dedup", kind=EventKind.MERGE, tracks=[1, 2],
                       lineage=[[[1, 0, 58]], [[2, 1, 59]]], frames=(1, 58),
                       params={"span": [1, 58]}, algo_score=0.9, signals={})
    ev.id = "dedup-r0-000001"
    plan = b.plan(ev)
    (label_a, a), (label_b, bb) = plan.rows
    assert (label_a, label_b) == ("A", "B")
    assert 0 < len(a) <= 6 and 0 < len(bb) <= 6
    assert {raw for raw, _ in a} == {1} and {raw for raw, _ in bb} == {2}
    assert all(1 <= f <= 58 for _, f in a + bb)  # only frames inside the span
    assert len(plan.contexts) == 1
    assert b.build(ev) is not None
