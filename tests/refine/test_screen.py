# tests/refine/test_screen.py
import numpy as np
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind
from dnt.refine.hints import ReclassHint, read_reclass_hints
from dnt.refine.primitives import iob_matrix, ramp
from dnt.refine.screen import (
    ScreenContext,
    _DupIndex,
    _duplicate,
    _unit,
    propose_screen,
)
from dnt.refine.verify import Band, route_without_vlm

from ._fixtures import box_rows, table

FPS = 10.0


def _work(*rows):
    return io.to_work(table(*rows)).work


def _ctx_tracks(*rows):
    t = table(*rows)
    return pd.DataFrame({"frame": t.frame, "track": t.track, "x": t.x, "y": t.y, "w": t.w,
                         "h": t.h, "cls": t.cls}), "tracks"


def _ctx_dets(*rows):
    t = table(*rows)
    return pd.DataFrame({"frame": t.frame, "track": -1, "x": t.x, "y": t.y, "w": t.w, "h": t.h,
                         "cls": t.cls}), "dets"


def _screen(work, cfg=None, ctx=None, hints=None, split=None, cuts=None):
    cfg = cfg or RefineConfig.defaults()
    boxes, fmt = ctx if ctx else (None, None)
    sctx = ScreenContext(boxes=boxes, fmt=fmt, hints=hints or {}, split_raw_ids=split or set())
    evs = propose_screen(work, cfg, FPS, sctx, cuts or {})
    route_without_vlm(evs, Band.of(cfg.screen))
    return {e.tracks[0]: e for e in evs}


def _ped(track, frames, x0=100.0, vx=3.0):
    return box_rows(track, frames, x0, 110.0, vx=vx, w=20.0, h=40.0)


def _car(track, frames, x0=80.0, vx=3.0, cls=2):
    return box_rows(track, frames, x0, 100.0, vx=vx, w=80.0, h=60.0, cls=cls)


def test_static_low_conf_hotspot_is_capped_and_pending():
    rows = [box_rows(t, range(120 * (t - 1), 120 * t), 300.0, 200.0, score=0.35)
            for t in (1, 2, 3)]
    evs = _screen(_work(*rows))
    for t in (1, 2, 3):
        ev = evs[t]
        assert ev.kind is EventKind.DROP and ev.params["reason"] == "static"
        assert ev.algo_score == pytest.approx(0.8)
        assert ev.decision is Decision.HUMAN_PENDING


def test_static_cue_ignores_missing_scores():
    ev = _screen(_work(box_rows(1, range(120), 300.0, 200.0, score=-1)))[1]
    assert ev.signals["C"] is None and ev.params["reason"] == "static"


def test_person_inside_moving_car_is_dropped():
    ev = _screen(_work(_ped(1, range(50))), ctx=_ctx_tracks(_car(9, range(50))))[1]
    assert ev.params["reason"] == "in_vehicle" and ev.decision is Decision.AUTO_ACCEPT


def test_boarding_passenger_is_kept():
    evs = _screen(_work(_ped(1, range(50))), ctx=_ctx_tracks(_car(9, range(10))))
    assert 1 not in evs


def test_detection_context_uses_persistence():
    ev = _screen(_work(_ped(1, range(50))), ctx=_ctx_dets(_car(9, range(50))))[1]
    assert ev.params["reason"] == "in_vehicle" and ev.algo_score == pytest.approx(1.0)


def test_fast_smooth_person_is_a_rider_and_walker_is_not():
    evs = _screen(_work(_ped(1, range(50), vx=12.0), _ped(2, range(50), x0=900.0, vx=3.2)))
    assert evs[1].kind is EventKind.RECLASS and evs[1].params["new_cls"] is None
    assert evs[1].decision is Decision.HUMAN_PENDING  # subtype still needed
    assert 2 not in evs


def test_localized_hint_settles_subtype():
    ev = _screen(_work(_ped(1, range(50), vx=12.0)),
                 hints={1: ReclassHint(1, 3, 0.95)})[1]
    assert ev.params["new_cls"] == 3 and ev.signals["subtype_source"] == "reclass"
    assert ev.decision is Decision.AUTO_ACCEPT


def test_hint_is_unlocalized_after_split():
    walk = _ped(1, range(0, 50), vx=3.0)
    ride = _ped(2, range(50, 100), x0=250.0, vx=12.0)
    solo = _ped(5, range(0, 50), x0=2000.0, vx=12.0)
    w = _work(walk, ride, solo)
    w.loc[w["track"] == 2, "raw_id"] = 1  # track 2 is the tail of a split of raw track 1
    hints = {1: ReclassHint(1, 3, 0.95), 5: ReclassHint(5, 3, 0.95)}
    evs = _screen(w, hints=hints, split={1})
    assert 1 not in evs
    assert evs[2].params["new_cls"] is None and evs[2].signals["hint_unlocalized"]["cls"] == 3
    assert evs[5].params["new_cls"] == 3


def test_vehicle_duplicate_drops_the_smaller():
    big = box_rows(1, range(30), 100.0, 100.0, vx=2.0, w=120.0, h=80.0, cls=2)
    small = box_rows(2, range(30), 110.0, 110.0, vx=2.0, w=50.0, h=40.0, cls=2)
    evs = _screen(_work(big, small), cfg=RefineConfig.defaults("vehicle"))
    assert 1 not in evs
    assert evs[2].params == {"reason": "duplicate", "of": 1, "spans": None}


def test_pending_split_gives_partial_drop():
    w = _work(_ped(1, range(100)))
    ctx = _ctx_tracks(_car(9, range(50, 100), x0=80.0 + 150.0))
    ev = _screen(w, ctx=ctx, cuts={1: [50]})[1]
    assert ev.params["spans"] == [[50, 99]] and ev.signals["partial"] is True
    assert ev.algo_score == pytest.approx(0.75) and ev.decision is Decision.HUMAN_PENDING


def test_applied_split_drops_only_the_passenger_track():
    w = _work(_ped(1, range(50)), _ped(2, range(50, 100), x0=250.0))
    ctx = _ctx_tracks(_car(9, range(50, 100), x0=230.0))
    evs = _screen(w, ctx=ctx)
    assert 1 not in evs and evs[2].params["reason"] == "in_vehicle"
    assert evs[2].params["spans"] is None


def test_every_segment_static_gives_a_whole_track_event():
    rows = (box_rows(1, range(0, 120), 300.0, 200.0, score=0.35)
            + box_rows(1, range(120, 240), 900.0, 500.0, score=0.35))
    ev = _screen(_work(rows), cuts={1: [120]})[1]  # whole-track static score alone would be 0
    assert ev.params["reason"] == "static" and ev.params["spans"] is None
    assert ev.signals["all_segments"] is True
    # per segment: ramp_R = ramp_T = 1; mean(ramp_J = 1, ramp_H = 0, ramp_C = 0.25 / 0.3)
    assert ev.algo_score == pytest.approx((1 + 0 + 0.25 / 0.3) / 3, abs=1e-3)


def test_context_with_no_overlapping_frames_is_harmless():
    w = _work(_ped(1, range(50)))
    assert _screen(w, ctx=_ctx_tracks(_car(9, range(500, 550)))) == {}
    empty = pd.DataFrame(columns=["frame", "track", "x", "y", "w", "h", "cls"])
    assert _screen(w, ctx=(empty, "tracks")) == {}


def test_single_row_track_has_no_screen_event():
    assert _screen(_work(box_rows(1, [7], 0.0, 0.0))) == {}


def test_read_reclass_hints(tmp_path, caplog):
    p = tmp_path / "h.csv"
    p.write_text("track,cls,avg_score\n1,3,0.95\n99,1,0.8\n")
    hints = read_reclass_hints(p, {1, 2})
    assert hints == {1: ReclassHint(1, 3, 0.95)} and "unknown" in caplog.text
    (tmp_path / "bad.csv").write_text("id,cls\n1,3\n")
    with pytest.raises(ValueError, match="track, cls, avg_score"):
        read_reclass_hints(tmp_path / "bad.csv", {1})


# ---- fix round 1: rules pinned one by one ----

def _events(work, cfg=None, ctx=None, hints=None, split=None, cuts=None):
    """Like ``_screen`` but keep every event (a list, not a per-track dict)."""
    cfg = cfg or RefineConfig.defaults()
    boxes, fmt = ctx if ctx else (None, None)
    sctx = ScreenContext(boxes=boxes, fmt=fmt, hints=hints or {}, split_raw_ids=split or set())
    evs = propose_screen(work, cfg, FPS, sctx, cuts or {})
    route_without_vlm(evs, Band.of(cfg.screen))
    return evs


def _ctx_extra(rows_by_cls):
    """Track-format context from ``[(cls, rows), ...]``."""
    parts = []
    for cls, rows in rows_by_cls:
        t = table(rows)
        parts.append(pd.DataFrame({"frame": t.frame, "track": t.track, "x": t.x, "y": t.y,
                                   "w": t.w, "h": t.h, "cls": cls}))
    return pd.concat(parts, ignore_index=True), "tracks"


def _still(track, frames, x0, y0, score):
    return box_rows(track, frames, x0, y0, w=30.0, h=60.0, score=score)


def test_weakest_segment_scores_the_whole_track_event():
    # Two static spots of unequal strength (mean confidence 0.5 vs 0.3), a pending cut between.
    # Per segment: ramp_R = ramp_T = 1, ramp_J = 1, ramp_H = 0 (only the segment itself is near).
    rows = (_still(1, range(0, 120), 300.0, 200.0, 0.5)
            + _still(1, range(120, 240), 900.0, 500.0, 0.3))
    ev = _screen(_work(rows), cuts={1: [120]})[1]
    weak, strong = (1 + 0 + 1 / 3) / 3, (1 + 0 + 1) / 3  # ramp_C = 0.1 / 0.3 and 1
    assert [s[2] for s in ev.signals["segments"]] == [
        pytest.approx(weak, abs=1e-6), pytest.approx(strong, abs=1e-6)]
    assert ev.params["spans"] is None and ev.signals["all_segments"] is True
    assert ev.algo_score == pytest.approx(weak, abs=1e-6)  # the LOWER segment, not max or mean


def _person_static_car(vx_person, jump=False, spread=None):
    """A person walking at ``vx_person`` inside wide vehicle detections that move with it."""
    ped = box_rows(1, range(40), 255.0, 110.0, vx=vx_person, w=20.0, h=40.0)
    if jump:  # two 400-wide vehicle boxes alternate: each holds the person, consecutive IoU < 0.5
        car = [box_rows(9, [f], 0.0 if f % 2 == 0 else 250.0, 100.0, w=400.0, h=60.0, cls=2)[0]
               for f in range(40)]
    else:
        car = box_rows(9, range(40), 250.0, 100.0, w=400.0, h=60.0, cls=2)
    return _work(ped), _ctx_dets(car)


def test_detection_context_needs_a_moving_person():
    w, ctx = _person_static_car(3.0)  # 0.75 box heights per second, above motion.moving_min
    assert _screen(w, ctx=ctx)[1].params["reason"] == "in_vehicle"
    w, ctx = _person_static_car(0.5)  # 0.125 box heights per second, below moving_min
    assert _screen(w, ctx=ctx) == {}


def test_detection_context_needs_persistent_vehicle_boxes():
    w, ctx = _person_static_car(3.0, jump=False)
    assert _screen(w, ctx=ctx)[1].signals["inside_frac"] == pytest.approx(39 / 40)
    w, ctx = _person_static_car(3.0, jump=True)
    assert _screen(w, ctx=ctx) == {}


def test_detection_context_needs_containment_at_the_previous_frame():
    # A person walks through a stationary vehicle box: the first contained frame has no
    # contained predecessor, so it does not count.
    ped = _ped(1, range(50), x0=170.0, vx=3.0)
    car = box_rows(9, range(50), 200.0, 100.0, w=200.0, h=60.0, cls=2)
    boxes = table(ped)[["x", "y", "w", "h"]].to_numpy()
    cbox = table(car)[["x", "y", "w", "h"]].to_numpy()[0]
    inter = (np.minimum(boxes[:, 0] + 20, cbox[0] + 200) - np.maximum(boxes[:, 0], cbox[0])).clip(0)
    inside = int(np.count_nonzero(inter / 20.0 >= 0.8))
    assert 30 < inside < 45
    ev = _screen(_work(ped), ctx=_ctx_dets(car))[1]
    assert ev.signals["inside_frac"] == pytest.approx((inside - 1) / 50)


def _lone_and_neighbours(offset, n_near=2, score=0.5):
    rows = [_still(1, range(120), 300.0, 200.0, score)]
    rows += [_still(2 + k, range(120), 300.0 + offset, 200.0 + 3.0 * k, score)
             for k in range(n_near)]
    return _work(*rows)


def test_hotspot_term_raises_the_static_score():
    # conf 0.5 keeps the score under the 0.8 cap: mean(J = 1, H, C = 1 / 3), R = T = 1.
    lone = _screen(_lone_and_neighbours(10.0, n_near=0))[1]
    hot = _screen(_lone_and_neighbours(10.0, n_near=2))[1]  # 10 px < 0.5 * 60 px
    far = _screen(_lone_and_neighbours(100.0, n_near=2))[1]  # 100 px > 0.5 * 60 px
    assert (lone.signals["H"], hot.signals["H"], far.signals["H"]) == (1, 3, 1)
    assert lone.algo_score == pytest.approx((1 + 0 + 1 / 3) / 3, abs=1e-6)
    assert hot.algo_score == pytest.approx((1 + 2 / 3 + 1 / 3) / 3, abs=1e-6)
    assert far.algo_score == pytest.approx(lone.algo_score)


def _pair(track_a, track_b, box_a, box_b, frames_a, frames_b, vx=2.0):
    """Two vehicle tracks moving with velocity ``vx`` along the same path."""
    fa, fb = list(frames_a), list(frames_b)
    return [box_rows(track_a, fa, box_a[0], 100.0, vx=vx, w=box_a[1], h=box_a[2], cls=2),
            box_rows(track_b, fb, box_b[0] + vx * (fb[0] - fa[0]), 100.0 + box_b[3], vx=vx,
                     w=box_b[1], h=box_b[2], cls=2)]


def _dups(rows):
    evs = _events(_work(*rows), cfg=RefineConfig.defaults("vehicle"))
    return [(e.tracks[0], e.params["of"]) for e in evs if e.params["reason"] == "duplicate"]


def test_equal_boxes_drop_the_higher_numbered_track_once():
    rows = _pair(4, 7, (100.0, 100.0, 70.0), (100.0, 100.0, 70.0, 0.0), range(30), range(30))
    assert _dups(rows) == [(7, 4)]


def test_smaller_area_decides_when_iob_holds_both_ways():
    # The 90 x 63 box lies inside the 100 x 70 box, so IoB is >= 0.7 in both directions
    # (1.0 and 0.81); only the smaller-area rule picks track 1 (the LOWER number) as the drop.
    rows = _pair(1, 2, (105.0, 90.0, 63.0), (100.0, 100.0, 70.0, -3.0), range(30), range(30))
    assert _dups(rows) == [(1, 2)]


def test_duplicate_needs_enough_shared_frames():
    def dups(shared):
        rows = _pair(1, 2, (100.0, 100.0, 70.0), (100.0, 100.0, 70.0, 0.0),
                     range(40), range(40 - shared, 80))
        return _dups(rows)
    assert dups(9) == []  # duplicate_min_frames is 10
    assert dups(10) == [(2, 1)]


def test_slow_person_with_a_localized_hint_is_a_rider_from_the_hint_alone():
    walker = _work(_ped(1, range(50), vx=1.0))
    assert _screen(walker) == {}
    ev = _screen(walker, hints={1: ReclassHint(1, 3, 0.95)})[1]
    assert ev.kind is EventKind.RECLASS and ev.signals["F"] == 0.0
    assert ev.algo_score == pytest.approx(1.0)
    assert ev.params["new_cls"] == RefineConfig.defaults().reclass_map["motorcycle"]
    assert ev.signals["subtype_source"] == "reclass" and ev.decision is Decision.AUTO_ACCEPT


def test_hint_class_outside_the_map_is_ignored():
    hints = {1: ReclassHint(1, 99, 0.99)}
    assert _screen(_work(_ped(1, range(50), vx=1.0)), hints=hints) == {}
    ev = _screen(_work(_ped(1, range(50), vx=12.0)), hints=hints)[1]  # rider by motion
    assert ev.signals["P"] is None and ev.params["new_cls"] is None
    assert "subtype_source" not in ev.signals


def test_hint_between_the_ramp_ends_raises_the_score_but_not_the_subtype():
    walker = _work(_ped(1, range(50), vx=1.0))
    ev = _screen(walker, hints={1: ReclassHint(1, 3, 0.87)})[1]  # ramp 0.75..0.9 -> 0.8
    assert ev.kind is EventKind.RECLASS and ev.algo_score == pytest.approx(0.8)
    assert ev.params["new_cls"] is None and "subtype_source" not in ev.signals
    assert ev.decision is Decision.HUMAN_PENDING


def test_slow_person_on_a_moving_two_wheeler_is_a_rider_from_k_alone():
    ped = _work(_ped(1, range(50), vx=1.0))
    bike = box_rows(9, range(50), 100.0, 110.0, vx=1.0, w=30.0, h=50.0)
    ev = _screen(ped, ctx=_ctx_extra([(3, bike)]))[1]
    assert ev.kind is EventKind.RECLASS and ev.signals["F"] == 0.0
    assert ev.signals["K"] == 1.0 and ev.algo_score == pytest.approx(1.0)
    assert _screen(ped, ctx=_ctx_extra([(9, bike)])) == {}  # class 9 is no two-wheeler


def test_jitter_lowers_the_static_score():
    frames = range(120)
    still = _still(1, frames, 300.0, 200.0, 0.3)
    shaky = _still(1, frames, 300.0, 200.0, 0.3)
    for row in shaky[::2]:
        row[2] += 1.2  # a 1.2 px step every frame = 0.02 box heights
    quiet, noisy = _screen(_work(still))[1], _screen(_work(shaky))[1]
    assert quiet.signals["J"] == pytest.approx(0.0)
    assert noisy.signals["J"] == pytest.approx(0.02)
    assert quiet.algo_score == pytest.approx((1 + 0 + 1) / 3, abs=1e-6)
    assert noisy.algo_score == pytest.approx((0.4 + 0 + 1) / 3, abs=1e-6)  # ramp_J(0.02) = 0.4


def test_short_static_track_gets_no_event():
    assert _screen(_work(_still(1, range(15), 300.0, 200.0, 0.3))) == {}  # 1.5 s < ramp_T low
    assert _screen(_work(_still(1, range(120), 300.0, 200.0, 0.3)))[1].params["reason"] == "static"


def test_partial_static_covers_only_the_supporting_segment():
    rows = (_still(1, range(0, 120), 300.0, 200.0, 0.35)
            + box_rows(1, range(120, 240), 300.0, 200.0, vx=6.0, w=30.0, h=60.0, score=0.35))
    ev = _screen(_work(rows), cuts={1: [120]})[1]
    assert ev.params["spans"] == [[0, 119]] and ev.signals["partial"] is True
    assert ev.params["reason"] == "static" and ev.frames == (0, 119)
    assert ev.algo_score == pytest.approx((1 + 0 + 0.25 / 0.3) / 3, abs=1e-6)


def test_partial_rider_covers_only_the_fast_segment_and_hint_is_unlocalized():
    walk = _ped(1, range(0, 50), vx=3.0)
    ride = box_rows(1, range(50, 100), 250.0, 110.0, vx=12.0, w=20.0, h=40.0)
    ev = _screen(_work(walk, ride), hints={1: ReclassHint(1, 3, 0.95)}, cuts={1: [50]})[1]
    assert ev.kind is EventKind.RECLASS and ev.params["spans"] == [[50, 99]]
    assert ev.signals["partial"] is True and ev.algo_score == pytest.approx(0.75)  # mixed cap
    assert ev.params["new_cls"] is None and ev.signals["hint_unlocalized"]["cls"] == 3


# ---- the vectorized duplicate cue equals the naive pairwise one ----

def _naive_duplicate(u, others, cfg, fps):
    """The original pairwise, per-frame duplicate cue (kept as the reference)."""
    sc = cfg.screen
    area_u = float(np.median(u.boxes[:, 2] * u.boxes[:, 3]))
    best = None
    for tid, o in others.items():
        if tid == u.track:
            continue
        common, iu, io_ = np.intersect1d(u.frames, o.frames, return_indices=True)
        if len(common) < sc.duplicate_min_frames:
            continue
        area_o = float(np.median(o.boxes[:, 2] * o.boxes[:, 3]))
        if area_u > area_o or (area_u == area_o and u.track < tid):
            continue
        ok = 0
        for a, b in zip(iu, io_, strict=True):
            iob = iob_matrix(u.boxes[a : a + 1], o.boxes[b : b + 1])[0, 0]
            dv = float(np.linalg.norm(u.vel[a] - o.vel[b]) / u.hmed * FPS)
            ok += int(iob >= sc.duplicate_iob and dv < sc.move_together)
        frac = ok / len(common)
        s = ramp(frac, *sc.ramps["D"])
        if best is None or s > best[0]:
            best = (s, {"D": frac}, {"reason": "duplicate", "of": int(tid)})
    return best


def _planted_duplicate_table(seed=11, n_objects=14, n_frames=120):
    rng = np.random.default_rng(seed)
    rows, tid = [], 1
    for _ in range(n_objects):
        start = int(rng.integers(0, n_frames // 2))
        frames = range(start, start + int(rng.integers(20, n_frames // 2)))
        x0, y0 = rng.uniform(0, 400), rng.uniform(0, 200)
        vx, vy = rng.uniform(-4, 4), rng.uniform(-2, 2)
        w, h = rng.uniform(40, 120), rng.uniform(30, 80)
        rows.append(box_rows(tid, frames, x0, y0, vx=vx, vy=vy, w=w, h=h, cls=2))
        for _ in range(int(rng.integers(0, 3))):  # planted duplicates of every flavour
            k = rng.choice([1.0, 0.9, 0.7, 0.4])  # equal size, near equal, smaller, too small
            sub = list(frames)[int(rng.integers(0, 6)): len(frames) - int(rng.integers(0, 6))]
            dvx = vx + (0.0 if rng.random() < 0.7 else 5.0)
            dup = box_rows(tid + 1, sub, x0 + (sub[0] - frames[0]) * vx + w * (1 - k) / 2,
                           y0 + (sub[0] - frames[0]) * vy + h * (1 - k) / 2,
                           vx=dvx, vy=vy, w=w * k, h=h * k, cls=2)
            rows.append(dup)
            tid += 1
        tid += 1
    return _work(*rows)


def test_fast_duplicate_cue_matches_the_naive_reference():
    cfg = RefineConfig.defaults("vehicle")
    work = _planted_duplicate_table()
    units = {int(t): _unit(t, g.sort_values("frame"), cfg, FPS) for t, g in work.groupby("track")}
    index = _DupIndex(units)
    hits = 0
    for u in units.values():
        fast, naive = _duplicate(u, index, cfg, FPS), _naive_duplicate(u, units, cfg, FPS)
        assert (fast is None) == (naive is None)
        if fast is not None:
            assert fast[0] == pytest.approx(naive[0], abs=1e-12) and fast[1] == naive[1]
            assert fast[2] == naive[2]
            hits += int(fast[0] > 0)
    assert hits >= 5  # the fixture really contains duplicates
    # segment units (a slice of one track) go through the same index
    g = work[work["track"] == next(iter(units))].sort_values("frame")
    seg = _unit(next(iter(units)), g.iloc[len(g) // 2 :], cfg, FPS)
    fast, naive = _duplicate(seg, index, cfg, FPS), _naive_duplicate(seg, units, cfg, FPS)
    assert (fast is None) == (naive is None) and (fast is None or fast[2] == naive[2])


# ---- hints validation ----

def test_hint_scores_must_be_finite_and_within_unit_interval(tmp_path):
    for i, bad in enumerate(("nan", "1.5", "-0.1", "abc")):
        p = tmp_path / f"s{i}.csv"
        p.write_text(f"track,cls,avg_score\n1,3,0.9\n2,3,{bad}\n")
        with pytest.raises(ValueError, match=rf"s{i}\.csv: row 3 \(track 2\): avg_score"):
            read_reclass_hints(p, {1, 2})


def test_hint_track_and_cls_must_be_integers(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("track,cls,avg_score\n1.5,3,0.9\n")
    with pytest.raises(ValueError, match=r"t\.csv: row 2: track 1\.5 is not an integer"):
        read_reclass_hints(p, {1})
    p = tmp_path / "c.csv"
    p.write_text("track,cls,avg_score\n1,x,0.9\n")
    with pytest.raises(ValueError, match=r"c\.csv: row 2: cls 'x' is not an integer"):
        read_reclass_hints(p, {1})
    p = tmp_path / "n.csv"
    p.write_text("track,cls,avg_score\n1,,0.9\n")
    with pytest.raises(ValueError, match=r"n\.csv: row 2: cls"):
        read_reclass_hints(p, {1})


def test_repeated_hint_rows_warn_and_keep_the_last(tmp_path, caplog):
    p = tmp_path / "r.csv"
    p.write_text("track,cls,avg_score\n1,3,0.8\n2,1,0.9\n1,36,0.95\n")
    hints = read_reclass_hints(p, {1, 2})
    assert hints[1] == ReclassHint(1, 36, 0.95) and hints[2] == ReclassHint(2, 1, 0.9)
    assert "repeated" in caplog.text
    assert read_reclass_hints(p, {2}) == {2: ReclassHint(2, 1, 0.9)}  # the repeats are unknown
