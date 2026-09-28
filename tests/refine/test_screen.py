# tests/refine/test_screen.py
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind
from dnt.refine.hints import ReclassHint, read_reclass_hints
from dnt.refine.screen import ScreenContext, propose_screen
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
