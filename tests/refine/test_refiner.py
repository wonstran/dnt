import logging
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine import refiner as refiner_mod
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.features import ArrayAppearance
from dnt.refine.refiner import TrackRefiner, resolve_fps

from ._fixtures import box_rows, load_raw, random_tracks, table


def _write(dirpath, df, name="t.txt"):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    df.to_csv(p, index=False, header=False)
    return p


def _refine(src, out, cfg=None, **kw):
    refiner = TrackRefiner(cfg)
    df = refiner.refine(src, out, verbose=False, **kw)
    assert df is refiner.last_result.tracks
    return refiner.last_result


def _crit1_rows(offset=0):
    rows = []
    for f in range(60):  # track 1: a car taken over by an untracked truck at frame 30
        truck = f >= 30
        cls = 5 if f == 30 else 7 if (f == 28 or f > 30) else 2
        rows.append([f + offset, 1, 100.0 + f, 100.0, 95.0 if truck else 53.0,
                     60.0 if truck else 56.0, 0.9, cls, -1, -1])
    rows += box_rows(2, [f + offset for f in range(120)], 400.0, 300.0, score=0.35)  # static
    rows += box_rows(3, [f + offset for f in range(40)], 600.0, 200.0, vx=3.0)  # fragment
    rows += box_rows(4, [f + offset for f in range(45, 90)], 735.0, 200.0, vx=3.0)
    return rows


def test_default_config_no_video_all_three_stages_score(tmp_path, monkeypatch):
    for name in ("transformers", "torchreid", "openai", "anthropic"):
        monkeypatch.setitem(sys.modules, name, None)  # a minimal install
    src = _write(tmp_path, table(_crit1_rows()))
    res = _refine(src, tmp_path / "o.csv", fps=10)
    assert {"switch", "screen", "link"} <= {e.stage for e in res.events}
    for e in res.events:
        if e.stage in ("switch", "screen"):
            assert e.decision is Decision.HUMAN_PENDING
    link = [e for e in res.events if e.stage == "link"]
    assert link[0].decision is Decision.AUTO_ACCEPT and link[0].applied
    header = Ledger.read(res.ledger_path).header
    assert header["fps"] == 10 and header["fps_source"] == "argument"
    assert header["inputs"]["tracks"]["sha256"] == io.sha256_file(src)
    out = pd.read_csv(tmp_path / "o.csv", header=None)
    assert out.shape[1] == 10
    assert sorted(out[1].unique()) == list(range(1, out[1].nunique() + 1))


def test_frame_offset_does_not_change_scores(tmp_path):
    def run(offset):
        d = tmp_path / f"o{offset}"
        res = _refine(_write(d, table(_crit1_rows(offset))), d / "o.csv", fps=10)
        return sorted((e.stage, str(e.kind), round(e.algo_score, 9)) for e in res.events)

    assert run(0) == run(120000)


def test_no_video_and_no_fps_fails_before_processing(tmp_path):
    src = _write(tmp_path, table(box_rows(1, range(5), 0.0, 0.0)))
    with pytest.raises(ValueError, match="fps="):
        _refine(src, tmp_path / "o.csv")
    assert not (tmp_path / "o.csv").exists()


def test_empty_track_file(tmp_path):
    (tmp_path / "e.txt").write_text("")
    res = _refine(tmp_path / "e.txt", tmp_path / "o.csv", fps=10)
    assert (tmp_path / "o.csv").read_text() == ""
    assert len(res.ledger_path.read_text().splitlines()) == 1 and res.events == []


def test_short_fragment_with_pending_link_is_deferred_not_dropped(tmp_path):
    rows = (box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0)
            + box_rows(2, [883], 160.0, 240.0, w=50.0, h=50.0)
            + box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=300.0))
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", fps=10)
    link = [e for e in res.events if e.stage == "link" and e.tracks == [1, 2]]
    assert link and link[0].decision is Decision.HUMAN_PENDING
    assert link[0].params["gate"] == "occluded"
    assert not any(e.stage == "orphan" for e in res.events)
    assert (pd.read_csv(tmp_path / "o.csv", header=None)[0] == 883).any()
    id_map = Ledger.read(res.ledger_path).header["id_map"]
    assert res.summary["orphan_deferred"] == [id_map["2"]]


def test_filled_input_rows_are_removed_and_refilled_once(tmp_path):
    df = table(box_rows(1, range(0, 10), 0.0, 0.0, vx=2.0),
               box_rows(1, range(15, 25), 30.0, 0.0, vx=2.0))
    filled = table(box_rows(1, range(10, 15), 20.0, 0.0))
    filled["r3"] = 1
    src = _write(tmp_path, pd.concat([df, filled]))
    res = _refine(src, tmp_path / "o.csv", fps=10)
    assert res.summary["filled_input_rows_removed"] == 5
    fills = [e for e in res.events if e.kind is EventKind.FILL]
    assert len(fills) == 1 and fills[0].params == {"gap": [9, 15], "n_rows": 5}
    assert fills[0].decision is Decision.AUTO_ACCEPT and fills[0].id == "fill-r0-000001"


def test_smoothing_writes_smooth_records(tmp_path):
    cfg = RefineConfig.defaults()
    cfg.fill.smooth_existing = True
    rng = np.random.default_rng(0)
    rows = box_rows(1, range(50), 100.0, 100.0, vx=2.0)
    for r in rows:
        r[2] += float(rng.normal(0, 3))
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", cfg=cfg, fps=10)
    sm = [e for e in res.events if e.kind is EventKind.SMOOTH]
    assert len(sm) == 1 and sm[0].params["n_rows"] > 0 and sm[0].signals["max_shift_px"] > 0
    assert Ledger.read(res.ledger_path).header["smoothing"] is True


def test_video_supplies_fps_frame_size_and_fingerprint(tmp_path, synthetic_video, caplog):
    video, _ = synthetic_video
    src = _write(tmp_path, table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)))
    with caplog.at_level(logging.WARNING):
        res = _refine(src, tmp_path / "o.csv", video_file=video)
    h = Ledger.read(res.ledger_path).header
    assert h["fps"] == pytest.approx(25.0) and h["fps_source"] == "video"
    assert h["frame_size"] == [320, 240]
    assert h["inputs"]["video"]["fingerprint"]["sha256"] == io.sha256_file(video)
    assert "motion-only" in caplog.text


def test_track_frames_beyond_the_video_are_rejected(tmp_path, synthetic_video):
    video, _ = synthetic_video
    src = _write(tmp_path, table(box_rows(1, range(495, 505), 0.0, 0.0)))
    with pytest.raises(ValueError, match="frame count"):
        _refine(src, tmp_path / "o.csv", video_file=video)


def test_refiner_api_matches_tracker(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg = RefineConfig.defaults("vehicle")
    cfg.to_yaml(cfg_file)
    refiner = TrackRefiner(config_yaml=str(cfg_file), device="cpu")
    assert refiner.config.target == "vehicle" and refiner.config.encoder.device == "cpu"
    with pytest.raises(ValueError, match="not both"):
        TrackRefiner(config=cfg, config_yaml=str(cfg_file))
    src = _write(tmp_path, table(box_rows(1, range(40), 100.0, 100.0, vx=2.0)))
    df = refiner.refine(src, tmp_path / "o.csv", fps=10, verbose=False)
    assert isinstance(df, pd.DataFrame) and refiner.last_result.ledger_path.exists()


def test_refine_batch_names_skips_and_overwrites(tmp_path):
    srcs = [_write(tmp_path / "in", table(box_rows(1, range(30), 0.0, 0.0, vx=2.0)),
                   f"cam{i}_track.txt") for i in (1, 2)]
    out_dir = tmp_path / "out"
    refiner = TrackRefiner()
    with pytest.raises(ValueError, match="output_path"):
        refiner.refine_batch(srcs, fps=10)
    first = refiner.refine_batch(srcs, output_path=out_dir, fps=10, verbose=False)
    assert first == [str(out_dir / "cam1_refined.txt"), str(out_dir / "cam2_refined.txt")]
    assert (out_dir / "cam1_refined.ledger.jsonl").exists()
    stamp = (out_dir / "cam1_refined.txt").stat().st_mtime_ns
    assert refiner.refine_batch(srcs, output_path=out_dir, fps=10, verbose=False) == first
    assert refiner.refine_batch(srcs, output_path=out_dir, fps=10, is_report=False,
                                verbose=False) == []
    assert (out_dir / "cam1_refined.txt").stat().st_mtime_ns == stamp
    again = refiner.refine_batch(srcs[:1], output_path=out_dir, fps=10, is_overwrite=True,
                                 verbose=False)
    assert again == first[:1]


# ---- rule-pinning, invariant, guard and timing tests (task 16 additions) --------------------

A_, B_ = np.eye(8)[0], np.eye(8)[1]


def _planted(offset=0):
    """A scene with every stage's trigger; ``offset`` shifts all frames.

    Raw track 1 is a person whose box and appearance change at frame 110 while a context car
    drives around the second half (a confident split, then a confident in-vehicle drop of the
    new track); 2 is a static low-confidence box (pending drop); 3 and 4 are a fragment pair
    (confident link, five filled rows); 5 is a passenger of context car 9 (confident drop);
    6 is a two-row orphan (confident drop).
    """
    o = offset
    rows = (
        box_rows(1, range(60 + o, 110 + o), 100.0, 300.0, vx=3.0, w=30.0, h=60.0)
        + box_rows(1, range(110 + o, 160 + o), 250.0, 300.0, vx=3.0, w=48.0, h=96.0)
        + box_rows(2, range(200 + o, 320 + o), 400.0, 500.0, score=0.35)
        + box_rows(3, range(20 + o, 60 + o), 600.0, 200.0, vx=3.0)
        + box_rows(4, range(65 + o, 110 + o), 735.0, 200.0, vx=3.0)
        + box_rows(5, range(100 + o, 150 + o), 100.0, 810.0, vx=3.0, w=20.0, h=40.0)
        + box_rows(6, [30 + o, 31 + o], 1500.0, 800.0)
    )
    ctx = (
        box_rows(9, range(100 + o, 150 + o), 80.0, 800.0, vx=3.0, w=80.0, h=60.0, cls=2)
        + box_rows(8, range(110 + o, 160 + o), 230.0, 280.0, vx=3.0, w=120.0, h=140.0, cls=2)
    )
    frames = list(range(60 + o, 160 + o))
    app = ArrayAppearance({
        1: (frames, np.array([A_ if f < 110 + o else B_ for f in frames])),
        3: (list(range(20 + o, 60 + o)), np.tile(A_, (40, 1))),
        4: (list(range(65 + o, 110 + o)), np.tile(A_, (45, 1))),
    })
    return table(rows), table(ctx), app


def _run_planted(d, cfg=None, offset=0, out="o.txt", **kw):
    rows, ctx, app = _planted(offset)
    src = _write(d, rows)
    ctx_file = _write(d, ctx, "c.txt")
    refiner = TrackRefiner(cfg, appearance_factory=lambda **_kw: app)
    refiner.refine(src, d / out, context_file=ctx_file, fps=10, verbose=False, **kw)
    return refiner.last_result


def _by(res, stage, kind=None):
    return [e for e in res.events if e.stage == stage and (kind is None or e.kind is kind)]


def _frames_of(res, track):
    t = res.tracks
    return sorted(t.loc[t["track"] == track, "frame"].tolist())


def test_confident_edits_are_applied_and_pending_edits_leave_the_tracks_alone(tmp_path):
    res = _run_planted(tmp_path)
    id_map = {int(k): v for k, v in Ledger.read(res.ledger_path).header["id_map"].items()}
    (split,) = _by(res, "switch")
    assert split.kind is EventKind.SPLIT and split.decision is Decision.AUTO_ACCEPT
    assert split.applied
    drops = {e.tracks[0]: e for e in _by(res, "screen", EventKind.DROP)}
    assert drops[5].params["reason"] == "in_vehicle" and drops[5].applied  # a passenger
    assert drops[7].params["reason"] == "in_vehicle" and drops[7].applied  # the split's tail
    assert drops[2].params["reason"] == "static"  # pending: not applied, rows kept
    assert drops[2].decision is Decision.HUMAN_PENDING and not drops[2].applied
    (link,) = _by(res, "link")
    assert link.tracks == [3, 4] and link.decision is Decision.AUTO_ACCEPT and link.applied
    (orphan,) = _by(res, "orphan")
    assert orphan.tracks == [6] and orphan.applied
    # the outputs: drops are gone, the pending drop is untouched, the link is one output id
    assert _frames_of(res, id_map[1]) == list(range(60, 110))
    assert _frames_of(res, id_map[2]) == list(range(200, 320))
    assert _frames_of(res, id_map[3]) == list(range(20, 110))
    assert set(id_map) == {1, 2, 3} and sorted(id_map.values()) == [1, 2, 3]
    assert res.tracks["track"].nunique() == 3 and len(res.tracks) == 50 + 120 + 90
    assert res.tracks["interp"].sum() == 5
    ids = [e.id for e in res.events]
    assert len(set(ids)) == len(ids)


def test_pending_split_and_pending_link_change_nothing(tmp_path):
    res = _refine(_write(tmp_path, table(_crit1_rows())), tmp_path / "o.csv", fps=10)
    pending = [e for e in res.events if e.stage in ("switch", "screen")]
    assert pending and all(not e.applied for e in pending)
    id_map = {int(k): v for k, v in Ledger.read(res.ledger_path).header["id_map"].items()}
    assert _frames_of(res, id_map[1]) == list(range(60))  # the takeover was not split
    assert _frames_of(res, id_map[2]) == list(range(120))  # the static box was not dropped
    # an occlusion-witnessed link is pending under the default band: two ids, no rows filled
    rows = (box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0)
            + box_rows(2, range(883, 900), 160.0, 240.0, w=50.0, h=50.0)
            + box_rows(99, range(821, 900), 100.0, 150.0, vx=0.5, w=300.0, h=300.0))
    d = tmp_path / "occ"
    res = _refine(_write(d, table(rows)), d / "o.csv", fps=10)
    (link,) = [e for e in res.events if e.stage == "link" and e.tracks == [1, 2]]
    assert link.decision is Decision.HUMAN_PENDING and not link.applied
    assert res.tracks["track"].nunique() == 3 and res.tracks["interp"].sum() == 0


def test_stage_order_each_stage_sees_the_previous_stages_applied_edits(tmp_path, monkeypatch):
    seen = []

    def spy(name):
        orig = getattr(refiner_mod, name)

        def wrapper(work, *a, **kw):
            spans = work.groupby("track")["frame"].agg(["min", "max"])
            seen.append((name, {int(t): (int(r["min"]), int(r["max"]))
                                for t, r in spans.iterrows()}))
            return orig(work, *a, **kw)

        monkeypatch.setattr(refiner_mod, name, wrapper)

    for name in ("propose_splits", "propose_screen", "run_link_stage", "propose_orphans",
                 "fill_stage"):
        spy(name)
    _run_planted(tmp_path)
    assert [n for n, _ in seen] == ["propose_splits", "propose_screen", "run_link_stage",
                                    "propose_orphans", "fill_stage"]
    views = dict(seen)
    assert set(views["propose_splits"]) == {1, 2, 3, 4, 5, 6}
    # screening sees the applied split: track 1 ends at 109 and its tail is a new track 7
    assert views["propose_screen"][1] == (60, 109) and views["propose_screen"][7] == (110, 159)
    # linking sees the applied drops: no passenger (5) and no split tail (7), fragments intact
    assert set(views["run_link_stage"]) == {1, 2, 3, 4, 6}
    # orphans see the merged chain (4 is folded into 3); fill sees the orphan removed
    assert set(views["propose_orphans"]) == {1, 2, 3, 6}
    assert views["propose_orphans"][3] == (20, 109)
    assert set(views["fill_stage"]) == {1, 2, 3}


@pytest.mark.parametrize("stage", ["switch", "screen", "link", "orphan", "fill"])
def test_a_disabled_stage_is_skipped(tmp_path, stage):
    base = _run_planted(tmp_path / "on")
    assert _by(base, stage)  # the scene does exercise every stage
    cfg = RefineConfig.defaults()
    getattr(cfg, stage).enabled = False
    res = _run_planted(tmp_path / "off", cfg=cfg)
    assert _by(res, stage) == []
    kept = {int(k): v for k, v in Ledger.read(res.ledger_path).header["id_map"].items()}
    if stage == "switch":
        assert not any(e.kind is EventKind.SPLIT for e in res.events)
    elif stage == "screen":
        assert _frames_of(res, kept[5]) == list(range(100, 150))  # the passenger stays
    elif stage == "link":
        assert 3 in kept and 4 in kept  # the fragments stay two tracks
    elif stage == "orphan":
        assert _frames_of(res, kept[6]) == [30, 31]
    else:
        assert res.tracks["interp"].sum() == 0


def test_accepted_occluded_link_protects_its_gap_from_filling(tmp_path):
    rows = (box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0)
            + box_rows(2, range(883, 900), 160.0, 240.0, w=50.0, h=50.0)
            + box_rows(99, range(821, 900), 100.0, 150.0, vx=0.5, w=300.0, h=300.0))
    cfg = RefineConfig.defaults()
    cfg.fill.max_gap = 100.0  # seconds: far more than the 6.2 s gap
    refiner = TrackRefiner(cfg)
    # validate() keeps occluded_score_cap below accept_above, so no valid config auto-accepts an
    # occlusion-witnessed link (Plan 3's VLM or a human does). Lower the band after validation
    # to exercise the wiring that protects the gap of such an accepted link.
    cfg.link.accept_above = 0.5  # this 6.3 s gap scores 0.57
    refiner.refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", fps=10, verbose=False)
    res = refiner.last_result
    (link,) = [e for e in res.events if e.stage == "link" and e.tracks == [1, 2]]
    assert link.params["gate"] == "occluded" and link.decision is Decision.AUTO_ACCEPT
    assert link.applied
    t = res.tracks
    ids = t.loc[t["frame"] == 780, "track"].unique().tolist()
    tail = t.loc[(t["frame"] == 883) & (t["x"] == 160.0), "track"].tolist()
    assert tail == ids  # the two fragments are one output id
    assert t["interp"].sum() == 0 and not (t["frame"].between(821, 882) & t["track"].eq(ids[0])
                                           ).any()
    assert not any(e.kind is EventKind.FILL for e in res.events)


def test_a_normal_link_gap_is_filled_and_recorded(tmp_path):
    res = _run_planted(tmp_path)
    (fill,) = _by(res, "fill", EventKind.FILL)
    assert fill.params == {"gap": [59, 65], "n_rows": 5} and fill.applied
    t = res.tracks
    filled = t[t["interp"] == 1]
    assert filled["frame"].tolist() == [60, 61, 62, 63, 64]
    assert filled["track"].nunique() == 1


def test_orphan_is_dropped_and_deferred_ids_are_final(tmp_path):
    res = _run_planted(tmp_path)
    assert res.summary["orphan_deferred"] == []
    (orphan,) = _by(res, "orphan")
    assert orphan.decision is Decision.AUTO_ACCEPT and orphan.applied
    assert not (res.tracks["x"] == 1500.0).any()  # the orphan's boxes are gone
    # a deferred fragment is reported by its FINAL id, which differs from its raw id here
    rows = (box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0)
            + box_rows(2, [883], 160.0, 240.0, w=50.0, h=50.0)
            + box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=300.0))
    d = tmp_path / "def"
    res = _refine(_write(d, table(rows)), d / "o.csv", fps=10)
    final = int(res.tracks.loc[res.tracks["frame"] == 883, "track"].iloc[0])
    assert final == 3 and res.summary["orphan_deferred"] == [3]
    assert Ledger.read(res.ledger_path).header["summary"]["orphan_deferred"] == [3]


def test_a_short_track_that_was_linked_is_not_an_orphan(tmp_path):
    # two one-row tracks that link into a 0.2 s chain: short, but accepted-linked, so kept
    src = _write(tmp_path, table(box_rows(1, [0], 100.0, 100.0), box_rows(2, [2], 100.0, 100.0)))
    res = _refine(src, tmp_path / "o.txt", fps=10)
    (link,) = _by(res, "link")
    assert link.tracks == [1, 2] and link.applied
    assert _by(res, "orphan") == []
    assert res.tracks["frame"].tolist() == [0, 1, 2] and res.tracks["track"].nunique() == 1


def test_ids_are_renumbered_by_first_frame_and_id_map_is_recorded(tmp_path):
    res = _run_planted(tmp_path)
    header = Ledger.read(res.ledger_path).header
    # work ids at the end -> output ids, ordered by first frame: 3 (20), 1 (60), 2 (200)
    assert header["id_map"] == {"3": 1, "1": 2, "2": 3}
    assert sorted(res.tracks["track"].unique()) == [1, 2, 3]
    assert header["n_filled_input_rows_removed"] == 0
    assert header["summary"] == res.summary
    assert res.summary["before"]["tracks"] == 6 and res.summary["after"]["tracks"] == 3
    assert res.summary["after"]["interpolated_rows"] == 5
    assert res.summary["events"]["switch/SPLIT/AUTO_ACCEPT"] == 1
    assert res.summary["events"]["fill/FILL/AUTO_ACCEPT"] == 1


def test_frame_offset_shifts_the_output_and_keeps_every_decision(tmp_path):
    a = _run_planted(tmp_path / "a", offset=0)
    b = _run_planted(tmp_path / "b", offset=120000)
    ta, tb = a.tracks.copy(), b.tracks.copy()
    tb["frame"] -= 120000
    pd.testing.assert_frame_equal(ta, tb)

    def key(res):
        return [(e.id, e.stage, str(e.kind), str(e.decision), e.applied, e.tracks,
                 round(e.algo_score, 9)) for e in res.events]

    assert key(a) == key(b)


def test_context_file_identical_to_the_track_file_is_rejected(tmp_path):
    src = _write(tmp_path, table(box_rows(1, range(40), 100.0, 100.0, vx=2.0)))
    copy = tmp_path / "copy.txt"
    shutil.copy(src, copy)
    for ctx in (src, copy):
        with pytest.raises(ValueError, match=r"context_file.*same content"):
            TrackRefiner().refine(src, tmp_path / "o.txt", context_file=ctx, fps=10,
                                  verbose=False)
    assert not (tmp_path / "o.txt").exists()
    assert not (tmp_path / "o.ledger.jsonl").exists()
    other = _write(tmp_path, table(box_rows(9, range(40), 100.0, 100.0, vx=2.0)), "ctx.txt")
    TrackRefiner().refine(src, tmp_path / "o.txt", context_file=other, fps=10, verbose=False)
    assert (tmp_path / "o.txt").exists()


def test_fps_resolution_order_and_its_errors(tmp_path, synthetic_video, monkeypatch):
    video, _ = synthetic_video
    src = _write(tmp_path, table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)))

    def header(cfg_fps, arg):
        cfg = RefineConfig.defaults()
        cfg.fps = cfg_fps
        res = _refine(src, tmp_path / "o.txt", cfg=cfg, video_file=video, fps=arg)
        h = Ledger.read(res.ledger_path).header
        return h["fps"], h["fps_source"]

    assert header(10.0, 20.0) == (20.0, "argument")
    assert header(10.0, None) == (10.0, "config")
    assert header(None, None) == (pytest.approx(25.0), "video")
    cfg = RefineConfig.defaults()
    cfg.fps = 12.5
    res = _refine(src, tmp_path / "o2.txt", cfg=cfg)  # no video: the config decides
    assert Ledger.read(res.ledger_path).header["fps"] == 12.5
    # no explicit fps and no usable video fps: an error naming fps, before any output
    for bad in (0.0, float("nan")):
        monkeypatch.setattr(io, "video_info", lambda _p, bad=bad: {
            "fps": bad, "frame_count": 500, "width": 320, "height": 240})
        with pytest.raises(ValueError, match="fps="):
            _refine(src, tmp_path / "o3.txt", video_file=tmp_path / "missing.mp4")
    assert not (tmp_path / "o3.txt").exists()
    assert resolve_fps(None, None, 30.0) == (30.0, "video")
    assert resolve_fps(5, 10, 30.0) == (5.0, "argument")
    with pytest.raises(ValueError, match="positive"):
        resolve_fps(0, None, 30.0)
    with pytest.raises(ValueError, match="positive"):
        resolve_fps(None, -1, 30.0)


def test_fps_that_disagrees_with_the_video_warns_but_wins(tmp_path, synthetic_video, caplog):
    video, _ = synthetic_video
    src = _write(tmp_path, table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)))
    with caplog.at_level(logging.WARNING):
        res = _refine(src, tmp_path / "o.txt", video_file=video, fps=10)
    assert Ledger.read(res.ledger_path).header["fps"] == 10.0
    assert "differs from the video" in caplog.text


def test_filled_input_rows_and_duplicates_are_counted_in_the_header(tmp_path):
    df = table(box_rows(1, range(0, 10), 0.0, 0.0, vx=2.0),
               box_rows(1, range(15, 25), 30.0, 0.0, vx=2.0))
    filled = table(box_rows(1, range(10, 15), 20.0, 0.0))
    filled["r3"] = 1
    dup = df.iloc[[0]]
    res = _refine(_write(tmp_path, pd.concat([df, filled, dup])), tmp_path / "o.csv", fps=10)
    header = Ledger.read(res.ledger_path).header
    assert header["n_filled_input_rows_removed"] == 5
    assert header["summary"]["filled_input_rows_removed"] == 5
    assert header["summary"]["duplicate_input_rows_removed"] == 1
    assert len(res.tracks) == 25 and int(res.tracks["interp"].sum()) == 5  # refilled once
    assert res.tracks["frame"].tolist() == list(range(25))


def test_refine_batch_extras(tmp_path):
    rows, ctx_a, _ = _planted(0)
    _, ctx_b, _ = _planted(1000)
    src_a = _write(tmp_path / "in", rows, "a_track.txt")
    src_b = _write(tmp_path / "in", table(box_rows(1, range(40), 0.0, 0.0, vx=2.0)),
                   "my_track_cam.txt")
    src_c = _write(tmp_path / "in", table(box_rows(1, range(40), 0.0, 0.0, vx=2.0)), "plain.txt")
    ctx_a_file = _write(tmp_path / "in", ctx_a, "ca.txt")
    ctx_b_file = _write(tmp_path / "in", ctx_b, "cb.txt")
    out_dir = tmp_path / "out" / "nested"
    refiner = TrackRefiner()
    with pytest.raises(ValueError, match="paired by position"):
        refiner.refine_batch([src_a, src_b], context_files=[ctx_a_file], output_path=out_dir,
                             fps=10)
    got = refiner.refine_batch([src_a, src_b, src_c], context_files=[ctx_a_file, ctx_b_file,
                               ctx_a_file], output_path=out_dir, fps=10, verbose=False)
    assert [p.rsplit("/", 1)[1] for p in got] == [
        "a_refined.txt", "my_track_cam_refined.txt", "plain_refined.txt"]
    for name, ctx in (("a", ctx_a_file), ("my_track_cam", ctx_b_file), ("plain", ctx_a_file)):
        h = Ledger.read(out_dir / f"{name}_refined.ledger.jsonl").header
        assert h["inputs"]["context"]["sha256"] == io.sha256_file(ctx)


def test_refine_batch_overwrite_rewrites_the_output(tmp_path):
    src = _write(tmp_path / "in", table(box_rows(1, range(30), 0.0, 0.0, vx=2.0)), "a_track.txt")
    out = tmp_path / "out" / "a_refined.txt"
    refiner = TrackRefiner()
    refiner.refine_batch([src], output_path=out.parent, fps=10, verbose=False)
    before = out.read_bytes()
    _write(tmp_path / "in", table(box_rows(1, range(30), 5.0, 5.0, vx=2.0)), "a_track.txt")
    refiner.refine_batch([src], output_path=out.parent, fps=10, verbose=False)
    assert out.read_bytes() == before  # skipped: the stale output is kept
    refiner.refine_batch([src], output_path=out.parent, fps=10, is_overwrite=True,
                         verbose=False)
    assert out.read_bytes() != before
    assert pd.read_csv(out, header=None)[2].iloc[0] == 5


# ---- invariants on real-looking data ---------------------------------------------------------


def _dropped_keys(events, work):
    """(raw_id, frame) keys of the input rows removed by applied DROP events."""
    keys = set()
    for e in events:
        if e.kind is not EventKind.DROP or not e.applied:
            continue
        spans = e.params.get("spans")
        for raw, f0, f1 in e.lineage[0]:
            g = work[(work["raw_id"] == raw) & work["frame"].between(f0, f1)]
            if spans:
                g = g[np.logical_or.reduce([g["frame"].between(a, b) for a, b in spans])]
            keys |= {(int(r), int(f)) for r, f in zip(g["raw_id"], g["frame"], strict=True)}
    return keys


def _check_invariants(src, out_file, res, cfg):
    tin = io.read_tracks(src)
    out = res.tracks
    file_df = pd.read_csv(out_file, header=None, float_precision="round_trip")
    assert not out.duplicated(["frame", "track"]).any()
    assert sorted(out["track"].unique()) == list(range(1, out["track"].nunique() + 1))
    header = Ledger.read(res.ledger_path).header
    assert sorted(header["id_map"].values()) == list(range(1, out["track"].nunique() + 1))
    # the file is the returned table, sorted by frame then track
    assert file_df[[0, 1]].to_numpy().tolist() == out[["frame", "track"]].to_numpy().tolist()
    # the file holds integer boxes (the dnt track format), rounded like io.write_tracks does
    np.testing.assert_array_equal(file_df[[2, 3, 4, 5]].to_numpy(),
                                  out[["x", "y", "w", "h"]].round().astype(int).to_numpy())
    # every observed output box is an input box (rounded) on the same frame; filled rows are flagged
    obs = out[out["interp"] == 0]
    box = ["frame", "x", "y", "w", "h"]
    inp = tin.work[box].round().astype(int).drop_duplicates()
    merged = obs[box].round().astype(int).merge(inp, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert set(out["interp"].unique()) <= {0, 1}
    # row accounting
    dropped = _dropped_keys(res.events, tin.work)
    filled_rows = sum(e.params["n_rows"] for e in res.events if e.kind is EventKind.FILL)
    assert int((out["interp"] == 1).sum()) == filled_rows
    lost_to_overlap = len(tin.work) - len(dropped) - len(obs)
    n_overlap_links = sum(1 for e in res.events if e.stage == "link" and e.applied
                          and e.params.get("gate") == "overlap")
    assert 0 <= lost_to_overlap <= cfg.link.overlap_frames * n_overlap_links
    assert len(out) == len(tin.work) - len(dropped) - lost_to_overlap + filled_rows
    assert header["summary"]["after"]["observed_rows"] == len(obs)
    return len(dropped), filled_rows


@pytest.mark.parametrize("scene", ["baseline0", "baseline1", "baseline2", "planted"])
def test_output_invariants_and_determinism(tmp_path, scene):
    if scene == "planted":
        rows, ctx, app = _planted(0)
        kw = {"context_file": _write(tmp_path, ctx, "c.txt")}
    else:
        rows, app, kw = load_raw(int(scene[-1])), None, {}
    src = _write(tmp_path, rows)
    cfg = RefineConfig.defaults()
    outs = []
    for i in (1, 2):
        refiner = TrackRefiner(cfg, appearance_factory=None if app is None
                               else lambda **_kw: app)
        refiner.refine(src, tmp_path / f"run{i}" / "o.txt", fps=10, verbose=False, **kw)
        outs.append((tmp_path / f"run{i}" / "o.txt", refiner.last_result))
    n_dropped, n_filled = _check_invariants(src, *outs[0], cfg)
    assert n_filled > 0  # the scenes do exercise filling
    if scene == "planted":
        assert n_dropped == 50 + 50 + 2
    (o1, _), (o2, _) = outs
    assert o1.read_bytes() == o2.read_bytes()
    assert o1.with_suffix(".ledger.jsonl").read_bytes() == o2.with_suffix(
        ".ledger.jsonl").read_bytes()


@pytest.mark.slow
@pytest.mark.parametrize("target", ["person", "vehicle"])
def test_refine_a_large_track_file_in_reasonable_time(tmp_path, target):
    df = random_tracks(seed=5, n_objects=300, n_frames=900)
    assert len(df) > 50000
    src = _write(tmp_path, df)
    cfg = RefineConfig.defaults(target)
    t0 = time.perf_counter()
    res = _refine(src, tmp_path / "o.txt", cfg=cfg, fps=10)
    elapsed = time.perf_counter() - t0
    assert res.tracks["track"].nunique() < df["track"].nunique()
    assert elapsed < 60, f"{elapsed:.1f} s"
