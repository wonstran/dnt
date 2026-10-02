import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine import refiner as refiner_mod
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.features import ArrayAppearance
from dnt.refine.primitives import box_centers
from dnt.refine.refiner import TrackRefiner, resolve_fps

from ._fixtures import box_rows, load_raw, random_tracks, table


def _write(dirpath, df, name="t.txt"):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    df.to_csv(p, index=False, header=False)
    return p


def _refine(src, out, cfg=None, **kw):
    if kw.get("video_file") is not None:
        cfg = cfg if cfg is not None else RefineConfig.defaults()
        cfg.encoder.kind = "none"  # these tests are about the video's metadata, not appearance
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
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []  # kind none: silent


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


# ---- fix round 1: chain-representative keying, ledger values, hints and MOT input ------------

_OCC, _ORD = "occluded", "normal"


def _chain_rows(kinds, seg=40):
    """Track 1 -> 2 -> 3 along a line; ``kinds`` gives the gap before track 2 and track 3.

    An ordinary gap is 5 missing frames; an occluded gap is 61 missing frames with a big box
    (track 91 / 92) that hides the path. Boxes are 30x60 at 3 px/frame, ``h == 60`` marks them.
    """
    def x(f):
        return 600.0 + 3.0 * (f - 20)

    rows = box_rows(1, range(20, 20 + seg), x(20), 200.0, vx=3.0)
    end = 20 + seg - 1
    for i, kind in enumerate(kinds):
        gap = 5 if kind == _ORD else 61
        start = end + gap + 1
        if kind == _OCC:
            rows += box_rows(91 + i, range(end + 1, start), x(end) - 20.0, 100.0,
                             w=3.0 * (gap + 1) + 70.0, h=250.0)
        rows += box_rows(2 + i, range(start, start + seg), x(start), 200.0, vx=3.0)
        end = start + seg - 1
    return rows


def _refine_chain(d, kinds, key_check=True):
    cfg = RefineConfig.defaults()
    cfg.fill.max_gap = 100.0  # seconds: far more than any gap here
    refiner = TrackRefiner(cfg)
    # validate() keeps occluded_score_cap below accept_above, so no valid config auto-accepts an
    # occlusion-witnessed link (Plan 3's VLM or a human does). Lower the band after validation
    # to exercise the wiring that protects the gap of such an accepted link.
    cfg.link.accept_above = 0.45
    refiner.refine(_write(d, table(_chain_rows(kinds))), d / "o.txt", fps=10, verbose=False)
    res = refiner.last_result
    links = {tuple(e.tracks): e for e in _by(res, "link")}
    assert set(links) == {(1, 2), (2, 3)}
    for (a, b), kind in zip(((1, 2), (2, 3)), kinds, strict=True):
        assert links[(a, b)].params["gate"] == kind
        assert links[(a, b)].decision is Decision.AUTO_ACCEPT and links[(a, b)].applied
    return res, links


@pytest.mark.parametrize("kinds", [(_ORD, _OCC), (_OCC, _OCC), (_OCC, _ORD)],
                         ids=["head-ordinary-tail-occluded", "both-occluded",
                              "head-occluded-tail-ordinary"])
def test_protected_gaps_follow_the_chain_representative(tmp_path, kinds):
    res, links = _refine_chain(tmp_path, kinds)
    t = res.tracks
    obj = t[t["h"] == 60]  # the moving object; the occluder boxes are 250 high
    assert obj["track"].nunique() == 1  # A, B and C are one output id
    expected, end = list(range(20, 60)), 59
    for kind in kinds:  # gap of 5 or 61 missing frames, then a 40-frame segment
        start = end + (6 if kind == _ORD else 62)
        expected += range(start, start + 40)
        end = start + 39
    assert sorted(obj.loc[obj["interp"] == 0, "frame"]) == expected
    for (a, b), kind in zip(((1, 2), (2, 3)), kinds, strict=True):
        f_before, f_after = links[(a, b)].params["gap"]
        gap_rows = obj[(obj["frame"] > f_before) & (obj["frame"] < f_after)]
        if kind == _OCC:
            assert f_after - f_before - 1 == 61 and len(gap_rows) == 0  # left empty
        else:
            assert len(gap_rows) == 5 and (gap_rows["interp"] == 1).all()  # filled
    fills = [tuple(e.params["gap"]) for e in _by(res, "fill", EventKind.FILL)]
    occluded = [tuple(links[k].params["gap"]) for k, kind in zip(((1, 2), (2, 3)), kinds,
                                                                  strict=True) if kind == _OCC]
    assert not set(fills) & set(occluded)
    assert int(obj["interp"].sum()) == 5 * kinds.count(_ORD)


def test_ordinary_gaps_in_the_same_chain_are_all_filled(tmp_path):
    res, _ = _refine_chain(tmp_path, (_ORD, _ORD))  # control: geometry alone leaves no gap empty
    obj = res.tracks[res.tracks["h"] == 60]
    assert obj["track"].nunique() == 1 and int(obj["interp"].sum()) == 10
    assert len(_by(res, "fill", EventKind.FILL)) == 2


def test_fill_record_values_on_a_constant_velocity_track(tmp_path):
    # 6 px/frame, 30x60 boxes, observed on frames 0-9 and 15-24: the gap is 10 -> 15
    rows = (box_rows(1, range(0, 10), 0.0, 0.0, vx=6.0)
            + box_rows(1, range(15, 25), 90.0, 0.0, vx=6.0))
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", fps=10)
    (fill,) = _by(res, "fill")
    assert fill.kind is EventKind.FILL and fill.tracks == [1]
    assert fill.frames == (9, 15)
    assert fill.params == {"gap": [9, 15], "n_rows": 5}
    assert fill.lineage == [[[1, 0, 24]]]  # one raw track spanning frames 0..24
    assert fill.algo_score == 1.0
    assert fill.signals["gap_seconds"] == pytest.approx(5 / 10)  # 5 missing frames at 10 fps
    # observed ends: centers (54 + 15, 30) and (90 + 15, 30) -> 36 px apart; median height 60
    assert fill.signals["chord_h"] == pytest.approx(36.0 / 60.0)
    # 6 px per frame = 60 px/s over 60 px high boxes = 1 box height per second
    assert fill.signals["max_fill_speed_h_s"] == pytest.approx(1.0)
    assert fill.decision is Decision.AUTO_ACCEPT and fill.applied
    assert fill.decision_history == [{"decision": "AUTO_ACCEPT", "round": 0, "source": "auto"}]
    assert fill.edit == {"kind": "FILL", "params": {"gap": [9, 15], "n_rows": 5}}
    filled = res.tracks[res.tracks["interp"] == 1]
    assert filled["frame"].tolist() == [10, 11, 12, 13, 14]
    assert filled["x"].tolist() == [60, 66, 72, 78, 84]  # the straight line, as integer boxes


def test_fill_speed_signal_is_the_fastest_step_across_the_gap(tmp_path):
    # 6 px/frame before the gap and after it, but the object covers 72 px over the six steps
    # of the gap (12 px/frame on average): the filled path must speed up, so steps differ
    rows = (box_rows(1, range(0, 10), 0.0, 0.0, vx=6.0)
            + box_rows(1, range(15, 25), 54.0 + 72.0, 0.0, vx=6.0))
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", fps=10)
    (fill,) = _by(res, "fill")
    seg = res.tracks[res.tracks["frame"].between(9, 15)].sort_values("frame")
    steps = np.abs(np.diff(seg["x"].to_numpy(float)))  # px per frame; y and heights are constant
    assert len(steps) == 6 and steps.sum() == 72 and steps.min() < steps.max()
    assert fill.signals["max_fill_speed_h_s"] == pytest.approx(steps.max() * 10 / 60.0)
    assert fill.signals["max_fill_speed_h_s"] > 2.0  # above the mean 72 / 6 px per frame
    assert fill.signals["chord_h"] == pytest.approx(72.0 / 60.0)


def test_smooth_record_values_with_one_displaced_box(tmp_path):
    cfg = RefineConfig.defaults()
    cfg.fill.smooth_existing = True
    rows = box_rows(1, range(41), 100.0, 100.0, vx=2.0)
    rows[20][2] += 30.0  # a single displaced middle box (frame 20)
    src = _write(tmp_path, table(rows))
    res = _refine(src, tmp_path / "o.csv", cfg=cfg, fps=10)
    (sm,) = _by(res, "fill", EventKind.SMOOTH)
    inp = io.read_tracks(src).work.set_index("frame")
    out = res.tracks.set_index("frame")
    shift = np.linalg.norm(box_centers(out.loc[inp.index, ["x", "y", "w", "h"]].to_numpy())
                           - box_centers(inp[["x", "y", "w", "h"]].to_numpy()), axis=1)
    # the smoother pulls the displaced box back and (slightly) moves its neighbours
    assert 0 < shift[20] < 30 and shift.argmax() == 20
    assert sm.signals["max_shift_frame"] == 20
    assert sm.signals["max_shift_px"] == pytest.approx(shift[20])
    assert sm.signals["mean_shift_px"] == pytest.approx(shift.mean())  # over every row
    assert sm.params["n_rows"] == int((shift > 0).sum()) and sm.params["n_rows"] >= 3
    assert sm.frames == (0, 40) and sm.tracks == [1] and sm.lineage == [[[1, 0, 40]]]
    assert sm.decision is Decision.AUTO_ACCEPT and sm.applied
    assert not _by(res, "fill", EventKind.FILL)
    # a straight track barely moves: at most rounding-level shifts, never a displaced box
    d = tmp_path / "straight"
    res = _refine(_write(d, table(box_rows(1, range(41), 100.0, 100.0, vx=6.0))), d / "o.csv",
                  cfg=cfg, fps=10)
    for e in _by(res, "fill", EventKind.SMOOTH):
        assert e.signals["max_shift_px"] <= 1.5 and e.params["n_rows"] <= 3


def test_summary_counts_on_a_two_class_table(tmp_path):
    rows = (box_rows(1, range(0, 10), 0.0, 0.0, vx=6.0, cls=0)
            + box_rows(1, range(15, 25), 90.0, 0.0, vx=6.0, cls=0)  # gap of 5 frames, refilled
            + box_rows(2, range(100, 130), 800.0, 400.0, vx=2.0, cls=2)  # 3.0 s
            + box_rows(3, range(200, 250), 300.0, 700.0, vx=-2.0, cls=0))  # 5.0 s
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.csv", fps=10)
    assert _by(res, "link") == []
    # observed spans: track 1 frames 0..24 = 2.5 s, track 2 = 3.0 s, track 3 = 5.0 s
    before = {"tracks": 3, "tracks_per_class": {"0": 2, "2": 1}, "observed_rows": 100,
              "interpolated_rows": 0, "median_track_seconds": 3.0}
    assert res.summary["before"] == before
    assert res.summary["after"] == {**before, "interpolated_rows": 5}
    assert res.summary["events"] == {"fill/FILL/AUTO_ACCEPT": 1}
    assert res.summary["vlm"] == {"calls": 0, "cache_hits": 0, "failures": 0}
    assert Ledger.read(res.ledger_path).header["summary"] == res.summary
    # an even count of tracks: the median is the mean of the two middle durations
    d = tmp_path / "four"
    res = _refine(_write(d, table(rows, box_rows(4, range(400, 420), 100.0, 100.0, vx=2.0,
                                                  cls=0))), d / "o.csv", fps=10)
    assert res.summary["before"]["median_track_seconds"] == pytest.approx((2.5 + 3.0) / 2)


def test_header_records_video_context_hints_and_frame_size(tmp_path, synthetic_video):
    video, _ = synthetic_video
    info = io.video_info(video)
    src = _write(tmp_path, table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)))
    ctx = table(box_rows(9, range(100), 200.0, 150.0, vx=1.0, w=80.0, h=60.0, cls=2))
    ctx_tracks = _write(tmp_path, ctx, "ctx_tracks.txt")
    dets = pd.DataFrame({"frame": ctx["frame"], "res": -1, "x": ctx["x"], "y": ctx["y"],
                         "w": ctx["w"], "h": ctx["h"], "conf": 0.9, "cls": ctx["cls"]})
    ctx_dets = _write(tmp_path, dets, "ctx_dets.txt")
    hints = tmp_path / "hints.csv"
    hints.write_text("track,cls,avg_score\n1,3,0.5\n")
    assert info["frame_count"] >= 100  # the track frames fit the video

    res = _refine(src, tmp_path / "a.txt", video_file=video, context_file=ctx_tracks,
                  reclass_file=hints)
    h = Ledger.read(res.ledger_path).header
    assert h["inputs"]["video"]["frame_count"] == info["frame_count"]
    assert h["inputs"]["video"]["fingerprint"] == {
        "sha256": io.sha256_file(video), "size": video.stat().st_size,
        "frame_count": info["frame_count"]}
    assert h["inputs"]["video"]["path"] == str(video)
    assert h["inputs"]["context"]["format"] == "tracks"
    assert h["inputs"]["context"]["sha256"] == io.sha256_file(ctx_tracks)
    assert h["inputs"]["context"]["path"] == str(ctx_tracks)
    assert h["inputs"]["hints"] == {"reclass": {
        "path": str(hints), "abs_path": str(hints.resolve()), "sha256": io.sha256_file(hints)}}
    assert h["inputs"]["tracks"]["format"] == "dnt" and h["inputs"]["features"] is None
    assert h["frame_size"] == [info["width"], info["height"]]  # from the video
    assert (h["fps"], h["fps_source"]) == (pytest.approx(info["fps"]), "video")

    cfg = RefineConfig.defaults()
    cfg.frame_size = [640, 480]  # a config value wins over the video's size
    res = _refine(src, tmp_path / "b.txt", cfg=cfg, video_file=video, context_file=ctx_dets)
    h = Ledger.read(res.ledger_path).header
    assert h["frame_size"] == [640, 480]
    assert h["inputs"]["context"]["format"] == "dets"
    assert h["inputs"]["hints"] is None

    res = _refine(src, tmp_path / "c.txt", fps=10)  # no video, no override, no context
    h = Ledger.read(res.ledger_path).header
    assert h["frame_size"] is None and h["inputs"]["video"] is None
    assert h["inputs"]["context"] is None and h["fps_source"] == "argument"
    cfg = RefineConfig.defaults()
    cfg.fps = 10.0
    cfg.frame_size = [640, 480]
    res = _refine(src, tmp_path / "d.txt", cfg=cfg)
    h = Ledger.read(res.ledger_path).header
    assert h["frame_size"] == [640, 480] and h["fps_source"] == "config"


def _rider_rows(track=1, frames=range(50), x0=100.0):
    """A fast, smooth person (3 box heights per second): a rider candidate."""
    return box_rows(track, frames, x0, 110.0, vx=12.0, w=20.0, h=40.0)


def _hints(dirpath, *rows, name="hints.csv"):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text("track,cls,avg_score\n" + "".join(f"{t},{c},{a}\n" for t, c, a in rows))
    return p


@pytest.mark.parametrize("ncols", [7, 10])
def test_mot_format_matches_the_equivalent_dnt_file(tmp_path, ncols):
    cfg = RefineConfig.defaults()
    cfg.class_ids = [2]
    rows = (box_rows(1, range(0, 10), 0.0, 0.0, vx=6.0, cls=2, score=0.8)
            + box_rows(1, range(15, 25), 90.0, 0.0, vx=6.0, cls=2, score=0.8)
            + box_rows(2, range(0, 30), 500.0, 300.0, vx=-3.0, cls=2, score=0.6))
    dnt = _write(tmp_path / "dnt", table(rows))
    mot_df = table(rows).iloc[:, :ncols].copy()  # frame, id, x, y, w, h, conf[, r3, r4, r5]
    if ncols == 10:
        mot_df.iloc[:, 7:] = -1
    mot = _write(tmp_path / "mot", mot_df)
    a = _refine(dnt, tmp_path / "a.txt", cfg=cfg, fps=10)
    b = _refine(mot, tmp_path / "b.txt", cfg=cfg, fps=10, fmt="mot")
    pd.testing.assert_frame_equal(a.tracks, b.tracks)
    assert (tmp_path / "a.txt").read_bytes() == (tmp_path / "b.txt").read_bytes()
    assert set(b.tracks["cls"]) == {2}  # the class comes from config.class_ids[0]
    assert (b.tracks["score"].round(2).isin([0.8, 0.6, -1.0])).all()
    assert Ledger.read(b.ledger_path).header["inputs"]["tracks"]["format"] == "mot"
    assert int(b.tracks["interp"].sum()) == 5
    with pytest.raises(ValueError, match="unknown track format"):
        _refine(mot, tmp_path / "c.txt", fps=10, fmt="csv")


def test_reclass_hint_settles_the_rider_subtype_and_is_applied(tmp_path, caplog):
    hints = _hints(tmp_path, (1, 3, 0.95), (99, 1, 0.8))  # 99 is not a track of this file
    src = _write(tmp_path, table(_rider_rows()))
    with caplog.at_level(logging.WARNING):
        res = _refine(src, tmp_path / "o.txt", fps=10, reclass_file=hints)
    assert "unknown track IDs" in caplog.text
    (ev,) = _by(res, "screen")
    assert ev.kind is EventKind.RECLASS and ev.params["new_cls"] == 3
    assert ev.signals["subtype_source"] == "reclass" and ev.signals["subtype"] == "motorcycle"
    assert ev.decision is Decision.AUTO_ACCEPT and ev.applied
    assert set(res.tracks["cls"]) == {3}  # every row carries the motorcycle class
    h = Ledger.read(res.ledger_path).header
    assert h["inputs"]["hints"]["reclass"]["sha256"] == io.sha256_file(hints)
    assert h["inputs"]["hints"]["reclass"]["path"] == str(hints)
    # without the hint the rider is only a pending candidate that needs a subtype
    d = tmp_path / "nohint"
    res = _refine(_write(d, table(_rider_rows())), d / "o.txt", fps=10)
    (ev,) = _by(res, "screen")
    assert ev.params["new_cls"] is None and ev.decision is Decision.HUMAN_PENDING
    assert not ev.applied and set(res.tracks["cls"]) == {0}


def test_a_malformed_hints_file_fails_before_any_output(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,cls\n1,3\n")
    src = _write(tmp_path, table(_rider_rows()))
    with pytest.raises(ValueError, match="track, cls, avg_score"):
        _refine(src, tmp_path / "o.txt", fps=10, reclass_file=bad)
    assert not (tmp_path / "o.txt").exists() and not (tmp_path / "o.ledger.jsonl").exists()


def _split_rider_scene():
    """Raw track 1 walks, then (size jump + appearance step) rides fast: a confident split."""
    rows = (box_rows(1, range(0, 50), 100.0, 110.0, vx=3.0, w=20.0, h=40.0)
            + box_rows(1, range(50, 100), 250.0, 110.0, vx=24.0, w=32.0, h=64.0))
    frames = list(range(100))
    app = ArrayAppearance({1: (frames, np.array([A_ if f < 50 else B_ for f in frames]))})
    return table(rows), app


def test_a_reclass_hint_is_unlocalized_once_the_raw_track_was_split(tmp_path):
    rows, app = _split_rider_scene()
    hints = _hints(tmp_path, (1, 3, 0.95))
    refiner = TrackRefiner(appearance_factory=lambda **_kw: app)
    refiner.refine(_write(tmp_path, rows), tmp_path / "o.txt", fps=10, verbose=False,
                   reclass_file=hints)
    res = refiner.last_result
    (split,) = _by(res, "switch")
    assert split.decision is Decision.AUTO_ACCEPT and split.applied  # an accepted split
    assert split.params["cut_frame"] == 50
    (ev,) = _by(res, "screen")
    assert ev.kind is EventKind.RECLASS and ev.tracks == [2]  # the fast tail, a new track
    # the hint was made for the whole raw track: recorded, but it cannot settle the subtype
    assert ev.signals["hint_unlocalized"] == {"cls": 3, "avg_score": 0.95}
    assert ev.params["new_cls"] is None and "subtype_source" not in ev.signals
    assert ev.signals["needs_subtype"] is True
    assert ev.decision is Decision.HUMAN_PENDING and not ev.applied
    assert set(res.tracks["cls"]) == {0}
    # the same scene without the split: the hint stays localized and settles the tail
    # (a raw track that was never split)
    d = tmp_path / "nosplit"
    src = _write(d, table(_rider_rows()))
    res = _refine(src, d / "o.txt", fps=10, reclass_file=_hints(d, (1, 3, 0.95)))
    (ev,) = _by(res, "screen")
    assert ev.params["new_cls"] == 3 and "hint_unlocalized" not in ev.signals


def test_refine_batch_passes_reclass_and_context_files_by_position(tmp_path):
    src_a = _write(tmp_path / "in", table(_rider_rows()), "a_track.txt")
    src_b = _write(tmp_path / "in", table(_rider_rows()), "b_track.txt")
    hint_a = _hints(tmp_path / "in", (1, 3, 0.95), name="ha.csv")  # motorcycle
    hint_b = _hints(tmp_path / "in", (1, 1, 0.95), name="hb.csv")  # cyclist
    out_dir = tmp_path / "out"
    refiner = TrackRefiner()
    got = refiner.refine_batch([src_a, src_b], reclass_files=[hint_a, hint_b],
                               output_path=out_dir, fps=10, verbose=False)
    cls = [set(pd.read_csv(p, header=None)[7]) for p in got]
    assert cls == [{3}, {1}]
    for name, hint in (("a", hint_a), ("b", hint_b)):
        h = Ledger.read(out_dir / f"{name}_refined.ledger.jsonl").header
        assert h["inputs"]["hints"]["reclass"]["sha256"] == io.sha256_file(hint)
    # context files: a passenger inside a car in file 1's context, a car far away in file 2's
    ped = table(box_rows(1, range(50), 100.0, 110.0, vx=3.0, w=20.0, h=40.0))
    src_c = _write(tmp_path / "in", ped, "c_track.txt")
    src_d = _write(tmp_path / "in", ped, "d_track.txt")
    car_over = _write(tmp_path / "in", table(box_rows(9, range(50), 80.0, 100.0, vx=3.0, w=80.0,
                                                      h=60.0, cls=2)), "car_over.txt")
    car_far = _write(tmp_path / "in", table(box_rows(9, range(50), 900.0, 700.0, vx=3.0, w=80.0,
                                                     h=60.0, cls=2)), "car_far.txt")
    out2 = tmp_path / "out2"
    got = refiner.refine_batch([src_c, src_d], context_files=[car_over, car_far],
                               output_path=out2, fps=10, verbose=False)
    assert [len(pd.read_csv(p, header=None)) if Path(p).stat().st_size else 0
            for p in got] == [0, 50]  # the passenger is dropped only where the car is
    for name, ctx in (("c", car_over), ("d", car_far)):
        h = Ledger.read(out2 / f"{name}_refined.ledger.jsonl").header
        assert h["inputs"]["context"]["sha256"] == io.sha256_file(ctx)


def test_the_ledger_is_written_before_the_output_track_file(tmp_path, monkeypatch):
    src = _write(tmp_path, table(box_rows(1, range(30), 0.0, 0.0, vx=2.0)))

    def boom(self, path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Ledger, "write", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        TrackRefiner().refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    assert not (tmp_path / "o.txt").exists()  # an existing output would be skipped by a batch
    monkeypatch.undo()
    res = _refine(src, tmp_path / "o.txt", fps=10)
    assert (tmp_path / "o.txt").exists() and res.ledger_path.exists()
    # the ledger already exists whenever the output does: check the write order directly
    order = []
    orig_write, orig_tracks = Ledger.write, io.write_tracks
    monkeypatch.setattr(Ledger, "write", lambda self, p: (order.append("ledger"),
                                                          orig_write(self, p)))
    monkeypatch.setattr(io, "write_tracks", lambda w, p: (order.append("tracks"),
                                                          orig_tracks(w, p)))
    _refine(src, tmp_path / "o2.txt", fps=10)
    assert order == ["ledger", "tracks"]


def test_an_empty_track_file_may_come_with_an_empty_context_file(tmp_path):
    (tmp_path / "e.txt").write_text("")
    (tmp_path / "c.txt").write_text("")
    res = _refine(tmp_path / "e.txt", tmp_path / "o.csv", fps=10, context_file=tmp_path / "c.txt")
    assert (tmp_path / "o.csv").read_text() == "" and res.events == []
    assert Ledger.read(res.ledger_path).header["inputs"]["context"]["format"] == "tracks"
    (tmp_path / "w.txt").write_text("\n \n")  # blank lines count as empty too
    _refine(tmp_path / "w.txt", tmp_path / "o2.csv", fps=10, context_file=tmp_path / "w.txt")
    # a non-empty track file still may not be its own context
    src = _write(tmp_path, table(box_rows(1, range(40), 100.0, 100.0, vx=2.0)))
    with pytest.raises(ValueError, match="same content"):
        _refine(src, tmp_path / "o3.csv", fps=10, context_file=src)


@pytest.mark.parametrize("count", [0, -1])
def test_an_unknown_video_frame_count_skips_the_frame_check(tmp_path, synthetic_video,
                                                            monkeypatch, count):
    video, _ = synthetic_video
    real = io.video_info(video)
    monkeypatch.setattr(io, "video_info", lambda _p: {**real, "frame_count": count})
    src = _write(tmp_path, table(box_rows(1, range(495, 505), 0.0, 0.0)))  # beyond a 150-frame clip
    res = _refine(src, tmp_path / "o.txt", video_file=video)
    assert Ledger.read(res.ledger_path).header["inputs"]["video"]["frame_count"] == count
    assert len(res.tracks) == 10


def test_the_fps_error_explains_both_ways_to_get_a_frame_rate(tmp_path):
    for video_fps in (None, 0.0):
        with pytest.raises(ValueError) as err:
            resolve_fps(None, None, video_fps)
        msg = str(err.value)
        assert "frame rate is needed" in msg and "fps=" in msg
        assert "no video is given" in msg and "no usable frame rate" in msg
    assert "ValueError" in TrackRefiner.refine_batch.__doc__


# ---- final review I1: stage 1 -> stage 2 cut wiring, split order, fill fallback ---------------


def _ped_then_passenger(d):
    """Raw track 1: a pedestrian (frames 0-9), then a passenger in context car 9 (10-99).

    The box shrinks 2x at frame 10, so stage 1 finds a motion-only cut there (S = 0.70,
    capped below accept). As a whole track the in-vehicle cue holds on 90% of the rows, which
    would auto-drop the track, pedestrian rows included.
    """
    rows = (box_rows(1, range(0, 10), 100.0, 300.0, vx=3.0, w=30.0, h=60.0)
            + box_rows(1, range(10, 100), 140.0, 330.0, vx=3.0, w=15.0, h=30.0))
    car = box_rows(9, range(10, 100), 120.0, 310.0, vx=3.0, w=80.0, h=60.0, cls=2)
    return _write(d, table(rows)), _write(d, table(car), "c.txt")


@pytest.mark.parametrize("cut", ["pending", "weak"])
def test_an_unresolved_stage1_cut_makes_screening_partial_and_keeps_the_pedestrian(tmp_path,
                                                                                   cut):
    src, ctx = _ped_then_passenger(tmp_path)
    cfg = RefineConfig.defaults()
    if cut == "weak":  # the 0.70 candidate now lands in [screen.segment_at, switch.reject_below)
        cfg.switch.reject_below = 0.75
    res = _refine(src, tmp_path / "o.txt", cfg=cfg, context_file=ctx, fps=10)
    splits = _by(res, "switch")
    if cut == "pending":
        (split,) = splits
        assert split.params["cut_frame"] == 10 and split.algo_score == pytest.approx(0.70)
        assert split.decision is Decision.HUMAN_PENDING and not split.applied
    else:
        assert splits == []  # below reject_below: no event, only a weak cut for screening
    (drop,) = _by(res, "screen")
    assert drop.kind is EventKind.DROP and drop.params["reason"] == "in_vehicle"
    assert drop.params["spans"] == [[10, 99]] and drop.signals["partial"] is True
    assert drop.algo_score == pytest.approx(0.75)  # screen.mixed_score_cap
    assert drop.decision is Decision.HUMAN_PENDING and not drop.applied
    # nothing was auto-dropped: the pedestrian rows (and, pending review, the rest) are kept
    assert _frames_of(res, 1) == list(range(100))


def _two_splits(tmp_path):
    """Raw track 1 changes object (appearance, size, position) at frames 50 and 100."""
    e = np.eye(8)
    rows = (box_rows(1, range(0, 50), 100.0, 110.0, vx=3.0, w=20.0, h=40.0)
            + box_rows(1, range(50, 100), 500.0, 400.0, vx=3.0, w=32.0, h=64.0)
            + box_rows(1, range(100, 150), 900.0, 110.0, vx=3.0, w=20.0, h=40.0))
    frames = list(range(150))
    app = ArrayAppearance({1: (frames, np.array([e[0] if f < 50 else e[1] if f < 100 else e[2]
                                                 for f in frames]))})
    refiner = TrackRefiner(appearance_factory=lambda **_kw: app)
    refiner.refine(_write(tmp_path, table(rows)), tmp_path / "o.txt", fps=10, verbose=False)
    return refiner.last_result


def test_two_accepted_splits_on_one_track_both_take_effect(tmp_path):
    res = _two_splits(tmp_path)
    splits = sorted(_by(res, "switch"), key=lambda ev: ev.params["cut_frame"])
    assert [s.params["cut_frame"] for s in splits] == [50, 100]
    for s in splits:
        assert s.decision is Decision.AUTO_ACCEPT and s.applied
    assert len(res.tracks) == 150  # every row is kept
    spans = res.tracks.groupby("track")["frame"].agg(["min", "max", "count"])
    assert spans.values.tolist() == [[0, 49, 50], [50, 99, 50], [100, 149, 50]]


def test_a_pending_split_leaves_a_raw_hint_unlocalized_for_every_segment(tmp_path):
    # spec 11.1: a mixed pedestrian/rider track with a strong raw hint, split pending
    rows = (box_rows(1, range(0, 50), 100.0, 110.0, vx=3.0, w=20.0, h=40.0)
            + box_rows(1, range(50, 100), 250.0, 110.0, vx=12.0, w=32.0, h=64.0))
    src = _write(tmp_path, table(rows))
    res = _refine(src, tmp_path / "o.txt", fps=10, reclass_file=_hints(tmp_path, (1, 3, 0.95)))
    (split,) = _by(res, "switch")
    assert split.params["cut_frame"] == 50 and split.decision is Decision.HUMAN_PENDING
    (ev,) = _by(res, "screen")
    assert ev.kind is EventKind.RECLASS and ev.params["spans"] == [[50, 99]]  # the rider only
    assert ev.params["new_cls"] is None and "subtype_source" not in ev.signals
    assert ev.signals["hint_unlocalized"] == {"cls": 3, "avg_score": 0.95}
    assert ev.signals["P"] is None  # the hint is not a cue either
    assert ev.decision is Decision.HUMAN_PENDING and not ev.applied
    assert set(res.tracks["cls"]) == {0}


def test_fill_max_gap_null_falls_back_to_link_max_gap(tmp_path):
    # a pedestrian waits 3.1 s off camera at one spot: a static-gate link (spec 6.4). The gap
    # is joined but not filled, because fill.max_gap null means link.max_gap (1 s); a 5-frame
    # gap of the same track is filled.
    rows = (box_rows(1, range(0, 20), 300.0, 200.0) + box_rows(1, range(25, 50), 300.0, 200.0)
            + box_rows(2, range(80, 130), 301.0, 200.0))
    cfg = RefineConfig.defaults()
    assert cfg.fill.max_gap is None and cfg.link.max_gap == 1.0
    res = _refine(_write(tmp_path, table(rows)), tmp_path / "o.txt", cfg=cfg, fps=10)
    (link,) = _by(res, "link")
    assert link.params == {"gate": "static", "gap": [49, 80]} and link.applied
    assert res.tracks["track"].nunique() == 1
    (fill,) = _by(res, "fill", EventKind.FILL)
    assert fill.params == {"gap": [19, 25], "n_rows": 5}
    t = res.tracks
    assert t.loc[t["interp"] == 1, "frame"].tolist() == [20, 21, 22, 23, 24]
    assert not t["frame"].between(50, 79).any()  # the 3 s wait stays empty
    # an explicit fill.max_gap above the wait fills it
    cfg.fill.max_gap = 4.0
    d = tmp_path / "long"
    res = _refine(_write(d, table(rows)), d / "o.txt", cfg=cfg, fps=10)
    assert res.tracks["frame"].between(50, 79).sum() == 30


# ---- final review I3 (R22): a same-run detection file as context -----------------------------


def _beside_the_path(tmp_path, shift):
    """Tracks 1 and 2 are one pedestrian with a 30-frame gap; pedestrian 3 walks beside the path.

    Track 3's box covers 48% of each hidden box of the gap, below ``link.witness_iob``. The
    context is a detection file of the same run: every row's box, with track 3's detections
    ``shift`` px closer to the path.
    """
    t = table(box_rows(1, range(60, 101), 20.0, 100.0, vx=2.0, w=50.0, h=50.0)
              + box_rows(2, range(131, 170), 162.0, 100.0, vx=2.0, w=50.0, h=50.0)
              + box_rows(3, range(60, 171), 46.0, 100.0, vx=2.0, w=50.0, h=50.0))
    dets = pd.DataFrame({"frame": t.frame, "res": -1, "x": t.x + (t.track == 3) * shift,
                         "y": t.y, "w": t.w, "h": t.h, "conf": 0.9, "cls": t.cls})
    d = tmp_path / f"shift{-shift:g}"
    return _refine(_write(d, t), d / "o.txt", fps=10, context_file=_write(d, dets, "dets.txt"))


def test_own_detections_in_the_context_do_not_witness_an_occlusion(tmp_path):
    # 2 px closer: IoU 0.92 with track 3's own box, so it is track 3's detection, not an
    # occluder, though it would cover 52% of each hidden box
    res = _beside_the_path(tmp_path, -2.0)
    assert Ledger.read(res.ledger_path).header["inputs"]["context"]["format"] == "dets"
    assert _by(res, "link") == []
    assert res.tracks["track"].nunique() == 3
    # 8 px closer: IoU 0.73, a different box (a genuine occluder) -> the gap is witnessed
    res = _beside_the_path(tmp_path, -8.0)
    (link,) = _by(res, "link")
    assert link.tracks == [1, 2] and link.params["gate"] == "occluded"
    assert link.signals["witness"] == pytest.approx(1.0) and link.signals["occluders"] == [-1]
    assert link.decision is Decision.HUMAN_PENDING


def test_screening_cues_still_see_context_boxes_that_duplicate_a_row(tmp_path):
    # R22 applies to the occlusion mask and the witness only: a two-wheeler box (tracks
    # context) on the person's own box, moving with it, is the rider cue K
    ped = table(box_rows(1, range(50), 100.0, 110.0, vx=1.0, w=20.0, h=40.0))
    bike = table(box_rows(9, range(50), 100.0, 110.0, vx=1.0, w=20.0, h=40.0, cls=3))
    res = _refine(_write(tmp_path, ped), tmp_path / "o.txt", fps=10,
                  context_file=_write(tmp_path, bike, "bike.txt"))
    (ev,) = _by(res, "screen")
    assert ev.kind is EventKind.RECLASS and ev.signals["K"] == 1.0


# ---- final review M2 / M3: outputs never overwrite inputs or each other -----------------------


def test_refining_a_file_into_one_of_its_inputs_is_rejected(tmp_path):
    src = _write(tmp_path, table(box_rows(1, range(30), 0.0, 0.0, vx=2.0)), "t.txt")
    ctx = _write(tmp_path, table(box_rows(9, range(30), 500.0, 0.0, cls=2)), "c.txt")
    hints = _hints(tmp_path, (1, 3, 0.95))
    before = {p: p.read_bytes() for p in (src, ctx, hints)}
    cases = [
        ({}, src, r"out_file .* is track_file"),
        ({}, tmp_path / "sub" / ".." / "t.txt", r"out_file .* is track_file"),  # same file
        ({"context_file": ctx}, ctx, r"out_file .* is context_file"),
        ({"reclass_file": hints}, hints, r"out_file .* is reclass_file"),
        # the ledger written next to o.txt would replace the context file
        ({"context_file": tmp_path / "o.ledger.jsonl"}, tmp_path / "o.txt",
         r"the ledger file .* is context_file"),
    ]
    (tmp_path / "sub").mkdir()
    shutil.copy(ctx, tmp_path / "o.ledger.jsonl")
    for kw, out, match in cases:
        with pytest.raises(ValueError, match=match):
            TrackRefiner().refine(src, out, fps=10, verbose=False, **kw)
    assert {p: p.read_bytes() for p in before} == before  # nothing was overwritten
    assert not (tmp_path / "o.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.txt", "hints.csv",
                                                          "o.ledger.jsonl", "sub", "t.txt"]


def test_refine_batch_rejects_two_inputs_with_the_same_output_name(tmp_path):
    rows = table(box_rows(1, range(30), 0.0, 0.0, vx=2.0))
    a = _write(tmp_path / "a", rows, "day1_track.txt")
    b = _write(tmp_path / "b", rows, "day1_track.txt")
    c = _write(tmp_path / "c", rows, "day1.txt")  # "_track" is stripped: also day1_refined
    other = _write(tmp_path / "a", rows, "day2_track.txt")
    out_dir = tmp_path / "out"
    for files in ([a, b], [other, a, c]):
        with pytest.raises(ValueError, match=r"day1_refined\.txt"):
            TrackRefiner().refine_batch(files, output_path=out_dir, fps=10, verbose=False)
    assert not out_dir.exists()  # checked before any work starts
    got = TrackRefiner().refine_batch([a, other], output_path=out_dir, fps=10, verbose=False)
    assert got == [str(out_dir / "day1_refined.txt"), str(out_dir / "day2_refined.txt")]


def test_an_applied_split_records_its_tail_track_so_output_rows_trace_back(tmp_path):
    # final review M5: the tail's new ID is in the ledger, and id_map maps it to the output
    _two_splits(tmp_path)
    ledger = Ledger.read(tmp_path / "o.ledger.jsonl")
    id_map = ledger.header["id_map"]
    out = pd.read_csv(tmp_path / "o.txt", header=None)
    splits = {e.params["cut_frame"]: e for e in ledger.events if e.kind is EventKind.SPLIT}
    # splits apply latest cut first: the tail from 100 gets work ID 2, the one from 50 gets 3
    assert {c: e.signals["new_track"] for c, e in splits.items()} == {100: 2, 50: 3}
    for (cut, ev), end in zip(sorted(splits.items()), (99, 149), strict=True):
        assert ev.applied and ev.lineage == [[[1, 0, 149]]]
        tail = out[out[1] == id_map[str(ev.signals["new_track"])]]  # output rows of the tail
        assert tail[0].tolist() == list(range(cut, end + 1))
    assert out[out[1] == id_map["1"]][0].tolist() == list(range(50))  # the head keeps raw ID 1


@pytest.mark.parametrize("fill", [True, False])
def test_the_returned_table_is_the_written_file(tmp_path, fill):
    # final review M6: no extra raw_id column, and integer boxes even when fill is off
    rng = np.random.default_rng(1)
    rows = box_rows(1, range(0, 10), 10.3, 20.6, vx=2.7) + box_rows(1, range(15, 25), 50.2,
                                                                     20.6, vx=2.7)
    for r in rows:
        r[6] = round(float(rng.uniform(0.3, 0.95)), 3)
    cfg = RefineConfig.defaults()
    cfg.fill.enabled = fill
    refiner = TrackRefiner(cfg)
    got = refiner.refine(_write(tmp_path, table(rows)), tmp_path / "o.txt", fps=10,
                         verbose=False)
    assert got is refiner.last_result.tracks
    written = pd.read_csv(tmp_path / "o.txt", header=None, names=io.OUT_COLUMNS,
                          float_precision="round_trip")
    pd.testing.assert_frame_equal(got, written)
    assert list(got.columns) == io.OUT_COLUMNS and "raw_id" not in got
    assert int(got["interp"].sum()) == (5 if fill else 0)
