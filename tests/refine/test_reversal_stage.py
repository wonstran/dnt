import numpy as np
import pandas as pd
import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind
from dnt.refine.refiner import TrackRefiner, _Stages
from dnt.refine.reversal import propose_reversal_splits

from ._fixtures import box_rows, table
from ._reversal import FPS, rows, var_height, work_of


def _cfg():
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    return cfg


def _refine(tmp_path, df, cfg=None, name="t.txt", **kw):
    src = tmp_path / name
    df.to_csv(src, index=False, header=False)
    refiner = TrackRefiner(cfg or _cfg())
    out = refiner.refine(src, tmp_path / f"o_{name}", fps=FPS, verbose=kw.pop("verbose", False))
    return refiner, out


def test_events_have_the_spec_fields():
    work = work_of(rows(1, range(60)))
    (ev,) = propose_reversal_splits(work, _cfg(), FPS)
    assert (ev.stage, ev.kind, ev.tracks, ev.frames) == ("reversal", EventKind.SPLIT, [1], (30, 30))
    assert ev.params == {"cut_frame": 30} and ev.algo_score == 1.0 and ev.decision is None
    assert ev.lineage == [[[1, 0, 59]]]
    assert set(ev.signals) == {"angle", "change", "run", "speed_before", "speed_after",
                               "rows_before", "rows_after"}
    assert ev.signals["run"] == [23, 38]


def test_keys_do_not_depend_on_work_ids():
    work = work_of(rows(1, range(60)))
    moved = work.copy()
    moved["track"] += 1000
    (a,), (b,) = (propose_reversal_splits(w, _cfg(), FPS) for w in (work, moved))
    assert a.proposal_key == b.proposal_key and b.tracks == [1001]


def test_keys_identify_the_raw_track():
    (a,) = propose_reversal_splits(work_of(rows(1, range(60))), _cfg(), FPS)
    (b,) = propose_reversal_splits(work_of(rows(7, range(60))), _cfg(), FPS)
    assert a.proposal_key != b.proposal_key


def test_the_stage_splits_applies_and_records_the_sides():
    stages = _Stages(_cfg(), FPS, None, None, None, None, {})
    out, events = stages.run(work_of(rows(1, range(60))))
    (ev,) = [e for e in events if e.stage == "reversal"]
    assert ev.id == "reversal-r0-000001" and ev.decision is Decision.AUTO_ACCEPT and ev.applied
    assert ev.edit == {"kind": "SPLIT", "params": {"cut_frame": 30}}
    new = ev.signals["new_track"]
    assert ev.signals["sides"] == [[1, 0], [1, 1]]
    assert set(out["track"]) == {1, new}
    assert out.loc[out["track"] == 1, "frame"].max() == 29
    assert out.loc[out["track"] == new, "frame"].min() == 30
    assert stages.reversal_sides.cuts == {1: [30]}


def test_two_reversals_make_three_ids_with_three_sides():
    t = np.arange(110)
    x = {int(f): v for f, v in zip(t, 100.0 + np.r_[0.0, np.cumsum(
        np.where(t <= 30, 2.0, np.where(t <= 70, -2.0, 2.0))[1:])], strict=True)}
    stages = _Stages(_cfg(), FPS, None, None, None, None, {})
    out, events = stages.run(work_of(rows(1, t, lambda f: x[int(f)])))
    revs = sorted((e for e in events if e.stage == "reversal"),
                  key=lambda e: e.params["cut_frame"])
    assert [e.params["cut_frame"] for e in revs] == [30, 70]
    assert [e.signals["sides"] for e in revs] == [[[1, 0], [1, 1]], [[1, 1], [1, 2]]]
    assert out["track"].nunique() == 3


def test_the_screen_stage_is_told_which_raw_ids_were_split(monkeypatch):
    seen = {}

    def fake_screen(work, cfg, fps, sctx, cuts):
        seen["raw"] = set(sctx.split_raw_ids)
        return []

    monkeypatch.setattr("dnt.refine.refiner.propose_screen", fake_screen)
    _Stages(_cfg(), FPS, None, None, None, None, {}).run(work_of(rows(1, range(60))))
    assert 1 in seen["raw"]


def test_the_person_target_leaves_tracks_alone():
    cfg = RefineConfig.defaults("person")
    stages = _Stages(cfg, FPS, None, None, None, None, {})
    _, events = stages.run(work_of(rows(1, range(60), cls=0)))
    assert not [e for e in events if e.stage == "reversal"]
    assert stages._sides() is None


def test_a_table_of_short_tracks_is_left_alone():
    work = work_of(rows(1, range(9)), rows(2, range(20, 29)))
    stages = _Stages(_cfg(), FPS, None, None, None, None, {})
    _, events = stages.run(work)
    assert not [e for e in events if e.stage == "reversal"] and stages._sides() is None


@pytest.mark.parametrize("h_of", [lambda f: 10.0, var_height])
def test_refine_writes_the_two_vehicles_as_two_tracks(tmp_path, h_of):
    refiner, out = _refine(tmp_path, table(rows(1, range(60), h_of=h_of)))
    assert out["track"].nunique() == 2
    ev = next(e for e in refiner.last_result.events if e.stage == "reversal")
    assert ev.params["cut_frame"] == (30 if h_of(0) == h_of(20) else 25)
    assert not [e for e in refiner.last_result.events if e.kind is EventKind.LINK and e.applied]


def test_progress_numbers_the_new_stage_first(tmp_path, capsys):
    _refine(tmp_path, table(rows(1, range(60))), verbose=True)
    err = capsys.readouterr().err
    assert "Stage 1/8 reversal" in err and "Stage 2/8 switch" in err and "Stage 8/8 Save" in err


def test_a_reused_refiner_does_not_carry_cuts_between_clips(tmp_path):
    clip1 = table(rows(1, range(60)))
    clip2 = table(box_rows(1, range(0, 60), 100.0, 100.0, vx=3.0, w=40.0, h=30.0, cls=2),
                  box_rows(5, range(61, 100), 100.0 + 3.0 * 61, 100.0, vx=3.0, w=40.0,
                           h=30.0, cls=2))
    refiner = TrackRefiner(_cfg())
    for i, df in enumerate((clip1, clip2)):
        p = tmp_path / f"c{i}.txt"
        df.to_csv(p, index=False, header=False)
        reused = refiner.refine(p, tmp_path / f"r{i}.txt", fps=FPS, verbose=False)
    _, fresh = _refine(tmp_path, clip2, name="fresh.txt")
    pd.testing.assert_frame_equal(reused.reset_index(drop=True), fresh.reset_index(drop=True))
