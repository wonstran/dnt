import copy
import json
import shutil

from dnt.refine import RefineConfig, TrackRefiner
from dnt.refine.events import Ledger

from ._fixtures import DATA

BASE = DATA / "reversal_baseline"


def _strip_paths(obj):
    if isinstance(obj, dict):
        return {k: _strip_paths(v) for k, v in obj.items() if k not in ("path", "abs_path")}
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    return obj


def _header(h):
    h = copy.deepcopy(h)
    h.pop("dnt_version")
    h["config"].pop("reversal", None)
    h["inputs"] = _strip_paths(h["inputs"])
    return h


def test_the_disabled_stage_reproduces_the_pre_change_output(tmp_path):
    raw = tmp_path / "raw.csv"
    shutil.copy(BASE / "raw.csv", raw)
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    cfg.reversal.enabled = False
    refiner = TrackRefiner(cfg)
    refiner.refine(raw, tmp_path / "tracks.csv", fps=10, verbose=False)
    assert (tmp_path / "tracks.csv").read_bytes() == (BASE / "tracks.csv").read_bytes()
    new, old = Ledger.read(refiner.last_result.ledger_path), Ledger.read(BASE / "ledger.jsonl")
    assert [e.to_dict() for e in new.events] == [e.to_dict() for e in old.events]
    assert _header(new.header) == _header(old.header)
    assert json.dumps(new.header["config"]["reversal"])  # the new block is the only addition


def test_the_enabled_stage_splits_the_scene_vehicle(tmp_path):
    raw = tmp_path / "raw.csv"
    shutil.copy(BASE / "raw.csv", raw)
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    refiner = TrackRefiner(cfg)
    refiner.refine(raw, tmp_path / "tracks.csv", fps=10, verbose=False)
    (cut,) = [e for e in refiner.last_result.events if e.stage == "reversal"]
    assert cut.tracks == [1] and cut.params["cut_frame"] == 30 and cut.applied
