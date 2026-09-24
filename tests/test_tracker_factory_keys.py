import json
import shutil
from pathlib import Path

import boxmot
import pytest
import yaml
from boxmot.trackers.tracker_zoo import get_tracker_config

from dnt import _device
from dnt.track.tracker import BoTSORTConfig, ByteTrackConfig, Tracker, _plan_tracker

DATA = Path(__file__).parent / "data"
SNAPSHOT = json.loads((DATA / "boxmot_16.0.11_yaml_defaults.json").read_text())
REID_NONE = json.loads((DATA / "tracker_type_reid_none.json").read_text())


@pytest.fixture
def spy(monkeypatch):
    seen = {}
    monkeypatch.setattr(boxmot, "create_tracker", lambda **kw: seen.update(kw) or "T")
    return seen


# ---- evolve_param_dict (the path review 3 identified) ------------------------
def test_evolve_param_dict_passed_verbatim(spy):
    cfg = ByteTrackConfig(extra_kwargs={"evolve_param_dict": {"track_thresh": 0.7}})
    with pytest.warns(DeprecationWarning, match="evolve_param_dict"):
        Tracker._build_boxmot_tracker(cfg.model, cfg, device="cpu", half=False)
    assert spy["evolve_param_dict"] == {"track_thresh": 0.7}


def test_evolve_param_dict_full_replacement_semantics():
    cfg = ByteTrackConfig(extra_kwargs={"evolve_param_dict": {"track_thresh": 0.7}})
    with pytest.warns(DeprecationWarning):
        t = Tracker._build_boxmot_tracker(cfg.model, cfg, device="cpu", half=False)
    assert t.track_thresh == 0.7
    assert t.match_thresh == 0.8  # ByteTrack constructor default, not YAML 0.9 — as in 0.3.2.4


def test_field_overlays_evolve_param_dict():
    cfg = ByteTrackConfig(match_thresh=0.5, extra_kwargs={"evolve_param_dict": {"track_thresh": 0.7}})
    with pytest.warns(DeprecationWarning):
        plan = _plan_tracker(cfg, device="cpu")
    assert plan.evolve_param_dict == {"track_thresh": 0.7, "match_thresh": 0.5}


def test_passthrough_key_boxmot_ignores_warns_but_passes():
    cfg = ByteTrackConfig(extra_kwargs={"evolve_param_dict": {"bogus_key": 1}})
    with pytest.warns(UserWarning, match="bogus_key"):
        plan = _plan_tracker(cfg, device="cpu")
    assert plan.evolve_param_dict == {"bogus_key": 1}


# ---- tracker_config -------------------------------------------------------------
def test_tracker_config_file_is_base(tmp_path):
    src = yaml.safe_load(Path(get_tracker_config("bytetrack")).read_text())
    src["track_thresh"]["default"] = 0.65
    path = tmp_path / "bt.yaml"
    path.write_text(yaml.safe_dump(src))
    cfg = ByteTrackConfig(match_thresh=0.5, extra_kwargs={"tracker_config": str(path)})
    with pytest.warns(DeprecationWarning, match="tracker_config"):
        plan = _plan_tracker(cfg, device="cpu")
    assert plan.evolve_param_dict == {**SNAPSHOT["bytetrack"], "track_thresh": 0.65, "match_thresh": 0.5}


def test_tracker_config_missing_file():
    with pytest.raises(FileNotFoundError):
        _plan_tracker(ByteTrackConfig(extra_kwargs={"tracker_config": "/nope.yaml"}), device="cpu")


def test_both_factory_keys_rejected(tmp_path):
    path = tmp_path / "bt.yaml"
    shutil.copy(get_tracker_config("bytetrack"), path)
    cfg = ByteTrackConfig(extra_kwargs={"evolve_param_dict": {}, "tracker_config": str(path)})
    with pytest.raises(ValueError, match="only one"):
        _plan_tracker(cfg, device="cpu")


# ---- per_class / reid_weights -----------------------------------------------------
def test_per_class_in_extra_wins():
    with pytest.warns(DeprecationWarning, match="per_class"):
        plan = _plan_tracker(ByteTrackConfig(extra_kwargs={"per_class": True}), device="cpu")
    assert plan.per_class is True


def test_reid_weights_in_extra_on_reid_tracker_wins():
    with pytest.warns(DeprecationWarning, match="reid_weights"):
        plan = _plan_tracker(BoTSORTConfig(extra_kwargs={"reid_weights": "clip_vehicleid.pt"}), device="cpu")
    assert plan.reid_weights == "clip_vehicleid.pt"


def test_reid_weights_in_extra_on_motion_tracker_raises():
    with pytest.raises(ValueError, match="uses no ReID weights"):
        _plan_tracker(ByteTrackConfig(extra_kwargs={"reid_weights": "x.pt"}), device="cpu")


# ---- tracker_type (§2.8.1) --------------------------------------------------------
def test_tracker_type_equal_to_model_is_noop(recwarn):
    plan = _plan_tracker(ByteTrackConfig(extra_kwargs={"tracker_type": "bytetrack"}), device="cpu")
    assert plan.tracker_type == "bytetrack"
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_tracker_type_override_regression(spy):
    cfg = ByteTrackConfig(extra_kwargs={"tracker_type": "ocsort"})
    with pytest.warns(DeprecationWarning, match="OCSORTConfig"):
        Tracker._build_boxmot_tracker(cfg.model, cfg, device="cpu", half=False)
    assert spy["tracker_type"] == "ocsort"
    assert spy["evolve_param_dict"] == SNAPSHOT["ocsort"]
    assert spy["reid_weights"] is None


def test_tracker_type_override_builds_target_class():
    from boxmot.trackers.ocsort.ocsort import OcSort

    cfg = ByteTrackConfig(extra_kwargs={"tracker_type": "ocsort"})
    with pytest.warns(DeprecationWarning):
        assert isinstance(Tracker._build_boxmot_tracker(cfg.model, cfg, device="cpu", half=False), OcSort)


def test_tracker_type_override_with_source_field_raises():
    with pytest.raises(ValueError, match="set them on OCSORTConfig"):
        _plan_tracker(ByteTrackConfig(track_thresh=0.5, extra_kwargs={"tracker_type": "ocsort"}), device="cpu")


def test_tracker_type_override_with_evolve_uses_dict():
    cfg = ByteTrackConfig(extra_kwargs={"tracker_type": "ocsort", "evolve_param_dict": {"det_thresh": 0.4}})
    with pytest.warns(DeprecationWarning):
        plan = _plan_tracker(cfg, device="cpu")
    assert (plan.tracker_type, plan.evolve_param_dict) == ("ocsort", {"det_thresh": 0.4})


def test_tracker_type_unknown_raises():
    with pytest.raises(ValueError, match="unknown tracker_type"):
        _plan_tracker(ByteTrackConfig(extra_kwargs={"tracker_type": "sort"}), device="cpu")


def test_tracker_type_carries_source_reid_weights():
    cfg = BoTSORTConfig(extra_kwargs={"tracker_type": "strongsort"})
    with pytest.warns(DeprecationWarning):
        plan = _plan_tracker(cfg, device="cpu")
    assert plan.reid_weights == BoTSORTConfig().reid_weights


@pytest.mark.parametrize("target", sorted(REID_NONE))
def test_non_reid_config_to_reid_target_matches_0324(target):
    cfg = ByteTrackConfig(extra_kwargs={"tracker_type": target})
    if REID_NONE[target] == "ok":
        with pytest.warns(DeprecationWarning):
            assert _plan_tracker(cfg, device="cpu").reid_weights is None
    else:
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="needs ReID weights"):
            _plan_tracker(cfg, device="cpu")


# ---- device / half precedence -------------------------------------------------
def test_extra_device_used_when_tracker_device_none():
    with pytest.warns(DeprecationWarning, match="device"):
        assert _plan_tracker(ByteTrackConfig(extra_kwargs={"device": "cpu"}), device=None).device == "cpu"


def test_extra_device_equal_is_dropped():
    assert _plan_tracker(ByteTrackConfig(extra_kwargs={"device": "cpu"}), device="cpu").device == "cpu"


def test_extra_device_conflict_raises(monkeypatch):
    monkeypatch.setattr(_device, "_available", lambda b: b in ("cuda", "cpu"))
    with pytest.raises(ValueError, match="Tracker\\(device"):
        _plan_tracker(ByteTrackConfig(extra_kwargs={"device": "cuda:0"}), device="cpu")


def test_extra_half_precedence():
    with pytest.raises(ValueError, match=r"Tracker\(.*half"):
        _plan_tracker(ByteTrackConfig(extra_kwargs={"half": True}), device="cpu", half=False)
    with pytest.warns(DeprecationWarning, match="half"):
        assert _plan_tracker(ByteTrackConfig(extra_kwargs={"half": True}), device="cpu", half=None).half is True


# ---- extra_kwargs None values are dropped, not treated as user overrides (0.3.2.4 parity) -----
def test_extra_kwargs_none_tuning_value_gives_untuned_snapshot():
    cfg = ByteTrackConfig(extra_kwargs={"track_thresh": None})
    plan = _plan_tracker(cfg, device="cpu")
    assert plan.evolve_param_dict == SNAPSHOT["bytetrack"]


def test_extra_kwargs_none_per_class_keeps_default_with_no_warning(recwarn):
    plan = _plan_tracker(ByteTrackConfig(extra_kwargs={"per_class": None}), device="cpu")
    assert plan.per_class is False
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
