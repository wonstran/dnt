import importlib.metadata
import importlib.util
import json
from pathlib import Path

import boxmot
import pytest

from dnt.track import _boxmot_compat as bx
from dnt.track.tracker import (
    BoostTrackConfig,
    BoTSORTConfig,
    ByteTrackConfig,
    DeepOCSORTConfig,
    HybridSORTConfig,
    OCSORTConfig,
    SFSORTConfig,
    StrongSORTConfig,
    Tracker,
    _build_tracker_args,
    _plan_tracker,
)

DATA = Path(__file__).parent / "data"
SNAPSHOT = json.loads((DATA / "boxmot_16.0.11_yaml_defaults.json").read_text())
EVIDENCE = json.loads((DATA / "boxmot_16.0.11_param_evidence.json").read_text())
ALL = [BoTSORTConfig, BoostTrackConfig, ByteTrackConfig, OCSORTConfig,
       StrongSORTConfig, DeepOCSORTConfig, HybridSORTConfig, SFSORTConfig]


# ---- group 1: propagation (NOT evidence of effect) --------------------------------
@pytest.mark.parametrize("cls", ALL)
def test_untuned_args_equal_boxmot_16011_snapshot(cls):
    cfg = cls()
    assert _build_tracker_args(cfg, device="cpu") == SNAPSHOT[cfg.model.value]


def test_installed_boxmot_yaml_equals_snapshot():
    installed = importlib.metadata.version("boxmot")
    assert installed == bx.BOXMOT_VERSION, f"installed boxmot {installed}"
    for name in bx.TRACKER_TYPES:
        assert bx.boxmot_yaml_defaults(name) == SNAPSHOT[name], name


PROPAGATION = [
    (ByteTrackConfig(track_thresh=0.55, match_thresh=0.7),
     lambda t: (t.track_thresh, t.match_thresh), (0.55, 0.7)),
    (BoTSORTConfig(track_high_thresh=0.55, match_thresh=0.7),
     lambda t: (t.track_high_thresh, t.match_thresh), (0.55, 0.7)),
    (OCSORTConfig(det_thresh=0.55, max_age=44), lambda t: (t.det_thresh, t.max_age), (0.55, 44)),
    (DeepOCSORTConfig(det_thresh=0.55, max_age=44),
     lambda t: (t.det_thresh, t.max_age), (0.55, 44)),
    (StrongSORTConfig(max_age=44, max_iou_dist=0.66),
     lambda t: (t.max_age, t.tracker.max_iou_dist), (44, 0.66)),
    (HybridSORTConfig(det_thresh=0.55, max_age=44),
     lambda t: (t.det_thresh, t.max_age), (0.55, 44)),
    (BoostTrackConfig(det_thresh=0.55, max_age=44),
     lambda t: (t.det_thresh, t.max_age), (0.55, 44)),
    (SFSORTConfig(high_th=0.65, match_th_second=0.35),
     lambda t: (t.high_th, t.match_th_second), (0.65, 0.35)),
]


@pytest.mark.parametrize(("cfg", "read", "expected"), PROPAGATION, ids=lambda x: type(x).__name__)
def test_values_reach_constructed_tracker(cfg, read, expected):
    tracker = Tracker._build_boxmot_tracker(cfg.model, cfg, device="cpu", half=False)
    assert read(tracker) == expected


def test_create_tracker_kwargs_untuned_bytetrack(monkeypatch):
    seen = {}
    monkeypatch.setattr(boxmot, "create_tracker", lambda **kw: seen.update(kw) or "T")
    assert Tracker._build_boxmot_tracker(None, ByteTrackConfig(), device="cpu", half=False) == "T"
    assert seen == {
        "tracker_type": "bytetrack", "reid_weights": None, "device": "cpu", "half": False,
        "per_class": False, "evolve_param_dict": SNAPSHOT["bytetrack"],
    }


def test_reid_default_weight_for_reid_trackers_only():
    assert _plan_tracker(BoTSORTConfig(), device="cpu").reid_weights == Tracker.DEFAULT_REID_WEIGHT
    assert _plan_tracker(ByteTrackConfig(), device="cpu").reid_weights is None


# ---- group 2: source evidence ----------------------------------------------------
def test_effective_params_equal_evidence_keys():
    assert {t: frozenset(f) for t, f in EVIDENCE.items()} == bx.EFFECTIVE_PARAMS


def test_evidence_lines_present_in_installed_source():
    root = Path(importlib.util.find_spec("boxmot").origin).parent / "trackers"
    for tracker, entries in EVIDENCE.items():
        for field, ev in entries.items():
            line = (root / ev["file"]).read_text().splitlines()[ev["line"] - 1].strip()
            assert line == ev["code"], (tracker, field, ev)


def test_allowlist_names_are_real_parameters():
    for t, names in bx.EFFECTIVE_PARAMS.items():
        assert names <= bx.accepted_params(t), (t, names - bx.accepted_params(t))


# ---- validation -------------------------------------------------------------------
@pytest.mark.parametrize(("cfg", "match"), [
    (SFSORTConfig(det_thresh=0.5), "use 'high_th'"),
    (SFSORTConfig(max_age=5), "not exposed in dnt 0.3.3"),
    (SFSORTConfig(min_hits=2), "no equivalent"),
    (SFSORTConfig(iou_threshold=0.2), "no equivalent"),
    (SFSORTConfig(asso_func="giou"), "no equivalent"),
    (BoostTrackConfig(asso_func="giou"), "no equivalent"),
])
def test_non_effective_field_raises_with_hint(cfg, match):
    with pytest.raises(ValueError, match=match):
        _build_tracker_args(cfg, device="cpu")


def test_non_effective_via_attribute_and_extra_kwargs():
    cfg = SFSORTConfig()
    cfg.max_age = 5
    with pytest.raises(ValueError, match="no effect on sfsort"):
        _build_tracker_args(cfg, device="cpu")
    with pytest.raises(ValueError, match="no effect on sfsort"):
        _build_tracker_args(SFSORTConfig(extra_kwargs={"min_hits": 2}), device="cpu")


def test_unknown_extra_key_raises():
    with pytest.raises(ValueError, match="not an effective BoxMOT parameter for bytetrack"):
        _build_tracker_args(ByteTrackConfig(extra_kwargs={"typo_thresh": 1}), device="cpu")


def test_field_and_extra_conflict_raises():
    with pytest.raises(ValueError, match="both"):
        cfg = ByteTrackConfig(track_thresh=0.5, extra_kwargs={"track_thresh": 0.4})
        _build_tracker_args(cfg, device="cpu")


@pytest.mark.parametrize("cfg", [
    SFSORTConfig(match_th_first=0.8),
    SFSORTConfig(new_track_th=0.5),             # below BoxMOT high_th 0.6
    SFSORTConfig(low_th=0.7),                   # above BoxMOT high_th 0.6
    SFSORTConfig(high_th=0.9, new_track_th=0.8),
    SFSORTConfig(high_th=1.2),
])
def test_sfsort_out_of_range_raises(cfg):
    with pytest.raises(ValueError, match="range"):
        _build_tracker_args(cfg, device="cpu")


def test_sfsort_unset_field_clamped_by_boxmot_is_allowed():
    args = _build_tracker_args(SFSORTConfig(high_th=0.95), device="cpu")
    assert args["high_th"] == 0.95 and args["new_track_th"] == SNAPSHOT["sfsort"]["new_track_th"]
