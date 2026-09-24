import json
import warnings
from pathlib import Path

import pytest
import yaml

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
LEGACY = DATA / "legacy_0.3.2"
SNAPSHOT = json.loads((DATA / "boxmot_16.0.11_yaml_defaults.json").read_text())
CLASSES = {"botsort": BoTSORTConfig, "boosttrack": BoostTrackConfig, "bytetrack": ByteTrackConfig,
           "ocsort": OCSORTConfig, "strongsort": StrongSORTConfig, "deepocsort": DeepOCSORTConfig,
           "hybridsort": HybridSORTConfig, "sfsort": SFSORTConfig}


def _no_legacy_warning(record):
    return not [w for w in record if "0.3.2.x" in str(w.message)]


def test_legacy_constants_match_0324_snapshots():
    defaults = json.loads((DATA / "legacy_0324_defaults.json").read_text())
    fields = json.loads((DATA / "legacy_0324_fields.json").read_text())
    assert defaults == bx.LEGACY_0324_DEFAULTS
    assert {k: sorted(v) for k, v in bx.LEGACY_0324_FIELDS.items()} == fields


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_old_yaml_files_give_untuned_args(name):
    path = LEGACY / "yaml" / f"{name}.yaml"
    for load in (lambda: CLASSES[name].import_yaml(str(path)),
                 lambda: Tracker.import_config_from_yaml(str(path)),
                 lambda: Tracker(config_yaml=str(path)).boxmot_config):
        with pytest.warns(DeprecationWarning, match="0.3.2.x"):
            cfg = load()
        assert _build_tracker_args(cfg, device="cpu") == SNAPSHOT[name]


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_old_dicts_through_existing_from_dict(name):
    old = json.loads((LEGACY / "dict" / f"{name}.json").read_text())
    for load in (CLASSES[name].from_dict, CLASSES[name].from_legacy_dict):
        with pytest.warns(DeprecationWarning, match="0.3.2.x"):
            cfg = load(old)
        assert _build_tracker_args(cfg, device="cpu") == SNAPSHOT[name]


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_033_roundtrip_is_strict_and_lossless(name):
    cls = CLASSES[name]
    tuned_field = {"sfsort": "high_th", "strongsort": "max_age", "botsort": "match_thresh",
                   "bytetrack": "match_thresh"}.get(name, "max_age")
    value = 0.66 if tuned_field in ("high_th", "match_thresh") else 44
    for cfg in (cls(), cls(**{tuned_field: value})):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert cls.from_dict(cfg.to_dict()) == cfg
            assert cls(**cfg.to_dict()) == cfg
        assert _no_legacy_warning(rec)
        assert next(iter(cfg.to_dict())) == "dnt_config_version"


def test_edited_legacy_values():
    old = json.loads((LEGACY / "dict" / "bytetrack.json").read_text())
    with pytest.warns(DeprecationWarning):
        cfg = ByteTrackConfig.from_dict({**old, "track_thresh": 0.55})
    assert cfg.tuning_values() == {"track_thresh": 0.55}
    old_sf = json.loads((LEGACY / "dict" / "sfsort.json").read_text())
    with pytest.raises(ValueError, match="use 'high_th'"):
        SFSORTConfig.from_dict({**old_sf, "det_thresh": 0.5})


def test_partial_dict_is_strict():
    with pytest.raises(ValueError, match="use 'high_th'"):
        _build_tracker_args(SFSORTConfig.from_dict({"det_thresh": 0.9}), device="cpu")


def test_hand_written_partial_yaml_is_strict(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump({"model": "bytetrack", "track_thres": 0.5}))  # typo
    cfg = Tracker.import_config_from_yaml(str(path))
    with pytest.raises(ValueError, match="'track_thres' is not an effective"):
        _build_tracker_args(cfg, device="cpu")


def test_legacy_flag_overrides():
    with pytest.raises(ValueError, match="not a dnt 0.3.2.x config"):  # noqa: RUF043 -- "." wildcards are harmless here; re.escape() would tighten matching
        ByteTrackConfig.from_dict({"model": "bytetrack"}, legacy=True)
    old = json.loads((LEGACY / "dict" / "bytetrack.json").read_text())
    cfg = ByteTrackConfig.from_dict(old, legacy=False)
    assert cfg.tuning_values()["track_thresh"] == 0.5


def test_unpacked_old_dict_warns():
    old = json.loads((LEGACY / "dict" / "bytetrack.json").read_text())
    with pytest.warns(UserWarning, match="unpacked 0.3.2.x dict"):
        ByteTrackConfig(**old)


def test_unknown_version_rejected():
    with pytest.raises(ValueError, match="dnt_config_version"):
        ByteTrackConfig.from_dict({"dnt_config_version": 3, "model": "bytetrack"})


def test_export_writes_version_first(tmp_path):
    path = tmp_path / "bt.yaml"
    ByteTrackConfig(track_thresh=0.5).export_yaml(str(path))
    assert path.read_text().splitlines()[0] == "dnt_config_version: 2"
    Tracker.export_config_to_yaml(str(path), ByteTrackConfig())
    assert path.read_text().splitlines()[0] == "dnt_config_version: 2"


def test_saved_passthrough_files_keep_0324_meaning():
    with pytest.warns(DeprecationWarning):
        cfg = Tracker(config_yaml=str(LEGACY / "passthrough" / "bytetrack_evolve.yaml")).boxmot_config
        plan = _plan_tracker(cfg, device="cpu")
    assert plan.evolve_param_dict == {"track_thresh": 0.7}
    for loaded in (lambda: Tracker.import_config_from_yaml(str(LEGACY / "passthrough" / "bytetrack_to_ocsort.yaml")),
                   lambda: ByteTrackConfig.from_dict(
                       json.loads((LEGACY / "passthrough" / "bytetrack_to_ocsort.json").read_text()))):
        with pytest.warns(DeprecationWarning):
            plan = _plan_tracker(loaded(), device="cpu")
        assert (plan.tracker_type, plan.evolve_param_dict) == ("ocsort", SNAPSHOT["ocsort"])
