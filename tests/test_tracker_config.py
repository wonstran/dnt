import dataclasses
import warnings

import pytest

from dnt.track.tracker import (
    CONFIG_VERSION,
    BoostTrackConfig,
    BoTSORTConfig,
    ByteTrackConfig,
    DeepOCSORTConfig,
    HybridSORTConfig,
    MOTModels,
    OCSORTConfig,
    ReIDWeights,
    SFSORTConfig,
    StrongSORTConfig,
)

ALL = [BoTSORTConfig, BoostTrackConfig, ByteTrackConfig, OCSORTConfig,
       StrongSORTConfig, DeepOCSORTConfig, HybridSORTConfig, SFSORTConfig]


@pytest.mark.parametrize("cls", ALL)
def test_positional_arguments_rejected(cls):
    with pytest.raises(TypeError):
        cls(ReIDWeights.CLIP_VEHICLEID)


@pytest.mark.parametrize("cls", ALL)
def test_untuned_config_has_no_tuning_values(cls):
    cfg = cls()
    assert cfg.tuning_values() == {}
    assert cfg.dnt_config_version == CONFIG_VERSION == 2


def test_new_fields_exist():
    names = {c: {f.name for f in dataclasses.fields(c)} for c in ALL}
    assert "min_conf" in names[ByteTrackConfig]
    assert {"cmc_method", "frame_rate"} <= names[BoTSORTConfig]
    assert {"high_th", "low_th", "new_track_th", "match_th_first", "match_th_second"} <= names[SFSORTConfig]
    assert "max_cos_dist" in names[StrongSORTConfig]


def test_model_string_coerced():
    assert ByteTrackConfig(model="bytetrack").model is MOTModels.BYTE_TRACK


def test_bad_version_rejected():
    with pytest.raises(ValueError, match="dnt_config_version"):
        ByteTrackConfig(dnt_config_version=3)


def test_max_dist_alias_warns_and_maps():
    with pytest.warns(DeprecationWarning, match="max_cos_dist"):
        cfg = StrongSORTConfig(max_dist=0.3)
    assert cfg.max_cos_dist == 0.3
    assert cfg.max_dist is None


def test_max_dist_and_max_cos_dist_conflict():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="both"):
            StrongSORTConfig(max_dist=0.3, max_cos_dist=0.4)


def test_reid_default_kept():
    assert BoTSORTConfig().reid_weights == ReIDWeights.OSNET_X1_0_MSMT17


def test_max_dist_set_after_construction_is_not_ignored():
    from dnt.track.tracker import _build_tracker_args

    cfg = StrongSORTConfig()
    cfg.max_dist = 0.3
    with pytest.warns(DeprecationWarning, match="max_cos_dist"):
        args = _build_tracker_args(cfg, device="cpu")
    assert args["max_cos_dist"] == 0.3


def test_max_dist_set_after_construction_conflicts_with_max_cos_dist():
    cfg = StrongSORTConfig(max_cos_dist=0.4)
    cfg.max_dist = 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="both"):
            cfg.tuning_values()
