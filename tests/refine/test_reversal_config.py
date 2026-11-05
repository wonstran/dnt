import re

import pytest

from dnt.refine.config import RefineConfig


def test_vehicle_defaults_turn_the_stage_on_and_person_defaults_leave_it_off():
    assert RefineConfig.defaults("vehicle").reversal.enabled is True
    assert RefineConfig.defaults("person").reversal.enabled is False


def test_defaults_are_the_script_values():
    rc = RefineConfig.defaults("vehicle").reversal
    assert (rc.window, rc.min_speed, rc.min_angle, rc.min_rows, rc.min_side_rows) == (
        2.0, 0.5, 135.0, 5, 3,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("window", 0), ("window", -1.0), ("window", "2"), ("window", True),
        ("min_speed", -0.1), ("min_speed", float("nan")), ("min_speed", True),
        ("min_angle", 90), ("min_angle", 181), ("min_angle", "135"),
        ("min_rows", 0), ("min_rows", 5.0), ("min_rows", True),
        ("min_side_rows", 1), ("min_side_rows", 3.0),
        ("enabled", "yes"),
    ],
)
def test_bad_values_are_rejected_with_the_field_name(key, value):
    with pytest.raises(ValueError, match=re.escape(f"reversal.{key}")):
        RefineConfig.from_dict({"target": "vehicle", "reversal": {key: value}})


def test_a_partial_block_overlays_the_target_defaults():
    cfg = RefineConfig.from_dict({"target": "vehicle", "reversal": {"min_angle": 150}})
    assert cfg.reversal.enabled is True
    assert cfg.reversal.min_angle == 150 and cfg.reversal.window == 2.0


def test_unknown_reversal_keys_raise():
    with pytest.raises(ValueError, match=re.escape("reversal.min_angel")):
        RefineConfig.from_dict({"reversal": {"min_angel": 150}})


def test_the_block_round_trips_through_yaml(tmp_path):
    cfg = RefineConfig.defaults("vehicle")
    cfg.reversal.min_rows = 7
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert RefineConfig.from_yaml(path).reversal.min_rows == 7
