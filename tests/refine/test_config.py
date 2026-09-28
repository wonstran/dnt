import pytest
import yaml

from dnt.refine.config import RefineConfig, to_frames


def test_to_frames():
    assert to_frames(1.0, 25) == 25 and to_frames(0.01, 10) == 1 and to_frames(0.5, 10) == 5


def test_target_defaults():
    p, v = RefineConfig.defaults("person"), RefineConfig.defaults("vehicle")
    assert p.class_ids == [0] and p.link.class_groups == []
    assert v.class_ids == [2, 5, 7] and v.link.class_groups == [[2, 7]]
    assert p.switch.size_gate == 1.5 and p.link.weights == {"mot": 0.45, "app": 0.40, "gap": 0.15}
    assert p.screen.ramps["R"] == [0.3, 0.05] and p.orphan.ramp == [0.5, 0.1]


def test_yaml_round_trip_keeps_int_keys(tmp_path):
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.max_gap = 2.0
    cfg.to_yaml(tmp_path / "c.yaml")
    back = RefineConfig.from_yaml(tmp_path / "c.yaml")
    assert back == cfg and 36 in back.hints.reclass_class_map


def test_partial_yaml_overlays_target_defaults(tmp_path):
    (tmp_path / "v.yaml").write_text("target: vehicle\nlink:\n  accept_above: 0.85\n")
    cfg = RefineConfig.from_yaml(tmp_path / "v.yaml")
    assert cfg.link.accept_above == 0.85 and cfg.link.class_groups == [[2, 7]]
    cfg2 = RefineConfig.from_dict({"screen": {"ramps": {"R": [0.4, 0.1]}}})
    assert cfg2.screen.ramps["R"] == [0.4, 0.1] and cfg2.screen.ramps["J"] == [0.03, 0.005]


@pytest.mark.parametrize("data, match", [
    ({"link": {"bogus": 1}}, "link.bogus"),
    ({"screen": {"ramps": {"Q": [0, 1]}}}, "screen.ramps.Q"),
    ({"switch": {"accept_above": 0.4, "reject_below": 0.5}}, "switch"),
    ({"screen": {"static_score_cap": 0.9}}, "static_score_cap"),
    ({"link": {"weights": {"mot": 0.5, "app": 0.5, "gap": 0.5}}}, "link.weights"),
    ({"link": {"occluded_score_cap": 0.9}}, "occluded_score_cap"),
    ({"link": {"ambiguous_cap": 0.9}}, "ambiguous_cap"),
    ({"link": {"max_gap": 9.0}}, "max_gap_occluded"),
    ({"link": {"class_groups": [[2, 7], [7, 5]]}}, "class_groups"),
    ({"screen": {"mixed_score_cap": 0.9}}, "mixed_score_cap"),
    ({"screen": {"segment_at": 0.6}}, "segment_at"),
    ({"target": "vehicle", "encoder": {"kind": "reid"}}, "weights"),
    ({"vlm": {"backend": "openai_compat"}}, "vlm.model"),
    ({"encoder": {"kind": "clip"}}, "encoder.kind"),
    ({"fps": 0}, "fps"),
    ({"screen": {"ramps": {"R": [0.3, 0.3]}}}, "screen.ramps.R"),
])
def test_validation_rules(data, match):
    with pytest.raises(ValueError, match=match):
        RefineConfig.from_dict(data)


def test_anthropic_backend_needs_no_model():
    RefineConfig.from_dict({"vlm": {"backend": "anthropic"}})


def test_from_yaml_rejects_non_mapping(tmp_path):
    (tmp_path / "l.yaml").write_text(yaml.safe_dump([1, 2]))
    with pytest.raises(ValueError, match="mapping"):
        RefineConfig.from_yaml(tmp_path / "l.yaml")


@pytest.mark.parametrize("data, match", [
    ({"fps": "ten"}, "fps"),
    ({"link": {"max_gap": None}}, "link.max_gap"),
    ({"switch": {"accept_above": None}}, "switch.accept_above"),
    ({"screen": {"ramps": {"R": 5}}}, "screen.ramps.R"),
    ({"screen": {"ramps": {"R": [1, 2, 3]}}}, "screen.ramps.R"),
    ({"frame_size": 5}, "frame_size"),
    ({"link": {"class_groups": [2, 7]}}, r"link\.class_groups\[0\]"),
    ({"link": {"weights": {"mot": "a"}}}, "link.weights.mot"),
    ({"encoder": {"sample_every": "five"}}, "encoder.sample_every"),
    ({"switch": {"nis_hi": "abc"}}, "switch.nis_hi"),
    ({"class_ids": 5}, "class_ids"),
    ({"link": {"max_gap": True}}, "link.max_gap"),
])
def test_wrong_type_rejected_with_path(data, match):
    with pytest.raises(ValueError, match=match):
        RefineConfig.from_dict(data)


def test_valid_nulls_and_strings():
    cfg1 = RefineConfig.from_dict({"fps": None})
    assert cfg1.fps is None
    cfg2 = RefineConfig.from_dict({"frame_size": [640, 480]})
    assert cfg2.frame_size == [640, 480]
    cfg3 = RefineConfig.from_dict({"fill": {"max_gap": None}})
    assert cfg3.fill.max_gap is None
    cfg4 = RefineConfig.from_dict({"fps": 10})
    assert cfg4.fps == 10
    cfg5 = RefineConfig.from_dict({"hints": {"reclass_class_map": {"36": "scooter"}}})
    assert 36 in cfg5.hints.reclass_class_map


def test_mutation_independence():
    shared_dict = {"class_ids": [1, 2], "link": {"class_groups": [[2, 7]]}}
    cfg1 = RefineConfig.from_dict(shared_dict)
    cfg2 = RefineConfig.from_dict(shared_dict)
    cfg1.class_ids.append(3)
    cfg1.link.class_groups[0].append(5)
    assert cfg2.class_ids == [1, 2]
    assert cfg2.link.class_groups == [[2, 7]]


@pytest.mark.parametrize("data, match", [
    ({"orphan": {"ramp": [0.5]}}, "orphan.ramp"),
    ({"orphan": {"ramp": [1, 2, 3]}}, "orphan.ramp"),
    ({"orphan": {"ramp": "x"}}, "orphan.ramp"),
    ({"orphan": {"ramp": None}}, "orphan.ramp"),
    ({"hints": {"reclass_ramp": [0.5]}}, "hints.reclass_ramp"),
])
def test_plain_ramp_fields_rejected_with_path(data, match):
    with pytest.raises(ValueError, match=match):
        RefineConfig.from_dict(data)


def test_direct_ramp_assignment_validated():
    cfg = RefineConfig.defaults()
    cfg.orphan.ramp = [0.5]
    with pytest.raises(ValueError, match=r"orphan\.ramp"):
        cfg.validate()
