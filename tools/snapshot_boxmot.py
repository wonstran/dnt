"""Snapshot BoxMOT 16.0.11 YAML defaults and 0.3.2.4 config defaults/field sets.

    ../dnt-baseline-venv/bin/python tools/snapshot_boxmot.py tests/data
"""

import importlib.metadata
import json
import sys
from dataclasses import MISSING, fields
from enum import Enum
from pathlib import Path

import yaml
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, get_tracker_config

import dnt
from dnt.track.tracker import Tracker

# NOTE: boxmot.__version__ misreports "16.0.10" for the 16.0.11 wheel; use
# importlib.metadata for the authoritative installed version (controller ruling).
assert importlib.metadata.version("boxmot") == "16.0.11" and dnt.__version__ == "0.3.2.4"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

yaml_defaults = {}
for name in sorted(TRACKER_MAPPING):
    data = yaml.safe_load(Path(get_tracker_config(name)).read_text())
    yaml_defaults[name] = {k: v["default"] for k, v in data.items()}
(out / "boxmot_16.0.11_yaml_defaults.json").write_text(json.dumps(yaml_defaults, indent=2, sort_keys=True))

NON_TUNING = {"model", "per_class", "extra_kwargs", "reid_weights"}
legacy_defaults, legacy_fields = {}, {}
for name in sorted(TRACKER_MAPPING):
    from dnt.track.tracker import MOTModels

    cls = Tracker._params_class_for_model(MOTModels(name))
    legacy_fields[name] = sorted(f.name for f in fields(cls))
    legacy_defaults[name] = {
        f.name: (f.default.value if isinstance(f.default, Enum) else f.default)
        for f in fields(cls)
        if f.name not in NON_TUNING and f.default is not MISSING
    }
(out / "legacy_0324_defaults.json").write_text(json.dumps(legacy_defaults, indent=2, sort_keys=True))
(out / "legacy_0324_fields.json").write_text(json.dumps(legacy_fields, indent=2, sort_keys=True))
print("ok")
