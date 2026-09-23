"""Export 0.3.2.4 tracker configs as YAML and to_dict() JSON (spec §2.6, §2.7).

Run ONLY with the 0.3.2.4 baseline interpreter:
    ../dnt-baseline-venv/bin/python tools/export_legacy_fixtures.py tests/data/legacy_0.3.2
"""

import json
import sys
from pathlib import Path

import dnt
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
)

assert dnt.__version__ == "0.3.2.4", dnt.__version__
out = Path(sys.argv[1])
CLASSES = {
    "botsort": BoTSORTConfig,
    "boosttrack": BoostTrackConfig,
    "bytetrack": ByteTrackConfig,
    "ocsort": OCSORTConfig,
    "strongsort": StrongSORTConfig,
    "deepocsort": DeepOCSORTConfig,
    "hybridsort": HybridSORTConfig,
    "sfsort": SFSORTConfig,
}
for sub in ("yaml", "dict", "passthrough"):
    (out / sub).mkdir(parents=True, exist_ok=True)
for name, cls in CLASSES.items():
    Tracker.export_config_to_yaml(str(out / "yaml" / f"{name}.yaml"), cls())
    (out / "dict" / f"{name}.json").write_text(json.dumps(cls().to_dict(), indent=2))

evolve = ByteTrackConfig(extra_kwargs={"evolve_param_dict": {"track_thresh": 0.7}})
Tracker.export_config_to_yaml(str(out / "passthrough" / "bytetrack_evolve.yaml"), evolve)
override = ByteTrackConfig(extra_kwargs={"tracker_type": "ocsort"})
Tracker.export_config_to_yaml(str(out / "passthrough" / "bytetrack_to_ocsort.yaml"), override)
(out / "passthrough" / "bytetrack_to_ocsort.json").write_text(json.dumps(override.to_dict(), indent=2))
print("exported", sorted(p.name for p in out.rglob("*.*")))
