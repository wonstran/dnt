"""Record the source line for every effective parameter (spec Appendix A).

    ../dnt-baseline-venv/bin/python tools/extract_param_evidence.py tests/data/boxmot_16.0.11_param_evidence.json
"""

import importlib.metadata
import json
import sys
from pathlib import Path

import boxmot

TRACKERS = Path(boxmot.__file__).parent / "trackers"
# (tracker, field, file relative to boxmot/trackers, 1-based line) — spec Appendix A
EVIDENCE = [
    ("bytetrack", "track_thresh", "bytetrack/bytetrack.py", 200),
    ("bytetrack", "match_thresh", "bytetrack/bytetrack.py", 232),
    ("bytetrack", "track_buffer", "bytetrack/bytetrack.py", 168),
    ("bytetrack", "frame_rate", "bytetrack/bytetrack.py", 168),
    ("bytetrack", "min_conf", "bytetrack/bytetrack.py", 202),
    ("botsort", "track_high_thresh", "botsort/botsort.py", 191),
    ("botsort", "track_low_thresh", "botsort/botsort.py", 188),
    ("botsort", "new_track_thresh", "botsort/botsort.py", 363),
    ("botsort", "match_thresh", "botsort/botsort.py", 253),
    ("botsort", "track_buffer", "botsort/botsort.py", 94),
    ("botsort", "frame_rate", "botsort/botsort.py", 94),
    ("botsort", "proximity_thresh", "botsort/botsort.py", 240),
    ("botsort", "appearance_thresh", "botsort/botsort.py", 246),
    ("botsort", "with_reid", "botsort/botsort.py", 124),
    ("botsort", "cmc_method", "botsort/botsort.py", 107),
    ("ocsort", "det_thresh", "ocsort/ocsort.py", 279),
    ("ocsort", "max_age", "ocsort/ocsort.py", 440),
    ("ocsort", "min_hits", "ocsort/ocsort.py", 430),
    ("ocsort", "iou_threshold", "ocsort/ocsort.py", 239),
    ("ocsort", "asso_func", "ocsort/ocsort.py", 337),
    ("ocsort", "delta_t", "ocsort/ocsort.py", 146),
    ("ocsort", "inertia", "ocsort/ocsort.py", 322),
    ("deepocsort", "det_thresh", "deepocsort/deepocsort.py", 340),
    ("deepocsort", "max_age", "deepocsort/deepocsort.py", 495),
    ("deepocsort", "min_hits", "deepocsort/deepocsort.py", 485),
    ("deepocsort", "iou_threshold", "deepocsort/deepocsort.py", 433),
    ("deepocsort", "asso_func", "deepocsort/deepocsort.py", 427),
    ("deepocsort", "delta_t", "deepocsort/deepocsort.py", 157),
    ("deepocsort", "inertia", "deepocsort/deepocsort.py", 406),
    ("strongsort", "max_cos_dist", "strongsort/strongsort.py", 80),
    ("strongsort", "nn_budget", "strongsort/strongsort.py", 80),
    ("strongsort", "max_iou_dist", "strongsort/sort/tracker.py", 148),
    ("strongsort", "max_age", "strongsort/sort/tracker.py", 132),
    ("strongsort", "n_init", "strongsort/sort/tracker.py", 164),
    ("strongsort", "ema_alpha", "strongsort/sort/track.py", 181),
    ("strongsort", "mc_lambda", "strongsort/sort/tracker.py", 119),
    ("hybridsort", "det_thresh", "hybridsort/hybridsort.py", 511),
    ("hybridsort", "max_age", "hybridsort/hybridsort.py", 723),
    ("hybridsort", "min_hits", "hybridsort/hybridsort.py", 710),
    ("hybridsort", "iou_threshold", "hybridsort/hybridsort.py", 625),
    ("hybridsort", "asso_func", "hybridsort/hybridsort.py", 578),
    ("boosttrack", "det_thresh", "boosttrack/boosttrack.py", 272),
    ("boosttrack", "max_age", "boosttrack/boosttrack.py", 338),
    ("boosttrack", "min_hits", "boosttrack/boosttrack.py", 332),
    ("boosttrack", "iou_threshold", "boosttrack/boosttrack.py", 299),
    ("sfsort", "high_th", "sfsort/sfsort.py", 251),
    ("sfsort", "low_th", "sfsort/sfsort.py", 216),
    ("sfsort", "new_track_th", "sfsort/sfsort.py", 252),
    ("sfsort", "match_th_first", "sfsort/sfsort.py", 253),
    ("sfsort", "match_th_second", "sfsort/sfsort.py", 224),
]
# NOTE: boxmot.__version__ misreports "16.0.10" for the 16.0.11 wheel; use
# importlib.metadata for the authoritative installed version (controller ruling).
assert importlib.metadata.version("boxmot") == "16.0.11"
result: dict[str, dict[str, dict]] = {}
for tracker, field, rel, line in EVIDENCE:
    code = (TRACKERS / rel).read_text().splitlines()[line - 1].strip()
    token = "asso_func_name" if (tracker, field) == ("hybridsort", "asso_func") else field
    assert token in code or field in ("track_buffer", "frame_rate"), (tracker, field, code)
    result.setdefault(tracker, {})[field] = {"file": rel, "line": line, "code": code}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, sort_keys=True))
print(len(EVIDENCE), "entries")
