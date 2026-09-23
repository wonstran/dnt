"""BoxMOT 16.0.11 compatibility layer for dnt 0.3.3 (spec §2).

Everything that depends on the BoxMOT version lives here, so the 0.4
migration to BoxMOT 25 replaces this one module.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

BOXMOT_VERSION = "16.0.11"
TRACKER_TYPES = (
    "botsort", "boosttrack", "bytetrack", "ocsort", "strongsort", "deepocsort", "hybridsort",
    "sfsort",
)
REID_TRACKERS = frozenset({"botsort", "strongsort", "deepocsort", "hybridsort", "boosttrack"})

# Fields dnt exposes that BoxMOT 16.0.11 reads after construction (spec Appendix A).
EFFECTIVE_PARAMS: dict[str, frozenset[str]] = {
    "bytetrack": frozenset(
        {"track_thresh", "match_thresh", "track_buffer", "frame_rate", "min_conf"}
    ),
    "botsort": frozenset({
        "track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh", "track_buffer",
        "proximity_thresh", "appearance_thresh", "with_reid", "cmc_method", "frame_rate",
    }),
    "ocsort": frozenset(
        {"det_thresh", "max_age", "min_hits", "iou_threshold", "asso_func", "delta_t", "inertia"}
    ),
    "deepocsort": frozenset(
        {"det_thresh", "max_age", "min_hits", "iou_threshold", "asso_func", "delta_t", "inertia"}
    ),
    "strongsort": frozenset(
        {"max_cos_dist", "max_iou_dist", "max_age", "n_init", "nn_budget", "ema_alpha", "mc_lambda"}
    ),
    "hybridsort": frozenset({"det_thresh", "max_age", "min_hits", "iou_threshold", "asso_func"}),
    "boosttrack": frozenset({"det_thresh", "max_age", "min_hits", "iou_threshold"}),
    "sfsort": frozenset({"high_th", "low_th", "new_track_th", "match_th_first", "match_th_second"}),
}

_NO_EQUIVALENT = "no equivalent for this tracker; remove the argument"
NON_EFFECTIVE_HINTS: dict[str, dict[str, str]] = {
    "sfsort": {
        "det_thresh": "use 'high_th' for the detection threshold",
        "max_age": (
            "SF-SORT's track timeouts (marginal_timeout/central_timeout) are not exposed "
            "in dnt 0.3.3; remove 'max_age'"
        ),
        "min_hits": _NO_EQUIVALENT,
        "iou_threshold": _NO_EQUIVALENT,
        "asso_func": _NO_EQUIVALENT,
    },
    "boosttrack": {"asso_func": _NO_EQUIVALENT},
}

# SF-SORT clamps these silently (sfsort.py:104-111, 336-340); dnt rejects instead.
SFSORT_DEFAULT_HIGH_TH = 0.6
SFSORT_RANGES: dict[str, tuple[float | str, float | str]] = {
    "high_th": (0.0, 1.0),
    "match_th_first": (0.0, 0.67),
    "new_track_th": ("high_th", 1.0),
    "low_th": (0.0, "high_th"),
    "match_th_second": (0.0, 1.0),
}


@cache
def tracker_class(tracker_type: str) -> type:
    """Return BoxMOT's tracker class for `tracker_type`."""
    from boxmot.trackers.tracker_zoo import TRACKER_MAPPING

    module_path, class_name = TRACKER_MAPPING[tracker_type].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


@cache
def accepted_params(tracker_type: str) -> frozenset[str]:
    """Parameters the tracker's __init__ or BaseTracker.__init__ name explicitly."""
    from boxmot.trackers.basetracker import BaseTracker

    names: set[str] = set()
    for fn in (tracker_class(tracker_type).__init__, BaseTracker.__init__):
        for p in inspect.signature(fn).parameters.values():
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                names.add(p.name)
    return frozenset(names)


def load_boxmot_yaml(path: str | Path) -> dict[str, Any]:
    """Load a BoxMOT tracker YAML the way `create_tracker` does: {k: v['default']}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {k: v["default"] for k, v in data.items()}


def boxmot_yaml_defaults(tracker_type: str) -> dict[str, Any]:
    """BoxMOT's own default arguments for `tracker_type`."""
    from boxmot.trackers.tracker_zoo import get_tracker_config

    return load_boxmot_yaml(get_tracker_config(tracker_type))


@dataclass(frozen=True)
class TrackerBuild:
    """Everything `create_tracker` needs, resolved from a dnt config (spec §2.3, §2.8)."""

    tracker_type: str
    evolve_param_dict: dict[str, Any]
    reid_weights: Any
    per_class: bool
    device: str | None
    half: bool | None


def validate_user_values(
    config_name: str, target: str, user: dict[str, Any], base: dict[str, Any]
) -> None:
    """Raise ValueError for non-effective, unknown or silently-clamped user settings."""
    allowed = EFFECTIVE_PARAMS[target]
    hints = NON_EFFECTIVE_HINTS.get(target, {})
    for key in user:
        if key in hints:
            msg = (
                f"{config_name}: '{key}' has no effect on {target} in BoxMOT {BOXMOT_VERSION}; "
                f"{hints[key]}."
            )
            raise ValueError(msg)
        if key not in allowed:
            msg = (
                f"{config_name}: '{key}' is not an effective BoxMOT parameter for {target}. "
                f"Valid: {sorted(allowed)}"
            )
            raise ValueError(msg)
    if target != "sfsort":
        return
    merged = {**base, **user}
    high = float(merged.get("high_th", SFSORT_DEFAULT_HIGH_TH))
    for key, (lo, hi) in SFSORT_RANGES.items():
        if key not in user:
            continue
        lo_v = high if lo == "high_th" else float(lo)
        hi_v = high if hi == "high_th" else float(hi)
        if not lo_v <= float(user[key]) <= hi_v:
            msg = (
                f"{config_name}: {key}={user[key]} is outside SF-SORT's range [{lo_v}, {hi_v}]; "
                "BoxMOT would silently clamp it."
            )
            raise ValueError(msg)


def build_tracker_args(
    *,
    config_name: str,
    model: str,
    tuning: dict[str, Any],
    per_class: bool,
    reid_weights: Any,
    extra_kwargs: dict[str, Any],
    tracker_device: str | None,
    tracker_half: bool | None,
    default_reid_weight: str,
    config_names: dict[str, str],
) -> TrackerBuild:
    """Resolve a dnt config into `create_tracker` arguments (spec §2.3)."""
    base = boxmot_yaml_defaults(model)
    user = dict(tuning)
    for key, value in extra_kwargs.items():
        if key in user:
            raise ValueError(f"{config_name}: '{key}' is set both as a field and in extra_kwargs.")
        user[key] = value
    validate_user_values(config_name, model, user, base)
    if reid_weights is None and model in REID_TRACKERS:
        reid_weights = default_reid_weight
    if model not in REID_TRACKERS:
        reid_weights = None
    return TrackerBuild(
        model, {**base, **user}, reid_weights, per_class, tracker_device, tracker_half
    )
