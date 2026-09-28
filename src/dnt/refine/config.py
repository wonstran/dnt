"""``RefineConfig``: every refinement setting, with per-target defaults (spec 9)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

TARGETS = ("person", "vehicle")
ENCODERS = ("dino", "reid", "none")
BACKENDS = ("none", "openai_compat", "anthropic")
_FIXED_KEY_DICTS = {"ramps", "weights", "weights_occluded", "legacy_weights", "reclass_map"}


def to_frames(seconds: float, fps: float) -> int:
    """Convert a duration in seconds to a whole number of frames (at least 1)."""
    return max(1, round(seconds * fps))


@dataclass(kw_only=True)
class ContextConfig:
    """Context file settings (spec 2.5)."""

    format: str = "auto"
    vehicle_classes: list[int] = field(default_factory=lambda: [2, 5, 7])
    twowheeler_classes: list[int] = field(default_factory=lambda: [1, 3])


@dataclass(kw_only=True)
class HintsConfig:
    """ReClass hint settings (spec 6.2)."""

    reclass_class_map: dict[int, str] = field(
        default_factory=lambda: {1: "cyclist", 3: "motorcycle", 36: "scooter"}
    )
    reclass_ramp: list[float] = field(default_factory=lambda: [0.75, 0.9])
    subtype_min: float = 1.0

    def __post_init__(self):
        """Normalize class keys to int (YAML and JSON may deliver strings)."""
        self.reclass_class_map = {int(k): str(v) for k, v in self.reclass_class_map.items()}


@dataclass(kw_only=True)
class MotionConfig:
    """Shared motion model settings (spec 5.1, 5.2)."""

    height_window: int = 15
    process_var: float = 10.0
    meas_var_pos: float = 25.0
    meas_var_size: float = 16.0
    moving_min: float = 0.3


@dataclass(kw_only=True)
class EncoderConfig:
    """Appearance encoder settings (spec 5.3, 5.5)."""

    kind: str = "dino"
    model: str = "facebook/dinov2-small"
    weights: str | None = None
    device: str = "auto"
    sample_every: int = 5
    occlusion_iou: float = 0.3
    batch_size: int = 64


@dataclass(kw_only=True)
class SwitchConfig:
    """Stage 1 settings (spec 6.1)."""

    enabled: bool = True
    accept_above: float = 0.90
    reject_below: float = 0.50
    window: float = 1.0
    min_side_seconds: float = 0.5
    nis_hi: float = 18.47
    w_app: float = 0.65
    w_mot: float = 0.35
    motion_only_cap: float = 0.70
    class_change_gate: bool = True
    size_gate: float = 1.5
    delta: float = 0.5
    nms_seconds: float = 1.0
    contact_iou: float = 0.1
    mad_floor: float = 0.02
    bimodal_purity: float = 0.9
    bimodal_silhouette_min: float = 0.25
    swap_boost: float = 0.2
    ramps: dict[str, list[float]] = field(default_factory=lambda: {
        "z_app": [2.0, 5.0], "silhouette": [0.25, 0.5], "jump": [0.15, 0.4], "cross": [0.0, 0.4],
    })


@dataclass(kw_only=True)
class ScreenConfig:
    """Stage 2 settings (spec 6.2)."""

    enabled: bool = True
    accept_above: float = 0.85
    reject_below: float = 0.40
    static_score_cap: float = 0.80
    vehicle_static_score_cap: float = 0.60
    mixed_score_cap: float = 0.75
    segment_at: float = 0.30
    in_vehicle_iob: float = 0.8
    move_together: float = 0.3
    rider_speed: float = 1.8
    twowheeler_iou: float = 0.3
    duplicate_iob: float = 0.7
    duplicate_min_frames: int = 10
    hotspot_radius: float = 0.5
    persistence_iou: float = 0.5
    ramps: dict[str, list[float]] = field(default_factory=lambda: {
        "R": [0.3, 0.05], "J": [0.03, 0.005], "C": [0.6, 0.3], "T": [2.0, 10.0],
        "H": [1.0, 4.0], "inside": [0.5, 0.9], "F": [0.3, 0.7], "S": [0.5, 0.9],
        "K": [0.2, 0.6], "D": [0.5, 0.9],
    })


@dataclass(kw_only=True)
class LinkConfig:
    """Stage 3 settings (spec 6.3)."""

    enabled: bool = True
    mode: str = "scored"
    accept_above: float = 0.80
    reject_below: float = 0.40
    max_gap: float = 1.0
    max_gap_static: float = 10.0
    max_gap_occluded: float = 8.0
    witness_iob: float = 0.5
    witness_min: float = 0.7
    max_heading_change: float = 120.0
    speed_factor: float = 1.5
    min_feasible_speed: float = 0.5
    occluded_score_cap: float = 0.75
    margin_min: float = 0.10
    ambiguous_cap: float = 0.75
    max_passes: int = 3
    weights_occluded: dict[str, float] = field(
        default_factory=lambda: {"mot": 0.25, "app": 0.60, "gap": 0.15}
    )
    class_groups: list[list[int]] = field(default_factory=list)
    size_ratio_max: float = 2.0
    dist_mult: float = 2.5
    dist_growth: float = 0.03
    iou_min: float = 0.05
    vel_frames: int = 5
    legacy_weights: dict[str, float] = field(
        default_factory=lambda: {"d": 1.0, "iou": 1.0, "s": 0.3}
    )
    legacy_cost_hi: float = 3.0
    weights: dict[str, float] = field(
        default_factory=lambda: {"mot": 0.45, "app": 0.40, "gap": 0.15}
    )
    static_speed: float = 0.2
    static_seconds: float = 0.5
    static_radius: float = 0.5
    overlap_frames: int = 2
    overlap_iou: float = 0.5
    k_embed: int = 5
    border_margin: float = 0.5
    speed_seconds: float = 1.0
    heading_min_speed: float = 0.2
    n_alternatives: int = 2


@dataclass(kw_only=True)
class OrphanConfig:
    """Orphan pass settings (spec 6.2)."""

    enabled: bool = True
    min_seconds: float = 0.5
    accept_above: float = 0.70
    reject_below: float = 0.30
    ramp: list[float] = field(default_factory=lambda: [0.5, 0.1])


@dataclass(kw_only=True)
class FillConfig:
    """Stage 4 settings (spec 6.4)."""

    enabled: bool = True
    max_gap: float | None = None
    smooth_existing: bool = False


@dataclass(kw_only=True)
class VLMConfig:
    """VLM verification settings (spec 7); used from Plan 3 on."""

    backend: str = "none"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    json_mode: bool = True
    min_conf: float = 0.7
    votes: int = 1
    vote_temperature: float = 0.7
    max_calls: int = 500
    max_concurrency: int = 4
    timeout_s: float = 60.0
    send_context_frames: bool = True
    cache_dir: str = "~/.cache/dnt/vlm"


@dataclass(kw_only=True)
class RefineConfig:
    """All refinement settings for one target (person or vehicle)."""

    target: str = "person"
    class_ids: list[int] = field(default_factory=lambda: [0])
    fps: float | None = None
    reclass_map: dict[str, int] = field(
        default_factory=lambda: {"cyclist": 1, "motorcycle": 3, "scooter": 36}
    )
    frame_size: list[int] | None = None
    context: ContextConfig = field(default_factory=ContextConfig)
    hints: HintsConfig = field(default_factory=HintsConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    orphan: OrphanConfig = field(default_factory=OrphanConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    @classmethod
    def defaults(cls, target: str = "person") -> RefineConfig:
        """Return the defaults for ``target`` (``person`` or ``vehicle``)."""
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
        cfg = cls(target=target)
        if target == "vehicle":
            cfg.class_ids = [2, 5, 7]
            cfg.link.class_groups = [[2, 7]]
        return cfg

    @classmethod
    def from_dict(cls, data: Mapping | None) -> RefineConfig:
        """Overlay ``data`` on the target's defaults; unknown keys raise ValueError."""
        data = dict(data or {})
        cfg = cls.defaults(data.get("target", "person"))
        _overlay(cfg, data, "")
        cfg.hints.reclass_class_map = {int(k): str(v)
                                       for k, v in cfg.hints.reclass_class_map.items()}
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path) -> RefineConfig:
        """Load a config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is not None and not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Return the config as plain Python data."""
        return asdict(self)

    def to_yaml(self, path) -> None:
        """Write the config to a YAML file."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def validate(self) -> None:
        """Raise ValueError listing every rule of spec 9 the config breaks."""
        p: list[str] = []
        if self.target not in TARGETS:
            p.append(f"target must be one of {TARGETS}")
        if self.encoder.kind not in ENCODERS:
            p.append(f"encoder.kind must be one of {ENCODERS}")
        if self.vlm.backend not in BACKENDS:
            p.append(f"vlm.backend must be one of {BACKENDS}")
        if self.link.mode not in ("scored", "legacy"):
            p.append("link.mode must be 'scored' or 'legacy'")
        if self.context.format not in ("auto", "tracks", "dets"):
            p.append("context.format must be 'auto', 'tracks' or 'dets'")
        for name in ("switch", "screen", "link", "orphan"):
            s = getattr(self, name)
            if not 0.0 <= s.reject_below < s.accept_above <= 1.0:
                p.append(f"{name}: need 0 <= reject_below < accept_above <= 1")
        sc, lc = self.screen, self.link
        if not sc.static_score_cap < sc.accept_above:
            p.append("screen.static_score_cap must be below screen.accept_above")
        if not sc.vehicle_static_score_cap < sc.accept_above:
            p.append("screen.vehicle_static_score_cap must be below screen.accept_above")
        if not sc.mixed_score_cap < sc.accept_above:
            p.append("screen.mixed_score_cap must be below screen.accept_above")
        if not sc.segment_at < self.switch.reject_below:
            p.append("screen.segment_at must be below switch.reject_below")
        for wname in ("weights", "weights_occluded"):
            if abs(sum(getattr(lc, wname).values()) - 1.0) > 1e-6:
                p.append(f"link.{wname} must sum to 1")
        if not lc.occluded_score_cap < lc.accept_above:
            p.append("link.occluded_score_cap must be below link.accept_above")
        if not lc.ambiguous_cap < lc.accept_above:
            p.append("link.ambiguous_cap must be below link.accept_above")
        if not lc.max_gap < lc.max_gap_occluded:
            p.append("link.max_gap must be below link.max_gap_occluded")
        seen: set[int] = set()
        for group in lc.class_groups:
            if seen & set(group):
                p.append("link.class_groups: a class appears in more than one group")
            seen |= set(group)
        if self.encoder.kind == "reid" and self.target == "vehicle" and not self.encoder.weights:
            p.append("encoder.weights is required for reid with the vehicle target")
        if self.vlm.backend not in ("none", "anthropic") and not self.vlm.model:
            p.append("vlm.model is required for this backend")
        ramps = {f"switch.ramps.{k}": v for k, v in self.switch.ramps.items()}
        ramps |= {f"screen.ramps.{k}": v for k, v in sc.ramps.items()}
        ramps |= {"orphan.ramp": self.orphan.ramp, "hints.reclass_ramp": self.hints.reclass_ramp}
        for name, (lo, hi) in ramps.items():
            if lo == hi:
                p.append(f"{name} needs lo != hi")
        if self.fps is not None and self.fps <= 0:
            p.append("fps must be positive")
        if self.frame_size is not None and (len(self.frame_size) != 2
                                            or min(self.frame_size) <= 0):
            p.append("frame_size must be [width, height] with positive values")
        if not self.class_ids:
            p.append("class_ids must not be empty")
        if not set(self.hints.reclass_class_map.values()) <= set(self.reclass_map):
            p.append("hints.reclass_class_map values must be keys of reclass_map")
        if p:
            raise ValueError("invalid refine config: " + "; ".join(p))


def _overlay(obj, data: Mapping, path: str) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{path}{key}' must be a mapping")
            _overlay(current, value, f"{path}{key}.")
        elif key in _FIXED_KEY_DICTS and isinstance(current, dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{path}{key}' must be a mapping")
            for sub in value:
                if sub not in current:
                    raise ValueError(f"unknown config key '{path}{key}.{sub}'")
            setattr(obj, key, {**current, **{k: (list(v) if isinstance(v, list | tuple) else v)
                                             for k, v in value.items()}})
        else:
            setattr(obj, key, value)
