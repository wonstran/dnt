"""``RefineConfig``: every refinement setting, with per-target defaults (spec 9)."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints
from urllib.parse import urlsplit

import yaml

TARGETS = ("person", "vehicle")
ENCODERS = ("dino", "reid", "none")
_ENDPOINT_KEYS = {"backend", "base_url", "model", "api_key_env", "api_key_file", "max_tokens", "extra_body"}
BACKENDS = ("none", "openai_compat", "anthropic")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FIXED_KEY_DICTS = {"ramps", "weights", "weights_occluded", "legacy_weights", "reclass_map"}
_RAMP_DICTS = {"ramps", "ramp", "reclass_ramp"}
_RAMP_FIELDS = {"orphan.ramp", "hints.reclass_ramp"}


def load_env_file(path) -> bool:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Variables that are already set are left alone, so the shell wins over the file.

    Parameters
    ----------
    path : str or Path
        The ``.env`` file; a missing file is ignored.

    Returns
    -------
    bool
        True if the file existed and was read.

    """
    path = Path(path)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


def _json_ok(value) -> bool:
    """Return True if ``value`` is plain JSON data (str keys; no NaN or infinity)."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return all(isinstance(k, str) for k in _keys(value))


def _keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


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
    min_crop_px: int = 40
    ramps: dict[str, list[float]] = field(
        default_factory=lambda: {
            "z_app": [2.0, 5.0],
            "silhouette": [0.25, 0.5],
            "jump": [0.15, 0.4],
            "cross": [0.0, 0.4],
        }
    )


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
    ramps: dict[str, list[float]] = field(
        default_factory=lambda: {
            "R": [0.3, 0.05],
            "J": [0.03, 0.005],
            "C": [0.6, 0.3],
            "T": [2.0, 10.0],
            "H": [1.0, 4.0],
            "inside": [0.5, 0.9],
            "F": [0.3, 0.7],
            "S": [0.5, 0.9],
            "K": [0.2, 0.6],
            "D": [0.5, 0.9],
        }
    )


@dataclass(kw_only=True)
class LinkConfig:
    """Stage 3 settings (spec 6.3)."""

    enabled: bool = True
    mode: str = "scored"
    accept_above: float = 0.62
    reject_below: float = 0.40
    max_gap: float = 1.0
    max_gap_static: float = 10.0
    max_gap_occluded: float = 8.0
    witness_iob: float = 0.5
    witness_min: float = 0.7
    max_heading_change: float = 120.0
    speed_factor: float = 1.5
    min_feasible_speed: float = 0.5
    occluded_score_cap: float = 0.60
    margin_min: float = 0.10
    ambiguous_cap: float = 0.60
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
    min_crop_px: int = 0


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
    """VLM verification settings (spec 7); used from Plan 3 on.

    Attributes
    ----------
    base_url : str or None
        The endpoint of either backend: an OpenAI-compatible server for ``openai_compat``, or a
        proxy or gateway for ``anthropic`` (None: the vendor's endpoint). An ``http://`` or
        ``https://`` URL with a host and a valid port, without whitespace, a user name,
        password, query string, or fragment.
    api_key_env : str or None
        The name of the environment variable that holds the key (None: ``OPENAI_API_KEY`` or
        ``ANTHROPIC_API_KEY``).
    api_key_file : str or None
        The path of a file whose content (stripped) is the key: one line of printable ASCII
        without whitespace; ``~`` is expanded. It takes
        precedence over ``api_key_env``, and ``TrackRefiner(vlm_api_key=...)`` over both. The
        config holds no key itself, since it is copied into the ledger header.

    """

    backend: str = "none"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    api_key_file: str | None = None
    json_mode: bool = True
    min_conf: float = 0.7
    votes: int = 1
    vote_temperature: float = 0.7
    max_calls: int = 500
    max_concurrency: int = 4
    timeout_s: float = 60.0
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    send_context_frames: bool = True
    cache_dir: str = "~/.cache/dnt/vlm"
    endpoints: dict[str, dict] = field(default_factory=dict)
    use: str | None = None

    def resolve(self) -> VLMConfig:
        """Return the settings with the endpoint named by ``use`` applied.

        Each entry of ``endpoints`` may set ``backend``, ``base_url``, ``model`` and
        ``api_key_env``, ``api_key_file``, ``max_tokens`` and ``extra_body``; any other ``vlm`` setting is shared by all endpoints.
        With no ``use`` the settings are returned as they are.

        Returns
        -------
        VLMConfig
            A copy with the chosen endpoint's fields filled in, and no ``endpoints``/``use``.

        """
        out = copy.deepcopy(self)
        if self.use is not None:
            for k, v in self.endpoints[self.use].items():
                setattr(out, k, v)
        out.endpoints, out.use = {}, None
        return out


@dataclass(kw_only=True)
class RefineConfig:
    """All refinement settings for one target (person or vehicle).

    Attributes
    ----------
    class_ids : list of int
        The target's class IDs (``[0]`` for person, ``[2, 5, 7]`` for vehicle). In this
        release only ``class_ids[0]`` is used: it becomes the class of every row read from a
        MOTChallenge file, which has no class column. Rows are not filtered by class: every
        row of the track file is screened, linked and orphan-checked as the target, so pass a
        file that holds the target's tracks only.

    """

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
        data = dict(copy.deepcopy(data) or {})
        cfg = cls.defaults(data.get("target", "person"))
        _overlay(cfg, data, "")
        cfg.hints.reclass_class_map = {
            int(k): str(v) for k, v in cfg.hints.reclass_class_map.items()
        }
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path) -> RefineConfig:
        """Load a config from a YAML file."""
        return cls.from_dict(read_yaml_mapping(path))

    def to_dict(self) -> dict:
        """Return the config as plain Python data."""
        return asdict(self)

    def to_yaml(self, path) -> None:
        """Write the config to a YAML file."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def validate(self) -> None:
        """Raise ValueError listing every rule of spec 9 the config breaks.

        An empty ``encoder.weights`` (``""`` from YAML) is first set to ``None``: the default
        weights.
        """
        if isinstance(self.encoder.weights, str) and not self.encoder.weights.strip():
            self.encoder.weights = None
        for name in ("api_key_env", "api_key_file", "base_url"):
            v = getattr(self.vlm, name)
            if isinstance(v, str) and not v.strip():
                setattr(self.vlm, name, None)  # "" from YAML: the default (variable, endpoint)
        p: list[str] = []
        if self.target not in TARGETS:
            p.append(f"target must be one of {TARGETS}")
        if self.encoder.kind not in ENCODERS:
            p.append(f"encoder.kind must be one of {ENCODERS}")
        vlm = self.vlm
        endpoints_ok = True
        for ename, ep in vlm.endpoints.items():
            bad = set(ep) - _ENDPOINT_KEYS
            if bad:
                p.append(f"vlm.endpoints.{ename}: unknown keys {sorted(bad)}")
                endpoints_ok = False
        if vlm.use is not None and vlm.use not in vlm.endpoints:
            p.append(f"vlm.use '{vlm.use}' is not in vlm.endpoints")
            endpoints_ok = False
        if endpoints_ok:
            vlm = vlm.resolve()
        if vlm.backend not in BACKENDS:
            p.append(f"vlm.backend must be one of {BACKENDS}")
        for name in ("switch", "link"):
            mcp = getattr(self, name).min_crop_px
            if isinstance(mcp, bool) or not isinstance(mcp, int) or mcp < 0:
                p.append(f"{name}.min_crop_px must be a whole number of pixels >= 0, not {mcp!r}")
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
        if vlm.backend not in ("none", "anthropic") and not vlm.model:
            p.append("vlm.model is required for this backend")
        vl = vlm

        def _num(x):
            return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)

        if not (isinstance(vl.votes, int) and not isinstance(vl.votes, bool) and vl.votes >= 1):
            p.append("vlm.votes must be an integer >= 1")
        if not (_num(vl.min_conf) and 0.0 <= vl.min_conf <= 1.0):
            p.append("vlm.min_conf must be a number in [0, 1]")
        if not (
            isinstance(vl.max_calls, int)
            and not isinstance(vl.max_calls, bool)
            and vl.max_calls >= 0
        ):
            p.append("vlm.max_calls must be an integer >= 0")
        if not (
            isinstance(vl.max_concurrency, int)
            and not isinstance(vl.max_concurrency, bool)
            and vl.max_concurrency >= 1
        ):
            p.append("vlm.max_concurrency must be an integer >= 1")
        if not isinstance(vl.extra_body, dict) or not _json_ok(vl.extra_body):
            p.append("vlm.extra_body must be a mapping of plain JSON values")
        mt = vl.max_tokens
        if mt is not None and not (isinstance(mt, int) and not isinstance(mt, bool) and mt >= 1):
            p.append("vlm.max_tokens must be a whole number >= 1 (or unset)")
        if not (_num(vl.timeout_s) and vl.timeout_s > 0):
            p.append("vlm.timeout_s must be a number > 0")
        if not (_num(vl.vote_temperature) and vl.vote_temperature >= 0):
            p.append("vlm.vote_temperature must be a number >= 0")
        if not (isinstance(vl.cache_dir, str) and vl.cache_dir.strip()):
            p.append("vlm.cache_dir must be a non-empty string")
        if vl.api_key_env is not None and not (
            isinstance(vl.api_key_env, str) and _ENV_NAME.fullmatch(vl.api_key_env)
        ):
            # never echo the value: a key pasted here would land in the message and the ledger
            p.append(
                "vlm.api_key_env must be the name of an environment variable (letters, digits "
                "and underscores, not starting with a digit), not the key itself"
            )
        p.extend(_url_problems(vl.base_url))
        if vl.api_key_file is not None and not isinstance(vl.api_key_file, str):
            p.append("vlm.api_key_file must be the path of a file that holds the key")
        elif isinstance(vl.api_key_file, str) and vl.api_key_file.strip().startswith("sk-"):
            p.append("vlm.api_key_file must be the path of a file that holds the key, not the key")
        ramps = {f"switch.ramps.{k}": v for k, v in self.switch.ramps.items()}
        ramps |= {f"screen.ramps.{k}": v for k, v in sc.ramps.items()}
        ramps |= {"orphan.ramp": self.orphan.ramp, "hints.reclass_ramp": self.hints.reclass_ramp}
        for name, v in ramps.items():
            try:
                lo, hi = v
            except (TypeError, ValueError):
                p.append(f"{name} must be a list of exactly 2 numbers")
                continue
            if lo == hi:
                p.append(f"{name} needs lo != hi")
        durations = {
            "switch.window": self.switch.window,
            "switch.delta": self.switch.delta,
            "switch.nms_seconds": self.switch.nms_seconds,
            "switch.min_side_seconds": self.switch.min_side_seconds,
            "link.max_gap": lc.max_gap,
            "link.max_gap_static": lc.max_gap_static,
            "link.max_gap_occluded": lc.max_gap_occluded,
            "link.static_seconds": lc.static_seconds,
            "link.speed_seconds": lc.speed_seconds,
        }
        if self.fill.max_gap is not None:
            durations["fill.max_gap"] = self.fill.max_gap
        for name, v in durations.items():
            if isinstance(v, int | float) and not v > 0:
                p.append(f"{name} must be a positive number of seconds, not {v}")
        if isinstance(lc.max_passes, int) and lc.max_passes < 1:
            p.append(f"link.max_passes must be at least 1, not {lc.max_passes}")
        if self.fps is not None and self.fps <= 0:
            p.append("fps must be positive")
        if self.frame_size is not None and (len(self.frame_size) != 2 or min(self.frame_size) <= 0):
            p.append("frame_size must be [width, height] with positive values")
        if not self.class_ids:
            p.append("class_ids must not be empty")
        if not set(self.hints.reclass_class_map.values()) <= set(self.reclass_map):
            p.append("hints.reclass_class_map values must be keys of reclass_map")
        if p:
            raise ValueError("invalid refine config: " + "; ".join(p))


def read_yaml_mapping(path) -> dict:
    """Return the top-level mapping of a config YAML file (empty if the file is empty).

    Nothing is checked beyond the shape; ``RefineConfig.from_dict`` checks the keys and values.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is not None and not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return dict(data or {})


def _url_problems(url) -> list[str]:
    """Return what is wrong with ``vlm.base_url``; the messages never echo the value.

    A URL can carry a key (in ``user:pass@`` or a query string), and the config is copied into
    the ledger header, so neither is allowed.
    """
    if url is None:
        return []
    if not (isinstance(url, str) and url.lower().startswith(("http://", "https://"))):
        return ["vlm.base_url must be an http:// or https:// URL"]
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        return ["vlm.base_url must not contain whitespace or control characters"]
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # raises ValueError for a port that is not a number in range
    except ValueError:
        return ["vlm.base_url is not a valid URL (check the host and the port)"]
    if not host:
        return ["vlm.base_url must name a host"]
    if "@" in parts.netloc:
        return [
            "vlm.base_url must not hold a user name or password (user:pass@); give the key "
            "with vlm.api_key_file, vlm.api_key_env, or TrackRefiner(vlm_api_key=...) instead"
        ]
    if parts.query or parts.fragment or url.rstrip().endswith(("?", "#")):
        return [
            "vlm.base_url must not have a query string or fragment; give a key with "
            "vlm.api_key_file, vlm.api_key_env, or TrackRefiner(vlm_api_key=...)"
        ]
    return []


def _check_type(path: str, value, hint) -> None:
    """Check value against type hint; raise ValueError if mismatch."""
    if hint is type(None):
        if value is not None:
            raise ValueError(f"config key '{path}' must be null, not {type(value).__name__}")
        return

    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        if hint is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"config key '{path}' must be an int, not {type(value).__name__}")
        elif hint is float:
            if isinstance(value, bool) or (not isinstance(value, (int, float))):
                msg = f"config key '{path}' must be a number, not {type(value).__name__}"
                raise ValueError(msg)
            if not math.isfinite(value):
                raise ValueError(f"config key '{path}' must be finite")
        elif hint is bool:
            if not isinstance(value, bool):
                raise ValueError(f"config key '{path}' must be a bool, not {type(value).__name__}")
        elif hint is str and not isinstance(value, str):
            raise ValueError(f"config key '{path}' must be a string, not {type(value).__name__}")
        return

    if origin is type(None) or (hasattr(hint, "__origin__") and hint.__origin__ is type(None)):
        return

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"config key '{path}' must be a list, not {type(value).__name__}")
        if args:
            for i, item in enumerate(value):
                _check_type(f"{path}[{i}]", item, args[0])
        return

    if origin is dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"config key '{path}' must be a mapping, not {type(value).__name__}")
        if args and len(args) >= 2:
            for k, v in value.items():
                _check_type(f"{path}.{k}", v, args[1])
        return

    if hasattr(hint, "__args__"):
        hint_args = hint.__args__
        if type(None) in hint_args:
            if value is None:
                return
            non_none_types = [t for t in hint_args if t is not type(None)]
            if len(non_none_types) == 1:
                _check_type(path, value, non_none_types[0])
                return
    raise ValueError(f"config key '{path}' has unknown type hint {hint}")


def _check_ramp_shape(path: str, value) -> None:
    """Check that a ramp field is a list of exactly 2 numbers."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config key '{path}' must be a list of exactly 2 numbers")
    if len(value) != 2:
        raise ValueError(f"config key '{path}' must be a list of exactly 2 numbers")
    ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
    if not ok:
        raise ValueError(f"config key '{path}' must be a list of exactly 2 numbers")


def _overlay(obj, data: Mapping, path: str) -> None:
    names = {f.name for f in fields(obj)}
    hints = get_type_hints(type(obj))

    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key '{path}{key}'")
        current = getattr(obj, key)
        full_path = f"{path}{key}"

        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{full_path}' must be a mapping")
            _overlay(current, value, f"{full_path}.")
        elif key in _FIXED_KEY_DICTS and isinstance(current, dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"config key '{full_path}' must be a mapping")
            for sub in value:
                if sub not in current:
                    raise ValueError(f"unknown config key '{full_path}.{sub}'")

            if key in _RAMP_DICTS:
                processed = {}
                for k, v in value.items():
                    if isinstance(v, (list, tuple)):
                        if len(v) != 2:
                            msg = f"config key '{full_path}.{k}' must be a list of "
                            raise ValueError(msg + "exactly 2 numbers")
                        ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
                        if not ok:
                            msg = f"config key '{full_path}.{k}' must be a list of "
                            raise ValueError(msg + "exactly 2 numbers")
                        processed[k] = list(v)
                    else:
                        msg = f"config key '{full_path}.{k}' must be a list of exactly "
                        raise ValueError(msg + "2 numbers")
                setattr(obj, key, {**current, **processed})
            else:
                processed = {}
                if key in hints:
                    dict_hint = hints[key]
                    dict_args = get_args(dict_hint)
                    value_hint = dict_args[1] if len(dict_args) >= 2 else None
                else:
                    value_hint = None

                for k, v in value.items():
                    if value_hint:
                        _check_type(f"{full_path}.{k}", v, value_hint)
                    if isinstance(v, (list, tuple)):
                        processed[k] = list(v)
                    else:
                        processed[k] = v
                setattr(obj, key, {**current, **processed})
        else:
            if key in hints:
                hint = hints[key]
                _check_type(full_path, value, hint)

            if full_path in _RAMP_FIELDS:
                _check_ramp_shape(full_path, value)

            setattr(obj, key, value)
