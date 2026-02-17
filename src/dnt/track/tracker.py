"""BoxMOT tracking utilities and unified tracker interface.

This module provides the Tracker class for BoxMOT tracking and
post-processing utilities for infilling, clustering, and filtering tracks.
"""

import contextlib
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, StrEnum
from inspect import signature
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ensure local imports work if this is run as a script
sys.path.append(os.path.dirname(__file__))


def _configure_boxmot_logging(boxmot_verbose: bool) -> None:
    """Configure BoxMOT logger verbosity.

    If `boxmot_verbose` is False, set BoxMOT logging level to ERROR.
    """
    if boxmot_verbose:
        return

    try:
        from boxmot.utils import logger as boxmot_logger  # type: ignore

        boxmot_logger.remove()
        boxmot_logger.add(sys.stderr, level="ERROR")
    except Exception:
        # Do not block tracking if logger configuration is unavailable.
        pass


def _patch_boxmot_requirements_installer() -> None:
    """Patch BoxMOT dependency installer to use pip when uv is unavailable."""
    try:
        from boxmot.utils.checks import RequirementsChecker  # type: ignore
    except Exception:
        return

    install_fn_name = "_install_packages"
    original = getattr(RequirementsChecker, install_fn_name, None)
    if original is None:
        return
    if getattr(original, "__name__", "") == "_install_packages_with_pip_fallback":
        return

    def _install_packages_with_pip_fallback(self, packages, extra_args=None):  # type: ignore[no-untyped-def]
        cmd = ["uv", "pip", "install", "--no-cache-dir"]
        if extra_args:
            cmd += list(extra_args)
        cmd += list(packages)
        try:
            subprocess.check_call(cmd)
            return
        except FileNotFoundError:
            pip_cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages]
            if extra_args:
                pip_cmd[4:4] = list(extra_args)
            subprocess.check_call(pip_cmd)

    setattr(RequirementsChecker, install_fn_name, _install_packages_with_pip_fallback)


class MOTModels(StrEnum):
    """Supported tracker backends exposed by BoxMOT.

    Attributes
    ----------
    BOTSORT : str
        BoT-SORT tracker name used by BoxMOT.
        Good default when you want motion + appearance matching.
    BOOSTTRACK : str
        BoostTrack tracker name used by BoxMOT.
        Usually improves association under difficult motion/crowding.
    BYTE_TRACK : str
        ByteTrack tracker name used by BoxMOT.
        Faster and simpler; does not require ReID weights.
    OCSORT : str
        OCSORT tracker name used by BoxMOT.
        Motion-centric tracker; useful when appearance features are unreliable.
    STRONGSORT : str
        StrongSORT tracker name used by BoxMOT.
        Appearance-heavy tracker; typically more robust to long occlusions.
    DEEPOCSORT : str
        DeepOCSORT tracker name used by BoxMOT.
        OCSORT variant enhanced with appearance features.
    HYBRIDSORT : str
        HybridSORT tracker name used by BoxMOT.
        Hybrid strategy between motion and appearance matching.
    SFSORT : str
        SFSort tracker name used by BoxMOT.
        Lightweight motion-centric tracking for real-time pipelines.

    """

    BOTSORT = "botsort"
    BOOSTTRACK = "boosttrack"
    BYTE_TRACK = "bytetrack"
    OCSORT = "ocsort"
    STRONGSORT = "strongsort"
    DEEPOCSORT = "deepocsort"
    HYBRIDSORT = "hybridsort"
    SFSORT = "sfsort"


class ReIDWeights(StrEnum):
    """Built-in BoxMOT ReID weight file names.

    Use these enum values for `reid_weights` in tracker parameter dataclasses.
    """

    RESNET50_MARKET1501 = "resnet50_market1501.pt"
    RESNET50_DUKEMTMCREID = "resnet50_dukemtmcreid.pt"
    RESNET50_MSMT17 = "resnet50_msmt17.pt"
    RESNET50_FC512_MARKET1501 = "resnet50_fc512_market1501.pt"
    RESNET50_FC512_DUKEMTMCREID = "resnet50_fc512_dukemtmcreid.pt"
    RESNET50_FC512_MSMT17 = "resnet50_fc512_msmt17.pt"
    MLFN_MARKET1501 = "mlfn_market1501.pt"
    MLFN_DUKEMTMCREID = "mlfn_dukemtmcreid.pt"
    MLFN_MSMT17 = "mlfn_msmt17.pt"
    HACNN_MARKET1501 = "hacnn_market1501.pt"
    HACNN_DUKEMTMCREID = "hacnn_dukemtmcreid.pt"
    HACNN_MSMT17 = "hacnn_msmt17.pt"
    MOBILENETV2_X1_0_MARKET1501 = "mobilenetv2_x1_0_market1501.pt"
    MOBILENETV2_X1_0_DUKEMTMCREID = "mobilenetv2_x1_0_dukemtmcreid.pt"
    MOBILENETV2_X1_0_MSMT17 = "mobilenetv2_x1_0_msmt17.pt"
    MOBILENETV2_X1_4_MARKET1501 = "mobilenetv2_x1_4_market1501.pt"
    MOBILENETV2_X1_4_DUKEMTMCREID = "mobilenetv2_x1_4_dukemtmcreid.pt"
    MOBILENETV2_X1_4_MSMT17 = "mobilenetv2_x1_4_msmt17.pt"
    OSNET_X1_0_MARKET1501 = "osnet_x1_0_market1501.pt"
    OSNET_X1_0_DUKEMTMCREID = "osnet_x1_0_dukemtmcreid.pt"
    OSNET_X1_0_MSMT17 = "osnet_x1_0_msmt17.pt"
    OSNET_X0_75_MARKET1501 = "osnet_x0_75_market1501.pt"
    OSNET_X0_75_DUKEMTMCREID = "osnet_x0_75_dukemtmcreid.pt"
    OSNET_X0_75_MSMT17 = "osnet_x0_75_msmt17.pt"
    OSNET_X0_5_MARKET1501 = "osnet_x0_5_market1501.pt"
    OSNET_X0_5_DUKEMTMCREID = "osnet_x0_5_dukemtmcreid.pt"
    OSNET_X0_5_MSMT17 = "osnet_x0_5_msmt17.pt"
    OSNET_X0_25_MARKET1501 = "osnet_x0_25_market1501.pt"
    OSNET_X0_25_DUKEMTMCREID = "osnet_x0_25_dukemtmcreid.pt"
    OSNET_X0_25_MSMT17 = "osnet_x0_25_msmt17.pt"
    OSNET_IBN_X1_0_MSMT17 = "osnet_ibn_x1_0_msmt17.pt"
    OSNET_AIN_X1_0_MSMT17 = "osnet_ain_x1_0_msmt17.pt"
    LMBN_N_DUKE = "lmbn_n_duke.pt"
    LMBN_N_MARKET = "lmbn_n_market.pt"
    LMBN_N_CUHK03_D = "lmbn_n_cuhk03_d.pt"
    CLIP_MARKET1501 = "clip_market1501.pt"
    CLIP_DUKE = "clip_duke.pt"
    CLIP_VERI = "clip_veri.pt"
    CLIP_VEHICLEID = "clip_vehicleid.pt"


@dataclass
class MOTBaseConfig:
    """Common configuration fields for BoxMOT tracker creation.

    Attributes
    ----------
    model : MOTModels
        BoxMOT tracker backend for this parameter bundle
        (default: `MOTModels.BOTSORT`).
    per_class : bool
        Whether to run tracking independently per class (default: `False`).
        Options: `True`, `False`.
        Setting `True` reduces cross-class ID switches but can create more tracks.
    extra_kwargs : dict[str, Any]
        Additional kwargs merged into tracker construction arguments
        (default: `{}`).
        Use this for BoxMOT arguments not explicitly represented in dataclasses.

    """

    model: MOTModels = MOTModels.BOTSORT
    per_class: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Convert dataclass fields to keyword arguments for BoxMOT tracker creation."""
        kwargs = asdict(self)
        kwargs.pop("model", None)
        kwargs.pop("extra_kwargs", None)
        kwargs.update(self.extra_kwargs)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        """Return dataclass values as a serializable dictionary."""
        return self._yaml_safe(asdict(self))

    @staticmethod
    def _yaml_safe(value: Any) -> Any:
        """Recursively convert Enum values into YAML-serializable primitives."""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: MOTBaseConfig._yaml_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [MOTBaseConfig._yaml_safe(v) for v in value]
        if isinstance(value, tuple):
            return tuple(MOTBaseConfig._yaml_safe(v) for v in value)
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MOTBaseConfig":
        """Build a parameter object from a dictionary.

        Unknown keys are stored in `extra_kwargs`.
        """
        valid_fields = {f.name for f in fields(cls)}
        known = {k: v for k, v in data.items() if k in valid_fields}
        unknown = {k: v for k, v in data.items() if k not in valid_fields}

        if "model" in known and not isinstance(known["model"], MOTModels):
            known["model"] = MOTModels(str(known["model"]))

        params = cls(**known)
        if unknown:
            params.extra_kwargs.update(unknown)
        return params

    def export_yaml(self, yaml_file: str) -> None:
        """Export parameters to a YAML file."""
        out_path = Path(yaml_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def import_yaml(cls, yaml_file: str) -> "MOTBaseConfig":
        """Import parameters from a YAML file."""
        with Path(yaml_file).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            msg = f"Invalid YAML content in {yaml_file}: expected a mapping."
            raise ValueError(msg)
        return cls.from_dict(data)


@dataclass
class BoTSORTConfig(MOTBaseConfig):
    """BoTSORT-specific parameters.

    Attributes
    ----------
    reid_weights : ReIDWeights | str | None
        Optional ReID weights path (default: `"osnet_x1_0_msmt17.pt"`).
        Options: `None`, file name (auto-resolved), or absolute path.
        Built-in downloadable options:
        `'resnet50_market1501.pt'`, `'resnet50_dukemtmcreid.pt'`,
        `'resnet50_msmt17.pt'`, `'resnet50_fc512_market1501.pt'`,
        `'resnet50_fc512_dukemtmcreid.pt'`, `'resnet50_fc512_msmt17.pt'`,
        `'mlfn_market1501.pt'`, `'mlfn_dukemtmcreid.pt'`, `'mlfn_msmt17.pt'`,
        `'hacnn_market1501.pt'`, `'hacnn_dukemtmcreid.pt'`,
        `'hacnn_msmt17.pt'`, `'mobilenetv2_x1_0_market1501.pt'`,
        `'mobilenetv2_x1_0_dukemtmcreid.pt'`,
        `'mobilenetv2_x1_0_msmt17.pt'`, `'mobilenetv2_x1_4_market1501.pt'`,
        `'mobilenetv2_x1_4_dukemtmcreid.pt'`,
        `'mobilenetv2_x1_4_msmt17.pt'`, `'osnet_x1_0_market1501.pt'`,
        `'osnet_x1_0_dukemtmcreid.pt'`, `'osnet_x1_0_msmt17.pt'`,
        `'osnet_x0_75_market1501.pt'`, `'osnet_x0_75_dukemtmcreid.pt'`,
        `'osnet_x0_75_msmt17.pt'`, `'osnet_x0_5_market1501.pt'`,
        `'osnet_x0_5_dukemtmcreid.pt'`, `'osnet_x0_5_msmt17.pt'`,
        `'osnet_x0_25_market1501.pt'`, `'osnet_x0_25_dukemtmcreid.pt'`,
        `'osnet_x0_25_msmt17.pt'`, `'osnet_ibn_x1_0_msmt17.pt'`,
        `'osnet_ain_x1_0_msmt17.pt'`, `'lmbn_n_duke.pt'`,
        `'lmbn_n_market.pt'`, `'lmbn_n_cuhk03_d.pt'`,
        `'clip_market1501.pt'`, `'clip_duke.pt'`, `'clip_veri.pt'`,
        `'clip_vehicleid.pt'`.
        Suggestions: use `"osnet_x1_0_msmt17.pt"` for pedestrians or
        `"clip_vehicleid.pt"` / `"clip_veri.pt"` for vehicles.
    track_high_thresh : float
        High score threshold for first association (default: `0.5`).
        Increasing this is stricter and may reduce false matches but miss tracks.
    track_low_thresh : float
        Lower score threshold for second association (default: `0.1`).
        Increasing this keeps fewer low-confidence detections.
    new_track_thresh : float
        Threshold to initialize new tracks (default: `0.6`).
        Increasing this creates fewer new tracks and can reduce false positives.
    match_thresh : float
        Matching threshold for association (default: `0.8`).
        Increasing this makes association more permissive.
    track_buffer : int
        Number of frames to keep lost tracks (default: `30`).
        Increasing this preserves IDs longer through occlusion, but may cause
        stale tracks to survive longer.
    with_reid : bool
        Whether to enable ReID-assisted association (default: `True`).
        Options: `True`, `False`.
        Disabling this speeds up tracking but may increase ID switches.
    proximity_thresh : float
        Proximity threshold for ReID matching (default: `0.5`).
        Increasing this requires stronger geometric overlap before ReID is used.
    appearance_thresh : float
        Appearance similarity threshold for ReID matching (default: `0.25`).
        Increasing this requires closer appearance match and is more conservative.

    """

    model: MOTModels = MOTModels.BOTSORT
    reid_weights: ReIDWeights | str | None = ReIDWeights.OSNET_X1_0_MSMT17
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30
    with_reid: bool = True
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25


@dataclass
class BoostTrackConfig(MOTBaseConfig):
    """BoostTrack-specific parameters.

    Attributes
    ----------
    reid_weights : ReIDWeights | str | None
        Optional ReID weights path (default: `"osnet_x1_0_msmt17.pt"`).
        Options: same built-in `.pt` names listed in `BoTSORTParams.reid_weights`.
    det_thresh : float
        Detection confidence threshold (default: `0.3`).
        Increasing this keeps only higher-confidence detections.
    max_age : int
        Maximum age of unmatched tracks (default: `30`).
        Increasing this keeps tracks alive longer when unmatched.
    min_hits : int
        Minimum hits before track confirmation (default: `3`).
        Increasing this delays confirmation and reduces short noisy tracks.
    iou_threshold : float
        IoU threshold for association (default: `0.3`).
        Increasing this demands tighter overlap to match detections.
    asso_func : str
        Association function name (default: `"iou"`).
        Typical options: `"iou"`, `"giou"`, `"diou"`, `"ciou"`, `"centroid"`.

    """

    model: MOTModels = MOTModels.BOOSTTRACK
    reid_weights: ReIDWeights | str | None = ReIDWeights.OSNET_X1_0_MSMT17
    det_thresh: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    asso_func: str = "iou"


@dataclass
class ByteTrackConfig(MOTBaseConfig):
    """ByteTrack-specific parameters.

    Attributes
    ----------
    track_thresh : float
        Detection confidence threshold (default: `0.5`).
        Increasing this filters more weak detections.
    match_thresh : float
        Threshold for matching detections to tracks (default: `0.8`).
        Increasing this generally allows looser matching.
    track_buffer : int
        Number of frames to keep lost tracks (default: `30`).
        Increasing this keeps unmatched tracks longer.
    frame_rate : int
        Source video frame rate used by the tracker (default: `30`).
        Set this close to real FPS for best temporal behavior.

    """

    model: MOTModels = MOTModels.BYTE_TRACK
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


@dataclass
class OCSORTConfig(MOTBaseConfig):
    """OCSORT-specific parameters.

    Attributes
    ----------
    det_thresh : float
        Detection confidence threshold (default: `0.3`).
        Increasing this reduces low-confidence detections.
    max_age : int
        Maximum age of unmatched tracks (default: `30`).
        Increasing this keeps tracks alive through longer gaps.
    min_hits : int
        Minimum hits before track confirmation (default: `3`).
        Increasing this reduces short-lived false tracks.
    iou_threshold : float
        IoU threshold for association (default: `0.3`).
        Increasing this makes matching stricter.
    asso_func : str
        Association function name (default: `"iou"`).
        Typical options: `"iou"`, `"giou"`, `"diou"`, `"ciou"`, `"centroid"`.
    delta_t : int
        Time gap used by motion compensation (default: `3`).
        Increasing this smooths longer motion history, but may lag quick turns.
    inertia : float
        Motion inertia weight (default: `0.2`).
        Increasing this trusts previous velocity more.

    """

    model: MOTModels = MOTModels.OCSORT
    det_thresh: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    asso_func: str = "iou"
    delta_t: int = 3
    inertia: float = 0.2


@dataclass
class StrongSORTConfig(MOTBaseConfig):
    """StrongSORT-specific parameters.

    Attributes
    ----------
    reid_weights : ReIDWeights | str | None
        Optional ReID weights path (default: `"osnet_x1_0_msmt17.pt"`).
        Options: same built-in `.pt` names listed in `BoTSORTParams.reid_weights`.
    max_dist : float
        Maximum cosine distance for appearance matching (default: `0.2`).
        Increasing this allows less similar appearance matches.
    max_iou_dist : float
        Maximum IoU distance for geometric matching (default: `0.7`).
        Increasing this allows looser geometric matches.
    max_age : int
        Maximum age of unmatched tracks (default: `70`).
        Increasing this retains tracks through longer occlusions.
    n_init : int
        Minimum hits before track confirmation (default: `3`).
        Increasing this delays confirmation and reduces unstable IDs.
    nn_budget : int
        Maximum size of appearance feature gallery (default: `100`).
        Increasing this improves long-term matching memory at higher memory cost.
    ema_alpha : float
        EMA factor for appearance embeddings (default: `0.9`).
        Increasing this smooths features more and reduces noise.
    mc_lambda : float
        Motion compensation blending factor (default: `0.995`).
        Increasing this gives more weight to motion compensation.

    """

    model: MOTModels = MOTModels.STRONGSORT
    reid_weights: ReIDWeights | str | None = ReIDWeights.OSNET_X1_0_MSMT17
    max_dist: float = 0.2
    max_iou_dist: float = 0.7
    max_age: int = 70
    n_init: int = 3
    nn_budget: int = 100
    ema_alpha: float = 0.9
    mc_lambda: float = 0.995


@dataclass
class DeepOCSORTConfig(MOTBaseConfig):
    """DeepOCSORT-specific parameters.

    Attributes
    ----------
    reid_weights : ReIDWeights | str | None
        Optional ReID weights path (default: `"osnet_x1_0_msmt17.pt"`).
        Options: same built-in `.pt` names listed in `BoTSORTParams.reid_weights`.
    det_thresh : float
        Detection confidence threshold (default: `0.3`).
        Increasing this keeps fewer low-confidence detections.
    max_age : int
        Maximum age of unmatched tracks (default: `30`).
        Increasing this keeps unmatched tracks alive longer.
    min_hits : int
        Minimum hits before track confirmation (default: `3`).
        Increasing this delays track confirmation.
    iou_threshold : float
        IoU threshold for association (default: `0.3`).
        Increasing this requires tighter overlap.
    asso_func : str
        Association function name (default: `"iou"`).
        Typical options: `"iou"`, `"giou"`, `"diou"`, `"ciou"`, `"centroid"`.
    delta_t : int
        Time gap used by motion compensation (default: `3`).
        Increasing this smooths over longer temporal windows.
    inertia : float
        Motion inertia weight (default: `0.2`).
        Increasing this emphasizes velocity continuity.

    """

    model: MOTModels = MOTModels.DEEPOCSORT
    reid_weights: ReIDWeights | str | None = ReIDWeights.OSNET_X1_0_MSMT17
    det_thresh: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    asso_func: str = "iou"
    delta_t: int = 3
    inertia: float = 0.2


@dataclass
class HybridSORTConfig(MOTBaseConfig):
    """HybridSORT-specific parameters.

    Attributes
    ----------
    reid_weights : ReIDWeights | str | None
        Optional ReID weights path (default: `"osnet_x1_0_msmt17.pt"`).
        Options: same built-in `.pt` names listed in `BoTSORTParams.reid_weights`.
    det_thresh : float
        Detection confidence threshold (default: `0.3`).
        Increasing this reduces weak detections.
    max_age : int
        Maximum age of unmatched tracks (default: `30`).
        Increasing this keeps tracks longer during occlusion.
    min_hits : int
        Minimum hits before track confirmation (default: `3`).
        Increasing this reduces early noisy tracks.
    iou_threshold : float
        IoU threshold for association (default: `0.3`).
        Increasing this makes IoU matching stricter.
    asso_func : str
        Association function name (default: `"iou"`).
        Typical options: `"iou"`, `"giou"`, `"diou"`, `"ciou"`, `"centroid"`.

    """

    model: MOTModels = MOTModels.HYBRIDSORT
    reid_weights: ReIDWeights | str | None = ReIDWeights.OSNET_X1_0_MSMT17
    det_thresh: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    asso_func: str = "iou"


@dataclass
class SFSORTConfig(MOTBaseConfig):
    """SFSORT-specific parameters.

    Attributes
    ----------
    det_thresh : float
        Detection confidence threshold (default: `0.3`).
        Increasing this reduces weak detections.
    max_age : int
        Maximum age of unmatched tracks (default: `30`).
        Increasing this keeps tracks longer through brief misses.
    min_hits : int
        Minimum hits before track confirmation (default: `3`).
        Increasing this reduces short-lived noisy tracks.
    iou_threshold : float
        IoU threshold for association (default: `0.3`).
        Increasing this requires tighter overlap for matching.
    asso_func : str
        Association function name (default: `"iou"`).
        Typical options: `"iou"`, `"giou"`, `"diou"`, `"ciou"`, `"centroid"`.

    """

    model: MOTModels = MOTModels.SFSORT
    det_thresh: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    asso_func: str = "iou"


BoxMOTModelParams = (
    BoTSORTConfig
    | BoostTrackConfig
    | ByteTrackConfig
    | OCSORTConfig
    | StrongSORTConfig
    | DeepOCSORTConfig
    | HybridSORTConfig
    | SFSORTConfig
)


class Tracker:
    """Unified interface for BoxMOT tracking and track post-processing.

    This class runs BoxMOT tracking given a detection file and source video.
    It also provides post-processing utilities to infill missing frames,
    split tracks by large gaps, and drop short tracks.

    Attributes
    ----------
    TRACK_FIELDS : list[str]
        Standard output columns for tracking and post-processing utilities
        (default: class constant).
    device : str
        Device string used by deep trackers (default: `"auto"`).
        Options: `"auto"`, `"cpu"`, `"cuda"`, `"mps"`.
    half : bool
        Whether half precision is enabled for deep trackers (default: `False`).
        Options: `True`, `False`. Enabling can improve speed on supported GPUs.
    boxmot_model : MOTModels
        Selected BoxMOT tracker backend (default: `MOTModels.BOTSORT`).
        Options: `MOTModels.BOTSORT`, `MOTModels.BOOSTTRACK`,
        `MOTModels.BYTE_TRACK`, `MOTModels.OCSORT`,
        `MOTModels.STRONGSORT`, `MOTModels.DEEPOCSORT`,
        `MOTModels.HYBRIDSORT`, `MOTModels.SFSORT`.
    boxmot_config : BoxMOTModelConfig
        Configuration dataclass instance for BoxMOT tracker creation
        (default: model-specific defaults).
    boxmot_verbose : bool
        If False, suppress BoxMOT INFO/SUCCESS logging output.
    output_score_cls : bool
        Whether to include tracker `score` and `cls` values in outputs.
        If `False`, both fields are exported as `-1` to keep file schema stable.
    REID_WEIGHTS_DIR : pathlib.Path
        Directory where relative ReID weights are resolved and stored.
    DEFAULT_REID_WEIGHT : str
        Fallback ReID weight file name used when a model expects ReID
        and no weight is explicitly set.

    """

    TRACK_FIELDS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]  # noqa: RUF012
    REID_WEIGHTS_DIR = Path(__file__).resolve().parent / "reid_weights"
    DEFAULT_REID_WEIGHT = ReIDWeights.OSNET_X1_0_MSMT17.value

    def __init__(
        self,
        config: BoxMOTModelParams | None = None,
        config_yaml: str | None = None,
        device: str = "auto",
        half: bool = False,
        output_score_cls: bool = True,
        boxmot_verbose: bool = False,
    ) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        config : BoxMOTModelConfig, optional
            Configuration bundle for BoxMOT tracker creation. Tracker backend
            is selected from `config.model`.
        config_yaml : str, optional
            YAML file containing model-aware config. When provided,
            values loaded from YAML override `config` input.
        device : str, optional
            Device string used by deep trackers (default: `"auto"`, "cpu", "cuda", "mps").
        half : bool, optional
            Whether half precision is enabled for deep trackers (default: `False`).
        output_score_cls : bool, optional
            If True, output tracker confidence and class values in `score` and
            `cls` columns. If False, export `-1` for both fields.
        boxmot_verbose : bool, optional
            If False, suppress BoxMOT INFO/SUCCESS logging output.

        """
        self.device = device
        self.half = half
        self.boxmot_verbose = boxmot_verbose
        self.output_score_cls = output_score_cls
        yaml_path = config_yaml
        resolved_config = config or config
        self.model_config_yaml = yaml_path

        if yaml_path:
            resolved_config = self.import_config_from_yaml(yaml_path)

        if resolved_config is None:
            resolved_config = self._default_boxmot_config()

        self.boxmot_model = resolved_config.model
        self.boxmot_config = resolved_config

    def track(
        self,
        det_file: str,
        out_file: str,
        video_file: str | None = None,
        show: bool = False,
        video_index: int | None = None,
        total_videos: int | None = None,
    ) -> pd.DataFrame:
        """Run tracking on a single detection file using BoxMOT.

        Parameters
        ----------
        det_file : str
            Path to detection file (CSV format with columns:
            frame, x1, y1, width, height, confidence, class_id).
        out_file : str
            Path to write tracking results. If empty string, results are not saved.
        video_file : str, optional
            Path to source video file. Required for BoxMOT tracker.
        show : bool, optional
            If True (default: False), display live tracking preview with bounding
            boxes and track IDs. Press 's' to toggle preview, 'ESC' to hide,
            'q' to stop tracking early.
        video_index : int, optional
            Index of current video in batch (for progress bar display).
        total_videos : int, optional
            Total number of videos in batch (for complete progress context).

        Returns
        -------
        pd.DataFrame
            Tracking results with columns: frame, track, x, y, w, h, score, cls, r3, r4
            Each row represents one detected object per frame.

        Raises
        ------
        FileNotFoundError
            If det_file or video_file does not exist.
        ValueError
            If video_file is None.

        Notes
        -----
        The tracker processes detections frame-by-frame, maintaining track IDs
        across frames. Detection coordinates are converted from (x1, y1, x2, y2)
        to (x, y, width, height) format for BoxMOT.

        Track IDs are persistent across frame sequences and reused if tracks
        are lost and then re-acquired within track_buffer frames.

        """
        if not Path(det_file).exists():
            msg = f"Detection file not found: {det_file}"
            raise FileNotFoundError(msg)

        if video_file is None:
            msg = "Video file required for BoxMOT tracking but not provided."
            raise ValueError(msg)
        if not Path(video_file).exists():
            msg = f"Video file not found: {video_file}"
            raise FileNotFoundError(msg)

        return self._track_boxmot(
            video_file=video_file,
            det_file=det_file,
            out_file=out_file,
            show=show,
            video_index=video_index,
            total_videos=total_videos,
        )

    def track_batch(
        self,
        det_files: list[str] | None = None,
        video_files: list[str] | None = None,
        output_path: str | None = None,
        is_overwrite: bool = False,
        is_report: bool = True,
    ) -> list[str]:
        """Run tracking on multiple detection files sequentially.

        Parameters
        ----------
        det_files : list[str] | None, optional
            List of detection file paths. Each file should contain frame-level
            detections in CSV format. If None (default), returns empty list.
        video_files : list[str] | None, optional
            List of corresponding source video file paths for each detection file.
            Length should match det_files. Required for BoxMOT tracking.
        output_path : str | None, optional
            Directory to save tracking results. Track files are named based on
            input filename with '_track.txt' suffix. If None (default),
            tracking still runs but results are not persisted.
        is_overwrite : bool, optional
            If False (default), skip tracking for videos with existing output files.
        is_report : bool, optional
            If True (default), include skipped files in returned list.

        Returns
        -------
        list[str]
            List of output track file paths. Includes both newly created and
            existing files (if is_report=True). Empty list if det_files is None.

        Notes
        -----
        Processing is sequential (not parallel). Each detection file is tracked
        in order with progress display showing "Tracking X of Y".

        Files matching between det_files and video_files by index position.
        If video_files is shorter than det_files, missing videos are left None
        and those detections are skipped.

        """
        if det_files is None:
            return []

        results: list[str] = []
        total_videos = len(det_files)

        for idx, det_file in enumerate(det_files, start=1):
            base_filename = os.path.splitext(os.path.basename(det_file))[0].replace("_iou", "")

            track_file = None
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                track_file = os.path.join(output_path, base_filename + "_track.txt")

            if track_file and not is_overwrite and os.path.exists(track_file):
                if is_report:
                    results.append(track_file)
                continue

            # BoxMOT requires a matching source video.
            video_file = None
            if video_files is not None:  # noqa: SIM102
                if idx - 1 < len(video_files):
                    video_file = video_files[idx - 1]

            # run tracking
            self.track(
                det_file=det_file,
                out_file=track_file if track_file else "",  # track() expects a path
                video_file=video_file,
                video_index=idx,
                total_videos=total_videos,
            )

            if track_file:
                results.append(track_file)

        return results

    @staticmethod
    def _build_boxmot_tracker(
        model: MOTModels,
        config: BoxMOTModelParams,
        boxmot_verbose: bool = False,
        device: str | None = None,
        half: bool | None = None,
    ) -> Any:
        """Build a BoxMOT tracker instance from a model enum and typed config."""
        _configure_boxmot_logging(boxmot_verbose)
        _patch_boxmot_requirements_installer()

        try:
            # BoxMOT API moved across versions:
            # - newer: boxmot.create_tracker / boxmot.trackers.tracker_zoo.create_tracker
            # - older: boxmot.tracker_zoo.create_tracker
            try:
                from boxmot import create_tracker  # type: ignore
            except ImportError:
                try:
                    from boxmot.trackers.tracker_zoo import create_tracker  # type: ignore
                except ImportError:
                    from boxmot.tracker_zoo import create_tracker  # type: ignore
        except ImportError as exc:
            msg = "BoxMOT support requires the `boxmot` package. Install it before using `Tracker`."
            raise ImportError(msg) from exc

        tracker_kwargs = {"tracker_type": model.value}
        tracker_kwargs.update(config.to_kwargs())
        if device is not None:
            tracker_kwargs["device"] = device
        if half is not None:
            tracker_kwargs["half"] = half
        tracker_kwargs["device"] = Tracker._resolve_boxmot_device(tracker_kwargs.get("device"))
        if tracker_kwargs.get("reid_weights") is None and model in {
            MOTModels.BOTSORT,
            MOTModels.BOOSTTRACK,
            MOTModels.STRONGSORT,
            MOTModels.DEEPOCSORT,
            MOTModels.HYBRIDSORT,
        }:
            tracker_kwargs["reid_weights"] = Tracker.DEFAULT_REID_WEIGHT

        tracker_kwargs["reid_weights"] = Tracker._resolve_reid_weights_path(tracker_kwargs.get("reid_weights"))

        sig = signature(create_tracker)
        filtered_kwargs = {
            key: value for key, value in tracker_kwargs.items() if key in sig.parameters and value is not None
        }
        return create_tracker(**filtered_kwargs)

    @staticmethod
    def _resolve_boxmot_device(device: Any) -> str | None:
        """Normalize device value for BoxMOT and avoid invalid `device='auto'`."""
        if device is None:
            return None

        device_str = str(device).strip()
        if device_str.lower() != "auto":
            return device_str

        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower() == "auto":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
        except Exception:
            pass

        return "cpu"

    @staticmethod
    def _resolve_reid_weights_path(reid_weights: ReIDWeights | str | None) -> str | None:
        """Resolve relative ReID weight names into `dnt/track/reid_weights`.

        If `reid_weights` is a bare file name (for example
        `osnet_x1_0_msmt17.pt`), this method rewrites it to
        `src/dnt/track/reid_weights/<name>`. BoxMOT will then auto-download
        known weights to this path when the file is missing.
        """
        if not reid_weights:
            return None

        if isinstance(reid_weights, ReIDWeights):
            reid_weights = reid_weights.value

        weight_path = Path(str(reid_weights))
        if weight_path.is_absolute():
            return str(weight_path)
        if weight_path.parent != Path("."):
            return str(weight_path)

        Tracker.REID_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        resolved = Tracker.REID_WEIGHTS_DIR / weight_path.name
        return str(resolved)

    @staticmethod
    def _default_boxmot_config(
        model: MOTModels = MOTModels.BOTSORT,
    ) -> BoxMOTModelParams:
        """Return default config dataclass for a given BoxMOT model."""
        if model == MOTModels.BOTSORT:
            return BoTSORTConfig()
        if model == MOTModels.BOOSTTRACK:
            return BoostTrackConfig()
        if model == MOTModels.BYTE_TRACK:
            return ByteTrackConfig()
        if model == MOTModels.OCSORT:
            return OCSORTConfig()
        if model == MOTModels.STRONGSORT:
            return StrongSORTConfig()
        if model == MOTModels.DEEPOCSORT:
            return DeepOCSORTConfig()
        if model == MOTModels.HYBRIDSORT:
            return HybridSORTConfig()
        if model == MOTModels.SFSORT:
            return SFSORTConfig()
        return BoTSORTConfig()

    @staticmethod
    def _default_boxmot_params(
        model: MOTModels = MOTModels.BOTSORT,
    ) -> BoxMOTModelParams:
        """Backward-compatible wrapper for `_default_boxmot_config`."""
        return Tracker._default_boxmot_config(model=model)

    @staticmethod
    def _params_class_for_model(model: MOTModels) -> type[MOTBaseConfig]:
        """Return parameter dataclass type for a BoxMOT model."""
        mapping: dict[MOTModels, type[MOTBaseConfig]] = {
            MOTModels.BOTSORT: BoTSORTConfig,
            MOTModels.BOOSTTRACK: BoostTrackConfig,
            MOTModels.BYTE_TRACK: ByteTrackConfig,
            MOTModels.OCSORT: OCSORTConfig,
            MOTModels.STRONGSORT: StrongSORTConfig,
            MOTModels.DEEPOCSORT: DeepOCSORTConfig,
            MOTModels.HYBRIDSORT: HybridSORTConfig,
            MOTModels.SFSORT: SFSORTConfig,
        }
        return mapping[model]

    @staticmethod
    def export_config_to_yaml(
        yaml_file: str,
        config: BoxMOTModelParams,
    ) -> None:
        """Export model-aware BoxMOT config to a YAML file."""
        out_path = Path(yaml_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config.to_dict(), f, sort_keys=False)

    @staticmethod
    def import_config_from_yaml(yaml_file: str) -> BoxMOTModelParams:
        """Import model-aware BoxMOT config from a YAML file."""
        with Path(yaml_file).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            msg = f"Invalid YAML content in {yaml_file}: expected a mapping."
            raise ValueError(msg)

        model = MOTModels(str(data.get("model", MOTModels.BOTSORT.value)))
        param_cls = Tracker._params_class_for_model(model)
        return param_cls.from_dict(data)

    # Backward-compatible wrappers.
    @staticmethod
    def export_params_to_yaml(
        yaml_file: str,
        params: BoxMOTModelParams,
    ) -> None:
        """Export model-aware BoxMOT params to a YAML file (backward-compatible wrapper)."""
        Tracker.export_config_to_yaml(yaml_file=yaml_file, config=params)

    @staticmethod
    def import_params_from_yaml(yaml_file: str) -> BoxMOTModelParams:
        """Import model-aware BoxMOT params from a YAML file (backward-compatible wrapper)."""
        return Tracker.import_config_from_yaml(yaml_file=yaml_file)

    def export_current_config_to_yaml(self, yaml_file: str) -> None:
        """Export this tracker's active model and config to YAML."""
        self.export_config_to_yaml(
            yaml_file=yaml_file,
            config=self.boxmot_config,
        )

    def export_current_params_to_yaml(self, yaml_file: str) -> None:
        """Export this tracker's active model and config to YAML (backward-compatible wrapper)."""
        self.export_current_config_to_yaml(yaml_file)

    @staticmethod
    def _parse_boxmot_outputs(
        outputs: Any,
        frame_id: int,
        output_score_cls: bool = True,
    ) -> list[list[float | int]]:
        """Normalize BoxMOT outputs into the standard DNT track row layout."""
        if outputs is None:
            return []

        tracks = outputs if isinstance(outputs, np.ndarray) else np.asarray(outputs)

        if tracks.size == 0:
            return []
        if tracks.ndim == 1:
            tracks = np.expand_dims(tracks, axis=0)

        parsed: list[list[float | int]] = []
        for row in tracks:
            if len(row) < 5:
                continue
            x1, y1, x2, y2, tid = row[:5]
            if output_score_cls:
                score = round(float(row[5]), 2) if len(row) > 5 else -1
                cls = int(row[6]) if len(row) > 6 else -1
            else:
                score = -1
                cls = -1
            x = round(float(x1))
            y = round(float(y1))
            w = round(float(x2 - x1))
            h = round(float(y2 - y1))
            parsed.append([
                int(frame_id),
                int(tid),
                x,
                y,
                w,
                h,
                score,
                cls,
                -1,
                -1,
            ])
        return parsed

    def _track_boxmot(
        self,
        video_file: str,
        det_file: str,
        out_file: str,
        show: bool = False,
        video_index: int | None = None,
        total_videos: int | None = None,
    ) -> pd.DataFrame:
        """Execute BoxMOT tracking on one video with corresponding detections.

        Internal method that runs the main tracking loop per video. Processes
        detections frame-by-frame, maintains track associations using BoxMOT,
        and optionally displays a live preview window.

        Parameters
        ----------
        video_file : str
            Path to input video file.
        det_file : str
            Path to detections CSV file.
        out_file : str
            Output path for tracking results. If empty, results are not saved.
        show : bool, optional
            If True, display tracking preview (default: False).
        video_index : int | None, optional
            Index in batch progression (for UI display).
        total_videos : int | None, optional
            Total videos in batch (for UI display).

        Returns
        -------
        pd.DataFrame
            Tracking results with TRACK_FIELDS columns.

        """
        """Run BoxMOT tracking on one video/detection pair and return track records."""
        detections = pd.read_csv(det_file, header=None).to_numpy()
        if detections.size == 0 or len(detections) == 0:
            return pd.DataFrame(columns=self.TRACK_FIELDS)

        start_frame = int(detections[:, 0].min())
        end_frame = int(detections[:, 0].max())
        total_frames = max(end_frame - start_frame + 1, 0)

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Couldn't open video {video_file}")
            return pd.DataFrame(columns=self.TRACK_FIELDS)

        tracker = self._build_boxmot_tracker(
            self.boxmot_model,
            self.boxmot_config,
            boxmot_verbose=self.boxmot_verbose,
            device=self.device,
            half=self.half,
        )

        desc = f"Tracking {Path(video_file).stem}"
        pbar = tqdm(total=total_frames, desc=desc, unit="frames")
        if video_index is not None and total_videos is not None:
            pbar.set_description_str(f"Tracking {video_index} of {total_videos} - {Path(video_file).stem}")

        results: list[list[float | int]] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        preview_enabled = show
        window_title = "Tracking Preview | s: show preview | ESC: hide preview | q: quit tracking"
        control_title = "Tracking Control"

        while cap.isOpened():
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret or frame_id > end_frame:
                break

            frame_dets = detections[detections[:, 0] == frame_id]
            if frame_dets.size > 0:
                # Convert detection format: (x, y, w, h) -> (x1, y1, x2, y2)
                # Input:  [frame_id, _, x, y, w, h, score, class_id]
                # Output: [x1, y1, x2, y2, score, class_id]
                dets = np.array(
                    [[d[2], d[3], d[2] + d[4], d[3] + d[5], d[6], d[7]] for d in frame_dets],
                    dtype=float,
                )
            else:
                dets = np.empty((0, 6), dtype=float)

            try:
                outputs = tracker.update(dets, frame)
            except TypeError:
                outputs = tracker.update(dets, frame, None)

            frame_tracks = self._parse_boxmot_outputs(
                outputs,
                frame_id,
                output_score_cls=self.output_score_cls,
            )
            results.extend(frame_tracks)

            if show:
                control = np.zeros((80, 720, 3), dtype=np.uint8)
                status = "preview ON" if preview_enabled else "preview OFF"
                cv2.putText(
                    control,
                    f"{status} | ESC: hide preview | s: show preview | q: quit tracking",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(control_title, control)

                if preview_enabled:
                    preview = frame.copy()
                    for row in frame_tracks:
                        x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                        tid = int(row[1])
                        cls = int(row[7])
                        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"id={tid} cls={cls}" if self.output_score_cls else f"id={tid}"
                        cv2.putText(
                            preview,
                            label,
                            (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.imshow(window_title, preview)
                else:
                    with contextlib.suppress(cv2.error):
                        cv2.destroyWindow(window_title)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC: hide preview window, keep tracking
                    preview_enabled = False
                elif key == ord("s"):  # s: show preview window again
                    preview_enabled = True
                elif key == ord("q"):  # q: stop tracking early
                    break

            pbar.update()

        cap.release()
        pbar.close()
        if show:
            cv2.destroyAllWindows()

        df = pd.DataFrame(results, columns=self.TRACK_FIELDS)
        if out_file:
            out_dir = Path(out_file).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_file, index=False, header=None)
        return df


if __name__ == "__main__":
    input_video = "/mnt/d/videos/sample/traffic.mp4"
    det_file = "/mnt/d/videos/sample/dets/traffic_det.txt"
    output_file = "/mnt/d/videos/sample/tracks/traffic_track.txt"

    config = BoTSORTConfig(
        reid_weights=ReIDWeights.OSNET_X1_0_MSMT17,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.8,
        track_buffer=30,
        with_reid=True,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
    )

    fps = cv2.VideoCapture(input_video).get(cv2.CAP_PROP_FPS)

    c2 = ByteTrackConfig(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=fps,
    )

    c3 = SFSORTConfig(
        det_thresh=0.3,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        asso_func="iou",
    )

    tracker = Tracker(
        # This is overridden by `config_yaml` when present.
        config=c3,
        device="auto",
        half=True,
        output_score_cls=False,
    )

    tracker.track(
        det_file=det_file,
        out_file=output_file,
        video_file=input_video,
        show=True,
    )
