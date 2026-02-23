"""Track module for detection tracking functionality.

This module provides tracking utilities and classes for object detection.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from .re_class import ReClass as ReClass
except ModuleNotFoundError:
    # Keep tracker imports usable even when optional re-class dependencies
    # (for example `cython_bbox`) are not installed.
    ReClass = None  # type: ignore[assignment]

from .post_process import interpolate_tracks_rts as interpolate_tracks_rts
from .post_process import link_tracklets as link_tracklets
from .tracker import (
    BoostTrackConfig as BoostTrackConfig,
)
from .tracker import (
    BoTSORTConfig as BoTSORTConfig,
)
from .tracker import (
    ByteTrackConfig as ByteTrackConfig,
)
from .tracker import (
    DeepOCSORTConfig as DeepOCSORTConfig,
)
from .tracker import (
    HybridSORTConfig as HybridSORTConfig,
)
from .tracker import (
    MOTBaseConfig as MOTBaseConfig,
)
from .tracker import (
    MOTModels as MOTModels,
)
from .tracker import (
    OCSORTConfig as OCSORTConfig,
)
from .tracker import (
    ReIDWeights as ReIDWeights,
)
from .tracker import (
    SFSORTConfig as SFSORTConfig,
)
from .tracker import (
    StrongSORTConfig as StrongSORTConfig,
)
from .tracker import (
    Tracker as Tracker,
)

# Backward-compat aliases for historical exported names.
BOTSORTConfig = BoTSORTConfig
SFSortConfig = SFSORTConfig

__all__ = [
    "BoTSORTConfig",
    "BoostTrackConfig",
    "ByteTrackConfig",
    "DeepOCSORTConfig",
    "HybridSORTConfig",
    "MOTBaseConfig",
    "MOTModels",
    "OCSORTConfig",
    "ReClass",
    "ReIDWeights",
    "SFSORTConfig",
    "StrongSORTConfig",
    "Tracker",
    "interpolate_tracks_rts",
    "link_tracklets",
]
