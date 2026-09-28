"""Track refinement: switch splitting, false-track screening, linking, and gap filling.

Design: docs/superpowers/specs/2026-09-27-track-refinement-design.md.
"""

from .config import RefineConfig
from .interpolate import interpolate_tracks_rts
from .link import link_tracklets
from .refiner import RefineResult, TrackRefiner

__all__ = [
    "RefineConfig",
    "RefineResult",
    "TrackRefiner",
    "interpolate_tracks_rts",
    "link_tracklets",
]
