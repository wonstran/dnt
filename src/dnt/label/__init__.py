"""Label package for detection tracking."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from .labeler import ElementType as ElementType
from .labeler import Encoder as Encoder
from .labeler import Labeler
from .labeler import LabelMethod as LabelMethod
from .labeler import Preset as Preset
from .labeler import TrackClipMethod as TrackClipMethod

__all__ = ["ElementType", "Encoder", "LabelMethod", "Labeler", "Preset", "TrackClipMethod"]
