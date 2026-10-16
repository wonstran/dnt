"""Fakes for ReClass and VehReClass tests: a frame reader and a scripted detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

DET_FIELDS = ["frame", "res", "x", "y", "w", "h", "conf", "class"]
TRACK_FIELDS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]


class FakeReader:
    """Mimic ``refine.crops.FrameReader``: a failed frame raises and ends that generator."""

    def __init__(self, path, bad=(), shape=(1080, 1920)):
        self.path = str(path)
        self.bad = set(bad)
        self.shape = shape
        self.requests: list[list[int]] = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True

    def frames(self, wanted):
        wanted = sorted({int(v) for v in wanted})
        self.requests.append(wanted)
        for f in wanted:
            if f in self.bad:
                raise ValueError(f"cannot read frame {f}")
            yield f, np.full((*self.shape, 3), f % 255, np.uint8)


class FakeDetector:
    """Return scripted boxes per frame; record every frame it is asked about."""

    DET_FIELDS = DET_FIELDS

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.seen: list[int] = []

    def _rows(self, frame):
        return [
            {"frame": frame, "res": -1, "x": x, "y": y, "w": w, "h": h, "conf": c, "class": k}
            for (x, y, w, h, c, k) in self.script.get(int(frame), [])
        ]

    def detect_image(self, image, frame=0):
        self.seen.append(int(frame))
        return pd.DataFrame(self._rows(frame), columns=DET_FIELDS)

    def detect_frames(self, input_video, frames, verbose=True):
        rows = [r for f in frames for r in self._rows(f)]
        self.seen.extend(int(f) for f in frames)
        return pd.DataFrame(rows, columns=DET_FIELDS)


def vehicle_rows(track, frames, box, **cols) -> pd.DataFrame:
    """Rows of one track at ``frames`` with box ``(x, y, w, h)``; ``cols`` override columns."""
    x, y, w, h = box
    df = pd.DataFrame({
        "frame": list(frames), "track": track, "x": x, "y": y, "w": w, "h": h,
        "score": -1.0, "cls": -1, "r3": -1, "r4": -1,
    })
    for k, v in cols.items():
        df[k] = v
    return df[TRACK_FIELDS + [c for c in cols if c not in TRACK_FIELDS]]
