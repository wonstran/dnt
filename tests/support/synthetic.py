"""Synthetic traffic scene and stub detector for dnt tests.

Plain module (no pytest dependency) so the wheel smoke test can copy it
outside the repository and run it with an installed dnt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WIDTH, HEIGHT, FPS, N_FRAMES = 320, 240, 25, 150
GAP = range(60, 68)  # object 1 hidden: interpolation gap
OBJ3_START = 30  # object 3 enters late: confirmation probes
DET_FIELDS = ["frame", "res", "x", "y", "w", "h", "conf", "class"]


@dataclass(frozen=True)
class GroundTruth:
    """Per-frame boxes (columns frame, obj, x, y, w, h) and hidden frame ranges."""

    boxes: pd.DataFrame
    hidden: dict[int, range]


def _box(obj: int, f: int) -> tuple[float, float, int, int] | None:
    if obj == 1:
        return None if f in GAP else (10 + 1.5 * f, 40.0, 30, 20)
    if obj == 2:
        return (150.0, 10 + 1.2 * f, 24, 24)
    if obj == 3:
        return None if f < OBJ3_START else (300 - 2.1 * (f - OBJ3_START), 130.0, 28, 22)
    raise ValueError(obj)


_COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}


def make_synthetic_video(path: str | Path) -> GroundTruth:
    """Write a 150-frame 320x240 mp4 with three moving boxes; return the ground truth."""
    path = Path(path)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    rows = []
    try:
        for f in range(N_FRAMES):
            img = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
            for obj in (1, 2, 3):
                b = _box(obj, f)
                if b is None:
                    continue
                x, y, w, h = b
                cv2.rectangle(img, (int(x), int(y)), (int(x) + w, int(y) + h), _COLORS[obj], -1)
                rows.append((f, obj, int(x), int(y), w, h))
            writer.write(img)
    finally:
        writer.release()
    boxes = pd.DataFrame(rows, columns=["frame", "obj", "x", "y", "w", "h"])
    return GroundTruth(boxes=boxes, hidden={1: GAP})


class StubDetector:
    """Stand-in for `dnt.detect.Detector`: same `detect()` call and output, no model."""

    DET_FIELDS = DET_FIELDS

    def __init__(
        self,
        truth: GroundTruth,
        *,
        seed: int = 0,
        noise_px: int = 1,
        drop_rate: float = 0.03,
        cls: int = 2,
        conf: float = 0.9,
        low_conf: float | None = None,
    ) -> None:
        self.truth = truth
        self.seed = seed
        self.noise_px = noise_px
        self.drop_rate = drop_rate
        self.cls = cls
        self.conf = conf
        self.low_conf = low_conf

    def detect(self, input_video: str | Path, iou_file: str | Path | None = None) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(input_video))
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        rng = np.random.default_rng(self.seed)
        b = self.truth.boxes[self.truth.boxes.frame < n_frames].reset_index(drop=True)
        keep = rng.random(len(b)) >= self.drop_rate
        noise = rng.integers(-self.noise_px, self.noise_px + 1, size=(len(b), 4))
        b = b[keep].reset_index(drop=True)
        noise = noise[keep]
        conf = np.full(len(b), self.conf)
        if self.low_conf is not None:
            conf = np.where(b["frame"].to_numpy() % 2 == 1, self.low_conf, self.conf)
        df = pd.DataFrame({
            "frame": b["frame"].astype(int),
            "res": -1,
            "x": (b["x"] + noise[:, 0]).astype(int),
            "y": (b["y"] + noise[:, 1]).astype(int),
            "w": (b["w"] + noise[:, 2]).astype(int),
            "h": (b["h"] + noise[:, 3]).astype(int),
            "conf": conf.round(2),
            "class": self.cls,
        })[DET_FIELDS].sort_values(["frame", "x"]).reset_index(drop=True)
        if iou_file is not None:
            Path(iou_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(iou_file, index=False, header=False)
        return df
