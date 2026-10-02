"""A tiny colored-box video and a stub encoder for the appearance tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FPS = 10.0
WIDTH, HEIGHT = 320, 240
BACKGROUND = 90
RED = (0, 0, 200)  # BGR
BLUE = (200, 0, 0)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


RED_DIR = _unit(np.array([200.0, 0.0, 0.0]) - BACKGROUND)  # RGB, away from the background
BLUE_DIR = _unit(np.array([0.0, 0.0, 200.0]) - BACKGROUND)


def make_color_video(path, rows, n_frames, fps=FPS):
    """Write an mp4 where each ``(frame, x, y, w, h, bgr)`` row is a filled rectangle."""
    by_frame: dict[int, list] = {}
    for f, x, y, w, h, color in rows:
        by_frame.setdefault(int(f), []).append((x, y, w, h, color))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    try:
        for f in range(n_frames):
            img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
            for x, y, w, h, color in by_frame.get(f, []):
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
            writer.write(img)
    finally:
        writer.release()
    return Path(path)


def video_rows(track_rows, color):
    """Turn track rows ``[f, track, x, y, w, h, ...]`` into ``make_color_video`` rows."""
    return [(r[0], r[2], r[3], r[4], r[5], color) for r in track_rows]


class ColorEncoder:
    """Stub encoder: the direction of a crop's mean RGB color away from the gray background."""

    name = "stub"
    model_name = "stub-color"
    preprocess_id = "stub-v1"
    weights_sha = None
    dim = 3

    def __init__(self):
        self.calls = 0
        self.crops = 0
        self.max_batch = 0

    def encode(self, crops):
        self.calls += 1
        self.crops += len(crops)
        self.max_batch = max(self.max_batch, len(crops))
        out = []
        for c in crops:
            v = c.reshape(-1, 3).mean(axis=0) - BACKGROUND
            n = np.linalg.norm(v)
            out.append(v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0]))
        return np.asarray(out, dtype=np.float32).reshape(-1, 3)
