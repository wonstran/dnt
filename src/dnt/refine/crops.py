"""Frame access and box crops for appearance encoding (spec 5.3)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

#: A box is enlarged by this factor on each axis before it is cropped. Part of the cache key.
CROP_PAD = 1.1
#: A jump of more than this many frames is done by seeking; a shorter one by decoding on.
SEEK_GAP = 100


def crop_box(frame: np.ndarray, box, pad: float = CROP_PAD) -> np.ndarray | None:
    """Return the RGB crop of ``box`` enlarged by ``pad``, or ``None`` if nothing is left.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image, as read by OpenCV.
    box : sequence of float
        ``(x, y, w, h)`` in pixels.
    pad : float
        Enlargement of the box on each axis before clipping to the frame.

    Returns
    -------
    numpy.ndarray or None
        ``uint8`` array ``(h, w, 3)`` in RGB order, or ``None`` when the box is empty,
        non-finite, or has fewer than 2 pixels left on an axis after clipping.

    """
    height, width = frame.shape[:2]
    x, y, w, h = (float(v) for v in box)
    if not (np.isfinite([x, y, w, h]).all() and w > 0 and h > 0):
        return None
    cx, cy = x + w / 2.0, y + h / 2.0
    # round, not floor/ceil: 100 * 1.1 / 2 is 55.00000000000001, which would add a pixel
    x0 = max(0, round(cx - w * pad / 2.0))
    x1 = min(width, round(cx + w * pad / 2.0))
    y0 = max(0, round(cy - h * pad / 2.0))
    y1 = min(height, round(cy + h * pad / 2.0))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.ascontiguousarray(frame[y0:y1, x0:x1, ::-1])


class FrameReader:
    """Read chosen frames of a video, decoding on over short gaps and seeking over long ones."""

    def __init__(self, path):
        """Open ``path``; raise ``ValueError`` if OpenCV cannot."""
        import cv2

        self.path = str(path)
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        self._next = 0

    def __enter__(self) -> FrameReader:
        """Return the reader."""
        return self

    def __exit__(self, *exc) -> None:
        """Release the video."""
        self.close()

    def close(self) -> None:
        """Release the video."""
        self._cap.release()

    def frames(self, wanted: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, BGR frame)`` for each unique wanted index, in increasing order.

        Raises
        ------
        ValueError
            If a frame cannot be read (for example, it is past the end of the video).

        """
        for f in sorted({int(v) for v in wanted}):
            if f < self._next or f - self._next > SEEK_GAP:
                self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, f)
                self._next = f
            while self._next < f:
                if not self._cap.grab():
                    raise ValueError(
                        f"cannot read frame {f} of {self.path}: "
                        f"the video ends at frame {self._next}"
                    )
                self._next += 1
            ok, img = self._cap.read()
            if not ok or img is None:
                raise ValueError(f"cannot read frame {f} of {self.path}")
            self._next += 1
            yield f, img
