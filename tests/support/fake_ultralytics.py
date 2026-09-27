"""Minimal stand-in for ultralytics YOLO/RTDETR used by detector unit tests."""

from __future__ import annotations

import numpy as np


class _T:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self):
        return len(self.xyxy.numpy())


class _Result:
    def __init__(self, boxes, masks=None):
        self.boxes = boxes
        self.masks = masks

    def plot(self):
        raise NotImplementedError


class FakeModel:
    """Returns one fixed box with fractional coordinates for every frame."""

    BOX = (10.6, 20.4, 50.9, 70.2)

    def __init__(self, path: str):
        self.path = path

    def predict(self, source=None, **kwargs):
        self.last_kwargs = kwargs
        return [_Result(_Boxes([self.BOX], [0.876], [2]))]
