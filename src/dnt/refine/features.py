"""The appearance interface the stages use (spec 5.3); real providers arrive in Plan 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np


class Appearance(Protocol):
    """Source of clean (unoccluded), L2-normalized embeddings per raw track and frame."""

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames, embeddings)`` for ``raw_id`` within ``[f0, f1]``, sorted by frame."""
        ...


class ArrayAppearance:
    """In-memory ``Appearance`` built from arrays (tests and callers with precomputed features)."""

    def __init__(self, table: Mapping[int, tuple[Sequence[int], np.ndarray]]):
        """Store ``{raw_id: (frames, embeddings)}``, sorted and L2-normalized."""
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for raw, (frames, emb) in table.items():
            f = np.asarray(list(frames), dtype=int)
            e = np.asarray(emb, dtype=float)
            order = np.argsort(f, kind="stable")
            f, e = f[order], e[order]
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            self._t[int(raw)] = (f, e)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored samples of ``raw_id`` within ``[f0, f1]``."""
        if int(raw_id) not in self._t:
            return np.empty(0, dtype=int), np.empty((0, 0))
        f, e = self._t[int(raw_id)]
        m = (f >= f0) & (f <= f1)
        return f[m], e[m]


def track_embeddings(appearance: Appearance, lineage) -> tuple[np.ndarray, np.ndarray]:
    """Return a track's clean samples across its lineage spans, sorted by frame."""
    parts = [appearance.clean_embeddings(int(r), int(a), int(b)) for r, a, b in lineage]
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return np.empty(0, dtype=int), np.empty((0, 0))
    f = np.concatenate([p[0] for p in parts])
    e = np.vstack([p[1] for p in parts])
    order = np.argsort(f, kind="stable")
    return f[order], e[order]
