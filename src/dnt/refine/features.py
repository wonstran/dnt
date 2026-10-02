"""The appearance interface the stages use (spec 5.3), array providers, and the embedding cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .io import sha256_file

log = logging.getLogger(__name__)

#: Bumped whenever the crop or embedding code changes, so old caches are not reused.
#: Version 2 added the keys of skipped (empty or unreadable) crops to the file.
#: Version 3 stopped embedding boxes smaller than ``encoder.min_crop_px`` (part of the key).
FEATURES_VERSION = 3


class Appearance(Protocol):
    """Source of clean (unoccluded), L2-normalized embeddings per raw track and frame."""

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames, embeddings)`` for ``raw_id`` within ``[f0, f1]``, sorted by frame."""
        ...


class DenseAppearance(Appearance, Protocol):
    """An ``Appearance`` that can also give every clean frame around a stage 1 candidate.

    ``clean_embeddings`` then returns the coarse samples only. Stage 1 finds candidates on them
    and calls ``prefetch_dense`` once with every window it will need, then ``dense_embeddings``.
    """

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every clean observed frame of ``raw_id`` within ``[f0, f1]``, sorted."""
        ...

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Compute the embeddings of ``(raw_id, f0, f1)`` windows in one pass."""
        ...


class ArrayAppearance:
    """In-memory ``Appearance`` built from arrays (tests and callers with precomputed features)."""

    def __init__(self, table: Mapping[int, tuple[Sequence[int], np.ndarray]]):
        """Store ``{raw_id: (frames, embeddings)}``, sorted and L2-normalized."""
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for raw, (frames, emb) in table.items():
            f = np.asarray(list(frames), dtype=int)
            e = np.asarray(emb, dtype=float)
            if len(f) != len(e):
                raise ValueError(f"raw_id {raw}: frames and embeddings have different lengths")
            if e.ndim != 2 or e.shape[0] == 0:
                raise ValueError(f"raw_id {raw}: embeddings must be 2-D with at least one row")
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


class CoarseArrayAppearance(ArrayAppearance):
    """``ArrayAppearance`` that behaves like a video provider: coarse by default, dense on request.

    ``clean_embeddings`` returns every ``every``-th stored sample (by ordinal within the raw
    track); ``dense_embeddings`` returns all stored samples.
    """

    def __init__(self, table: Mapping[int, tuple[Sequence[int], np.ndarray]], every: int = 5):
        """Store the samples; ``every`` is the coarse stride."""
        if int(every) < 1:
            raise ValueError("every must be at least 1")
        super().__init__(table)
        self._every = int(every)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the coarse samples of ``raw_id`` within ``[f0, f1]``."""
        if int(raw_id) not in self._t:
            return np.empty(0, dtype=int), np.empty((0, 0))
        f, e = self._t[int(raw_id)]
        m = (np.arange(len(f)) % self._every == 0) & (f >= f0) & (f <= f1)
        return f[m], e[m]

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every stored sample of ``raw_id`` within ``[f0, f1]``."""
        return ArrayAppearance.clean_embeddings(self, raw_id, f0, f1)

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Do nothing: the samples are already in memory."""
        return None


def _join(parts) -> tuple[np.ndarray, np.ndarray]:
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return np.empty(0, dtype=int), np.empty((0, 0))
    f = np.concatenate([p[0] for p in parts])
    e = np.vstack([p[1] for p in parts])
    order = np.argsort(f, kind="stable")
    return f[order], e[order]


def track_embeddings(appearance: Appearance, lineage) -> tuple[np.ndarray, np.ndarray]:
    """Return a track's clean (coarse) samples across its lineage spans, sorted by frame."""
    return _join([appearance.clean_embeddings(int(r), int(a), int(b)) for r, a, b in lineage])


def dense_track_embeddings(
    appearance: DenseAppearance, lineage, f0: int, f1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a track's dense samples within ``[f0, f1]`` across its lineage spans, sorted."""
    parts = []
    for r, a, b in lineage:
        lo, hi = max(int(a), int(f0)), min(int(b), int(f1))
        if lo <= hi:
            parts.append(appearance.dense_embeddings(int(r), lo, hi))
    return _join(parts)


def features_key(
    *, tracks_sha, video, context_sha, encoder, sample_every, occlusion_iou, crop_pad, min_crop_px
) -> str:
    """Return the SHA-256 key of every input that can change an embedding (spec 5.3).

    Parameters
    ----------
    tracks_sha : str
        SHA-256 of the input track file.
    video : dict
        Video fingerprint with ``sha256``, ``size`` and ``frame_count``.
    context_sha : str or None
        SHA-256 of the context file, or ``None``.
    encoder : object
        Anything with ``name``, ``model_name``, ``weights_sha`` and ``preprocess_id``.
    sample_every, occlusion_iou, crop_pad : float
        Sampling stride, occlusion threshold, and crop padding.
    min_crop_px : int
        Minimum longer box side, in pixels, of an embedded crop (``0``: no minimum).

    Returns
    -------
    str
        Hex digest.

    """
    parts = {
        "version": FEATURES_VERSION,
        "tracks": tracks_sha,
        "video": [video["sha256"], int(video["size"]), int(video["frame_count"])],
        "context": context_sha or "none",
        "encoder": [encoder.name, encoder.model_name, encoder.weights_sha, encoder.preprocess_id],
        "sample_every": int(sample_every),
        "occlusion_iou": float(occlusion_iou),
        "crop_pad": float(crop_pad),
        "min_crop_px": int(min_crop_px),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _keys_ok(ids, frames) -> bool:
    """Return whether ``ids`` and ``frames`` are 1-D integer arrays of one length, no pair twice."""
    if ids.ndim != 1 or frames.ndim != 1 or len(ids) != len(frames):
        return False
    if ids.dtype.kind not in "iu" or frames.dtype.kind not in "iu":
        return False
    return len(set(zip(ids.tolist(), frames.tolist(), strict=True))) == len(ids)


def _checked(ids, frames, emb, dim) -> np.ndarray | None:
    """Return the embeddings as float32 if the cache arrays are usable, else ``None``.

    Usable means: 1-D integer ids and frames, a 2-D float matrix with one row each, the width
    ``dim`` when it is given (and at least one column otherwise), finite values *after* the
    conversion to float32, and no duplicate ``(raw_id, frame)`` pair.
    """
    if not _keys_ok(ids, frames) or emb.ndim != 2 or emb.dtype.kind != "f":
        return None
    if emb.shape[0] != len(ids):
        return None
    if len(ids) and (emb.shape[1] == 0 or (dim is not None and emb.shape[1] != dim)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        out = emb.astype(np.float32)
    if not bool(np.isfinite(out).all()):
        return None
    return out


class FeatureStore:
    """Embeddings per (raw track id, frame), saved as a deterministic ``.npz`` under a key.

    The store also records the boxes whose crop was empty or unreadable (``skip``), so a rerun
    does not read the video for them again and a replay can tell them from missing entries.
    """

    def __init__(self, key: str):
        """Create an empty store for cache key ``key``."""
        self.key = key
        self.dirty = False
        self._dim: int | None = None
        self._d: dict[int, dict[int, np.ndarray]] = {}
        self._skipped: set[tuple[int, int]] = set()

    def __len__(self) -> int:
        """Return the number of stored embeddings."""
        return sum(len(v) for v in self._d.values())

    def has(self, raw_id: int, frame: int) -> bool:
        """Return whether ``(raw_id, frame)`` has an embedding."""
        return int(frame) in self._d.get(int(raw_id), {})

    def is_skipped(self, raw_id: int, frame: int) -> bool:
        """Return whether ``(raw_id, frame)`` was recorded as having no usable crop."""
        return (int(raw_id), int(frame)) in self._skipped

    def skip(self, raw_id: int, frame: int) -> None:
        """Record that ``(raw_id, frame)`` has no usable crop (empty or unreadable)."""
        key = (int(raw_id), int(frame))
        if self.has(*key):
            raise ValueError(f"{key} has an embedding; it cannot also be skipped")
        if key not in self._skipped:
            self._skipped.add(key)
            self.dirty = True

    def put(self, raw_id: int, frame: int, emb) -> None:
        """Store one embedding; every embedding of a store has the same width."""
        if self.is_skipped(raw_id, frame):
            raise ValueError(f"{(int(raw_id), int(frame))} is skipped; it cannot be embedded")
        e = np.array(emb, dtype=np.float32)
        if e.ndim != 1 or (self._dim is not None and e.shape[0] != self._dim):
            raise ValueError(
                f"embedding of shape {e.shape} does not match the store's width {self._dim}"
            )
        self._dim = int(e.shape[0])
        self._d.setdefault(int(raw_id), {})[int(frame)] = e
        self.dirty = True

    def get(self, raw_id: int, frames: Sequence[int]) -> np.ndarray:
        """Return the embeddings of ``frames`` of ``raw_id`` as a ``(len(frames), D)`` array."""
        rows = self._d[int(raw_id)]
        return np.stack([rows[int(f)] for f in frames])

    def save(self, path) -> str:
        """Write the store atomically and return the file's SHA-256."""
        ids, frames, embs = [], [], []
        for rid in sorted(self._d):
            for f in sorted(self._d[rid]):
                ids.append(rid)
                frames.append(f)
                embs.append(self._d[rid][f])
        dim = embs[0].shape[0] if embs else 0
        skipped = sorted(self._skipped)
        arrays = {
            "key": np.array(self.key),
            "raw_id": np.asarray(ids, dtype=np.int64),
            "frame": np.asarray(frames, dtype=np.int64),
            "emb": np.asarray(embs, dtype=np.float32).reshape(len(ids), dim),
            "skipped_raw_id": np.asarray([k[0] for k in skipped], dtype=np.int64),
            "skipped_frame": np.asarray([k[1] for k in skipped], dtype=np.int64),
        }
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        try:
            with tmp.open("wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, p)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self.dirty = False
        return sha256_file(p)

    @classmethod
    def load(cls, path, key: str, dim: int | None = None) -> FeatureStore | None:
        """Return the stored cache, or ``None`` if it is missing, damaged, or for other inputs.

        Parameters
        ----------
        path : path-like
            The ``.features.npz`` file.
        key : str
            The cache key the file must carry.
        dim : int, optional
            The encoder's embedding width; a file with another width is a miss.

        """
        p = Path(path)
        if not p.is_file():
            return None
        try:
            with p.open("rb") as fh, np.load(fh, allow_pickle=False) as z:
                key_array = z["key"]
                ids, frames, emb = z["raw_id"], z["frame"], z["emb"]
                skip_ids, skip_frames = z["skipped_raw_id"], z["skipped_frame"]
        except Exception as err:  # a damaged or old-format cache is a miss, never an exception
            log.info("feature cache %s is unreadable (%s); recomputing", p, err)
            return None
        arrays = (key_array, ids, frames, emb, skip_ids, skip_frames)
        if not all(isinstance(a, np.ndarray) for a in arrays):
            log.info("feature cache %s does not hold arrays; recomputing", p)
            return None
        if key_array.ndim != 0 or key_array.dtype.kind != "U" or str(key_array) != key:
            log.info("feature cache %s was built with different inputs; recomputing", p)
            return None
        emb32 = _checked(ids, frames, emb, dim)
        if emb32 is None:
            log.info("feature cache %s is malformed or has another width; recomputing", p)
            return None
        skipped = (
            set(zip(skip_ids.tolist(), skip_frames.tolist(), strict=True))
            if _keys_ok(skip_ids, skip_frames)
            else None
        )
        if skipped is None or not skipped.isdisjoint(
            zip(ids.tolist(), frames.tolist(), strict=True)
        ):
            log.info("feature cache %s has malformed skipped keys; recomputing", p)
            return None
        store = cls(key)
        for r, f, e in zip(ids.tolist(), frames.tolist(), emb32, strict=True):
            store._d.setdefault(r, {})[f] = e
        store._skipped = skipped
        if len(emb32):
            store._dim = int(emb32.shape[1])
        return store
