"""``Appearance`` provider that embeds crops of a video, with an on-disk cache (spec 5.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .crops import CROP_PAD, FrameReader, crop_box
from .features import FeatureStore

#: Crops are encoded as soon as this many batches' worth are waiting, so memory stays bounded.
_FLUSH_BATCHES = 8


class VideoAppearance:
    """Embed the clean crops of raw tracks, on demand, and keep them in a ``FeatureStore``.

    A raw track's *coarse* samples are the observed frames with ordinal ``0, k, 2k, ...``
    (``k = sample_every``); *dense* samples are all of its observed frames. A sample is used
    only if its row is clean (not occluded, and its box's longer side is at least
    ``min_crop_px``) and its crop is not empty. Rows that are not clean still count for the
    ordinals.

    ``view(min_crop_px)`` gives a stage its own, stricter size rule over the same embeddings.

    Attributes
    ----------
    coarse_clean : int
        Coarse samples of all raw tracks whose rows are clean.
    coarse_too_small : int
        Coarse samples whose rows are not occluded but whose boxes are smaller than
        ``min_crop_px``, so they are not embedded.

    """

    def __init__(
        self,
        work: pd.DataFrame,
        occluded: pd.Series,
        video_file,
        encoder,
        store: FeatureStore,
        *,
        sample_every: int,
        batch_size: int,
        min_crop_px: int,
        crop_pad: float = CROP_PAD,
    ):
        """Index the raw tracks of ``work``.

        Parameters
        ----------
        work : pandas.DataFrame
            Raw work table with ``raw_id``, ``frame``, ``x``, ``y``, ``w``, ``h``.
        occluded : pandas.Series
            True for rows whose crop is occluded (``primitives.occlusion_flags``), indexed
            like ``work``.
        video_file : path-like
            The video the track file belongs to.
        encoder : AppearanceEncoder
            Embeds RGB crops.
        store : FeatureStore
            Holds, and receives, the embeddings.
        sample_every : int
            Coarse stride, in observed frames.
        batch_size : int
            Crops per ``encoder.encode`` call.
        min_crop_px : int
            Rows whose box has a longer side (``max(w, h)``, in pixels) below this are treated
            like occluded rows: never cropped or embedded. ``0`` turns the rule off.
        crop_pad : float
            Box enlargement before cropping.

        """
        if not isinstance(occluded, pd.Series):
            raise ValueError(f"occluded must be a pandas Series, got {type(occluded).__name__}")
        if not work.index.is_unique or not occluded.index.is_unique:
            raise ValueError("work and occluded must each have a unique index")
        missing = work.index.difference(occluded.index)
        if len(missing):
            raise ValueError(
                f"occluded does not cover {len(missing)} row(s) of work (first label: {missing[0]})"
            )
        if int(min_crop_px) < 0:
            raise ValueError(f"min_crop_px must be at least 0, got {min_crop_px}")
        self.video_file = video_file
        self.encoder = encoder
        self.store = store
        self.batch_size = max(1, int(batch_size))
        self.crop_pad = float(crop_pad)
        self.min_crop_px = int(min_crop_px)
        self.coarse_clean = 0
        self.coarse_too_small = 0
        self._tr: dict[int, tuple[np.ndarray, ...]] = {}
        w = work.sort_values(["raw_id", "frame"])
        w = w.assign(_occ=occluded.loc[w.index].to_numpy(bool))
        every = max(1, int(sample_every))
        for raw_id, g in w.groupby("raw_id", sort=True):
            boxes = g[["x", "y", "w", "h"]].to_numpy(float)
            visible = ~g["_occ"].to_numpy(bool)
            side = boxes[:, 2:4].max(axis=1)  # the longer side, as in the track file
            clean = visible & _big_enough(side, self.min_crop_px)
            coarse = np.arange(len(g)) % every == 0  # ordinals count every observed frame
            self._tr[int(raw_id)] = (g["frame"].to_numpy(int), boxes, clean, coarse, visible, side)
        self.coarse_clean, self.coarse_too_small = self._coarse_counts(self.min_crop_px)

    def _coarse_counts(self, min_crop_px: int) -> tuple[int, int]:
        """Return the coarse samples that are clean, and those only too small, for a minimum."""
        clean = small = 0
        for _, _, _, coarse, visible, side in self._tr.values():
            big = _big_enough(side, min_crop_px)
            clean += int((coarse & visible & big).sum())
            small += int((coarse & visible & ~big).sum())
        return clean, small

    def view(self, min_crop_px: int) -> AppearanceView:
        """Return a view that also leaves out boxes whose longer side is below ``min_crop_px``.

        The view shares this provider's store and rows. It asks the provider for the same
        embeddings as the provider itself would, and filters them when it collects them, so
        what it returns does not depend on the order of calls.

        Raises
        ------
        ValueError
            If ``min_crop_px`` is below the provider's own ``min_crop_px``: those boxes were
            never embedded.

        """
        return AppearanceView(self, min_crop_px)

    def _rows(self, raw_id: int, f0: int, f1: int, dense: bool, min_crop_px: int = 0):
        tr = self._tr.get(int(raw_id))
        if tr is None:
            return None
        frames, boxes, clean, coarse, _, side = tr
        m = (frames >= f0) & (frames <= f1) & clean & _big_enough(side, min_crop_px)
        if not dense:
            m &= coarse
        return frames, boxes, np.flatnonzero(m)

    def _ensure(self, windows: Sequence[tuple[int, int, int]], dense: bool) -> None:
        need: dict[int, dict[tuple[int, int], np.ndarray]] = {}
        for raw_id, f0, f1 in windows:
            rows = self._rows(raw_id, f0, f1, dense)
            if rows is None:
                continue
            frames, boxes, idx = rows
            for i in idx:
                key = (int(raw_id), int(frames[i]))
                if self.store.has(*key) or self.store.is_skipped(*key):
                    continue
                need.setdefault(key[1], {})[key] = boxes[i]
        if not need:
            return
        crops: list[np.ndarray] = []
        owners: list[tuple[int, int]] = []
        with FrameReader(self.video_file) as reader:
            for f, img in reader.frames(need):
                for key in sorted(need[f]):
                    crop = crop_box(img, need[f][key], self.crop_pad)
                    if crop is None:  # recorded in the cache, so no rerun reads it again
                        self.store.skip(*key)
                        continue
                    crops.append(crop)
                    owners.append(key)
                if len(crops) >= self.batch_size * _FLUSH_BATCHES:
                    self._flush(crops, owners)
        self._flush(crops, owners)

    def _flush(self, crops: list[np.ndarray], owners: list[tuple[int, int]]) -> None:
        for i in range(0, len(crops), self.batch_size):
            chunk = crops[i : i + self.batch_size]
            emb = self.encoder.encode(chunk)
            if len(emb) != len(chunk):
                raise ValueError(f"encoder returned {len(emb)} embeddings for {len(chunk)} crops")
            bad = int((~np.isfinite(np.asarray(emb, dtype=np.float32)).all(axis=1)).sum())
            if bad:
                raise ValueError(
                    f"encoder returned {bad} non-finite embedding(s) for {len(chunk)} crops"
                )
            for (raw_id, frame), e in zip(owners[i : i + self.batch_size], emb, strict=True):
                self.store.put(raw_id, frame, e)
        crops.clear()
        owners.clear()

    def _collect(self, raw_id: int, f0: int, f1: int, dense: bool, min_crop_px: int = 0):
        rows = self._rows(raw_id, f0, f1, dense, min_crop_px)
        if rows is None:
            return np.empty(0, dtype=int), np.empty((0, 0))
        frames, _, idx = rows
        keep = [int(frames[i]) for i in idx if self.store.has(raw_id, int(frames[i]))]
        if not keep:
            return np.empty(0, dtype=int), np.empty((0, 0))
        return np.asarray(keep, dtype=int), self.store.get(raw_id, keep)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the coarse clean samples of ``raw_id`` within ``[f0, f1]``, sorted by frame."""
        self._ensure([(raw_id, f0, f1)], dense=False)
        return self._collect(raw_id, f0, f1, dense=False)

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every clean observed frame of ``raw_id`` within ``[f0, f1]``, sorted."""
        self._ensure([(raw_id, f0, f1)], dense=True)
        return self._collect(raw_id, f0, f1, dense=True)

    def prefetch_coarse(self) -> None:
        """Embed every coarse sample of every raw track in one pass over the video."""
        windows = [(rid, int(tr[0][0]), int(tr[0][-1])) for rid, tr in self._tr.items()]
        self._ensure(windows, dense=False)

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Embed the clean frames of ``(raw_id, f0, f1)`` windows in one pass over the video."""
        self._ensure(list(windows), dense=True)


def _big_enough(side: np.ndarray, min_crop_px: int) -> np.ndarray:
    """Return which boxes have a longer side of at least ``min_crop_px`` (all of them for 0)."""
    if min_crop_px <= 0:
        return np.ones(len(side), dtype=bool)
    return side >= min_crop_px


class AppearanceView:
    """A ``VideoAppearance`` seen by one stage, with that stage's minimum box size (spec 5.3).

    It implements ``clean_embeddings``, ``dense_embeddings`` and ``prefetch_dense`` over the
    provider's store and rows. Rows whose box has a longer side below ``min_crop_px`` are left
    out when the embeddings are collected; the coarse ordinals still count every observed row.
    Embedding is the provider's: a view never embeds a row the provider leaves out.
    """

    def __init__(self, provider: VideoAppearance, min_crop_px: int):
        """Wrap ``provider``; ``min_crop_px`` must be at least the provider's own value."""
        if int(min_crop_px) < provider.min_crop_px:
            raise ValueError(
                f"a view's min_crop_px ({min_crop_px}) cannot be below the provider's "
                f"({provider.min_crop_px}): smaller boxes were never embedded"
            )
        self.provider = provider
        self.min_crop_px = int(min_crop_px)
        self.coarse_clean, self.coarse_too_small = provider._coarse_counts(self.min_crop_px)

    def clean_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the coarse samples of ``raw_id`` within ``[f0, f1]`` that are large enough."""
        self.provider._ensure([(raw_id, f0, f1)], dense=False)
        return self.provider._collect(raw_id, f0, f1, dense=False, min_crop_px=self.min_crop_px)

    def dense_embeddings(self, raw_id: int, f0: int, f1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return every large-enough clean observed frame of ``raw_id`` within ``[f0, f1]``."""
        self.provider._ensure([(raw_id, f0, f1)], dense=True)
        return self.provider._collect(raw_id, f0, f1, dense=True, min_crop_px=self.min_crop_px)

    def prefetch_dense(self, windows: Sequence[tuple[int, int, int]]) -> None:
        """Embed the clean frames of ``(raw_id, f0, f1)`` windows in one pass (the provider's)."""
        self.provider.prefetch_dense(windows)
