import numpy as np
import pandas as pd
import pytest

from dnt.refine import io, video_appearance
from dnt.refine.features import FeatureStore
from dnt.refine.video_appearance import VideoAppearance

from ._fixtures import box_rows, table
from ._video import BLUE, RED, RED_DIR, ColorEncoder, make_color_video, video_rows


def _make(tmp_path, tracks, colors, *, occluded=None, every=5, batch=4, n_frames=60, enc=None,
          store=None):
    rows = [r for t in tracks for r in t]
    vrows = [v for t, c in zip(tracks, colors, strict=True) for v in video_rows(t, c)]
    video = make_color_video(tmp_path / "v.mp4", vrows, n_frames)
    work = io.to_work(table(*tracks)).work
    occ = pd.Series(False, index=work.index) if occluded is None else occluded(work)
    enc = enc or ColorEncoder()
    store = store or FeatureStore("k")
    app = VideoAppearance(work, occ, video, enc, store, sample_every=every, batch_size=batch)
    assert len(rows) == len(work)
    return app, enc, store, video


def _walker(track, frames, x0=100.0):
    return box_rows(track, frames, x0, 60.0, vx=2.0, w=40.0, h=80.0)


def test_coarse_samples_are_every_kth_observed_frame_with_the_right_color(tmp_path):
    app, enc, *_ = _make(tmp_path, [_walker(1, range(60))], [RED])
    f, e = app.clean_embeddings(1, 0, 59)
    assert list(f) == list(range(0, 60, 5)) and e.shape == (12, 3)
    assert (e @ RED_DIR).min() > 0.98
    assert enc.crops == 12


def test_ordinals_count_observed_frames_not_frame_numbers(tmp_path):
    frames = list(range(10)) + list(range(20, 30))
    app, *_ = _make(tmp_path, [_walker(1, frames)], [RED])
    f, _ = app.clean_embeddings(1, 0, 59)
    assert list(f) == [0, 5, 20, 25]


def test_dense_returns_every_clean_frame_and_reuses_coarse_samples(tmp_path):
    app, enc, *_ = _make(tmp_path, [_walker(1, range(60))], [RED])
    fc, ec = app.clean_embeddings(1, 10, 20)
    assert list(fc) == [10, 15, 20]
    before = enc.crops
    fd, ed = app.dense_embeddings(1, 10, 20)
    assert list(fd) == list(range(10, 21))
    assert enc.crops == before + 8  # only the 8 frames not already embedded
    assert np.array_equal(ed[[0, 5, 10]], ec)
    # the coarse view is unchanged by dense samples sitting in the store
    assert list(app.clean_embeddings(1, 10, 20)[0]) == [10, 15, 20]


def test_occluded_rows_are_not_embedded(tmp_path):
    app, *_ = _make(
        tmp_path, [_walker(1, range(60))], [RED], occluded=lambda w: w["frame"].between(20, 29)
    )
    f, _ = app.clean_embeddings(1, 0, 59)
    assert 20 not in f and 25 not in f and 30 in f and 15 in f
    f, _ = app.dense_embeddings(1, 15, 35)
    assert list(f) == [15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35]


def test_a_track_with_no_clean_crops_gives_empty_arrays(tmp_path):
    app, enc, *_ = _make(
        tmp_path, [_walker(1, range(30))], [RED], occluded=lambda w: w["frame"] >= 0
    )
    f, e = app.clean_embeddings(1, 0, 59)
    assert f.size == 0 and e.size == 0 and enc.calls == 0
    assert app.dense_embeddings(1, 0, 59)[0].size == 0
    assert app.clean_embeddings(99, 0, 59)[0].size == 0  # unknown raw id


def test_boxes_outside_the_frame_are_skipped_without_error(tmp_path):
    inside, outside = _walker(1, range(30)), _walker(2, range(30), x0=1000.0)
    app, enc, *_ = _make(tmp_path, [inside, outside], [RED, BLUE])
    assert app.clean_embeddings(2, 0, 29)[0].size == 0
    assert list(app.clean_embeddings(1, 0, 29)[0]) == [0, 5, 10, 15, 20, 25]
    assert enc.crops == 6
    assert app.dense_embeddings(2, 0, 29)[0].size == 0  # no retry storm on the unreadable ones
    assert enc.crops == 6


def test_samples_in_the_store_are_never_re_read(tmp_path, monkeypatch):
    app, enc, *_ = _make(tmp_path, [_walker(1, range(60))], [RED])
    first = app.clean_embeddings(1, 0, 59)
    opened = []
    real = video_appearance.FrameReader

    def counting(path):
        opened.append(path)
        return real(path)

    monkeypatch.setattr(video_appearance, "FrameReader", counting)
    calls = enc.calls
    again = app.clean_embeddings(1, 0, 59)
    assert opened == [] and enc.calls == calls
    assert np.array_equal(first[1], again[1])


def test_prefetch_coarse_reads_the_video_once_for_all_tracks(tmp_path, monkeypatch):
    tracks = [_walker(1, range(60)), _walker(2, range(10, 50), x0=200.0)]
    app, *_ = _make(tmp_path, tracks, [RED, BLUE])
    opened = []
    real = video_appearance.FrameReader
    monkeypatch.setattr(video_appearance, "FrameReader", lambda p: opened.append(p) or real(p))
    app.prefetch_coarse()
    assert len(opened) == 1
    app.clean_embeddings(1, 0, 59)
    app.clean_embeddings(2, 0, 59)
    assert len(opened) == 1


def test_prefetch_dense_embeds_overlapping_windows_once(tmp_path):
    app, enc, *_ = _make(tmp_path, [_walker(1, range(60))], [RED])
    app.prefetch_dense([(1, 10, 20), (1, 15, 25), (1, 10, 20)])
    assert enc.crops == 16  # frames 10..25 once each
    assert list(app.dense_embeddings(1, 10, 25)[0]) == list(range(10, 26))
    assert enc.crops == 16


def test_batches_are_bounded_and_batch_size_does_not_change_the_embeddings(tmp_path):
    tracks = [_walker(1, range(60))]
    small, enc_small, *_ = _make(tmp_path, tracks, [RED], batch=3, every=1)
    big, enc_big, *_ = _make(tmp_path, tracks, [RED], batch=64, every=1)
    a = small.clean_embeddings(1, 0, 59)[1]
    b = big.clean_embeddings(1, 0, 59)[1]
    assert enc_small.max_batch <= 3 and enc_big.max_batch <= 64 and enc_small.calls > enc_big.calls
    assert np.allclose(a, b, atol=1e-6)


def test_a_saved_store_serves_the_coarse_samples_without_the_video(tmp_path):
    app, _, store, _ = _make(tmp_path, [_walker(1, range(60))], [RED])
    first = app.clean_embeddings(1, 0, 59)
    path = tmp_path / "f.features.npz"
    store.save(path)
    loaded = FeatureStore.load(path, "k")
    work = io.to_work(table(_walker(1, range(60)))).work
    enc2 = ColorEncoder()
    app2 = VideoAppearance(
        work, pd.Series(False, index=work.index), tmp_path / "gone.mp4", enc2, loaded,
        sample_every=5, batch_size=4,
    )
    again = app2.clean_embeddings(1, 0, 59)
    assert enc2.calls == 0 and np.array_equal(first[1], again[1])
    with pytest.raises(ValueError, match="cannot open video"):
        app2.dense_embeddings(1, 0, 59)  # needs frames that were never embedded


def test_a_frame_past_the_end_of_the_video_names_the_frame(tmp_path):
    tracks = [_walker(1, [10, 500])]
    app, *_ = _make(tmp_path, tracks, [RED], n_frames=60, every=1)
    with pytest.raises(ValueError, match="frame 500"):
        app.clean_embeddings(1, 0, 600)


def test_ordinals_are_not_frame_numbers_when_the_gap_is_not_a_multiple_of_k(tmp_path):
    frames = list(range(10)) + list(range(23, 33))
    app, *_ = _make(tmp_path, [_walker(1, frames)], [RED])
    f, _ = app.clean_embeddings(1, 0, 59)
    assert list(f) == [0, 5, 23, 28]  # ordinals 0, 5, 10, 15; not the frames divisible by 5


def test_unreadable_crops_are_remembered_and_the_video_is_not_reopened(tmp_path, monkeypatch):
    app, enc, *_ = _make(tmp_path, [_walker(2, range(30), x0=1000.0)], [BLUE])
    assert app.clean_embeddings(2, 0, 29)[0].size == 0
    opened = []
    real = video_appearance.FrameReader
    monkeypatch.setattr(video_appearance, "FrameReader", lambda p: opened.append(p) or real(p))
    assert app.clean_embeddings(2, 0, 29)[0].size == 0
    assert opened == [] and enc.crops == 0


def test_crops_are_encoded_while_the_video_is_read_not_held_until_the_end(tmp_path, monkeypatch):
    seen = {"read": 0, "first_encode": None}
    real = video_appearance.FrameReader

    class Counting:
        def __init__(self, path):
            self._r = real(path)

        def __enter__(self):
            self._r.__enter__()
            return self

        def __exit__(self, *exc):
            return self._r.__exit__(*exc)

        def frames(self, wanted):
            for item in self._r.frames(wanted):
                seen["read"] += 1
                yield item

    class Recording(ColorEncoder):
        def encode(self, crops):
            if seen["first_encode"] is None:
                seen["first_encode"] = seen["read"]
            return super().encode(crops)

    monkeypatch.setattr(video_appearance, "FrameReader", Counting)
    app, *_ = _make(tmp_path, [_walker(1, range(60))], [RED], every=1, batch=1, enc=Recording())
    f, _ = app.clean_embeddings(1, 0, 59)
    assert f.size == 60
    assert seen["first_encode"] <= video_appearance._FLUSH_BATCHES  # not after all 60 frames


def test_crops_of_one_frame_are_encoded_in_raw_id_order(tmp_path):
    class Recording(ColorEncoder):
        def __init__(self):
            super().__init__()
            self.order = []

        def encode(self, crops):
            out = super().encode(crops)
            self.order += ["red" if e @ RED_DIR > 0.9 else "blue" for e in out]
            return out

    tracks = [_walker(1, range(4)), _walker(2, range(4), x0=200.0)]
    app, enc, *_ = _make(tmp_path, tracks, [RED, BLUE], batch=64, enc=Recording())
    app.prefetch_dense([(2, 0, 3), (1, 0, 3)])  # asked for in the opposite order
    assert enc.order == ["red", "blue"] * 4
