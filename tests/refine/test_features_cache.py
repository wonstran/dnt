import logging
from types import SimpleNamespace

import numpy as np
import pytest

from dnt.refine import features, io
from dnt.refine.features import (
    ArrayAppearance,
    CoarseArrayAppearance,
    FeatureStore,
    dense_track_embeddings,
    features_key,
    track_embeddings,
)


def _enc(**kw):
    base = dict(name="stub", model_name="m", weights_sha=None, preprocess_id="p1")
    return SimpleNamespace(**{**base, **kw})


BASE = dict(
    tracks_sha="t",
    video={"sha256": "v", "size": 10, "frame_count": 5},
    context_sha=None,
    encoder=_enc(),
    sample_every=5,
    occlusion_iou=0.3,
    crop_pad=1.1,
)


def test_key_is_a_stable_sha256():
    k = features_key(**BASE)
    assert k == features_key(**BASE) and len(k) == 64 and int(k, 16) >= 0


@pytest.mark.parametrize(
    "change",
    [
        {"tracks_sha": "t2"},
        {"video": {"sha256": "v2", "size": 10, "frame_count": 5}},
        {"video": {"sha256": "v", "size": 11, "frame_count": 5}},
        {"video": {"sha256": "v", "size": 10, "frame_count": 6}},
        {"context_sha": "c"},
        {"encoder": _enc(name="other")},
        {"encoder": _enc(model_name="m2")},
        {"encoder": _enc(weights_sha="w")},
        {"encoder": _enc(preprocess_id="p2")},
        {"sample_every": 4},
        {"occlusion_iou": 0.2},
        {"crop_pad": 1.2},
    ],
)
def test_key_changes_with_every_input(change):
    assert features_key(**{**BASE, **change}) != features_key(**BASE)


def test_key_changes_with_the_features_version(monkeypatch):
    before = features_key(**BASE)
    monkeypatch.setattr(features, "FEATURES_VERSION", features.FEATURES_VERSION + 1)
    assert features_key(**BASE) != before


def _store(key="k"):
    s = FeatureStore(key)
    s.put(2, 7, [0.0, 1.0])
    s.put(1, 3, [1.0, 0.0])
    s.put(1, 4, [0.6, 0.8])
    return s


def test_store_has_put_get_len_and_dirty():
    s = FeatureStore("k")
    assert not s.dirty and len(s) == 0 and not s.has(1, 3)
    s.put(1, 3, [1.0, 0.0])
    assert s.dirty and s.has(1, 3) and not s.has(1, 4) and not s.has(2, 3) and len(s) == 1
    s.put(1, 4, [0.0, 1.0])
    got = s.get(1, [4, 3])
    assert got.shape == (2, 2) and got.dtype == np.float32 and got[0, 1] == 1.0


def test_store_round_trips_and_is_byte_deterministic(tmp_path):
    a, b = tmp_path / "a.features.npz", tmp_path / "b.features.npz"
    sha_a = _store().save(a)
    other = FeatureStore("k")  # same content inserted in another order
    other.put(1, 4, [0.6, 0.8])
    other.put(2, 7, [0.0, 1.0])
    other.put(1, 3, [1.0, 0.0])
    assert other.save(b) == sha_a == io.sha256_file(a)
    assert a.read_bytes() == b.read_bytes()
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = FeatureStore.load(a, "k")
    assert loaded is not None and not loaded.dirty and len(loaded) == 3
    assert np.allclose(loaded.get(1, [3, 4]), [[1.0, 0.0], [0.6, 0.8]])
    assert loaded.has(2, 7)


def test_an_empty_store_round_trips(tmp_path):
    p = tmp_path / "e.features.npz"
    FeatureStore("k").save(p)
    loaded = FeatureStore.load(p, "k")
    assert loaded is not None and len(loaded) == 0


def test_a_different_key_is_a_miss_logged_at_info(tmp_path, caplog):
    p = tmp_path / "f.features.npz"
    _store("k").save(p)
    with caplog.at_level(logging.INFO, logger="dnt.refine.features"):
        assert FeatureStore.load(p, "other") is None
    assert "different inputs" in caplog.text


def test_missing_garbage_and_truncated_files_are_misses(tmp_path):
    assert FeatureStore.load(tmp_path / "none.npz", "k") is None
    bad = tmp_path / "bad.features.npz"
    bad.write_bytes(b"not a zip at all")
    assert FeatureStore.load(bad, "k") is None
    good = tmp_path / "good.features.npz"
    _store().save(good)
    cut = tmp_path / "cut.features.npz"
    cut.write_bytes(good.read_bytes()[:60])
    assert FeatureStore.load(cut, "k") is None
    empty = tmp_path / "empty.features.npz"
    empty.write_bytes(b"")
    assert FeatureStore.load(empty, "k") is None


def _write_npz(path, **over):
    arrays = {
        "key": np.array("k"),
        "raw_id": np.array([1, 1], dtype=np.int64),
        "frame": np.array([3, 4], dtype=np.int64),
        "emb": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    arrays.update(over)
    np.savez(path, **arrays)


def test_the_helper_writes_a_loadable_archive(tmp_path):
    p = tmp_path / "ok.features.npz"
    _write_npz(p)
    loaded = FeatureStore.load(p, "k")
    assert loaded is not None and len(loaded) == 2 and loaded.has(1, 4)


@pytest.mark.parametrize(
    "over",
    [
        {"emb": np.array([1.0, 0.0], dtype=np.float32)},  # 1-D embeddings
        {"emb": np.array([["a", "b"], ["c", "d"]])},  # non-numeric
        {"emb": np.array([[1.0, 0.0]], dtype=np.float32)},  # wrong number of rows
        {"emb": np.zeros((2, 0), dtype=np.float32)},  # zero width with rows
        {"emb": np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)},  # not finite
        {"emb": np.array([[1.0, np.inf], [0.0, 1.0]], dtype=np.float32)},
        {"raw_id": np.array(1, dtype=np.int64)},  # 0-D
        {"frame": np.array([[3, 4]], dtype=np.int64)},  # 2-D
        {"frame": np.array([3.0, 4.0])},  # float frames
        {"raw_id": np.array(["a", "b"])},  # non-numeric ids
        {"frame": np.array([3, 3], dtype=np.int64)},  # duplicate (raw_id, frame)
        {"key": np.array(["k", "k"])},  # key is not a scalar string
        {"key": np.array(3)},
    ],
)
def test_readable_but_malformed_archives_are_misses(tmp_path, over):
    p = tmp_path / "bad.features.npz"
    _write_npz(p, **over)
    assert FeatureStore.load(p, "k") is None


def test_an_archive_with_a_missing_array_is_a_miss(tmp_path):
    p = tmp_path / "part.features.npz"
    np.savez(p, key=np.array("k"), raw_id=np.array([1]), frame=np.array([3]))
    assert FeatureStore.load(p, "k") is None


def test_an_archive_of_the_wrong_embedding_width_is_a_miss_for_that_encoder(tmp_path):
    p = tmp_path / "w.features.npz"
    _write_npz(p)  # width 2
    assert FeatureStore.load(p, "k", dim=3) is None
    assert FeatureStore.load(p, "k", dim=2) is not None
    assert FeatureStore.load(p, "k") is not None  # no encoder to compare with


def test_values_that_overflow_float32_are_a_miss(tmp_path):
    p = tmp_path / "big.features.npz"
    _write_npz(p, emb=np.array([[1e300, 0.0], [0.0, 1.0]]))  # finite as float64, inf as float32
    assert FeatureStore.load(p, "k", dim=2) is None
    ok = tmp_path / "f64.features.npz"
    _write_npz(ok, emb=np.array([[0.5, 0.25], [0.0, 1.0]]))  # float64 is fine when it converts
    loaded = FeatureStore.load(ok, "k", dim=2)
    assert loaded is not None and loaded.get(1, [3]).dtype == np.float32


def test_put_rejects_an_embedding_of_another_width():
    s = FeatureStore("k")
    s.put(1, 3, [1.0, 0.0])
    with pytest.raises(ValueError, match="width"):
        s.put(1, 4, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="width"):
        s.put(1, 5, [[1.0, 0.0]])  # not a vector


def _table():
    frames = list(range(20))
    emb = np.eye(4)[np.arange(20) % 4]
    return {1: (frames, emb)}


def test_coarse_provider_returns_every_kth_sample_and_dense_all():
    app = CoarseArrayAppearance(_table(), every=5)
    f, e = app.clean_embeddings(1, 0, 19)
    assert list(f) == [0, 5, 10, 15] and e.shape == (4, 4)
    f, e = app.clean_embeddings(1, 3, 12)
    assert list(f) == [5, 10]
    f, e = app.dense_embeddings(1, 3, 12)
    assert list(f) == list(range(3, 13))
    assert app.prefetch_dense([(1, 0, 5)]) is None
    assert app.clean_embeddings(9, 0, 5)[0].size == 0 and app.dense_embeddings(9, 0, 5)[0].size == 0
    with pytest.raises(ValueError, match="every"):
        CoarseArrayAppearance(_table(), every=0)


def test_plain_array_provider_has_no_dense_methods():
    assert not hasattr(ArrayAppearance(_table()), "dense_embeddings")


def test_dense_track_embeddings_follow_the_lineage_and_clip_to_the_window():
    app = CoarseArrayAppearance({1: (range(10), np.eye(3)[np.arange(10) % 3]),
                                 2: (range(10, 20), np.eye(3)[np.arange(10) % 3])}, every=5)
    lineage = [[1, 0, 9], [2, 10, 19]]
    f, e = dense_track_embeddings(app, lineage, 7, 12)
    assert list(f) == [7, 8, 9, 10, 11, 12] and e.shape == (6, 3)
    f, e = dense_track_embeddings(app, lineage, 30, 40)
    assert f.size == 0 and e.size == 0
    f, _ = track_embeddings(app, lineage)  # coarse only: ordinals 0 and 5 of each raw track
    assert list(f) == [0, 5, 10, 15]


def test_a_failed_save_leaves_the_previous_cache_intact(tmp_path, monkeypatch):
    p = tmp_path / "keep.features.npz"
    sha = _store().save(p)

    def _boom(fh, **arrays):
        fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", _boom)
    with pytest.raises(OSError, match="disk full"):
        _store().save(p)
    assert io.sha256_file(p) == sha
    monkeypatch.undo()
    loaded = FeatureStore.load(p, "k")
    assert loaded is not None and len(loaded) == 3
