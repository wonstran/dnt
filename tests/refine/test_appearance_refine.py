import sys

import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.features import FeatureStore
from dnt.refine.refiner import TrackRefiner

from ._fakes import install_fake_torchreid, install_fake_transformers
from ._fixtures import box_rows, table
from ._video import BLUE, RED, ColorEncoder, make_color_video, video_rows


def _scene(tmp_path, n_frames=120, name="t.txt", video="v.mp4"):
    """One track: a red object for 60 frames, then a bigger blue one takes the ID over."""
    red = box_rows(1, range(60), 20.0, 100.0, vx=2.0, w=20.0, h=40.0)
    blue = box_rows(1, range(60, 120), 140.0, 100.0, vx=2.0, w=35.0, h=40.0)
    vid = make_color_video(
        tmp_path / video, video_rows(red, RED) + video_rows(blue, BLUE), n_frames
    )
    src = tmp_path / name
    table(red + blue).to_csv(src, index=False, header=False)
    return src, vid


def _cfg(**encoder):
    cfg = RefineConfig.defaults()
    cfg.link.enabled = False  # keep the two pieces apart: these tests are about stage 1
    for k, v in encoder.items():
        setattr(cfg.encoder, k, v)
    return cfg


def _run(src, out, video, enc, cfg=None, context=None):
    refiner = TrackRefiner(cfg or _cfg(), encoder_factory=lambda c, t: enc)
    refiner.refine(src, out, video_file=video, context_file=context, verbose=False)
    return refiner.last_result


def _header(res):
    return Ledger.read(res.ledger_path).header


def test_a_takeover_is_split_at_the_right_frame_from_appearance(tmp_path):
    src, video = _scene(tmp_path)
    res = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    splits = [e for e in res.events if e.kind is EventKind.SPLIT]
    assert len(splits) == 1
    ev = splits[0]
    assert ev.params["cut_frame"] == 60 and ev.signals["motion_only"] is False
    assert ev.algo_score >= RefineConfig.defaults().switch.accept_above
    assert ev.decision is Decision.AUTO_ACCEPT and ev.applied
    spans = res.tracks.groupby("track")["frame"].agg(["min", "max"]).to_numpy().tolist()
    assert spans == [[0, 59], [60, 119]]


def test_the_cache_is_written_recorded_and_reused(tmp_path):
    src, video = _scene(tmp_path)
    enc1 = ColorEncoder()
    r1 = _run(src, tmp_path / "o.txt", video, enc1)
    feats = tmp_path / "o.features.npz"
    rec = _header(r1)["inputs"]["features"]
    assert enc1.crops > 0 and feats.is_file()
    assert rec["sha256"] == io.sha256_file(feats) and len(rec["cache_key"]) == 64
    assert rec["path"] == str(feats)
    first_bytes = feats.read_bytes()

    enc2 = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video, enc2)
    assert enc2.calls == 0
    assert r2.tracks.equals(r1.tracks)
    assert _header(r2)["inputs"]["features"] == rec
    assert feats.read_bytes() == first_bytes
    assert [(e.id, e.kind, e.decision, e.algo_score) for e in r2.events] == [
        (e.id, e.kind, e.decision, e.algo_score) for e in r1.events
    ]


def test_a_different_video_is_a_cache_miss(tmp_path):
    src, video = _scene(tmp_path)
    r1 = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    key1 = _header(r1)["inputs"]["features"]["cache_key"]  # the next run rewrites this ledger
    _, video2 = _scene(tmp_path, n_frames=121, video="v2.mp4")
    enc = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video2, enc)
    assert enc.crops > 0
    assert _header(r2)["inputs"]["features"]["cache_key"] != key1


def test_a_different_encoder_setting_is_a_cache_miss(tmp_path):
    src, video = _scene(tmp_path)
    _run(src, tmp_path / "o.txt", video, ColorEncoder())
    enc = ColorEncoder()
    _run(src, tmp_path / "o.txt", video, enc, cfg=_cfg(sample_every=4))
    assert enc.crops > 0


def test_a_damaged_cache_is_rebuilt_not_fatal(tmp_path):
    src, video = _scene(tmp_path)
    feats = tmp_path / "o.features.npz"
    feats.write_bytes(b"garbage")
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc)
    assert enc.crops > 0 and len(res.tracks)
    key = _header(res)["inputs"]["features"]["cache_key"]
    assert FeatureStore.load(feats, key) is not None


def test_a_cache_with_the_right_key_but_the_wrong_width_is_rebuilt(tmp_path):
    import numpy as np

    src, video = _scene(tmp_path)
    r1 = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    rec = _header(r1)["inputs"]["features"]
    feats = tmp_path / "o.features.npz"
    store = FeatureStore.load(feats, rec["cache_key"])
    assert store is not None and len(store) > 0
    # same key, readable, but 5 columns where this encoder writes 3
    np.savez(
        feats,
        key=np.array(rec["cache_key"]),
        raw_id=np.array([1, 1], dtype=np.int64),
        frame=np.array([0, 5], dtype=np.int64),
        emb=np.ones((2, 5), dtype=np.float32),
    )
    enc = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video, enc)
    assert enc.crops > 0 and r2.tracks.equals(r1.tracks)
    assert FeatureStore.load(feats, rec["cache_key"], dim=3) is not None


def test_a_replaced_weights_file_reloads_the_encoder_and_misses_the_cache(tmp_path, monkeypatch):
    seen = install_fake_torchreid(monkeypatch)
    src, video = _scene(tmp_path)
    weights = tmp_path / "osnet.pt"
    weights.write_bytes(b"v1")
    refiner = TrackRefiner(_cfg(kind="reid", weights=str(weights), device="cpu"))

    def run():
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
        return _header(refiner.last_result)["inputs"]["features"]["cache_key"]

    key1 = run()
    assert seen["loads"] == 1
    assert run() == key1 and seen["loads"] == 1  # same weights: model and cache are reused
    weights.write_bytes(b"v2")  # replaced in place, same path, same refiner
    key3 = run()
    assert seen["loads"] == 2 and key3 != key1


def test_the_same_hub_model_name_with_new_weights_misses_the_cache(tmp_path, monkeypatch):
    seen = install_fake_transformers(monkeypatch)
    src, video = _scene(tmp_path)

    def run():
        refiner = TrackRefiner(_cfg(kind="dino", device="cpu"))  # a new refiner loads the model
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
        return _header(refiner.last_result)["inputs"]["features"]["cache_key"]

    key1 = run()
    assert run() == key1  # the name still resolves to the same weights: the cache is valid
    seen["seed"] = 1  # facebook/dinov2-small now resolves to other weights
    assert run() != key1


def test_a_detection_file_of_the_same_run_does_not_make_every_crop_occluded(tmp_path):
    src, video = _scene(tmp_path)
    raw = pd.read_csv(src, header=None)
    det = pd.DataFrame(
        {"f": raw[0], "res": -1, "x": raw[2], "y": raw[3], "w": raw[4], "h": raw[5],
         "conf": 0.9, "cls": 0}
    )
    ctx = tmp_path / "t_iou.txt"
    det.to_csv(ctx, index=False, header=False)
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc, context=ctx)
    assert enc.crops > 0
    assert [e.params["cut_frame"] for e in res.events if e.kind is EventKind.SPLIT] == [60]


def test_a_track_with_no_clean_crops_is_handled(tmp_path):
    rows = box_rows(1, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0) + box_rows(
        2, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0
    )
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 60)
    src = tmp_path / "t.txt"
    table(rows).to_csv(src, index=False, header=False)
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc)  # both boxes always overlap each other
    assert enc.calls == 0 and len(res.tracks) > 0


def test_kind_none_with_a_video_is_motion_only_and_writes_no_cache(tmp_path):
    src, video = _scene(tmp_path)
    enc = ColorEncoder()
    with_video = _run(src, tmp_path / "o.txt", video, enc, cfg=_cfg(kind="none"))
    assert enc.calls == 0 and _header(with_video)["inputs"]["features"] is None
    assert not (tmp_path / "o.features.npz").exists()
    refiner = TrackRefiner(_cfg(kind="none"))
    no_video = refiner.refine(src, tmp_path / "p.txt", fps=10, verbose=False)
    assert no_video.equals(with_video.tracks)


def test_a_missing_package_fails_before_any_file_is_written(tmp_path, monkeypatch):
    src, video = _scene(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", None)
    refiner = TrackRefiner(_cfg())  # kind "dino", no encoder_factory
    with pytest.raises(ImportError, match=r"dnt\[refine-dino\]"):
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]
    # the same call without a video does not need the package
    refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)


def test_refine_batch_builds_the_encoder_once(tmp_path):
    srcs, videos = [], []
    for name in ("a", "b"):
        s, v = _scene(tmp_path, name=f"{name}_track.txt", video=f"{name}.mp4")
        srcs.append(s)
        videos.append(v)
    built = []
    enc = ColorEncoder()

    def factory(cfg, target):
        built.append(target)
        return enc

    refiner = TrackRefiner(_cfg(), encoder_factory=factory)
    outs = refiner.refine_batch(srcs, video_files=videos, output_path=tmp_path / "out", verbose=False)
    assert len(outs) == 2 and built == ["person"]
    assert all((tmp_path / "out" / f"{n}_refined.features.npz").is_file() for n in ("a", "b"))


def test_an_empty_valid_cache_is_still_a_cache_hit_and_is_recorded(tmp_path, monkeypatch):
    from dnt.refine import video_appearance

    rows = box_rows(1, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0) + box_rows(
        2, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0
    )
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 60)
    src = tmp_path / "t.txt"
    table(rows).to_csv(src, index=False, header=False)
    feats = tmp_path / "o.features.npz"
    r1 = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    rec = _header(r1)["inputs"]["features"]  # an empty store is still saved and recorded
    assert feats.is_file() and rec["sha256"] == io.sha256_file(feats)
    assert len(FeatureStore.load(feats, rec["cache_key"])) == 0

    opened, loaded = [], []
    real_reader, real_load = video_appearance.FrameReader, FeatureStore.load

    def spy_reader(*a, **k):
        opened.append(a)
        return real_reader(*a, **k)

    def spy_load(path, key, dim=None):
        loaded.append(real_load(path, key, dim))
        return loaded[-1]

    monkeypatch.setattr(video_appearance, "FrameReader", spy_reader)
    monkeypatch.setattr(FeatureStore, "load", staticmethod(spy_load))
    enc = ColorEncoder()
    r2 = _run(src, tmp_path / "o.txt", video, enc)
    assert enc.calls == 0 and not opened
    assert len(loaded) == 1 and loaded[0] is not None and len(loaded[0]) == 0
    assert _header(r2)["inputs"]["features"] == rec


def test_a_failed_cache_save_leaves_no_ledger_and_no_output(tmp_path, monkeypatch):
    src, video = _scene(tmp_path)

    def boom(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(FeatureStore, "save", boom)
    with pytest.raises(OSError, match="disk full"):
        _run(src, tmp_path / "o.txt", video, ColorEncoder())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]
