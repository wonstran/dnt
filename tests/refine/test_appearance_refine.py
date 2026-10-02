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


def test_a_failing_stage_keeps_the_embeddings_but_writes_no_ledger_or_output(
    tmp_path, monkeypatch
):
    import numpy as np

    from dnt.refine.refiner import _Stages

    src, video = _scene(tmp_path)

    def crash(self, *args, **kwargs):
        raise RuntimeError("stage 2 crashed")

    monkeypatch.setattr(_Stages, "_screen", crash)  # after stage 1 used the embeddings
    enc = ColorEncoder()
    with pytest.raises(RuntimeError, match="stage 2 crashed"):
        _run(src, tmp_path / "o.txt", video, enc)
    assert enc.crops > 0
    feats = tmp_path / "o.features.npz"
    assert feats.is_file()
    assert not (tmp_path / "o.txt").exists() and not (tmp_path / "o.ledger.jsonl").exists()
    with np.load(feats) as z:
        key = str(z["key"])
    store = FeatureStore.load(feats, key, dim=3)
    assert store is not None and len(store) == enc.crops
    monkeypatch.undo()  # the next run succeeds and encodes nothing again
    again = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, again)
    assert again.calls == 0 and _header(res)["inputs"]["features"]["cache_key"] == key


def test_a_crash_in_the_coarse_pass_keeps_the_embeddings_encoded_before_it(tmp_path):
    import numpy as np

    class FailsOnSecondCall(ColorEncoder):
        def encode(self, crops):
            if self.calls == 1:
                raise RuntimeError("encoder crashed")
            return super().encode(crops)

    src, video = _scene(tmp_path)
    enc = FailsOnSecondCall()
    with pytest.raises(RuntimeError, match="encoder crashed"):
        _run(src, tmp_path / "o.txt", video, enc, cfg=_cfg(batch_size=1))
    feats = tmp_path / "o.features.npz"
    with np.load(feats) as z:
        key = str(z["key"])
    store = FeatureStore.load(feats, key, dim=3)
    assert store is not None and len(store) == enc.crops == 1
    assert not (tmp_path / "o.ledger.jsonl").exists()


def test_a_failing_save_after_a_crash_does_not_hide_the_crash(tmp_path, monkeypatch, caplog):
    from dnt.refine.refiner import _Stages

    src, video = _scene(tmp_path)

    def crash(self, *args, **kwargs):
        raise RuntimeError("stage 2 crashed")

    def no_save(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(_Stages, "_screen", crash)
    monkeypatch.setattr(FeatureStore, "save", no_save)
    with pytest.raises(RuntimeError, match="stage 2 crashed"):
        _run(src, tmp_path / "o.txt", video, ColorEncoder())
    assert "disk full" in caplog.text


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


def test_an_empty_valid_cache_is_saved_recorded_and_loadable(tmp_path):
    """Pin what an empty cache can prove.

    The scene has no clean crops, so the encoder and the video are untouched with or without
    a cache hit; a hit cannot be told from a miss here. What can be shown is that an empty
    store is still saved, recorded in the header, and loads as a valid store with that key.
    """
    rows = box_rows(1, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0) + box_rows(
        2, range(60), 100.0, 100.0, vx=1.0, w=30.0, h=60.0
    )
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 60)
    src = tmp_path / "t.txt"
    table(rows).to_csv(src, index=False, header=False)
    feats = tmp_path / "o.features.npz"
    enc = ColorEncoder()
    res = _run(src, tmp_path / "o.txt", video, enc)
    rec = _header(res)["inputs"]["features"]
    assert enc.calls == 0 and feats.is_file()
    assert rec["sha256"] == io.sha256_file(feats) and rec["path"] == str(feats)
    store = FeatureStore.load(feats, rec["cache_key"], dim=enc.dim)
    assert store is not None and store.key == rec["cache_key"] and len(store) == 0
    assert FeatureStore.load(feats, "0" * 64) is None


def _stored_frames(feats, raw_id=1):
    import numpy as np

    with np.load(feats) as z:
        return set(z["frame"][z["raw_id"] == raw_id].tolist())


def _overlap_context(tmp_path, src, frames, name="ctx.txt"):
    """A detection file of another object: shifted 8 px from the track's box, IoU about 0.43."""
    raw = pd.read_csv(src, header=None)
    raw = raw[raw[0].isin(list(frames))]
    det = pd.DataFrame(
        {"f": raw[0], "res": -1, "x": raw[2] + 8.0, "y": raw[3], "w": raw[4], "h": raw[5],
         "conf": 0.9, "cls": 0}
    )
    ctx = tmp_path / name
    det.to_csv(ctx, index=False, header=False)
    return ctx


def test_context_boxes_occlude_crops_and_the_context_changes_the_cache_key(tmp_path):
    src, video = _scene(tmp_path)
    ctx = _overlap_context(tmp_path, src, range(30, 60))
    with_ctx = _run(src, tmp_path / "a.txt", video, ColorEncoder(), context=ctx)
    frames = _stored_frames(tmp_path / "a.features.npz")
    assert any(f < 30 for f in frames) and any(f >= 60 for f in frames)
    assert not any(30 <= f < 60 for f in frames)  # occluded by the other object

    without = _run(src, tmp_path / "b.txt", video, ColorEncoder())
    assert any(30 <= f < 60 for f in _stored_frames(tmp_path / "b.features.npz"))
    key_ctx = _header(with_ctx)["inputs"]["features"]["cache_key"]
    assert _header(without)["inputs"]["features"]["cache_key"] != key_ctx


def test_the_occlusion_threshold_is_part_of_the_cache_key(tmp_path):
    src, video = _scene(tmp_path)
    r1 = _run(src, tmp_path / "o.txt", video, ColorEncoder())
    key1 = _header(r1)["inputs"]["features"]["cache_key"]
    cfg = _cfg()
    cfg.encoder.occlusion_iou = 0.5
    r2 = _run(src, tmp_path / "o.txt", video, ColorEncoder(), cfg=cfg)
    assert _header(r2)["inputs"]["features"]["cache_key"] != key1


def test_a_changed_device_rebuilds_the_encoder_but_the_same_settings_do_not(tmp_path):
    src, video = _scene(tmp_path)
    built = []

    def factory(cfg, target):
        built.append(cfg.device)
        return ColorEncoder()

    refiner = TrackRefiner(_cfg(device="cpu"), encoder_factory=factory)

    def run():
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)

    run()
    run()
    assert built == ["cpu"]
    refiner.config.encoder.device = "cuda"
    run()
    assert built == ["cpu", "cuda"]


def test_a_failed_cache_save_leaves_no_ledger_and_no_output(tmp_path, monkeypatch):
    src, video = _scene(tmp_path)

    def boom(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(FeatureStore, "save", boom)
    with pytest.raises(OSError, match="disk full"):
        _run(src, tmp_path / "o.txt", video, ColorEncoder())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]
