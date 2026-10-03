import json
import logging
import sys
import threading

import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.refiner import TrackRefiner
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import VLMRunner

from ._fixtures import box_rows, table
from ._video import RED, make_color_video, takeover_scene, video_rows


def reply(answer, conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf, "reason": "ok"})


def cfg_for(tmp_path, **vlm):
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"  # motion-only: a takeover split is capped at 0.70 (uncertain band)
    cfg.link.enabled = False
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "m"
    cfg.vlm.cache_dir = str(tmp_path / "vlmcache")
    for k, v in vlm.items():
        setattr(cfg.vlm, k, v)
    return cfg


def run(tmp_path, cfg, backend, src, video, out="o.txt", **kw):
    refiner = TrackRefiner(cfg, vlm_backend_factory=lambda c: backend)
    refiner.refine(src, tmp_path / out, video_file=video, verbose=False, **kw)
    return refiner.last_result


def splits(res):
    return [e for e in res.events if e.kind is EventKind.SPLIT]


def test_a_sure_answer_decides_and_applies_the_split(tmp_path):
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": reply("different")})
    res = run(tmp_path, cfg_for(tmp_path), backend, src, video)
    (ev,) = splits(res)
    assert ev.decision is Decision.VLM_ACCEPT and ev.applied and ev.params["cut_frame"] == 60
    assert ev.vlm["answer"] == "different" and ev.vlm["backend"] == "fake"
    assert res.tracks.track.nunique() == 2
    assert res.summary["vlm"] == {
        "calls": 1, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0,
        "no_evidence": 0,
    }
    assert backend.calls[0]["tag"] == f"SPLIT:{ev.id}"
    led = Ledger.read(res.ledger_path)
    (back,) = [e for e in led.events if e.kind is EventKind.SPLIT]
    assert back.decision is Decision.VLM_ACCEPT and back.vlm == ev.vlm
    assert led.header["summary"]["vlm"]["calls"] == 1


def test_a_rejecting_answer_keeps_the_track_whole(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("same_individual")}),
              src, video)
    (ev,) = splits(res)
    assert ev.decision is Decision.VLM_REJECT and not ev.applied
    assert res.tracks.track.nunique() == 1


def test_an_unsure_or_failed_answer_leaves_it_pending_and_the_run_completes(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    assert splits(res)[0].decision is Decision.HUMAN_PENDING and res.tracks.track.nunique() == 1
    res2 = run(tmp_path, cfg_for(tmp_path, cache_dir=str(tmp_path / "other")),
               FakeBackend({"SPLIT": [RuntimeError("boom")]}), src, video, out="o2.txt")
    ev = splits(res2)[0]
    assert ev.decision is Decision.HUMAN_PENDING and ev.vlm["error"].startswith("RuntimeError")
    assert res2.summary["vlm"]["failures"] == 1


def test_the_budget_is_per_run_and_counted(tmp_path):
    src, video = takeover_scene(tmp_path, n_tracks=2)
    backend = FakeBackend({"SPLIT": reply("different")})
    res = run(tmp_path, cfg_for(tmp_path, max_calls=1), backend, src, video)
    decided = [e for e in splits(res) if e.decision is Decision.VLM_ACCEPT]
    left = [e for e in splits(res) if e.decision is Decision.HUMAN_PENDING]
    assert len(decided) == 1 and len(left) == 1 and left[0].vlm["error"] == "budget"
    assert res.summary["vlm"]["calls"] == 1 and res.summary["vlm"]["budget_skipped"] == 1


def test_a_retry_is_reported_in_the_result_the_ledger_and_the_budget(tmp_path):
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": ["not json", reply("different")]})  # an invalid reply, no sleep
    res = run(tmp_path, cfg_for(tmp_path, max_calls=2), backend, src, video)
    assert res.summary["vlm"] == {
        "calls": 1, "retries": 1, "cache_hits": 0, "failures": 0, "budget_skipped": 0,
        "no_evidence": 0,
    }
    assert len(backend.calls) == 2
    assert Ledger.read(res.ledger_path).header["summary"]["vlm"]["retries"] == 1
    # max_calls is a hard limit on invocations: with one unit the retry is not allowed
    tight = FakeBackend({"SPLIT": ["not json", reply("different")]})
    res2 = run(tmp_path, cfg_for(tmp_path, max_calls=1, cache_dir=str(tmp_path / "c2")), tight,
               src, video, out="o2.txt")
    assert len(tight.calls) == 1 and res2.summary["vlm"]["retries"] == 0
    assert splits(res2)[0].decision is Decision.HUMAN_PENDING
    assert splits(res2)[0].vlm["error"] == "budget" and res2.summary["vlm"]["budget_skipped"] == 1


def test_a_rerun_with_the_same_cache_makes_no_backend_calls(tmp_path):
    src, video = takeover_scene(tmp_path)
    cfg = cfg_for(tmp_path)
    run(tmp_path, cfg, FakeBackend({"SPLIT": reply("different")}), src, video)
    again = FakeBackend({})  # asking would raise: the run must be served from the cache
    res = run(tmp_path, cfg, again, src, video, out="o2.txt")
    (ev,) = splits(res)
    assert again.calls == [] and ev.decision is Decision.VLM_ACCEPT and ev.vlm["cached"] is True
    assert res.summary["vlm"] == {
        "calls": 0, "retries": 0, "cache_hits": 1, "failures": 0, "budget_skipped": 0,
        "no_evidence": 0,
    }


def test_context_frames_can_be_left_out_of_the_image(tmp_path):
    src, video = takeover_scene(tmp_path)
    with_ctx = FakeBackend({"SPLIT": reply("different")})
    run(tmp_path, cfg_for(tmp_path), with_ctx, src, video)
    without = FakeBackend({"SPLIT": reply("different")})
    run(tmp_path, cfg_for(tmp_path, send_context_frames=False, cache_dir=str(tmp_path / "c2")),
        without, src, video, out="o2.txt")
    assert without.calls[0]["image_len"] < with_ctx.calls[0]["image_len"]


def test_a_screen_event_can_be_redirected_to_a_reclass(tmp_path):
    # a motionless low-confidence box: its static score (about 0.61) is in the uncertain band
    rows = box_rows(1, range(120), 100.0, 80.0, w=40.0, h=80.0, score=0.35)
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 120)
    src = tmp_path / "t.txt"
    table(rows).to_csv(src, index=False, header=False)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"DROP": reply("cyclist")}), src, video)
    (ev,) = [e for e in res.events if e.stage == "screen"]
    assert ev.kind is EventKind.DROP and ev.decision is Decision.VLM_ACCEPT and ev.applied
    assert ev.edit["kind"] == "RECLASS" and ev.edit["params"]["new_cls"] == 1
    assert set(res.tracks["cls"]) == {1}


def test_an_always_overlapping_screen_event_reaches_the_vlm(tmp_path):
    # a person who is inside a context car for the whole track: every row is occluded, yet the
    # screen stage must still show the observed crops (the in-vehicle hypothesis relies on them)
    rows = box_rows(1, range(120), 100.0, 80.0, vx=2.0, w=20.0, h=40.0, score=0.6)
    # the car holds the person whole (IoB 1) at IoU 0.4: an occluder (>= encoder.occlusion_iou
    # 0.3), not the row's own detection (a matched IoU >= 0.5 is dropped as a duplicate)
    ctx = box_rows(9, range(120), 90.0, 75.0, vx=2.0, w=40.0, h=50.0, cls=2)
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 120)
    src, ctx_file = tmp_path / "t.txt", tmp_path / "c.txt"
    table(rows).to_csv(src, index=False, header=False)
    table(ctx).to_csv(ctx_file, index=False, header=False)
    cfg = cfg_for(tmp_path)
    # inside_frac is 1.0, which the default ramp scores 1.0 (>= accept_above, so AUTO_ACCEPT);
    # a wider ramp scores it 0.5, inside the band, so the VLM is asked
    cfg.screen.ramps["inside"] = [0.5, 1.5]
    backend = FakeBackend({"DROP": reply("person_in_vehicle")})
    refiner = TrackRefiner(cfg, vlm_backend_factory=lambda c: backend)
    refiner.refine(src, tmp_path / "o.txt", video_file=video, context_file=ctx_file,
                   verbose=False)
    res = refiner.last_result
    (ev,) = [e for e in res.events if e.stage == "screen"]
    assert len(backend.calls) == 1 and backend.calls[0]["image_len"] > 0
    assert ev.decision is Decision.VLM_ACCEPT and ev.applied
    assert ev.edit["kind"] == "DROP" and ev.edit["params"]["reason"] == "in_vehicle"
    assert res.tracks.empty


def test_the_backend_is_closed_after_a_run_and_after_a_failing_one(tmp_path, monkeypatch):
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": reply("different")})
    run(tmp_path, cfg_for(tmp_path), backend, src, video)
    assert backend.closed == 1
    assert not [t for t in threading.enumerate() if t.name == "dnt-vlm-loop"]
    real = VLMRunner.ask_many

    def ask_then_fail(self, questions):
        real(self, questions)  # the loop is started and the client has been used
        raise RuntimeError("stage failed")

    monkeypatch.setattr(VLMRunner, "ask_many", ask_then_fail)
    failing = FakeBackend({"SPLIT": reply("different")})
    with pytest.raises(RuntimeError, match="stage failed"):
        run(tmp_path, cfg_for(tmp_path, cache_dir=str(tmp_path / "c2")), failing, src, video,
            out="o2.txt")
    assert failing.closed == 1
    assert not [t for t in threading.enumerate() if t.name == "dnt-vlm-loop"]


def test_without_a_video_the_backend_is_ignored_with_a_warning(tmp_path, caplog):
    src, _ = takeover_scene(tmp_path)
    backend = FakeBackend({})
    refiner = TrackRefiner(cfg_for(tmp_path), vlm_backend_factory=lambda c: backend)
    with caplog.at_level(logging.WARNING):
        refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    assert backend.calls == [] and "ignored" in caplog.text
    assert splits(refiner.last_result)[0].decision is Decision.HUMAN_PENDING
    assert refiner.last_result.summary["vlm"]["calls"] == 0


def test_a_missing_package_or_key_fails_before_any_file_is_written(tmp_path, monkeypatch):
    src, video = takeover_scene(tmp_path)
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-vlm\]"):
        TrackRefiner(cfg_for(tmp_path)).refine(src, tmp_path / "o.txt", video_file=video,
                                               verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]
    import importlib.machinery
    import types

    fake = types.ModuleType("anthropic")
    fake.__spec__ = importlib.machinery.ModuleSpec("anthropic", None)
    fake.AsyncAnthropic = object
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = cfg_for(tmp_path)
    cfg.vlm.backend, cfg.vlm.model = "anthropic", None
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        TrackRefiner(cfg).refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]


def test_backend_none_changes_nothing(tmp_path):
    src, video = takeover_scene(tmp_path)
    cfg = cfg_for(tmp_path)
    cfg.vlm.backend, cfg.vlm.model = "none", None
    refiner = TrackRefiner(cfg)
    refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert splits(refiner.last_result)[0].decision is Decision.HUMAN_PENDING
    assert refiner.last_result.summary["vlm"] == {
        "calls": 0, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0,
        "no_evidence": 0,
    }


def test_refine_writes_the_review_for_pending_events_and_records_the_image(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    assert res.review_path == tmp_path / "o.review.html" and res.review_path.is_file()
    (ev,) = splits(res)
    html = res.review_path.read_text()
    assert f'data-id="{ev.id}"' in html and (tmp_path / "o.review" / f"{ev.id}.jpg").is_file()
    led = Ledger.read(res.ledger_path)
    (back,) = [e for e in led.events if e.kind is EventKind.SPLIT]
    assert back.vlm["evidence"] == f"o.review/{ev.id}.jpg"


def test_the_review_snippet_names_the_renumbered_output_tracks(tmp_path):
    # raw id 7 becomes output id 1; the snippet must point at the id the output file holds
    src, video = takeover_scene(tmp_path, first_id=7)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    html = res.review_path.read_text()
    assert "track_ids=[1]" in html and "track_ids=[]" not in html and "track_ids=[7]" not in html
    assert sorted(res.tracks.track.unique()) == [1]


def test_regenerating_a_review_for_other_inputs_changes_the_run_and_card_keys(tmp_path):
    def keys(html):
        return html.split('data-run="')[1].split('"')[0], html.split('data-key="')[1].split('"')[0]

    src, video = takeover_scene(tmp_path)
    run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    first = (tmp_path / "o.review.html").read_text()
    # the same output name and the same event id (switch-r0-000001), but another track file
    src2, video2 = takeover_scene(tmp_path, name="u.txt", video="w.mp4", first_id=5)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src2, video2)
    second = res.review_path.read_text()
    assert 'data-id="switch-r0-000001"' in first and 'data-id="switch-r0-000001"' in second
    assert keys(first)[0] != keys(second)[0]  # the run key (what localStorage is namespaced by)
    assert keys(first)[1] != keys(second)[1]  # the proposal key (what each saved choice is checked by)


def test_the_review_lists_exactly_the_pending_events(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    pend = [e for e in res.events if e.decision is Decision.HUMAN_PENDING
            and e.kind not in (EventKind.FILL, EventKind.SMOOTH)]
    assert pend  # an "unsure" answer keeps the split pending
    html = res.review_path.read_text()
    assert html.count('class="card"') == len(pend)
    assert all(f'data-id="{e.id}"' in html for e in pend)


def test_the_review_is_removed_when_a_rerun_leaves_nothing_pending(tmp_path):
    src, video = takeover_scene(tmp_path)
    run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    assert (tmp_path / "o.review.html").is_file() and (tmp_path / "o.review").is_dir()
    # a "different" answer confirms the split, so nothing is pending any more
    cfg = cfg_for(tmp_path, cache_dir=str(tmp_path / "other-cache"))  # not the cached "unsure"
    res = run(tmp_path, cfg, FakeBackend({"SPLIT": reply("different")}), src, video)
    assert not any(e.decision is Decision.HUMAN_PENDING and e.kind is EventKind.SPLIT
                   for e in res.events)
    assert res.review_path is None
    assert not (tmp_path / "o.review.html").exists() and not (tmp_path / "o.review").exists()


def test_a_run_without_a_video_still_writes_a_signals_only_review(tmp_path):
    src, _ = takeover_scene(tmp_path)
    refiner = TrackRefiner(cfg_for(tmp_path))
    refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    html = refiner.last_result.review_path.read_text()
    assert "<img" not in html and "no image" in html


def test_an_unknown_frame_count_still_reaches_the_backend(tmp_path, monkeypatch, caplog):
    from dnt.refine import io

    real = io.video_info
    monkeypatch.setattr(io, "video_info", lambda path: {**real(path), "frame_count": 0})
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": reply("different")})
    with caplog.at_level(logging.WARNING, logger="dnt.refine"):
        res = run(tmp_path, cfg_for(tmp_path), backend, src, video)
    (ev,) = splits(res)
    assert len(backend.calls) == 1 and ev.decision is Decision.VLM_ACCEPT
    assert res.summary["vlm"]["calls"] == 1 and res.summary["vlm"]["no_evidence"] == 0
    unknown = [r for r in caplog.records if "frame count unknown" in r.getMessage()]
    assert len(unknown) == 1 and unknown[0].levelno == logging.WARNING


def test_the_frame_count_warning_needs_a_backend(tmp_path, monkeypatch, caplog):
    from dnt.refine import io

    real = io.video_info
    monkeypatch.setattr(io, "video_info", lambda path: {**real(path), "frame_count": 0})
    src, video = takeover_scene(tmp_path)
    cfg = cfg_for(tmp_path)
    cfg.vlm.backend, cfg.vlm.model = "none", None
    with caplog.at_level(logging.WARNING, logger="dnt.refine"):
        TrackRefiner(cfg).refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert not [r for r in caplog.records if "frame count unknown" in r.getMessage()]


def test_events_without_an_evidence_image_are_counted(tmp_path, monkeypatch):
    from dnt.refine import evidence

    def broken(path):
        raise ValueError("cannot open video")

    monkeypatch.setattr(evidence, "FrameReader", broken)
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": reply("different")})
    res = run(tmp_path, cfg_for(tmp_path), backend, src, video)
    (ev,) = splits(res)
    assert backend.calls == [] and ev.vlm["error"] == "no evidence image"
    assert res.summary["vlm"] == {
        "calls": 0, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0,
        "no_evidence": 1,
    }


def test_the_review_snippet_names_the_output_file_by_its_absolute_path(tmp_path, monkeypatch):
    import html as html_mod

    src, video = takeover_scene(tmp_path)
    monkeypatch.chdir(tmp_path)
    refiner = TrackRefiner(cfg_for(tmp_path), vlm_backend_factory=lambda c: FakeBackend(
        {"SPLIT": reply("unsure")}))
    refiner.refine(src, "o.txt", video_file=video, verbose=False)
    page = html_mod.unescape(refiner.last_result.review_path.read_text())
    assert f"track_file={str((tmp_path / 'o.txt').resolve())!r}" in page
    assert "track_file='o.txt'" not in page


def test_the_runner_is_made_only_after_the_inputs_are_read_and_is_closed(tmp_path, monkeypatch):
    from dnt.refine import refiner as refiner_mod

    made = []

    class Recording(VLMRunner):
        def __init__(self, *a, **k):
            made.append(self)
            super().__init__(*a, **k)

    monkeypatch.setattr(refiner_mod, "VLMRunner", Recording)
    src, video = takeover_scene(tmp_path)
    bad = tmp_path / "bad.txt"
    bad.write_text("this is not a track file\n")
    built = []
    refiner = TrackRefiner(cfg_for(tmp_path), vlm_backend_factory=lambda c: built.append(c) or
                           FakeBackend({"SPLIT": reply("different")}))
    with pytest.raises(ValueError):
        refiner.refine(bad, tmp_path / "o.txt", video_file=video, verbose=False)
    assert len(built) == 1 and made == []  # the backend is checked first; no runner was made
    refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert len(made) == 1 and made[0]._closed


def test_the_endpoint_named_by_use_is_what_the_backend_factory_receives(tmp_path):
    src, video = takeover_scene(tmp_path)
    cfg = cfg_for(tmp_path)
    cfg.vlm.endpoints = {
        "a": {"backend": "openai_compat", "model": "model-a", "base_url": "http://a.example/v1"},
        "b": {"backend": "openai_compat", "model": "model-b", "base_url": "http://b.example/v1"},
    }
    cfg.vlm.use = "b"
    backend, seen = FakeBackend({"SPLIT": reply("different")}), []
    refiner = TrackRefiner(cfg, vlm_backend_factory=lambda c: seen.append(c) or backend)
    refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    (got,) = seen
    assert (got.model, got.base_url) == ("model-b", "http://b.example/v1")
    assert got.endpoints == {} and got.use is None
    assert len(backend.calls) == 1


def test_max_tokens_is_validated_and_selectable_per_endpoint():
    cfg = RefineConfig.defaults()
    cfg.vlm.endpoints = {"r": {"backend": "openai_compat", "model": "m", "max_tokens": 4096}}
    cfg.vlm.use = "r"
    cfg.validate()
    assert cfg.vlm.resolve().max_tokens == 4096
    for bad in (0, -1, True, 1.5, "300"):
        cfg = RefineConfig.defaults()
        cfg.vlm.max_tokens = bad
        with pytest.raises(ValueError, match="max_tokens"):
            cfg.validate()


def test_extra_body_is_validated_and_selectable_per_endpoint():
    cfg = RefineConfig.defaults()
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    cfg.vlm.endpoints = {"q": {"backend": "openai_compat", "model": "m", "extra_body": body}}
    cfg.vlm.use = "q"
    cfg.validate()
    assert cfg.vlm.resolve().extra_body == body
    for bad in ([1], "x", {1: "a"}, {"a": float("nan")}, {"a": {1: 2}}, {"a": object()}):
        cfg = RefineConfig.defaults()
        cfg.vlm.extra_body = bad
        with pytest.raises(ValueError, match="extra_body"):
            cfg.validate()
