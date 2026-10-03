"""The VLM endpoint (``vlm.base_url``) and the API key sources, and that a key never leaks."""

import json
import logging
import re

import pytest

from dnt.refine import cli
from dnt.refine.config import RefineConfig, VLMConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.refiner import TrackRefiner
from dnt.refine.vlm import make_backend, resolve_api_key
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import Question, VLMRunner

from ._video import takeover_scene
from ._vlm_fakes import APIStatusError, install_fake_anthropic, install_fake_openai

GOOD = json.dumps({"answer": "different", "confidence": 0.9, "reason": "r"})
LOCAL = "http://localhost:8000/v1"
ENV_KEY, FILE_KEY, RT_KEY = "sk-from-env-111", "sk-from-file-222", "sk-from-arg-333"
DEFAULT_ENV = {"openai_compat": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "MY_VLM_KEY"):
        monkeypatch.delenv(name, raising=False)


def install(monkeypatch, backend, replies=(GOOD,)):
    fake = install_fake_openai if backend == "openai_compat" else install_fake_anthropic
    return fake(monkeypatch, list(replies))


def vcfg(backend, **kw):
    base = {"backend": backend, "model": "m"}
    if backend == "openai_compat":
        base["base_url"] = LOCAL
    return VLMConfig(**{**base, **kw})


def key_file(tmp_path, text, name="vlm.key"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---- endpoint -----------------------------------------------------------------------------


def test_anthropic_honours_base_url(monkeypatch):
    seen = install(monkeypatch, "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)
    make_backend(vcfg("anthropic", base_url="https://gateway.example/anthropic"))
    assert seen["client_kwargs"]["base_url"] == "https://gateway.example/anthropic"
    make_backend(vcfg("anthropic"))  # unset: the SDK's own default endpoint
    assert seen["client_kwargs"].get("base_url") is None


# ---- key sources and their priority -------------------------------------------------------


@pytest.mark.parametrize("backend", ["openai_compat", "anthropic"])
def test_key_priority_is_runtime_then_file_then_env(monkeypatch, tmp_path, backend):
    seen = install(monkeypatch, backend)
    monkeypatch.setenv(DEFAULT_ENV[backend], ENV_KEY)
    kf = str(key_file(tmp_path, FILE_KEY))
    b = make_backend(vcfg(backend, api_key_file=kf), api_key=RT_KEY)
    assert seen["client_kwargs"]["api_key"] == RT_KEY and b.secret == RT_KEY
    b = make_backend(vcfg(backend, api_key_file=kf))
    assert seen["client_kwargs"]["api_key"] == FILE_KEY and b.secret == FILE_KEY
    b = make_backend(vcfg(backend))
    assert seen["client_kwargs"]["api_key"] == ENV_KEY and b.secret == ENV_KEY
    monkeypatch.setenv("MY_VLM_KEY", "sk-named-444")
    make_backend(vcfg(backend, api_key_env="MY_VLM_KEY"))
    assert seen["client_kwargs"]["api_key"] == "sk-named-444"


def test_key_file_is_stripped_and_tilde_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    key_file(tmp_path, "  \n" + FILE_KEY + " \r\n\n")
    assert resolve_api_key(vcfg("anthropic", api_key_file="~/vlm.key"), "X") == FILE_KEY


@pytest.mark.parametrize("content", [None, "", "  \n\t\n", b"\xff\xfe" + FILE_KEY.encode(), "dir"])
def test_an_unreadable_or_empty_key_file_names_the_path_not_the_content(
    monkeypatch, tmp_path, content
):
    install(monkeypatch, "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", ENV_KEY)  # a broken file is an error, not a fallback
    path = tmp_path / "keys" / "vlm.key"
    path.parent.mkdir()
    if content == "dir":
        path.mkdir()
    elif isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content)
    with pytest.raises(ValueError, match=r"vlm\.api_key_file") as err:
        make_backend(vcfg("anthropic", api_key_file=str(path)))
    msg = str(err.value)
    assert str(path) in msg and FILE_KEY not in msg and ENV_KEY not in msg
    # no chained exception carries the file's content into a traceback
    assert err.value.__cause__ is None
    assert err.value.__context__ is None or err.value.__suppress_context__


@pytest.mark.parametrize("backend", ["openai_compat", "anthropic"])
def test_no_key_anywhere_names_all_three_ways(monkeypatch, backend):
    install(monkeypatch, backend)
    cfg = vcfg(backend, base_url=None)
    with pytest.raises(ValueError) as err:
        make_backend(cfg)
    msg = str(err.value)
    for way in ("vlm_api_key", "vlm.api_key_file", DEFAULT_ENV[backend]):
        assert way in msg, way


def test_openai_compat_with_an_endpoint_and_no_key_sends_a_placeholder(monkeypatch):
    seen = install(monkeypatch, "openai_compat")
    b = make_backend(vcfg("openai_compat"))  # a local server needs no key
    assert seen["client_kwargs"]["api_key"] == "EMPTY" and b.secret is None
    monkeypatch.setenv("OPENAI_BASE_URL", LOCAL)  # the SDK's own endpoint variable counts too
    make_backend(vcfg("openai_compat", base_url=None))
    assert seen["client_kwargs"]["api_key"] == "EMPTY"


def test_a_blank_runtime_key_is_rejected():
    with pytest.raises(ValueError, match="vlm_api_key"):
        TrackRefiner(vlm_api_key="  ")


# ---- the key never leaks ------------------------------------------------------------------


def test_the_runner_scrubs_the_backends_resolved_key():
    b = FakeBackend({"*": [APIStatusError(f"denied for {RT_KEY}", 401)]})  # trips the breaker
    b.secret = RT_KEY
    cfg = VLMConfig(backend="openai_compat", model="m")
    with VLMRunner(cfg, b, None) as r:
        (v,) = r.ask_many([Question("LINK:x", b"img", "P", ["a", "b"])])
    assert v.answer is None and RT_KEY not in v.error and "***" in v.error
    assert r.fatal_error is not None and RT_KEY not in r.fatal_error


def _refine_cfg(tmp_path, **vlm):
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"  # motion-only: the takeover split lands in the uncertain band
    cfg.link.enabled = False
    cfg.vlm.backend, cfg.vlm.model, cfg.vlm.base_url = "openai_compat", "m", LOCAL
    cfg.vlm.cache_dir = str(tmp_path / "vlmcache")
    for k, v in vlm.items():
        setattr(cfg.vlm, k, v)
    return cfg


@pytest.mark.parametrize("source", ["runtime", "file", "env"])
def test_a_key_from_any_source_never_reaches_config_ledger_summary_or_logs(
    monkeypatch, tmp_path, caplog, source
):
    key = {"runtime": RT_KEY, "file": FILE_KEY, "env": ENV_KEY}[source]
    seen = install(monkeypatch, "openai_compat", [APIStatusError(f"invalid key {key}", 401)])
    src, video = takeover_scene(tmp_path)
    cfg = _refine_cfg(tmp_path)
    if source == "file":
        cfg.vlm.api_key_file = str(key_file(tmp_path, key + "\n"))
    if source == "env":
        monkeypatch.setenv("OPENAI_API_KEY", key)
    refiner = TrackRefiner(cfg, vlm_api_key=key if source == "runtime" else None)
    with caplog.at_level(logging.DEBUG):
        refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    res = refiner.last_result
    assert seen["client_kwargs"]["api_key"] == key and seen["requests"]  # it was really used
    (ev,) = [e for e in res.events if e.kind is EventKind.SPLIT]
    assert ev.decision is Decision.HUMAN_PENDING and "***" in ev.vlm["error"]
    assert key not in json.dumps(cfg.to_dict()) and key not in repr(cfg)
    assert key not in res.ledger_path.read_text()
    assert key not in json.dumps(res.summary, default=str)
    assert key not in caplog.text and "***" in caplog.text
    assert "remaining VLM questions are skipped" in caplog.text  # the breaker tripped
    page = res.review_path.read_text()
    assert ev.id in page and key not in page  # the pending event is on the review page
    for f in res.review_path.with_suffix("").rglob("*"):  # the page's images and manifest
        if f.is_file():
            assert key.encode() not in f.read_bytes(), f
    assert Ledger.read(res.ledger_path).header["config"]["vlm"]["api_key_file"] == (
        cfg.vlm.api_key_file
    )


def test_a_bad_key_file_fails_before_any_input_is_read(monkeypatch, tmp_path):
    install(monkeypatch, "openai_compat")
    cfg = _refine_cfg(tmp_path, api_key_file=str(tmp_path / "missing.key"))
    with pytest.raises(ValueError, match=re.escape(str(tmp_path / "missing.key"))):
        # the track file does not exist: had it been read first, this would be FileNotFoundError
        TrackRefiner(cfg).refine(tmp_path / "no_tracks.txt", tmp_path / "o.txt",
                                 video_file=tmp_path / "no_video.mp4", verbose=False)


# ---- config validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "phrase"),
    [
        ("ftp://host/v1", "http:// or https://"),
        ("localhost:8000/v1", "http:// or https://"),
        (5, "http:// or https://"),
        ("http://", "host"),
        ("https://user:sk-in-url-555@gw.example/v1", "user:pass@"),
        ("https://sk-in-url-555@gw.example/v1", "user:pass@"),
        ("https://gw.example/v1?key=sk-in-url-555", "query string"),
    ],
)
def test_base_url_must_be_a_plain_http_url_and_is_never_echoed(url, phrase):
    cfg = RefineConfig.defaults()
    cfg.vlm.backend, cfg.vlm.model, cfg.vlm.base_url = "openai_compat", "m", url
    with pytest.raises(ValueError, match=re.escape(phrase)) as err:
        cfg.validate()
    assert "vlm.base_url" in str(err.value) and "sk-in-url-555" not in str(err.value)
    if phrase == "user:pass@":
        assert "vlm.api_key_file" in str(err.value)


@pytest.mark.parametrize("url", [LOCAL, "https://api.example.com/v1", "HTTPS://gw.example"])
def test_good_base_urls_validate(url):
    cfg = RefineConfig.defaults()
    cfg.vlm.base_url = url
    cfg.validate()
    assert cfg.vlm.base_url == url


def test_blank_base_url_and_key_file_from_yaml_mean_unset(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("vlm:\n  base_url: ''\n  api_key_file: '  '\n")
    cfg = RefineConfig.from_yaml(y)
    assert cfg.vlm.base_url is None and cfg.vlm.api_key_file is None


def test_api_key_file_is_a_path_kept_in_the_config():
    cfg = RefineConfig.from_dict({"vlm": {"api_key_file": "~/.config/dnt/vlm.key"}})
    assert cfg.to_dict()["vlm"]["api_key_file"] == "~/.config/dnt/vlm.key"
    with pytest.raises(ValueError, match="must be a string"):
        RefineConfig.from_dict({"vlm": {"api_key_file": 5}})
    cfg.vlm.api_key_file = 5
    with pytest.raises(ValueError, match=r"vlm\.api_key_file must be the path"):
        cfg.validate()
    cfg.vlm.api_key_file = "sk-ant-pasted-666"
    with pytest.raises(ValueError, match="not the key") as err:
        cfg.validate()
    assert "sk-ant-pasted-666" not in str(err.value)


# ---- CLI ----------------------------------------------------------------------------------


def _main(capsys, *args):
    code = cli.main([str(a) for a in args])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def _yaml(tmp_path, **vlm):
    cfg = _refine_cfg(tmp_path)
    for k, v in vlm.items():
        setattr(cfg.vlm, k, v)
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    return path


def test_cli_vlm_options_override_the_yaml(tmp_path, capsys):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, backend="anthropic", model="claude-x", base_url="http://a.example/v1",
              api_key_env="A_KEY", api_key_file="/a/vlm.key")
    # without a video no backend is built (and no key is read): only the header is checked
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                           "--out", tmp_path / "o.txt")
    assert code == 0, err
    vlm = Ledger.read(json.loads(out)["ledger"]).header["config"]["vlm"]
    assert (vlm["backend"], vlm["model"], vlm["base_url"], vlm["api_key_env"],
            vlm["api_key_file"]) == ("anthropic", "claude-x", "http://a.example/v1", "A_KEY",
                                     "/a/vlm.key")
    code, out, err = _main(
        capsys, "run", src, "--fps", 10, "--config", y, "--out", tmp_path / "o2.txt",
        "--vlm-backend", "openai_compat", "--vlm-model", "qwen",
        "--vlm-base-url", "https://b.example/v1", "--vlm-api-key-env", "B_KEY",
        "--vlm-api-key-file", "/b/vlm.key",
    )
    assert code == 0, err
    vlm = Ledger.read(json.loads(out)["ledger"]).header["config"]["vlm"]
    assert (vlm["backend"], vlm["model"], vlm["base_url"], vlm["api_key_env"],
            vlm["api_key_file"]) == ("openai_compat", "qwen", "https://b.example/v1", "B_KEY",
                                     "/b/vlm.key")
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                           "--out", tmp_path / "o3.txt", "--vlm-backend", "none",
                           "--vlm-base-url", "")
    assert code == 0, err
    vlm = Ledger.read(json.loads(out)["ledger"]).header["config"]["vlm"]
    assert vlm["backend"] == "none" and vlm["base_url"] is None


@pytest.mark.parametrize("url", ["ftp://h/v1", "https://u:sk-cli-777@h/v1"])
def test_cli_rejects_a_bad_base_url_with_exit_2_and_no_secret(tmp_path, capsys, url):
    src, _ = takeover_scene(tmp_path)
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", _yaml(tmp_path),
                           "--out", tmp_path / "o.txt", "--vlm-base-url", url)
    assert code == 2 and out == "" and err.startswith("dnt-refine: error: ")
    assert "vlm.base_url" in err and "sk-cli-777" not in err
    assert not (tmp_path / "o.txt").exists()


def test_cli_key_file_end_to_end(monkeypatch, tmp_path, capsys):
    seen = install(monkeypatch, "openai_compat")
    src, video = takeover_scene(tmp_path)
    kf = key_file(tmp_path, FILE_KEY + "\n")
    y = _yaml(tmp_path, base_url=None)  # api.openai.com: a key is required
    code, out, err = _main(capsys, "run", src, "--video", video, "--config", y,
                           "--out", tmp_path / "o.txt", "--vlm-api-key-file", kf)
    assert code == 0, err
    assert seen["client_kwargs"]["api_key"] == FILE_KEY and len(seen["requests"]) == 1
    ledger = Ledger.read(json.loads(out)["ledger"])
    assert ledger.header["config"]["vlm"]["api_key_file"] == str(kf)
    (ev,) = [e for e in ledger.events if e.kind is EventKind.SPLIT]
    assert ev.decision is Decision.VLM_ACCEPT
    assert FILE_KEY not in out + err and FILE_KEY not in (tmp_path / "o.ledger.jsonl").read_text()


def test_cli_missing_or_unreadable_key_exits_2(monkeypatch, tmp_path, capsys):
    install(monkeypatch, "openai_compat")
    src, video = takeover_scene(tmp_path)
    y = _yaml(tmp_path, base_url=None)
    code, out, err = _main(capsys, "run", src, "--video", video, "--config", y,
                           "--out", tmp_path / "o.txt")
    assert code == 2 and out == "" and "vlm.api_key_file" in err and "OPENAI_API_KEY" in err
    missing = tmp_path / "nope.key"
    code, out, err = _main(capsys, "run", src, "--video", video, "--config", y,
                           "--out", tmp_path / "o.txt", "--vlm-api-key-file", missing)
    assert code == 2 and out == "" and err.startswith("dnt-refine: error: ")
    assert str(missing) in err
    assert not (tmp_path / "o.txt").exists()
