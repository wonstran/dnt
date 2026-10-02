"""Key content checks, the cleartext warning, URL checks, and the CLI's ``--vlm-*`` semantics."""

import json
import logging
import os
import re
import subprocess
import sys
import threading

import pytest

from dnt.refine import cli
from dnt.refine.config import RefineConfig, VLMConfig
from dnt.refine.events import Ledger
from dnt.refine.refiner import TrackRefiner
from dnt.refine.vlm import make_backend
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import Question, VLMRunner

from ._video import takeover_scene
from ._vlm_fakes import install_fake_anthropic, install_fake_openai

GOOD = json.dumps({"answer": "different", "confidence": 0.9, "reason": "r"})
KEY = "sk-Good4Key9Zq"


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL",
                 "MY_VLM_KEY"):
        monkeypatch.delenv(name, raising=False)


def anthropic_cfg(**kw):
    return VLMConfig(**{"backend": "anthropic", **kw})


def tokens(text):
    """The distinctive pieces of a key or a file's content (runs of 4+ letters and digits)."""
    return re.findall(r"[A-Za-z0-9]{4,}", text)


# ---- key file content ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"\xef\xbb\xbf" + KEY.encode() + b"\n", KEY.encode() + b"\r\n", b"\n  " + KEY.encode() + b"  "],
)
def test_a_bom_crlf_or_padding_is_removed_from_a_key_file(monkeypatch, tmp_path, raw):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    f = tmp_path / "k"
    f.write_bytes(raw)
    make_backend(anthropic_cfg(api_key_file=str(f)))
    assert seen["client_kwargs"]["api_key"] == KEY


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("sk-Line1Part\nsk-Line2Part\n", "more than one line"),
        ("# Comment4Text\nsk-AfterComment9\n", "more than one line"),
        ("sk-Carriage1\r\nsk-Carriage2", "more than one line"),
        ("sk-Bare1Part\rsk-Bare2Part", "more than one line"),
        ("ANTHROPIC_API_KEY=sk-DotEnv7Val\n", "NAME=value"),
        ("export MYKEY=sk-Exported8Val\n", "NAME=value"),
        ("sk-Space1Part sk-Space2Part", "contains whitespace"),
        ("sk-Tab1Part\tsk-Tab2Part", "contains whitespace"),
        ("sk-AccentéPart1", "non-ASCII or control"),
        ("sk-Ctrl\x01Part2abc", "non-ASCII or control"),
    ],
)
def test_a_malformed_key_file_is_rejected_by_reason_without_the_content(
    monkeypatch, tmp_path, content, reason
):
    install_fake_anthropic(monkeypatch, [GOOD])
    f = tmp_path / "vlm.key"
    f.write_bytes(content.encode("utf-8"))
    with pytest.raises(ValueError, match=re.escape(reason)) as err:
        make_backend(anthropic_cfg(api_key_file=str(f)))
    msg = str(err.value)
    assert str(f) in msg
    for tok in tokens(content):
        assert tok not in msg, tok


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('"sk-Quoted1Part"', "wrapped in quotes"),
        ("'sk-Quoted2Part'", "wrapped in quotes"),
        ("OPENAI_API_KEY=", "NAME=value"),
        ("API_KEY=", "NAME=value"),  # 8 characters, but a variable name, not base64
        ("export OPENAI_API_KEY=", "NAME=value"),
        ("MYKEY==sk-Double9Eq", "NAME=value"),
        ("Short1Key=", "NAME=value"),  # not a multiple of 4: not base64 padding
    ],
)
def test_quoted_keys_and_empty_or_doubled_assignments_are_rejected(
    monkeypatch, tmp_path, content, reason
):
    install_fake_anthropic(monkeypatch, [GOOD])
    f = tmp_path / "vlm.key"
    f.write_text(content + "\n")
    with pytest.raises(ValueError, match=re.escape(reason)) as err:
        make_backend(anthropic_cfg(api_key_file=str(f)))
    for tok in tokens(content):
        assert tok not in str(err.value), tok


@pytest.mark.parametrize("key", ["QUJDREVGR0g=", "QUJDREVGRw==", "sk-proj-Abc4=", "abcdefg="])
def test_trailing_base64_padding_is_allowed(monkeypatch, tmp_path, key):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    f = tmp_path / "vlm.key"
    f.write_text(key + "\n")
    make_backend(anthropic_cfg(api_key_file=str(f)))
    assert seen["client_kwargs"]["api_key"] == key


def test_quoted_env_and_runtime_keys_are_rejected(monkeypatch):
    install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", '"sk-QuotedEnv1"')
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is wrapped"):
        make_backend(anthropic_cfg())
    with pytest.raises(ValueError, match="vlm_api_key argument is wrapped") as err:
        TrackRefiner(vlm_api_key="'sk-QuotedArg2'")
    assert "QuotedArg2" not in str(err.value)


@pytest.mark.parametrize(
    "case", ["not_utf8", "oversized", "missing", "directory", "empty", "multi_line", "quoted"]
)
def test_key_file_errors_carry_no_exception_context(monkeypatch, tmp_path, case):
    # a chained UnicodeDecodeError's .object would hold the file's bytes (the key)
    install_fake_anthropic(monkeypatch, [GOOD])
    f = tmp_path / "vlm.key"
    if case == "not_utf8":
        f.write_bytes(b"\xff" + KEY.encode())
    elif case == "oversized":
        f.write_bytes(KEY.encode() * 5000)
    elif case == "directory":
        f.mkdir()
    elif case == "empty":
        f.write_text(" \n")
    elif case == "multi_line":
        f.write_text(KEY + "\n" + KEY)
    elif case == "quoted":
        f.write_text(f'"{KEY}"')
    with pytest.raises(ValueError) as err:
        make_backend(anthropic_cfg(api_key_file=str(f)))
    assert err.value.__context__ is None and err.value.__cause__ is None
    assert KEY not in str(err.value)


def test_an_oversized_key_file_is_rejected_after_a_capped_read(monkeypatch, tmp_path):
    install_fake_anthropic(monkeypatch, [GOOD])
    f = tmp_path / "big.key"
    f.write_bytes(b"A" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="larger than 64 KiB"):
        make_backend(anthropic_cfg(api_key_file=str(f)))


@pytest.mark.skipif(not os.path.exists("/proc/self/statm"),
                    reason="needs /dev/zero, /proc and resource limits (Linux)")
def test_an_endless_key_file_is_cut_off():
    # in a child limited to 1 GiB more address space than its imports took, with a timeout: a
    # regression that reads to the end fails here instead of filling memory or hanging the run
    code = (
        "import os, resource\n"
        "from dnt.refine.config import VLMConfig\n"
        "from dnt.refine.vlm import resolve_api_key\n"
        "with open('/proc/self/statm') as f:\n"
        "    size = int(f.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')\n"
        "resource.setrlimit(resource.RLIMIT_AS, (size + (1 << 30), size + (1 << 30)))\n"
        "try:\n"
        "    resolve_api_key(VLMConfig(api_key_file='/dev/zero'), 'X')\n"
        "except ValueError as e:\n"
        "    print(e)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env=env)
    assert "larger than 64 KiB" in r.stdout, r.stderr[-500:]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_fifo_with_no_writer_does_not_hang(monkeypatch, tmp_path):
    install_fake_anthropic(monkeypatch, [GOOD])
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    got = {}

    def read():
        try:
            make_backend(anthropic_cfg(api_key_file=str(fifo)))
        except Exception as exc:
            got["error"] = exc

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(10)
    hung = t.is_alive()
    try:
        # a reader blocked in open() or read() returns once a writer opens and closes the pipe
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
    except OSError:  # ENXIO: no reader is waiting, nothing to release
        pass
    t.join(10)
    assert not hung, "reading a FIFO with no writer blocked"
    assert isinstance(got.get("error"), ValueError) and "is empty" in str(got["error"])


# ---- runtime and environment keys ---------------------------------------------------------


def test_env_keys_are_stripped_and_a_blank_one_counts_as_unset(monkeypatch):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"  {KEY}\n")
    make_backend(anthropic_cfg())
    assert seen["client_kwargs"]["api_key"] == KEY
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(ValueError, match="needs an API key and none was found"):
        make_backend(anthropic_cfg())


@pytest.mark.parametrize(
    ("value", "reason"),
    [("sk-Env1Part sk-Env2Part", "contains whitespace"), ("sk-EnvéPart3", "non-ASCII"),
     ("sk-EnvLine1\nsk-EnvLine2", "more than one line")],
)
def test_a_malformed_env_key_names_the_variable_not_the_value(monkeypatch, value, reason):
    install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("MY_VLM_KEY", value)
    with pytest.raises(ValueError, match=re.escape(reason)) as err:
        make_backend(anthropic_cfg(api_key_env="MY_VLM_KEY"))
    assert "MY_VLM_KEY environment variable" in str(err.value)
    for tok in tokens(value):
        assert tok not in str(err.value), tok


def test_a_malformed_runtime_key_names_the_argument_not_the_value(monkeypatch):
    install_fake_anthropic(monkeypatch, [GOOD])
    for value in ("sk-Rt1Part sk-Rt2Part", "sk-RtLine1\nsk-RtLine2", "sk-RtéPart3"):
        for call in (lambda v: TrackRefiner(vlm_api_key=v),
                     lambda v: make_backend(anthropic_cfg(), api_key=v)):
            with pytest.raises(ValueError, match="vlm_api_key argument") as err:
                call(value)
            for tok in tokens(value):
                assert tok not in str(err.value), tok
    with pytest.raises(ValueError, match="must be a string"):
        TrackRefiner(vlm_api_key=b"sk-bytes")


def test_a_runtime_key_with_a_custom_factory_is_an_error():
    with pytest.raises(ValueError, match="vlm_backend_factory") as err:
        TrackRefiner(vlm_api_key=KEY, vlm_backend_factory=lambda c: FakeBackend({}))
    assert KEY not in str(err.value)


# ---- cleartext warning --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "url", "key", "warned"),
    [
        ("anthropic", "http://gw.example.com/anthropic", True, True),
        ("anthropic", "https://gw.example.com/anthropic", True, False),
        ("anthropic", "http://localhost:4000", True, False),
        ("anthropic", "http://127.0.0.1:4000", True, False),
        ("anthropic", "http://[::1]:4000", True, False),
        ("openai_compat", "http://10.0.0.5:8000/v1", True, True),
        ("openai_compat", "http://10.0.0.5:8000/v1", False, False),  # placeholder: no secret
        ("openai_compat", "http://localhost:8000/v1", True, False),
    ],
)
def test_a_key_sent_over_http_to_a_remote_host_is_warned_about(
    monkeypatch, caplog, backend, url, key, warned
):
    (install_fake_anthropic if backend == "anthropic" else install_fake_openai)(monkeypatch, [GOOD])
    if key:
        monkeypatch.setenv(
            "ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY", KEY
        )
    with caplog.at_level(logging.WARNING):
        make_backend(VLMConfig(backend=backend, model="m", base_url=url))
    assert ("unencrypted" in caplog.text) is warned
    if warned:
        assert re.search(r"to (gw\.example\.com|10\.0\.0\.5),", caplog.text)
    assert KEY not in caplog.text


def test_the_sdk_base_url_variable_is_checked_too(monkeypatch, caplog):
    install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.example.net")
    with caplog.at_level(logging.WARNING):
        make_backend(anthropic_cfg())
    assert "unencrypted" in caplog.text and "proxy.example.net" in caplog.text


# ---- base_url checks and the scrubber -----------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://x/v1\n", "http://x/v1 ", "http://x /v1", "http://h:abc/v1", "http://h:99999/v1"]
)
def test_whitespace_and_bad_ports_are_rejected_by_validation(url):
    cfg = RefineConfig.defaults()
    cfg.vlm.base_url = url
    with pytest.raises(ValueError, match=r"vlm\.base_url"):
        cfg.validate()


def test_the_scrubber_skips_blank_and_very_short_secrets(monkeypatch):
    monkeypatch.setenv("MY_VLM_KEY", " ")
    b = FakeBackend({"*": [RuntimeError("a bad answer at row 7")]})
    b.secret = "a b"
    with VLMRunner(VLMConfig(backend="openai_compat", model="m", api_key_env="MY_VLM_KEY"),
                   b, None) as r:
        (v,) = r.ask_many([Question("LINK:x", b"img", "P", ["a", "b"])])
    assert v.error == "RuntimeError: a bad answer at row 7"


# ---- CLI ----------------------------------------------------------------------------------


def _main(capsys, *args):
    code = cli.main([str(a) for a in args])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def _yaml(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text("encoder:\n  kind: none\nlink:\n  enabled: false\n" + text)
    return path


def _header_vlm(out):
    return Ledger.read(json.loads(out)["ledger"]).header["config"]["vlm"]


VLLM = (
    "vlm:\n  backend: openai_compat\n  base_url: http://localhost:8000/v1\n  model: qwen\n"
    "  api_key_env: VLLM_KEY\n  api_key_file: /srv/vllm.key\n"
)


def test_switching_backend_drops_the_files_backend_bound_settings(tmp_path, capsys):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, VLLM)
    run = ("run", src, "--fps", 10, "--config", y)
    code, out, err = _main(capsys, *run, "--out", tmp_path / "a.txt", "--vlm-backend", "anthropic")
    assert code == 0, err
    vlm = _header_vlm(out)
    assert vlm["backend"] == "anthropic"
    assert (vlm["model"], vlm["base_url"], vlm["api_key_env"], vlm["api_key_file"]) == (
        None, None, None, None)
    code, out, err = _main(capsys, *run, "--out", tmp_path / "b.txt", "--vlm-backend", "anthropic",
                           "--vlm-base-url", "https://gw.example.com", "--vlm-model", "claude-x")
    assert code == 0, err
    vlm = _header_vlm(out)
    assert (vlm["model"], vlm["base_url"], vlm["api_key_env"]) == (
        "claude-x", "https://gw.example.com", None)
    # the same backend keeps everything; flags not given never clobber the file
    code, out, err = _main(capsys, *run, "--out", tmp_path / "c.txt",
                           "--vlm-backend", "openai_compat", "--vlm-model", "qwen2")
    assert code == 0, err
    vlm = _header_vlm(out)
    assert (vlm["model"], vlm["base_url"], vlm["api_key_env"], vlm["api_key_file"]) == (
        "qwen2", "http://localhost:8000/v1", "VLLM_KEY", "/srv/vllm.key")
    code, out, err = _main(capsys, *run, "--out", tmp_path / "d.txt", "--vlm-backend", "none")
    assert code == 0, err
    assert _header_vlm(out)["backend"] == "none"


def test_switching_to_anthropic_never_sends_its_key_to_the_files_endpoint(
    monkeypatch, tmp_path, capsys
):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    src, video = takeover_scene(tmp_path)
    y = _yaml(tmp_path, VLLM + f"  cache_dir: {tmp_path / 'cache'}\n")
    code, out, err = _main(capsys, "run", src, "--video", video, "--config", y,
                           "--out", tmp_path / "o.txt", "--vlm-backend", "anthropic")
    assert code == 0, err
    assert KEY not in out + err and seen["client_kwargs"]["api_key"] == KEY
    assert "base_url" not in seen["client_kwargs"] and seen["requests"]


@pytest.mark.parametrize(
    ("text", "flags"),
    [
        ("vlm:\n  backend: openai_compat\n", ["--vlm-model", "qwen"]),
        ("vlm:\n  backend: openai_compat\n  model: q\n  base_url: ftp://old/v1\n",
         ["--vlm-base-url", "http://localhost:8000/v1"]),
        # a switch drops the file's invalid base_url (it belongs to the other backend)
        ("vlm:\n  backend: anthropic\n  base_url: ftp://old\n",
         ["--vlm-backend", "openai_compat", "--vlm-model", "qwen"]),
    ],
)
def test_flags_can_complete_a_file_that_is_only_valid_with_them(tmp_path, capsys, text, flags):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, text)
    with pytest.raises(ValueError):
        RefineConfig.from_yaml(y)
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                           "--out", tmp_path / "o.txt", *flags)
    assert code == 0, err
    assert _header_vlm(out)["backend"] == "openai_compat"


def test_unknown_keys_still_fail_with_flags(tmp_path, capsys):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, "vlm:\n  backend: openai_compat\n  modle: qwen\n")
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                           "--out", tmp_path / "o.txt", "--vlm-model", "qwen")
    assert code == 2 and out == "" and "unknown config key 'vlm.modle'" in err


def test_blank_flags_reset_fields_to_their_defaults(tmp_path, capsys):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, "vlm:\n  backend: anthropic\n  model: claude-x\n  api_key_env: K\n")
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                           "--out", tmp_path / "o.txt", "--vlm-model", "",
                           "--vlm-api-key-env", " ")
    assert code == 0, err
    vlm = _header_vlm(out)
    assert vlm["model"] is None and vlm["api_key_env"] is None


def test_a_bad_port_is_a_clean_cli_error(tmp_path, capsys):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, "vlm:\n  backend: openai_compat\n  model: q\n")
    code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                           "--out", tmp_path / "o.txt", "--vlm-base-url", "http://h:abc/v1")
    assert code == 2 and out == "" and err.startswith("dnt-refine: error: ")
    assert "vlm.base_url" in err and "Traceback" not in err


@pytest.mark.parametrize("backend_line", ["", "  backend: null\n"])
def test_a_file_without_a_backend_counts_as_none(tmp_path, capsys, caplog, backend_line):
    src, _ = takeover_scene(tmp_path)
    y = _yaml(tmp_path, "vlm:\n" + backend_line + "  base_url: http://x.example/v1\n"
              "  api_key_env: TEMPLATE_KEY\n")
    with caplog.at_level(logging.INFO, logger="dnt.refine.cli"):
        code, out, err = _main(capsys, "run", src, "--fps", 10, "--config", y,
                               "--out", tmp_path / "o.txt", "--vlm-backend", "anthropic")
    assert code == 0, err
    vlm = _header_vlm(out)
    assert vlm["backend"] == "anthropic" and vlm["base_url"] is None
    assert vlm["api_key_env"] is None
    assert "replaces the config's none" in caplog.text and "None" not in caplog.text
    assert "x.example" not in caplog.text and "TEMPLATE_KEY" not in caplog.text  # names only
