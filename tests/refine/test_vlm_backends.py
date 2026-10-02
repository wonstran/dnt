import asyncio
import base64
import json
import sys

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import DEFAULT_ANTHROPIC_MODEL, VLMAnswer, VLMTransientError, make_backend
from dnt.refine.vlm.fake import FakeBackend

from ._vlm_fakes import install_fake_anthropic, install_fake_openai

OPTS = ["same_individual", "different", "unsure"]
GOOD = json.dumps({"answer": "different", "confidence": 0.9, "reason": "r"})


def ask(backend, tag="LINK:link-r0-000001", temperature=0.0):
    return asyncio.run(backend.ask(b"\xff\xd8jpeg", "PROMPT", OPTS, temperature, tag=tag))


# ---- fake ----

def test_fake_lookup_order_and_call_log():
    b = FakeBackend({"LINK:link-r0-000001": GOOD, "LINK": {"answer": "unsure", "confidence": 0.1,
                                                          "reason": ""}, "*": GOOD})
    assert ask(b).answer == "different"  # whole tag
    assert ask(b, tag="LINK:link-r0-000002").answer == "unsure"  # kind
    assert ask(b, tag="SPLIT:switch-r0-000001").answer == "different"  # default
    assert b.calls[0]["tag"] == "LINK:link-r0-000001" and b.calls[0]["image_len"] == 6
    assert b.calls[0]["options"] == OPTS and b.calls[0]["prompt"] == "PROMPT"


def test_fake_event_id_key_and_sequences_and_exceptions():
    b = FakeBackend({"link-r0-000001": [RuntimeError("boom"), "garbage", GOOD]})
    with pytest.raises(RuntimeError, match="boom"):
        ask(b)
    with pytest.raises(ValueError):
        ask(b)  # an invalid reply is parsed like a real one
    assert ask(b).answer == "different"
    assert ask(b).answer == "different"  # the last item repeats


def test_fake_callable_returns_items_and_unknown_tag_raises():
    b = FakeBackend(lambda tag, options, temperature: {"answer": options[1], "confidence": 1.0,
                                                       "reason": str(temperature)})
    assert ask(b, temperature=0.7).reason == "0.7"
    with pytest.raises(RuntimeError, match="no scripted answer"):
        ask(FakeBackend({}))
    ans = VLMAnswer("different", 0.5, "x", "raw")
    assert ask(FakeBackend({"*": ans})) is ans


# ---- openai_compat ----

def cfg(**kw):
    return VLMConfig(**{"backend": "openai_compat", "base_url": "http://localhost:8000/v1",
                        "model": "qwen", **kw})


def test_openai_request_shape(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    b = make_backend(cfg(timeout_s=12.0))
    ans = ask(b, temperature=0.3)
    assert ans.answer == "different" and b.name == "openai_compat" and b.model == "qwen"
    ck = seen["client_kwargs"]
    assert ck["base_url"] == "http://localhost:8000/v1" and ck["api_key"] == "sk-secret"
    assert ck["timeout"] == 12.0 and ck["max_retries"] == 0
    req = seen["requests"][0]
    assert req["model"] == "qwen" and req["temperature"] == 0.3
    assert req["response_format"] == {"type": "json_object"}
    content = req["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "PROMPT"}
    url = content[1]["image_url"]["url"]
    assert url == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()


def test_openai_json_mode_off_and_missing_key_for_local_servers(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    b = make_backend(cfg(json_mode=False))
    ask(b)
    assert "response_format" not in seen["requests"][0]
    assert seen["client_kwargs"]["api_key"]  # a local server needs a placeholder, not None


def test_openai_custom_key_env(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.setenv("MY_VLM_KEY", "abc")
    make_backend(cfg(api_key_env="MY_VLM_KEY"))
    assert seen["client_kwargs"]["api_key"] == "abc"


@pytest.mark.parametrize(
    "name", ["APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"]
)
def test_openai_transient_errors_are_mapped(monkeypatch, name):
    install_fake_openai(monkeypatch, [])  # creates the module, so the exception class exists
    exc = getattr(sys.modules["openai"], name)("sk-secret failure")
    install_fake_openai(monkeypatch, [exc])
    with pytest.raises(VLMTransientError) as err:
        ask(make_backend(cfg()))
    assert "sk-secret" not in str(err.value)


def test_openai_other_errors_propagate_and_5xx_status_is_transient(monkeypatch):
    install_fake_openai(monkeypatch, [])
    status_error = sys.modules["openai"].APIStatusError
    install_fake_openai(monkeypatch, [status_error("bad request", 400)])
    with pytest.raises(status_error):
        ask(make_backend(cfg()))
    install_fake_openai(monkeypatch, [status_error("down", 503)])
    with pytest.raises(VLMTransientError):
        ask(make_backend(cfg()))


# ---- anthropic ----

def test_anthropic_request_shape_and_default_model(monkeypatch):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    b = make_backend(VLMConfig(backend="anthropic", timeout_s=9.0))
    assert b.model == DEFAULT_ANTHROPIC_MODEL and b.name == "anthropic"
    assert ask(b, temperature=0.2).answer == "different"  # text block found after a non-text one
    ck = seen["client_kwargs"]
    assert ck["api_key"] == "sk-ant" and ck["timeout"] == 9.0 and ck["max_retries"] == 0
    req = seen["requests"][0]
    assert req["model"] == DEFAULT_ANTHROPIC_MODEL and req["temperature"] == 0.2
    assert req["max_tokens"] > 0
    blocks = req["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[0]["source"]["data"] == base64.b64encode(b"\xff\xd8jpeg").decode()
    assert blocks[1] == {"type": "text", "text": "PROMPT"}


def test_anthropic_model_override_missing_key_and_errors(monkeypatch):
    install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_backend(VLMConfig(backend="anthropic"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert make_backend(VLMConfig(backend="anthropic", model="claude-x")).model == "claude-x"
    install_fake_anthropic(monkeypatch, [])
    exc = sys.modules["anthropic"].RateLimitError("sk-ant slow down")
    install_fake_anthropic(monkeypatch, [exc])
    with pytest.raises(VLMTransientError) as err:
        ask(make_backend(VLMConfig(backend="anthropic")))
    assert "sk-ant" not in str(err.value)


def test_backends_close_their_client_and_the_fake_counts_closes(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    asyncio.run(make_backend(cfg()).aclose())
    assert seen["closed"] == 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    seen2 = install_fake_anthropic(monkeypatch, [GOOD])
    asyncio.run(make_backend(VLMConfig(backend="anthropic")).aclose())
    assert seen2["closed"] == 1
    fake = FakeBackend({})
    asyncio.run(fake.aclose())
    assert fake.closed == 1
