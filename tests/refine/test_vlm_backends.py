import asyncio
import base64
import json
import sys

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import DEFAULT_ANTHROPIC_MODEL, VLMAnswer, VLMTransientError, make_backend
from dnt.refine.vlm.fake import FakeBackend

from ._vlm_fakes import (
    anthropic_reply,
    install_fake_anthropic,
    install_fake_openai,
    openai_reply,
)

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


def test_openai_max_tokens_defaults_to_1024_and_follows_the_setting(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    ask(make_backend(cfg()))
    ask(make_backend(cfg(max_tokens=4000)))
    assert [r["max_tokens"] for r in seen["requests"]] == [1024, 4000]


def test_an_openai_reply_cut_off_before_its_answer_names_max_tokens(monkeypatch):
    cut = openai_reply('{"answer": "diff', finish_reason="length")
    install_fake_openai(monkeypatch, [cut])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    with pytest.raises(RuntimeError, match=r"max_tokens=2048.*raise vlm\.max_tokens") as err:
        ask(make_backend(cfg(max_tokens=2048)))
    assert "sk-secret" not in str(err.value)


def test_a_complete_openai_answer_that_hit_the_limit_is_still_used(monkeypatch):
    install_fake_openai(monkeypatch, [openai_reply(GOOD, finish_reason="length")])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    assert ask(make_backend(cfg())).answer == "different"


def test_a_bad_reply_that_was_not_cut_off_keeps_its_parse_error(monkeypatch):
    install_fake_openai(monkeypatch, [openai_reply("no json here", finish_reason="stop")])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    with pytest.raises(ValueError, match="no JSON object"):
        ask(make_backend(cfg()))


def test_anthropic_max_tokens_setting_overrides_the_default(monkeypatch):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    ask(make_backend(VLMConfig(backend="anthropic", max_tokens=3000)))
    assert seen["requests"][0]["max_tokens"] == 3000


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
    assert req["model"] == DEFAULT_ANTHROPIC_MODEL
    # the default model rejects a non-default temperature (HTTP 400) and thinks by default:
    # no sampling parameters, effort "low", and room for the thinking before the answer
    assert "temperature" not in req and "top_p" not in req and "top_k" not in req
    assert "thinking" not in req and "output_config" not in req
    assert req["extra_body"] == {"output_config": {"effort": "low"}}
    assert req["max_tokens"] == 2048
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


@pytest.mark.parametrize(
    ("model", "new"),
    [
        ("claude-sonnet-5-5", True),
        ("claude-sonnet-5", True),
        ("claude-opus-5-1", True),
        ("claude-opus-4-7", True),
        ("claude-opus-4-8-20260101", True),
        ("claude-fable-1", True),
        ("claude-mythos-2", True),
        ("claude-haiku-4-5", False),
        ("claude-sonnet-4-6", False),
        ("claude-opus-4-6", False),
        ("claude-opus-4-1", False),
        ("claude-3-5-sonnet-latest", False),
        ("", False),
    ],
)
def test_new_family_models_are_recognized(model, new):
    from dnt.refine.vlm.anthropic import _new_family

    assert _new_family(model) is new


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-6"])
def test_older_claude_models_get_the_temperature_and_no_effort(monkeypatch, model):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    b = make_backend(VLMConfig(backend="anthropic", model=model))
    ask(b, temperature=0.0)
    ask(b, temperature=0.7)
    first, second = seen["requests"]
    assert first["temperature"] == 0.0 and second["temperature"] == 0.7
    for req in (first, second):
        assert req["model"] == model and req["max_tokens"] == 1024
        assert "extra_body" not in req and "output_config" not in req and "thinking" not in req


def test_votes_reach_an_older_model_at_the_vote_temperature_and_skip_it_on_new_ones(monkeypatch):
    from dnt.refine.vlm.runner import Question, VLMRunner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    for model, expect in (("claude-haiku-4-5", 0.7), (DEFAULT_ANTHROPIC_MODEL, None)):
        seen = install_fake_anthropic(monkeypatch, [GOOD])
        cfg_ = VLMConfig(backend="anthropic", model=model, votes=3, vote_temperature=0.7)
        with VLMRunner(cfg_, make_backend(cfg_)) as r:
            (v,) = r.ask_many([Question("LINK:link-r0-000001", b"img", "P", list(OPTS))])
        assert v.answer == "different" and len(seen["requests"]) == 3
        assert [req.get("temperature") for req in seen["requests"]] == [expect] * 3


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (anthropic_reply(None, stop_reason="refusal"), "refused"),
        (anthropic_reply(GOOD, stop_reason="refusal"), "refused"),
        (anthropic_reply(None, stop_reason="max_tokens"), "max_tokens"),
        (anthropic_reply('{"answer": "diff', stop_reason="max_tokens"), "max_tokens"),
        (anthropic_reply(None, stop_reason="end_turn"), "no text"),
        (anthropic_reply("  ", stop_reason="end_turn", thinking=False), "no text"),
    ],
)
def test_a_refusal_a_cut_off_or_a_textless_reply_is_an_error(monkeypatch, reply, message):
    install_fake_anthropic(monkeypatch, [reply])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    with pytest.raises(RuntimeError, match=message) as err:
        ask(make_backend(VLMConfig(backend="anthropic")))
    assert "sk-ant" not in str(err.value)


def test_a_complete_answer_that_hit_max_tokens_is_still_used(monkeypatch):
    install_fake_anthropic(monkeypatch, [anthropic_reply(GOOD, stop_reason="max_tokens")])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert ask(make_backend(VLMConfig(backend="anthropic"))).answer == "different"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens", "end_turn"])
def test_the_runner_does_not_retry_a_refusal_a_cut_off_or_a_textless_reply(monkeypatch, stop):
    from dnt.refine.vlm.runner import Question, VLMRunner

    seen = install_fake_anthropic(monkeypatch, [anthropic_reply(None, stop_reason=stop)])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    cfg_ = VLMConfig(backend="anthropic")
    with VLMRunner(cfg_, make_backend(cfg_)) as r:
        (v,) = r.ask_many([Question("LINK:link-r0-000001", b"img", "P", list(OPTS))])
    assert v.answer is None and v.error.startswith("RuntimeError: ")
    assert len(seen["requests"]) == 1 and r.calls == 1 and r.retries == 0 and r.failures == 1
    assert "sk-ant" not in v.error


@pytest.mark.parametrize(
    "make",
    [
        lambda m: m.APIConnectionError("sk-ant connection reset"),
        lambda m: m.APITimeoutError("sk-ant timed out"),
        lambda m: m.InternalServerError("sk-ant overloaded"),
        lambda m: m.APIStatusError("sk-ant bad gateway", 502),
        lambda m: m.APIStatusError("sk-ant overloaded", 529),
        lambda m: m.RateLimitError("sk-ant slow down"),
    ],
)
def test_anthropic_transient_errors_are_mapped(monkeypatch, make):
    install_fake_anthropic(monkeypatch, [])
    exc = make(sys.modules["anthropic"])
    install_fake_anthropic(monkeypatch, [exc])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    with pytest.raises(VLMTransientError) as err:
        ask(make_backend(VLMConfig(backend="anthropic")))
    assert "sk-ant" not in str(err.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_anthropic_client_errors_propagate_unchanged(monkeypatch, status):
    install_fake_anthropic(monkeypatch, [])
    exc = sys.modules["anthropic"].APIStatusError("bad", status)
    install_fake_anthropic(monkeypatch, [exc])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    with pytest.raises(type(exc)) as err:
        ask(make_backend(VLMConfig(backend="anthropic")))
    assert err.value is exc and err.value.status_code == status
