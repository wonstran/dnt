import importlib.machinery
import sys
import types

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import (
    DEFAULT_ANTHROPIC_MODEL,
    VLMAnswer,
    check_vlm_dependencies,
    make_backend,
    parse_answer,
)

OPTS = ["same_individual", "different", "unsure"]


def test_plain_json_is_parsed():
    a = parse_answer('{"answer": "different", "confidence": 0.9, "reason": "other clothes"}', OPTS)
    assert a == VLMAnswer("different", 0.9, "other clothes", a.raw)


def test_a_code_fence_and_text_around_the_json_are_tolerated():
    text = 'Sure!\n```json\n{"answer": "same_individual", "confidence": 0.8, "reason": "x"}\n```\nDone'
    assert parse_answer(text, OPTS).answer == "same_individual"
    text2 = 'I think {"answer": "unsure", "confidence": 0.2, "reason": "a {b} c"} ok'
    assert parse_answer(text2, OPTS).reason == "a {b} c"  # braces inside a string


def test_a_percentage_confidence_is_scaled():
    a = parse_answer('{"answer": "different", "confidence": 85, "reason": ""}', OPTS)
    assert a.confidence == pytest.approx(0.85)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        '{"answer": "maybe", "confidence": 0.9, "reason": ""}',  # not an option
        '{"answer": "different", "reason": ""}',  # no confidence
        '{"answer": "different", "confidence": "high", "reason": ""}',
        '{"answer": "different", "confidence": true, "reason": ""}',
        '{"answer": "different", "confidence": NaN, "reason": ""}',
        '{"answer": "different", "confidence": -0.1, "reason": ""}',
        '{"answer": "different", "confidence": 120, "reason": ""}',
        '{"answer": 3, "confidence": 0.5, "reason": ""}',
        '["different"]',
        '{"answer": "different", "confidence": 0.5',  # unterminated
    ],
)
def test_invalid_replies_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_answer(text, OPTS)


def _present(monkeypatch, name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    monkeypatch.setitem(sys.modules, name, mod)


def test_dependency_check_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-vlm\]") as err:
        check_vlm_dependencies(VLMConfig(backend="openai_compat", model="m"))
    assert "vlm.backend: none" in str(err.value)
    with pytest.raises(ImportError, match=r"dnt\[refine-vlm\]"):
        check_vlm_dependencies(VLMConfig(backend="anthropic"))
    check_vlm_dependencies(VLMConfig(backend="none"))  # needs nothing


def test_an_installed_package_passes(monkeypatch):
    _present(monkeypatch, "openai")
    check_vlm_dependencies(VLMConfig(backend="openai_compat", model="m"))
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))  # no __spec__
    check_vlm_dependencies(VLMConfig(backend="anthropic"))


def test_make_backend_rejects_none_and_unknown():
    for name in ("none", "nope"):
        with pytest.raises(ValueError, match="no VLM backend"):
            make_backend(VLMConfig(backend=name))


def test_the_default_model_is_pinned():
    assert DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-5-5"
