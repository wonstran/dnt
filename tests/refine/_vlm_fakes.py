"""Fake ``openai`` and ``anthropic`` packages for the backend tests."""

from __future__ import annotations

import importlib.machinery
import sys
import types


def _module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(Exception):
    def __init__(self, message="", status_code=500):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIStatusError):
    def __init__(self, message=""):
        super().__init__(message, 429)


class InternalServerError(APIStatusError):
    def __init__(self, message=""):
        super().__init__(message, 500)


def _errors(mod):
    # module-level classes, shared by every install, so an exception built from one fake
    # module is still caught after the test installs the module again
    for cls in (
        APIConnectionError, APITimeoutError, APIStatusError, RateLimitError, InternalServerError
    ):
        setattr(mod, cls.__name__, cls)


def install_fake_openai(monkeypatch, replies):
    """Install a fake ``openai``; ``replies`` is a list of reply texts or exceptions."""
    seen = {"requests": [], "client_kwargs": None, "closed": 0}
    queue = list(replies)
    mod = _module("openai")
    _errors(mod)

    class _Completions:
        async def create(self, **kwargs):
            seen["requests"].append(kwargs)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            msg = types.SimpleNamespace(content=item)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class AsyncOpenAI:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

        async def close(self):
            seen["closed"] += 1

    mod.AsyncOpenAI = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return seen


def install_fake_anthropic(monkeypatch, replies):
    """Install a fake ``anthropic``; ``replies`` is a list of reply texts or exceptions."""
    seen = {"requests": [], "client_kwargs": None, "closed": 0}
    queue = list(replies)
    mod = _module("anthropic")
    _errors(mod)

    class _Messages:
        async def create(self, **kwargs):
            seen["requests"].append(kwargs)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            block = types.SimpleNamespace(type="text", text=item)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="thinking"), block])

    class AsyncAnthropic:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs
            self.messages = _Messages()

        async def close(self):
            seen["closed"] += 1

    mod.AsyncAnthropic = AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return seen
