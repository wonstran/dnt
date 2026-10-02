"""Opt-in smoke tests against real VLM backends: ``pytest -m live``.

Each test is skipped (not failed) when its package is not installed or its environment
variables are not set:

- anthropic: ``ANTHROPIC_API_KEY`` (``DNT_LIVE_ANTHROPIC_MODEL`` picks another model);
- openai_compat: ``DNT_LIVE_OPENAI_BASE_URL`` and ``DNT_LIVE_OPENAI_MODEL``, and
  ``OPENAI_API_KEY`` if the server needs one.
"""

import asyncio
import importlib.util
import os

import numpy as np
import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import VLMAnswer, make_backend

pytestmark = pytest.mark.live

OPTIONS = ["red", "blue", "unsure"]
PROMPT = (
    "The image is a single solid color. Which color is it? Reply with only a JSON object: "
    '{"answer": one of "red", "blue", "unsure", "confidence": a number from 0 to 1, '
    '"reason": a few words}.'
)


def _jpeg() -> bytes:
    import cv2

    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = (0, 0, 255)  # BGR: red
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _need(module: str, *env: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"the {module!r} package is not installed")
    missing = [name for name in env if not os.environ.get(name)]
    if missing:
        pytest.skip(f"set {', '.join(missing)} to run this live test")


def _ask(cfg: VLMConfig) -> VLMAnswer:
    backend = make_backend(cfg)

    async def go():
        try:
            return await backend.ask(_jpeg(), PROMPT, OPTIONS, 0.0, tag="LIVE:smoke")
        finally:
            await backend.aclose()

    return asyncio.run(go())


def _check(ans: VLMAnswer) -> None:
    assert isinstance(ans, VLMAnswer)
    assert ans.answer in OPTIONS and 0.0 <= ans.confidence <= 1.0


def test_live_anthropic():
    _need("anthropic", "ANTHROPIC_API_KEY")
    model = os.environ.get("DNT_LIVE_ANTHROPIC_MODEL") or None
    _check(_ask(VLMConfig(backend="anthropic", model=model, timeout_s=120.0)))


def test_live_openai_compat():
    _need("openai", "DNT_LIVE_OPENAI_BASE_URL", "DNT_LIVE_OPENAI_MODEL")
    cfg = VLMConfig(
        backend="openai_compat",
        base_url=os.environ["DNT_LIVE_OPENAI_BASE_URL"],
        model=os.environ["DNT_LIVE_OPENAI_MODEL"],
        timeout_s=120.0,
    )
    _check(_ask(cfg))
