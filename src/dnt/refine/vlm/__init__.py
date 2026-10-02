"""VLM verification backends (spec 7.3): protocol, errors, answer parsing, and the factory.

``openai`` and ``anthropic`` are imported only inside the backend constructors.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..config import VLMConfig

#: For each backend: the module it needs and the pip extra that provides it.
BACKEND_REQUIRES = {
    "openai_compat": ("openai", "refine-vlm"),
    "anthropic": ("anthropic", "refine-vlm"),
}
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class VLMTransientError(Exception):
    """A failure worth retrying: a timeout, HTTP 429, HTTP 5xx, or a dropped connection."""


@dataclass
class VLMAnswer:
    """One parsed answer: the chosen option, its confidence, a reason, and the raw reply."""

    answer: str
    confidence: float
    reason: str
    raw: str


class VLMBackend(Protocol):
    """Asks a vision-language model one multiple-choice question about one image."""

    name: str
    model: str

    async def ask(
        self,
        image_jpeg: bytes,
        prompt: str,
        options: list[str],
        temperature: float,
        *,
        tag: str = "",
    ) -> VLMAnswer:
        """Return the model's parsed answer; raise ``VLMTransientError`` to ask for a retry."""
        ...


def _json_object(text: str) -> dict:
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    start = body.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(body)):
        ch = body[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(body[start : i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("the reply's JSON is not an object")
                return obj
    raise ValueError("unterminated JSON object in the reply")


def parse_answer(text: str, options: list[str]) -> VLMAnswer:
    """Parse a model reply into a ``VLMAnswer``.

    The JSON object may sit inside a code fence or in surrounding text. The answer must be one
    of ``options`` and the confidence a number in [0, 1] (a value in (1, 100] is read as a
    percentage).

    Raises
    ------
    ValueError
        If the reply holds no valid answer.

    """
    obj = _json_object(text)
    ans = obj.get("answer")
    if not isinstance(ans, str) or ans.strip() not in options:
        raise ValueError(f"answer {ans!r} is not one of {options}")
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, int | float) or not math.isfinite(conf):
        raise ValueError("confidence must be a finite number")
    conf = float(conf)
    if 1.0 < conf <= 100.0:
        conf /= 100.0
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence {conf} is outside [0, 1]")
    reason = obj.get("reason", "")
    reason = reason if isinstance(reason, str) else str(reason)
    return VLMAnswer(ans.strip(), conf, reason.strip()[:500], text)


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):  # a loaded module whose __spec__ is None
        return sys.modules.get(module) is not None


def check_vlm_dependencies(cfg: VLMConfig) -> None:
    """Raise ``ImportError`` naming the pip extra if the backend's package is not installed."""
    if cfg.backend == "none":
        return
    module, extra = BACKEND_REQUIRES[cfg.backend]
    if not _importable(module):
        raise ImportError(
            f"vlm.backend={cfg.backend!r} needs the {module!r} package. Install it with "
            f"pip install 'dnt[{extra}]', or set vlm.backend: none to leave uncertain events "
            "for review."
        )


def make_backend(cfg: VLMConfig) -> VLMBackend:
    """Build the backend selected by ``cfg.backend``.

    Raises
    ------
    ValueError
        If ``cfg.backend`` is ``none`` or unknown, or a required key is not set.

    """
    if cfg.backend == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg)
    if cfg.backend == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(cfg)
    raise ValueError(f"no VLM backend for vlm.backend={cfg.backend!r}")
