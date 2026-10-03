"""VLM verification backends (spec 7.3): protocol, errors, answer parsing, and the factory.

``openai`` and ``anthropic`` are imported only inside the backend constructors.
"""

from __future__ import annotations

import contextlib
import importlib.util
import ipaddress
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from ..config import VLMConfig

log = logging.getLogger(__name__)

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


def missing_key_message(backend: str, env: str) -> str:
    """Return the error for a backend that found no API key; it names every way to give one."""
    return (
        f"vlm.backend: {backend} needs an API key and none was found. Pass it in Python "
        f"(TrackRefiner(..., vlm_api_key=...)), set vlm.api_key_file (--vlm-api-key-file) to "
        f"a file that holds it, or set the {env} environment variable (vlm.api_key_env, "
        "--vlm-api-key-env, names another variable)"
    )


#: The largest key file read; a longer one is an error (a key is a few hundred bytes at most).
KEY_FILE_MAX_BYTES = 64 * 1024
_ENV_ASSIGNMENT = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)(=+)(.*)", re.DOTALL)
_SHOUTED_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")  # OPENAI_API_KEY, MY_KEY, ...
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


def key_problem(key: str) -> str | None:
    """Return why ``key`` (already stripped) cannot be an API key, or None if it can.

    The reason never quotes the key: a key is printable ASCII with no whitespace; a ``.env``
    line (``NAME=value``, ``NAME=``, ``NAME==value``), a quoted key, or several lines are
    reported as such. Trailing base64 padding (``abcd=``, ``abcdef==``) is allowed when the
    whole key is a multiple of 4 characters long and does not look like a variable name.
    """
    if "\n" in key or "\r" in key:
        return "has more than one line"
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        return "is wrapped in quotes; put only the key in it, without quotes"
    m = _ENV_ASSIGNMENT.match(key)
    if m is not None:
        name, eqs, rest = m.groups()
        padding = (
            not rest
            and len(eqs) <= 2
            and key == name + eqs
            and len(key) % 4 == 0
            and not _SHOUTED_NAME.fullmatch(name)
        )
        if not padding:
            return "looks like NAME=value; put only the key in it"
    if any(c.isspace() for c in key):
        return "contains whitespace"
    if any(not (32 < ord(c) < 127) for c in key):
        return "contains non-ASCII or control characters"
    return None


def clean_api_key(value, source: str) -> str | None:
    """Strip ``value``; return None if it is blank, the key if it is valid.

    Raises
    ------
    ValueError
        If ``value`` is not a string or is not a valid key; the message names ``source`` (an
        argument, a variable, or a file), never the value.

    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string")
    key = value.strip()
    if not key:
        return None
    why = key_problem(key)
    if why is not None:
        raise ValueError(f"{source} {why}")
    return key


def _read_key_file(path: Path) -> bytes:
    """Read at most ``KEY_FILE_MAX_BYTES + 1`` bytes; a FIFO with no writer reads as empty."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)  # non-blocking open: a FIFO with no writer must not hang
    try:
        if getattr(os, "O_NONBLOCK", 0):
            os.set_blocking(fd, True)  # then read normally, to the end of what is written
        chunks, size = [], 0
        while size <= KEY_FILE_MAX_BYTES:
            chunk = os.read(fd, KEY_FILE_MAX_BYTES + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def resolve_api_key(cfg: VLMConfig, default_env: str, runtime_key: str | None = None) -> str | None:
    """Return the API key for a backend, or None when no source has one.

    The sources, first match wins:

    1. ``runtime_key``: the ``vlm_api_key`` argument of ``TrackRefiner``;
    2. ``cfg.api_key_file``: a file whose content, stripped of surrounding whitespace (and a
       UTF-8 byte order mark), is the key (``~`` is expanded; at most 64 KiB is read);
    3. the environment variable ``cfg.api_key_env``, or ``default_env`` when that is unset (a
       blank value counts as unset).

    Each key is checked with ``key_problem``: one line of printable ASCII without whitespace.
    The key itself never goes into the config, a message, or a log line.

    Raises
    ------
    ValueError
        If ``cfg.api_key_file`` is set but cannot be read, is too large, is empty, or does not
        hold a valid key, or a runtime or environment key is not valid. The message names the
        file, argument or variable, never the content.

    """
    key = clean_api_key(runtime_key, "the vlm_api_key argument")
    if key is not None:
        return key
    key_file = getattr(cfg, "api_key_file", None)
    if key_file:
        path = Path(key_file).expanduser()
        where = f"the VLM key file {path} (vlm.api_key_file)"
        # every ValueError below is raised outside an ``except`` block, so it carries no
        # exception context: a UnicodeDecodeError's ``object`` would hold the file's bytes
        raw, why = None, None
        try:
            raw = _read_key_file(path)
        except OSError as exc:
            why = exc.strerror or type(exc).__name__
        if raw is None:
            raise ValueError(f"cannot read {where}: {why}")
        if len(raw) > KEY_FILE_MAX_BYTES:
            raise ValueError(f"{where} is larger than {KEY_FILE_MAX_BYTES // 1024} KiB")
        text = None
        with contextlib.suppress(UnicodeDecodeError):
            text = raw.decode("utf-8-sig")
        if text is None:
            raise ValueError(f"{where} is not UTF-8 text")
        key = clean_api_key(text, where)
        if key is None:
            raise ValueError(f"{where} is empty")
        return key
    env = cfg.api_key_env or default_env
    return clean_api_key(os.environ.get(env), f"the {env} environment variable")


def warn_if_cleartext(url: str | None, backend: str) -> None:
    """Log one warning if the key would go over plain ``http://`` to a host that is not local.

    Only the host is logged (validation already forbids a user name, password, or query).
    """
    if not url:
        return
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
    except ValueError:
        return
    if parts.scheme.lower() != "http":
        return
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host.lower() in _LOOPBACK_NAMES or host.lower().endswith(".localhost")
    if not local:
        log.warning(
            "vlm.backend %s: the endpoint uses http:// to %s, so the API key is sent "
            "unencrypted; use https:// unless the network is trusted",
            backend,
            host,
        )


def make_backend(cfg: VLMConfig, api_key: str | None = None) -> VLMBackend:
    """Build the backend selected by ``cfg.backend``.

    ``api_key`` is a key given at run time (``TrackRefiner(vlm_api_key=...)``); it takes
    precedence over ``vlm.api_key_file`` and the environment (see ``resolve_api_key``).

    Raises
    ------
    ValueError
        If ``cfg.backend`` is ``none`` or unknown, a required key is not found, or the key file
        cannot be read.

    """
    if cfg.backend == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg, api_key=api_key)
    if cfg.backend == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(cfg, api_key=api_key)
    raise ValueError(f"no VLM backend for vlm.backend={cfg.backend!r}")
