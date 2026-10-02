"""Anthropic Messages API backend (spec 7.3)."""

from __future__ import annotations

import base64
import os

from . import (
    DEFAULT_ANTHROPIC_MODEL,
    VLMAnswer,
    VLMTransientError,
    missing_key_message,
    parse_answer,
    resolve_api_key,
    warn_if_cleartext,
)

#: Model id prefixes of the Claude models that reject a non-default ``temperature`` (HTTP 400),
#: think adaptively by default, and take an effort level instead.
_NEW_FAMILY_PREFIXES = (
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable",
    "claude-mythos",
)
_NEW_FAMILY_MAX_TOKENS = 2048  # thinking tokens count against max_tokens
_OLDER_MAX_TOKENS = 1024


def _new_family(model: str) -> bool:
    """Return whether ``model`` belongs to the newer Claude models.

    Those models answer HTTP 400 to a non-default ``temperature``, ``top_p`` or ``top_k``, run
    adaptive thinking by default, and accept an effort level; older models (for example
    ``claude-haiku-4-5`` or ``claude-sonnet-4-6``) take a ``temperature`` and no effort.
    """
    return str(model).startswith(_NEW_FAMILY_PREFIXES)


class AnthropicBackend:
    """Ask Claude; the image goes in as a base64 image block."""

    name = "anthropic"

    def __init__(self, cfg, *, api_key: str | None = None):
        """Create the client; ``vlm.base_url``, when set, replaces Anthropic's endpoint.

        The key is ``api_key``, else the content of ``vlm.api_key_file``, else the variable
        named by ``vlm.api_key_env`` (``ANTHROPIC_API_KEY``).

        Raises
        ------
        ValueError
            If no key is found, or the key file cannot be read.

        """
        from anthropic import AsyncAnthropic

        key = resolve_api_key(cfg, "ANTHROPIC_API_KEY", api_key)
        if key is None:
            raise ValueError(
                missing_key_message("anthropic", cfg.api_key_env or "ANTHROPIC_API_KEY")
            )
        # the SDK reads ANTHROPIC_BASE_URL itself when no base_url is given
        warn_if_cleartext(cfg.base_url or os.environ.get("ANTHROPIC_BASE_URL"), self.name)
        self.secret = key  # the runner scrubs it from every error text
        self.model = cfg.model or DEFAULT_ANTHROPIC_MODEL
        extra = {"base_url": cfg.base_url} if cfg.base_url else {}
        self._client = AsyncAnthropic(api_key=key, timeout=cfg.timeout_s, max_retries=0, **extra)

    def _request(self, content: list, temperature: float) -> dict:
        """Return the keyword arguments of ``messages.create`` for this model.

        A newer model gets no sampling parameters (it would answer HTTP 400), effort ``low``,
        and room for its thinking; an older model gets ``temperature`` and no effort.
        """
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if _new_family(self.model):
            kwargs["max_tokens"] = _NEW_FAMILY_MAX_TOKENS
            # extra_body works on every SDK version; a named output_config argument may not
            kwargs["extra_body"] = {"output_config": {"effort": "low"}}
        else:
            kwargs["max_tokens"] = _OLDER_MAX_TOKENS
            kwargs["temperature"] = float(temperature)
        return kwargs

    async def ask(self, image_jpeg, prompt, options, temperature, *, tag: str = "") -> VLMAnswer:
        """Return the parsed answer; timeouts, 429 and 5xx raise ``VLMTransientError``.

        ``temperature`` is sent only to models that accept sampling parameters (see
        ``_new_family``). A refusal, a reply cut off at ``max_tokens`` before its answer, or a
        reply with no text raises ``RuntimeError``, which the runner does not retry.
        """
        import anthropic

        data = base64.b64encode(image_jpeg).decode()
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
            },
            {"type": "text", "text": prompt},
        ]
        try:
            resp = await self._client.messages.create(**self._request(content, temperature))
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise VLMTransientError(type(exc).__name__) from None
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise VLMTransientError(f"HTTP {exc.status_code}") from None
            raise
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            raise RuntimeError("the model refused (stop_reason=refusal)")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if stop == "max_tokens":
            try:
                return parse_answer(text, options)
            except ValueError:
                raise RuntimeError(
                    "the model stopped at max_tokens before giving an answer"
                ) from None
        if not text.strip():
            raise RuntimeError(f"the reply has no text (stop_reason={stop})")
        return parse_answer(text, options)

    async def aclose(self) -> None:
        """Close the HTTP client; the runner calls this on the event loop that used it."""
        await self._client.close()
