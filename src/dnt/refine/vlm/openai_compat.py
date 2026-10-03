"""OpenAI-compatible chat backend: vLLM, Ollama, or any server speaking that API (spec 7.3)."""

from __future__ import annotations

import base64
import os

from . import (
    VLMAnswer,
    VLMTransientError,
    missing_key_message,
    parse_answer,
    resolve_api_key,
    warn_if_cleartext,
)

#: Output budget when ``vlm.max_tokens`` is unset; a reasoning model spends part of it thinking.
DEFAULT_MAX_TOKENS = 1024


class OpenAICompatBackend:
    """Ask a chat-completions endpoint; the image goes in as a base64 data URL."""

    name = "openai_compat"

    def __init__(self, cfg, *, api_key: str | None = None):
        """Create the client.

        The key is ``api_key``, else the content of ``vlm.api_key_file``, else the variable
        named by ``vlm.api_key_env`` (``OPENAI_API_KEY``). A server at ``vlm.base_url`` (or
        ``OPENAI_BASE_URL``) may need no key, and a placeholder is sent; without either, the
        client would call api.openai.com, which needs one.

        Raises
        ------
        ValueError
            If no key is found and no endpoint is set, or the key file cannot be read.

        """
        from openai import AsyncOpenAI

        self.model = cfg.model
        self._json_mode = bool(cfg.json_mode)
        self._max_tokens = cfg.max_tokens or DEFAULT_MAX_TOKENS
        self._extra_body = dict(cfg.extra_body or {})
        key = resolve_api_key(cfg, "OPENAI_API_KEY", api_key)
        endpoint = cfg.base_url or os.environ.get("OPENAI_BASE_URL")
        if key is None and not endpoint:
            raise ValueError(
                missing_key_message("openai_compat", cfg.api_key_env or "OPENAI_API_KEY")
                + "; a local server needs vlm.base_url instead"
            )
        if key is not None:  # the placeholder is no secret
            warn_if_cleartext(endpoint, self.name)
        self.secret = key  # the runner scrubs it from every error text
        self._client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=key or "EMPTY", timeout=cfg.timeout_s, max_retries=0
        )

    async def ask(self, image_jpeg, prompt, options, temperature, *, tag: str = "") -> VLMAnswer:
        """Return the parsed answer; timeouts, 429 and 5xx raise ``VLMTransientError``."""
        import openai

        data = base64.b64encode(image_jpeg).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        ]
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature),
            "max_tokens": self._max_tokens,
        }
        if self._extra_body:  # provider switches, e.g. {"reasoning": {"enabled": false}}
            kwargs["extra_body"] = self._extra_body
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise VLMTransientError(type(exc).__name__) from None
        except openai.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise VLMTransientError(f"HTTP {exc.status_code}") from None
            raise
        choice = resp.choices[0]
        text = choice.message.content or ""
        try:
            return parse_answer(text, options)
        except ValueError:
            if getattr(choice, "finish_reason", None) == "length":
                raise RuntimeError(
                    f"the reply was cut off at max_tokens={self._max_tokens} before its answer "
                    "(a reasoning model spends the budget thinking): raise vlm.max_tokens or "
                    "turn off the model's reasoning"
                ) from None
            raise

    async def aclose(self) -> None:
        """Close the HTTP client; the runner calls this on the event loop that used it."""
        await self._client.close()
