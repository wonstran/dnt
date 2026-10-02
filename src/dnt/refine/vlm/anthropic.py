"""Anthropic Messages API backend (spec 7.3)."""

from __future__ import annotations

import base64
import os

from . import DEFAULT_ANTHROPIC_MODEL, VLMAnswer, VLMTransientError, parse_answer


class AnthropicBackend:
    """Ask Claude; the image goes in as a base64 image block."""

    name = "anthropic"

    def __init__(self, cfg):
        """Create the client; the key comes from ``vlm.api_key_env`` (``ANTHROPIC_API_KEY``)."""
        from anthropic import AsyncAnthropic

        env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(env)
        if not key:
            raise ValueError(f"set the {env} environment variable to use vlm.backend: anthropic")
        self.model = cfg.model or DEFAULT_ANTHROPIC_MODEL
        self._client = AsyncAnthropic(api_key=key, timeout=cfg.timeout_s, max_retries=0)

    async def ask(self, image_jpeg, prompt, options, temperature, *, tag: str = "") -> VLMAnswer:
        """Return the parsed answer; timeouts, 429 and 5xx raise ``VLMTransientError``."""
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
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=float(temperature),
                messages=[{"role": "user", "content": content}],
            )
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise VLMTransientError(type(exc).__name__) from None
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise VLMTransientError(f"HTTP {exc.status_code}") from None
            raise
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return parse_answer(text, options)

    async def aclose(self) -> None:
        """Close the HTTP client; the runner calls this on the event loop that used it."""
        await self._client.close()
