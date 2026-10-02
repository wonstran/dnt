"""OpenAI-compatible chat backend: vLLM, Ollama, or any server speaking that API (spec 7.3)."""

from __future__ import annotations

import base64
import os

from . import VLMAnswer, VLMTransientError, parse_answer


class OpenAICompatBackend:
    """Ask a chat-completions endpoint; the image goes in as a base64 data URL."""

    name = "openai_compat"

    def __init__(self, cfg):
        """Create the client. A local server needs no key; a placeholder is sent."""
        from openai import AsyncOpenAI

        self.model = cfg.model
        self._json_mode = bool(cfg.json_mode)
        key = os.environ.get(cfg.api_key_env or "OPENAI_API_KEY") or "EMPTY"
        self._client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=key, timeout=cfg.timeout_s, max_retries=0
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
            "max_tokens": 300,
        }
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
        return parse_answer(resp.choices[0].message.content or "", options)

    async def aclose(self) -> None:
        """Close the HTTP client; the runner calls this on the event loop that used it."""
        await self._client.close()
