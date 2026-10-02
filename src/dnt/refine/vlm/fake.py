"""A scripted VLM backend for tests (spec 7.3)."""

from __future__ import annotations

import json

from . import VLMAnswer, parse_answer


class FakeBackend:
    """Return scripted replies; record every call in ``calls``."""

    def __init__(self, script, *, name: str = "fake", model: str = "fake-1"):
        """``script``: a dict (tag, event id, kind or ``"*"`` -> item or list) or a callable."""
        self.script = script
        self.name = name
        self.model = model
        self.calls: list[dict] = []
        self.closed = 0
        self._pos: dict[str, int] = {}

    async def aclose(self) -> None:
        """Count the close (the runner closes its backend once)."""
        self.closed += 1

    def _item(self, tag: str, options, temperature):
        if callable(self.script):
            return self.script(tag, options, temperature)
        kind, _, event_id = tag.partition(":")
        for key in (tag, event_id, kind, "*"):
            if key and key in self.script:
                entry = self.script[key]
                if not isinstance(entry, list):
                    return entry
                i = self._pos.get(key, 0)
                self._pos[key] = i + 1
                return entry[min(i, len(entry) - 1)]
        raise RuntimeError(f"no scripted answer for {tag!r}")

    async def ask(self, image_jpeg, prompt, options, temperature, *, tag: str = "") -> VLMAnswer:
        """Return the scripted reply for ``tag`` (parsed like a real reply)."""
        self.calls.append(
            {
                "tag": tag,
                "options": list(options),
                "temperature": temperature,
                "prompt": prompt,
                "image_len": len(image_jpeg),
            }
        )
        item = self._item(tag, options, temperature)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, VLMAnswer):
            return item
        text = item if isinstance(item, str) else json.dumps(item)
        return parse_answer(text, options)
