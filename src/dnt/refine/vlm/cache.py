"""On-disk cache of VLM answers (spec 7.4): one JSON file per question."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
from pathlib import Path

from . import VLMAnswer

log = logging.getLogger(__name__)


class AnswerCache:
    """Best-effort answer cache keyed by the image, the question, and the model settings."""

    def __init__(self, directory):
        """Use ``directory`` (``~`` is expanded); it is created on the first write."""
        self.dir = Path(directory).expanduser()

    @staticmethod
    def key(image_jpeg, prompt, options, backend, model, temperature, vote_index) -> str:
        """Return the SHA-256 of the image, prompt, options, backend, model, temperature, vote."""
        h = hashlib.sha256()
        h.update(hashlib.sha256(image_jpeg).digest())
        parts = [prompt, list(options), backend, model, round(float(temperature), 6)]
        parts.append(int(vote_index))
        h.update(json.dumps(parts).encode())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str, options: list[str] | None = None) -> VLMAnswer | None:
        """Return the stored answer, or ``None`` if it is missing, damaged, or not a valid answer.

        An entry is valid when its answer is a string (one of ``options`` when given) and its
        confidence is a finite number in [0, 1] that is not a bool: the same rules a fresh reply
        must pass, so a damaged entry can never decide an event.
        """
        try:
            d = json.loads(self._path(key).read_text())
            answer, conf = d["answer"], d["confidence"]
            if not isinstance(answer, str) or (options is not None and answer not in options):
                return None
            if isinstance(conf, bool) or not isinstance(conf, int | float):
                return None
            if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
                return None
            return VLMAnswer(answer, float(conf), str(d["reason"]), str(d["raw"]))
        except Exception:
            return None

    def put(self, key: str, answer: VLMAnswer) -> None:
        """Store an answer atomically; a failure is logged and ignored."""
        path = self._path(key)
        tmp = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(
                    {
                        "answer": answer.answer,
                        "confidence": answer.confidence,
                        "reason": answer.reason,
                        "raw": answer.raw,
                    }
                )
            )
            os.replace(tmp, path)
        except OSError as err:
            log.warning("could not write the VLM answer cache %s: %s", path, err)
            with contextlib.suppress(OSError):  # e.g. the parent is not a directory
                tmp.unlink(missing_ok=True)
