"""Runs VLM questions with votes, retries, a call budget, bounded concurrency, and a cache."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from collections import Counter
from dataclasses import dataclass, field

from . import VLMAnswer, VLMTransientError
from .cache import AnswerCache

log = logging.getLogger(__name__)
_BACKOFF_FIRST_S = 2.0
_ATTEMPTS = 3
_DEFAULT_KEY_ENVS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@dataclass
class Question:
    """One multiple-choice question about one image."""

    tag: str
    image: bytes
    prompt: str
    options: list[str]


@dataclass
class Verdict:
    """The outcome of a question: an answer, or ``error`` (and then ``answer`` is None)."""

    answer: str | None
    confidence: float
    reason: str
    votes: dict[str, int]
    cached: bool
    error: str | None
    raw: list[str] = field(default_factory=list)


def _check(answer: VLMAnswer, options: list[str]) -> None:
    """Raise ``ValueError`` unless ``answer`` is an option with a confidence in [0, 1].

    The bundled backends already parse replies strictly; this guards a custom backend that
    returns a ``VLMAnswer`` object directly.
    """
    c = answer.confidence
    if answer.answer not in options:
        raise ValueError(f"answer {answer.answer!r} is not one of {options}")
    if isinstance(c, bool) or not isinstance(c, int | float) or not math.isfinite(c):
        raise ValueError("confidence must be a finite number")
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"confidence {c} is outside [0, 1]")


class VLMRunner:
    """Ask many questions of one backend, within a budget, on one long-lived event loop."""

    def __init__(self, cfg, backend, cache: AnswerCache | None = None, *, sleep=None):
        """``cfg`` is a ``VLMConfig``; ``sleep`` replaces ``asyncio.sleep`` in tests."""
        self.cfg = cfg
        self.backend = backend
        self.cache = cache
        self._sleep = sleep
        self.calls = 0
        self.retries = 0
        self.cache_hits = 0
        self.failures = 0
        self.budget_skipped = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    # ---- the long-lived loop ----

    @staticmethod
    def _serve(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("the VLM runner is closed")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                ready = threading.Event()
                thread = threading.Thread(
                    target=self._serve, args=(loop, ready), daemon=True, name="dnt-vlm-loop"
                )
                thread.start()
                ready.wait()
                self._loop, self._thread = loop, thread
            return self._loop

    def close(self) -> None:
        """Close the backend's client on the loop that used it, then stop the loop (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is None:
            return
        aclose = getattr(self.backend, "aclose", None)
        if aclose is not None:
            try:
                asyncio.run_coroutine_threadsafe(aclose(), loop).result(timeout=10.0)
            except Exception as err:
                log.warning("could not close the VLM backend cleanly: %s", err)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10.0)

    def __enter__(self) -> VLMRunner:
        """Return the runner; it is closed on exit."""
        return self

    def __exit__(self, *exc) -> None:
        """Close the runner."""
        self.close()

    # ---- one question ----

    def _scrub(self, text: str) -> str:
        names = {self.cfg.api_key_env, *_DEFAULT_KEY_ENVS} - {None}
        for name in names:
            secret = os.environ.get(name)
            if secret:
                text = text.replace(secret, "***")
        return text[:300]

    def _take_retry(self, slack: list[int]) -> bool:
        """Spend one unit of the retry allowance; False (and nothing spent) when it is empty."""
        if slack[0] <= 0:
            return False
        slack[0] -= 1
        self.retries += 1
        return True

    async def _ask_backend(self, q: Question, temperature: float, slack: list[int]):
        """Send one vote, retrying transient failures and one invalid reply; return (ans, err).

        Every retry takes a unit of ``slack``; with none left the result is ``(None, "budget")``.
        """
        sleep = self._sleep or asyncio.sleep
        prompt = q.prompt
        invalid = ""
        for invalid_try in range(2):
            answer = None
            for attempt in range(_ATTEMPTS):
                try:
                    got = await asyncio.wait_for(
                        self.backend.ask(q.image, prompt, q.options, temperature, tag=q.tag),
                        float(self.cfg.timeout_s),
                    )
                    _check(got, q.options)
                    answer = got  # only a validated answer is ever kept
                    break
                except (VLMTransientError, TimeoutError) as exc:
                    if attempt == _ATTEMPTS - 1:
                        what = self._scrub(f"{type(exc).__name__} {exc}".strip())
                        return None, f"transient: {what} after {_ATTEMPTS} attempts"
                    if not self._take_retry(slack):
                        return None, "budget"
                    await sleep(_BACKOFF_FIRST_S * 2**attempt)
                except ValueError as exc:  # the reply was not a valid answer
                    invalid = self._scrub(str(exc))
                    break
                except Exception as exc:
                    return None, self._scrub(f"{type(exc).__name__}: {exc}")
            if answer is not None:
                return answer, None
            if invalid_try == 0:
                if not self._take_retry(slack):
                    return None, "budget"
                prompt = (
                    q.prompt
                    + f"\n\nYour previous reply was not valid ({invalid}). Reply with only the "
                    'JSON object, with "answer" set to exactly one of: '
                    + ", ".join(q.options)
                    + "."
                )
        return None, f"invalid output: {invalid}"

    def _keys_and_hits(self, q: Question, n: int, temperature: float):
        keys, hits = [], []
        for i in range(n):
            key = ans = None
            if self.cache is not None:
                key = self.cache.key(
                    q.image,
                    q.prompt,
                    q.options,
                    self.backend.name,
                    self.backend.model,
                    temperature,
                    i,
                )
                ans = self.cache.get(key, q.options)
            keys.append(key)
            hits.append(ans)
        return keys, hits

    async def _one(
        self, q: Question, n: int, temperature: float, keys, hits, slack: list[int]
    ) -> Verdict:
        answers: list[VLMAnswer] = []
        for i in range(n):
            ans = hits[i]
            if ans is not None:
                self.cache_hits += 1
            else:
                self.calls += 1
                ans, err = await self._ask_backend(q, temperature, slack)
                if ans is None:
                    votes = dict(Counter(a.answer for a in answers))
                    return Verdict(None, 0.0, "", votes, False, err, [a.raw for a in answers])
                if keys[i] is not None:
                    self.cache.put(keys[i], ans)
            answers.append(ans)
        counts = Counter(a.answer for a in answers)
        ranked = counts.most_common()
        raws = [a.raw for a in answers]
        all_cached = all(h is not None for h in hits)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return Verdict(None, 0.0, "", dict(counts), all_cached, "tie", raws)
        best = ranked[0][0]
        first = next(a for a in answers if a.answer == best)
        conf = first.confidence if n == 1 else ranked[0][1] / n
        return Verdict(best, conf, first.reason, dict(counts), all_cached, None, raws)

    async def _gather(self, questions: list[Question]) -> list[Verdict]:
        n = max(1, int(self.cfg.votes))
        temperature = float(self.cfg.vote_temperature) if n > 1 else 0.0
        # Admission, in list order and before anything is sent: cached votes are free, and a
        # question is admitted only if all its uncached votes fit in what is left of the budget.
        left = max(0, int(self.cfg.max_calls) - self.calls - self.retries)
        plans = []
        for q in questions:
            keys, hits = self._keys_and_hits(q, n, temperature)
            need = sum(h is None for h in hits)
            if need > left:
                plans.append(None)
            else:
                left -= need
                plans.append((keys, hits))
        slack = [left]  # what admission left over: the retries of this call draw from it
        sem = asyncio.Semaphore(max(1, int(self.cfg.max_concurrency)))

        async def one(q: Question, plan) -> Verdict:
            if plan is None:
                self.budget_skipped += 1
                return Verdict(None, 0.0, "", {}, False, "budget", [])
            async with sem:
                v = await self._one(q, n, temperature, *plan, slack)
            if v.error == "budget":
                self.budget_skipped += 1
            elif v.error is not None:
                self.failures += 1
            return v

        tasks = [one(q, p) for q, p in zip(questions, plans, strict=True)]
        return list(await asyncio.gather(*tasks))

    def ask_many(self, questions: list[Question]) -> list[Verdict]:
        """Ask every question and return the verdicts in order (blocks until all are done)."""
        if not questions:
            return []
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(self._gather(questions), loop).result()
