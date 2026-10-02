# dnt.refine Track Refinement — Plan 3 of 4: VLM Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `TrackRefiner.refine` send the events that fall in the uncertain score band to a vision-language model (VLM), apply its sure answers during the run (`VLM_ACCEPT` / `VLM_REJECT`, including redirected edits and the rider subtype), leave everything else `HUMAN_PENDING`, and write a static review page for the pending events. This is spec §7, §8.1, and the VLM counters of §8.3.

**Architecture:**
- **A small `vlm` package.** A `VLMBackend` protocol with `openai_compat` (vLLM, Ollama, any OpenAI-compatible server), `anthropic`, and a scripted `fake` backend; one runner that adds votes, retries, a call budget, concurrency, an on-disk answer cache, and a sync public face over `asyncio`.
- **`evidence.py` builds one composite JPEG per event** (crops plus context frames) from the video, using P2's `FrameReader` and `crop_box`.
- **`verify.py` gains `route_with_vlm`.** It routes by band (as today), sends the uncertain band to the VLM in order of distance from the band midpoint, and maps answers to decisions and edits (spec §7.2). `_Stages._route` calls it when a backend is configured and a video is given; otherwise the P1 `route_without_vlm` path is unchanged.
- **`review.py` writes `OUT.review.html`** and `OUT.review/*.jpg` for the pending events, plus the `decisions.json` export that Plan 4's `apply` will read.

**Tech Stack:** Python >=3.11, numpy, OpenCV (already required), `asyncio`. Optional extra `refine-vlm = ["openai>=1.40", "anthropic>=0.40"]`, imported only inside the backend constructors.

**Spec:** [`docs/superpowers/specs/2026-09-27-track-refinement-design.md`](../specs/2026-09-27-track-refinement-design.md) (rev. 7 plus the Plan 2 follow-ups). Sections such as "§7.2" point there. Plans 1 and 2 are merged; this plan builds on their real interfaces (`Event`, `Decision`, `Band`, `decide`, `FrameReader`, `crop_box`, `TrackRefiner._Stages`).

## Plan series

| Plan | Scope | Status |
|---|---|---|
| P1 Core | package, stages, ledger, `refine`, `dnt-refine run` | merged |
| P2 Appearance | encoders, feature cache, dense rescoring | merged (plus the size rule and the lower link band) |
| **P3 (this plan)** | VLM backends, runner, cache, evidence, routing, review page | §7, §8.1, §8.3 VLM counters |
| P4 Replay and audit | `apply`, decisions, rounds, `audit`, reference case, quickstart | not written |

## Global Constraints

- **Dependencies.** `requires-python = ">=3.11"`. **No new required dependencies.** `openai` and `anthropic` are imported only inside the backend constructors (`refine-vlm` extra). `import dnt.refine` must not import them (a subprocess test pins it).
- **Dependency rule (§2.2).** `dnt.refine` never imports `dnt.track`, `dnt.detect`, `dnt.label`, `dnt.filter`, or `boxmot`. The review page only *prints* a `Labeler.draw_track_clips(...)` snippet; it does not import `Labeler`.
- **Proposal vs edit (§4.1).** `kind`, `params`, and `proposal_key` never change. A redirected answer changes `edit` only.
- **Which events go to a VLM.** Only `switch`, `screen`, and `link` events in the uncertain band, plus an `AUTO_ACCEPT` rider `RECLASS` that has no subtype yet. Orphan, fill, and smooth events never do.
- **Failures never abort a run and never apply an edit.** Timeout, 429, 5xx, invalid output, a budget stop, and any other error leave the event `HUMAN_PENDING` with `vlm.error`.
- **Secrets.** API keys come from environment variables named in the config (`vlm.api_key_env`, default `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). A key must never appear in the ledger, the config dict, a log line, or an exception message.
- **Defaults (§9).** `vlm.min_conf` 0.7, `votes` 1, `vote_temperature` 0.7, `max_calls` 500, `max_concurrency` 4, `timeout_s` 60, `send_context_frames` true, `cache_dir` `~/.cache/dnt/vlm`. The default Anthropic model is `claude-sonnet-5-5`.
- **Lint.** Ruff rules `E,F,I,UP,B,SIM,RUF,D` (line length 100, numpy-style docstrings); ASCII only; `zip(..., strict=True)`; no new entries in the pyproject per-file lint baseline; `ruff format src/dnt/refine` clean. Do not touch `site/` or the version numbers.

## Review Focus

The spec is silent on these inputs, and each would hurt a person running this on real clips. Each line has a test in the task that owns the code.

1. **A VLM reply that is almost right:** JSON in a code fence, text around it, a confidence written as a percentage or a string, an answer outside the options, an empty reply. Parse leniently, validate strictly, retry once, then leave the event pending. Never crash, never apply. (Tasks 1, 4)
2. **Cost control:** hundreds of uncertain events must stay within `max_calls`, closest-to-midpoint first; cache hits are free; a rerun on the same inputs makes zero backend calls; concurrency stays bounded; votes and retries each count against `max_calls`, a hard limit on backend invocations. (Tasks 2, 4, 6, 7)
3. **Secrets and privacy:** the key never reaches the ledger, logs, or error messages; images are sent only when a backend is configured; `send_context_frames: false` really leaves context frames out of the image. (Tasks 3, 5, 7)
4. **Evidence edge cases:** a track with fewer than the wanted clean crops, frames missing at the end of the video, boxes partly outside the frame, an event on the first frame, no video at all (cards show signals only). (Tasks 5, 8)
5. **Where it runs:** `refine` called from a thread that already has a running event loop (Jupyter) must work, and successive stage calls must reuse the backend's connections without `Event loop is closed` (tested against a local keep-alive HTTP server); the review page must escape every text it shows (a VLM `reason` containing `<script>` is plain text). (Tasks 4, 8)
6. **Damaged or hostile state on disk:** a readable cache entry with a NaN or out-of-range confidence, a boolean confidence, or an answer outside the question's options is a miss, never a verdict; a review directory holding the user's own `my-photo.jpg` keeps it, and a manifest cannot point outside the directory. (Tasks 2, 6, 8)
7. **Regenerated reports:** the same output name with different inputs reuses event ids such as `switch-r0-000001`; a saved browser choice is restored only for the same run and the same `proposal_key`. (Task 8)

---

### Task 1: The `vlm` package core: protocol, errors, answer parsing, factory

**Files:**
- Create: `src/dnt/refine/vlm/__init__.py`
- Test: `tests/refine/test_vlm_core.py`

**Interfaces:**
- Consumes: `dnt.refine.config.VLMConfig` (`backend`, `model`, ...).
- Produces:
  - `VLMTransientError(Exception)`: raised by a backend for a failure worth retrying.
  - `VLMAnswer(answer: str, confidence: float, reason: str, raw: str)` (dataclass).
  - `VLMBackend` (Protocol): attributes `name: str`, `model: str`; `async ask(image_jpeg: bytes, prompt: str, options: list[str], temperature: float, *, tag: str = "") -> VLMAnswer`. `tag` (`"<KIND>:<event id>"`) is for the scripted backend and logs; real backends ignore it and it is never part of the cache key.
  - `parse_answer(text: str, options: list[str]) -> VLMAnswer`: raises `ValueError` when the reply is not a valid answer.
  - `BACKEND_REQUIRES = {"openai_compat": ("openai", "refine-vlm"), "anthropic": ("anthropic", "refine-vlm")}`, `DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"`.
  - `check_vlm_dependencies(cfg: VLMConfig) -> None` (raises `ImportError` naming the extra), `make_backend(cfg: VLMConfig) -> VLMBackend`.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_vlm_core.py`:

```python
import importlib.machinery
import sys
import types

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import (
    DEFAULT_ANTHROPIC_MODEL,
    VLMAnswer,
    check_vlm_dependencies,
    make_backend,
    parse_answer,
)

OPTS = ["same_individual", "different", "unsure"]


def test_plain_json_is_parsed():
    a = parse_answer('{"answer": "different", "confidence": 0.9, "reason": "other clothes"}', OPTS)
    assert a == VLMAnswer("different", 0.9, "other clothes", a.raw)


def test_a_code_fence_and_text_around_the_json_are_tolerated():
    text = 'Sure!\n```json\n{"answer": "same_individual", "confidence": 0.8, "reason": "x"}\n```\nDone'
    assert parse_answer(text, OPTS).answer == "same_individual"
    text2 = 'I think {"answer": "unsure", "confidence": 0.2, "reason": "a {b} c"} ok'
    assert parse_answer(text2, OPTS).reason == "a {b} c"  # braces inside a string


def test_a_percentage_confidence_is_scaled():
    a = parse_answer('{"answer": "different", "confidence": 85, "reason": ""}', OPTS)
    assert a.confidence == pytest.approx(0.85)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        '{"answer": "maybe", "confidence": 0.9, "reason": ""}',  # not an option
        '{"answer": "different", "reason": ""}',  # no confidence
        '{"answer": "different", "confidence": "high", "reason": ""}',
        '{"answer": "different", "confidence": true, "reason": ""}',
        '{"answer": "different", "confidence": NaN, "reason": ""}',
        '{"answer": "different", "confidence": -0.1, "reason": ""}',
        '{"answer": "different", "confidence": 120, "reason": ""}',
        '{"answer": 3, "confidence": 0.5, "reason": ""}',
        '["different"]',
        '{"answer": "different", "confidence": 0.5',  # unterminated
    ],
)
def test_invalid_replies_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_answer(text, OPTS)


def _present(monkeypatch, name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    monkeypatch.setitem(sys.modules, name, mod)


def test_dependency_check_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-vlm\]") as err:
        check_vlm_dependencies(VLMConfig(backend="openai_compat", model="m"))
    assert "vlm.backend: none" in str(err.value)
    with pytest.raises(ImportError, match=r"dnt\[refine-vlm\]"):
        check_vlm_dependencies(VLMConfig(backend="anthropic"))
    check_vlm_dependencies(VLMConfig(backend="none"))  # needs nothing


def test_an_installed_package_passes(monkeypatch):
    _present(monkeypatch, "openai")
    check_vlm_dependencies(VLMConfig(backend="openai_compat", model="m"))
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))  # no __spec__
    check_vlm_dependencies(VLMConfig(backend="anthropic"))


def test_make_backend_rejects_none_and_unknown():
    for name in ("none", "nope"):
        with pytest.raises(ValueError, match="no VLM backend"):
            make_backend(VLMConfig(backend=name))


def test_the_default_model_is_pinned():
    assert DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-5-5"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_core.py -q`
Expected: collection error `No module named 'dnt.refine.vlm'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/vlm/__init__.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_core.py tests/test_refine_independence.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/vlm/__init__.py tests/refine/test_vlm_core.py
git commit -m "feat(refine): add the VLM backend protocol, answer parsing and dependency check" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Answer cache and prompts

**Files:**
- Create: `src/dnt/refine/vlm/cache.py`, `src/dnt/refine/vlm/prompts.py`
- Test: `tests/refine/test_vlm_cache_prompts.py`

**Interfaces:**
- Consumes: Task 1 (`VLMAnswer`); `dnt.refine.events.Event`, `EventKind`.
- Produces:
  - `AnswerCache(directory)`: `key(image_jpeg, prompt, options, backend, model, temperature, vote_index) -> str` (static; SHA-256 hex), `get(key, options=None) -> VLMAnswer | None`, `put(key, answer) -> None`. A hit is validated as strictly as a fresh reply: `answer` must be a string (and one of `options` when they are given), `confidence` a finite number in [0, 1] that is not a bool; anything else is a miss. One JSON file per key at `<dir>/<key[:2]>/<key>.json`; atomic writes; a missing, unreadable or malformed file is a miss; a write failure is logged and ignored (the cache is best-effort).
  - `prompts.PERSON_SCREEN`, `VEHICLE_SCREEN`, `SAME` (option lists), `RIDER_OPTIONS = {"cyclist": "cyclist", "motorcycle_rider": "motorcycle", "scooter_rider": "scooter"}` (option -> `reclass_map` key).
  - `options_for(event, target) -> list[str] | None` (None for events that are never sent to a VLM) and `build_prompt(event, target, options, *, fps) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_vlm_cache_prompts.py`:

```python
import json

import pytest

from dnt.refine.events import Event, EventKind
from dnt.refine.vlm import VLMAnswer
from dnt.refine.vlm.cache import AnswerCache
from dnt.refine.vlm.prompts import (
    PERSON_SCREEN,
    RIDER_OPTIONS,
    SAME,
    VEHICLE_SCREEN,
    build_prompt,
    options_for,
)


def _key(**over):
    args = dict(
        image_jpeg=b"img", prompt="p", options=["a", "b"], backend="fake", model="m",
        temperature=0.0, vote_index=0,
    )
    args.update(over)
    return AnswerCache.key(**args)


def test_the_key_depends_on_every_part():
    base = _key()
    assert base == _key() and len(base) == 64
    for over in (
        {"image_jpeg": b"img2"}, {"prompt": "q"}, {"options": ["a", "c"]}, {"backend": "x"},
        {"model": "m2"}, {"temperature": 0.7}, {"vote_index": 1},
    ):
        assert _key(**over) != base, over
    assert _key(options=["a", "b"]) != _key(options=["b", "a"])  # order matters


def test_put_get_round_trip_and_misses(tmp_path):
    c = AnswerCache(tmp_path / "vlm")
    k = _key()
    assert c.get(k) is None
    ans = VLMAnswer("different", 0.8, "r", '{"raw": 1}')
    c.put(k, ans)
    assert c.get(k) == ans
    path = tmp_path / "vlm" / k[:2] / f"{k}.json"
    assert path.is_file() and json.loads(path.read_text())["answer"] == "different"
    path.write_text("not json")
    assert c.get(k) is None  # damaged file is a miss
    path.write_text(json.dumps({"answer": 3}))
    assert c.get(k) is None  # wrong shape is a miss
    assert list((tmp_path / "vlm").rglob("*.tmp")) == []


@pytest.mark.parametrize(
    "entry",
    [
        {"answer": "different", "confidence": float("nan"), "reason": "", "raw": ""},
        {"answer": "different", "confidence": float("inf"), "reason": "", "raw": ""},
        {"answer": "different", "confidence": 1.5, "reason": "", "raw": ""},
        {"answer": "different", "confidence": -0.1, "reason": "", "raw": ""},
        {"answer": "different", "confidence": True, "reason": "", "raw": ""},
        {"answer": "different", "confidence": "0.9", "reason": "", "raw": ""},
        {"answer": "different", "confidence": None, "reason": "", "raw": ""},
        {"answer": "different", "reason": "", "raw": ""},
        {"answer": "maybe", "confidence": 0.9, "reason": "", "raw": ""},
        {"answer": 3, "confidence": 0.9, "reason": "", "raw": ""},
    ],
)
def test_readable_but_invalid_entries_are_misses(tmp_path, entry):
    c = AnswerCache(tmp_path)
    k = _key()
    path = tmp_path / k[:2] / f"{k}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(entry))  # json.dumps writes NaN / Infinity literals
    assert c.get(k, ["same_individual", "different", "unsure"]) is None


def test_the_option_check_is_skipped_without_options_and_a_good_entry_is_a_hit(tmp_path):
    c = AnswerCache(tmp_path)
    k = _key()
    c.put(k, VLMAnswer("maybe", 0.5, "r", "raw"))
    assert c.get(k, ["different"]) is None  # not one of the question's options
    assert c.get(k) == VLMAnswer("maybe", 0.5, "r", "raw")  # no options to check against
    c.put(k, VLMAnswer("different", 0.0, "r", "raw"))  # 0.0 and 1.0 are valid
    assert c.get(k, ["different"]).confidence == 0.0


def test_a_failing_write_is_ignored(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    c = AnswerCache(blocker / "sub")  # a directory cannot be created under a file
    c.put(_key(), VLMAnswer("a", 1.0, "", ""))  # must not raise
    assert c.get(_key()) is None


def test_the_directory_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = AnswerCache("~/cache")
    c.put(_key(), VLMAnswer("a", 1.0, "", ""))
    assert (tmp_path / "cache").is_dir()


def _ev(kind, stage, **params):
    return Event.propose(
        stage=stage, kind=kind, tracks=[1], lineage=[[[1, 0, 9]]], frames=(0, 9),
        params=params, algo_score=0.5, signals={},
    )


def test_options_by_kind_and_target():
    assert options_for(_ev(EventKind.SPLIT, "switch", cut_frame=5), "person") == SAME
    assert options_for(_ev(EventKind.LINK, "link", gap=[5, 8]), "vehicle") == SAME
    drop = _ev(EventKind.DROP, "screen", reason="static", spans=None)
    assert options_for(drop, "person") == PERSON_SCREEN
    assert options_for(drop, "vehicle") == VEHICLE_SCREEN
    assert options_for(_ev(EventKind.RECLASS, "screen", new_cls=None, spans=None), "person") == PERSON_SCREEN
    assert options_for(_ev(EventKind.DROP, "orphan", reason="orphan", spans=None), "person") is None
    assert options_for(_ev(EventKind.FILL, "fill", gap=[1, 4]), "person") is None
    assert "unsure" in PERSON_SCREEN and "unsure" in VEHICLE_SCREEN and "unsure" in SAME
    assert set(RIDER_OPTIONS) <= set(PERSON_SCREEN)


def test_a_partial_screen_prompt_explains_the_two_rows():
    partial = _ev(EventKind.RECLASS, "screen", new_cls=None, spans=[[10, 40]])
    whole = _ev(EventKind.RECLASS, "screen", new_cls=None, spans=None)
    p = build_prompt(partial, "person", options_for(partial, "person"), fps=10.0)
    assert "Row A shows crops from that part" in p and "Row B shows crops from the rest" in p
    assert "row A" in p
    w = build_prompt(whole, "person", options_for(whole, "person"), fps=10.0)
    assert "Row B" not in w and "Row A" not in w


@pytest.mark.parametrize(
    "ev,target,needle",
    [
        (_ev(EventKind.SPLIT, "switch", cut_frame=5), "person", "before"),
        (_ev(EventKind.LINK, "link", gap=[100, 130]), "person", "3.0 s"),
        (_ev(EventKind.DROP, "screen", reason="static", spans=None), "person", "pedestrian"),
        (_ev(EventKind.DROP, "screen", reason="duplicate", spans=None, of=2), "vehicle", "vehicle"),
    ],
)
def test_prompts_name_the_options_and_the_reply_format(ev, target, needle):
    options = options_for(ev, target)
    p = build_prompt(ev, target, options, fps=10.0)
    assert needle in p
    for o in options:
        assert o in p
    assert '"answer"' in p and '"confidence"' in p and '"reason"' in p
    assert p == build_prompt(ev, target, options, fps=10.0)  # deterministic
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_cache_prompts.py -q`
Expected: collection error `No module named 'dnt.refine.vlm.cache'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/vlm/cache.py`:

```python
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
```

Create `src/dnt/refine/vlm/prompts.py`:

```python
"""Prompt templates and answer options per event kind and target (spec 7.2)."""

from __future__ import annotations

from ..events import Event, EventKind

PERSON_SCREEN = [
    "pedestrian",
    "cyclist",
    "motorcycle_rider",
    "scooter_rider",
    "person_in_vehicle",
    "not_a_person",
    "unsure",
]
VEHICLE_SCREEN = [
    "vehicle",
    "part_or_duplicate_of_another_vehicle",
    "not_a_vehicle",
    "unsure",
]
SAME = ["same_individual", "different", "unsure"]
#: Rider answers and the ``RefineConfig.reclass_map`` key each one selects.
RIDER_OPTIONS = {"cyclist": "cyclist", "motorcycle_rider": "motorcycle", "scooter_rider": "scooter"}

_GLOSS = {
    "pedestrian": "a person on foot",
    "cyclist": "a person riding a bicycle",
    "motorcycle_rider": "a person riding a motorcycle",
    "scooter_rider": "a person riding a scooter",
    "person_in_vehicle": "a person inside a vehicle",
    "not_a_person": "not a person at all (a shadow, a sign, a pole, a bag, ...)",
    "vehicle": "a complete road vehicle",
    "part_or_duplicate_of_another_vehicle": "a part of, or a second box on, another vehicle",
    "not_a_vehicle": "not a vehicle at all",
    "same_individual": "the same individual in both rows",
    "different": "two different individuals",
    "unsure": "you cannot tell",
}


def options_for(event: Event, target: str) -> list[str] | None:
    """Return the answer options for ``event``, or ``None`` if it is never sent to a VLM."""
    if event.stage in ("switch", "link") and event.kind in (EventKind.SPLIT, EventKind.LINK):
        return list(SAME)
    if event.stage == "screen" and event.kind in (EventKind.DROP, EventKind.RECLASS):
        return list(PERSON_SCREEN if target == "person" else VEHICLE_SCREEN)
    return None


def build_prompt(event: Event, target: str, options: list[str], *, fps: float) -> str:
    """Return the question for ``event`` (deterministic text, no event identifiers)."""
    if event.kind is EventKind.SPLIT:
        body = (
            "The tracker kept one ID across a possible change of object. Row A shows crops of "
            "the tracked object before the cut; row B shows crops after it. Is the object in "
            "row B the same individual as in row A?"
        )
    elif event.kind is EventKind.LINK:
        t_e, t_s = event.params["gap"]
        gap = (int(t_s) - int(t_e)) / float(fps)
        body = (
            "The tracker lost an object and later started a new track. Row A shows the last "
            f"crops of the first track; row B shows the first crops of the second track, "
            f"{gap:.1f} s later. Are A and B the same individual?"
        )
    else:
        what = "a pedestrian" if target == "person" else "a road vehicle"
        if event.params.get("spans"):  # a partial edit: show what it would change and what not
            body = (
                f"The tracker reports {what}, but the tracker's cue fires only on part of one "
                "track. Row A shows crops from that part, which is the part that would be "
                "changed. Row B shows crops from the rest of the same track. What is the "
                "tracked object in row A? Use row B only to see what the object looked like "
                "before or after."
            )
        else:
            body = (
                f"The tracker reports {what}. The crops were taken at evenly spaced moments "
                "of one track (a context frame follows when present). What is the tracked "
                "object?"
            )
    lines = [body, "", "Options:"]
    lines += [f"- {o}: {_GLOSS[o]}" for o in options]
    lines += [
        "",
        "Reply with only a JSON object, no other text:",
        '{"answer": "<one of the options>", "confidence": <number from 0 to 1>, '
        '"reason": "<one short sentence>"}',
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_cache_prompts.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/vlm/cache.py src/dnt/refine/vlm/prompts.py tests/refine/test_vlm_cache_prompts.py
git commit -m "feat(refine): add the VLM answer cache and prompts" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The `fake`, `openai_compat` and `anthropic` backends

**Files:**
- Create: `src/dnt/refine/vlm/fake.py`, `src/dnt/refine/vlm/openai_compat.py`, `src/dnt/refine/vlm/anthropic.py`, `tests/refine/_vlm_fakes.py`
- Test: `tests/refine/test_vlm_backends.py`

**Interfaces:**
- Consumes: Task 1 (`VLMAnswer`, `VLMTransientError`, `parse_answer`, `DEFAULT_ANTHROPIC_MODEL`), `VLMConfig`.
- Produces:
  - `FakeBackend(script, *, name="fake", model="fake-1")`. `script` is a dict or a callable. A dict maps a key to an *item* or a list of items; the key is looked up as the whole tag (`"LINK:link-r0-000003"`), then the event id part, then the kind part (`"LINK"`), then `"*"`. A list is consumed one item per call (the last item repeats). An item is a `str` (the raw reply text), a `dict` (JSON-encoded as the reply), an `Exception` instance (raised), or a `VLMAnswer` (returned as is). A callable is called as `script(tag, options, temperature)` and returns an item. No matching key raises `RuntimeError`. `.calls` records `{"tag", "options", "temperature", "prompt", "image_len"}` per call.
  - `OpenAICompatBackend(cfg)` and `AnthropicBackend(cfg)`: lazy-import their package, read the key from the environment, send the image as base64, parse the reply with `parse_answer`, and map timeouts, connection errors, HTTP 429 and HTTP >= 500 to `VLMTransientError` (other errors propagate unchanged).
  - Every backend has an optional `async aclose()` that closes its HTTP client; the runner calls it on the event loop that used the client (Task 4). `FakeBackend.closed` counts closes.
  - Test helpers in `tests/refine/_vlm_fakes.py`: `install_fake_openai(monkeypatch, replies)` and `install_fake_anthropic(monkeypatch, replies)` return a `seen` dict (`seen["requests"]` list of kwargs, `seen["client_kwargs"]`).

- [ ] **Step 1: Write the test helpers and the failing tests**

Create `tests/refine/_vlm_fakes.py`:

```python
"""Fake ``openai`` and ``anthropic`` packages for the backend tests."""

from __future__ import annotations

import importlib.machinery
import sys
import types


def _module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(Exception):
    def __init__(self, message="", status_code=500):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIStatusError):
    def __init__(self, message=""):
        super().__init__(message, 429)


class InternalServerError(APIStatusError):
    def __init__(self, message=""):
        super().__init__(message, 500)


def _errors(mod):
    # module-level classes, shared by every install, so an exception built from one fake
    # module is still caught after the test installs the module again
    for cls in (
        APIConnectionError, APITimeoutError, APIStatusError, RateLimitError, InternalServerError
    ):
        setattr(mod, cls.__name__, cls)


def install_fake_openai(monkeypatch, replies):
    """Install a fake ``openai``; ``replies`` is a list of reply texts or exceptions."""
    seen = {"requests": [], "client_kwargs": None, "closed": 0}
    queue = list(replies)
    mod = _module("openai")
    _errors(mod)

    class _Completions:
        async def create(self, **kwargs):
            seen["requests"].append(kwargs)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            msg = types.SimpleNamespace(content=item)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class AsyncOpenAI:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=_Completions())

        async def close(self):
            seen["closed"] += 1

    mod.AsyncOpenAI = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return seen


def install_fake_anthropic(monkeypatch, replies):
    """Install a fake ``anthropic``; ``replies`` is a list of reply texts or exceptions."""
    seen = {"requests": [], "client_kwargs": None, "closed": 0}
    queue = list(replies)
    mod = _module("anthropic")
    _errors(mod)

    class _Messages:
        async def create(self, **kwargs):
            seen["requests"].append(kwargs)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            block = types.SimpleNamespace(type="text", text=item)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="thinking"), block])

    class AsyncAnthropic:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs
            self.messages = _Messages()

        async def close(self):
            seen["closed"] += 1

    mod.AsyncAnthropic = AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return seen
```

Create `tests/refine/test_vlm_backends.py`:

```python
import asyncio
import base64
import json
import sys

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import DEFAULT_ANTHROPIC_MODEL, VLMAnswer, VLMTransientError, make_backend
from dnt.refine.vlm.fake import FakeBackend

from ._vlm_fakes import install_fake_anthropic, install_fake_openai

OPTS = ["same_individual", "different", "unsure"]
GOOD = json.dumps({"answer": "different", "confidence": 0.9, "reason": "r"})


def ask(backend, tag="LINK:link-r0-000001", temperature=0.0):
    return asyncio.run(backend.ask(b"\xff\xd8jpeg", "PROMPT", OPTS, temperature, tag=tag))


# ---- fake ----

def test_fake_lookup_order_and_call_log():
    b = FakeBackend({"LINK:link-r0-000001": GOOD, "LINK": {"answer": "unsure", "confidence": 0.1,
                                                          "reason": ""}, "*": GOOD})
    assert ask(b).answer == "different"  # whole tag
    assert ask(b, tag="LINK:link-r0-000002").answer == "unsure"  # kind
    assert ask(b, tag="SPLIT:switch-r0-000001").answer == "different"  # default
    assert b.calls[0]["tag"] == "LINK:link-r0-000001" and b.calls[0]["image_len"] == 6
    assert b.calls[0]["options"] == OPTS and b.calls[0]["prompt"] == "PROMPT"


def test_fake_event_id_key_and_sequences_and_exceptions():
    b = FakeBackend({"link-r0-000001": [RuntimeError("boom"), "garbage", GOOD]})
    with pytest.raises(RuntimeError, match="boom"):
        ask(b)
    with pytest.raises(ValueError):
        ask(b)  # an invalid reply is parsed like a real one
    assert ask(b).answer == "different"
    assert ask(b).answer == "different"  # the last item repeats


def test_fake_callable_returns_items_and_unknown_tag_raises():
    b = FakeBackend(lambda tag, options, temperature: {"answer": options[1], "confidence": 1.0,
                                                       "reason": str(temperature)})
    assert ask(b, temperature=0.7).reason == "0.7"
    with pytest.raises(RuntimeError, match="no scripted answer"):
        ask(FakeBackend({}))
    ans = VLMAnswer("different", 0.5, "x", "raw")
    assert ask(FakeBackend({"*": ans})) is ans


# ---- openai_compat ----

def cfg(**kw):
    return VLMConfig(**{"backend": "openai_compat", "base_url": "http://localhost:8000/v1",
                        "model": "qwen", **kw})


def test_openai_request_shape(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    b = make_backend(cfg(timeout_s=12.0))
    ans = ask(b, temperature=0.3)
    assert ans.answer == "different" and b.name == "openai_compat" and b.model == "qwen"
    ck = seen["client_kwargs"]
    assert ck["base_url"] == "http://localhost:8000/v1" and ck["api_key"] == "sk-secret"
    assert ck["timeout"] == 12.0 and ck["max_retries"] == 0
    req = seen["requests"][0]
    assert req["model"] == "qwen" and req["temperature"] == 0.3
    assert req["response_format"] == {"type": "json_object"}
    content = req["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "PROMPT"}
    url = content[1]["image_url"]["url"]
    assert url == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()


def test_openai_json_mode_off_and_missing_key_for_local_servers(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    b = make_backend(cfg(json_mode=False))
    ask(b)
    assert "response_format" not in seen["requests"][0]
    assert seen["client_kwargs"]["api_key"]  # a local server needs a placeholder, not None


def test_openai_custom_key_env(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    monkeypatch.setenv("MY_VLM_KEY", "abc")
    make_backend(cfg(api_key_env="MY_VLM_KEY"))
    assert seen["client_kwargs"]["api_key"] == "abc"


@pytest.mark.parametrize(
    "name", ["APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"]
)
def test_openai_transient_errors_are_mapped(monkeypatch, name):
    install_fake_openai(monkeypatch, [])  # creates the module, so the exception class exists
    exc = getattr(sys.modules["openai"], name)("sk-secret failure")
    install_fake_openai(monkeypatch, [exc])
    with pytest.raises(VLMTransientError) as err:
        ask(make_backend(cfg()))
    assert "sk-secret" not in str(err.value)


def test_openai_other_errors_propagate_and_5xx_status_is_transient(monkeypatch):
    install_fake_openai(monkeypatch, [])
    status_error = sys.modules["openai"].APIStatusError
    install_fake_openai(monkeypatch, [status_error("bad request", 400)])
    with pytest.raises(status_error):
        ask(make_backend(cfg()))
    install_fake_openai(monkeypatch, [status_error("down", 503)])
    with pytest.raises(VLMTransientError):
        ask(make_backend(cfg()))


# ---- anthropic ----

def test_anthropic_request_shape_and_default_model(monkeypatch):
    seen = install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    b = make_backend(VLMConfig(backend="anthropic", timeout_s=9.0))
    assert b.model == DEFAULT_ANTHROPIC_MODEL and b.name == "anthropic"
    assert ask(b, temperature=0.2).answer == "different"  # text block found after a non-text one
    ck = seen["client_kwargs"]
    assert ck["api_key"] == "sk-ant" and ck["timeout"] == 9.0 and ck["max_retries"] == 0
    req = seen["requests"][0]
    assert req["model"] == DEFAULT_ANTHROPIC_MODEL and req["temperature"] == 0.2
    assert req["max_tokens"] > 0
    blocks = req["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[0]["source"]["data"] == base64.b64encode(b"\xff\xd8jpeg").decode()
    assert blocks[1] == {"type": "text", "text": "PROMPT"}


def test_anthropic_model_override_missing_key_and_errors(monkeypatch):
    install_fake_anthropic(monkeypatch, [GOOD])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_backend(VLMConfig(backend="anthropic"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert make_backend(VLMConfig(backend="anthropic", model="claude-x")).model == "claude-x"
    install_fake_anthropic(monkeypatch, [])
    exc = sys.modules["anthropic"].RateLimitError("sk-ant slow down")
    install_fake_anthropic(monkeypatch, [exc])
    with pytest.raises(VLMTransientError) as err:
        ask(make_backend(VLMConfig(backend="anthropic")))
    assert "sk-ant" not in str(err.value)


def test_backends_close_their_client_and_the_fake_counts_closes(monkeypatch):
    seen = install_fake_openai(monkeypatch, [GOOD])
    asyncio.run(make_backend(cfg()).aclose())
    assert seen["closed"] == 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    seen2 = install_fake_anthropic(monkeypatch, [GOOD])
    asyncio.run(make_backend(VLMConfig(backend="anthropic")).aclose())
    assert seen2["closed"] == 1
    fake = FakeBackend({})
    asyncio.run(fake.aclose())
    assert fake.closed == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_backends.py -q`
Expected: collection error `No module named 'dnt.refine.vlm.fake'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/vlm/fake.py`:

```python
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
```

Create `src/dnt/refine/vlm/openai_compat.py`:

```python
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
```

Create `src/dnt/refine/vlm/anthropic.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_backends.py tests/test_refine_independence.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/vlm/fake.py src/dnt/refine/vlm/openai_compat.py src/dnt/refine/vlm/anthropic.py tests/refine/_vlm_fakes.py tests/refine/test_vlm_backends.py
git commit -m "feat(refine): add the fake, OpenAI-compatible and Anthropic VLM backends" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The runner: votes, retries, budget, concurrency, cache, one long-lived event loop

**Files:**
- Create: `src/dnt/refine/vlm/runner.py`
- Test: `tests/refine/test_vlm_runner.py`

**Interfaces:**
- Consumes: Tasks 1-3 (`VLMBackend`, `VLMTransientError`, `AnswerCache`), `VLMConfig`.
- Produces:
  - `Question(tag, image, prompt, options)` and `Verdict(answer, confidence, reason, votes, cached, error, raw)` (dataclasses). `Verdict.answer` is `None` when `error` is set (`"budget"`, `"tie"`, `"transient: ..."`, `"invalid output: ..."`, `"<ExcType>: ..."`).
  - `VLMRunner(cfg, backend, cache=None, *, sleep=None)` with counters `calls` (first attempts: one per uncached vote sent), `retries` (extra backend invocations after a transient failure or an invalid reply), `cache_hits`, `failures` (questions that ended in an error other than `"budget"`), `budget_skipped` (questions that ended with `error="budget"`, whether refused at admission or out of retry allowance); **`calls + retries <= max_calls` always: `max_calls` is a hard limit on backend invocations**; `ask_many(questions) -> list[Verdict]` (synchronous, order preserved); `close()` (idempotent; also `with VLMRunner(...) as r:`).
  - **One live event loop.** The runner starts a daemon thread with its own event loop on the first `ask_many` and keeps it for its whole life, so the backend's HTTP client (which keeps pooled connections bound to the loop that opened them) sees one loop across every stage call. `ask_many` submits to that loop and blocks, so it works from plain code and from a thread that already runs a loop (Jupyter). `close()` awaits `backend.aclose()` (when the backend has one) on that loop, stops the loop, and joins the thread; `ask_many` after `close()` raises `RuntimeError`.
  - **Budget in priority order, whole questions only.** Before anything is sent, the runner admits the questions in list order (the caller sorts them by priority): cached votes are free, and a question is admitted only when **all its uncached votes** fit in the remaining `max_calls`; otherwise it gets `error="budget"` and no partial work. A later question that does fit (for example one with cached votes) is still admitted, so the budget is used fully while higher-priority questions always come first. **Retries draw from an allowance of their own: whatever is left of `max_calls` after admission** (`max_calls` minus invocations already made minus the votes admitted). Each retry (after a transient failure or an invalid reply) takes one unit; when the allowance is empty the question ends with `error="budget"` (stays pending) instead of retrying. A retry therefore never takes a unit that an admitted question needs, and the run can never make more than `max_calls` invocations (a tight budget leaves no allowance, so then a transient failure leaves the question pending).
  - Rules (§7.2, §7.4): `votes = n > 1` asks the same question n times at `vote_temperature`; the answer is the majority option with `confidence = count / n` and a tie gives `error="tie"`; with `n == 1` the temperature is 0 and the confidence is the model's own. A reply that does not parse is retried once with a reminder appended to the prompt; a second failure gives `"invalid output: ..."`. A timeout (`vlm.timeout_s`) or `VLMTransientError` is retried with exponential backoff, 3 attempts, first wait 2 s (`sleep` is injectable). Any other exception ends the question with `"<ExcType>: <message>"`, no retry. A cached vote is used only if it passes the same validation as a fresh reply (Task 2). Concurrency is bounded by `max_concurrency` (FIFO in question order). Known secret values (the configured key variable and the two default key variables) are scrubbed from every error text.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_vlm_runner.py`:

```python
import asyncio
import http.server
import json
import threading

import pytest

from dnt.refine.config import VLMConfig
from dnt.refine.vlm import VLMAnswer, VLMTransientError, parse_answer
from dnt.refine.vlm.cache import AnswerCache
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import Question, VLMRunner

OPTS = ["same_individual", "different", "unsure"]
CREATED = []


@pytest.fixture(autouse=True)
def _close_runners():
    yield
    for r in CREATED:
        r.close()
    CREATED.clear()


def reply(answer="different", conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf, "reason": "r"})


def q(tag="LINK:link-r0-000001", image=b"img"):
    return Question(tag, image, "PROMPT", list(OPTS))


class Sleeps:
    def __init__(self):
        self.waits = []

    async def __call__(self, s):
        self.waits.append(s)


def runner(script, cache=None, sleep=None, backend=None, **cfg):
    backend = backend or FakeBackend(script)
    r = VLMRunner(
        VLMConfig(backend="openai_compat", model="m", **cfg), backend, cache, sleep=sleep or Sleeps()
    )
    CREATED.append(r)
    return r, backend


def test_one_vote_uses_the_models_confidence_and_temperature_zero():
    r, b = runner({"*": reply("different", 0.83)})
    (v,) = r.ask_many([q()])
    assert (v.answer, v.confidence, v.error, v.cached) == ("different", 0.83, None, False)
    assert v.reason == "r" and v.votes == {"different": 1}
    assert b.calls[0]["temperature"] == 0.0 and r.calls == 1 and r.cache_hits == 0


def test_majority_vote_confidence_and_temperature():
    r, b = runner(
        {"*": [reply("different"), reply("same_individual"), reply("different")]},
        votes=3,
        vote_temperature=0.7,
    )
    (v,) = r.ask_many([q()])
    assert v.answer == "different" and v.confidence == pytest.approx(2 / 3)
    assert v.votes == {"different": 2, "same_individual": 1}
    assert [c["temperature"] for c in b.calls] == [0.7, 0.7, 0.7] and r.calls == 3


def test_a_tie_has_no_answer():
    r, _ = runner({"*": [reply("different"), reply("same_individual")]}, votes=2)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error == "tie"
    assert v.votes == {"different": 1, "same_individual": 1}
    assert r.failures == 1


def test_cache_hits_are_free_and_votes_are_cached_separately(tmp_path):
    cache = AnswerCache(tmp_path)
    r, _ = runner(
        {"*": [reply("different"), reply("same_individual"), reply("different")]},
        cache=cache,
        votes=3,
    )
    (v1,) = r.ask_many([q()])
    assert r.calls == 3 and len(list(tmp_path.rglob("*.json"))) == 3
    r2, b2 = runner({}, cache=cache, votes=3)  # an empty script would raise if asked
    (v2,) = r2.ask_many([q()])
    assert b2.calls == [] and r2.calls == 0 and r2.cache_hits == 3
    assert v2.answer == v1.answer and v2.cached is True and v1.cached is False


def test_a_cached_entry_with_a_bad_confidence_or_option_is_not_a_hit(tmp_path):
    cache = AnswerCache(tmp_path)
    key = AnswerCache.key(b"img", "PROMPT", OPTS, "fake", "fake-1", 0.0, 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    for entry in (
        {"answer": "different", "confidence": float("nan"), "reason": "", "raw": ""},
        {"answer": "maybe", "confidence": 0.99, "reason": "", "raw": ""},
    ):
        path.write_text(json.dumps(entry))
        r, b = runner({"*": reply("same_individual", 0.8)}, cache=cache)
        (v,) = r.ask_many([q()])
        assert v.answer == "same_individual" and v.cached is False and len(b.calls) == 1
        assert r.cache_hits == 0  # the fresh answer then overwrote the bad entry


def test_invalid_output_is_retried_once_with_a_reminder_and_is_one_call():
    r, b = runner({"*": ["not json at all", reply("same_individual", 0.8)]})
    (v,) = r.ask_many([q()])
    assert v.answer == "same_individual" and r.calls == 1 and r.retries == 1
    assert b.calls[0]["prompt"] == "PROMPT"
    assert b.calls[1]["prompt"].startswith("PROMPT") and "not valid" in b.calls[1]["prompt"]
    assert all(o in b.calls[1]["prompt"] for o in OPTS)


def test_repeated_invalid_output_ends_the_question_and_is_not_cached(tmp_path):
    r, _ = runner({"*": ["nope"]}, cache=AnswerCache(tmp_path))
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("invalid output")
    assert r.calls == 1 and r.retries == 1 and r.failures == 1
    assert list(tmp_path.rglob("*.json")) == []


def test_transient_errors_back_off_and_recover():
    sl = Sleeps()
    r, _ = runner({"*": [VLMTransientError("429"), VLMTransientError("503"), reply()]}, sleep=sl)
    (v,) = r.ask_many([q()])
    assert v.answer == "different" and sl.waits == [2.0, 4.0]
    assert r.calls == 1 and r.retries == 2


def test_three_transient_failures_give_up():
    sl = Sleeps()
    r, _ = runner({"*": [VLMTransientError("503")]}, sleep=sl)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("transient") and sl.waits == [2.0, 4.0]
    assert r.failures == 1 and r.calls == 1 and r.retries == 2


def test_a_timeout_is_retried():
    class Slow:
        name, model = "slow", "m"
        n = 0

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            Slow.n += 1
            if Slow.n == 1:
                await asyncio.sleep(1.0)
            return parse_answer(reply(), options)

    r, _ = runner(None, backend=Slow(), timeout_s=0.05)
    (v,) = r.ask_many([q()])
    assert v.answer == "different" and Slow.n == 2


def test_other_exceptions_do_not_retry_and_never_leak_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    r, _ = runner({"*": [RuntimeError("auth failed for sk-very-secret")]})
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("RuntimeError") and r.calls == 1
    assert r.retries == 0
    assert "sk-very-secret" not in v.error and "***" in v.error
    monkeypatch.setenv("MY_KEY", "tok-123")
    r2, _ = runner({"*": [RuntimeError("bad tok-123")]}, api_key_env="MY_KEY")
    (v2,) = r2.ask_many([q()])
    assert "tok-123" not in v2.error


def test_a_backend_returning_a_bad_answer_object_is_treated_as_invalid_output():
    bad = VLMAnswer("different", float("nan"), "", "")
    r, _ = runner({"*": [bad]})
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("invalid output") and r.retries == 1
    for answer in (VLMAnswer("maybe", 0.9, "", ""), VLMAnswer("different", 1.5, "", ""),
                   VLMAnswer("different", -0.1, "", ""), VLMAnswer("different", True, "", ""),
                   VLMAnswer("different", float("inf"), "", "")):
        r2, b2 = runner({"*": [answer]})
        (v2,) = r2.ask_many([q()])
        assert v2.answer is None and v2.error.startswith("invalid output")
        assert len(b2.calls) == 2  # the one retry was made, then it gave up
    r3, _ = runner({"*": [VLMAnswer("different", 0.9, "", ""), ]})
    assert r3.ask_many([q()])[0].answer == "different"
    # a bad object followed by a good one: the good one wins
    r4, _ = runner({"*": [bad, VLMAnswer("different", 0.9, "r", "")]})
    assert r4.ask_many([q()])[0].answer == "different"


def test_three_invalid_votes_never_add_up_to_a_majority():
    bad = VLMAnswer("different", float("nan"), "", "")
    r, _ = runner({"*": [bad]}, votes=3, vote_temperature=0.7)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error.startswith("invalid output") and v.confidence == 0.0


def test_the_budget_is_spent_in_question_order():
    r, b = runner({"*": [reply()]}, max_calls=3)
    vs = r.ask_many([q(f"LINK:link-r0-{i:06d}") for i in range(5)])
    assert [v.error for v in vs] == [None, None, None, "budget", "budget"]
    assert r.calls == 3 and r.budget_skipped == 2 and r.failures == 0 and len(b.calls) == 3


def test_a_question_gets_all_its_votes_or_none_even_with_concurrency():
    # three votes need three calls: the first question takes the whole budget, the second
    # gets nothing (before: two calls to the first and one to the second, and neither finished)
    r, b = runner({"*": [reply("different")]}, votes=3, max_calls=3, max_concurrency=2)
    v1, v2 = r.ask_many([q("LINK:a"), q("LINK:b")])
    assert v1.answer == "different" and v1.votes == {"different": 3}
    assert v2.error == "budget" and v2.votes == {}
    assert r.calls == 3 and r.budget_skipped == 1 and len(b.calls) == 3


def test_a_cheaper_later_question_still_uses_what_is_left(tmp_path):
    cache = AnswerCache(tmp_path)
    k0 = AnswerCache.key(b"img3", "PROMPT", OPTS, "fake", "fake-1", 0.7, 0)
    cache.put(k0, VLMAnswer("different", 0.9, "r", "raw"))  # q3 has one of its two votes cached
    r, _ = runner({"*": [reply("different")]}, cache=cache, votes=2, max_calls=3, max_concurrency=2)
    qs = [q("LINK:a", b"img1"), q("LINK:b", b"img2"), q("LINK:c", b"img3")]
    v1, v2, v3 = r.ask_many(qs)  # needs: 2, 2, 1; budget 3
    assert v1.error is None and v2.error == "budget" and v3.error is None
    assert r.calls == 3 and r.cache_hits == 1


def test_retries_draw_only_from_what_admission_left_over():
    sl = Sleeps()
    script = {"LINK:a": [VLMTransientError("503"), reply()], "LINK:b": [reply()]}
    # budget 3: a and b are admitted (2 votes), one unit is left over for a's retry
    r, b = runner(script, sleep=sl, max_calls=3, max_concurrency=2)
    v1, v2 = r.ask_many([q("LINK:a"), q("LINK:b")])
    assert v1.error is None and v2.error is None
    assert r.calls == 2 and r.retries == 1 and sl.waits == [2.0] and len(b.calls) == 3
    # budget 2: nothing is left over, so a's transient failure leaves a pending ("budget") and
    # b, whose vote was reserved, is still answered
    sl2 = Sleeps()
    r2, b2 = runner(script, sleep=sl2, max_calls=2, max_concurrency=2)
    w1, w2 = r2.ask_many([q("LINK:a"), q("LINK:b")])
    assert w1.error == "budget" and w1.answer is None and w2.error is None
    assert r2.calls == 2 and r2.retries == 0 and len(b2.calls) == 2 and sl2.waits == []
    assert r2.budget_skipped == 1 and r2.failures == 0


def test_max_calls_is_a_hard_limit_on_backend_invocations():
    # the review's case: transient, transient, invalid, transient, transient, then a valid reply
    # would be six invocations; with max_calls=1 only the first may be made
    T = VLMTransientError
    script = {"*": [T("1"), T("2"), "not json", T("3"), T("4"), reply()]}
    r, b = runner(script, max_calls=1)
    (v,) = r.ask_many([q()])
    assert v.answer is None and v.error == "budget"
    assert len(b.calls) == 1 and r.calls == 1 and r.retries == 0
    for cap in (2, 3, 4, 5):
        r, b = runner(script, max_calls=cap)
        r.ask_many([q()])
        assert len(b.calls) <= cap and r.calls + r.retries == len(b.calls)


def test_the_limit_holds_across_stage_calls_and_with_many_questions():
    T = VLMTransientError
    r, b = runner({"*": [T("x"), reply(), reply()]}, max_calls=5, max_concurrency=3)
    for batch in range(3):
        r.ask_many([q(f"LINK:{batch}-{i}") for i in range(4)])
    assert len(b.calls) <= 5 and r.calls + r.retries == len(b.calls)


def test_cache_hits_do_not_consume_the_budget(tmp_path):
    cache = AnswerCache(tmp_path)
    r1, _ = runner({"*": [reply()]}, cache=cache, max_calls=1)
    r1.ask_many([q()])
    r2, b2 = runner({"*": [reply()]}, cache=cache, max_calls=0)
    (v,) = r2.ask_many([q()])
    assert v.answer == "different" and b2.calls == [] and r2.budget_skipped == 0


def test_concurrency_is_bounded_and_order_is_preserved():
    class Probe:
        name, model = "probe", "m"
        live = peak = 0

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            Probe.live += 1
            Probe.peak = max(Probe.peak, Probe.live)
            await asyncio.sleep(0.01)
            Probe.live -= 1
            return VLMAnswer(options[0], 1.0, tag, "")

    r, _ = runner(None, backend=Probe(), max_concurrency=2)
    vs = r.ask_many([q(f"LINK:{i}") for i in range(7)])
    assert Probe.peak == 2 and [v.reason for v in vs] == [f"LINK:{i}" for i in range(7)]


def test_it_works_inside_a_running_event_loop():
    r, _ = runner({"*": reply()})

    async def main():
        return r.ask_many([q()])  # a synchronous call from a coroutine (Jupyter)

    (v,) = asyncio.run(main())
    assert v.answer == "different"


def test_an_empty_batch_is_fine_and_starts_no_thread():
    r, _ = runner({})
    assert r.ask_many([]) == [] and r._thread is None


# ---- one live loop for the runner's whole life (review: pooled connections are loop-bound) ----


def test_every_call_runs_on_the_same_live_loop_and_close_stops_it():
    seen = []

    class Loops:
        name, model = "loops", "m"

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            seen.append(asyncio.get_running_loop())
            return VLMAnswer(options[0], 1.0, "", "")

    r, _ = runner(None, backend=Loops())
    r.ask_many([q("LINK:a")])
    r.ask_many([q("LINK:b")])  # a second stage
    assert len(seen) == 2 and seen[0] is seen[1] and not seen[0].is_closed()
    thread = r._thread
    r.close()
    assert not thread.is_alive() and seen[0].is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        r.ask_many([q()])
    r.close()  # idempotent


def test_close_closes_the_backend_once_on_its_own_loop():
    closes = []

    class Closing:
        name, model = "closing", "m"

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            return VLMAnswer(options[0], 1.0, "", "")

        async def aclose(self):
            closes.append(asyncio.get_running_loop())

    r, _ = runner(None, backend=Closing())
    r.ask_many([q()])
    loop = r._loop
    r.close()
    r.close()
    assert closes == [loop]
    quiet, fake = runner({})
    quiet.close()  # never used: nothing to close, no thread
    assert fake.closed == 0 and quiet._thread is None


class _Quiet(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like a real API server

    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def make_http_backend(url):
    import httpx

    class HttpBackend:
        name, model = "http", "m"

        def __init__(self):
            self._client = None

        async def ask(self, image, prompt, options, temperature, *, tag=""):
            if self._client is None:  # created on the first loop, like AsyncOpenAI
                self._client = httpx.AsyncClient()
            resp = await self._client.get(url)  # reuses a pooled keep-alive connection
            assert resp.status_code == 200
            return VLMAnswer(options[0], 1.0, "", "")

        async def aclose(self):
            if self._client is not None:
                await self._client.aclose()

    return HttpBackend()


def test_pooled_http_connections_survive_successive_stage_calls(local_server):
    pytest.importorskip("httpx")
    r, _ = runner(None, backend=make_http_backend(local_server))
    for stage in ("switch", "screen", "link-1", "link-2"):  # one ask_many per stage call
        (v,) = r.ask_many([q(f"LINK:{stage}")])
        assert v.error is None, (stage, v.error)  # a per-call loop fails with "Event loop is closed"


def test_pooled_http_connections_also_work_from_a_notebook_loop(local_server):
    pytest.importorskip("httpx")
    r, _ = runner(None, backend=make_http_backend(local_server))

    async def main():
        first = r.ask_many([q("LINK:a")])
        second = r.ask_many([q("LINK:b")])
        return first, second

    first, second = asyncio.run(main())
    assert first[0].error is None and second[0].error is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_runner.py -q`
Expected: collection error `No module named 'dnt.refine.vlm.runner'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/vlm/runner.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_runner.py -q -W error`
Expected: all pass. To confirm the loop tests bite, temporarily make `ask_many` use `asyncio.run(self._gather(questions))` and check that `test_pooled_http_connections_survive_successive_stage_calls` and `test_every_call_runs_on_the_same_live_loop_and_close_stops_it` fail; restore it.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/vlm/runner.py tests/refine/test_vlm_runner.py
git commit -m "feat(refine): add the VLM runner with votes, retries, a priority budget and one live loop" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Evidence images

**Files:**
- Create: `src/dnt/refine/evidence.py`
- Test: `tests/refine/test_evidence.py` (uses `tests/refine/_video.py` from Plan 2)

**Interfaces:**
- Consumes: P2's `crops.FrameReader` and `crops.crop_box`; the raw work table (columns `frame, raw_id, x, y, w, h`, one row per observed box, index = row labels) and the raw occlusion flags (`primitives.occlusion_flags`, a bool Series with the same index); an `Event` (`kind`, `params`, `lineage`, `frames`).
- Produces:
  - `EvidenceBuilder(video_file, raw_work, occluded, *, frame_count, send_context_frames=True)`.
  - `plan(event) -> EvidencePlan`: the crop rows and context frames the image will show, without reading the video. `EvidencePlan.rows` is a list of `(label, [(raw_id, frame), ...])`; `EvidencePlan.contexts` is a list of `ContextTile(frame, caption, boxes)` with `ContextBox(label, color, dashed, xywh)`; `EvidencePlan.is_empty`.
  - `build_many(events) -> dict[str, bytes | None]` maps `event.id` to one composite JPEG (or `None` when nothing could be drawn). Frames are read in sorted order, 64 events per pass, and each frame is rendered into small tiles as soon as it is decoded, so memory stays bounded. `build(event)` is `build_many([event])[event.id]`.
  - Tile rules (§7.1): crops are padded to 1.5x the box and upscaled to a height of at least 160 px; context frames are downscaled to 768 px wide with the event's boxes drawn (`A` green, `B` red, hidden path dashed yellow, other tracks gray); DROP/RECLASS: 6 **observed** crops spread evenly over the track plus 1 context frame at mid-life (overlapping rows are kept: overlap is what the duplicate, in-vehicle and rider hypotheses are about, so an always-overlapping track still has evidence), and, for a partial edit (`params["spans"]` given), two rows instead: row A 3 observed crops from the supported segments (the part that would change) and row B 3 observed crops from the rest of the same track, with the context frame taken inside the supported part; SPLIT at t: row A the last 3 clean crops before t, row B the first 3 from t on, plus the context frame at t with nearby tracks drawn; LINK: row A the last 3 clean crops of the first track, row B the first 3 of the second, plus context frames at `gap[0]` and `gap[1]`, and for `gate == "occluded"` one at mid-gap with the interpolated hidden box dashed. A crop is *observed* when the row exists, its box has a positive size and its frame is inside the video; it is *clean* when, in addition, it is not flagged occluded. Identity questions (SPLIT, LINK) use clean crops only. With `send_context_frames=False` no context frame is drawn.

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_evidence.py`:

```python
import cv2
import numpy as np
import pandas as pd
import pytest

from dnt.refine import io
from dnt.refine.events import Event, EventKind
from dnt.refine.evidence import EvidenceBuilder

from ._fixtures import box_rows, table
from ._video import BLUE, RED, make_color_video, video_rows

N = 120


def _work(*row_lists):
    work = io.to_work(table(*row_lists)).work
    return work


def _builder(tmp_path, work, colors, occluded=None, **kw):
    rows = []
    for raw, color in colors.items():
        rows += video_rows(work[work.raw_id == raw].assign(track=raw).pipe(_as_rows), color)
    video = make_color_video(tmp_path / "v.mp4", rows, N)
    occ = pd.Series(False, index=work.index) if occluded is None else occluded
    return EvidenceBuilder(video, work, occ, frame_count=N, **kw)


def _as_rows(df):
    return [[r.frame, r.track, r.x, r.y, r.w, r.h] for r in df.itertuples()]


def _ev(kind, stage, tracks, lineage, frames, **params):
    ev = Event.propose(stage=stage, kind=kind, tracks=tracks, lineage=lineage, frames=frames,
                       params=params, algo_score=0.5, signals={})
    ev.id = f"{stage}-r0-000001"
    return ev


def _walker(raw, frames, x0=40.0, y0=60.0):
    return box_rows(raw, frames, x0, y0, vx=1.0, w=30.0, h=70.0)


def test_screen_event_plan_has_six_even_crops_and_one_context(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    plan = b.plan(ev)
    (_, tiles), = plan.rows
    assert len(tiles) == 6 and tiles[0] == (1, 0) and tiles[-1] == (1, 59)
    assert [f for _, f in tiles] == sorted(f for _, f in tiles)
    assert len(plan.contexts) == 1 and plan.contexts[0].frame in range(25, 35)  # mid-life


def test_a_partial_screen_event_shows_the_supported_and_the_unsupported_segments(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.RECLASS, "screen", [1], [[[1, 0, 59]]], (20, 40), new_cls=None,
             spans=[[20, 40]])
    plan = b.plan(ev)
    (la, a), (lb, bb) = plan.rows
    assert (la, lb) == ("A", "B")
    assert 1 <= len(a) <= 3 and all(20 <= f <= 40 for _, f in a)
    assert 1 <= len(bb) <= 3 and all(f < 20 or f > 40 for _, f in bb)
    assert 20 <= plan.contexts[0].frame <= 40  # the context frame is inside the supported part
    jpeg = b.build(ev)
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] >= 2 * 160 + 240  # two rows of crops (160 px+) plus the context frame (a single row plus context is only 418 px)


def test_a_partial_event_that_covers_the_whole_track_has_an_empty_second_row(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.RECLASS, "screen", [1], [[[1, 0, 59]]], (0, 59), new_cls=None,
             spans=[[0, 59]])
    (_, a), (_, bb) = b.plan(ev).rows
    assert len(a) == 3 and bb == []


def test_screen_evidence_keeps_overlapping_rows_that_identity_evidence_drops(tmp_path):
    w = _work(_walker(1, range(60)))
    occ = pd.Series(True, index=w.index)  # always overlapping another box
    b = _builder(tmp_path, w, {1: RED}, occluded=occ)
    drop = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="duplicate",
               spans=None, of=2)
    plan = b.plan(drop)
    (_, tiles), = plan.rows
    assert len(tiles) == 6 and len(plan.contexts) == 1 and not plan.is_empty
    assert b.build(drop)[:2] == b"\xff\xd8"
    split = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    assert b.plan(split).rows == [("A", []), ("B", [])]  # identity questions need clean crops


def test_split_plan_uses_three_clean_crops_each_side(tmp_path):
    w = _work(_walker(1, range(60)))
    occ = pd.Series(False, index=w.index)
    occ[w.frame.isin([28, 29])] = True  # the two frames before the cut are occluded
    b = _builder(tmp_path, w, {1: RED}, occluded=occ)
    ev = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    plan = b.plan(ev)
    (_, a), (_, bb) = plan.rows
    assert [f for _, f in a] == [25, 26, 27] and [f for _, f in bb] == [30, 31, 32]
    assert plan.contexts[0].frame == 30


def test_link_plan_rows_contexts_and_the_hidden_path(tmp_path):
    w = _work(_walker(1, range(0, 40)), _walker(2, range(60, 100), x0=100.0))
    b = _builder(tmp_path, w, {1: RED, 2: BLUE})
    base = dict(gap=[39, 60])
    ev = _ev(EventKind.LINK, "link", [1, 2], [[[1, 0, 39]], [[2, 60, 99]]], (39, 60),
             gate="normal", **base)
    plan = b.plan(ev)
    (_, a), (_, bb) = plan.rows
    assert [f for _, f in a] == [37, 38, 39] and [f for _, f in bb] == [60, 61, 62]
    assert [c.frame for c in plan.contexts] == [39, 60]
    occ = _ev(EventKind.LINK, "link", [1, 2], [[[1, 0, 39]], [[2, 60, 99]]], (39, 60),
              gate="occluded", **base)
    mid = b.plan(occ).contexts
    assert [c.frame for c in mid] == [39, 49, 60]
    hidden = [bx for bx in mid[1].boxes if bx.dashed]
    assert len(hidden) == 1 and hidden[0].xywh is not None
    # A's last box (frame 39) is at x = 79, B's first (frame 60) at x = 100; frame 49 is 10/21 of
    # the way, so the hidden box is at x = 89 with the same y, w, h
    assert hidden[0].xywh == pytest.approx((89.0, 60.0, 30.0, 70.0))


def test_fewer_clean_crops_than_wanted_and_no_context_option(tmp_path):
    w = _work(_walker(1, range(2)))
    b = _builder(tmp_path, w, {1: RED}, send_context_frames=False)
    ev = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 1]]], (1, 1), cut_frame=1)
    plan = b.plan(ev)
    (_, a), (_, bb) = plan.rows
    assert [f for _, f in a] == [0] and [f for _, f in bb] == [1]
    assert plan.contexts == []


def test_build_makes_a_jpeg_of_bounded_width(tmp_path):
    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    jpeg = b.build(ev)
    assert jpeg[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.ndim == 3 and 100 < img.shape[1] <= 800 and img.shape[0] >= 160
    # a crop tile is at least 160 px high: the first row's tallest tile
    no_ctx = _builder(tmp_path, w, {1: RED}, send_context_frames=False).build(ev)
    img2 = cv2.imdecode(np.frombuffer(no_ctx, np.uint8), cv2.IMREAD_COLOR)
    assert img2.shape[0] < img.shape[0]


def test_build_many_keys_by_event_id_and_skips_frames_past_the_video(tmp_path):
    w = _work(_walker(1, range(60)), _walker(2, [10, 500], x0=150.0))
    b = _builder(tmp_path, w, {1: RED})
    e1 = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static", spans=None)
    e2 = _ev(EventKind.DROP, "screen", [2], [[[2, 10, 500]]], (10, 500), reason="static",
             spans=None)
    e2.id = "screen-r0-000002"
    out = b.build_many([e1, e2])
    assert set(out) == {"screen-r0-000001", "screen-r0-000002"}
    assert out["screen-r0-000001"] is not None
    assert out["screen-r0-000002"] is not None  # frame 10 is drawn, frame 500 is skipped


def test_an_event_with_nothing_to_draw_gives_none(tmp_path):
    # identity evidence needs clean crops: an always-overlapping track has none, so a SPLIT on it
    # (no context frames either) has nothing to draw; a screen event on the same track does
    # (see test_screen_evidence_keeps_overlapping_rows_that_identity_evidence_drops)
    w = _work(_walker(1, range(60)))
    occ = pd.Series(True, index=w.index)
    b = _builder(tmp_path, w, {1: RED}, occluded=occ, send_context_frames=False)
    split = _ev(EventKind.SPLIT, "switch", [1], [[[1, 0, 59]]], (30, 30), cut_frame=30)
    assert b.plan(split).is_empty and b.build(split) is None
    # a screen event whose lineage matches no observed row has nothing to draw either
    clean = _builder(tmp_path, w, {1: RED}, send_context_frames=False)
    ghost = _ev(EventKind.DROP, "screen", [1], [[[1, 500, 509]]], (500, 509), reason="static",
                spans=None)
    assert clean.plan(ghost).is_empty and clean.build(ghost) is None


def test_a_box_partly_outside_the_frame_still_gives_a_tile(tmp_path):
    w = _work(box_rows(1, range(10), -20.0, 200.0, w=40.0, h=60.0))  # cut at the left and bottom
    b = _builder(tmp_path, w, {1: RED})
    ev = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 9]]], (0, 9), reason="static", spans=None)
    assert len(b.plan(ev).rows[0][1]) == 6
    jpeg = b.build(ev)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"


def test_frames_are_read_in_one_sorted_pass_per_chunk(tmp_path, monkeypatch):
    from dnt.refine import evidence

    w = _work(_walker(1, range(60)))
    b = _builder(tmp_path, w, {1: RED})
    evs = []
    for i in range(3):
        e = _ev(EventKind.DROP, "screen", [1], [[[1, 0, 59]]], (0, 59), reason="static",
                spans=None)
        e.id = f"screen-r0-{i:06d}"
        evs.append(e)
    opened = []
    real = evidence.FrameReader

    def counting(path):
        opened.append(path)
        return real(path)

    monkeypatch.setattr(evidence, "FrameReader", counting)
    b.build_many(evs)
    assert len(opened) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_evidence.py -q`
Expected: collection error `No module named 'dnt.refine.evidence'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/evidence.py`:

```python
"""Composite evidence images for VLM questions and review cards (spec 7.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crops import FrameReader, crop_box
from .events import Event, EventKind

BACKGROUND = 128
EVIDENCE_PAD = 1.5
MIN_TILE_HEIGHT = 160
MAX_TILE_WIDTH = 360
CONTEXT_WIDTH = 768
GAP = 6
JPEG_QUALITY = 85
CHUNK = 64
GREEN, RED, YELLOW, GRAY = (0, 200, 0), (0, 0, 220), (0, 220, 220), (170, 170, 170)


@dataclass
class ContextBox:
    """A box drawn on a context frame."""

    label: str
    color: tuple
    dashed: bool
    xywh: tuple | None


@dataclass
class ContextTile:
    """A context frame, its caption, and the boxes drawn on it."""

    frame: int
    caption: str
    boxes: list[ContextBox] = field(default_factory=list)


@dataclass
class EvidencePlan:
    """What an evidence image shows: crop rows and context frames."""

    rows: list[tuple[str, list[tuple[int, int]]]] = field(default_factory=list)
    contexts: list[ContextTile] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to draw."""
        return not any(tiles for _, tiles in self.rows) and not self.contexts


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


class EvidenceBuilder:
    """Builds one composite JPEG per event from the video and the raw tracks."""

    def __init__(
        self,
        video_file,
        raw_work: pd.DataFrame,
        occluded: pd.Series,
        *,
        frame_count: int,
        send_context_frames: bool = True,
    ):
        """Index the raw boxes by ``(raw_id, frame)``; ``occluded`` aligns with ``raw_work``."""
        self.video_file = video_file
        self.frame_count = int(frame_count)
        self.send_context_frames = bool(send_context_frames)
        w = raw_work.assign(_occ=occluded.reindex(raw_work.index).fillna(False).to_numpy(bool))
        self._box = {
            (int(r), int(f)): (float(x), float(y), float(ww), float(hh))
            for r, f, x, y, ww, hh in zip(
                w["raw_id"], w["frame"], w["x"], w["y"], w["w"], w["h"], strict=True
            )
        }
        self._occ = {
            (int(r), int(f)): bool(o)
            for r, f, o in zip(w["raw_id"], w["frame"], w["_occ"], strict=True)
        }
        self._by_frame: dict[int, list[int]] = {}
        for r, f in self._box:
            self._by_frame.setdefault(f, []).append(r)

    # ---- planning (no video access) ----

    def _clean(self, spans, lo: int | None = None, hi: int | None = None):
        """Clean ``(raw_id, frame)`` pairs of the lineage ``spans``, sorted by frame."""
        out = []
        for raw, f0, f1 in spans:
            for f in range(int(f0), int(f1) + 1):
                if (lo is not None and f < lo) or (hi is not None and f > hi):
                    continue
                key = (int(raw), f)
                box = self._box.get(key)
                if box is None or self._occ[key] or box[2] <= 0 or box[3] <= 0:
                    continue
                if f >= self.frame_count:
                    continue
                out.append(key)
        return sorted(out, key=lambda k: k[1])

    def _observed(self, spans, lo: int | None = None, hi: int | None = None):
        """Like ``_clean`` but keeps rows flagged occluded (screen evidence, spec 7.1)."""
        out = []
        for raw, f0, f1 in spans:
            for f in range(int(f0), int(f1) + 1):
                if (lo is not None and f < lo) or (hi is not None and f > hi):
                    continue
                box = self._box.get((int(raw), f))
                if box is None or box[2] <= 0 or box[3] <= 0 or f >= self.frame_count:
                    continue
                out.append((int(raw), f))
        return sorted(out, key=lambda k: k[1])

    def _box_at(self, spans, frame: int):
        for raw, f0, f1 in spans:
            if int(f0) <= frame <= int(f1) and (int(raw), frame) in self._box:
                return self._box[(int(raw), frame)]
        return None

    def _ctx(self, frame, caption, boxes):
        if not self.send_context_frames or not 0 <= frame < self.frame_count:
            return []
        return [ContextTile(int(frame), caption, boxes)]

    def plan(self, event: Event) -> EvidencePlan:
        """Return the tiles the image of ``event`` will show (spec 7.1)."""
        plan = EvidencePlan()
        a_spans = event.lineage[0] if event.lineage else []
        if event.kind in (EventKind.DROP, EventKind.RECLASS):
            spans = event.params.get("spans")
            if spans:  # a partial edit: the supported segments (A) and the rest of the track (B)
                inside = sorted(
                    {k for lo, hi in spans for k in self._observed(a_spans, lo, hi)},
                    key=lambda k: k[1],
                )
                chosen = set(inside)
                outside = [k for k in self._observed(a_spans) if k not in chosen]
                plan.rows += [("A", _spread(inside, 3)), ("B", _spread(outside, 3))]
                anchor = inside
            else:
                anchor = self._observed(a_spans)
                plan.rows.append(("A", _spread(anchor, 6)))
            if anchor:
                f = anchor[len(anchor) // 2][1]
                box = self._box_at(a_spans, f)
                plan.contexts += self._ctx(f, "A", [ContextBox("A", GREEN, False, box)])
        elif event.kind is EventKind.SPLIT:
            t = int(event.params["cut_frame"])
            before = self._clean(a_spans, None, t - 1)[-3:]
            after = self._clean(a_spans, t, None)[:3]
            plan.rows += [("A", before), ("B", after)]
            boxes = [ContextBox("A", GREEN, False, self._box_at(a_spans, t))]
            own = {int(r) for r, _, _ in a_spans}
            mine = self._box_at(a_spans, t)
            for raw in self._by_frame.get(t, []):
                if raw in own or mine is None:
                    continue
                ob = self._box[(raw, t)]
                if abs(ob[0] - mine[0]) < 3 * mine[3] and abs(ob[1] - mine[1]) < 3 * mine[3]:
                    boxes.append(ContextBox("", GRAY, False, ob))
            plan.contexts += self._ctx(t, f"cut at {t}", boxes)
        elif event.kind is EventKind.LINK:
            b_spans = event.lineage[1]
            t_e, t_s = (int(v) for v in event.params["gap"])
            plan.rows += [
                ("A", self._clean(a_spans)[-3:]),
                ("B", self._clean(b_spans)[:3]),
            ]
            plan.contexts += self._ctx(
                t_e, "A ends", [ContextBox("A", GREEN, False, self._box_at(a_spans, t_e))]
            )
            if event.params.get("gate") == "occluded":
                a_last, b_first = self._box_at(a_spans, t_e), self._box_at(b_spans, t_s)
                mid = (t_e + t_s) // 2
                hidden = None
                if a_last is not None and b_first is not None and t_s > t_e:
                    k = (mid - t_e) / (t_s - t_e)
                    hidden = tuple(a + k * (b - a) for a, b in zip(a_last, b_first, strict=True))
                plan.contexts += self._ctx(
                    mid, "hidden path", [ContextBox("?", YELLOW, True, hidden)]
                )
            plan.contexts += self._ctx(
                t_s, "B starts", [ContextBox("B", RED, False, self._box_at(b_spans, t_s))]
            )
        return plan

    # ---- rendering ----

    def build(self, event: Event) -> bytes | None:
        """Return the composite JPEG for one event, or ``None``."""
        return self.build_many([event]).get(event.id)

    def build_many(self, events: list[Event]) -> dict[str, bytes | None]:
        """Return ``{event.id: JPEG or None}``; frames are read once per chunk, in order."""
        out: dict[str, bytes | None] = {}
        for i in range(0, len(events), CHUNK):
            chunk = events[i : i + CHUNK]
            plans = {e.id: self.plan(e) for e in chunk}
            parts: dict[tuple, np.ndarray] = {}
            requests: dict[int, list[tuple]] = {}
            for e in chunk:
                p = plans[e.id]
                for ri, (_, tiles) in enumerate(p.rows):
                    for ti, (raw, f) in enumerate(tiles):
                        requests.setdefault(f, []).append((e.id, "crop", ri, ti, raw))
                for ci, ctx in enumerate(p.contexts):
                    requests.setdefault(ctx.frame, []).append((e.id, "ctx", ci, 0, 0))
            if requests:
                with FrameReader(self.video_file) as reader:
                    for f, img in reader.frames(requests):
                        for eid, kind, a, b, raw in requests[f]:
                            if kind == "crop":
                                tile = self._crop_tile(img, raw, f)
                            else:
                                tile = self._context_tile(img, plans[eid].contexts[a])
                            if tile is not None:
                                parts[(eid, kind, a, b)] = tile
            for e in chunk:
                out[e.id] = self._compose(plans[e.id], parts, e.id)
        return out

    def _crop_tile(self, img, raw: int, frame: int):
        import cv2

        crop = crop_box(img, self._box[(raw, frame)], EVIDENCE_PAD)
        if crop is None:
            return None
        h, w = crop.shape[:2]
        scale = max(1.0, MIN_TILE_HEIGHT / h)
        scale = min(scale, MAX_TILE_WIDTH / w) if w * scale > MAX_TILE_WIDTH else scale
        tile = cv2.resize(
            np.ascontiguousarray(crop[..., ::-1]),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.putText(tile, f"f{frame}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return tile

    def _context_tile(self, img, ctx: ContextTile):
        import cv2

        _, w = img.shape[:2]
        scale = CONTEXT_WIDTH / w if w > CONTEXT_WIDTH else 1.0
        frame = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()
        for box in ctx.boxes:
            if box.xywh is None:
                continue
            x, y, bw, bh = (v * scale for v in box.xywh)
            p0, p1 = (int(x), int(y)), (int(x + bw), int(y + bh))
            if box.dashed:
                _dashed_rect(frame, p0, p1, box.color)
            else:
                cv2.rectangle(frame, p0, p1, box.color, 1)
            if box.label:
                cv2.putText(
                    frame,
                    box.label,
                    (p0[0], max(12, p0[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box.color,
                    1,
                )
        cv2.putText(
            frame,
            f"{ctx.caption} (f{ctx.frame})",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame

    def _compose(self, plan: EvidencePlan, parts: dict, eid: str) -> bytes | None:
        import cv2

        rows = []
        for ri, (label, tiles) in enumerate(plan.rows):
            imgs = [
                parts[(eid, "crop", ri, ti)]
                for ti in range(len(tiles))
                if (eid, "crop", ri, ti) in parts
            ]
            if imgs:
                rows.append(_hstack(imgs, label))
        for ci in range(len(plan.contexts)):
            if (eid, "ctx", ci, 0) in parts:
                rows.append(parts[(eid, "ctx", ci, 0)])
        if not rows:
            return None
        width = max(r.shape[1] for r in rows)
        canvas = np.full(
            (sum(r.shape[0] + GAP for r in rows) + GAP, width + 2 * GAP, 3), BACKGROUND, np.uint8
        )
        y = GAP
        for r in rows:
            canvas[y : y + r.shape[0], GAP : GAP + r.shape[1]] = r
            y += r.shape[0] + GAP
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None


def _hstack(imgs: list[np.ndarray], label: str) -> np.ndarray:
    import cv2

    height = max(i.shape[0] for i in imgs)
    cols = [np.full((height, 28, 3), BACKGROUND, np.uint8)]
    cv2.putText(cols[0], label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for im in imgs:
        pad = np.full((height - im.shape[0], im.shape[1], 3), BACKGROUND, np.uint8)
        cols += [np.vstack([im, pad]), np.full((height, GAP, 3), BACKGROUND, np.uint8)]
    return np.hstack(cols)


def _dashed_rect(img, p0, p1, color, dash: int = 6) -> None:
    import cv2

    (x0, y0), (x1, y1) = p0, p1
    for x in range(x0, x1, 2 * dash):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, 1)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, 1)
    for y in range(y0, y1, 2 * dash):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, 1)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, 1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_evidence.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/evidence.py tests/refine/test_evidence.py
git commit -m "feat(refine): build composite evidence images for VLM questions" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Routing with a VLM in `verify.py`

**Files:**
- Modify: `src/dnt/refine/verify.py`
- Test: `tests/refine/test_verify_vlm.py`

**Interfaces:**
- Consumes: Tasks 1-5 (`VLMRunner`, `Question`, `Verdict`, `options_for`, `build_prompt`, `RIDER_OPTIONS`, an evidence object with `build_many(events) -> {event.id: bytes | None}`); P1's `Band`, `band_route`, `decide`, `route_without_vlm`; `RefineConfig` (`target`, `reclass_map`, `vlm.min_conf`).
- Produces:
  - `VLMRouting(runner, evidence, cfg, fps)` (dataclass) and `route_with_vlm(events, band, *, vlm: VLMRouting, round=0) -> None`.
  - `interpret(event, verdict, *, mode, cfg) -> tuple[Decision, dict | None]`: the decision and the redirected edit (`{"kind": ..., "params": ...}`, or `None` to keep the proposal's edit). `mode` is `"decide"` (an uncertain-band event) or `"subtype"` (an `AUTO_ACCEPT` rider `RECLASS` with no subtype yet).
  - **Events must have ids before routing** (the question tag is `"<KIND>:<event id>"`); Task 7 moves id assignment in front of routing.
- Behavior (§4.3, §7.2): each event is first banded. `AUTO_ACCEPT` / `AUTO_REJECT` are decided with source `auto`. An uncertain-band event of kind SPLIT/LINK (stages `switch`/`link`) or DROP/RECLASS (stage `screen`) is asked; any other uncertain event (orphan, fill, smooth) becomes `HUMAN_PENDING` without a call. An `AUTO_ACCEPT` RECLASS with `new_cls is None` is asked in `subtype` mode. The asked events go to the runner **ordered by `|algo_score - (accept_above + reject_below) / 2|`, closest first** (ties keep list order), so the budget is spent where the algorithm is least sure. An event with no evidence image, a verdict with an error, a tie, `unsure`, or `confidence < vlm.min_conf` becomes `HUMAN_PENDING` (source `vlm` when a call was made) with `event.vlm` recording the backend, model, answer, confidence, votes, reason, `evidence: None`, `cached`, and `error`.
  - Answer mapping: SPLIT `different` -> `VLM_ACCEPT`, `same_individual` -> `VLM_REJECT`; LINK the other way round. Person screen: `pedestrian` -> `VLM_REJECT`; `person_in_vehicle` -> `VLM_ACCEPT` with edit `DROP{reason: "in_vehicle", spans}`; `not_a_person` -> `VLM_ACCEPT` with `DROP{reason: "static", spans}`; `cyclist` / `motorcycle_rider` / `scooter_rider` -> `VLM_ACCEPT` with `RECLASS{new_cls: reclass_map[RIDER_OPTIONS[answer]], spans}` (a missing `reclass_map` key leaves the event pending). Vehicle screen: `vehicle` -> `VLM_REJECT`; `part_or_duplicate_of_another_vehicle` -> `VLM_ACCEPT` (edit unchanged) only for a `duplicate` event and otherwise treated as `unsure`; `not_a_vehicle` -> `VLM_ACCEPT` with `DROP{reason: "static", spans}`. `spans` is copied from the proposal. Subtype mode: a rider answer keeps the event `AUTO_ACCEPT` (source `auto`) and sets `edit.params.new_cls`; any other answer, `unsure`, low confidence, or an error leaves it `HUMAN_PENDING` with `signals["needs_subtype"] = True`.
  - `kind`, `params` and `proposal_key` are never changed (only `edit` is).

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_verify_vlm.py`:

```python
import json

import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.verify import Band, VLMRouting, interpret, route_with_vlm, route_without_vlm
from dnt.refine.vlm import VLMTransientError
from dnt.refine.vlm.cache import AnswerCache
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.prompts import build_prompt, options_for
from dnt.refine.vlm.runner import Verdict, VLMRunner

BAND = Band(accept_above=0.875, reject_below=0.375)  # midpoint 0.625, exact in binary


def reply(answer, conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf, "reason": "because"})


class Images:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.asked = []

    def build_many(self, events):
        self.asked += [e.id for e in events]
        return {e.id: (None if e.id in self.missing else b"jpeg") for e in events}


def make(kind, stage, score, idx=1, tracks=(1,), **params):
    lineage = [[[t, 0, 99]] for t in tracks]
    ev = Event.propose(stage=stage, kind=kind, tracks=list(tracks), lineage=lineage,
                       frames=(0, 99), params=params, algo_score=score, signals={})
    ev.id = f"{stage}-r0-{idx:06d}"
    return ev


CREATED = []


@pytest.fixture(autouse=True)
def _close_runners():
    yield
    for r in CREATED:
        r.close()  # stop each runner's loop thread, so later tests see no stray threads
    CREATED.clear()


def routing(script, target="person", **vlm):
    cfg = RefineConfig.defaults(target)
    for k, v in vlm.items():
        setattr(cfg.vlm, k, v)
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "m"
    backend = FakeBackend(script)
    runner = VLMRunner(cfg.vlm, backend, None, sleep=lambda s: _noop())
    CREATED.append(runner)
    images = Images()
    return VLMRouting(runner, images, cfg, 10.0), backend, images


async def _noop():
    return None


def link(score, idx=1, gate="normal"):
    return make(EventKind.LINK, "link", score, idx, tracks=(1, 2), gap=[50, 60], gate=gate)


def test_banded_events_never_call_the_vlm():
    r, b, _ = routing({})
    hi, lo = link(0.95, 1), link(0.2, 2)
    route_with_vlm([hi, lo], BAND, vlm=r)
    assert (hi.decision, lo.decision) == (Decision.AUTO_ACCEPT, Decision.AUTO_REJECT)
    assert b.calls == [] and hi.vlm is None
    assert [h["source"] for h in hi.decision_history] == ["auto"]


@pytest.mark.parametrize(
    "answer,conf,decision,edit",
    [
        ("same_individual", 0.9, Decision.VLM_ACCEPT, True),
        ("different", 0.9, Decision.VLM_REJECT, False),
        ("unsure", 0.99, Decision.HUMAN_PENDING, False),
        ("same_individual", 0.5, Decision.HUMAN_PENDING, False),  # below min_conf 0.7
    ],
)
def test_link_answers(answer, conf, decision, edit):
    r, b, _ = routing({"*": reply(answer, conf)})
    ev = link(0.6)
    key, kind, params = ev.proposal_key, ev.kind, dict(ev.params)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision
    assert (ev.edit is not None) is edit
    if edit:
        assert ev.edit == {"kind": "LINK", "params": params}
    assert (ev.proposal_key, ev.kind, ev.params) == (key, kind, params)
    assert ev.vlm["answer"] == answer and ev.vlm["confidence"] == conf
    assert ev.vlm["backend"] == "fake" and ev.vlm["model"] == "fake-1"
    # a confident answer has no error; one below vlm.min_conf says why it stayed pending
    assert (ev.vlm["error"] is None) if conf >= 0.7 else ("min_conf" in ev.vlm["error"])
    assert ev.vlm["evidence"] is None and ev.vlm["cached"] is False
    assert ev.decision_history[-1]["source"] == "vlm"
    assert b.calls[0]["tag"] == f"LINK:{ev.id}" and b.calls[0]["options"][0] == "same_individual"


def test_split_answers_are_inverted():
    r, _, _ = routing({"SPLIT": [reply("different"), reply("same_individual")]},
                      max_concurrency=1)
    a = make(EventKind.SPLIT, "switch", 0.625, 1, cut_frame=40)  # asked first (closest to mid)
    b = make(EventKind.SPLIT, "switch", 0.5, 2, cut_frame=70)
    route_with_vlm([a, b], BAND, vlm=r)
    assert a.decision is Decision.VLM_ACCEPT and b.decision is Decision.VLM_REJECT


@pytest.mark.parametrize(
    "answer,decision,edit_kind,edit_params",
    [
        ("pedestrian", Decision.VLM_REJECT, None, None),
        ("person_in_vehicle", Decision.VLM_ACCEPT, "DROP", {"reason": "in_vehicle", "spans": None}),
        ("not_a_person", Decision.VLM_ACCEPT, "DROP", {"reason": "static", "spans": None}),
        ("cyclist", Decision.VLM_ACCEPT, "RECLASS", {"new_cls": 1, "spans": None}),
        ("motorcycle_rider", Decision.VLM_ACCEPT, "RECLASS", {"new_cls": 3, "spans": None}),
        ("scooter_rider", Decision.VLM_ACCEPT, "RECLASS", {"new_cls": 36, "spans": None}),
        ("unsure", Decision.HUMAN_PENDING, None, None),
    ],
)
def test_person_screen_answers_redirect_the_edit_not_the_proposal(
    answer, decision, edit_kind, edit_params
):
    r, _, _ = routing({"*": reply(answer)})
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason="static", spans=None)
    key, params = ev.proposal_key, dict(ev.params)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision
    if edit_kind is None:
        assert ev.edit is None
    else:
        assert ev.edit == {"kind": edit_kind, "params": edit_params}
    assert ev.kind is EventKind.DROP and ev.params == params and ev.proposal_key == key


def test_partial_spans_are_copied_into_the_redirected_edit():
    r, _, _ = routing({"*": reply("cyclist")})
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason="static", spans=[[10, 20]])
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.edit == {"kind": "RECLASS", "params": {"new_cls": 1, "spans": [[10, 20]]}}


def test_a_missing_reclass_map_key_leaves_the_event_pending():
    r, _, _ = routing({"*": reply("scooter_rider")})
    del r.cfg.reclass_map["scooter"]
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason="static", spans=None)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and "scooter" in ev.vlm["error"]


@pytest.mark.parametrize(
    "reason,answer,decision",
    [
        ("duplicate", "part_or_duplicate_of_another_vehicle", Decision.VLM_ACCEPT),
        ("static", "part_or_duplicate_of_another_vehicle", Decision.HUMAN_PENDING),
        ("duplicate", "vehicle", Decision.VLM_REJECT),
        ("static", "not_a_vehicle", Decision.VLM_ACCEPT),
    ],
)
def test_vehicle_screen_answers(reason, answer, decision):
    r, b, _ = routing({"*": reply(answer)}, target="vehicle")
    extra = {"of": 2} if reason == "duplicate" else {}
    ev = make(EventKind.DROP, "screen", 0.6, 1, reason=reason, spans=None, **extra)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision
    assert b.calls[0]["options"][0] == "vehicle"
    if decision is Decision.VLM_ACCEPT and answer == "not_a_vehicle":
        assert ev.edit == {"kind": "DROP", "params": {"reason": "static", "spans": None}}
    if decision is Decision.VLM_ACCEPT and reason == "duplicate":
        assert ev.edit["params"]["reason"] == "duplicate" and ev.edit["params"]["of"] == 2


@pytest.mark.parametrize(
    "answer,decision,new_cls",
    [
        ("cyclist", Decision.AUTO_ACCEPT, 1),
        ("scooter_rider", Decision.AUTO_ACCEPT, 36),
        ("pedestrian", Decision.HUMAN_PENDING, None),
        ("not_a_person", Decision.HUMAN_PENDING, None),
        ("unsure", Decision.HUMAN_PENDING, None),
    ],
)
def test_rider_subtype_call_cannot_overturn_the_rider_decision(answer, decision, new_cls):
    r, b, _ = routing({"*": reply(answer)})
    ev = make(EventKind.RECLASS, "screen", 0.95, 1, new_cls=None, spans=None)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is decision and len(b.calls) == 1
    if new_cls is not None:
        assert ev.edit["params"]["new_cls"] == new_cls and ev.params["new_cls"] is None
        assert ev.decision_history[-1]["source"] == "auto"
    else:
        assert ev.edit is None and ev.signals["needs_subtype"] is True


def test_a_non_finite_confidence_never_accepts_an_edit():
    cfg = RefineConfig.defaults()
    for conf in (float("nan"), float("inf")):
        v = Verdict("different", conf, "", {"different": 1}, False, None)
        split = make(EventKind.SPLIT, "switch", 0.6, 1, cut_frame=40)
        assert interpret(split, v, mode="decide", cfg=cfg) == (Decision.HUMAN_PENDING, None)
        rider = make(EventKind.RECLASS, "screen", 0.95, 2, new_cls=None, spans=None)
        v2 = Verdict("cyclist", conf, "", {"cyclist": 1}, False, None)
        assert interpret(rider, v2, mode="subtype", cfg=cfg) == (Decision.HUMAN_PENDING, None)


def test_a_damaged_cache_entry_cannot_accept_an_edit(tmp_path):
    ev = make(EventKind.SPLIT, "switch", 0.6, 1, cut_frame=40)
    r, _, _ = routing({})  # nothing is scripted: any backend call fails
    prompt = build_prompt(ev, "person", options_for(ev, "person"), fps=10.0)
    key = AnswerCache.key(b"jpeg", prompt, options_for(ev, "person"), "fake", "fake-1", 0.0, 0)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"answer": "different", "confidence": float("nan"),
                                "reason": "r", "raw": "x"}))  # NaN literal, readable JSON
    r.runner.cache = AnswerCache(tmp_path)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and ev.edit is None
    assert r.runner.cache_hits == 0 and ev.vlm["error"].startswith("RuntimeError")


def test_a_rider_with_a_hinted_subtype_needs_no_call():
    r, b, _ = routing({})
    ev = make(EventKind.RECLASS, "screen", 0.95, 1, new_cls=3, spans=None)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.AUTO_ACCEPT and b.calls == []


def test_the_budget_goes_to_the_events_closest_to_the_band_midpoint():
    r, b, _ = routing({"*": reply("same_individual")}, max_calls=2, max_concurrency=1)
    scores = [0.5, 0.625, 0.75, 0.5625, 0.6875]  # distances from 0.625: .125 0 .125 1/16 1/16
    evs = [link(s, i + 1) for i, s in enumerate(scores)]
    route_with_vlm(evs, BAND, vlm=r)
    answered = [e.id for e in evs if e.decision is Decision.VLM_ACCEPT]
    assert answered == ["link-r0-000002", "link-r0-000004"]  # 0, then 1/16 (list order tie)
    left = [e for e in evs if e.decision is Decision.HUMAN_PENDING]
    assert len(left) == 3 and all(e.vlm["error"] == "budget" for e in left)
    assert [c["tag"] for c in b.calls] == ["LINK:link-r0-000002", "LINK:link-r0-000004"]


def test_orphans_and_position_records_are_never_asked():
    r, b, images = routing({})
    orphan = make(EventKind.DROP, "orphan", 0.5, 1, reason="orphan", spans=None)
    route_with_vlm([orphan], Band(0.7, 0.3), vlm=r)
    assert orphan.decision is Decision.HUMAN_PENDING and b.calls == [] and images.asked == []


def test_no_evidence_image_means_pending_without_a_call():
    r, b, images = routing({"*": reply("same_individual")})
    images.missing = {"link-r0-000001"}
    ev = link(0.6)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and b.calls == []
    assert ev.vlm["error"] == "no evidence image"


def test_a_failing_evidence_builder_leaves_events_pending_not_crashed():
    class Broken:
        def build_many(self, events):
            raise ValueError("cannot read frame 5")

    r, b, _ = routing({"*": reply("same_individual")})
    r.evidence = Broken()
    ev = link(0.6)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and b.calls == []
    assert ev.vlm["error"] == "no evidence image"


def test_backend_failures_leave_the_event_pending():
    r, _, _ = routing({"*": [RuntimeError("boom")]})
    ev = link(0.6)
    route_with_vlm([ev], BAND, vlm=r)
    assert ev.decision is Decision.HUMAN_PENDING and ev.edit is None
    assert ev.vlm["answer"] is None and ev.vlm["error"].startswith("RuntimeError")
    r2, _, _ = routing({"*": [VLMTransientError("429")]})
    ev2 = link(0.6)
    route_with_vlm([ev2], BAND, vlm=r2)
    assert ev2.decision is Decision.HUMAN_PENDING and ev2.vlm["error"].startswith("transient")


def test_the_without_vlm_path_is_unchanged():
    ev = link(0.6)
    route_without_vlm([ev], BAND)
    assert ev.decision is Decision.HUMAN_PENDING and ev.vlm is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_verify_vlm.py -q`
Expected: ImportError (`VLMRouting`, `route_with_vlm`).

- [ ] **Step 3: Implement**

In `src/dnt/refine/verify.py` change the module docstring first line to `"""Routing proposals by confidence band and by a VLM (spec 4.3, 7.2)."""`, and extend the imports:

```python
import copy
import logging
import math
from dataclasses import dataclass

from .events import ACCEPTED, Decision, Event, EventKind
from .vlm.prompts import RIDER_OPTIONS, build_prompt, options_for
from .vlm.runner import Question, Verdict
```

(the module still imports no heavy package: `dnt.refine.vlm` only imports the standard library at import time), and define `log = logging.getLogger(__name__)` after the imports.

Append to `src/dnt/refine/verify.py`:

```python
@dataclass
class VLMRouting:
    """Everything `route_with_vlm` needs: the runner, the evidence builder, the config, fps."""

    runner: object
    evidence: object
    cfg: object
    fps: float


def _vlm_record(vlm: VLMRouting, verdict: Verdict | None, error: str | None) -> dict:
    return {
        "backend": vlm.runner.backend.name,
        "model": vlm.runner.backend.model,
        "answer": None if verdict is None else verdict.answer,
        "confidence": 0.0 if verdict is None else verdict.confidence,
        "votes": {} if verdict is None else verdict.votes,
        "reason": "" if verdict is None else verdict.reason,
        "evidence": None,
        "cached": False if verdict is None else verdict.cached,
        "error": error,
    }


def _screen_edit(event: Event, kind: str, **params) -> dict:
    return {"kind": kind, "params": {**params, "spans": copy.deepcopy(event.params.get("spans"))}}


def interpret(event: Event, verdict: Verdict, *, mode: str, cfg) -> tuple[Decision, dict | None]:
    """Map a verdict to ``(decision, redirected edit or None)`` (spec 7.2).

    A returned edit of ``None`` means "keep the proposal's edit" (or no edit when the decision
    is not an acceptance). Errors, ties, ``unsure`` and low confidence are ``HUMAN_PENDING``.
    """
    pending = (Decision.HUMAN_PENDING, None)
    ans = verdict.answer
    if verdict.error is not None or ans is None or ans == "unsure":
        return pending
    if not math.isfinite(verdict.confidence) or verdict.confidence < float(cfg.vlm.min_conf):
        return pending
    if mode == "subtype":
        if ans not in RIDER_OPTIONS or RIDER_OPTIONS[ans] not in cfg.reclass_map:
            return pending
        new_cls = int(cfg.reclass_map[RIDER_OPTIONS[ans]])
        return Decision.AUTO_ACCEPT, _screen_edit(event, "RECLASS", new_cls=new_cls)
    if event.kind is EventKind.SPLIT:
        return (Decision.VLM_ACCEPT if ans == "different" else Decision.VLM_REJECT), None
    if event.kind is EventKind.LINK:
        return (Decision.VLM_ACCEPT if ans == "same_individual" else Decision.VLM_REJECT), None
    if cfg.target == "person":
        if ans == "pedestrian":
            return Decision.VLM_REJECT, None
        if ans == "person_in_vehicle":
            return Decision.VLM_ACCEPT, _screen_edit(event, "DROP", reason="in_vehicle")
        if ans == "not_a_person":
            return Decision.VLM_ACCEPT, _screen_edit(event, "DROP", reason="static")
        if ans in RIDER_OPTIONS and RIDER_OPTIONS[ans] in cfg.reclass_map:
            new_cls = int(cfg.reclass_map[RIDER_OPTIONS[ans]])
            return Decision.VLM_ACCEPT, _screen_edit(event, "RECLASS", new_cls=new_cls)
        return pending
    if ans == "vehicle":
        return Decision.VLM_REJECT, None
    if ans == "part_or_duplicate_of_another_vehicle":
        if event.params.get("reason") == "duplicate":
            return Decision.VLM_ACCEPT, None
        return pending
    if ans == "not_a_vehicle":
        return Decision.VLM_ACCEPT, _screen_edit(event, "DROP", reason="static")
    return pending


def route_with_vlm(events: list[Event], band: Band, *, vlm: VLMRouting, round: int = 0) -> None:
    """Route ``events`` by band, sending the uncertain ones to the VLM (spec 4.3, 7.2, 7.4).

    Expects undecided events that already have ids. See ``interpret`` for the answer mapping.
    """
    cfg = vlm.cfg
    asked: list[tuple[int, Event, str, list[str]]] = []
    for i, ev in enumerate(events):
        d = band_route(ev.algo_score, band)
        options = options_for(ev, cfg.target)
        subtype = (
            d is Decision.AUTO_ACCEPT
            and ev.kind is EventKind.RECLASS
            and ev.params.get("new_cls") is None
        )
        if subtype:
            ev.signals["needs_subtype"] = True
        if options is not None and (d is None or subtype):
            asked.append((i, ev, "subtype" if subtype else "decide", options))
        else:
            decide(ev, Decision.HUMAN_PENDING if d is None else d, source="auto", round=round)
    if not asked:
        return
    mid = (band.accept_above + band.reject_below) / 2.0
    asked.sort(key=lambda t: (abs(t[1].algo_score - mid), t[0]))
    try:
        images = vlm.evidence.build_many([ev for _, ev, _, _ in asked])
    except Exception as err:  # VLM trouble never aborts a run (spec 7.4)
        log.warning("could not build the evidence images: %s", err)
        images = {}
    todo, questions = [], []
    for _, ev, mode, options in asked:
        image = images.get(ev.id)
        if image is None:
            ev.vlm = _vlm_record(vlm, None, "no evidence image")
            decide(ev, Decision.HUMAN_PENDING, source="auto", round=round)
            continue
        prompt = build_prompt(ev, cfg.target, options, fps=vlm.fps)
        questions.append(Question(f"{ev.kind}:{ev.id}", image, prompt, options))
        todo.append((ev, mode))
    verdicts = vlm.runner.ask_many(questions)
    for (ev, mode), verdict in zip(todo, verdicts, strict=True):
        decision, edit = interpret(ev, verdict, mode=mode, cfg=cfg)
        error = verdict.error
        if error is None and decision is Decision.HUMAN_PENDING and verdict.answer is not None:
            error = _pending_reason(ev, verdict, mode, cfg)
        ev.vlm = _vlm_record(vlm, verdict, error)
        source = "auto" if decision is Decision.AUTO_ACCEPT else "vlm"
        decide(ev, decision, source=source, round=round)
        if edit is not None:
            ev.edit = edit
        if mode == "subtype" and decision is Decision.AUTO_ACCEPT:
            ev.signals.pop("needs_subtype", None)


def _pending_reason(ev: Event, verdict: Verdict, mode: str, cfg) -> str | None:
    """Say why an answered event stayed pending (None for a plain `unsure`)."""
    ans = verdict.answer
    if ans == "unsure":
        return None
    if not math.isfinite(verdict.confidence):
        return "the confidence is not a finite number"
    if verdict.confidence < float(cfg.vlm.min_conf):
        return f"confidence {verdict.confidence:.2f} is below vlm.min_conf"
    if ans in RIDER_OPTIONS and RIDER_OPTIONS[ans] not in cfg.reclass_map:
        return f"reclass_map has no {RIDER_OPTIONS[ans]!r} entry"
    if mode == "subtype":
        return f"answered {ans!r}, which disagrees with the rider decision"
    return f"answered {ans!r}, which does not settle this event"
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_verify_vlm.py tests/refine/test_verify_features.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/verify.py tests/refine/test_verify_vlm.py
git commit -m "feat(refine): route uncertain events through a VLM and map its answers" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Wire the VLM into `TrackRefiner`

**Files:**
- Modify: `src/dnt/refine/refiner.py`, `tests/refine/_video.py` (one new helper)
- Test: `tests/refine/test_vlm_refine.py`; update `tests/refine/test_refiner.py::test_summary_counts_on_a_two_class_table`, the only existing test that pins the exact summary `vlm` dict (`{"calls": 0, "cache_hits": 0, "failures": 0}`), which now has two more keys (`retries`, `budget_skipped`).

**Interfaces:**
- Consumes: Tasks 1-6 and P2's `TrackRefiner` / `_Stages` / `stage_views`.
- Produces:
  - `TrackRefiner(..., vlm_backend_factory=None)`: `vlm_backend_factory(vlm_cfg) -> VLMBackend` replaces `make_backend` (tests, custom backends) and skips the dependency check.
  - Behavior of `refine` when `cfg.vlm.backend != "none"`:
    - **With a video:** `check_vlm_dependencies` runs first (`ImportError` naming the `refine-vlm` extra, before any file is read or written, unless a factory is given); the backend is built then (a missing API key raises `ValueError` naming the variable, also before any work); a `VLMRunner` with an `AnswerCache(cfg.vlm.cache_dir)` is created per `refine` call, so `max_calls` is per run, and it is **closed in a `finally`** around the stage run (success or failure: the loop thread stops and the backend's client is closed on its own loop); `_Stages` builds an `EvidenceBuilder` from the raw work table and the raw occlusion flags and routes the `switch`, `screen` and `link` stages through `route_with_vlm`.
    - **Without a video:** a WARNING says the backend is ignored; nothing is sent; uncertain events stay `HUMAN_PENDING` (spec §10).
  - `_Stages._route` assigns event ids **before** routing (the question tag needs them); `_Stages.evidence` holds the `EvidenceBuilder` whenever a video is given (Task 8 reuses it for the review images).
  - The summary's `vlm` entry is `{"calls", "retries", "cache_hits", "failures", "budget_skipped"}` from the runner (all zero without a backend).
  - Helper `_video.takeover_scene(tmp_path, n_tracks=1, name="t.txt", video="v.mp4", first_id=1)` -> `(track_file, video)`; track `k` gets the raw id `first_id + k`: per track a red object for 60 frames, then a bigger blue object takes the ID over at frame 60 (the Plan 2 scene), tracks stacked vertically. With `encoder.kind: none` each gives a motion-only `SPLIT` capped at 0.70, which is in the uncertain band.

- [ ] **Step 1: Write the helper and the failing tests**

Append to `tests/refine/_video.py`:

```python
def takeover_scene(tmp_path, n_tracks=1, name="t.txt", video="v.mp4", first_id=1):
    """Per track: a red object for 60 frames, then a bigger blue one takes the ID over."""
    from ._fixtures import box_rows, table

    rows, vrows = [], []
    for k in range(n_tracks):
        y = 30.0 + 70.0 * k
        red = box_rows(first_id + k, range(60), 20.0, y, vx=2.0, w=20.0, h=40.0)
        blue = box_rows(first_id + k, range(60, 120), 140.0, y, vx=2.0, w=35.0, h=40.0)
        rows += red + blue
        vrows += video_rows(red, RED) + video_rows(blue, BLUE)
    vid = make_color_video(tmp_path / video, vrows, 120)
    src = tmp_path / name
    table(rows).to_csv(src, index=False, header=False)
    return src, vid
```

Create `tests/refine/test_vlm_refine.py`:

```python
import json
import logging
import sys
import threading

import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.refiner import TrackRefiner
from dnt.refine.vlm.fake import FakeBackend
from dnt.refine.vlm.runner import VLMRunner

from ._fixtures import box_rows, table
from ._video import RED, make_color_video, takeover_scene, video_rows


def reply(answer, conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf, "reason": "ok"})


def cfg_for(tmp_path, **vlm):
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"  # motion-only: a takeover split is capped at 0.70 (uncertain band)
    cfg.link.enabled = False
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "m"
    cfg.vlm.cache_dir = str(tmp_path / "vlmcache")
    for k, v in vlm.items():
        setattr(cfg.vlm, k, v)
    return cfg


def run(tmp_path, cfg, backend, src, video, out="o.txt", **kw):
    refiner = TrackRefiner(cfg, vlm_backend_factory=lambda c: backend)
    refiner.refine(src, tmp_path / out, video_file=video, verbose=False, **kw)
    return refiner.last_result


def splits(res):
    return [e for e in res.events if e.kind is EventKind.SPLIT]


def test_a_sure_answer_decides_and_applies_the_split(tmp_path):
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": reply("different")})
    res = run(tmp_path, cfg_for(tmp_path), backend, src, video)
    (ev,) = splits(res)
    assert ev.decision is Decision.VLM_ACCEPT and ev.applied and ev.params["cut_frame"] == 60
    assert ev.vlm["answer"] == "different" and ev.vlm["backend"] == "fake"
    assert res.tracks.track.nunique() == 2
    assert res.summary["vlm"] == {
        "calls": 1, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0
    }
    assert backend.calls[0]["tag"] == f"SPLIT:{ev.id}"
    led = Ledger.read(res.ledger_path)
    (back,) = [e for e in led.events if e.kind is EventKind.SPLIT]
    assert back.decision is Decision.VLM_ACCEPT and back.vlm == ev.vlm
    assert led.header["summary"]["vlm"]["calls"] == 1


def test_a_rejecting_answer_keeps_the_track_whole(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("same_individual")}),
              src, video)
    (ev,) = splits(res)
    assert ev.decision is Decision.VLM_REJECT and not ev.applied
    assert res.tracks.track.nunique() == 1


def test_an_unsure_or_failed_answer_leaves_it_pending_and_the_run_completes(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    assert splits(res)[0].decision is Decision.HUMAN_PENDING and res.tracks.track.nunique() == 1
    res2 = run(tmp_path, cfg_for(tmp_path, cache_dir=str(tmp_path / "other")),
               FakeBackend({"SPLIT": [RuntimeError("boom")]}), src, video, out="o2.txt")
    ev = splits(res2)[0]
    assert ev.decision is Decision.HUMAN_PENDING and ev.vlm["error"].startswith("RuntimeError")
    assert res2.summary["vlm"]["failures"] == 1


def test_the_budget_is_per_run_and_counted(tmp_path):
    src, video = takeover_scene(tmp_path, n_tracks=2)
    backend = FakeBackend({"SPLIT": reply("different")})
    res = run(tmp_path, cfg_for(tmp_path, max_calls=1), backend, src, video)
    decided = [e for e in splits(res) if e.decision is Decision.VLM_ACCEPT]
    left = [e for e in splits(res) if e.decision is Decision.HUMAN_PENDING]
    assert len(decided) == 1 and len(left) == 1 and left[0].vlm["error"] == "budget"
    assert res.summary["vlm"]["calls"] == 1 and res.summary["vlm"]["budget_skipped"] == 1


def test_a_retry_is_reported_in_the_result_the_ledger_and_the_budget(tmp_path):
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": ["not json", reply("different")]})  # an invalid reply, no sleep
    res = run(tmp_path, cfg_for(tmp_path, max_calls=2), backend, src, video)
    assert res.summary["vlm"] == {
        "calls": 1, "retries": 1, "cache_hits": 0, "failures": 0, "budget_skipped": 0
    }
    assert len(backend.calls) == 2
    assert Ledger.read(res.ledger_path).header["summary"]["vlm"]["retries"] == 1
    # max_calls is a hard limit on invocations: with one unit the retry is not allowed
    tight = FakeBackend({"SPLIT": ["not json", reply("different")]})
    res2 = run(tmp_path, cfg_for(tmp_path, max_calls=1, cache_dir=str(tmp_path / "c2")), tight,
               src, video, out="o2.txt")
    assert len(tight.calls) == 1 and res2.summary["vlm"]["retries"] == 0
    assert splits(res2)[0].decision is Decision.HUMAN_PENDING
    assert splits(res2)[0].vlm["error"] == "budget" and res2.summary["vlm"]["budget_skipped"] == 1


def test_a_rerun_with_the_same_cache_makes_no_backend_calls(tmp_path):
    src, video = takeover_scene(tmp_path)
    cfg = cfg_for(tmp_path)
    run(tmp_path, cfg, FakeBackend({"SPLIT": reply("different")}), src, video)
    again = FakeBackend({})  # asking would raise: the run must be served from the cache
    res = run(tmp_path, cfg, again, src, video, out="o2.txt")
    (ev,) = splits(res)
    assert again.calls == [] and ev.decision is Decision.VLM_ACCEPT and ev.vlm["cached"] is True
    assert res.summary["vlm"] == {
        "calls": 0, "retries": 0, "cache_hits": 1, "failures": 0, "budget_skipped": 0
    }


def test_context_frames_can_be_left_out_of_the_image(tmp_path):
    src, video = takeover_scene(tmp_path)
    with_ctx = FakeBackend({"SPLIT": reply("different")})
    run(tmp_path, cfg_for(tmp_path), with_ctx, src, video)
    without = FakeBackend({"SPLIT": reply("different")})
    run(tmp_path, cfg_for(tmp_path, send_context_frames=False, cache_dir=str(tmp_path / "c2")),
        without, src, video, out="o2.txt")
    assert without.calls[0]["image_len"] < with_ctx.calls[0]["image_len"]


def test_a_screen_event_can_be_redirected_to_a_reclass(tmp_path):
    # a motionless low-confidence box: its static score (about 0.61) is in the uncertain band
    rows = box_rows(1, range(120), 100.0, 80.0, w=40.0, h=80.0, score=0.35)
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 120)
    src = tmp_path / "t.txt"
    table(rows).to_csv(src, index=False, header=False)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"DROP": reply("cyclist")}), src, video)
    (ev,) = [e for e in res.events if e.stage == "screen"]
    assert ev.kind is EventKind.DROP and ev.decision is Decision.VLM_ACCEPT and ev.applied
    assert ev.edit["kind"] == "RECLASS" and ev.edit["params"]["new_cls"] == 1
    assert set(res.tracks["cls"]) == {1}


def test_an_always_overlapping_screen_event_reaches_the_vlm(tmp_path):
    # a person who is inside a context car for the whole track: every row is occluded, yet the
    # screen stage must still show the observed crops (the in-vehicle hypothesis relies on them)
    rows = box_rows(1, range(120), 100.0, 80.0, vx=2.0, w=20.0, h=40.0, score=0.6)
    # the car holds the person whole (IoB 1) at IoU 0.4: an occluder (>= encoder.occlusion_iou
    # 0.3), not the row's own detection (a matched IoU >= 0.5 is dropped as a duplicate)
    ctx = box_rows(9, range(120), 90.0, 75.0, vx=2.0, w=40.0, h=50.0, cls=2)
    video = make_color_video(tmp_path / "v.mp4", video_rows(rows, RED), 120)
    src, ctx_file = tmp_path / "t.txt", tmp_path / "c.txt"
    table(rows).to_csv(src, index=False, header=False)
    table(ctx).to_csv(ctx_file, index=False, header=False)
    cfg = cfg_for(tmp_path)
    # inside_frac is 1.0, which the default ramp scores 1.0 (>= accept_above, so AUTO_ACCEPT);
    # a wider ramp scores it 0.5, inside the band, so the VLM is asked
    cfg.screen.ramps["inside"] = [0.5, 1.5]
    backend = FakeBackend({"DROP": reply("person_in_vehicle")})
    refiner = TrackRefiner(cfg, vlm_backend_factory=lambda c: backend)
    refiner.refine(src, tmp_path / "o.txt", video_file=video, context_file=ctx_file,
                   verbose=False)
    res = refiner.last_result
    (ev,) = [e for e in res.events if e.stage == "screen"]
    assert len(backend.calls) == 1 and backend.calls[0]["image_len"] > 0
    assert ev.decision is Decision.VLM_ACCEPT and ev.applied
    assert ev.edit["kind"] == "DROP" and ev.edit["params"]["reason"] == "in_vehicle"
    assert res.tracks.empty


def test_the_backend_is_closed_after_a_run_and_after_a_failing_one(tmp_path, monkeypatch):
    src, video = takeover_scene(tmp_path)
    backend = FakeBackend({"SPLIT": reply("different")})
    run(tmp_path, cfg_for(tmp_path), backend, src, video)
    assert backend.closed == 1
    assert not [t for t in threading.enumerate() if t.name == "dnt-vlm-loop"]
    real = VLMRunner.ask_many

    def ask_then_fail(self, questions):
        real(self, questions)  # the loop is started and the client has been used
        raise RuntimeError("stage failed")

    monkeypatch.setattr(VLMRunner, "ask_many", ask_then_fail)
    failing = FakeBackend({"SPLIT": reply("different")})
    with pytest.raises(RuntimeError, match="stage failed"):
        run(tmp_path, cfg_for(tmp_path, cache_dir=str(tmp_path / "c2")), failing, src, video,
            out="o2.txt")
    assert failing.closed == 1
    assert not [t for t in threading.enumerate() if t.name == "dnt-vlm-loop"]


def test_without_a_video_the_backend_is_ignored_with_a_warning(tmp_path, caplog):
    src, _ = takeover_scene(tmp_path)
    backend = FakeBackend({})
    refiner = TrackRefiner(cfg_for(tmp_path), vlm_backend_factory=lambda c: backend)
    with caplog.at_level(logging.WARNING):
        refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    assert backend.calls == [] and "ignored" in caplog.text
    assert splits(refiner.last_result)[0].decision is Decision.HUMAN_PENDING
    assert refiner.last_result.summary["vlm"]["calls"] == 0


def test_a_missing_package_or_key_fails_before_any_file_is_written(tmp_path, monkeypatch):
    src, video = takeover_scene(tmp_path)
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-vlm\]"):
        TrackRefiner(cfg_for(tmp_path)).refine(src, tmp_path / "o.txt", video_file=video,
                                               verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]
    import importlib.machinery
    import types

    fake = types.ModuleType("anthropic")
    fake.__spec__ = importlib.machinery.ModuleSpec("anthropic", None)
    fake.AsyncAnthropic = object
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = cfg_for(tmp_path)
    cfg.vlm.backend, cfg.vlm.model = "anthropic", None
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        TrackRefiner(cfg).refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt", "v.mp4"]


def test_backend_none_changes_nothing(tmp_path):
    src, video = takeover_scene(tmp_path)
    cfg = cfg_for(tmp_path)
    cfg.vlm.backend, cfg.vlm.model = "none", None
    refiner = TrackRefiner(cfg)
    refiner.refine(src, tmp_path / "o.txt", video_file=video, verbose=False)
    assert splits(refiner.last_result)[0].decision is Decision.HUMAN_PENDING
    assert refiner.last_result.summary["vlm"] == {
        "calls": 0, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_refine.py -q`
Expected: failures (`TrackRefiner` has no `vlm_backend_factory`).

- [ ] **Step 3: Implement**

In `src/dnt/refine/refiner.py`:

1. Imports. Replace `from .verify import Band, decide, route_without_vlm` with

```python
from .verify import Band, VLMRouting, decide, route_with_vlm, route_without_vlm
```

   add `from .evidence import EvidenceBuilder` after `from .events import ...` (isort puts `.evidence` after `.events`), and after `from .video_appearance import VideoAppearance` add

```python
from .vlm import check_vlm_dependencies, make_backend
from .vlm.cache import AnswerCache
from .vlm.runner import VLMRunner
```

2. Add before `def _event_counts`:

```python
def _vlm_counts(runner) -> dict:
    if runner is None:
        return {"calls": 0, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0}
    return {
        "calls": runner.calls,
        "retries": runner.retries,
        "cache_hits": runner.cache_hits,
        "failures": runner.failures,
        "budget_skipped": runner.budget_skipped,
    }
```

3. `_Stages.__init__`: replace its signature and the first lines

```python
    def __init__(
        self, cfg: RefineConfig, fps: float, frame_size, appearance, ctx_boxes, ctx_fmt, hints
    ):
        """Hold the per-run inputs."""
```

   with

```python
    def __init__(
        self,
        cfg: RefineConfig,
        fps: float,
        frame_size,
        appearance,
        ctx_boxes,
        ctx_fmt,
        hints,
        *,
        vlm_runner=None,
        video=None,
        frame_count: int = 0,
    ):
        """Hold the per-run inputs."""
        self.vlm_runner, self.video, self.frame_count = vlm_runner, video, frame_count
        self.vlm: VLMRouting | None = None
        self.evidence: EvidenceBuilder | None = None
```

4. `_Stages._route`: replace

```python
        route_without_vlm(evs, band)
        for e in evs:
            self.seq[stage] += 1
            e.id = f"{stage}-r0-{self.seq[stage]:06d}"
        self.events.extend(evs)
```

   with

```python
        for e in evs:  # ids first: the VLM question tag uses them
            self.seq[stage] += 1
            e.id = f"{stage}-r0-{self.seq[stage]:06d}"
        if self.vlm is not None:
            route_with_vlm(evs, band, vlm=self.vlm)
        else:
            route_without_vlm(evs, band)
        self.events.extend(evs)
```

5. `_Stages.run`: directly after the line `occluded = occlusion_flags(work, self.ctx_boxes, self.cfg.encoder.occlusion_iou)` insert

```python
        if self.video is not None:
            self.evidence = EvidenceBuilder(
                self.video,
                work,
                occluded,
                frame_count=self.frame_count,
                send_context_frames=self.cfg.vlm.send_context_frames,
            )
            if self.vlm_runner is not None:
                self.vlm = VLMRouting(self.vlm_runner, self.evidence, self.cfg, self.fps)
```

6. `TrackRefiner.__init__`: add the keyword `vlm_backend_factory: Callable[..., object] | None = None,` after `encoder_factory`, and store `self.vlm_backend_factory = vlm_backend_factory`.

7. `TrackRefiner.refine`: directly after the block

```python
            check_encoder_dependencies(cfg.encoder)  # before any processing (spec 5.5)
```

   insert

```python
        runner = None
        if cfg.vlm.backend != "none" and video_file is None:
            log.warning(
                "vlm.backend %r is ignored without a video (no evidence images can be made); "
                "uncertain events stay pending",
                cfg.vlm.backend,
            )
        elif cfg.vlm.backend != "none":
            if self.vlm_backend_factory is None:
                check_vlm_dependencies(cfg.vlm)  # before any processing (spec 5.5)
            backend = (self.vlm_backend_factory or make_backend)(cfg.vlm)
            runner = VLMRunner(cfg.vlm, backend, AnswerCache(cfg.vlm.cache_dir))
```

   Extend the `Raises` section of the `refine` docstring. Replace

```
            package is not installed; the message names the pip extra.
```

   with

```
            package is not installed; or if a video is given, ``vlm.backend`` is set, and the
            backend's package is not installed; the message names the pip extra.
```

   and replace

```
            the input files; if no frame rate is known; or if an input is malformed.
```

   with

```
            the input files; if no frame rate is known; if an input is malformed; or, with a
            video and a VLM backend, if the backend's API key variable is not set.
```

8. Replace the `_Stages(...)` construction line

```python
            stages = _Stages(cfg, fps_val, frame_size, appearance, ctx_boxes, ctx_fmt, hint_map)
```

   with

```python
            stages = _Stages(
                cfg,
                fps_val,
                frame_size,
                appearance,
                ctx_boxes,
                ctx_fmt,
                hint_map,
                vlm_runner=runner,
                video=video_file,
                frame_count=(vinfo or {}).get("frame_count", 0),
            )
```

   replace `"vlm": {"calls": 0, "cache_hits": 0, "failures": 0},` with `"vlm": _vlm_counts(runner),`, and close the runner when the stages are done, whether they succeeded or not: the existing `try:` that wraps `stages.run` gets a `finally` after its `except BaseException:` block,

```python
        except BaseException:
            # (existing body unchanged)
            _save_on_failure(store, paths["features"])
            raise
        finally:
            if runner is not None:
                runner.close()  # stop the loop thread; close the backend's client on its loop
```

   The runner is created before the `try`, but it starts nothing until its first question, so an error in between (for example in `_appearance`) leaves no thread behind. The counters stay readable after `close()`, which is what `_vlm_counts(runner)` reads.

9. `tests/refine/test_refiner.py`, in `test_summary_counts_on_a_two_class_table`: replace

```python
    assert res.summary["vlm"] == {"calls": 0, "cache_hits": 0, "failures": 0}
```

   with

```python
    assert res.summary["vlm"] == {
        "calls": 0, "retries": 0, "cache_hits": 0, "failures": 0, "budget_skipped": 0
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_vlm_refine.py -q -W error` then `.venv/bin/python -m pytest tests/refine tests/test_refine_independence.py -q`
Expected: all pass (without item 9, `test_summary_counts_on_a_two_class_table` fails on the new keys).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/refiner.py tests/refine/_video.py tests/refine/test_vlm_refine.py tests/refine/test_refiner.py
git commit -m "feat(refine): route uncertain events through the VLM during refine" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The review page

**Files:**
- Create: `src/dnt/refine/review.py`
- Modify: `src/dnt/refine/refiner.py` (call it; set `RefineResult.review_path`), `src/dnt/refine/cli.py` (print the review path)
- Test: `tests/refine/test_review.py`; extend `tests/refine/test_vlm_refine.py`

**Interfaces:**
- Consumes: Task 5's `EvidenceBuilder` (`_Stages.evidence`, set whenever a video is given), the `HUMAN_PENDING` events, `RefineConfig.reclass_map`, the header's `id_map`, `output_paths(out)["review"]` (`OUT.review.html`; the image directory is `OUT.review/`).
- Produces:
  - `write_review(events, *, review_path, evidence, id_map, fps, video_file, track_file, reclass_map, title, run_key) -> Path | None`. It writes `OUT.review.html` and `OUT.review/<event id>.jpg` for the `HUMAN_PENDING` events and a manifest `OUT.review/.dnt-review.json` (`{"images": [names]}`) listing exactly the images it wrote; the dot-name is the report's own, so a user's `manifest.json` is never read, written or deleted. **Ownership is proven by that manifest only.** It deletes only files named in it (an unrelated `my-photo.jpg` or note in the directory is never touched, and with no valid manifest nothing but the page is removed). It never overwrites a file it does not own: if `<event id>.jpg` already exists and is not listed in the manifest, that card is shown without an image (with a WARNING) and the file stays intact. A manifest that is unreadable, not a JSON object, or whose `images` is not a list of plain `*.jpg` names proves nothing (it is treated as empty and replaced on the next pending write; a no-pending run leaves the directory alone). On a rerun it first removes the images the previous run listed that are not in the new set. It returns the html path, or `None` (after removing its own stale page, listed images and manifest, then the directory if empty) when nothing is pending. `id_map` maps work track ids to output ids; its keys are ints (as `renumber` returns) and string keys are accepted too. `FILL` and `SMOOTH` records are never pending and never shown. For a pending event that has a `vlm` record, `event.vlm["evidence"]` is set to the image path relative to the html file. Without a video (`evidence is None`) cards show signals only.
  - The page (§8.1) is static: inline CSS and JS, no network request, no external URL. Every dynamic text is HTML-escaped. One card per pending event with: the image, kind, reason (`params["reason"]` or `signals["hypothesis"]`), tracks, frames, `algo_score`, the top numeric signals, the VLM answer and reason, `LINK` alternatives (`signals["alternatives"]`), partial-segment scores (`signals["segments"]`), accept / reject radio buttons, a class picker (`<select class="cls">` of `reclass_map`, shown for `RECLASS` events and for screen events whose VLM answer was a rider type), and a copy-to-clipboard `Labeler.draw_track_clips(...)` snippet (output track ids from `id_map`; read the `Labeler.draw_track_clips` docstring and choose `start_frame_offset` / `end_frame_offset` so the clip covers the event's frame span +/- 2 s, with a comment saying how). The page filters by stage, sorts by score, and saves choices in `localStorage` (inside try/catch; convenience only) under a key made from `run_key` (a hash of the run's defining inputs and its config, set by `refine`); each saved choice also records the card's `proposal_key`, and it is restored only when the card's `data-key` equals it, so a regenerated page whose event `switch-r0-000001` is a different proposal never inherits the old choice. **Export decisions** downloads `decisions.json` as `{event_id: "accept" | "reject" | {"accept": true, "new_cls": 3}}` (§4.2) for the cards that have a choice.
  - `check_output_paths` also rejects an input file that lives inside `OUT.review/` (a generated image could overwrite it).
  - `refine` writes the review before the ledger (so `vlm.evidence` is recorded) and sets `RefineResult.review_path`; `dnt-refine run` prints it as `"review"` (null when none).

- [ ] **Step 1: Write the failing tests**

Create `tests/refine/test_review.py`:

```python
import json
import re
import shutil
import subprocess

import pytest

from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.review import _JS_PURE, write_review
from dnt.refine.verify import decide

RECLASS_MAP = {"cyclist": 1, "motorcycle": 3, "scooter": 36}


class Images:
    def build_many(self, events):
        return {e.id: b"\xff\xd8fakejpeg" for e in events}


def pending(kind, stage, idx, score=0.6, tracks=(1,), signals=None, vlm=None, **params):
    ev = Event.propose(stage=stage, kind=kind, tracks=list(tracks),
                       lineage=[[[t, 0, 99]] for t in tracks], frames=(10, 90), params=params,
                       algo_score=score, signals=signals or {})
    ev.id = f"{stage}-r0-{idx:06d}"
    ev.vlm = vlm
    decide(ev, Decision.HUMAN_PENDING, source="auto")
    return ev


IMAGES = Images()


def write(tmp_path, events, evidence=IMAGES, **kw):
    args = dict(review_path=tmp_path / "o.review.html", evidence=evidence,
                id_map={1: 1, 2: 2}, fps=10.0, video_file="/v/cam.mp4",
                track_file=tmp_path / "o.txt", reclass_map=RECLASS_MAP, title="o",
                run_key="run-1")
    args.update(kw)
    return write_review(events, **args)


def test_one_card_per_pending_event_and_none_for_decided_or_position_records(tmp_path):
    a = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal")
    b = pending(EventKind.SPLIT, "switch", 2, cut_frame=40)
    done = pending(EventKind.SPLIT, "switch", 3, cut_frame=70)
    decide(done, Decision.AUTO_ACCEPT, source="auto")
    fill = pending(EventKind.FILL, "fill", 4, gap=[5, 9], n_rows=3)
    html_path = write(tmp_path, [a, b, done, fill])
    html = html_path.read_text()
    assert len(re.findall(r'class="card"', html)) == 2
    assert 'data-id="link-r0-000001"' in html and 'data-id="switch-r0-000002"' in html
    assert "switch-r0-000003" not in html and "fill-r0-000004" not in html
    for ev in (a, b):
        img = tmp_path / "o.review" / f"{ev.id}.jpg"
        assert img.read_bytes().startswith(b"\xff\xd8")
        assert f'src="o.review/{ev.id}.jpg"' in html


def test_the_page_is_static_and_escapes_every_text(tmp_path):
    ev = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal",
                 vlm={"backend": "b<script>", "model": "m", "answer": "unsure", "confidence": 0.4,
                      "votes": {}, "reason": "<script>alert(1)</script>", "evidence": None,
                      "cached": False, "error": "x\"><img src=y onerror=z>"})
    html = write(tmp_path, [ev], title="<b>t</b>").read_text()
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "onerror=z>" not in html
    assert "<b>t</b>" not in html
    assert not re.search(r"https?://", html)
    assert "decisions.json" in html and "localStorage" in html and "Export decisions" in html


def test_the_image_path_is_recorded_on_the_vlm_record(tmp_path):
    vlm = {"backend": "b", "model": "m", "answer": "unsure", "confidence": 0.4, "votes": {},
           "reason": "r", "evidence": None, "cached": False, "error": None}
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40, vlm=vlm)
    other = pending(EventKind.SPLIT, "switch", 2, cut_frame=60)
    write(tmp_path, [ev, other])
    assert ev.vlm["evidence"] == "o.review/switch-r0-000001.jpg"
    assert other.vlm is None  # no VLM record, none is invented


def test_without_a_video_cards_show_signals_only(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, signals={"mot": 0.75, "app": 0.0}, cut_frame=40)
    html = write(tmp_path, [ev], evidence=None).read_text()
    assert "<img" not in html and "no image" in html and "mot" in html
    assert not (tmp_path / "o.review").exists() or not list((tmp_path / "o.review").glob("*.jpg"))


def test_reclass_card_has_a_class_picker_and_a_link_card_shows_alternatives(tmp_path):
    rc = pending(EventKind.RECLASS, "screen", 1, new_cls=None, spans=None,
                 signals={"hypothesis": "rider", "needs_subtype": True})
    lk = pending(EventKind.LINK, "link", 2, tracks=(1, 2), gap=[50, 60], gate="normal",
                 signals={"alternatives": [{"i": 1, "j": 3, "score": 0.55}]})
    seg = pending(EventKind.DROP, "screen", 3, reason="static", spans=[[10, 40]],
                  signals={"segments": [[10, 40, 0.8], [41, 90, 0.1]], "hypothesis": "static"})
    html = write(tmp_path, [rc, lk, seg]).read_text()
    picker = re.findall(r'<select class="cls">(.*?)</select>', html, re.S)
    assert len(picker) == 1 and "cyclist" in picker[0] and 'value="36"' in picker[0]
    assert "0.55" in html and "1 &rarr; 3" in html
    assert "10-40" in html and "0.80" in html and "0.10" in html


def test_the_labeler_snippet_names_the_output_tracks_and_the_video(tmp_path):
    ev = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal")
    html = write(tmp_path, [ev], id_map={1: 7, 2: 9}).read_text()  # renumber() returns int keys
    assert "draw_track_clips(" in html and "/v/cam.mp4" in html
    assert "[7, 9]" in html and "frame_offset" in html
    # the ledger header stores string keys (JSON); a map read back from it works as well
    assert "[7, 9]" in write(tmp_path, [ev], id_map={"1": 7, "2": 9}).read_text()
    # a track that no longer exists in the output has no id; it is left out, never invented
    assert "track_ids=[9]" in write(tmp_path, [ev], id_map={2: 9}).read_text()


def test_cards_and_the_page_carry_the_proposal_key_and_the_run_key(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html = write(tmp_path, [ev], run_key="abc123").read_text()
    assert f'data-key="{ev.proposal_key}"' in html and 'data-run="abc123"' in html
    other = pending(EventKind.SPLIT, "switch", 1, cut_frame=55)  # same id, a different proposal
    assert other.id == ev.id and other.proposal_key != ev.proposal_key
    html2 = write(tmp_path, [other], run_key="abc123").read_text()
    assert f'data-key="{other.proposal_key}"' in html2 and ev.proposal_key not in html2


NODE = shutil.which("node")


def run_node(body: str):
    out = subprocess.run([NODE, "-e", _JS_PURE + "\n" + body], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_saved_choice_is_restored_only_for_the_proposal_it_was_made_on():
    body = """
    var saved = {"switch-r0-000001": {choice: "accept", cls: "3", key: "AAA"}};
    console.log(JSON.stringify([
      restorable(saved, "switch-r0-000001", "AAA"),
      restorable(saved, "switch-r0-000001", "BBB"),
      restorable(saved, "switch-r0-000002", "AAA"),
      restorable({x: {choice: "maybe", key: "K"}}, "x", "K"),
      restorable({x: {choice: "reject"}}, "x", "K"),
      restorable(null, "x", "K")
    ]));"""
    got = run_node(body)
    assert got[0] == {"choice": "accept", "cls": "3", "key": "AAA"}
    assert got[1:] == [None] * 5  # a different proposal, another id, a bad choice, no key, no store


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_storage_key_is_namespaced_by_the_run_and_decisions_export_in_the_spec_format():
    body = """
    console.log(JSON.stringify({
      a: storageKey("r1"), b: storageKey("r2"),
      out: exportDecisions([
        {id: "e1", choice: "accept", cls: ""},
        {id: "e2", choice: "accept", cls: "36"},
        {id: "e3", choice: "reject", cls: "3"},
        {id: "e4", choice: null, cls: ""}
      ])
    }));"""
    got = run_node(body)
    assert got["a"] != got["b"] and got["a"].endswith("r1")
    assert got["out"] == {"e1": "accept", "e2": {"accept": True, "new_cls": 36}, "e3": "reject"}


def test_cards_are_filterable_and_sortable(tmp_path):
    a = pending(EventKind.LINK, "link", 1, score=0.5, tracks=(1, 2), gap=[5, 9], gate="normal")
    b = pending(EventKind.SPLIT, "switch", 2, score=0.7, cut_frame=40)
    html = write(tmp_path, [a, b]).read_text()
    assert 'data-stage="link"' in html and 'data-stage="switch"' in html
    assert 'data-score="0.500"' in html and 'data-score="0.700"' in html
    assert 'id="stage-filter"' in html and 'id="sort"' in html


def test_a_failing_evidence_builder_still_writes_a_signals_only_page(tmp_path):
    class Broken:
        def build_many(self, events):
            raise OSError("video went away")

    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html = write(tmp_path, [ev], evidence=Broken()).read_text()
    assert "no image" in html and f'data-id="{ev.id}"' in html


def test_nothing_pending_returns_none_and_removes_only_its_own_files(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html_path = write(tmp_path, [ev])
    d = tmp_path / "o.review"
    assert html_path.exists() and (d / f"{ev.id}.jpg").exists() and (d / ".dnt-review.json").exists()
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": [f"{ev.id}.jpg"]}
    (d / "keep.jpg").write_bytes(b"my photo")
    (d / "keep.txt").write_text("my note")
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert not html_path.exists() and not (d / f"{ev.id}.jpg").exists()
    assert not (d / ".dnt-review.json").exists()
    assert (d / "keep.jpg").read_bytes() == b"my photo" and (d / "keep.txt").exists()
    (d / "keep.jpg").unlink()
    (d / "keep.txt").unlink()
    ev2 = pending(EventKind.SPLIT, "switch", 2, cut_frame=40)
    write(tmp_path, [ev2])
    decide(ev2, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev2]) is None
    assert not d.exists()  # nothing of the user's lives there: the directory goes too


def test_without_a_manifest_no_image_is_deleted(tmp_path):
    d = tmp_path / "o.review"
    d.mkdir()
    (d / "stranger.jpg").write_bytes(b"x")
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    (tmp_path / "o.review.html").write_text("old page")
    assert write(tmp_path, [ev]) is None
    assert (d / "stranger.jpg").exists() and not (tmp_path / "o.review.html").exists()


def test_a_rerun_replaces_the_previous_images_it_owns_and_nothing_else(tmp_path):
    first = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [first])
    d = tmp_path / "o.review"
    (d / "keep.jpg").write_bytes(b"mine")
    second = pending(EventKind.SPLIT, "switch", 2, cut_frame=40)
    write(tmp_path, [second])
    assert not (d / f"{first.id}.jpg").exists() and (d / f"{second.id}.jpg").exists()
    assert (d / "keep.jpg").read_bytes() == b"mine"
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": [f"{second.id}.jpg"]}


def test_a_foreign_file_with_an_image_name_is_never_overwritten_or_deleted(tmp_path, caplog):
    d = tmp_path / "o.review"
    d.mkdir()
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    mine = d / f"{ev.id}.jpg"
    mine.write_bytes(b"the user's own file")  # same name as a generated image, no manifest
    with caplog.at_level("WARNING"):
        html = write(tmp_path, [ev]).read_text()
    assert mine.read_bytes() == b"the user's own file" and "no image" in html
    assert "not overwriting" in caplog.text and not (d / ".dnt-review.json").exists()
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None  # cleanup deletes nothing it never wrote
    assert mine.read_bytes() == b"the user's own file"
    # a manifest that lists other names does not make this file ours either
    (d / ".dnt-review.json").write_text(json.dumps({"images": ["other.jpg"]}))
    ev2 = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [ev2])
    assert mine.read_bytes() == b"the user's own file"
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": []}


def test_a_users_manifest_json_is_left_alone(tmp_path):
    d = tmp_path / "o.review"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"images": ["x.jpg"]}))
    (d / "x.jpg").write_bytes(b"x")
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [ev])
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert (d / "x.jpg").exists() and json.loads((d / "manifest.json").read_text())["images"] == [
        "x.jpg"
    ]


BAD_MANIFESTS = [
    "null", "1", '"text"', "[]", '{"images": null}', '{"images": 1}', '{"images": "a.jpg"}',
    '{"images": {"a.jpg": 1}}', '{"images": [1, null, "../x.jpg", "a.txt"]}', "{not json", "",
]


@pytest.mark.parametrize("text", BAD_MANIFESTS)
def test_a_damaged_manifest_never_breaks_the_review(tmp_path, text):
    d = tmp_path / "o.review"
    d.mkdir()
    (d / ".dnt-review.json").write_text(text)
    (d / "keep.jpg").write_bytes(b"mine")
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    assert write(tmp_path, [ev]) is not None  # a pending run replaces the damaged manifest
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": [f"{ev.id}.jpg"]}
    assert (d / "keep.jpg").read_bytes() == b"mine"
    (d / ".dnt-review.json").write_text(text)  # damaged again, then nothing is pending
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert (d / "keep.jpg").read_bytes() == b"mine"
    assert not (tmp_path / "o.review.html").exists()


def test_a_manifest_cannot_point_outside_the_review_directory(tmp_path):
    victim = tmp_path / "precious.jpg"
    victim.write_bytes(b"precious")
    d = tmp_path / "o.review"
    d.mkdir()
    (d / ".dnt-review.json").write_text(json.dumps({"images": ["../precious.jpg", "/etc/passwd"]}))
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    write(tmp_path, [ev])
    assert victim.read_bytes() == b"precious"


def test_an_input_inside_the_review_directory_is_rejected(tmp_path):
    from dnt.refine.refiner import check_output_paths

    d = tmp_path / "o.review"
    d.mkdir()
    for name in (".dnt-review.json", "e1.jpg", "my-photo.jpg"):
        with pytest.raises(ValueError, match="review"):
            check_output_paths(tmp_path / "o.txt", {"video_file": d / name})
    check_output_paths(tmp_path / "o.txt", {"video_file": tmp_path / "o.review.mp4"})  # a sibling


def test_the_output_directory_is_created_when_missing(tmp_path):
    # refine writes the review before the ledger, which is what creates the directory
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    page = tmp_path / "new" / "dir" / "o.review.html"
    assert write(tmp_path, [ev], review_path=page) == page
    assert page.is_file() and (page.parent / "o.review" / f"{ev.id}.jpg").is_file()
    nothing = tmp_path / "other" / "o.review.html"
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev], review_path=nothing) is None  # no directory is made for nothing
    assert not nothing.parent.exists()


def test_a_segment_without_a_score_does_not_crash_the_page(tmp_path):
    # a screen event keeps None for a segment where the hypothesis has no score (e.g. duplicate)
    ev = pending(EventKind.DROP, "screen", 1, reason="duplicate", spans=[[10, 40]], of=2,
                 signals={"segments": [[10, 40, 0.8], [41, 90, None]], "hypothesis": "duplicate"})
    html = write(tmp_path, [ev]).read_text()
    assert "10-40 0.80" in html and "41-90 -" in html
```

Append to `tests/refine/test_vlm_refine.py`:

```python
def test_refine_writes_the_review_for_pending_events_and_records_the_image(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    assert res.review_path == tmp_path / "o.review.html" and res.review_path.is_file()
    (ev,) = splits(res)
    html = res.review_path.read_text()
    assert f'data-id="{ev.id}"' in html and (tmp_path / "o.review" / f"{ev.id}.jpg").is_file()
    led = Ledger.read(res.ledger_path)
    (back,) = [e for e in led.events if e.kind is EventKind.SPLIT]
    assert back.vlm["evidence"] == f"o.review/{ev.id}.jpg"


def test_the_review_snippet_names_the_renumbered_output_tracks(tmp_path):
    # raw id 7 becomes output id 1; the snippet must point at the id the output file holds
    src, video = takeover_scene(tmp_path, first_id=7)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    html = res.review_path.read_text()
    assert "track_ids=[1]" in html and "track_ids=[]" not in html and "track_ids=[7]" not in html
    assert sorted(res.tracks.track.unique()) == [1]


def test_regenerating_a_review_for_other_inputs_changes_the_run_and_card_keys(tmp_path):
    def keys(html):
        return html.split('data-run="')[1].split('"')[0], html.split('data-key="')[1].split('"')[0]

    src, video = takeover_scene(tmp_path)
    run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src, video)
    first = (tmp_path / "o.review.html").read_text()
    # the same output name and the same event id (switch-r0-000001), but another track file
    src2, video2 = takeover_scene(tmp_path, name="u.txt", video="w.mp4", first_id=5)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("unsure")}), src2, video2)
    second = res.review_path.read_text()
    assert 'data-id="switch-r0-000001"' in first and 'data-id="switch-r0-000001"' in second
    assert keys(first)[0] != keys(second)[0]  # the run key (what localStorage is namespaced by)
    assert keys(first)[1] != keys(second)[1]  # the proposal key (what each saved choice is checked by)


def test_the_review_lists_exactly_the_pending_events(tmp_path):
    src, video = takeover_scene(tmp_path)
    res = run(tmp_path, cfg_for(tmp_path), FakeBackend({"SPLIT": reply("different")}), src, video)
    pend = [e for e in res.events if e.decision is Decision.HUMAN_PENDING
            and e.kind not in (EventKind.FILL, EventKind.SMOOTH)]
    if pend:
        html = res.review_path.read_text()
        assert html.count('class="card"') == len(pend)
        assert all(f'data-id="{e.id}"' in html for e in pend)
    else:
        assert res.review_path is None and not (tmp_path / "o.review.html").exists()


def test_a_run_without_a_video_still_writes_a_signals_only_review(tmp_path):
    src, _ = takeover_scene(tmp_path)
    refiner = TrackRefiner(cfg_for(tmp_path))
    refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    html = refiner.last_result.review_path.read_text()
    assert "<img" not in html and "no image" in html
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/refine/test_review.py -q`
Expected: collection error `No module named 'dnt.refine.review'`.

- [ ] **Step 3: Implement**

Create `src/dnt/refine/review.py` (static page; write it in this shape, filling the helpers):

```python
"""Static review page for HUMAN_PENDING events (spec 8.1)."""

from __future__ import annotations

import contextlib
import html
import json
import logging
import os
from pathlib import Path

from .events import Decision, Event, EventKind

log = logging.getLogger(__name__)

_CSS = """
body{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ccc;border-radius:6px;margin:12px 0;padding:10px}
.card img{max-width:100%;border:1px solid #ddd}
.meta span{margin-right:12px}.sig{color:#555}pre{background:#f0f0f0;padding:6px;overflow:auto}
.bar{position:sticky;top:0;background:#fafafa;padding:6px 0}
"""

_JS_PURE = """
function storageKey(run) { return "dnt-refine-review:" + run; }
function restorable(saved, id, proposalKey) {
  var s = saved && saved[id];
  if (!s || s.key !== proposalKey) { return null; }
  return (s.choice === "accept" || s.choice === "reject") ? s : null;
}
function exportDecisions(rows) {
  var out = {};
  rows.forEach(function (r) {
    if (r.choice === "accept") {
      out[r.id] = r.cls ? {accept: true, new_cls: parseInt(r.cls, 10)} : "accept";
    } else if (r.choice === "reject") { out[r.id] = "reject"; }
  });
  return out;
}
"""

_JS_DOM = """
(function () {
  var KEY = storageKey(document.body.dataset.run);
  var cards = [].slice.call(document.querySelectorAll(".card"));
  var saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  function choice(c) {
    var r = c.querySelector("input[type=radio]:checked");
    var sel = c.querySelector("select.cls");
    if (!r) { return null; }
    return {id: c.dataset.id, key: c.dataset.key, choice: r.value, cls: sel ? sel.value : ""};
  }
  cards.forEach(function (c) {
    var s = restorable(saved, c.dataset.id, c.dataset.key);
    if (!s) { return; }
    var r = c.querySelector('input[value="' + s.choice + '"]');
    if (r) { r.checked = true; }
    var sel = c.querySelector("select.cls");
    if (sel && s.cls) { sel.value = s.cls; }
  });
  document.addEventListener("change", function () {
    var s = {};
    cards.forEach(function (c) { var v = choice(c); if (v) { s[c.dataset.id] = v; } });
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
  });
  document.getElementById("export").onclick = function () {
    var rows = cards.map(choice).filter(Boolean);
    var blob = new Blob([JSON.stringify(exportDecisions(rows), null, 2)],
                        {type: "application/json"});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
  };
  function refresh() {
    var stage = document.getElementById("stage-filter").value;
    var key = document.getElementById("sort").value;
    var box = document.getElementById("cards");
    cards.sort(function (a, b) {
      return key === "score-asc" ? a.dataset.score - b.dataset.score
           : key === "score-desc" ? b.dataset.score - a.dataset.score : 0;
    }).forEach(function (c) {
      box.appendChild(c);
      c.style.display = (!stage || c.dataset.stage === stage) ? "" : "none";
    });
  }
  document.getElementById("stage-filter").onchange = refresh;
  document.getElementById("sort").onchange = refresh;
  [].forEach.call(document.querySelectorAll("button.copy"), function (b) {
    b.onclick = function () { navigator.clipboard.writeText(b.nextElementSibling.textContent); };
  });
})();
"""
_JS = _JS_PURE + _JS_DOM
```

```python
def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _top_signals(signals: dict, n: int = 6) -> list[str]:
    nums = [
        (k, v)
        for k, v in signals.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and v == v
    ]
    nums.sort(key=lambda kv: -abs(kv[1]))
    return [f"{k}={v:.3f}" for k, v in nums[:n]]


def _reason(ev: Event) -> str:
    return str(ev.params.get("reason") or ev.signals.get("hypothesis") or ev.kind)


def _snippet(ev: Event, id_map: dict, fps: float, video_file, track_file) -> str:
    ids = [i for i in (id_map.get(t, id_map.get(str(t))) for t in ev.tracks) if i is not None]
    margin = round(2.0 * fps)
    # the clip spans the selected tracks' first to last frame, plus a 2 s margin on each side
    return (
        "labeler.draw_track_clips(\n"
        f"    input_video={str(video_file)!r}, output_path='clips/{ev.id}',\n"
        f"    track_file={str(track_file)!r}, method='specify', track_ids={ids},\n"
        f"    start_frame_offset={margin}, end_frame_offset={margin},\n"
        ")"
    )


def _extras(ev: Event) -> str:
    out = []
    for alt in ev.signals.get("alternatives") or []:
        out.append(f"<div>next best: {_e(alt['i'])} &rarr; {_e(alt['j'])} {alt['score']:.2f}</div>")
    segs = ev.signals.get("segments")
    if segs:
        parts = ", ".join(
            f"{int(a)}-{int(b)} " + ("-" if c is None else f"{c:.2f}") for a, b, c in segs
        )
        out.append(f"<div>segments (score): {_e(parts)}</div>")
    return "".join(out)


def _vlm_block(ev: Event) -> str:
    v = ev.vlm
    if not v:
        return ""
    ans = "-" if v.get("answer") is None else _e(v["answer"])
    err = f" error: {_e(v['error'])}" if v.get("error") else ""
    return (
        f'<div class="vlm">VLM {_e(v.get("backend"))}/{_e(v.get("model"))}: {ans} '
        f"({float(v.get('confidence') or 0):.2f}) {_e(v.get('reason', ''))}{err}</div>"
    )


def _picker(ev: Event, reclass_map: dict) -> str:
    rider = (ev.vlm or {}).get("answer") in ("cyclist", "motorcycle_rider", "scooter_rider")
    if ev.kind is not EventKind.RECLASS and not (ev.stage == "screen" and rider):
        return ""
    opts = '<option value="">(keep)</option>' + "".join(
        f'<option value="{_e(c)}">{_e(name)} ({_e(c)})</option>' for name, c in reclass_map.items()
    )
    return f'<label>class <select class="cls">{opts}</select></label>'


def _card(ev: Event, img_rel: str | None, snippet: str, reclass_map: dict) -> str:
    image = f'<img src="{_e(img_rel)}" alt="evidence">' if img_rel else "<div>no image</div>"
    name = _e(ev.id)
    return (
        f'<div class="card" data-id="{name}" data-key="{_e(ev.proposal_key)}" '
        f'data-stage="{_e(ev.stage)}" data-score="{ev.algo_score:.3f}">'
        f'{image}<div class="meta"><span><b>{_e(ev.kind)}</b> {_e(_reason(ev))}</span>'
        f"<span>tracks {_e(ev.tracks)}</span><span>frames {_e(ev.frames[0])}-{_e(ev.frames[1])}"
        f"</span><span>score {ev.algo_score:.3f}</span></div>"
        f'<div class="sig">{_e(" ".join(_top_signals(ev.signals)))}</div>'
        f"{_vlm_block(ev)}{_extras(ev)}"
        f'<div><label><input type="radio" name="{name}" value="accept"> accept</label> '
        f'<label><input type="radio" name="{name}" value="reject"> reject</label> '
        f"{_picker(ev, reclass_map)}</div>"
        f'<button class="copy" type="button">copy clip snippet</button><pre>{_e(snippet)}</pre>'
        "</div>"
    )


_MANIFEST = ".dnt-review.json"  # the report's own name: a user's manifest.json is never touched


def _listed_images(img_dir: Path) -> list[str]:
    """Return the image names the previous run wrote here (plain ``*.jpg`` names), or ``[]``.

    Anything that is not a JSON object with a list of names proves no ownership.
    """
    try:
        names = json.loads((img_dir / _MANIFEST).read_text())["images"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if not isinstance(names, list):
        return []
    return [n for n in names if isinstance(n, str) and n == Path(n).name and n.endswith(".jpg")]


def _remove_listed(img_dir: Path, keep: set[str] = frozenset()) -> None:
    """Delete the images the manifest lists (except ``keep``); never anything else."""
    for name in _listed_images(img_dir):
        if name not in keep:
            (img_dir / name).unlink(missing_ok=True)


def _remove_own_files(review_path: Path, img_dir: Path) -> None:
    review_path.unlink(missing_ok=True)
    if img_dir.is_dir():
        owned = _listed_images(img_dir)
        _remove_listed(img_dir)
        if owned:  # a manifest that proves nothing is left where it is
            (img_dir / _MANIFEST).unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            img_dir.rmdir()  # only if nothing else lives there


def write_review(
    events: list[Event],
    *,
    review_path,
    evidence,
    id_map: dict,
    fps: float,
    video_file,
    track_file,
    reclass_map: dict,
    title: str,
    run_key: str,
) -> Path | None:
    """Write ``OUT.review.html`` and ``OUT.review/*.jpg`` for the pending events (spec 8.1).

    Returns the page path, or ``None`` (after removing this output's stale page and images)
    when no event is pending.
    """
    review_path = Path(review_path)
    img_dir = review_path.parent / review_path.name.removesuffix(".html")
    pend = [
        e
        for e in events
        if e.decision is Decision.HUMAN_PENDING and e.kind not in (EventKind.FILL, EventKind.SMOOTH)
    ]
    if not pend:
        _remove_own_files(review_path, img_dir)
        return None
    images: dict = {}
    if evidence is not None:
        try:
            images = evidence.build_many(pend)
        except Exception as err:  # a damaged video must not stop the review being written
            log.warning("could not build the review images: %s", err)
    cards, stages = [], sorted({e.stage for e in pend})
    written: list[str] = []
    owned = set(_listed_images(img_dir))
    for ev in pend:
        rel = None
        data = images.get(ev.id)
        name = f"{ev.id}.jpg"
        if data is not None and name not in owned and os.path.lexists(img_dir / name):
            log.warning("not overwriting %s: it is not an image of this report", img_dir / name)
            data = None  # the card is shown without an image; the foreign file stays intact
        if data is not None:
            img_dir.mkdir(parents=True, exist_ok=True)
            (img_dir / name).write_bytes(data)
            written.append(name)
            rel = f"{img_dir.name}/{ev.id}.jpg"
            if ev.vlm:
                ev.vlm["evidence"] = rel
        snippet = _snippet(ev, id_map, fps, video_file, track_file)
        cards.append(_card(ev, rel, snippet, reclass_map))
    if written or owned:  # never start a manifest in a directory that holds none of our images
        _remove_listed(img_dir, keep=set(written))  # images of events that are gone
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / _MANIFEST).write_text(json.dumps({"images": written}))
    stage_opts = '<option value="">all stages</option>' + "".join(
        f'<option value="{_e(s)}">{_e(s)}</option>' for s in stages
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head>"
        f'<body data-run="{_e(run_key)}">'
        f"<h2>{_e(title)}: {len(pend)} event(s) to review</h2>"
        '<div class="bar"><select id="stage-filter">' + stage_opts + "</select> "
        '<select id="sort"><option value="score-desc">score, high first</option>'
        '<option value="score-asc">score, low first</option><option value="order">order</option>'
        '</select> <button id="export" type="button">Export decisions</button></div>'
        '<div id="cards">' + "".join(cards) + f"</div><script>{_JS}</script></body></html>"
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)  # refine writes this before the ledger
    review_path.write_text(page, encoding="utf-8")
    return review_path
```

In `refiner.py`, after `work, id_map = renumber(work)` and before the summary is built, add

```python
        review_path = write_review(
            events,
            review_path=paths["review"],
            evidence=stages.evidence,
            id_map=id_map,
            fps=fps_val,
            video_file=None if video_file is None else str(Path(video_file).resolve()),
            track_file=out,
            reclass_map=cfg.reclass_map,
            title=out.stem,
            run_key=run_key,
        )
```

(import `from .review import write_review` between the `.primitives` and `.screen` imports; `import hashlib` and `import json` at the top of the module if they are not there yet). Compute `run_key` once, right after `inputs` is built:

```python
        run_key = hashlib.sha256(
            json.dumps(
                {
                    "tracks": track_sha,
                    "context": context_sha,
                    "video": (inputs["video"] or {}).get("fingerprint"),
                    "hints": ((inputs["hints"] or {}).get("reclass") or {}).get("sha256"),
                    "config": cfg.to_dict(),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
```

and pass `review_path=review_path` to `RefineResult(...)` in place of `None`. In `cli.py` replace the printed dict `{"out": args.out, "ledger": str(res.ledger_path), "summary": res.summary}` with

```python
            {
                "out": args.out,
                "ledger": str(res.ledger_path),
                "review": None if res.review_path is None else str(res.review_path),
                "summary": res.summary,
            },
```

In `tests/refine/test_cli.py::test_success_prints_json_with_out_ledger_and_summary` change `assert set(doc) == {"out", "ledger", "summary"}` to `assert set(doc) == {"out", "ledger", "review", "summary"}` and add after the next assertion `assert doc["review"] is None  # nothing is pending in this scene, so there is no review page`.

In `check_output_paths` (`refiner.py`), after the existing loop over `written`, add the image directory (a generated image must not overwrite a recorded input):

```python
    review_dir = output_paths(out)["review"].with_suffix("")  # OUT.review
    for in_name, in_path in inputs.items():
        if in_path is not None and review_dir.resolve() in Path(in_path).resolve().parents:
            raise ValueError(
                f"{in_name} ({in_path}) is inside the review image directory ({review_dir}); "
                "refine would overwrite or delete it. Move the input or write the output elsewhere."
            )
```

The review is written before the ledger, and the ledger write is what creates a missing output directory (`dnt-refine run --out new_dir/o.txt`), so `write_review` creates `review_path.parent` itself before it writes the page (the `mkdir` line above the `write_text`).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine/test_review.py tests/refine/test_vlm_refine.py -q -W error` and then `.venv/bin/python -m pytest tests/refine tests/test_refine_independence.py -q`.
Expected: all pass. No existing test pins `review_path is None` or the absence of `o.review.html`; the one existing test that pins the CLI JSON keys is updated above (without that edit `test_success_prints_json_with_out_ledger_and_summary` fails, and without the `mkdir` in `write_review` `test_output_invariants_and_determinism` fails with `FileNotFoundError`).

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src/dnt/refine
git add src/dnt/refine/review.py src/dnt/refine/refiner.py src/dnt/refine/cli.py tests/refine/test_review.py tests/refine/test_vlm_refine.py tests/refine/test_cli.py
git commit -m "feat(refine): write a static review page for pending events" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Config validation, extras, docs, spec follow-ups, and the final gate

**Files:**
- Modify: `src/dnt/refine/config.py` (validate the `vlm` block), `pyproject.toml` (extras), `docs/api/refine/index.md`, `docs/api/refine/appearance.md` or a new `docs/api/refine/verification.md` (+ `mkdocs.yml` nav), `docs/changelog.md` and `CHANGELOG.md` (byte-identical), `docs/superpowers/specs/2026-09-27-track-refinement-design.md`, `tests/refine/test_encoders.py` (the extras test), `tests/refine/test_cli.py` (the limitation pins)
- Test: `tests/refine/test_config.py` (additions), `tests/refine/test_minimal_install_vlm.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `RefineConfig.validate()` rejects: `vlm.votes` not an int >= 1; `vlm.min_conf` outside [0, 1]; `vlm.max_calls` not an int >= 0; `vlm.max_concurrency` not an int >= 1; `vlm.timeout_s` not > 0; `vlm.vote_temperature` < 0; `vlm.cache_dir` not a non-empty string; bool is never accepted for a number. The existing rules stay (a non-`none` backend other than `anthropic` needs `vlm.model`).
  - Extras: `refine-vlm = ["openai>=1.40", "anthropic>=0.40"]`; `refine = ["transformers>=4.40", "torchreid", "tensorboard", "openai>=1.40", "anthropic>=0.40"]`; neither package is a required dependency.
  - Docs: a "Verification with a VLM" section (config example for a local OpenAI-compatible server and for Anthropic; the answer-to-decision table; budget, cache, votes; what the review page is and that applying `decisions.json` arrives with Plan 4), the changelog, and the spec follow-ups below.
  - Spec follow-ups (the spec stays authoritative and must match the code): §7.3 `ask(..., *, tag: str = "")` and `VLMTransientError`; the pinned default Anthropic model `claude-sonnet-5-5`; §7.2/§4.3 orphan events are never sent to a VLM; §7.1 context frames for a `LINK` also include "A ends" and "B starts", and the hidden-path frame has no occluder label; evidence is built in chunks of 64 events; §7.4 `max_calls` is a hard limit on backend invocations (a retry after a transient failure or an invalid reply draws one unit from the allowance left after admission, and when none is left the event stays pending with `vlm.error: "budget"`; `calls` + `retries` never exceed it, and both are reported), whole questions are admitted in priority order (a question whose uncached votes do not all fit is skipped with `budget`, a later one that fits still runs), cached answers are validated like fresh replies, the sync public API keeps one event loop alive for the runner's life and closes the client on it, and `budget_skipped` is reported; §8.3 `vlm` summary has `retries` and `budget_skipped`; §10 the no-video row also says a signals-only review page is written; §8.1 the review page deletes only the images listed in its own `.dnt-review.json`, and saved choices are namespaced by a run key and checked against each event's `proposal_key`.
  - The limitation note in `docs/api/refine/index.md` and the changelog must stay true: **without a VLM backend** (the default) the capped edits are recorded as `HUMAN_PENDING` and not applied; **with a backend and a video** the VLM decides them when it is sure. Rewrite the sentences and the pins in `test_docs_and_changelog_state_the_limitation_accurately` together (keep pinning "ID-switch splits found from motion alone", "links across occlusions", "ambiguous assignment margin", "static objects or of mixed tracks", `HUMAN_PENDING`, and add the phrase "Without a VLM backend" to both notes' pins), and replace "VLM verification ... follow in later releases" by "applying review decisions follows in a later release".

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/refine/test_config.py` (`pytest` and `RefineConfig` are already imported at the top of that file; a second import block mid-file fails ruff with E402 and F811):

```python
@pytest.mark.parametrize(
    "key,value",
    [
        ("votes", 0), ("votes", 1.5), ("votes", True),
        ("min_conf", -0.1), ("min_conf", 1.1), ("min_conf", True),
        ("max_calls", -1), ("max_calls", 2.5),
        ("max_concurrency", 0), ("max_concurrency", False),
        ("timeout_s", 0), ("timeout_s", -3),
        ("vote_temperature", -0.5),
        ("cache_dir", ""), ("cache_dir", 7),
    ],
)
def test_vlm_settings_are_validated(key, value):
    cfg = RefineConfig.defaults()
    setattr(cfg.vlm, key, value)
    with pytest.raises(ValueError, match=f"vlm.{key}"):
        cfg.validate()


def test_good_vlm_settings_validate_and_round_trip(tmp_path):
    cfg = RefineConfig.defaults()
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "qwen"
    cfg.vlm.votes, cfg.vlm.min_conf, cfg.vlm.max_calls = 3, 0.0, 0
    cfg.validate()
    cfg.to_yaml(tmp_path / "c.yaml")
    assert RefineConfig.from_yaml(tmp_path / "c.yaml").vlm.votes == 3
```

In `tests/refine/test_encoders.py` update `test_extras_are_declared_and_not_required` to the new extras (the exact lists above, and assert neither `openai` nor `anthropic` is a required dependency).

Create `tests/refine/test_minimal_install_vlm.py`:

```python
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import logging, sys
    for name in ("transformers", "torchreid", "openai", "anthropic"):
        sys.modules[name] = None  # a minimal install
    from pathlib import Path

    from dnt.refine import RefineConfig, TrackRefiner

    # tests/ has no __init__.py and the same package name pytest uses here is `refine`; importing
    # `tests.refine` instead fails when another distribution installs a top-level `tests`
    sys.path.insert(0, sys.argv[2])
    from refine._video import takeover_scene

    out = Path(sys.argv[1])
    src, vid = takeover_scene(out)
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"
    cfg.link.enabled = False
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "m"
    refiner = TrackRefiner(cfg)
    try:
        refiner.refine(src, out / "a.txt", video_file=vid, verbose=False)
    except ImportError as err:
        assert "refine-vlm" in str(err), err
    else:
        raise SystemExit("expected an ImportError for a video with a VLM backend")
    assert not (out / "a.txt").exists() and not (out / "a.ledger.jsonl").exists()
    refiner.refine(src, out / "b.txt", fps=10, verbose=False)  # no video: the backend is ignored
    assert (out / "b.txt").is_file() and refiner.last_result.summary["vlm"]["calls"] == 0
    cfg.vlm.backend, cfg.vlm.model = "none", None
    refiner.refine(src, out / "c.txt", video_file=vid, verbose=False)
    assert (out / "c.txt").is_file()
    print("OK")
    """
)


def test_a_minimal_install_runs_without_a_vlm_and_fails_early_with_one_requested(tmp_path):
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), tests_dir], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/refine/test_config.py tests/refine/test_encoders.py tests/refine/test_minimal_install_vlm.py -q`
Expected: the 15 config validation cases (`DID NOT RAISE ValueError`) and the extras test fail; the minimal-install test already passes if Task 7 is correct (it pins behavior). The script imports the helper as `refine._video` with `tests/` on `sys.path`, the package name pytest itself uses for `tests/refine` (it has an `__init__.py`; `tests/` has none). Do not write `from tests.refine._video import ...`: it fails with `ModuleNotFoundError` whenever another distribution installed a top-level `tests` package into the environment, because a regular package beats the repository's `tests/` namespace directory.

- [ ] **Step 3: Implement**

1. `src/dnt/refine/config.py`, inside `validate()` next to the existing `vlm` checks, add (matching the file's existing helpers and style for number checks; reject `bool` explicitly because `bool` is an `int`):

```python
        v = self.vlm

        def _num(x):
            return isinstance(x, int | float) and not isinstance(x, bool)

        if not (isinstance(v.votes, int) and not isinstance(v.votes, bool) and v.votes >= 1):
            p.append("vlm.votes must be an integer >= 1")
        if not (_num(v.min_conf) and 0.0 <= v.min_conf <= 1.0):
            p.append("vlm.min_conf must be a number in [0, 1]")
        if not (
            isinstance(v.max_calls, int) and not isinstance(v.max_calls, bool) and v.max_calls >= 0
        ):
            p.append("vlm.max_calls must be an integer >= 0")
        if not (
            isinstance(v.max_concurrency, int)
            and not isinstance(v.max_concurrency, bool)
            and v.max_concurrency >= 1
        ):
            p.append("vlm.max_concurrency must be an integer >= 1")
        if not (_num(v.timeout_s) and v.timeout_s > 0):
            p.append("vlm.timeout_s must be a number > 0")
        if not (_num(v.vote_temperature) and v.vote_temperature >= 0):
            p.append("vlm.vote_temperature must be a number >= 0")
        if not (isinstance(v.cache_dir, str) and v.cache_dir.strip()):
            p.append("vlm.cache_dir must be a non-empty string")
```

2. `pyproject.toml`: add `refine-vlm = ["openai>=1.40", "anthropic>=0.40"]` and extend `refine` as above (nothing else).
3. Docs, changelog (`Unreleased`: a "New" bullet for the VLM verification placed before the "This release scores with motion only" bullet, a "Changed" bullet that the capped edits are decided by a VLM when a backend and a video are given, and the extras line), the mirror `cp docs/changelog.md CHANGELOG.md`, and the spec follow-ups listed above.

   The limitation note (`docs/api/refine/index.md`, `!!! note "Current limitations"`, and the changelog bullet that starts "This release scores with motion only") must be rewritten together with the pins in `test_docs_and_changelog_state_the_limitation_accurately`. In both texts, replace the sentences from "Other edits are ..." to the end of the note by (the index note indents each line by four spaces; the pins normalise whitespace, so line breaks are free):

   > Without a VLM backend (the default), other edits are proposed but never applied yet [changelog: "are capped below auto-accept, recorded as `HUMAN_PENDING`, and not applied yet"], because their scores are capped below auto-accept: ID-switch splits found from motion alone, links across occlusions, links with an ambiguous assignment margin, and false-track drops of static objects or of mixed tracks. Rider reclasses whose subtype no ReClass hint settles are pending too, however high they score, unless a VLM backend names the subtype. They appear in the ledger as `HUMAN_PENDING` and leave the tracks unchanged. With a VLM backend and a video (see Verification with a VLM below), the VLM decides these edits when it is sure; the rest stay `HUMAN_PENDING` and go on a review page. In-vehicle drops need a context file with the vehicles' boxes (`context_file=`, or `--context`); without one the in-vehicle cue is skipped. Applying review decisions follows in a later release.
   > (In the changelog variant, drop the clause ", because their scores are capped below auto-accept": the bracketed text already says it.)

   (The old closing sentence "VLM verification and applying review decisions follow in later releases", and the old reason "because only a hint can choose the subtype in this release", are no longer true and must go. The existing pins `"never applied yet"` / `"not applied yet"` keep working.) In the test, replace `assert "static objects" in note and "mixed tracks" in note, where` by

```python
        assert "static objects or of mixed tracks" in note, where
        assert "links across occlusions" in note, where
```

   and add after the `"ambiguous assignment margin"` pin

```python
        assert "Without a VLM backend" in note, where
        assert "the VLM decides these edits when it is sure" in note, where
        assert "applying review decisions follows in a later release" in note.lower(), where
        assert "follow in later releases" not in note, where
```

   Add the section "Verification with a VLM" to `docs/api/refine/index.md` before the final `::: dnt.refine` line (the contents are listed under Docs above).
4. mkdocs: add `docs/api/refine/verification.md` containing `# Verification` and the blocks `::: dnt.refine.vlm`, `::: dnt.refine.verify`, `::: dnt.refine.evidence`, `::: dnt.refine.review` (blank line between blocks), and a nav entry `- Verification: api/refine/verification.md` after Appearance in `mkdocs.yml`. In `tests/refine/test_cli.py::test_docs_pages_exist_and_are_in_the_nav` add `"verification"` to the page tuple and, after that loop, check the four blocks:

```python
    verification = (ROOT / "docs/api/refine/verification.md").read_text()
    for module in ("vlm", "verify", "evidence", "review"):
        assert f"::: dnt.refine.{module}\n" in verification
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/refine tests/test_refine_independence.py -q`
Expected: all pass.

- [ ] **Step 5: Gate**

```bash
.venv/bin/python -m pytest -q                      # whole default suite (about 15 minutes: run it in the background, a 600000 ms foreground timeout is too short)
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src/dnt/refine
git status --short | grep -v '^??'                 # only the files named in this task are modified
```

Build the docs into a temporary directory outside the repository (`site/` must not change; the `with-pdf` plugin already fails the strict build on `main`, so the check uses a temporary config without it, with `docs_dir`, `site_dir`, mkdocstrings `paths` and `watch` rewritten to absolute paths, exactly as in Plan 2 Task 8):

```bash
T=$(mktemp -d) && .venv/bin/python - "$T" <<'EOF'
import pathlib
import re
import sys

t, root = pathlib.Path(sys.argv[1]), pathlib.Path.cwd()
s = pathlib.Path("mkdocs.yml").read_text()
s = re.sub(r"  - with-pdf:\n(?:      .*\n)+", "", s)
s = s.replace("docs_dir: docs", f"docs_dir: {root}/docs")
s = s.replace("site_dir: site", f"site_dir: {t}/site").replace("paths: [src]", f"paths: [{root}/src]")
s = s.replace("watch:\n  - src/dnt\n", f"watch:\n  - {root}/src/dnt\n")
(t / "mkdocs.yml").write_text(s)
EOF
.venv/bin/mkdocs build --strict -f "$T/mkdocs.yml"
git status --short site                            # empty
```

- [ ] **Step 6: Commit**

```bash
git add src/dnt/refine/config.py pyproject.toml docs/api/refine docs/changelog.md CHANGELOG.md mkdocs.yml docs/superpowers/specs/2026-09-27-track-refinement-design.md tests/refine/test_config.py tests/refine/test_encoders.py tests/refine/test_cli.py tests/refine/test_minimal_install_vlm.py
git commit -m "docs(refine): document VLM verification; validate vlm settings; add the refine-vlm extra" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review (run by the plan's author)

**Spec coverage.**
- §7.1 evidence packet: Task 5 (tiles per kind, padding, upscaling, context frames, `send_context_frames`, one pass per chunk). Not done on purpose: the occluder label on the hidden-path frame (the link stage does not export the occluder box), recorded in the Task 9 spec follow-ups.
- §7.2 prompts, options, answer mapping, votes, invalid output: Tasks 2, 4, 6.
- §7.3 backends `openai_compat`, `anthropic`, `fake`, `none`: Tasks 1, 3, 7; extra `refine-vlm`: Task 9.
- §7.4 budget (closest to the midpoint first, rider subtype from the same budget), concurrency, cache, failures never abort: Tasks 4, 6, 7.
- §4.3 exceptions (rider subtype call, `unsure` -> pending): Task 6. The static, mixed, occluded and ambiguous caps are already in the stages; they now put those events in the VLM band.
- §8.1 review page: Task 8. §8.3 `vlm` counters: Task 7. §10 rows (no video, missing extra, VLM errors): Tasks 7 and 9.
- Left to Plan 4 on purpose: `apply`/replay and the use of `decisions.json`, `audit`, `features_recomputed`, the reference-case test (§11.4).

**Placeholders.** None: every code step shows its code.

**Type consistency.** `VLMAnswer`, `Question`, `Verdict`, `VLMRunner` (Tasks 1, 4) are used with the same field names in Tasks 6-7; `route_with_vlm(events, band, *, vlm: VLMRouting, round=0)` is called by `_Stages._route` with `VLMRouting(runner, evidence, cfg, fps)`; `EvidenceBuilder.build_many(events) -> {event.id: bytes | None}` is the only evidence call used by routing and by the review page; `Event.vlm` keys are `backend, model, answer, confidence, votes, reason, evidence, cached, error` everywhere.

**Review Focus coverage.** (1) almost-right replies: Tasks 1 and 4. (2) cost control: Tasks 4, 6, 7. (3) secrets: Tasks 3 and 4 (scrubbing); `send_context_frames`: Task 7. (4) evidence edge cases: Tasks 5 and 8. (5) running event loop and escaping: Tasks 4 and 8.

**Known limits, stated plainly.**
(a) The real `openai` and `anthropic` clients and the answers of a real model are not exercised by the suite: the backends are tested against fake client modules, and the answer quality (prompts, options) must be audited on real clips with a real model before the bands are tuned. (b) A model that answers with very high confidence and is wrong is accepted: only the audit (Plan 4) measures VLM precision. (c) The default Anthropic model id is pinned at `claude-sonnet-5-5`; verify it against the current model list when releasing. (d) The occluder box is not drawn on the hidden-path frame. (e) Review decisions in `decisions.json` take effect only when Plan 4's `apply` exists.
