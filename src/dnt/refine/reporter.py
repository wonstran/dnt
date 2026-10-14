"""One ``Reporter`` owns the console output of a refinement run.

Stages only call it (``stage_start``, ``advance``, ``stage_done``); nothing else prints. It is
reached through a context variable (``current()``), so it need not be passed through every
module. Each stage has one bar, from 0 to 100%, and it stays on screen when the stage ends.
Everything else is a permanent log line. When the output is not a terminal there are no bars:
only a line per stage, and a progress line every 10% or 30 seconds of a stage's work.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from tqdm import tqdm

#: Without a terminal, a stage with fewer items than this logs no progress lines.
MIN_ITEMS_FOR_BAR = 256
_LOG_STEPS = 10  # a progress line every 1/10 of a sub-task
_LOG_SECONDS = 30.0
_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"

_current: ContextVar[Reporter | None] = ContextVar("dnt_refine_reporter", default=None)


def current() -> Reporter | None:
    """Return the active reporter, or None when the code runs as a library call."""
    return _current.get()


def advance(n: int = 1) -> None:
    """Count ``n`` finished items of the running stage; nothing when no reporter is active."""
    rep = _current.get()
    if rep is not None:
        rep.advance(n)


def extend(n: int) -> None:
    """Add ``n`` items to the running stage's work; nothing when no reporter is active."""
    rep = _current.get()
    if rep is not None:
        rep.extend(n)


@contextmanager
def use(reporter: Reporter) -> Iterator[Reporter]:
    """Make ``reporter`` the active one inside the block."""
    token = _current.set(reporter)
    try:
        yield reporter
    finally:
        _current.reset(token)


class _Bar:
    """Progress of one stage: ``update(n)`` counts ``n`` finished items of ``total``."""

    def __init__(self, reporter: Reporter, label: str, total: int = 0):
        self._rep, self.label, self.total = reporter, label, 0
        self.done = 0
        self._bar = None
        self._announced = False
        self._step = 1  # next 10% mark to log, as a multiple of 10%
        self._t_log = time.monotonic()
        self.extend(total)

    @property
    def _min_items(self) -> int:
        return 1 if self._rep.live else MIN_ITEMS_FOR_BAR

    def extend(self, n: int) -> None:
        """Add ``n`` items to the total, for work found only after the stage began."""
        self.total += int(n)
        rep = self._rep
        if not rep.enabled or self.total < self._min_items:
            return
        if rep.live:
            if self._bar is None:
                self._bar = tqdm(
                    total=self.total,
                    desc=self.label,
                    bar_format=_BAR_FORMAT,
                    file=rep.stream,
                    leave=True,
                )
                self._bar.update(self.done)
            else:
                self._bar.total = self.total
                self._bar.refresh()
        elif not self._announced:
            self._announced = True
            rep.emit(f"{self.label}: {self.total} items")

    def update(self, n: int = 1) -> None:
        """Count ``n`` more finished items."""
        self.done += n
        if self._bar is not None:
            self._bar.update(n)
        elif self._announced:
            now = time.monotonic()
            if (
                self.done * _LOG_STEPS >= self._step * self.total
                or now - self._t_log >= _LOG_SECONDS
            ):
                self._t_log = now
                self._step = self.done * _LOG_STEPS // self.total + 1
                self._rep.emit(
                    f"{self.label}: {100 * self.done // self.total}% ({self.done}/{self.total})"
                )

    def close(self) -> None:
        """Finish at 100% and leave the bar on screen."""
        if self._bar is not None:
            self._bar.n = self._bar.total
            self._bar.refresh()
            self._bar.close()
            self._bar = None


class Reporter:
    """Prints stage lines, the stage's progress bar and a closing summary for one clip.

    Parameters
    ----------
    enabled : bool
        False silences everything (``verbose=False``).
    tag : str
        Prefix of every line, e.g. ``"[Video 2/3]"``.
    n_stages : int
        Number of stages, shown in the stage label.
    stream : file-like, optional
        Where to write; ``sys.stderr`` by default.
    live : bool, optional
        Draw bars. Default: whether ``stream`` is a terminal.

    """

    def __init__(
        self, *, enabled: bool = True, tag: str = "", n_stages: int = 6, stream=None, live=None
    ):
        """Set up; the clock starts on ``__enter__``."""
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        isatty = getattr(self.stream, "isatty", lambda: False)()
        self.live = bool(enabled and (isatty if live is None else live))
        self.tag = f"{tag} " if tag else ""
        self.n_stages = n_stages
        self.embedded = 0
        self.cached = 0
        self._stage: tuple[str, int | None] | None = None
        self._progress: _Bar | None = None
        self._notes: Counter = Counter()
        self._t0 = self._t_stage = time.monotonic()

    def __enter__(self) -> Reporter:
        """Start the clip's clock."""
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        """Remove an unfinished stage bar."""
        self._close_progress()

    def emit(self, text: str) -> None:
        """Write one permanent line."""
        if not self.enabled:
            return
        if self.live:
            tqdm.write(text, file=self.stream)
        else:
            print(text, file=self.stream, flush=True)

    def info(self, text: str) -> None:
        """Write one permanent line with the video tag."""
        self.emit(f"{self.tag}{text}")

    def _label(self) -> str:
        name, index = self._stage
        return f"Stage {index}/{self.n_stages} {name}" if index else name

    def stage_start(self, name: str, index: int | None = None, total: int = 0) -> None:
        """Begin a stage; ``index`` (1-based) numbers it.

        ``total`` is the stage's amount of work (``advance`` counts it off, ``extend`` adds to
        it); the stage's bar appears once it is known.
        """
        self._close_progress()
        self._stage, self._t_stage = (name, index), time.monotonic()
        self._notes.clear()
        self._progress = _Bar(self, f"{self.tag}{self._label()}", int(total))
        if not self.live and self.enabled:
            self.info(f"{self._label()} ...")

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    def advance(self, n: int = 1) -> None:
        """Count ``n`` finished items of the stage's work."""
        if self._progress is not None:
            self._progress.update(n)

    def extend(self, n: int) -> None:
        """Add ``n`` items to the stage's work, found only after the stage began."""
        if self._progress is not None:
            self._progress.extend(n)

    def note(self, *, embedded: int = 0, cached: int = 0) -> None:
        """Record embedded and cached crops, for the stage line and the summary."""
        self._notes["embedded"] += embedded
        self._notes["cached"] += cached
        self.embedded += embedded
        self.cached += cached

    @contextmanager
    def task(self, name: str, total: int) -> Iterator[_Bar]:
        """Count ``total`` more items of the running stage (the stage has one bar only)."""
        if self._progress is None:
            self.stage_start(name)
        self._progress.extend(int(total))
        yield self._progress

    def stage_done(
        self,
        *,
        tracks_before: int | None = None,
        tracks_after: int | None = None,
        events: int | None = None,
        skipped: bool = False,
    ) -> None:
        """End the stage: its bar stays at 100%, and one permanent line gives time and counts."""
        dt = time.monotonic() - self._t_stage
        self._close_progress()
        parts = []
        if skipped:
            parts.append("skipped")
        else:
            if events is not None:
                parts.append(f"{events} events")
            if tracks_before is not None and tracks_after is not None:
                parts.append(f"tracks {tracks_before} -> {tracks_after}")
            emb, cached = self._notes["embedded"], self._notes["cached"]
            if emb or cached:
                parts.append(f"embedded {emb} crops ({cached} cached)")
        detail = " · ".join(parts)
        self.info(f"{self._label()} done in {dt:.1f}s" + (f" · {detail}" if detail else ""))

    def summary(self, tracks_before: int, tracks_after: int) -> None:
        """Write the per-clip summary line."""
        dt = time.monotonic() - self._t0
        self.info(
            f"Done in {dt:.1f}s · crops embedded {self.embedded}, cached {self.cached}"
            f" · tracks {tracks_before} -> {tracks_after}"
        )
