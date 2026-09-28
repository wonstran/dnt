# tests/refine/test_orphan.py
"""Tests for the orphan pass (spec 6.2)."""

from __future__ import annotations

import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import EventKind
from dnt.refine.screen import propose_orphans

from ._fixtures import box_rows, table


def _work(*rows):
    return io.to_work(table(*rows)).work


def test_short_unlinked_track_is_an_orphan_and_long_is_not():
    """Brief test 1: short unlinked is orphan, long is not."""
    w = _work(box_rows(1, range(2), 0.0, 0.0), box_rows(2, range(100), 50.0, 0.0),
              box_rows(3, range(3), 90.0, 0.0))
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                                    pending_endpoints=set())
    by = {e.tracks[0]: e for e in evs}
    assert set(by) == {1, 3} and deferred == []
    assert by[1].algo_score == pytest.approx(0.75)  # 0.2 s -> ramp(0.2; 0.5, 0.1)
    assert by[1].stage == "orphan" and by[1].params == {"reason": "orphan", "spans": None}


def test_linked_and_pending_tracks_are_skipped():
    """Brief test 2: linked is skipped, pending is deferred."""
    w = _work(box_rows(1, [0], 0.0, 0.0), box_rows(2, [5], 50.0, 0.0))
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks={1},
                                    pending_endpoints={2})
    assert evs == [] and deferred == [2]


def test_nearly_long_enough_track_scores_below_reject():
    """Brief test 3: 4 rows @ 10 fps = 0.4 s, scores < reject (0.3)."""
    w = _work(box_rows(1, range(4), 0.0, 0.0))  # 0.4 s -> 0.25 < reject 0.3
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    assert evs == []


def test_exactly_min_seconds_is_not_orphan():
    """Mutation: track with exactly min_seconds (5 rows @ 10 fps = 0.5 s) is NOT an orphan."""
    w = _work(box_rows(1, range(5), 0.0, 0.0))  # 0.5 s
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    assert evs == []


def test_three_rows_scores_from_ramp():
    """Mutation: 3 rows @ 10 fps = 0.3 s, ramp(0.3; 0.5, 0.1) = 0.5."""
    w = _work(box_rows(1, range(3), 0.0, 0.0))  # 0.3 s -> 0.5
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    assert len(evs) == 1 and evs[0].algo_score == pytest.approx(0.5)


def test_one_row_gives_max_score():
    """Mutation: 1 row @ 10 fps = 0.1 s, ramp(0.1; 0.5, 0.1) = 1.0 >= accept (0.70)."""
    w = _work(box_rows(1, range(1), 0.0, 0.0))  # 0.1 s -> 1.0
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    assert len(evs) == 1 and evs[0].algo_score == pytest.approx(1.0)


def test_linked_track_not_deferred():
    """Mutation: linked short track gives no event and is NOT in deferred list."""
    w = _work(box_rows(1, range(2), 0.0, 0.0))  # 0.2 s, orphan if not linked
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks={1},
                                    pending_endpoints=set())
    assert evs == [] and deferred == []


def test_pending_short_track_is_deferred():
    """Mutation: short track in pending_endpoints gives no event but IS deferred."""
    w = _work(box_rows(1, range(2), 0.0, 0.0))  # 0.2 s, would be orphan
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                                    pending_endpoints={1})
    assert evs == [] and deferred == [1]


def test_long_pending_track_not_deferred():
    """Mutation: long track in pending_endpoints is NOT deferred (not an orphan candidate)."""
    w = _work(box_rows(1, range(10), 0.0, 0.0))  # 1.0 s >= min_seconds, not orphan
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                                    pending_endpoints={1})
    assert evs == [] and deferred == []


def test_linked_overrides_pending():
    """Mutation: track in both linked_tracks and pending_endpoints is skipped as linked."""
    w = _work(box_rows(1, range(2), 0.0, 0.0))  # 0.2 s, would be orphan/pending
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks={1},
                                    pending_endpoints={1})
    # Should be skipped by linked check first, so NOT in deferred
    assert evs == [] and deferred == []


def test_event_fields():
    """Mutation: event has correct fields: stage, kind, params, signals, frames, lineage."""
    w = _work(box_rows(1, range(2), 0.0, 0.0))  # 0.2 s
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    assert len(evs) == 1
    ev = evs[0]
    assert ev.stage == "orphan"
    assert ev.kind == EventKind.DROP
    assert ev.params == {"reason": "orphan", "spans": None}
    assert ev.tracks == [1]
    assert ev.frames == (0, 1)
    assert "observed_seconds" in ev.signals
    assert ev.signals["observed_seconds"] == pytest.approx(0.2)
    assert len(ev.lineage) == 1


def test_events_ordered_by_track_id():
    """Mutation: events are ordered by track id."""
    w = _work(box_rows(3, range(2), 0.0, 0.0), box_rows(1, range(2), 10.0, 0.0),
              box_rows(2, range(2), 20.0, 0.0))
    evs, _ = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                             pending_endpoints=set())
    track_ids = [e.tracks[0] for e in evs]
    assert track_ids == sorted(track_ids) == [1, 2, 3]


def test_deferred_ordered_by_track_id():
    """Mutation: deferred list is ordered by track id."""
    w = _work(box_rows(3, range(2), 0.0, 0.0), box_rows(1, range(2), 10.0, 0.0),
              box_rows(2, range(2), 20.0, 0.0))
    _, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                                   pending_endpoints={3, 1, 2})
    assert deferred == sorted(deferred) == [1, 2, 3]


def test_empty_work_returns_empty():
    """Mutation: empty work table returns ([], [])."""
    w = _work()  # empty table
    evs, deferred = propose_orphans(w, RefineConfig.defaults(), 10.0, linked_tracks=set(),
                                    pending_endpoints=set())
    assert evs == [] and deferred == []
