import numpy as np
import pandas as pd

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import EventKind
from dnt.refine.link import describe_tracks, legacy_link_events, score_candidates
from dnt.refine.refiner import fill_stage
from dnt.refine.screen import propose_orphans

from ._fixtures import table
from ._video import BLUE, RED
from .test_dedup import _rows, _work
from .test_evidence import _builder, _walker
from .test_evidence import _work as _ev_work

FRAMES = [f for f in range(10, 31) if f != 20]  # frame 20 was dropped by a merge
EXCLUDED = {122: {20}}


class _Recording:
    """An appearance provider that records the spans it is asked for."""

    def __init__(self):
        self.calls = []

    def clean_embeddings(self, raw_id, f0, f1):
        self.calls.append((int(raw_id), int(f0), int(f1)))
        frames = np.arange(int(f0), int(f1) + 1)
        return frames, np.tile([1.0, 0.0], (len(frames), 1))


def _link_work():
    return _work(_rows(122, FRAMES), _rows(7, range(33, 50)))


def test_link_descriptors_and_embedding_requests_skip_excluded_frames():
    work, cfg = _link_work(), RefineConfig.defaults()
    occluded = pd.Series(False, index=work.index)
    control = _Recording()
    score_candidates(work, cfg, 10.0, appearance=control, context=None, frame_size=None,
                     occluded=occluded)
    assert (122, 10, 30) in control.calls  # the span reads the dropped frame 20
    rec = _Recording()
    cands, descs = score_candidates(work, cfg, 10.0, appearance=rec, context=None,
                                    frame_size=None, occluded=occluded, excluded=EXCLUDED)
    assert cands and descs[122].lineage == [[122, 10, 19], [122, 21, 30]]
    assert (122, 10, 30) not in rec.calls
    assert (122, 10, 19) in rec.calls and (122, 21, 30) in rec.calls
    assert describe_tracks(work, cfg, 10.0, occluded, EXCLUDED)[122].lineage == descs[122].lineage


def test_legacy_link_events_use_the_excluded_frames():
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    (ev,) = legacy_link_events(_link_work(), cfg, 10.0, EXCLUDED)
    assert ev.lineage == [[[122, 10, 19], [122, 21, 30]], [[7, 33, 49]]]
    (plain,) = legacy_link_events(_link_work(), cfg, 10.0)
    assert plain.lineage == [[[122, 10, 30]], [[7, 33, 49]]]


def test_orphan_events_use_the_excluded_frames():
    work = _work(_rows(1, [0, 2, 3]))
    kw = dict(linked_tracks=set(), pending_endpoints=set())
    (ev,) = propose_orphans(work, RefineConfig.defaults(), 10.0, excluded={1: {1}}, **kw)[0]
    assert ev.lineage == [[[1, 0, 0], [1, 2, 3]]]
    (plain,) = propose_orphans(work, RefineConfig.defaults(), 10.0, **kw)[0]
    assert plain.lineage == [[[1, 0, 3]]]


def test_fill_events_use_the_excluded_frames():
    # frame 4 of raw 1 was dropped and the merged track holds raw 2's row there
    work = io.to_work(table(_rows(1, [f for f in range(10) if f != 4]), _rows(1, range(15, 25)),
                            _rows(1, [4]))).work
    work.loc[work["frame"] == 4, "raw_id"] = 2
    _, events = fill_stage(work, RefineConfig.defaults(), 10.0, {}, {1: {4}})
    fills = [e for e in events if e.kind is EventKind.FILL]
    assert len(fills) == 1
    assert fills[0].lineage == [[[1, 0, 3], [2, 4, 4], [1, 5, 24]]]
    _, plain = fill_stage(work, RefineConfig.defaults(), 10.0, {})
    fill = next(e for e in plain if e.kind is EventKind.FILL)
    assert fill.lineage == [[[1, 0, 24], [2, 4, 4]]]


def test_the_evidence_of_a_later_link_never_shows_the_dropped_observation(tmp_path):
    # the builder is made from the ORIGINAL raw table, where (122, 20) still exists: only the
    # split spans of the downstream event keep that observation out of the evidence
    raw = _ev_work(_walker(122, range(10, 22)), _walker(7, range(24, 40), x0=54.0))
    after = raw[~((raw["raw_id"] == 122) & (raw["frame"] == 20))]
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    builder = _builder(tmp_path, raw, {122: RED, 7: BLUE})

    def tiles(ev):
        return [k for _, ts in builder.plan(ev).rows for k in ts]

    (plain,) = legacy_link_events(after, cfg, 10.0)
    assert (122, 20) in tiles(plain) and builder._box_at(plain.lineage[0], 20) is not None
    (kept,) = legacy_link_events(after, cfg, 10.0, EXCLUDED)
    assert (122, 20) not in tiles(kept) and (122, 19) in tiles(kept)
    assert builder._box_at(kept.lineage[0], 20) is None  # the own-track box lookup
