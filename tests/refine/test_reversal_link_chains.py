from dnt.refine.config import RefineConfig
from dnt.refine.events import Event, EventKind
from dnt.refine.link import resolve_chains, run_link_stage, score_candidates
from dnt.refine.primitives import occlusion_flags
from dnt.refine.reversal import ReversalSides, resolve_reversal_chains
from dnt.refine.verify import Band, route_without_vlm

from ._reversal import FPS, link_event, rows, work_of


def _cfg(mode="scored"):
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.mode = mode
    return cfg


def _link(work, sides=None, cfg=None):
    cfg = cfg or _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    band = Band.of(cfg.link)
    return run_link_stage(work, cfg, FPS, appearance=None, context=None, frame_size=None,
                          occluded=occ, route=lambda e: route_without_vlm(e, band),
                          reversal_sides=sides)


def _connected(pairs, a, b):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(j)] = find(i)
    return find(a) == find(b)


def _chain_work():
    # A = raw 1 frames 0-29, X = raw 2 frames 28-31, B = raw 1 frames 30-59 (review 21:16)
    return work_of(rows(1, range(0, 30)), rows(2, range(28, 32)), rows(3, range(30, 60)),
                   raw={3: 1})


def test_without_the_check_the_overlap_chain_merges_both_sides():
    res = _link(_chain_work())
    assert sorted(res.accepted) == [(1, 2), (2, 3)]


def test_the_chain_check_drops_one_link_of_the_overlap_chain():
    res = _link(_chain_work(), ReversalSides(cuts={1: [30]}))
    applied = [e for e in res.events if e.applied]
    dropped = [e for e in res.events if e.signals.get("skipped_reason") == "reversal"]
    assert len(applied) == 1 and len(dropped) == 1
    assert dropped[0].signals["conflicts_with"] == {"raw": 1, "sides": [0, 1], "tracks": [1, 3]}
    assert dropped[0] in res.skipped
    assert not _connected(res.accepted, 1, 3)


def test_the_path_follows_the_links_not_the_first_frames():
    # review 21:42: A 0-28 raw 1, X 30-31 raw 10, Y 29-32 raw 3, B 33-60 raw 10; cut raw 10 at 32
    work = work_of(rows(1, range(0, 29), lambda f: 100.0 + 3.0 * f),
                   rows(2, range(30, 32), lambda f: 100.0 + 3.0 * f),
                   rows(3, range(29, 33), lambda f: 100.0 + 3.0 * f),
                   rows(4, range(33, 61), lambda f: 100.0 + 3.0 * f), raw={2: 10, 4: 10})
    cfg = _cfg()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, descs = score_candidates(work, cfg, FPS, appearance=None, context=None,
                                    frame_size=None, occluded=occ)
    assert {(1, 2), (2, 3), (3, 4)} <= {(c.i, c.j) for c in cands}
    evs = [link_event(i, j, s, descs[i].lineage, descs[j].lineage)
           for (i, j), s in (((1, 2), 0.9), ((2, 3), 0.8), ((3, 4), 0.7))]
    _, skipped = resolve_chains(evs, descs, cfg.link.overlap_frames)
    assert skipped == []  # the frame-overlap check sees no problem
    sides = ReversalSides(cuts={10: [32]})
    tags = {t: sides.tags(t, d.lineage) for t, d in descs.items()}
    assert tags == {1: set(), 2: {(10, 0)}, 3: set(), 4: {(10, 1)}}
    (dropped,) = resolve_reversal_chains(evs, tags)
    assert dropped.tracks == [3, 4]  # the weakest on the path 2 -> 3 -> 4
    assert dropped.signals["conflicts_with"] == {"raw": 10, "sides": [0, 1], "tracks": [2, 4]}
    assert evs[0].applied  # A -> X is not on the path


def test_legacy_mode_runs_the_chain_check_too(monkeypatch):
    work = _chain_work()

    def fake_legacy(work, cfg, fps, excluded=None, *, reversal_sides=None):
        def lin(t):
            g = work[work["track"] == t]
            return [[int(g["raw_id"].iloc[0]), int(g["frame"].min()), int(g["frame"].max())]]

        return [
            Event.propose(stage="link", kind=EventKind.LINK, tracks=[a, b],
                          lineage=[lin(a), lin(b)], frames=(0, 1),
                          params={"gate": "legacy", "gap": [a, b]}, algo_score=1.0)
            for a, b in ((1, 2), (2, 3))
        ]

    monkeypatch.setattr("dnt.refine.link.legacy_link_events", fake_legacy)
    res = _link(work, ReversalSides(cuts={1: [30]}), cfg=_cfg("legacy"))
    assert len(res.accepted) == 1 and not _connected(res.accepted, 1, 3)
    assert [e.signals["skipped_reason"] for e in res.skipped] == ["reversal"]
