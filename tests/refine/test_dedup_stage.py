import pytest

from dnt.refine.config import RefineConfig
from dnt.refine.dedup import propose_merges
from dnt.refine.events import Decision, EventKind, Ledger
from dnt.refine.refiner import TrackRefiner, _Stages
from dnt.refine.verify import Band

from ._fixtures import table
from .test_dedup import _rows, _work


def _stages(cfg, fps=20.0):
    return _Stages(cfg, fps, None, None, None, None, {})


def _short_pair_cfg(link_on):
    cfg = RefineConfig.defaults()
    cfg.link.enabled = link_on
    cfg.fill.enabled = False  # keep the orphan assertions apart from interpolation
    cfg.dedup.min_observed = 1
    cfg.dedup.min_overlap_seconds = 0.1
    return cfg


@pytest.mark.parametrize("link_on", [True, False])
def test_an_accepted_merge_protects_its_representative_from_the_orphan_drop(link_on):
    work = _work(_rows(1, [0, 20]), _rows(2, [10, 30]))  # 4 rows in all: 0.2 s at 20 fps
    off = _short_pair_cfg(link_on)
    off.dedup.enabled = False
    out, events = _stages(off).run(work)
    assert out.empty and {e.stage for e in events if e.kind is EventKind.DROP} == {"orphan"}
    on = _short_pair_cfg(link_on)
    out, events = _stages(on).run(work)
    assert out["track"].unique().tolist() == [1] and len(out) == 4
    (merge,) = [e for e in events if e.stage == "dedup"]
    assert merge.decision is Decision.AUTO_ACCEPT and merge.applied
    assert not [e for e in events if e.stage == "orphan"]


@pytest.mark.parametrize("link_on", [True, False])
def test_a_pending_merge_defers_the_orphan_drop_of_both_endpoints(link_on):
    work = _work(_rows(1, [0, 20]), _rows(2, [10, 30], dx=14.0))
    stages = _stages(_short_pair_cfg(link_on))
    out, events = stages.run(work)
    (merge,) = [e for e in events if e.stage == "dedup"]
    assert merge.decision is Decision.HUMAN_PENDING
    assert not [e for e in events if e.stage == "orphan"]
    assert sorted(stages.orphan_deferred) == [1, 2]
    assert sorted(out["track"].unique()) == [1, 2]


def test_dedup_routing_never_goes_through_the_vlm():
    cfg = _short_pair_cfg(True)
    stages = _stages(cfg)
    stages.vlm = object()  # route_with_vlm would fail on it: it has no runner or evidence
    events = propose_merges(_work(_rows(1, [0, 20]), _rows(2, [10, 30], dx=14.0)), cfg, 20.0)
    stages._route(events, Band.of(cfg.dedup), "dedup", use_vlm=False)
    assert [e.decision for e in events] == [Decision.HUMAN_PENDING]
    assert events[0].id == "dedup-r0-000001"


def test_absorbed_ids_compose_across_dedup_and_link():
    work = _work(_rows(1, range(0, 60, 2)), _rows(2, range(1, 60, 2)), _rows(3, range(62, 111)))
    stages = _stages(RefineConfig.defaults(), fps=10.0)
    out, events = stages.run(work)
    assert out["track"].unique().tolist() == [1]
    assert stages.absorbed == {2: 1, 3: 1}  # 2 into 1 by the merge, 3 into 1 by the link
    kinds = {(e.stage, str(e.kind), str(e.decision)) for e in events if e.stage != "fill"}
    assert ("dedup", "MERGE", "AUTO_ACCEPT") in kinds and ("link", "LINK", "AUTO_ACCEPT") in kinds
    assert stages.merge_counts["applied"] == 1


def _two_hop():
    # C (1) is linked later. B (2) and A (3) interleave and merge. D (4) walks beside them, so
    # the A/D merge is only uncertain: a pending event whose endpoint A is absorbed twice.
    return _work(_rows(1, range(0, 41)), _rows(2, range(50, 112, 2)),
                 _rows(3, range(51, 112, 2)), _rows(4, range(52, 111, 2), dx=14.0))


def _spy_orphans(monkeypatch):
    seen = {}
    original = _Stages._orphans

    def spy(self, work, linked, pending):
        seen["linked"], seen["pending"] = set(linked), set(pending)
        return original(self, work, linked, pending)

    monkeypatch.setattr(_Stages, "_orphans", spy)
    return seen


def test_a_merged_track_that_link_then_absorbs_resolves_and_stays_protected(monkeypatch):
    seen = _spy_orphans(monkeypatch)
    stages = _stages(RefineConfig.defaults(), fps=10.0)
    out, events = stages.run(_two_hop())
    ab = next(e for e in events if e.stage == "dedup" and e.tracks == [2, 3])
    ad = next(e for e in events if e.stage == "dedup" and e.tracks == [3, 4])
    assert ab.decision is Decision.AUTO_ACCEPT and ab.applied
    assert ad.decision is Decision.HUMAN_PENDING
    link = next(e for e in events if e.stage == "link" and e.applied)
    assert link.tracks == [1, 2]  # the merged track (2) is absorbed by the earlier track 1
    # A (3) was absorbed by B (2) and B by C (1): the map must resolve A to C, not stop at B
    assert stages.absorbed == {2: 1, 3: 1}
    # what dedup produced, before link maps it
    assert stages.merged_reps == {2} and stages.merge_pending == {2, 4}
    # what the orphan pass received: both sets were mapped through link's representatives
    assert 1 in seen["linked"] and 2 not in seen["linked"]
    assert {1, 4} <= seen["pending"] and 2 not in seen["pending"]
    assert sorted(out["track"].unique()) == [1, 4]


def test_a_disabled_stage_leaves_no_trace():
    cfg = RefineConfig.defaults()
    cfg.dedup.enabled = False
    work = _work(_rows(1, range(0, 60, 2)), _rows(2, range(1, 60, 2)))
    stages = _stages(cfg, fps=10.0)
    out, events = stages.run(work)
    assert not [e for e in events if e.stage == "dedup"]
    assert stages.absorbed == {} and stages.excluded == {}
    assert stages.merge_counts == {"applied": 0, "redundant": 0, "conflict": 0, "dropped_rows": 0}
    assert sorted(out["track"].unique()) == [1, 2]


def _write(tmp_path, df):
    p = tmp_path / "t.txt"
    df.to_csv(p, index=False, header=False)
    return p


def test_the_ledger_header_and_summary_describe_the_merges(tmp_path):
    src = _write(tmp_path, table(_rows(1, range(0, 120, 2)), _rows(2, range(1, 120, 2))))
    refiner = TrackRefiner(RefineConfig.defaults())
    tracks = refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    assert tracks["track"].nunique() == 1 and len(tracks) == 120
    ledger = Ledger.read(refiner.last_result.ledger_path)
    assert ledger.header["absorbed"] == {"2": 1}
    summary = refiner.last_result.summary
    assert summary["dedup"] == {"applied": 1, "redundant": 0, "conflict": 0, "dropped_rows": 0}
    assert "dedup/MERGE/AUTO_ACCEPT" in summary["events"]
    off = RefineConfig.defaults()
    off.dedup.enabled = False
    refiner = TrackRefiner(off)
    tracks = refiner.refine(src, tmp_path / "off.txt", fps=10, verbose=False)
    assert tracks["track"].nunique() == 2  # fill gives each ID the other's frames
    assert Ledger.read(refiner.last_result.ledger_path).header["absorbed"] == {}
    assert refiner.last_result.summary["dedup"]["applied"] == 0
