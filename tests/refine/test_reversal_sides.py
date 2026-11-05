from dnt.refine.reversal import ReversalSides, resolve_reversal_chains, side_conflict

from ._reversal import link_event, rows, work_of


def test_own_tags_number_the_sides_of_each_cut_raw_id():
    s = ReversalSides(cuts={1: [30, 70]})
    assert s.own([[1, 0, 29]]) == {(1, 0)}
    assert s.own([[1, 30, 69]]) == {(1, 1)}  # a span starting at a cut is after it
    assert s.own([[1, 70, 99]]) == {(1, 2)}
    assert s.own([[2, 0, 99]]) == set()  # raw 2 has no cuts
    assert s.own([[1, 0, 29], [2, 30, 40]]) == {(1, 0)}


def test_side_conflict_names_the_smallest_raw_and_its_two_smallest_sides():
    assert side_conflict({(3, 0), (3, 1), (1, 2), (1, 0), (1, 1)}) == (1, 0, 1)
    assert side_conflict({(1, 0), (2, 1)}) is None
    assert side_conflict(set()) is None


def test_absorb_keeps_the_representatives_own_tags():
    # the review's minimal case: track 3 merged into 1, only 1 had a side, and 1's own rows
    # are all gone (its lineage now names raw 3 only)
    s = ReversalSides(cuts={1: [30]})
    s.absorb({3: 1}, {1: {(1, 0)}, 3: set()})
    assert s.tags(1, [[3, 0, 29]]) == {(1, 0)}


def test_tags_carried_by_dedup_survive_a_later_link_map():
    s = ReversalSides(cuts={1: [30]})
    s.absorb({3: 1}, {1: {(1, 0)}, 2: {(1, 1)}, 3: set()})  # dedup
    before_link = {1: s.tags(1, [[3, 0, 29]]), 5: set()}
    s.absorb({1: 1, 5: 1}, before_link)  # link: 5 joined 1
    assert (1, 0) in s.tags(1, [[3, 0, 29], [5, 40, 60]])


def test_table_tags_read_each_tracks_lineage():
    work = work_of(rows(1, range(0, 30)), rows(2, range(30, 60)), rows(3, range(0, 10)), raw={2: 1})
    s = ReversalSides(cuts={1: [30]})
    assert s.table_tags(work) == {1: {(1, 0)}, 2: {(1, 1)}, 3: set()}


def test_the_chain_check_drops_the_weakest_link_on_the_path_only():
    # 1 - 2 - 3 - 4 - 5; tracks 2 and 4 hold the two sides of raw 7's cut
    evs = [
        link_event(1, 2, 0.9),
        link_event(2, 3, 0.8),
        link_event(3, 4, 0.95),
        link_event(4, 5, 0.5),
    ]
    tags = {1: set(), 2: {(7, 0)}, 3: set(), 4: {(7, 1)}, 5: set()}
    (dropped,) = resolve_reversal_chains(evs, tags)
    assert dropped.tracks == [2, 3]  # 4 -> 5 is weaker but not on the 2..4 path
    assert dropped.applied is False and dropped.signals["skipped_reason"] == "reversal"
    assert dropped.signals["conflicts_with"] == {"raw": 7, "sides": [0, 1], "tracks": [2, 4]}
    assert all(e.applied for e in evs if e is not dropped)


def test_every_path_between_the_sides_is_broken():
    # a diamond: 1-2-4 and 1-3-4, with 1 and 4 on opposite sides
    evs = [
        link_event(1, 2, 0.9),
        link_event(2, 4, 0.6),
        link_event(1, 3, 0.9),
        link_event(3, 4, 0.7),
    ]
    tags = {1: {(7, 0)}, 2: set(), 3: set(), 4: {(7, 1)}}
    dropped = resolve_reversal_chains(evs, tags)
    assert [e.tracks for e in dropped] == [[2, 4], [3, 4]]


def test_a_two_way_pair_loses_both_links_weaker_first():
    # dropping the weaker link leaves the stronger one joining the same two sides, so the
    # check repeats until the component is clear
    evs = [link_event(1, 2, 0.9), link_event(2, 1, 0.7, [[2, 0, 0]], [[1, 1, 1]])]
    dropped = resolve_reversal_chains(evs, {1: {(7, 0)}, 2: {(7, 1)}})
    assert [e.tracks for e in dropped] == [[2, 1], [1, 2]]
    assert not any(e.applied for e in evs)


def test_a_pair_without_a_conflict_keeps_its_link():
    calm = [link_event(1, 2, 0.9)]
    assert resolve_reversal_chains(calm, {1: {(7, 0)}, 2: {(8, 1)}}) == []
    assert calm[0].applied


def test_a_track_that_holds_both_sides_alone_does_not_loop_forever():
    evs = [link_event(1, 2, 0.9)]
    assert resolve_reversal_chains(evs, {1: {(7, 0), (7, 1)}, 2: set()}) == []
