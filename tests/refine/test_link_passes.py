import numpy as np
import pytest

from dnt.refine import io
from dnt.refine import link as link_mod
from dnt.refine.apply import merge_chains
from dnt.refine.config import RefineConfig
from dnt.refine.events import ACCEPTED, Decision, Event, EventKind
from dnt.refine.link import (
    Candidate,
    assign_in_passes,
    describe_tracks,
    legacy_link_events,
    resolve_chains,
    run_link_stage,
)
from dnt.refine.primitives import occlusion_flags
from dnt.refine.verify import Band, decide, route_without_vlm

from ._fixtures import box_rows, load_raw, table


def _make_event(c, routed, extra):
    return Event.propose(stage="link", kind=EventKind.LINK, tracks=[c.i, c.j],
                         lineage=[[[c.i, 0, 9]], [[c.j, 20, 29]]], frames=(9, 20),
                         params={"gate": c.gate, "gap": [9, 20]}, algo_score=routed,
                         signals={**c.signals, **extra})


def _route(decisions):
    def route(evs):
        for ev in evs:
            decide(ev, decisions.get(tuple(ev.tracks), Decision.AUTO_ACCEPT), source="auto")
    return route


def _c(i, j, s):
    return Candidate(i, j, "normal", 5, s, {})


def _assign(cands, decisions=None, cfg=None):
    return assign_in_passes(cands, cfg or RefineConfig.defaults(), make_event=_make_event,
                            route=_route(decisions or {}))


def test_rejected_edge_frees_endpoint_for_next_pass():
    evs = assign_in_passes([_c(1, 2, 0.9), _c(1, 3, 0.7)], RefineConfig.defaults(),
                           make_event=_make_event, route=_route({(1, 2): Decision.VLM_REJECT}))
    assert [e.tracks for e in evs] == [[1, 2], [1, 3]]
    assert evs[1].signals["pass"] == 2 and evs[1].signals["replaces"] == evs[0].proposal_key
    assert evs[1].decision is Decision.AUTO_ACCEPT


def test_pending_edge_reserves_its_endpoints():
    evs = assign_in_passes([_c(1, 2, 0.9), _c(1, 3, 0.7)], RefineConfig.defaults(),
                           make_event=_make_event,
                           route=_route({(1, 2): Decision.HUMAN_PENDING}))
    assert [e.tracks for e in evs] == [[1, 2]]


def test_accepted_pairs_are_frozen_in_later_passes():
    cands = [_c(1, 10, 0.9), _c(2, 10, 0.95), _c(2, 11, 0.85), _c(1, 12, 0.6)]
    evs = assign_in_passes(cands, RefineConfig.defaults(), make_event=_make_event,
                           route=_route({(2, 11): Decision.AUTO_REJECT}))
    assert sorted(e.tracks for e in evs) == [[1, 10], [2, 11]]
    assert not any(e.tracks == [2, 10] for e in evs)


def test_ambiguous_pair_is_capped_and_lists_its_alternative():
    cfg = RefineConfig.defaults()
    evs = assign_in_passes([_c(1, 2, 0.9), _c(1, 3, 0.9)], cfg, make_event=_make_event,
                           route=lambda e: route_without_vlm(e, Band.of(cfg.link)))
    assert len(evs) == 1
    ev = evs[0]
    assert ev.algo_score == pytest.approx(0.75) and ev.decision is Decision.HUMAN_PENDING
    assert ev.signals["margin"] == pytest.approx(0.0)
    assert len(ev.signals["alternatives"]) == 1


def test_clear_winner_keeps_its_score():
    evs = assign_in_passes([_c(1, 2, 0.9)], RefineConfig.defaults(), make_event=_make_event,
                           route=_route({}))
    assert evs[0].algo_score == pytest.approx(0.9) and evs[0].signals["alternatives"] == []


def _descs(*rows):
    w = io.to_work(table(*rows)).work
    return describe_tracks(w, RefineConfig.defaults(), 10.0, occlusion_flags(w, None, 0.3))


def _acc(i, j, score):
    ev = _make_event(_c(i, j, score), score, {})
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    ev.applied = True
    return ev


def test_chain_with_a_repeated_frame_skips_its_weakest_link():
    descs = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(11, 21), 30.0, 0.0),
                   box_rows(3, range(5, 31), 60.0, 0.0))
    a, b = _acc(1, 2, 0.9), _acc(2, 3, 0.8)
    pairs, skipped = resolve_chains([a, b], descs, overlap_frames=2)
    assert pairs == [(1, 2)] and skipped == [b]
    assert b.applied is False and b.signals["skipped_reason"] == "overlap"


def test_small_allowed_overlap_is_kept():
    descs = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(9, 21), 30.0, 0.0))
    pairs, skipped = resolve_chains([_acc(1, 2, 0.9)], descs, overlap_frames=2)
    assert pairs == [(1, 2)] and skipped == []


def test_sparse_overlap_link_keeps_unique_observed_rows():
    a = box_rows(1, [*range(0, 9), 11], 100.0, 100.0, vx=2.0)
    b = box_rows(2, range(9, 13), 118.0, 100.0, vx=2.0)  # same box as A at frame 11 (g = -2)
    w = io.to_work(table(a, b)).work
    cfg = RefineConfig.defaults()
    res = run_link_stage(w, cfg, 10.0, appearance=None, context=None, frame_size=None,
                         occluded=occlusion_flags(w, None, 0.3),
                         route=lambda e: route_without_vlm(e, Band.of(cfg.link)))
    assert res.accepted == [(1, 2)] and res.events[0].params["gate"] == "overlap"
    merged, _ = merge_chains(w, res.accepted)
    assert merged["frame"].tolist() == list(range(0, 13))  # B's frames 9 and 10 survive


def test_run_link_stage_links_collinear_fragments():
    w = io.to_work(table(box_rows(1, range(40), 100.0, 100.0, vx=2.0),
                         box_rows(2, range(50, 90), 200.0, 100.0, vx=2.0))).work
    cfg = RefineConfig.defaults()
    res = run_link_stage(w, cfg, 25.0, appearance=None, context=None, frame_size=None,
                         occluded=occlusion_flags(w, None, 0.3),
                         route=lambda e: route_without_vlm(e, Band.of(cfg.link)))
    assert res.accepted == [(1, 2)] and res.events[0].applied is True
    assert res.pending_endpoints == set()
    assert np.isclose(res.events[0].algo_score, 1 - 0.25 * 11 / 25)


# ---- additional rule-pinning and edge-case tests -------------------------------------------


def test_no_candidates_gives_no_events():
    assert _assign([]) == []


def test_single_candidate_is_one_pass_one_event():
    (ev,) = _assign([_c(4, 5, 0.6)])
    assert ev.tracks == [4, 5] and ev.signals["pass"] == 1
    assert ev.signals["S_link"] == pytest.approx(0.6)
    assert ev.signals["margin"] == pytest.approx(0.6) and "replaces" not in ev.signals
    assert ev.params == {"gate": "normal", "gap": [9, 20]}


def test_candidates_below_reject_below_never_become_events():
    lo = RefineConfig.defaults().link.reject_below
    evs = _assign([_c(1, 2, lo - 0.01), _c(3, 4, lo), _c(5, 6, 0.0)])
    assert [e.tracks for e in evs] == [[3, 4]]  # the boundary value itself is kept
    assert _assign([_c(1, 2, lo - 0.01)]) == []


def test_only_below_threshold_alternative_is_not_listed_or_used_for_margin():
    lo = RefineConfig.defaults().link.reject_below
    (ev,) = _assign([_c(1, 2, 0.9), _c(1, 3, lo - 0.01)])
    assert ev.signals["alternatives"] == [] and ev.signals["margin"] == pytest.approx(0.9)


def test_better_candidate_wins_and_loser_is_not_proposed_while_winner_is_accepted():
    evs = _assign([_c(1, 3, 0.6), _c(2, 3, 0.9)])
    assert [e.tracks for e in evs] == [[2, 3]]
    assert evs[0].algo_score == pytest.approx(0.9)  # margin 0.3 is clear: no cap


def test_loser_is_proposed_in_pass_two_once_the_winner_is_rejected():
    evs = _assign([_c(1, 3, 0.6), _c(2, 3, 0.9)], {(2, 3): Decision.VLM_REJECT})
    assert [e.tracks for e in evs] == [[2, 3], [1, 3]]
    assert [e.signals["pass"] for e in evs] == [1, 2]
    assert evs[1].signals["replaces"] == evs[0].proposal_key
    assert "replaces" not in evs[0].signals


def test_rejected_edge_is_not_proposed_again():
    evs = _assign([_c(1, 2, 0.9)], {(1, 2): Decision.AUTO_REJECT})
    assert [e.tracks for e in evs] == [[1, 2]]


def test_start_side_takeover_also_records_replaces():
    evs = _assign([_c(1, 2, 0.9), _c(3, 2, 0.7)], {(1, 2): Decision.HUMAN_REJECT})
    assert [e.tracks for e in evs] == [[1, 2], [3, 2]]
    assert evs[1].signals["replaces"] == evs[0].proposal_key and evs[1].signals["pass"] == 2


def test_replaces_names_the_latest_rejection_in_a_rejection_chain():
    cands = [_c(1, 2, 0.9), _c(1, 3, 0.8), _c(1, 4, 0.7)]
    evs = _assign(cands, {(1, 2): Decision.AUTO_REJECT, (1, 3): Decision.AUTO_REJECT})
    assert [e.tracks for e in evs] == [[1, 2], [1, 3], [1, 4]]
    assert evs[1].signals["replaces"] == evs[0].proposal_key
    assert evs[2].signals["replaces"] == evs[1].proposal_key


@pytest.mark.parametrize("max_passes", [1, 2, 3, 4])
def test_max_passes_is_respected(max_passes):
    cfg = RefineConfig.defaults()
    cfg.link.max_passes = max_passes
    cands = [_c(1, 2, 0.9), _c(1, 3, 0.8), _c(1, 4, 0.7), _c(1, 5, 0.6), _c(1, 6, 0.5)]
    rej = {(1, j): Decision.AUTO_REJECT for j in range(2, 7)}
    evs = _assign(cands, rej, cfg)
    assert [e.signals["pass"] for e in evs] == list(range(1, max_passes + 1))
    assert [e.tracks for e in evs] == [[1, j] for j in range(2, 2 + max_passes)]


def test_pending_endpoints_stay_reserved_when_a_later_pass_runs():
    # Pass 1 proposes (1,2) and (11,12) [both pending] and (3,4) [rejected]. Pass 2 must not
    # touch end 1 (reserved by (1,2)) or start 12 (reserved by (11,12)); only (3,5) is free.
    cands = [_c(1, 2, 0.9), _c(1, 6, 0.7), _c(11, 12, 0.9), _c(13, 12, 0.7),
             _c(3, 4, 0.9), _c(3, 5, 0.7)]
    pend = {(1, 2): Decision.HUMAN_PENDING, (11, 12): Decision.HUMAN_PENDING,
            (3, 4): Decision.AUTO_REJECT}
    evs = _assign(cands, pend)
    assert [e.tracks for e in evs] == [[1, 2], [3, 4], [11, 12], [3, 5]]
    assert [e.signals["pass"] for e in evs] == [1, 1, 1, 2]


def test_accepted_pairs_of_pass_one_stay_frozen_with_both_endpoints_removed():
    # (1,10) accepted in pass 1; (2,11) rejected. Pass 2 must not reuse end 1 or start 10.
    cands = [_c(1, 10, 0.9), _c(2, 10, 0.85), _c(2, 11, 0.8), _c(1, 12, 0.7), _c(2, 13, 0.6)]
    evs = _assign(cands, {(2, 11): Decision.AUTO_REJECT})
    assert [e.tracks for e in evs] == [[1, 10], [2, 11], [2, 13]]
    for e in evs[1:]:
        assert e.tracks[0] != 1 and e.tracks[1] != 10


def _margin_cfg(margin_min):
    cfg = RefineConfig.defaults()
    cfg.link.margin_min = margin_min
    return cfg


def test_ambiguity_cap_is_not_applied_at_exactly_margin_min():
    # dyadic scores keep the subtraction exact: margin == 0.25 == margin_min
    (ev,) = _assign([_c(1, 2, 1.0), _c(1, 3, 0.75)], cfg=_margin_cfg(0.25))
    assert ev.signals["margin"] == 0.25
    assert ev.algo_score == 1.0


def test_ambiguity_cap_is_applied_just_below_margin_min():
    (ev,) = _assign([_c(1, 2, 1.0), _c(1, 3, 0.8125)], cfg=_margin_cfg(0.25))
    assert ev.signals["margin"] == pytest.approx(0.1875)
    assert ev.algo_score == pytest.approx(RefineConfig.defaults().link.ambiguous_cap)
    assert ev.signals["S_link"] == 1.0  # the raw score is kept as a signal


def test_ambiguity_cap_never_raises_a_low_score():
    cap = RefineConfig.defaults().link.ambiguous_cap
    (ev,) = _assign([_c(1, 2, 0.6), _c(1, 3, 0.58)])
    assert ev.algo_score == pytest.approx(0.6) and cap > 0.6


def test_start_side_competitor_also_makes_a_pair_ambiguous():
    (ev,) = _assign([_c(1, 2, 0.9), _c(3, 2, 0.88)])
    assert ev.tracks == [1, 2] and ev.algo_score == pytest.approx(0.75)


def test_alternatives_are_sorted_and_limited_to_n_alternatives():
    # by score (best first), ties by (i, j); the best-scoring alternative has a larger j
    cands = [_c(1, 2, 0.9), _c(1, 3, 0.5), _c(1, 5, 0.5), _c(1, 4, 0.7), _c(1, 6, 0.45)]
    (ev,) = _assign(cands)
    assert ev.signals["alternatives"] == [{"i": 1, "j": 4, "score": 0.7},
                                          {"i": 1, "j": 3, "score": 0.5}]
    assert ev.signals["margin"] == pytest.approx(0.2)
    cfg = RefineConfig.defaults()
    cfg.link.n_alternatives = 1
    (ev,) = _assign(cands, cfg=cfg)
    assert ev.signals["alternatives"] == [{"i": 1, "j": 4, "score": 0.7}]
    assert ev.signals["margin"] == pytest.approx(0.2)  # the margin still uses the best alternative
    cfg.link.n_alternatives = 4
    (ev,) = _assign(cands, cfg=cfg)
    assert [a["j"] for a in ev.signals["alternatives"]] == [4, 3, 5, 6]


def test_alternatives_include_start_side_competitors():
    # the two links (0.6 + 0.7) outweigh the single strong one (0.9): winners are (3,2), (1,4)
    by = {tuple(e.tracks): e for e in _assign([_c(1, 2, 0.9), _c(3, 2, 0.6), _c(1, 4, 0.7)])}
    assert set(by) == {(3, 2), (1, 4)}
    assert by[(1, 4)].signals["alternatives"] == [{"i": 1, "j": 2, "score": 0.9}]
    assert by[(3, 2)].signals["alternatives"] == [{"i": 1, "j": 2, "score": 0.9}]


def test_independent_components_are_assigned_independently():
    cands = [_c(1, 2, 0.9), _c(1, 3, 0.8), _c(10, 11, 0.7), _c(12, 11, 0.6)]
    evs = _assign(cands)
    assert [e.tracks for e in evs] == [[1, 2], [10, 11]]
    assert [a["j"] for a in evs[0].signals["alternatives"]] == [3]
    assert [a["i"] for a in evs[1].signals["alternatives"]] == [12]


def test_rejection_in_one_component_leaves_the_other_untouched():
    cands = [_c(1, 2, 0.9), _c(1, 3, 0.8), _c(10, 11, 0.7), _c(12, 11, 0.6)]
    evs = _assign(cands, {(1, 2): Decision.AUTO_REJECT})
    assert [e.tracks for e in evs] == [[1, 2], [10, 11], [1, 3]]
    assert [e.signals["pass"] for e in evs] == [1, 1, 2]
    assert "replaces" not in evs[1].signals


def test_assignment_is_deterministic_and_independent_of_input_order():
    cands = [_c(1, 2, 0.9), _c(1, 3, 0.8), _c(4, 3, 0.85), _c(5, 6, 0.7), _c(7, 6, 0.7)]
    rej = {(1, 2): Decision.AUTO_REJECT}
    a = _assign(cands, rej)
    b = _assign(list(reversed(cands)), rej)
    c = _assign(cands, rej)
    for other in (b, c):
        assert [e.tracks for e in a] == [e.tracks for e in other]
        assert [e.proposal_key for e in a] == [e.proposal_key for e in other]
        assert [e.algo_score for e in a] == [e.algo_score for e in other]
        assert [e.signals for e in a] == [e.signals for e in other]


# ---- assignment maximises the total score, unmatched allowed (spec 6.3) -------------------


def _pairs(evs):
    return [tuple(e.tracks) for e in evs]


def test_strong_single_link_beats_two_marginal_links():
    # (a,x)=0.95 alone (0.95) outweighs (a,y)+(b,x) (0.84); a count-first solver drops 0.95
    evs = _assign([_c(1, 10, 0.95), _c(1, 11, 0.42), _c(2, 10, 0.42)])
    assert _pairs(evs) == [(1, 10)]
    assert evs[0].signals["S_link"] == pytest.approx(0.95)
    assert evs[0].decision is Decision.AUTO_ACCEPT


def test_two_links_are_chosen_when_they_outweigh_the_single_best():
    evs = _assign([_c(1, 10, 0.95), _c(1, 11, 0.60), _c(2, 10, 0.60)])  # 1.2 > 0.95
    assert sorted(_pairs(evs)) == [(1, 11), (2, 10)]


def test_competition_chain_keeps_the_strong_links_and_leaves_weak_ones_unmatched():
    # path e1-s1 .42, e2-s1 .95, e2-s2 .42, e3-s2 .42, e3-s3 .95: the perfect matching is
    # 1.79, the two strong links 1.90 (count-first would take the perfect matching)
    cands = [_c(1, 10, 0.42), _c(2, 10, 0.95), _c(2, 11, 0.42), _c(3, 11, 0.42),
             _c(3, 12, 0.95)]
    assert _pairs(_assign(cands)) == [(2, 10), (3, 12)]


def test_strong_diagonal_wins_over_weak_off_diagonals():
    cands = [_c(i, 10 + j, 0.9 if i == j else 0.45) for i in (1, 2, 3) for j in (1, 2, 3)]
    assert _pairs(_assign(cands)) == [(1, 11), (2, 12), (3, 13)]


def test_exact_ties_are_deterministic():
    cands = [_c(1, 10, 0.8), _c(1, 11, 0.8), _c(2, 10, 0.8), _c(2, 11, 0.8)]
    first = _pairs(_assign(cands))
    assert len(first) == 2
    assert _pairs(_assign(cands)) == first
    assert _pairs(_assign(list(reversed(cands)))) == first


def test_component_where_count_and_total_agree_still_works():
    cands = [_c(1, 10, 0.9), _c(2, 11, 0.8), _c(2, 10, 0.5)]
    assert _pairs(_assign(cands)) == [(1, 10), (2, 11)]


def test_rejected_strong_first_choice_frees_endpoints_for_weaker_pass_two_edges():
    cands = [_c(1, 10, 0.95), _c(1, 11, 0.42), _c(2, 10, 0.42)]
    evs = _assign(cands, {(1, 10): Decision.VLM_REJECT})
    assert _pairs(evs) == [(1, 10), (1, 11), (2, 10)]
    assert [e.signals["pass"] for e in evs] == [1, 2, 2]
    assert all(e.signals["replaces"] == evs[0].proposal_key for e in evs[1:])


# ---- resolve_chains -----------------------------------------------------------------------


def test_chain_skips_the_first_link_when_it_is_the_weakest():
    # 1/2 share 3 frames and 2/3 share 2 (both allowed); 1 and 3 share 26-28 unlinked
    descs = _descs(box_rows(1, range(0, 29), 0.0, 0.0), box_rows(2, range(25, 28), 30.0, 0.0),
                   box_rows(3, range(26, 41), 60.0, 0.0))
    a, b = _acc(1, 2, 0.6), _acc(2, 3, 0.8)
    pairs, skipped = resolve_chains([a, b], descs, overlap_frames=2)
    assert pairs == [(2, 3)] and skipped == [a] and a.applied is False and b.applied is True


def test_overlap_boundary_is_overlap_frames_plus_one_shared_frames():
    ok = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(8, 21), 30.0, 0.0))
    assert resolve_chains([_acc(1, 2, 0.9)], ok, overlap_frames=2) == ([(1, 2)], [])  # 3 shared
    bad = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(7, 21), 30.0, 0.0))
    ev = _acc(1, 2, 0.9)
    pairs, skipped = resolve_chains([ev], bad, overlap_frames=2)  # 4 shared
    assert pairs == [] and skipped == [ev] and ev.applied is False
    assert ev.signals["skipped_reason"] == "overlap"
    pairs, skipped = resolve_chains([_acc(1, 2, 0.9)], bad, overlap_frames=3)
    assert pairs == [(1, 2)] and skipped == []


def test_single_shared_frame_between_unlinked_chain_members_is_a_conflict():
    # 1/2 and 2/3 overlap by 2 frames (allowed); 1 and 3 share only frame 26, unlinked
    descs = _descs(box_rows(1, range(0, 27), 0.0, 0.0), box_rows(2, range(25, 28), 30.0, 0.0),
                   box_rows(3, range(26, 41), 60.0, 0.0))
    a, b = _acc(1, 2, 0.9), _acc(2, 3, 0.8)
    pairs, skipped = resolve_chains([a, b], descs, overlap_frames=2)
    assert pairs == [(1, 2)] and skipped == [b]
    # without the 1/3 frame the same chain is fine
    ok = _descs(box_rows(1, range(0, 26), 0.0, 0.0), box_rows(2, range(25, 28), 30.0, 0.0),
                box_rows(3, range(26, 41), 60.0, 0.0))
    assert resolve_chains([_acc(1, 2, 0.9), _acc(2, 3, 0.8)], ok, 2)[0] == [(1, 2), (2, 3)]


def test_separate_chains_may_overlap_each_other():
    descs = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(20, 30), 30.0, 0.0),
                   box_rows(3, range(0, 11), 0.0, 500.0), box_rows(4, range(20, 30), 30.0, 500.0))
    pairs, skipped = resolve_chains([_acc(1, 2, 0.9), _acc(3, 4, 0.8)], descs, 2)
    assert pairs == [(1, 2), (3, 4)] and skipped == []


def test_resolve_chains_without_links_is_empty():
    assert resolve_chains([], {}, 2) == ([], [])


# ---- run_link_stage -----------------------------------------------------------------------


def _three_fragments():
    return io.to_work(table(
        box_rows(1, range(40), 100.0, 100.0, vx=2.0),
        box_rows(2, range(50, 90), 200.0, 100.0, vx=2.0),
        box_rows(3, range(100, 140), 300.0, 100.0, vx=2.0),
        box_rows(4, range(40), 100.0, 700.0, vx=2.0),
        box_rows(5, range(50, 90), 200.0, 700.0, vx=2.0),
    )).work


def _stage(work, decisions, cfg=None):
    cfg = cfg or RefineConfig.defaults()
    return run_link_stage(work, cfg, 25.0, appearance=None, context=None, frame_size=None,
                          occluded=occlusion_flags(work, None, 0.3), route=_route(decisions))


def test_applied_only_for_accepted_links_and_pending_endpoints_are_exact():
    w = _three_fragments()
    res = _stage(w, {(2, 3): Decision.HUMAN_PENDING, (4, 5): Decision.VLM_REJECT})
    by = {tuple(e.tracks): e for e in res.events}
    assert set(by) == {(1, 2), (2, 3), (4, 5)}
    assert by[(1, 2)].applied is True
    assert by[(2, 3)].applied is False and by[(2, 3)].decision is Decision.HUMAN_PENDING
    assert by[(4, 5)].applied is False and by[(4, 5)].decision is Decision.VLM_REJECT
    assert res.accepted == [(1, 2)] and res.skipped == []
    assert res.pending_endpoints == {2, 3}


def test_pending_endpoints_collect_every_pending_event():
    w = _three_fragments()
    res = _stage(w, {(1, 2): Decision.HUMAN_PENDING, (4, 5): Decision.HUMAN_PENDING})
    assert res.pending_endpoints == {1, 2, 4, 5}
    assert res.accepted == [] or all(p not in [(1, 2), (4, 5)] for p in res.accepted)


def test_all_links_accepted_gives_no_pending_and_all_applied():
    res = _stage(_three_fragments(), {})
    assert sorted(res.accepted) == [(1, 2), (2, 3), (4, 5)]
    assert all(e.applied for e in res.events) and res.pending_endpoints == set()


def test_run_link_stage_marks_a_skipped_chain_link_unapplied(monkeypatch):
    descs = _descs(box_rows(1, range(0, 11), 0.0, 0.0), box_rows(2, range(11, 21), 30.0, 0.0),
                   box_rows(3, range(5, 31), 60.0, 0.0))
    cands = [_c(1, 2, 0.9), _c(2, 3, 0.85)]
    monkeypatch.setattr(link_mod, "score_candidates", lambda *a, **k: (cands, descs))
    res = _stage(_three_fragments(), {})
    by = {tuple(e.tracks): e for e in res.events}
    assert res.accepted == [(1, 2)] and res.skipped == [by[(2, 3)]]
    assert by[(1, 2)].applied is True and by[(2, 3)].applied is False
    assert by[(2, 3)].decision in ACCEPTED  # still accepted, just not applied
    assert by[(2, 3)].signals["skipped_reason"] == "overlap"
    assert by[(1, 2)].params["gap"] == [descs[1].t_e, descs[2].t_s]
    assert by[(1, 2)].lineage == [descs[1].lineage, descs[2].lineage]
    assert by[(1, 2)].frames == (descs[1].t_e, descs[2].t_s)


def test_run_link_stage_with_no_candidates_or_tracks():
    w = io.to_work(table(box_rows(1, range(30), 0.0, 0.0))).work
    res = _stage(w, {})
    assert res.events == [] and res.accepted == [] and res.pending_endpoints == set()
    w = io.to_work(table(box_rows(1, range(30), 0.0, 0.0), box_rows(2, range(30), 900.0, 0.0),
                         )).work
    assert _stage(w, {}).events == []


def test_legacy_mode_routes_every_match_as_accepted_and_applies_it():
    raw = load_raw(1)
    w = io.to_work(raw).work
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    expected = legacy_link_events(w, cfg, 25.0)
    assert expected, "fixture must produce legacy matches"
    seen = []

    def route(evs):
        seen.append(len(evs))
        for ev in evs:
            decide(ev, Decision.AUTO_ACCEPT, source="auto")

    res = run_link_stage(w, cfg, 25.0, appearance=None, context=None, frame_size=None,
                         occluded=occlusion_flags(w, None, 0.3), route=route)
    assert seen == [len(expected)]
    assert [e.tracks for e in res.events] == [e.tracks for e in expected]
    assert res.accepted == [(e.tracks[0], e.tracks[1]) for e in expected]
    assert all(e.applied and e.decision is Decision.AUTO_ACCEPT for e in res.events)
    assert res.pending_endpoints == set() and res.skipped == []
    assert all(e.params["gate"] == "legacy" for e in res.events)


def test_legacy_mode_does_not_apply_rejected_matches():
    w = io.to_work(load_raw(1)).work
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    n = len(legacy_link_events(w, cfg, 25.0))
    assert n >= 1

    def route(evs):
        for ev in evs:
            decide(ev, Decision.AUTO_REJECT, source="auto")

    res = run_link_stage(w, cfg, 25.0, appearance=None, context=None, frame_size=None,
                         occluded=occlusion_flags(w, None, 0.3), route=route)
    assert len(res.events) == n and res.accepted == []
    assert not any(e.applied for e in res.events)
