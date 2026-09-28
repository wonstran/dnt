import math

import numpy as np

from dnt.refine.events import Decision, Event, EventKind, Ledger, assign_ids, proposal_key


def _split(tracks=(4,), cut=821, lineage=((12, 780, 900),)):
    return Event.propose(stage="switch", kind=EventKind.SPLIT, tracks=list(tracks),
                         lineage=[[list(x) for x in lineage]], frames=(cut, cut),
                         params={"cut_frame": cut}, algo_score=0.7, signals={"mot": 1.0})


def test_key_ignores_track_numbering_but_not_the_cut():
    assert _split(tracks=(4,)).proposal_key == _split(tracks=(99,)).proposal_key
    assert _split(cut=821).proposal_key != _split(cut=822).proposal_key


def test_key_ignores_score_signals_and_edit():
    # Build two events through Event.propose with different algo_score and signals
    a = Event.propose(stage="switch", kind=EventKind.SPLIT, tracks=[4],
                      lineage=[[list((12, 780, 900))]], frames=(821, 821),
                      params={"cut_frame": 821}, algo_score=0.7, signals={"mot": 1.0})
    b = Event.propose(stage="switch", kind=EventKind.SPLIT, tracks=[4],
                      lineage=[[list((12, 780, 900))]], frames=(821, 821),
                      params={"cut_frame": 821}, algo_score=0.1, signals={"x": 1})
    assert a.proposal_key == b.proposal_key
    # Test the module-level function with an extra non-defining param
    k1 = proposal_key("screen", "SPLIT", [[[1, 0, 9]]], {"cut_frame": 821, "note": "x"})
    k2 = proposal_key("screen", "SPLIT", [[[1, 0, 9]]], {"cut_frame": 821, "note": "y"})
    assert k1 == k2
    # Verify that changing a defining param does change the key
    k = proposal_key("screen", "DROP", [[[1, 0, 9]]], {"reason": "static", "spans": None})
    assert k != proposal_key("screen", "DROP", [[[1, 0, 9]]], {"reason": "in_vehicle",
                                                                "spans": None})


def test_round_trip_including_pending_and_nan(tmp_path):
    ev = _split()
    ev.signals["nis"] = float("nan")
    ev.decision = Decision.HUMAN_PENDING
    ev.decision_history.append({"decision": "HUMAN_PENDING", "round": 0, "source": "auto"})
    assign_ids([ev], "switch", 0)
    Ledger({"round": 0, "id_map": {3: 1}}, [ev]).write(tmp_path / "l.jsonl")
    header, *lines = (tmp_path / "l.jsonl").read_text().splitlines()
    assert '"round":0' in header and len(lines) == 1
    back = Ledger.read(tmp_path / "l.jsonl")
    got = back.events[0]
    assert got.id == "switch-r0-000001" and got.decision is Decision.HUMAN_PENDING
    assert got.frames == (821, 821) and got.kind is EventKind.SPLIT
    assert got.signals["nis"] is None and not math.isnan(got.algo_score)
    ev.signals["nis"] = None
    assert got == ev
    assert back.header["id_map"] == {"3": 1}


def test_write_is_deterministic(tmp_path):
    ev = _split()
    assign_ids([ev], "switch", 0)
    Ledger({"b": 1, "a": 2}, [ev]).write(tmp_path / "1.jsonl")
    Ledger({"a": 2, "b": 1}, [ev]).write(tmp_path / "2.jsonl")
    assert (tmp_path / "1.jsonl").read_bytes() == (tmp_path / "2.jsonl").read_bytes()


def test_assign_ids_continues_sequence():
    evs = [_split(), _split(cut=900)]
    assert assign_ids(evs, "switch", 1, start=5) == 7
    assert [e.id for e in evs] == ["switch-r1-000005", "switch-r1-000006"]


def test_proposal_key_normalizes_integral_floats():
    # int and integral float should produce the same key
    k_int = proposal_key("switch", EventKind.SPLIT, [[]], {"cut_frame": 821})
    k_float = proposal_key("switch", EventKind.SPLIT, [[]], {"cut_frame": 821.0})
    k_np = proposal_key("switch", EventKind.SPLIT, [[]], {"cut_frame": np.float64(821.0)})
    assert k_int == k_float == k_np
    # Non-integral float should differ
    k_nonint = proposal_key("switch", EventKind.SPLIT, [[]], {"cut_frame": 821.5})
    assert k_int != k_nonint
    # Test with lineage spans: int, float, and numpy float should match
    k_int_lin = proposal_key("screen", EventKind.DROP, [[[12, 780, 900]]], {"reason": "x", "spans": None})
    k_float_lin = proposal_key("screen", EventKind.DROP, [[[12, 780.0, 900.0]]], {"reason": "x", "spans": None})
    assert k_int_lin == k_float_lin
