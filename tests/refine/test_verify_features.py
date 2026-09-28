import numpy as np
import pytest

from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.features import ArrayAppearance, track_embeddings
from dnt.refine.verify import Band, band_route, decide, route_without_vlm


def _ev(kind, score, params):
    return Event.propose(stage="screen", kind=kind, tracks=[1], lineage=[[[1, 0, 9]]],
                         frames=(0, 9), params=params, algo_score=score)


def test_band_route():
    b = Band(0.85, 0.40)
    assert band_route(0.9, b) is Decision.AUTO_ACCEPT
    assert band_route(0.3, b) is Decision.AUTO_REJECT
    assert band_route(0.6, b) is None


def test_band_route_edges():
    """Test band_route at exact threshold values."""
    b = Band(0.85, 0.40)
    assert band_route(0.85, b) is Decision.AUTO_ACCEPT
    assert band_route(0.40, b) is None
    assert band_route(0.3999, b) is Decision.AUTO_REJECT


def test_route_without_vlm_pending_and_edits():
    acc = _ev(EventKind.DROP, 0.9, {"reason": "in_vehicle", "spans": None})
    mid = _ev(EventKind.DROP, 0.6, {"reason": "static", "spans": None})
    rider = _ev(EventKind.RECLASS, 1.0, {"new_cls": None, "spans": None})
    hinted = _ev(EventKind.RECLASS, 1.0, {"new_cls": 3, "spans": None})
    route_without_vlm([acc, mid, rider, hinted], Band(0.85, 0.40))
    assert acc.decision is Decision.AUTO_ACCEPT and acc.edit["kind"] == "DROP"
    assert mid.decision is Decision.HUMAN_PENDING and mid.edit is None
    assert rider.decision is Decision.HUMAN_PENDING and rider.signals["needs_subtype"] is True
    assert hinted.decision is Decision.AUTO_ACCEPT
    assert acc.decision_history == [{"decision": "AUTO_ACCEPT", "round": 0, "source": "auto"}]


def test_route_without_vlm_auto_reject():
    """Test that AUTO_REJECT event has edit=None and history entry."""
    ev = _ev(EventKind.DROP, 0.3, {"reason": "static", "spans": None})
    route_without_vlm([ev], Band(0.85, 0.40))
    assert ev.decision is Decision.AUTO_REJECT
    assert ev.edit is None
    assert len(ev.decision_history) == 1
    assert ev.decision_history[0]["decision"] == "AUTO_REJECT"


def test_decide_deep_copy_edit_params():
    """Test that edit params are deep copied and mutations don't affect proposal."""
    ev = _ev(EventKind.DROP, 0.9, {"reason": "in_vehicle", "spans": [[5, 9]]})
    decide(ev, Decision.AUTO_ACCEPT, source="test")
    assert ev.edit["params"] == ev.params
    assert ev.edit["params"] is not ev.params
    assert ev.edit["params"]["spans"] is not ev.params["spans"]
    ev.edit["params"]["spans"][0][0] = 999
    assert ev.params["spans"][0][0] == 5


def test_decide_edit_kind_matches():
    """Test that edit kind matches the event kind."""
    ev = _ev(EventKind.DROP, 0.9, {"reason": "in_vehicle", "spans": None})
    decide(ev, Decision.AUTO_ACCEPT, source="test")
    assert ev.edit["kind"] == "DROP"
    assert ev.edit["kind"] == str(ev.kind)


def test_decide_multiple_rounds():
    """Test that decide can be called multiple times with different rounds."""
    ev = _ev(EventKind.DROP, 0.6, {"reason": "static", "spans": None})
    decide(ev, Decision.AUTO_ACCEPT, source="round1", round=0)
    decide(ev, Decision.AUTO_REJECT, source="round2", round=1)
    assert len(ev.decision_history) == 2
    assert ev.decision_history[0]["round"] == 0
    assert ev.decision_history[1]["round"] == 1
    assert ev.edit is None


def test_decide_accepted_then_rejected_edit_follows_last():
    """Test that edit follows the last decision (accepted then rejected gives None)."""
    ev = _ev(EventKind.DROP, 0.9, {"reason": "in_vehicle", "spans": None})
    decide(ev, Decision.AUTO_ACCEPT, source="accept", round=0)
    first_edit = ev.edit
    assert first_edit is not None
    decide(ev, Decision.AUTO_REJECT, source="reject", round=1)
    assert ev.edit is None


def test_array_appearance_normalizes_and_filters():
    app = ArrayAppearance({7: ([3, 1, 2], np.array([[0, 3.0], [2.0, 0], [0, 5.0]]))})
    f, e = app.clean_embeddings(7, 1, 2)
    assert f.tolist() == [1, 2]
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)
    assert app.clean_embeddings(99, 0, 10)[0].size == 0


def test_array_appearance_sorts_and_preserves_pairing():
    """Test that sorting preserves frame-embedding pairing with distinct embeddings."""
    emb_3 = [1.0, 0.0, 0.0]
    emb_1 = [0.0, 1.0, 0.0]
    emb_2 = [0.0, 0.0, 1.0]
    app = ArrayAppearance({7: ([3, 1, 2], np.array([emb_3, emb_1, emb_2]))})
    f, e = app.clean_embeddings(7, 0, 10)
    assert f.tolist() == [1, 2, 3]
    np.testing.assert_allclose(e[0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(e[1], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(e[2], np.array([1.0, 0.0, 0.0]))


def test_array_appearance_validates_frame_embedding_length():
    """Test that ArrayAppearance raises ValueError for mismatched lengths."""
    with pytest.raises(ValueError, match=r"raw_id 7.*different lengths"):
        ArrayAppearance({7: ([1, 2, 3], np.array([[1.0], [2.0]]))})


def test_array_appearance_validates_embedding_2d():
    """Test that ArrayAppearance raises ValueError for non-2D embeddings."""
    with pytest.raises(ValueError, match=r"raw_id 7.*2-D"):
        ArrayAppearance({7: ([1, 2], np.array([1.0, 2.0]))})


def test_array_appearance_validates_embedding_nonempty():
    """Test that ArrayAppearance raises ValueError for empty embeddings."""
    with pytest.raises(ValueError, match=r"raw_id 7.*at least one row"):
        ArrayAppearance({7: ([], np.empty((0, 3)))})


def test_track_embeddings_follow_lineage():
    app = ArrayAppearance({1: (range(10), np.eye(10)), 2: (range(10, 20), np.eye(10))})
    f, e = track_embeddings(app, [[1, 5, 9], [2, 10, 11]])
    assert f.tolist() == [5, 6, 7, 8, 9, 10, 11] and e.shape == (7, 10)


def test_track_embeddings_sorts_out_of_order_lineage():
    """Test that track_embeddings sorts frames even with out-of-order lineage spans."""
    app = ArrayAppearance({
        1: (range(10), np.eye(10)),
        2: (range(10, 20), np.eye(10))
    })
    f, e = track_embeddings(app, [[2, 10, 11], [1, 5, 9]])
    assert f.tolist() == [5, 6, 7, 8, 9, 10, 11]
    assert e.shape == (7, 10)
    np.testing.assert_allclose(e[4], np.eye(10)[9])
    np.testing.assert_allclose(e[5], np.eye(10)[0])
    np.testing.assert_allclose(e[6], np.eye(10)[1])
