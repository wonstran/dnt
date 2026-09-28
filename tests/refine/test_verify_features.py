import numpy as np

from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.features import ArrayAppearance, track_embeddings
from dnt.refine.verify import Band, band_route, route_without_vlm


def _ev(kind, score, params):
    return Event.propose(stage="screen", kind=kind, tracks=[1], lineage=[[[1, 0, 9]]],
                         frames=(0, 9), params=params, algo_score=score)


def test_band_route():
    b = Band(0.85, 0.40)
    assert band_route(0.9, b) is Decision.AUTO_ACCEPT
    assert band_route(0.3, b) is Decision.AUTO_REJECT
    assert band_route(0.6, b) is None


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


def test_array_appearance_normalizes_and_filters():
    app = ArrayAppearance({7: ([3, 1, 2], np.array([[0, 3.0], [2.0, 0], [0, 5.0]]))})
    f, e = app.clean_embeddings(7, 1, 2)
    assert f.tolist() == [1, 2]
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)
    assert app.clean_embeddings(99, 0, 10)[0].size == 0


def test_track_embeddings_follow_lineage():
    app = ArrayAppearance({1: (range(10), np.eye(10)), 2: (range(10, 20), np.eye(10))})
    f, e = track_embeddings(app, [[1, 5, 9], [2, 10, 11]])
    assert f.tolist() == [5, 6, 7, 8, 9, 10, 11] and e.shape == (7, 10)
