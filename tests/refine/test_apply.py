import pytest

from dnt.refine import apply as A
from dnt.refine import io
from dnt.refine.events import Event, EventKind

from ._fixtures import box_rows, table


def _work(*rows):
    return io.to_work(table(*rows)).work


def test_lineage_after_split_keeps_raw_ids():
    w = A.split_track(_work(box_rows(12, range(10, 20), 0.0, 0.0)), 12, 15, 13)
    assert A.lineage(w, 12) == [[12, 10, 14]] and A.lineage(w, 13) == [[12, 15, 19]]


def test_split_tail_id_never_collides_with_sparse_ids():
    w = _work(box_rows(3, range(5), 0.0, 0.0), box_rows(10004, range(5), 50.0, 0.0),
              box_rows(97, range(5), 99.0, 0.0))
    new = A.next_track_id(w)
    assert new == 10005
    w = A.split_track(w, 3, 2, new)
    assert sorted(w["track"].unique()) == [3, 97, 10004, 10005]
    assert (w.loc[w["track"] == 10005, "raw_id"] == 3).all()


def test_drop_and_reclass_with_spans_preserve_index():
    w = _work(box_rows(1, range(10), 0.0, 0.0))
    d = A.drop_rows(w, 1, [[5, 9]])
    assert d["frame"].tolist() == [0, 1, 2, 3, 4] and d.index.tolist() == [0, 1, 2, 3, 4]
    r = A.reclass_rows(w, 1, 3, [[0, 1]])
    assert r["cls"].tolist()[:3] == [3, 3, 0]
    assert A.drop_rows(w, 1).empty


def test_merge_chains_keeps_earliest_id_and_drops_overlap_rows():
    w = _work(box_rows(5, range(0, 10), 0.0, 0.0), box_rows(2, range(9, 20), 20.0, 0.0),
              box_rows(8, range(25, 30), 40.0, 0.0))
    m, rep = A.merge_chains(w, [(5, 2), (2, 8)])
    assert rep == {5: 5, 2: 5, 8: 5}
    assert m["frame"].tolist() == [*range(0, 20), *range(25, 30)]
    assert not m.duplicated(["track", "frame"]).any()


def test_merge_keeps_unique_rows_inside_a_sparse_overlap():
    a = box_rows(1, [*range(0, 9), 11], 0.0, 0.0)  # A: frames 0-8 and 11
    b = box_rows(2, range(9, 13), 0.0, 0.0)  # B: frames 9-12; only frame 11 is shared
    m, _ = A.merge_chains(_work(a, b), [(1, 2)])
    assert m["frame"].tolist() == list(range(0, 13))
    assert m.loc[m["frame"] == 11, "raw_id"].tolist() == [1]  # the duplicate came from B


def test_renumber_is_contiguous_by_first_frame():
    w = _work(box_rows(50, range(5, 9), 0.0, 0.0), box_rows(7, range(0, 3), 0.0, 0.0))
    r, id_map = A.renumber(w)
    assert id_map == {7: 1, 50: 2} and sorted(r["track"].unique()) == [1, 2]


def test_apply_edit_uses_edit_not_proposal():
    w = _work(box_rows(1, range(10), 0.0, 0.0))
    ev = Event.propose(stage="screen", kind=EventKind.DROP, tracks=[1],
                       lineage=[A.lineage(w, 1)], frames=(0, 9),
                       params={"reason": "static", "spans": None}, algo_score=0.6)
    ev.edit = {"kind": "RECLASS", "params": {"new_cls": 1, "spans": None}}
    out = A.apply_edit(w, ev)
    assert len(out) == 10 and (out["cls"] == 1).all()
    ev.edit = None
    with pytest.raises(ValueError, match="no edit"):
        A.apply_edit(w, ev)
