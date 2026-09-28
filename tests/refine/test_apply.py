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


def test_empty_spans_is_noop():
    w = _work(box_rows(1, range(10), 0.0, 0.0))
    d = A.drop_rows(w, 1, [])
    assert len(d) == 10
    assert d["frame"].tolist() == list(range(10))
    r = A.reclass_rows(w, 1, 3, [])
    assert (r["cls"] == 0).all()
    r_tuple = A.reclass_rows(w, 1, 3, ((2, 4),))
    assert r_tuple["cls"].tolist()[:6] == [0, 0, 3, 3, 3, 0]
    import numpy as np
    r_np = A.reclass_rows(w, 1, 3, np.array([[1, 3]]))
    assert r_np["cls"].tolist()[:5] == [0, 3, 3, 3, 0]


def test_index_preservation_with_nonmonotonic_index():
    w = _work(box_rows(1, range(5), 0.0, 0.0), box_rows(2, range(3, 8), 10.0, 0.0))
    original_index = (w.index * 3 + 100)[::-1]
    w.index = original_index
    d = A.drop_rows(w, 1, [[1, 3]])
    assert len(d) == 7
    surviving_mask = ~((w["track"] == 1) & (w["frame"].isin([1, 2, 3])))
    expected_labels = original_index[surviving_mask]
    assert list(d.index) == list(expected_labels)
    w2 = _work(box_rows(1, range(5), 0.0, 0.0))
    w2.index = (w2.index * 2 + 50)[::-1]
    r = A.reclass_rows(w2, 1, 5, [[1, 2]])
    assert r["cls"].tolist() == [0, 5, 5, 0, 0]
    assert list(r.index) == list(w2.index)
    w3 = _work(box_rows(3, range(4), 0.0, 0.0))
    w3.index = (w3.index * 5 + 200)[::-1]
    s = A.split_track(w3, 3, 2, 10)
    assert list(s.index) == list(w3.index)


def test_apply_edit_split_and_drop_reclass_coverage():
    w = _work(box_rows(1, range(10), 0.0, 0.0), box_rows(2, range(5, 8), 10.0, 0.0))
    ev_split = Event.propose(stage="screen", kind=EventKind.SPLIT, tracks=[1],
                             lineage=[A.lineage(w, 1)], frames=(0, 9),
                             params={"cut_frame": 5}, algo_score=0.8)
    ev_split.edit = {"kind": "SPLIT", "params": {"cut_frame": 5}}
    result = A.apply_edit(w, ev_split)
    assert (result.loc[result["track"] == 1, "frame"] < 5).all()
    assert result[result["track"] != 1].index[0] >= 0
    ev_split_newid = Event.propose(stage="screen", kind=EventKind.SPLIT, tracks=[1],
                                   lineage=[A.lineage(w, 1)], frames=(0, 9),
                                   params={"cut_frame": 5}, algo_score=0.8)
    ev_split_newid.edit = {"kind": "SPLIT", "params": {"cut_frame": 5}}
    result2 = A.apply_edit(w, ev_split_newid, new_id=50)
    assert 50 in result2["track"].values
    ev_drop = Event.propose(stage="screen", kind=EventKind.DROP, tracks=[2],
                            lineage=[A.lineage(w, 2)], frames=(5, 7),
                            params={"reason": "test", "spans": [[5, 6]]}, algo_score=0.7)
    ev_drop.edit = {"kind": "DROP", "params": {"spans": [[5, 6]]}}
    result3 = A.apply_edit(w, ev_drop)
    assert len(result3.loc[result3["track"] == 2]) == 1
    ev_reclass = Event.propose(stage="screen", kind=EventKind.RECLASS, tracks=[1],
                               lineage=[A.lineage(w, 1)], frames=(0, 9),
                               params={"new_cls": 7, "spans": None}, algo_score=0.6)
    ev_reclass.edit = {"kind": "RECLASS", "params": {"new_cls": 7, "spans": None}}
    result4 = A.apply_edit(w, ev_reclass)
    assert (result4.loc[result4["track"] == 1, "cls"] == 7).all()
    ev_bad = Event.propose(stage="screen", kind=EventKind.RECLASS, tracks=[1],
                           lineage=[A.lineage(w, 1)], frames=(0, 9),
                           params={"spans": None}, algo_score=0.6)
    ev_bad.edit = {"kind": "RECLASS", "params": {"spans": None}}
    with pytest.raises(ValueError, match="new_cls"):
        A.apply_edit(w, ev_bad)
    ev_stage = Event.propose(stage="screen", kind=EventKind.LINK, tracks=[1, 2],
                             lineage=[A.lineage(w, 1), A.lineage(w, 2)], frames=(0, 9),
                             params={"gap": 3}, algo_score=0.5)
    ev_stage.edit = {"kind": "LINK", "params": {"gap": 3}}
    with pytest.raises(ValueError, match="stage level"):
        A.apply_edit(w, ev_stage)
    ev_no_id = Event.propose(stage="screen", kind=EventKind.DROP, tracks=[1],
                             lineage=[A.lineage(w, 1)], frames=(0, 9),
                             params={"reason": "test", "spans": None}, algo_score=0.6)
    ev_no_id.id = ""
    ev_no_id.edit = {"kind": "DROP", "params": {"spans": None}}
    result5 = A.apply_edit(w, ev_no_id)
    assert 1 not in result5["track"].values


def test_split_track_rejects_existing_new_id():
    w = _work(box_rows(1, range(5), 0.0, 0.0), box_rows(2, range(5), 10.0, 0.0))
    with pytest.raises(ValueError, match="new_id 2 already exists"):
        A.split_track(w, 1, 2, 2)
    with pytest.raises(ValueError, match="new_id 1 already exists"):
        A.split_track(w, 1, 2, 1)
