import hashlib
import logging

import pandas as pd
import pytest

from dnt.refine import io

from ._fixtures import box_rows, table


def _write(tmp_path, df, name="t.txt"):
    p = tmp_path / name
    df.to_csv(p, index=False, header=False)
    return p


def test_read_tracks_removes_filled_rows_and_builds_work_table(tmp_path):
    df = table(box_rows(7, range(5), 10.0, 20.0))
    df.loc[2, "r3"] = 1
    tin = io.read_tracks(_write(tmp_path, df))
    assert tin.n_filled_removed == 1
    assert list(tin.work.columns) == io.WORK_COLUMNS
    assert tin.work["frame"].tolist() == [0, 1, 3, 4]
    assert (tin.work["raw_id"] == 7).all() and (tin.work["interp"] == 0).all()


def test_to_work_recognizes_a_named_interp_column():
    df = pd.DataFrame({
        "frame": [0, 1, 2], "track": [4, 4, 4], "x": [1.0] * 3, "y": [1.0] * 3,
        "w": [5.0] * 3, "h": [9.0] * 3, "score": [0.9] * 3, "cls": [0] * 3,
        "interp": [0, 1, 0], "r4": [-1] * 3,
    })  # the named layout refine and interpolate_tracks_rts write (no r3 column)
    tin = io.to_work(df)
    assert tin.n_filled_removed == 1 and tin.work["frame"].tolist() == [0, 2]


def test_duplicate_track_frame_rows_keep_first(tmp_path, caplog):
    df = table(box_rows(1, [0, 1, 1, 2], 10.0, 20.0))
    df.loc[2, "x"] = 999.0
    with caplog.at_level(logging.WARNING):
        tin = io.read_tracks(_write(tmp_path, df))
    assert tin.n_duplicates_removed == 1
    assert 999.0 not in tin.work["x"].tolist()
    assert "duplicate" in caplog.text


def test_read_tracks_errors(tmp_path):
    (tmp_path / "few.txt").write_text("1,2,3,4,5\n")
    with pytest.raises(ValueError, match="at least 6 columns"):
        io.read_tracks(tmp_path / "few.txt")
    (tmp_path / "bad.txt").write_text("0,1,1,1,1,1,0.9,0,-1,-1\n1,1,abc,1,1,1,0.9,0,-1,-1\n")
    with pytest.raises(ValueError, match="line 2"):
        io.read_tracks(tmp_path / "bad.txt")


@pytest.mark.parametrize(("col", "bad"), [(6, "high"), (7, "car"), (8, "yes"), (9, "")])
def test_read_tracks_rejects_nonnumeric_score_cls_flag_and_r4(tmp_path, col, bad):
    fields = ["2", "1", "1", "1", "1", "1", "0.9", "0", "-1", "-1"]
    fields[col] = bad
    rows = ["0,1,1,1,1,1,0.9,0,-1,-1", "1,1,1,1,1,1,0.9,0,1,-1", ",".join(fields)]
    (tmp_path / "t.txt").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="line 3"):
        io.read_tracks(tmp_path / "t.txt")


def test_valid_filled_flag_is_still_removed(tmp_path):
    rows = ["0,1,1,1,1,1,0.9,0,-1,-1", "1,1,1,1,1,1,0.9,0,1,-1", "2,1,1,1,1,1,0.9,0,0,-1"]
    (tmp_path / "t.txt").write_text("\n".join(rows) + "\n")
    tin = io.read_tracks(tmp_path / "t.txt")
    assert tin.n_filled_removed == 1 and tin.work["frame"].tolist() == [0, 2]


def test_read_context_rejects_nonnumeric_class(tmp_path):
    (tmp_path / "d.txt").write_text("0,-1,1,2,3,4,0.9,car\n")
    with pytest.raises(ValueError, match="line 1"):
        io.read_context(tmp_path / "d.txt")


def test_read_tracks_empty_file(tmp_path):
    (tmp_path / "e.txt").write_text("")
    tin = io.read_tracks(tmp_path / "e.txt")
    assert tin.work.empty and list(tin.work.columns) == io.WORK_COLUMNS


def test_read_mot_sets_class(tmp_path):
    (tmp_path / "m.txt").write_text("1,4,10,20,30,60,0.8,-1,-1,-1\n2,4,11,20,30,60,0.7,-1,-1,-1\n")
    tin = io.read_tracks(tmp_path / "m.txt", fmt="mot", class_id=0)
    assert tin.work["cls"].tolist() == [0, 0] and tin.work["score"].tolist() == [0.8, 0.7]


def test_read_context_detects_format(tmp_path):
    (tmp_path / "d.txt").write_text("0,-1,1,2,3,4,0.9,2\n")
    (tmp_path / "t.txt").write_text("0,5,1,2,3,4,0.9,7,-1,-1\n")
    (tmp_path / "x.txt").write_text("0,5,1,2,3,4,0.9\n")
    d, fd = io.read_context(tmp_path / "d.txt")
    t, ft = io.read_context(tmp_path / "t.txt")
    assert (fd, d["track"].tolist(), d["cls"].tolist()) == ("dets", [-1], [2])
    assert (ft, t["track"].tolist(), t["cls"].tolist()) == ("tracks", [5], [7])
    with pytest.raises(ValueError, match="7 columns"):
        io.read_context(tmp_path / "x.txt")


def test_write_tracks_format(tmp_path):
    work = io.read_tracks(_write(tmp_path, table(box_rows(2, [1, 0], 10.4, 20.6)))).work
    out = tmp_path / "o.txt"
    io.write_tracks(work, out)
    lines = out.read_text().splitlines()
    assert lines == ["0,2,10,21,30,60,0.9,0,0,-1", "1,2,10,21,30,60,0.9,0,0,-1"]
    io.write_tracks(io.empty_work(), tmp_path / "empty.txt")
    assert (tmp_path / "empty.txt").read_text() == ""


def test_sha256_and_video_info(tmp_path, synthetic_video):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert io.sha256_file(p) == hashlib.sha256(b"abc" * 1000).hexdigest()
    video, _ = synthetic_video
    info = io.video_info(video)
    assert info == {"fps": pytest.approx(25.0), "frame_count": 150, "width": 320, "height": 240}
    fp = io.video_fingerprint(video, info["frame_count"])
    assert fp["sha256"] == io.sha256_file(video) and fp["frame_count"] == 150


def test_read_tracks_reports_true_file_line_with_blank_lines(tmp_path):
    """Blank lines should not affect the reported line number."""
    (tmp_path / "blank.txt").write_text("\n0,1,1,1,1,1,0.9,0,-1,-1\n\n0,1,abc,1,1,1,0.9,0,-1,-1\n")
    with pytest.raises(ValueError, match="line 4"):
        io.read_tracks(tmp_path / "blank.txt")


def test_read_tracks_names_file_for_ragged_rows(tmp_path):
    """ParserError on ragged rows should name the file."""
    (tmp_path / "ragged.txt").write_text("0,1,1,1,1,1,0.9,0,-1,-1\n1,1,1,1,1,1,0.9,0,-1,-1,extra\n")
    with pytest.raises(ValueError, match=r"ragged\.txt"):
        io.read_tracks(tmp_path / "ragged.txt")


def test_read_tracks_only_blank_lines(tmp_path):
    """A file with only blank lines should read as an empty work table."""
    (tmp_path / "blank_only.txt").write_text("\n\n\n")
    tin = io.read_tracks(tmp_path / "blank_only.txt")
    assert tin.work.empty and list(tin.work.columns) == io.WORK_COLUMNS


def test_read_tracks_ragged_true_line_with_blanks_11_fields(tmp_path):
    """Ragged row after blanks should report true file line number."""
    (tmp_path / "ragged2.txt").write_text("\n\n0,1,1,1,1,1,0.9,0,-1,-1\n0,1,1,1,1,1,0.9,0,-1,-1,extra\n")
    with pytest.raises(ValueError, match=r"ragged2\.txt.*line 4"):
        io.read_tracks(tmp_path / "ragged2.txt")


def test_read_tracks_ragged_true_line_with_blanks_short_first(tmp_path):
    """Field count mismatch after blanks should report true file line number."""
    (tmp_path / "ragged3.txt").write_text("\n\n0,1,1,1,1,1\n0,1,1,1,1,1,0.9,0,-1,-1\n")
    with pytest.raises(ValueError, match=r"line 4"):
        io.read_tracks(tmp_path / "ragged3.txt")


# ---- final review M1: non-finite and non-integral values are rejected with file and line -------


@pytest.mark.parametrize(("col", "bad"), [(0, "inf"), (1, "-inf"), (2, "inf"), (5, "nan"),
                                          (6, "inf"), (7, "NaN"), (8, "inf"), (9, "-inf")])
def test_read_tracks_rejects_non_finite_values(tmp_path, col, bad):
    fields = ["2", "1", "1", "1", "1", "1", "0.9", "0", "-1", "-1"]
    fields[col] = bad
    rows = ["0,1,1,1,1,1,0.9,0,-1,-1", "", "1,1,1,1,1,1,0.9,0,-1,-1", ",".join(fields)]
    path = tmp_path / "t.txt"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=r"t\.txt: non-(finite|numeric) value on line 4"):
        io.read_tracks(path)


@pytest.mark.parametrize("fmt", ["dnt", "mot"])
@pytest.mark.parametrize(("col", "bad"), [(0, "2.5"), (1, "1.5")])
def test_read_tracks_rejects_non_integer_frame_and_track(tmp_path, fmt, col, bad):
    # truncating 2.5 to 2 would re-create the (track, frame) duplicate removed before it
    fields = ["2", "1", "1", "1", "1", "1", "0.9", "0", "-1", "-1"]
    fields[col] = bad
    rows = ["2,1,1,1,1,1,0.9,0,-1,-1", ",".join(fields)]
    path = tmp_path / "t.txt"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=r"t\.txt: non-integer frame or track value on line 2"):
        io.read_tracks(path, fmt=fmt)
    fields[col] = "3.0"  # an integral float is fine
    path.write_text("\n".join([rows[0], ",".join(fields)]) + "\n")
    assert len(io.read_tracks(path, fmt=fmt).work) == 2


def test_read_context_rejects_non_finite_and_non_integer_frames(tmp_path):
    for text, what in (("0,-1,1,2,inf,4,0.9,2\n", "non-finite"),
                       ("0.5,-1,1,2,3,4,0.9,2\n", "non-integer frame")):
        (tmp_path / "d.txt").write_text(text)
        with pytest.raises(ValueError, match=rf"d\.txt: {what}.* line 1"):
            io.read_context(tmp_path / "d.txt")


def test_refine_rejects_an_inf_box_before_writing_anything(tmp_path):
    from dnt.refine import TrackRefiner

    path = tmp_path / "t.txt"
    path.write_text("0,1,1,1,1,1,0.9,0,-1,-1\n1,1,1,1,inf,1,0.9,0,-1,-1\n")
    with pytest.raises(ValueError, match="non-finite value on line 2"):
        TrackRefiner().refine(path, tmp_path / "o.txt", fps=10, verbose=False)
    assert not (tmp_path / "o.txt").exists() and not (tmp_path / "o.ledger.jsonl").exists()
