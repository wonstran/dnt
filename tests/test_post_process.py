import pandas as pd

from dnt.track.post_process import interpolate_tracks_rts


def test_track_file_first_row_kept(tmp_path):
    rows = [[0, 7, 10, 20, 30, 20, 0.9, 2, -1, -1],
            [1, 7, 12, 20, 30, 20, 0.9, 2, -1, -1],
            [2, 7, 14, 20, 30, 20, 0.9, 2, -1, -1]]
    path = tmp_path / "t.txt"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    out = interpolate_tracks_rts(track_file=str(path), verbose=False)
    assert (out["frame"] == 0).any()
    assert len(out) == 3


def test_empty_track_file_roundtrip(tmp_path):
    src = tmp_path / "empty.txt"
    src.write_text("")
    dst = tmp_path / "out.txt"
    out = interpolate_tracks_rts(track_file=str(src), output_file=str(dst), verbose=False)
    assert out.empty
    assert dst.read_text() == ""
