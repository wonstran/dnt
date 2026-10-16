"""ReClass results are pinned, and the shared reading helpers tolerate bad frames."""

import pandas as pd
from reclass_fakes import FakeDetector, FakeReader

from dnt.track import re_class


def _tracks():
    t1 = pd.DataFrame({"frame": [9, 1, 5], "track": 1, "x": 100, "y": 100, "w": 50, "h": 50})
    t2 = pd.DataFrame({
        "frame": [20, 21, 22, 23, 24], "track": 2, "x": 300, "y": 300,
        "w": [10, 20, 30, 40, 50], "h": 50,
    })
    t3 = pd.DataFrame({"frame": [40, 41], "track": 3, "x": 500, "y": 500, "w": [30, 60], "h": 60})
    return pd.concat([t1, t2, t3], ignore_index=True)


def test_reclass_results_are_pinned(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    det = FakeDetector({
        9: [(110, 110, 20, 20, 0.9, 1)],   # inside track 1 at frame 9 (the tie winner)
        1: [(110, 110, 20, 20, 0.9, 36)],  # frame 1 must NOT be sampled with num_frames=1
        20: [(305, 305, 5, 5, 0.9, 36)],   # track 2's smallest box: not among its top 3
        41: [(510, 510, 20, 20, 0.9, 36)],  # track 3's largest box
    })
    monkeypatch.setattr(re_class, "Detector", lambda **kw: det)
    monkeypatch.setattr(re_class, "FrameReader", FakeReader, raising=False)
    rc = re_class.ReClass(num_frames=1, threshold=0.75)
    out = rc.re_classify(_tracks(), str(video), verbose=False)
    assert out.values.tolist() == [[1, 1, 1.0], [2, 0, 0.0], [3, 36, 1.0]]


def test_read_frames_continues_after_a_bad_middle_frame():
    reader = FakeReader("v", bad={5})
    got = [(f, img is None) for f, img in re_class._read_frames(reader, [9, 5, 1, 9])]
    assert got == [(1, False), (5, True), (9, False)]
    assert reader.requests == [[1], [5], [9]]


def test_detect_frames_once_each_in_order():
    det = FakeDetector({1: [(0, 0, 5, 5, 0.5, 2)]})
    by_frame, unread = re_class._detect_frames(det, FakeReader("v", bad={4}), [7, 1, 4, 7])
    assert det.seen == [1, 7]
    assert unread == {4}
    assert sorted(by_frame) == [1, 7]
    assert len(by_frame[1]) == 1 and by_frame[7].empty


def test_dets_for_empty_has_columns():
    out = re_class._dets_for({}, [3])
    assert out.empty and list(out.columns) == re_class._DET_FIELDS
