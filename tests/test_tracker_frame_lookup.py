import numpy as np
import pytest

from dnt.track import tracker as tracker_module
from dnt.track.tracker import ByteTrackConfig, Tracker

# frame, res, x, y, w, h, conf, class -- rows deliberately out of frame order
DETS = np.array([
    [4, -1, 50, 60, 10, 20, 0.9, 0],
    [2, -1, 10, 20, 30, 40, 0.8, 2],
    [4, -1, 5, 6, 7, 8, 0.7, 1],
    [2, -1, 100, 110, 5, 6, 0.6, 0],
])
# frame -> x1, y1, x2, y2, conf, class, in the order the rows appear in DETS
WANT = {
    0: [],
    2: [[10, 20, 40, 60, 0.8, 2], [100, 110, 105, 116, 0.6, 0]],
    3: [],
    4: [[50, 60, 60, 80, 0.9, 0], [5, 6, 12, 14, 0.7, 1]],
    99: [],
}


def _want(frame_id: int) -> np.ndarray:
    return np.array(WANT[frame_id], dtype=float).reshape(-1, 6)


@pytest.mark.parametrize("frame_id", sorted(WANT))
def test_frame_detections_returns_xyxy_rows_in_file_order(frame_id):
    got = tracker_module._FrameDetections(DETS).xyxy(frame_id)
    assert got.dtype == float
    assert got.shape == _want(frame_id).shape
    np.testing.assert_array_equal(got, _want(frame_id))


def test_frame_detections_repeated_lookup_is_stable():
    index = tracker_module._FrameDetections(DETS)
    index.xyxy(2)[:] = -1  # a caller (BoxMOT) may modify what it is handed
    np.testing.assert_array_equal(index.xyxy(2), _want(2))


def test_track_hands_tracker_each_frames_detections_in_file_order(monkeypatch, tmp_path, synthetic_video):
    seen = []

    class Recorder:
        def update(self, dets, frame):
            seen.append(dets.copy())
            return np.empty((0, 7))

    monkeypatch.setattr(Tracker, "_build_boxmot_tracker", staticmethod(lambda *a, **k: Recorder()))
    det_file = tmp_path / "unsorted_iou.txt"
    np.savetxt(det_file, DETS, delimiter=",")
    video, _ = synthetic_video

    Tracker(config=ByteTrackConfig(), device="cpu").track(str(det_file), "", str(video), verbose=False)

    assert len(seen) == 3  # frames 2, 3, 4
    for got, frame_id in zip(seen, (2, 3, 4), strict=True):
        assert got.dtype == float
        assert got.shape == _want(frame_id).shape
        np.testing.assert_array_equal(got, _want(frame_id))
