import cv2
import pandas as pd
import pytest

from dnt.label import labeler as lab
from synthetic import StubDetector


@pytest.fixture
def capture_spy(monkeypatch):
    real = cv2.VideoCapture
    made = []

    class Spy:
        def __init__(self, *args):
            self._cap = real(*args)
            self.released = False
            made.append(self)

        def __getattr__(self, name):
            return getattr(self._cap, name)

        def release(self):
            self.released = True
            self._cap.release()

    monkeypatch.setattr(lab.cv2, "VideoCapture", Spy)
    return made


def test_draw_dets_accepts_detector_dataframe(synthetic_video, capture_spy):
    video, truth = synthetic_video
    named = StubDetector(truth).detect(video)
    df = lab.Labeler().draw_dets(input_video=str(video), output_video="", dets=named)
    assert len(df) == len(named)
    assert all(c.released for c in capture_spy)


def test_draw_tracks_releases_capture(synthetic_video, tmp_path, capture_spy):
    video, _ = synthetic_video
    tracks = tmp_path / "t.txt"
    pd.DataFrame([[0, 1, 10, 10, 20, 20, 0.9, 2, -1, -1]]).to_csv(tracks, index=False, header=False)
    lab.Labeler().draw_tracks(input_video=str(video), output_video="", track_file=str(tracks), verbose=False)
    assert capture_spy and all(c.released for c in capture_spy)


def test_export_track_frames_without_bbox_writes_files(synthetic_video, tmp_path):
    video, _ = synthetic_video
    tracks = pd.DataFrame([[0, 1, 10, 10, 20, 20, 0.9, 2, -1, -1],
                           [1, 1, 12, 10, 20, 20, 0.9, 2, -1, -1]])
    lab.Labeler.export_track_frames(str(video), tracks, str(tmp_path / "frames"), bbox=False)
    # naming unchanged from 0.3.2.4 (iterrows yields float frame numbers)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["1_0.0.jpg", "1_1.0.jpg"]
