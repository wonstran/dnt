import cv2
import pandas as pd
import pytest
from synthetic import StubDetector

from dnt.label import labeler as lab


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


@pytest.fixture
def pbar_desc_spy(monkeypatch):
    captured = []
    real_tqdm = lab.tqdm

    class Spy(real_tqdm):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs.get("desc"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lab, "tqdm", Spy)
    return captured


def _write_one_track(tmp_path):
    tracks = tmp_path / "t.txt"
    pd.DataFrame([[0, 1, 10, 10, 20, 20, 0.9, 2, -1, -1]]).to_csv(tracks, index=False, header=False)
    return tracks


def test_draw_tracks_message_default_omits_filename(synthetic_video, tmp_path, pbar_desc_spy):
    video, _ = synthetic_video
    tracks = _write_one_track(tmp_path)
    lab.Labeler().draw_tracks(input_video=str(video), output_video="", track_file=str(tracks), verbose=True)
    assert pbar_desc_spy == ["Generating labels"]


def test_draw_tracks_message_none_shows_input_video(synthetic_video, tmp_path, pbar_desc_spy):
    video, _ = synthetic_video
    tracks = _write_one_track(tmp_path)
    lab.Labeler().draw_tracks(
        input_video=str(video), output_video="", track_file=str(tracks), verbose=True, message=None
    )
    assert pbar_desc_spy == [f"Generating labels {video}"]


def test_draw_tracks_message_custom_string(synthetic_video, tmp_path, pbar_desc_spy):
    video, _ = synthetic_video
    tracks = _write_one_track(tmp_path)
    lab.Labeler().draw_tracks(
        input_video=str(video), output_video="", track_file=str(tracks), verbose=True, message="clip A"
    )
    assert pbar_desc_spy == ["Generating labels clip A"]


def test_draw_tracks_message_in_batch_uses_dash_separator(synthetic_video, tmp_path, pbar_desc_spy):
    video, _ = synthetic_video
    tracks = _write_one_track(tmp_path)
    lab.Labeler().draw_tracks(
        input_video=str(video), output_video="", track_file=str(tracks), verbose=True,
        video_index=1, video_tot=3, message="clip A",
    )
    assert pbar_desc_spy == ["Generating labels 1 of 3 - clip A"]


def test_draw_tracks_compress_message_ignores_message(synthetic_video, tmp_path, pbar_desc_spy):
    video, _ = synthetic_video
    tracks = _write_one_track(tmp_path)
    lab.Labeler(compress_message=True).draw_tracks(
        input_video=str(video), output_video="", track_file=str(tracks), verbose=True, message="clip A"
    )
    assert pbar_desc_spy == ["Generating labels"]


def test_export_track_frames_without_bbox_writes_files(synthetic_video, tmp_path):
    video, _ = synthetic_video
    tracks = pd.DataFrame([[0, 1, 10, 10, 20, 20, 0.9, 2, -1, -1],
                           [1, 1, 12, 10, 20, 20, 0.9, 2, -1, -1]])
    lab.Labeler.export_track_frames(str(video), tracks, str(tmp_path / "frames"), bbox=False)
    # naming unchanged from 0.3.2.4 (iterrows yields float frame numbers)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["1_0.0.jpg", "1_1.0.jpg"]


def test_ffmpeg_start_reports_a_missing_binary(monkeypatch):
    monkeypatch.setattr(lab.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="ffmpeg was not found on PATH"):
        lab._start_ffmpeg(["ffmpeg", "-y"])


def test_imwrite_handles_non_ascii_paths_and_roundtrips(tmp_path):
    import numpy as np

    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    path = tmp_path / "ünï-1.jpg"
    lab._imwrite(str(path), img)
    assert path.exists()
    assert cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR).shape == img.shape


def _tiny_video(path, n=5):
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(n):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()


@pytest.mark.parametrize("method", ["draw_tracks", "draw_dets"])
def test_an_empty_track_or_det_file_gives_an_unlabelled_video(tmp_path, method):
    src, out, empty = tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path / "empty.txt"
    _tiny_video(src)
    empty.write_text("")
    kw = {"track_file": str(empty)} if method == "draw_tracks" else {"det_file": str(empty)}
    df = getattr(lab.Labeler(method=lab.LabelMethod.OPENCV), method)(
        input_video=str(src), output_video=str(out), **kw
    )
    assert df.empty and out.exists()
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()
