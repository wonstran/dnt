from pathlib import Path

from dnt.track.tracker import ByteTrackConfig, Tracker


def test_track_batch_strips_only_trailing_iou(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(Tracker, "track", lambda self, det_file, out_file, video_file=None, **k: written.append(out_file))
    det = tmp_path / "cam_iou_north_iou.txt"
    det.write_text("")
    Tracker(config=ByteTrackConfig()).track_batch([str(det)], ["v.mp4"], output_path=str(tmp_path / "out"))
    assert Path(written[0]).name == "cam_iou_north_track.txt"


def test_track_batch_forwards_verbose_to_track(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        Tracker, "track", lambda self, det_file, out_file, video_file=None, **k: seen.update(k)
    )
    det = tmp_path / "cam_iou.txt"
    det.write_text("")
    Tracker(config=ByteTrackConfig()).track_batch([str(det)], ["v.mp4"], verbose=False)
    assert seen["verbose"] is False


def test_track_verbose_false_suppresses_progress_bar(capsys, synthetic_video, stub_dets):
    video, _ = synthetic_video
    Tracker(config=ByteTrackConfig(), device="cpu").track(str(stub_dets), "", str(video), verbose=False)
    assert capsys.readouterr().err == ""


def test_track_verbose_true_shows_progress_bar(capsys, synthetic_video, stub_dets):
    video, _ = synthetic_video
    Tracker(config=ByteTrackConfig(), device="cpu").track(str(stub_dets), "", str(video), verbose=True)
    assert capsys.readouterr().err != ""


def _track_err(capsys, synthetic_video, stub_dets, **kwargs):
    video, _ = synthetic_video
    Tracker(config=ByteTrackConfig(), device="cpu").track(str(stub_dets), "", str(video), **kwargs)
    return capsys.readouterr().err


def test_track_default_message_hides_file_name(capsys, synthetic_video, stub_dets):
    err = _track_err(capsys, synthetic_video, stub_dets, video_index=1, video_tot=3)
    assert "Tracking 1 of 3:" in err
    assert synthetic_video[0].stem not in err


def test_track_message_none_shows_file_name(capsys, synthetic_video, stub_dets):
    err = _track_err(capsys, synthetic_video, stub_dets, video_index=1, video_tot=3, message=None)
    assert f"Tracking 1 of 3 - {synthetic_video[0].stem}:" in err


def test_track_custom_message_without_batch_index(capsys, synthetic_video, stub_dets):
    assert "Tracking vehicle:" in _track_err(capsys, synthetic_video, stub_dets, message="vehicle")
