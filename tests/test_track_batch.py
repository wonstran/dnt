from pathlib import Path

from dnt.track.tracker import ByteTrackConfig, Tracker


def test_track_batch_strips_only_trailing_iou(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(Tracker, "track", lambda self, det_file, out_file, video_file=None, **k: written.append(out_file))
    det = tmp_path / "cam_iou_north_iou.txt"
    det.write_text("")
    Tracker(config=ByteTrackConfig()).track_batch([str(det)], ["v.mp4"], output_path=str(tmp_path / "out"))
    assert Path(written[0]).name == "cam_iou_north_track.txt"
