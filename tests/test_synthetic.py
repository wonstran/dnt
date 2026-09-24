import cv2
import pandas as pd
from synthetic import FPS, GAP, N_FRAMES, OBJ3_START, StubDetector

DET_FIELDS = ["frame", "res", "x", "y", "w", "h", "conf", "class"]


def test_video_has_expected_frames(synthetic_video):
    video, _ = synthetic_video
    cap = cv2.VideoCapture(str(video))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_FRAMES
        assert round(cap.get(cv2.CAP_PROP_FPS)) == FPS
    finally:
        cap.release()


def test_ground_truth_gap_and_late_object(synthetic_video):
    _, truth = synthetic_video
    obj1 = truth.boxes[truth.boxes.obj == 1]
    assert not set(obj1.frame) & set(GAP)
    obj3 = truth.boxes[truth.boxes.obj == 3]
    assert obj3.frame.min() == OBJ3_START


def test_stub_detector_schema_and_file(synthetic_video, tmp_path):
    video, truth = synthetic_video
    out = tmp_path / "d_iou.txt"
    df = StubDetector(truth).detect(video, iou_file=out)
    assert list(df.columns) == DET_FIELDS
    assert (df["conf"] == 0.9).all() and (df["class"] == 2).all()
    on_disk = pd.read_csv(out, header=None)
    assert on_disk.shape == df.shape
    assert 0.9 < len(df) / len(truth.boxes) <= 1.0


def test_stub_detector_low_conf_alternates(synthetic_video):
    video, truth = synthetic_video
    df = StubDetector(truth, low_conf=0.3, drop_rate=0.0).detect(video)
    assert set(df.loc[df.frame % 2 == 1, "conf"]) == {0.3}
    assert set(df.loc[df.frame % 2 == 0, "conf"]) == {0.9}
