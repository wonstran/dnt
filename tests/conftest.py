import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "support"))

from synthetic import GroundTruth, StubDetector, make_synthetic_video  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> tuple[Path, GroundTruth]:
    path = tmp_path_factory.mktemp("synthetic") / "scene.mp4"
    truth = make_synthetic_video(path)
    return path, truth


@pytest.fixture(scope="session")
def stub_dets(synthetic_video, tmp_path_factory) -> Path:
    video, truth = synthetic_video
    out = tmp_path_factory.mktemp("dets") / "scene_iou.txt"
    StubDetector(truth).detect(video, iou_file=out)
    return out
