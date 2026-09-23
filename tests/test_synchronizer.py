import numpy as np
import pytest

from dnt.shared import synhcro

FPS = {"v0": 25.0, "v1": 30.0, "v2": 25.0}
FRAMES = {"v0": 100, "v1": 60, "v2": 10}
R = 1_000_000


@pytest.fixture(autouse=True)
def fake_video_info(monkeypatch):
    monkeypatch.setattr(synhcro.Detector, "get_fps", staticmethod(lambda v: FPS[v]))
    monkeypatch.setattr(synhcro.Detector, "get_frames", staticmethod(lambda v: FRAMES[v]))


def _t(df, video, frame):
    return int(df[(df.video == video) & (df.frame == frame)].unix_time.iloc[0])


def _sync(videos, offsets):
    return synhcro.Synchronizer(videos, ref_frame=25, ref_time=R, offsets=offsets).process()


def test_worked_example_zero_offsets():
    df = _sync(["v0", "v1"], [0, 0])
    assert (_t(df, "v0", 25), _t(df, "v0", 99), _t(df, "v1", 0)) == (1_000_000, 1_002_960, 1_003_000)


def test_gap_offset():
    assert _t(_sync(["v0", "v1"], [0, 5]), "v1", 0) == 1_003_167


def test_overlap_offset():
    assert _t(_sync(["v0", "v1"], [0, -3]), "v1", 0) == 1_002_900


def test_first_offset_must_be_zero():
    with pytest.raises(ValueError, match=r"offsets\[0\] must be 0"):
        _sync(["v0", "v1"], [2, 0])


def test_non_integer_offset_rejected():
    with pytest.raises(ValueError, match="integers"):
        _sync(["v0", "v1"], [0, 1.5])


@pytest.mark.parametrize("offsets", [None, [0]])
def test_single_video_identical_to_0324(offsets):
    df = synhcro.Synchronizer(["v0"], ref_frame=25, ref_time=R, offsets=offsets).process()
    old = np.round(R + (np.arange(100) - 25) * (1000.0 / 25.0)).astype(np.int64)
    assert (df.unix_time.to_numpy() == old).all()


def test_three_videos_no_drift():
    df = _sync(["v0", "v1", "v2"], [0, 0, 0])
    assert _t(df, "v2", 0) == 1_005_000  # 1_003_000 + 60 frames * 33.333... ms, unrounded carry
