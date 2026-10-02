import numpy as np
import pytest

from dnt.refine.crops import CROP_PAD, FrameReader, crop_box


def _frame():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :, 2] = 200  # BGR: red
    return img


def test_crop_is_rgb_and_padded():
    crop = crop_box(_frame(), (50, 20, 40, 60))
    assert crop.dtype == np.uint8 and crop.shape[2] == 3
    assert crop[..., 0].min() == 200 and crop[..., 2].max() == 0  # R first
    # 40 x 60 enlarged by 1.1 -> about 44 x 66
    assert abs(crop.shape[1] - 40 * CROP_PAD) <= 2 and abs(crop.shape[0] - 60 * CROP_PAD) <= 2


def test_crop_is_padded_then_clipped_to_the_frame():
    # (-20, -10, 60, 40) enlarged by 1.1 spans x -23..43 and y -12..32; the frame keeps 0..43, 0..32
    assert crop_box(_frame(), (-20, -10, 60, 40)).shape == (32, 43, 3)
    # the same box away from the edges keeps its full padded size: 66 wide, 44 high
    assert crop_box(_frame(), (60, 30, 60, 40)).shape == (44, 66, 3)


@pytest.mark.parametrize(
    "box",
    [(500, 20, 40, 60), (10, 10, 0, 50), (10, 10, 50, -3), (np.nan, 1, 5, 5), (200, 100, 5, 5)],
)
def test_empty_or_outside_boxes_give_none(box):
    assert crop_box(_frame(), box) is None


def test_reader_yields_requested_frames_in_order(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        got = list(reader.frames([40, 3, 3, 10]))
    assert [f for f, _ in got] == [3, 10, 40]
    assert all(img.shape == (240, 320, 3) for _, img in got)


def test_seeking_and_sequential_reads_agree(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        sequential = {f: img for f, img in reader.frames(range(0, 150))}
    with FrameReader(video) as reader:
        sparse = {f: img for f, img in reader.frames([149, 5, 120])}  # gaps > SEEK_GAP seek
    for f, img in sparse.items():
        assert np.array_equal(img, sequential[f])


def test_a_second_call_may_go_backwards(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        later = dict(reader.frames([60]))
        earlier = dict(reader.frames([7]))
    assert set(later) == {60} and set(earlier) == {7}


def test_a_frame_past_the_end_names_the_frame(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader, pytest.raises(ValueError, match="frame 500"):
        list(reader.frames([500]))


def test_an_unopenable_video_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot open video"):
        FrameReader(tmp_path / "missing.mp4")
