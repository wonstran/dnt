import numpy as np
import pytest

from dnt.refine.crops import CROP_PAD, SEEK_GAP, FrameReader, crop_box


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


def test_the_padded_edges_are_rounded_to_the_nearest_pixel():
    # 100 * 1.1 / 2 is 55.00000000000001; truncating would start a column early (111 columns, not 110)
    img = np.zeros((100, 400, 3), np.uint8)
    assert crop_box(img, (10, 20, 100, 40)).shape == (44, 110, 3)


def test_a_crop_needs_two_pixels_on_each_axis():
    # 100 x 200 frame; the padded box spans x 198.95..204.45, so one column (199) is left
    assert crop_box(_frame(), (199.2, 50, 5, 5)) is None
    # and with the box 1 px further left, two columns (198, 199) are left
    assert crop_box(_frame(), (198.2, 50, 5, 5)).shape == (5, 2, 3)
    # the same on the y axis: frame height 100
    assert crop_box(_frame(), (50, 99.2, 5, 5)) is None
    assert crop_box(_frame(), (50, 98.2, 5, 5)).shape == (2, 5, 3)


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


class _SpyCap:
    """Wrap a cv2.VideoCapture: count seeks, optionally fail one read after performing it."""

    def __init__(self, cap):
        self._cap = cap
        self.seeks = 0
        self.fail_next_read = False

    def set(self, prop, value):
        self.seeks += 1
        return self._cap.set(prop, value)

    def read(self):
        ok, img = self._cap.read()
        if self.fail_next_read:
            self.fail_next_read = False
            return False, None
        return ok, img

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _spied(reader):
    reader._cap = _SpyCap(reader._cap)
    return reader._cap


def _sequential(video):
    with FrameReader(video) as reader:
        return dict(reader.frames(range(150)))


def test_seek_gap_is_one_hundred_frames():
    assert SEEK_GAP == 100


@pytest.mark.parametrize(
    ("wanted", "seeks"),
    [
        ([0], 0),
        ([100], 0),  # 100 frames skipped from the start: decoded on
        ([101], 1),  # 101 frames skipped: seek
        ([0, 101], 0),  # 100 skipped between them: decoded on
        ([0, 102], 1),  # 101 skipped: seek
        ([5, 6, 7, 8], 0),
        ([5, 120, 149], 1),  # only 5 -> 120 skips more than SEEK_GAP
    ],
)
def test_long_gaps_seek_and_short_ones_decode_on(synthetic_video, wanted, seeks):
    video, _ = synthetic_video
    sequential = _sequential(video)
    with FrameReader(video) as reader:
        spy = _spied(reader)
        got = dict(reader.frames(wanted))
    assert spy.seeks == seeks
    assert set(got) == set(wanted)
    for f, img in got.items():
        assert np.array_equal(img, sequential[f])


def test_a_backwards_jump_seeks(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader:
        spy = _spied(reader)
        list(reader.frames([60]))
        assert spy.seeks == 0
        list(reader.frames([59]))  # 59 < 61: only a seek can get there
        assert spy.seeks == 1


def test_a_second_call_may_go_backwards(synthetic_video):
    video, _ = synthetic_video
    sequential = _sequential(video)
    with FrameReader(video) as reader:
        later = dict(reader.frames([60]))
        earlier = dict(reader.frames([7]))
    assert set(later) == {60} and set(earlier) == {7}
    assert np.array_equal(later[60], sequential[60])
    assert np.array_equal(earlier[7], sequential[7])


def test_a_failed_read_makes_the_next_call_seek(synthetic_video):
    video, _ = synthetic_video
    sequential = _sequential(video)
    with FrameReader(video) as reader:
        spy = _spied(reader)
        spy.fail_next_read = True
        with pytest.raises(ValueError, match="cannot read frame 10"):
            list(reader.frames([10]))
        after = dict(reader.frames([11]))
    assert spy.seeks == 1
    assert np.array_equal(after[11], sequential[11])


def test_a_negative_index_is_a_value_error(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader, pytest.raises(ValueError, match="negative"):
        list(reader.frames([-1, 3]))


def test_a_frame_past_the_end_names_the_frame(synthetic_video):
    video, _ = synthetic_video
    with FrameReader(video) as reader, pytest.raises(ValueError, match="frame 500"):
        list(reader.frames([500]))


def test_an_unopenable_video_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot open video"):
        FrameReader(tmp_path / "missing.mp4")
