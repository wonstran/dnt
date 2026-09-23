import cv2
import pandas as pd
import pytest

from dnt.label import Labeler
from dnt.track import ByteTrackConfig, BoTSORTConfig, Tracker, interpolate_tracks_rts, link_tracklets
from synthetic import GAP, N_FRAMES


@pytest.mark.parametrize("cfg", [ByteTrackConfig(), BoTSORTConfig()], ids=["bytetrack", "botsort"])
def test_cpu_pipeline(cfg, synthetic_video, stub_dets, tmp_path):
    video, _ = synthetic_video
    tracks = Tracker(config=cfg, device="cpu").track(str(stub_dets), str(tmp_path / "t.txt"), str(video))
    assert len(tracks) > 0

    before_gap = tracks[(tracks.frame == GAP.start - 1) & (tracks.y < 60)]
    assert len(before_gap) == 1, "object 1 should be tracked just before the gap"
    obj1 = int(before_gap.track.iloc[0])

    interp = interpolate_tracks_rts(tracks=pd.read_csv(tmp_path / "t.txt", header=None), verbose=False,
                                    output_file=str(tmp_path / "i.txt"))
    filled = interp[(interp.track == obj1) & (interp["interp"] == 1)]
    assert set(GAP) <= set(filled.frame.astype(int))

    link_tracklets(track_file=str(tmp_path / "i.txt"), output_file=str(tmp_path / "l.txt"), verbose=False)
    Labeler().draw_tracks(input_video=str(video), output_video=str(tmp_path / "out.mp4"),
                          track_file=str(tmp_path / "l.txt"), label_class=True, verbose=False)
    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_FRAMES
    finally:
        cap.release()
