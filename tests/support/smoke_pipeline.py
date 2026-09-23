"""S1 smoke pipeline: run with an INSTALLED dnt, outside the repository (spec §5.3)."""

import sys
import sysconfig
from pathlib import Path

import cv2
import pandas as pd

import dnt

site = Path(sysconfig.get_paths()["purelib"]).resolve()
assert Path(dnt.__file__).resolve().is_relative_to(site), f"dnt imported from {dnt.__file__}, not {site}"

from synthetic import GAP, N_FRAMES, StubDetector, make_synthetic_video  # noqa: E402

from dnt.label import Labeler  # noqa: E402
from dnt.track import ByteTrackConfig, Tracker, interpolate_tracks_rts, link_tracklets  # noqa: E402

work = Path(sys.argv[1])
video = work / "scene.mp4"
truth = make_synthetic_video(video)
dets = work / "scene_iou.txt"
StubDetector(truth).detect(video, iou_file=dets)

tracks_file = work / "tracks.txt"
Tracker(config=ByteTrackConfig(), device="cpu").track(str(dets), str(tracks_file), str(video))
assert len(pd.read_csv(tracks_file, header=None)) > 0

interp_file = work / "interp.txt"
interp = interpolate_tracks_rts(track_file=str(tracks_file), output_file=str(interp_file), verbose=False)
assert set(GAP) <= set(interp.loc[interp["interp"] == 1, "frame"].astype(int)), "gap not filled"

linked_file = work / "linked.txt"
link_tracklets(track_file=str(interp_file), output_file=str(linked_file), verbose=False)

labeled = work / "labeled.mp4"
Labeler().draw_tracks(input_video=str(video), output_video=str(labeled), track_file=str(linked_file),
                      label_class=True, verbose=False)
cap = cv2.VideoCapture(str(labeled))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
assert n == N_FRAMES, n

# Bundled weights: present in the installed package, unmodified, and used without any download.
import hashlib  # noqa: E402

import dnt.detect.signal.detector as sig  # noqa: E402

pkg = Path(dnt.__file__).parent
osnet = pkg / "track" / "reid_weights" / "osnet_x1_0_msmt17.pt"
ped = pkg / "detect" / "signal" / "weights" / "ped_signal.pt"
for path, want in ((osnet, sys.argv[2]), (ped, sys.argv[3])):
    assert path.is_file(), f"bundled weight missing from the installed wheel: {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == want, f"bundled weight differs: {path}"
osnet_mtime = osnet.stat().st_mtime_ns

# default Tracker() is BoT-SORT with ReID: exercises the bundled OSNet weight
bot_file = work / "tracks_botsort.txt"
Tracker(device="cpu").track(str(dets), str(bot_file), str(video))
assert len(pd.read_csv(bot_file, header=None)) > 0
assert osnet.stat().st_mtime_ns == osnet_mtime, "OSNet weight was re-downloaded/overwritten"


def _no_download(*args, **kwargs):
    raise AssertionError(f"SignalDetector tried to download {args!r} instead of using the bundled weight")


sig.download_file = _no_download
sig.SignalDetector(det_zones=[(0, 0, 32, 32)], device="cpu")
print("SMOKE OK")
