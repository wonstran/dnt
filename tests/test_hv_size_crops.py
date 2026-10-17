import math
import sys
from pathlib import Path

import cv2
import pandas as pd
import pytest
from reclass_fakes import FakeReader, vehicle_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hv_size_crops as hv  # noqa: E402


def _tracks():
    big = vehicle_rows(1, range(10), (100, 100, 300, 150))  # size 300
    big.loc[big.frame == 4, ["w", "h"]] = (320, 160)  # its largest real box, frame 4
    small = vehicle_rows(2, range(10), (500, 100, 120, 80))  # size 120
    filled = vehicle_rows(3, range(10), (800, 100, 200, 100))  # size 200 ...
    filled.loc[filled.frame == 7, ["w", "h"]] = (210, 105)  # its largest real box, frame 7
    filled.loc[filled.frame == 9, ["w", "h", "r3"]] = (900, 600, 1)  # ... a filled giant is ignored
    short = vehicle_rows(4, range(2), (100, 400, 900, 500))  # 2 real rows < hv_min_count
    return pd.concat([big, small, filled, short], ignore_index=True)


def test_select_sorts_by_size_and_uses_largest_real_box():
    sel = hv.select_tracks(_tracks(), hv_min_count=3)
    assert sel.track.tolist() == [1, 3, 2]
    assert sel["size"].tolist() == [300, 200, 120]
    assert sel.loc[0, ["frame", "w", "h"]].tolist() == [4, 320, 160]
    assert sel.loc[1, "frame"] == 7  # the larger filled row (frame 9) is never the crop


def test_select_min_size_and_limit():
    assert hv.select_tracks(_tracks(), min_size=150).track.tolist() == [1, 3]
    assert hv.select_tracks(_tracks(), limit=1).track.tolist() == [1]
    assert hv.select_tracks(_tracks(), hv_min_count=2).track.tolist()[0] == 4


def test_main_writes_crops_csv_and_sheet(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    tracks = tmp_path / "t.txt"
    _tracks().to_csv(tracks, header=False, index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(hv, "FrameReader", lambda p: FakeReader(p, bad={7}))  # track 3's frame
    hv.main(["--tracks", str(tracks), "--video", str(video), "--out", str(out)])
    csv = pd.read_csv(out / "sizes.csv")
    assert csv.columns.tolist() == ["rank", "track", "size", "frame", "x", "y", "w", "h", "crop"]
    assert csv.track.tolist() == [1, 3, 2]
    assert csv.crop.tolist()[0] == "001_size300_track1.jpg"
    assert math.isnan(csv.crop[1])  # frame 7 could not be read: listed, no crop
    assert sorted(p.name for p in out.glob("0*.jpg")) == [
        "001_size300_track1.jpg", "003_size120_track2.jpg"]
    assert cv2.imread(str(out / "001_size300_track1.jpg")).shape[:2] == (176, 352)
    assert cv2.imread(str(out / "sheet.jpg")) is not None


def test_main_missing_video_exits(tmp_path):
    tracks = tmp_path / "t.txt"
    _tracks().to_csv(tracks, header=False, index=False)
    with pytest.raises(SystemExit):
        hv.main(["--tracks", str(tracks), "--video", str(tmp_path / "no.mp4"), "--out",
                 str(tmp_path / "o")])
