import sys

import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from dnt.filter import Filter
from dnt.track.post_process import interpolate_tracks_rts

ZONE = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])


def _dets():
    # positional 8-column detections; centres (15,15) and (25,25) inside ZONE, (115,115) outside
    return pd.DataFrame([[1, -1, 10, 10, 10, 10, 0.9, 2],
                         [1, -1, 20, 20, 10, 10, 0.9, 2],
                         [2, -1, 110, 110, 10, 10, 0.9, 2]])


@pytest.mark.parametrize("zones", [ZONE, MultiPolygon([ZONE]), [ZONE]], ids=["polygon", "multipolygon", "list"])
def test_filter_iou_with_zones(zones):
    out = Filter.filter_iou(_dets(), zones=zones)
    assert len(out) == 2
    assert set(out[0]) == {1}


@pytest.mark.parametrize("zones", [None, []])
def test_filter_iou_without_zones_unchanged(zones):
    assert len(Filter.filter_iou(_dets(), zones=zones)) == 3


def test_filter_interpolate_uses_package_module():
    tracks = pd.DataFrame({"frame": [0, 1, 5], "track": [1, 1, 1], "x": [10, 12, 20],
                           "y": [20, 22, 30], "w": [10, 10, 10], "h": [5, 5, 5]})
    a = Filter.interpolate_tracks_rts(tracks=tracks, verbose=False)
    b = interpolate_tracks_rts(tracks=tracks, verbose=False)
    pd.testing.assert_frame_equal(a, b)
    assert "dnt_track_post_process_dynamic" not in sys.modules
