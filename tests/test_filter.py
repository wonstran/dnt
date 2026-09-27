import sys

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from dnt.detect.yolo.detector import Detector
from dnt.filter import Filter
from dnt.track.post_process import interpolate_tracks_rts

ZONE = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])


def _dets():
    # positional 8-column detections; centres (15,15) and (25,25) inside ZONE, (115,115) outside
    return pd.DataFrame([[1, -1, 10, 10, 10, 10, 0.9, 2],
                         [1, -1, 20, 20, 10, 10, 0.9, 2],
                         [2, -1, 110, 110, 10, 10, 0.9, 2]])


def _dets_named():
    # same rows as _dets(), but with Detector.DET_FIELDS column names, matching what
    # Detector.detect() returns in memory (as opposed to a headerless CSV read back in).
    return pd.DataFrame(_dets().to_numpy(), columns=Detector.DET_FIELDS)


@pytest.mark.parametrize("zones", [ZONE, MultiPolygon([ZONE]), [ZONE]], ids=["polygon", "multipolygon", "list"])
def test_filter_iou_with_zones(zones):
    out = Filter.filter_iou(_dets(), zones=zones)
    assert len(out) == 2
    assert set(out[0]) == {1}


@pytest.mark.parametrize("zones", [None, []])
def test_filter_iou_without_zones_unchanged(zones):
    assert len(Filter.filter_iou(_dets(), zones=zones)) == 3


def test_filter_iou_accepts_detector_named_columns():
    # regression: Detector.detect() returns named columns, not the positional/headerless
    # layout filter_iou originally assumed; passing its output straight in used to raise
    # KeyError.
    out = Filter.filter_iou(_dets_named(), zones=ZONE)
    assert len(out) == 2
    assert set(out["frame"]) == {1}
    assert list(out.columns) == Detector.DET_FIELDS


def test_filter_interpolate_uses_package_module():
    tracks = pd.DataFrame({"frame": [0, 1, 5], "track": [1, 1, 1], "x": [10, 12, 20],
                           "y": [20, 22, 30], "w": [10, 10, 10], "h": [5, 5, 5]})
    a = Filter.interpolate_tracks_rts(tracks=tracks, verbose=False)
    b = interpolate_tracks_rts(tracks=tracks, verbose=False)
    pd.testing.assert_frame_equal(a, b)
    assert "dnt_track_post_process_dynamic" not in sys.modules


def test_deduplicate_boxes_empty_returns_empty():
    empty = pd.DataFrame(columns=range(8))
    out = Filter.deduplicate_boxes(empty)
    assert out.empty


def test_deduplicate_boxes_accepts_detector_named_columns():
    # regression: same KeyError as filter_iou when fed Detector.detect()'s named-column
    # output directly, e.g. `Filter.deduplicate_boxes(detector.detect(video))`.
    dets = pd.DataFrame([
        [1, -1, 10, 10, 20, 20, 0.6, 2],
        [1, -1, 11, 11, 20, 20, 0.9, 2],
    ], columns=Detector.DET_FIELDS)
    out = Filter.deduplicate_boxes(dets)
    assert len(out) == 1
    assert out.iloc[0]["conf"] == 0.9
    assert list(out.columns) == Detector.DET_FIELDS


def test_deduplicate_boxes_keeps_single_detection_per_frame():
    dets = pd.DataFrame([[1, -1, 10, 10, 20, 20, 0.9, 2]])
    out = Filter.deduplicate_boxes(dets)
    pd.testing.assert_frame_equal(out, dets)


def test_deduplicate_boxes_drops_near_identical_duplicate():
    # two near-identical boxes (IoU ~1.0) in the same frame; keep the higher-confidence one
    dets = pd.DataFrame([
        [1, -1, 10, 10, 20, 20, 0.6, 2],
        [1, -1, 11, 11, 20, 20, 0.9, 2],
    ])
    out = Filter.deduplicate_boxes(dets, iou_thresh=0.45)
    assert len(out) == 1
    assert out.iloc[0][6] == 0.9


def test_deduplicate_boxes_suppresses_nested_sub_box_by_containment():
    # a torso-sized box fully nested inside a full-body box: low IoU, high containment
    dets = pd.DataFrame([
        [1, -1, 0, 0, 100, 100, 0.95, 2],   # full body
        [1, -1, 20, 10, 30, 40, 0.5, 2],    # torso, fully inside, small IoU
    ])
    out = Filter.deduplicate_boxes(dets, iou_thresh=0.45, containment_thresh=0.65)
    assert len(out) == 1
    assert out.iloc[0][6] == 0.95


def test_deduplicate_boxes_keeps_distinct_boxes():
    dets = pd.DataFrame([
        [1, -1, 0, 0, 10, 10, 0.9, 2],
        [1, -1, 200, 200, 10, 10, 0.8, 2],
    ])
    out = Filter.deduplicate_boxes(dets)
    assert len(out) == 2


def test_deduplicate_boxes_independent_across_frames():
    dets = pd.DataFrame([
        [1, -1, 10, 10, 20, 20, 0.6, 2],
        [1, -1, 11, 11, 20, 20, 0.9, 2],
        [2, -1, 10, 10, 20, 20, 0.6, 2],
        [2, -1, 11, 11, 20, 20, 0.9, 2],
    ])
    out = Filter.deduplicate_boxes(dets)
    assert len(out) == 2
    assert set(out[0]) == {1, 2}


def test_deduplicate_boxes_preserves_original_row_order():
    dets = pd.DataFrame([
        [2, -1, 0, 0, 10, 10, 0.9, 2],
        [1, -1, 0, 0, 10, 10, 0.9, 2],
        [1, -1, 200, 200, 10, 10, 0.8, 2],
    ])
    out = Filter.deduplicate_boxes(dets)
    assert list(out.index) == [0, 1, 2]


def test_deduplicate_boxes_matches_reference_implementation():
    """Cross-check the vectorized implementation against a naive reference on random data."""
    rng = np.random.default_rng(0)
    rows = []
    for frame in range(5):
        n = rng.integers(2, 8)
        xs = rng.uniform(0, 50, n)
        ys = rng.uniform(0, 50, n)
        ws = rng.uniform(5, 30, n)
        hs = rng.uniform(5, 30, n)
        confs = rng.uniform(0.1, 1.0, n)
        for x, y, w, h, conf in zip(xs, ys, ws, hs, confs, strict=True):
            rows.append([frame, -1, x, y, w, h, conf, 2])
    dets = pd.DataFrame(rows)

    def _reference(df, iou_thresh=0.45, containment_thresh=0.65):
        # Mirrors dnt.engine.ious()'s cython_bbox backend exactly: the classic
        # Faster-RCNN bbox_overlaps() convention treats coordinates as pixel-inclusive
        # (+1 on width/height/intersection), unlike plain continuous geometry. Using
        # that same convention here makes this a check of deduplicate_boxes()'s
        # vectorized *algorithm*, not a comparison of two different IoU definitions.
        keep_indices = []
        for _, grp in df.groupby(0):
            if len(grp) == 1:
                keep_indices.append(grp.index[0])
                continue
            grp_sorted = grp.sort_values(6, ascending=False)
            boxes = grp_sorted[[2, 3, 4, 5]].to_numpy()
            indices = grp_sorted.index.to_numpy()
            x1, y1 = boxes[:, 0], boxes[:, 1]
            x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
            areas = (boxes[:, 2] + 1) * (boxes[:, 3] + 1)
            suppressed = np.zeros(len(boxes), dtype=bool)
            for i in range(len(boxes)):
                if suppressed[i]:
                    continue
                keep_indices.append(indices[i])
                for j in range(i + 1, len(boxes)):
                    if suppressed[j]:
                        continue
                    xi1, yi1 = max(x1[i], x1[j]), max(y1[i], y1[j])
                    xi2, yi2 = min(x2[i], x2[j]), min(y2[i], y2[j])
                    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
                    iou = inter / max(areas[i] + areas[j] - inter, 1e-6)
                    containment = inter / max(min(areas[i], areas[j]), 1e-6)
                    if iou > iou_thresh or containment > containment_thresh:
                        suppressed[j] = True
        return df.loc[sorted(keep_indices)].copy()

    expected = _reference(dets)
    actual = Filter.deduplicate_boxes(dets)
    assert sorted(actual.index) == sorted(expected.index)
