"""Filter module for detection and track filtering operations.

This module provides filtering utilities for detections and tracks based on:
- Intersection over Union (IoU) with zones
- Spatial containment within polygons
- Line crossing detection
- Various reference points and offsets
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, Point, Polygon, geometry
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from tqdm import tqdm

from ..engine import ious

# Detector.DET_FIELDS order. Not imported from dnt.detect.yolo.detector to avoid pulling
# in Ultralytics (and its import-time side effects) just to filter a detections table.
_DET_FIELD_POSITIONS = {"frame": 0, "res": 1, "x": 2, "y": 3, "w": 4, "h": 5, "conf": 6, "class": 7}


def _det_columns(detections: pd.DataFrame) -> dict[str, int | str]:
    """Map each detection field name to its actual column key in `detections`.

    Accepts either the positional/headerless layout produced by
    `pd.read_csv(file, header=None)` (columns `0..7`) or the named layout
    `Detector.detect()` returns in memory (`Detector.DET_FIELDS`).

    Parameters
    ----------
    detections : pd.DataFrame
        Detections in either layout.

    Returns
    -------
    dict[str, int | str]
        Maps `frame, res, x, y, w, h, conf, class` to the column key actually present.

    Raises
    ------
    KeyError
        If `detections` has neither layout.

    """
    columns = set(detections.columns)
    if set(_DET_FIELD_POSITIONS) <= columns:
        return {name: name for name in _DET_FIELD_POSITIONS}
    if set(_DET_FIELD_POSITIONS.values()) <= columns:
        return dict(_DET_FIELD_POSITIONS)
    msg = (
        f"detections must have Detector.DET_FIELDS columns {list(_DET_FIELD_POSITIONS)} "
        f"or the equivalent positional columns {list(_DET_FIELD_POSITIONS.values())}; "
        f"got columns {list(detections.columns)}."
    )
    raise KeyError(msg)


class Filter:
    """Filter class for detection and track filtering operations.

    Provides static methods for filtering detections and tracks based on:
    - Intersection over Union (IoU) with zones
    - Spatial containment within polygons
    - Line crossing detection
    - Various reference points and offsets

    Methods
    -------
    filter_iou(detections, zones, class_list, score_threshold)
        Filter detections by IoU with zones and class list.
    deduplicate_boxes(detections, iou_thresh, containment_thresh)
        Suppress duplicate and nested-sub-box detections within each frame.
    filter_tracks(tracks, include_zones, exclude_zones, video_index, video_tot)
        Filter tracks by inclusion and exclusion zones.
    filter_tracks_by_zones_agg(tracks, zones, method, ref_point, offset, col_names, video_index, video_tot)
        Filter tracks aggregated by zones with configurable reference point.
    filter_frames_by_zones_agg(tracks, zones, method, ref_point, offset, col_names, video_index, video_tot)
        Filter frames aggregated by zones with configurable reference point.
    filter_tracks_by_zones(tracks, zones, method, ref_point, offset, col_names, zone_name, video_index, video_tot)
        Filter tracks by zones with list, filter, or label methods.
    filter_tracks_by_lines(tracks, lines, method, video_index, video_tot)
        Filter tracks by line crossing detection.
    filter_tracks_by_lines_v2(tracks, lines, method, tolerance, bbox_size, force_line_indexes, video_index, video_tot)
        Advanced line crossing detection with configurable tolerance and forced line indexes.

    """

    def __init__(self) -> None:
        """Initialize the Filter class."""
        pass

    @staticmethod
    def filter_iou(
        detections: pd.DataFrame,
        zones: BaseGeometry | list | None = None,
        class_list: list[int] | None = None,
        score_threshold: float = 0,
    ) -> pd.DataFrame:
        """Filter detections by IoU with zones and class list.

        Parameters
        ----------
        detections : pd.DataFrame
            DataFrame of detections with columns for x, y, width, height, score, and class,
            either the positional/headerless layout (`pd.read_csv(file, header=None)`) or
            the named layout `Detector.detect()` returns (`Detector.DET_FIELDS`).
        zones : geometry.multipolygon, optional
            MultiPolygon zones to filter detections within. Default is None.
        class_list : list[int], optional
            List of class IDs to include in filtering. Default is None.
        score_threshold : float, optional
            Minimum confidence score threshold. Default is 0.

        Returns
        -------
        pd.DataFrame
            Filtered detections within zones and matching class list and score threshold.

        """
        cols = _det_columns(detections)
        detections = detections.loc[detections[cols["conf"]] >= score_threshold].copy()

        # filter classess
        if class_list:
            detections = detections.loc[detections[cols["class"]].isin(class_list)].copy()

        if zones is None or (not isinstance(zones, BaseGeometry) and len(zones) == 0):
            return detections

        area = zones if isinstance(zones, BaseGeometry) else unary_union(list(zones))
        centers = gpd.GeoSeries(
            gpd.points_from_xy(
                detections[cols["x"]] + detections[cols["w"]] / 2,
                detections[cols["y"]] + detections[cols["h"]] / 2,
            ),
            index=detections.index,
        )
        return detections.loc[centers.within(area).to_numpy()].reset_index(drop=True)

    @staticmethod
    def deduplicate_boxes(
        detections: pd.DataFrame,
        iou_thresh: float = 0.45,
        containment_thresh: float = 0.65,
    ) -> pd.DataFrame:
        """Suppress duplicate and nested-sub-box detections within each frame.

        Within each frame, detections are visited in descending confidence order (as in
        NMS). A lower-confidence detection is dropped against an already-kept one if
        either:

        - their IoU exceeds `iou_thresh` (near-duplicate boxes for the same object), or
        - its containment ratio (intersection area / smaller box's area) exceeds
          `containment_thresh` (e.g. a torso box nested inside a full-body box).

        Both ratios are derived from a single per-frame `dnt.engine.ious` call: IoU
        already encodes the intersection area given each box's own area
        (`inter = iou * (area_a + area_b) / (1 + iou)`), so no separate intersection
        pass is needed.

        Parameters
        ----------
        detections : pd.DataFrame
            Detections with columns `frame, res, x, y, w, h, conf, class`, either the
            positional/headerless layout (`pd.read_csv(file, header=None)`) or the named
            layout `Detector.detect()` returns (`Detector.DET_FIELDS`); the same layouts
            `filter_iou` accepts.
        iou_thresh : float, optional
            IoU above which the lower-confidence box of a pair is suppressed.
            Default is 0.45.
        containment_thresh : float, optional
            Containment ratio above which the lower-confidence box of a pair is
            suppressed. Default is 0.65.

        Returns
        -------
        pd.DataFrame
            `detections` with suppressed rows removed, in their original row order.

        """
        if detections.empty:
            return detections

        cols = _det_columns(detections)
        frames = detections[cols["frame"]].to_numpy()
        x = detections[cols["x"]].to_numpy(dtype=np.float64)
        y = detections[cols["y"]].to_numpy(dtype=np.float64)
        w = detections[cols["w"]].to_numpy(dtype=np.float64)
        h = detections[cols["h"]].to_numpy(dtype=np.float64)
        conf = detections[cols["conf"]].to_numpy(dtype=np.float64)
        tlbr = np.column_stack([x, y, x + w, y + h])
        # `ious()` wraps cython_bbox.bbox_overlaps(), which uses the classic Faster-RCNN
        # pixel-inclusive convention (area = (w+1)*(h+1), not continuous w*h). The
        # intersection-from-IoU algebra below is only exact if these areas match that
        # convention exactly.
        areas = (w + 1) * (h + 1)

        # group row positions by frame without pandas groupby overhead: sort by frame
        # once, then split at each frame boundary.
        frame_order = np.argsort(frames, kind="stable")
        boundaries = np.flatnonzero(np.diff(frames[frame_order])) + 1
        groups = np.split(frame_order, boundaries)

        keep = np.ones(len(detections), dtype=bool)
        for idx in groups:
            if len(idx) < 2:
                continue
            # highest confidence first, so earlier boxes suppress later, weaker ones
            order = idx[np.argsort(-conf[idx], kind="stable")]

            overlap = ious(tlbr[order], tlbr[order])
            sum_areas = areas[order][:, None] + areas[order][None, :]
            inter = overlap * sum_areas / (1.0 + overlap)
            min_area = np.minimum(areas[order][:, None], areas[order][None, :])
            containment = np.divide(inter, min_area, out=np.zeros_like(inter), where=min_area > 0)

            suppress_by = (overlap > iou_thresh) | (containment > containment_thresh)
            np.fill_diagonal(suppress_by, False)

            suppressed = np.zeros(len(order), dtype=bool)
            for i in range(len(order) - 1):
                if suppressed[i]:
                    continue
                suppressed[i + 1 :] |= suppress_by[i, i + 1 :]
            keep[order[suppressed]] = False

        return detections.loc[keep]

    @staticmethod
    def filter_tracks(
        tracks: pd.DataFrame,
        include_zones: geometry.MultiPolygon | None = None,
        exclude_zones: geometry.MultiPolygon | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Filter tracks by inclusion and exclusion zones.

        Parameters
        ----------
        tracks : pd.DataFrame
            DataFrame of tracks with columns for x, y, width, height, and track ID.
        include_zones : geometry.MultiPolygon, optional
            MultiPolygon zones to include tracks within. Default is None.
        exclude_zones : geometry.MultiPolygon, optional
            MultiPolygon zones to exclude tracks from. Default is None.
        video_index : int, optional
            Index of the current video being processed. Default is None.
        video_tot : int, optional
            Total number of videos being processed. Default is None.

        Returns
        -------
        pd.DataFrame
            Filtered tracks after applying inclusion and exclusion zones.

        """
        g = [geometry.Point(xy) for xy in zip((tracks[2] + tracks[4] / 2), (tracks[3] + tracks[5] / 2), strict=True)]
        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        track_ids = tracks[1].unique()
        include_ids = []
        exclude_ids = []

        pbar = tqdm(total=len(track_ids), unit=" tracks")
        if video_index and video_tot:
            pbar.set_description_str(f"Filtering zones {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Filtering zones ")

        for track_id in track_ids:
            if include_zones:
                selected_tracks = geo_tracks.loc[
                    (geo_tracks[1] == track_id) & (geo_tracks.geometry.within(include_zones))
                ]
                if len(selected_tracks) > 0:
                    include_ids.append(track_id)

            if exclude_zones:
                selected_tracks = geo_tracks.loc[
                    (geo_tracks[1] == track_id) & (geo_tracks.geometry.within(exclude_zones))
                ]
                if len(selected_tracks) > 0:
                    exclude_ids.append(track_id)

            pbar.update()

        pbar.close()

        if len(include_ids) > 0:
            results = tracks.loc[tracks[1].isin(include_ids)].copy()
        else:
            results = tracks.copy()

        if len(exclude_ids) > 0:
            results = results.loc[~results[1].isin(exclude_ids)].copy()

        return results

    @staticmethod
    def filter_tracks_by_zones_agg(
        tracks: pd.DataFrame,
        zones: geometry.MultiPolygon | None = None,
        method: str = "include",
        ref_point: str = "bc",
        offset: tuple | None = None,
        col_names: list | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Filter tracks by polygon zones using an aggregate ID match.

        Inputs:
            tracks - input tracks.
            zones - polygons used for matching.
            method - 'include' (default) or 'exclude'.
            ref_point - bbox reference point (default: 'bc'); options include
                bl, bc, br, tl, tc, tr, cl, cc, cr.
            offset - offset applied to the reference point, default is (0, 0).
            video_index - video index for progress display.
            video_tot - total video count for progress display.

        Returns:
            Filtered tracks.

        """
        if offset is None:
            offset = (0, 0)
        if col_names is None:
            col_names = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]

        try:
            tracks.columns = col_names
        except Exception:
            print("Tracks is invalid!")

        if ref_point == "cc":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] / 2 + offset[0]),
                    (tracks["y"] + tracks["h"] / 2 + offset[1]),
                    strict=True,
                )
            ]
        elif ref_point == "tc":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + offset[1]), strict=True)
            ]
        elif ref_point == "bc":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True
                )
            ]
        elif ref_point == "cl":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]), strict=True)
            ]
        elif ref_point == "cr":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]), strict=True
                )
            ]
        elif ref_point == "tl":
            g = [Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + offset[1]), strict=True)]
        elif ref_point == "tr":
            g = [
                Point(xy) for xy in zip((tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + offset[1]), strict=True)
            ]
        elif ref_point == "bl":
            g = [
                Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True)
            ]
        elif ref_point == "br":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True
                )
            ]
        else:
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True
                )
            ]

        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        matched_ids = []
        pbar = tqdm(total=len(zones), unit=" zones")
        if video_index and video_tot:
            pbar.set_description_str(f"Filtering zones {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Filtering zones ")

        for zone in zones:
            matched = geo_tracks[geo_tracks.geometry.within(zone)]
            if len(matched) > 0:
                matched_ids.extend(matched["track"].unique().tolist())
            pbar.update()

        pbar.close()

        if len(matched_ids) > 0:
            if method == "include":
                results = tracks.loc[tracks["track"].isin(matched_ids)].copy()
            else:
                results = tracks.loc[~tracks["track"].isin(matched_ids)].copy()
        else:
            results = tracks.copy()

        return results

    @staticmethod
    def filter_frames_by_zones_agg(
        tracks: pd.DataFrame,
        zones: geometry.MultiPolygon | None = None,
        method: str = "include",
        ref_point: str = "bc",
        offset: tuple | None = None,
        col_names: list | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Filter frame rows by polygon zones.

        Inputs:
            tracks - input tracks.
            zones - polygons used for matching.
            method - 'include' (default) keeps matching rows; 'exclude' drops them.
            ref_point - bbox reference point (default: 'bc').
            offset - offset applied to the reference point, default is (0, 0).
            video_index - video index for progress display.
            video_tot - total video count for progress display.

        Returns:
            Filtered frame rows.

        """
        if offset is None:
            offset = (0, 0)
        if col_names is None:
            col_names = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]

        try:
            tracks.columns = col_names
        except:
            print("Tracks is invalid!")

        if ref_point == "cc":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] / 2 + offset[0]),
                    (tracks["y"] + tracks["h"] / 2 + offset[1]),
                    strict=True,
                )
            ]
        elif ref_point == "tc":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + offset[1]), strict=True)
            ]
        elif ref_point == "bc":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True
                )
            ]
        elif ref_point == "cl":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]), strict=True)
            ]
        elif ref_point == "cr":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]), strict=True
                )
            ]
        elif ref_point == "tl":
            g = [Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + offset[1]), strict=True)]
        elif ref_point == "tr":
            g = [
                Point(xy) for xy in zip((tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + offset[1]), strict=True)
            ]
        elif ref_point == "bl":
            g = [
                Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True)
            ]
        elif ref_point == "br":
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True
                )
            ]
        else:
            g = [
                Point(xy)
                for xy in zip(
                    (tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] + offset[1]), strict=True
                )
            ]

        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        matched_frames = []
        pbar = tqdm(total=len(zones), unit=" zones")
        if video_index and video_tot:
            pbar.set_description_str(f"Filtering zones {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Filtering zones ")

        for zone in zones:
            matched = geo_tracks[geo_tracks.geometry.within(zone)]
            if len(matched) > 0:
                matched_frames.extend(matched.index.values.tolist())
            pbar.update()

        pbar.close()

        if len(matched_frames) > 0:
            if method == "include":
                results = tracks.iloc[matched_frames].copy()
            else:
                results = tracks.drop(matched_frames, axis=0).copy()
        else:
            results = tracks.copy()

        return results

    @staticmethod
    def filter_tracks_by_zones(
        tracks: pd.DataFrame,
        zones: list[Polygon] | None = None,
        method: str = "list",
        ref_point: str = "bc",
        offset: tuple | None = None,
        col_names: list | None = None,
        zone_name: str = "zone",
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Filter tracks by zones.

        Inputs:
            tracks - tracks
            zones - zones (polygon)
            method - 'list' (default) - List track ids within zones
                     'filter' - filter tracks within zones
                     'label' - label tracks with zone index
            ref_point - the reference point of a track bbox,
                        br - bottom_right,
                        bl - bottom_left
                        bc - bottom_center
                        cc - center_point,
                        cl - left_center,
                        cr - right_center,
                        tc - top_center,
                        tl - top_left,
                        tr - top_right,
            offset - the offset to ref_point, default is (0, 0)
            zone_name - field name used when method='label', default is 'zone'
            video_index - video index
            video_tot - total videos
        Returns:
            Filtered tracks
        """
        if offset is None:
            offset = (0, 0)
        if col_names is None:
            col_names = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]

        try:
            tracks.columns = col_names
        except Exception:
            print("Tracks is invalid!")

        if ref_point == "cc":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]))
            ]
        elif ref_point == "tc":
            g = [Point(xy) for xy in zip((tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + offset[1]))]
        elif ref_point == "bc":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] + offset[1]))
            ]
        elif ref_point == "cl":
            g = [Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]))]
        elif ref_point == "cr":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + tracks["h"] / 2 + offset[1]))
            ]
        elif ref_point == "tl":
            g = [Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + offset[1]))]
        elif ref_point == "tr":
            g = [Point(xy) for xy in zip((tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + offset[1]))]
        elif ref_point == "bl":
            g = [Point(xy) for xy in zip((tracks["x"] + offset[0]), (tracks["y"] + tracks["h"] + offset[1]))]
        elif ref_point == "br":
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] + offset[0]), (tracks["y"] + tracks["h"] + offset[1]))
            ]
        else:
            g = [
                Point(xy)
                for xy in zip((tracks["x"] + tracks["w"] / 2 + offset[0]), (tracks["y"] + tracks["h"] + offset[1]))
            ]

        geo_tracks = gpd.GeoDataFrame(tracks, geometry=g)

        matched_ids = []
        pbar = tqdm(total=len(zones), unit=" zones")
        if video_index and video_tot:
            pbar.set_description_str(f"Filtering zones {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Filtering zones ")

        for zone in zones:
            matched = geo_tracks[geo_tracks.geometry.within(zone)]
            if len(matched) > 0:
                matched_ids.append(matched["track"].unique().tolist())
            pbar.update()

        pbar.close()

        if (method == "filter") or (method == "label"):
            tracks[zone_name] = -1
            for i in range(len(matched_ids)):
                tracks.loc[tracks["track"].isin(matched_ids[i]), zone_name] = i
            results = tracks[tracks[zone_name] != -1].copy() if method == "filter" else tracks
        else:
            results = []
            if len(matched_ids) > 0:
                for i in range(len(matched_ids)):
                    result = tracks.loc[tracks["track"].isin(matched_ids[i])].copy()
                    results.append(result)

        return results

    @staticmethod
    def filter_tracks_by_lines(
        tracks: pd.DataFrame,
        lines: list[LineString] | None = None,
        method: str = "include",
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Filter tracks by lines.

        Inputs:
            tracks - a DataFrame of tracks, [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED, RESERVED, RESERVED]
            lines - a list of LineString
            method - filtering method, include (default) - including tracks crossing
                the lines, exclude - exclude tracks crossing the lines
            video_index - the index of video for processing
            video_tot - the total number of videos
        Returns:
            a DataFrame of [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED, RESERVED, RESERVED]
        """
        track_ids = tracks[1].unique()
        ids = []

        pbar = tqdm(total=len(track_ids), unit=" tracks")
        if video_index and video_tot:
            pbar.set_description_str(f"Filtering tracks {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Filtering tracks ")

        for track_id in track_ids:
            selected = tracks.loc[(tracks[1] == track_id)].copy()
            if len(selected) > 0:
                g = selected.apply(
                    lambda track: Polygon([
                        (track[2], track[3]),
                        (track[2] + track[4], track[3]),
                        (track[2] + track[4], track[3] + track[5]),
                        (track[2], track[3] + track[5]),
                    ]),
                    axis=1,
                )
                intersected = True
                for line in lines:
                    intersected = intersected and any(line.intersects(g).values.tolist())

                if intersected:
                    ids.append(track_id)

            pbar.update()

        pbar.close()

        results = []
        if method == "include":
            results = tracks.loc[tracks[1].isin(ids)].copy()
        elif method == "exclude":
            results = tracks.loc[~tracks[1].isin(ids)].copy()

        results.sort_values(by=[0, 1], inplace=True)
        return results

    @staticmethod
    def filter_tracks_by_lines_v2(
        tracks: pd.DataFrame,
        lines: list[LineString] | None = None,
        method: str = "include",
        tolerance: int = 0,
        bbox_size: int = 0,
        force_line_indexes: list[int] | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Filter tracks by lines.

        Inputs:
            tracks - a DataFrame of tracks, [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH,
                RESERVED, RESERVED, RESERVED]
            lines - a list of LineString
            method - filtering method, include (default) - including tracks crossing the
                lines, exclude - exclude tracks crossing the lines
            tolerance - if a bbox intersects the reference lines of (number of lanes -
                tolerance), it is hit. default is 0.
            force_line_indexes: the line indexes that a bbox must intersect for matching
            bbox_size - the size of detection bbox, default is 0 - the original bbox
            video_index - the index of video for processing
            video_tot - the total number of videos
        Returns:
            a DataFrame of [FRAME, TRACK_ID, TOPX, TOPY, WIDTH, LENGTH, RESERVED,
                RESERVED, RESERVED]
        """
        tracks.columns = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]
        track_ids = tracks["track"].unique()
        ids = []

        # set hit criterion
        hit_criterion = len(lines) - tolerance
        if (hit_criterion < 1) or (hit_criterion > len(lines)):
            hit_criterion = len(lines)

        pbar = tqdm(total=len(track_ids), unit=" tracks")
        if video_index and video_tot:
            pbar.set_description_str(f"Filtering tracks {video_index} of {video_tot}")
        else:
            pbar.set_description_str("Filtering tracks ")

        for track_id in track_ids:
            selected = tracks.loc[(tracks["track"] == track_id)].copy()
            hit_cnt = 0
            hit_force = True
            if len(selected) > 0:
                if bbox_size == 0:
                    g = selected.apply(
                        lambda track: Polygon([
                            (track["x"], track["y"]),
                            (track["x"] + track["w"], track["y"]),
                            (track["x"] + track["w"], track["y"] + track["h"]),
                            (track["x"], track["y"] + track["h"]),
                        ]),
                        axis=1,
                    )
                else:
                    g = selected.apply(
                        lambda track: Polygon([
                            (track["x"] + track["w"] / 2 - bbox_size, track["y"] + track["h"] - bbox_size),
                            (track["x"] + track["w"] / 2 + bbox_size, track["y"] + track["h"] - bbox_size),
                            (track["x"] + track["w"] / 2 + bbox_size, track["y"] + track["h"]),
                            (track["x"] + track["w"] / 2 - bbox_size, track["y"] + track["h"]),
                        ]),
                        axis=1,
                    )

                for line in lines:
                    if any(line.intersects(g).values.tolist()):
                        hit_cnt += 1

                if force_line_indexes is not None:
                    force_lines = [lines[i] for i in force_line_indexes]
                    for line in force_lines:
                        hit_force = any(line.intersects(g).values.tolist())

                if (hit_cnt >= hit_criterion) and hit_force:
                    ids.append(track_id)

            pbar.update()

        pbar.close()

        results = []
        if method == "include":
            results = tracks.loc[tracks["track"].isin(ids)].copy()
        elif method == "exclude":
            results = tracks.loc[~tracks["track"].isin(ids)].copy()

        results.sort_values(by=["frame", "track"], inplace=True)
        return results

    @staticmethod
    def interpolate_tracks_rts(
        tracks: pd.DataFrame | None = None,
        track_file: str | None = None,
        output_file: str | None = None,
        col_names: list[str] | None = None,
        fill_gaps_only: bool = True,
        smooth_existing: bool = False,
        process_var: float = 10.0,
        meas_var_pos: float = 25.0,
        meas_var_size: float = 16.0,
        min_track_len: int = 2,
        max_gap: int = 30,
        add_interp_flag: bool = True,
        interp_col: str = "interp",
        verbose: bool = True,
        video_index: int | None = None,
        video_tot: int | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible wrapper for :func:`dnt.refine.interpolate.interpolate_tracks_rts`."""
        from ..refine.interpolate import interpolate_tracks_rts as _interpolate_tracks_rts

        return _interpolate_tracks_rts(
            tracks=tracks,
            track_file=track_file,
            output_file=output_file,
            col_names=col_names,
            fill_gaps_only=fill_gaps_only,
            smooth_existing=smooth_existing,
            process_var=process_var,
            meas_var_pos=meas_var_pos,
            meas_var_size=meas_var_size,
            min_track_len=min_track_len,
            max_gap=max_gap,
            add_interp_flag=add_interp_flag,
            interp_col=interp_col,
            verbose=verbose,
            video_index=video_index,
            video_tot=video_tot,
        )


if __name__ == "__main__":
    pass
