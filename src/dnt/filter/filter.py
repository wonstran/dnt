"""Filter module for detection and track filtering operations.

This module provides filtering utilities for detections and tracks based on:
- Intersection over Union (IoU) with zones
- Spatial containment within polygons
- Line crossing detection
- Various reference points and offsets
"""

import geopandas as gpd
import pandas as pd
from shapely import LineString, Point, Polygon, geometry
from tqdm import tqdm


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
        zones: geometry.MultiPolygon | None = None,
        class_list: list[int] | None = None,
        score_threshold: float = 0,
    ) -> pd.DataFrame:
        """Filter detections by IoU with zones and class list.

        Parameters
        ----------
        detections : pd.DataFrame
            DataFrame of detections with columns for x, y, width, height, score, and class.
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
        detections = detections.loc[detections[6] >= score_threshold].copy()

        # filter classess
        if class_list:
            detections = detections.loc[detections[7].isin(class_list)].copy()

        if zones:
            # filter locations
            g = [
                geometry.Point(xy)
                for xy in zip((detections[2] + detections[4] / 2), (detections[3] + detections[5] / 2), strict=True)
            ]
            geo_detections = gpd.GeoDataFrame(detections, geometry=g)

            frames = geo_detections.loc[geo_detections.geometry.within(zones)].drop(columns="geometry")

            if frames:
                results = pd.concat(frames)
                results = results[~results.index.duplicated()].reset_index(drop=True)
            else:
                results = pd.DataFrame()

        else:
            results = detections

        return results

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
        """Filter tracks by zones.
        Inputs:
            tracks: tracks
            zones: a list of polygons
            method - 'include' (default), 'exclude'
            ref_point - the reference point of a track bbox,
                defalt is br, others - bl, bc, tl, tc, tr, cl, cc, cr
            offset - the offset to ref_point, default is (0, 0)
            video_index - video index
            video_tot - total videos
        Return:
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
        """Filter tracks by zones.
        Inputs:
            tracks - tracks
            zones - zones (polygon)
            method - 'include' (default) include the tracks if they are within the zones; 'exclude' exclude the tracks if they are within the zones
            ref_point - the reference point of a track bbox, defalt is bottom_point, center_point, left_up, right_up, left_buttom, right_buttom
            offset - the offset to ref_point, default is (0, 0)
            video_index - video index
            video_tot - total videos
        Return:
            Filtered tracks
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
                        br - buttom_right,
                        bl - bottom_left
                        bc - bottom_center
                        cc - center_point,
                        cl - left_center,
                        cr - right_center,
                        tc - top_center,
                        tl - top_left,
                        tr - top_right,
            offset - the offset to ref_point, default is (0, 0)
            aggregate - combine outputs to one dataframe, add zone column
            zone_name - if aggregate, the field name of zone variable, default is 'zone'
            video_index - video index
            video_tot - total videos
        Return:
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
        Return:
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
            tolerance - if a bbox intesect the reference lines of (number of lanes -
                tolerance), it is hit. default is 0.
            force_line_indexes: the line indexes that a bbox must intersect for matching
            bbox_size - the size of detection bbox, default is 0 - the orginal bbox
            video_index - the index of video for processing
            video_tot - the total number of videos
        Return:
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
        """Backward-compatible wrapper for :func:`dnt.track.post_process.interpolate_tracks_rts`."""
        import importlib.util
        import sys
        from pathlib import Path

        post_file = Path(__file__).resolve().parents[1] / "track" / "post_process.py"
        module_name = "dnt_track_post_process_dynamic"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, post_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load post_process module from: {post_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module

        _interpolate_tracks_rts = module.interpolate_tracks_rts

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
