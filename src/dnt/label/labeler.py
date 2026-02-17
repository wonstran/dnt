"""Video labeling and annotation module.

This module provides functionality for drawing and annotating video frames with labels,
tracks, and detections. It includes the Labeler class for drawing on videos using OpenCV
or FFmpeg backends, and utilities for extracting frames and clips from videos.
"""

import itertools
import os
import random
import subprocess
import sys
from ast import literal_eval
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11

    class StrEnum(str, Enum):  # noqa: UP042
        """Compatibility shim for StrEnum in Python < 3.11.

        This class provides a string-based enum for older Python versions
        that don't have StrEnum in the enum module.
        """

        pass


import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shared.util import load_classes

DRAW_COLUMNS = ["frame", "type", "coords", "color", "size", "thick", "desc", "fill", "alpha"]
TRACK_COLUMNS = ["frame", "track", "x", "y", "w", "h", "score", "cls", "r3", "r4"]
BRIGHT_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 128, 0),
    (128, 255, 0),
    (255, 0, 128),
    (128, 0, 255),
    (0, 128, 255),
)


def _bright_color(index: int) -> tuple[int, int, int]:
    """Get a bright color from the palette by index.

    Parameters
    ----------
    index : int
        Color index (wraps around using modulo).

    Returns
    -------
    tuple[int, int, int]
        BGR color tuple.

    """
    return BRIGHT_COLORS_BGR[index % len(BRIGHT_COLORS_BGR)]


class ElementType(StrEnum):
    """Supported drawing element types.

    Values
    ------
    - TXT: Text label
    - LINE: Line segment
    - BOX: Rectangle (box)
    - BBOX: Bounding box with label
    - CIRCLE: Circle or disk
    - POLYGON: Filled polygon
    - POLYLINES: Open polyline
    """

    TXT = "txt"
    LINE = "line"
    BOX = "box"
    BBOX = "bbox"
    CIRCLE = "circle"
    POLYGON = "polygon"
    POLYLINES = "polylines"


class LabelMethod(StrEnum):
    """Supported rendering backends.

    Values
    ------
    - OPENCV: use OpenCV to draw labels (default)
    - FFMPEG: use FFmpeg to draw labels with H.265 encoding for better compression
    - CHROME_SAFE: use FFmpeg to draw labels with H.264 encoding and browser-compatible settings for maximum compatibility
    """

    OPENCV = "opencv"
    FFMPEG = "ffmpeg"
    CHROME_SAFE = "chrome_safe"


class Encoder(StrEnum):
    """Supported FFmpeg encoders.

    Values
    ------
    - LIBX264: H.264 codec (AVC)
    - LIBX265: H.265 codec (HEVC)
    - H264_NVENC: NVIDIA GPU-accelerated H.264
    - HEVC_NVENC: NVIDIA GPU-accelerated H.265
    """

    LIBX264 = "libx264"
    LIBX265 = "libx265"
    H264_NVENC = "h264_nvenc"
    HEVC_NVENC = "hevc_nvenc"


class Preset(StrEnum):
    """Supported FFmpeg presets.

    Values
    ------
    - ULTRAFAST: Fastest encoding, largest file
    - SUPERFAST: Very fast encoding
    - VERYFAST: Fast encoding
    - FASTER: Faster encoding
    - FAST: Fast encoding
    - MEDIUM: Medium speed (default for most encoders)
    - SLOW: Slow encoding
    - SLOWER: Very slow encoding
    - VERYSLOW: Slowest encoding, smallest file
    """

    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"


class TrackClipMethod(StrEnum):
    """Track clip selection strategies.

    Values
    ------
    - ALL: include all tracks (default)
    - RANDOM: randomly select a specified number of tracks, specified by `random_number`
    - SPECIFY: specify track ids to include, provided in `track_ids`
    """

    ALL = "all"
    RANDOM = "random"
    SPECIFY = "specify"


class Labeler:
    """A video labeler for drawing and annotating video frames.

    This class provides functionality to draw labels, tracks, and detections on video files
    using either OpenCV or FFmpeg as the backend.

    Attributes
    ----------
    method : str
        The drawing method: 'opencv', 'ffmpeg', or 'chrome_safe'.
    encoder : str
        The video encoder to use with FFmpeg.
    preset : str
        The encoding preset for quality/speed tradeoff.
    crf : int
        Constant Rate Factor for compression quality.
    pix_fmt : str
        Pixel format for video output.
    compress_message : bool
        Whether to show compressed progress messages.
    nodraw_empty : bool
        Whether to skip drawing empty frames.

    """

    def __init__(
        self,
        method: LabelMethod | str = LabelMethod.OPENCV,
        encoder: Encoder | str = Encoder.LIBX264,
        preset: Preset | str = Preset.MEDIUM,
        crf: int = 23,
        pix_fmt: str = "bgr24",
        compress_message: bool = False,
        nodraw_empty: bool = True,
    ):
        """Initialize the Labeler.

        Parameters
        ----------
        method : LabelMethod | str
            'opencv' (default) - use opencv to draw labels
            'ffmpeg' - use ffmpeg to draw labels
            'chrome_safe' - use ffmpeg to draw labels with chrome compatible video format
        encoder : Encoder | str
            'libx264' (default) - use libx264 encoder for ffmpeg
            'libx265' - use libx265 encoder for ffmpeg
            'h264_nvenc' - use h264_nvenc encoder for ffmpeg
            'hevc_nvenc' - use hevc_nvenc encoder for ffmpeg
        preset : Preset | str
            'medium' (default) - use medium preset for ffmpeg
            'slow' - use slow preset for ffmpeg
            'fast' - use fast preset for ffmpeg
        crf : int
            23 (default) - use 23 crf for ffmpeg, lower is better quality
        pix_fmt : str
            Pixel format for video output. Default is 'bgr24'.
        compress_message : bool
            False (default) - show compress message in progress bar
        nodraw_empty : bool
            True (default) - not draw empty frames

        """
        if isinstance(method, LabelMethod):
            self.method = method.value
        else:
            method_value = str(method).lower().strip()
            valid_methods = {m.value for m in LabelMethod}
            if method_value not in valid_methods:
                raise ValueError(f"Invalid method={method!r}. Choose one of {sorted(valid_methods)}.")
            self.method = method_value

        if isinstance(encoder, Encoder):
            self.encoder = encoder.value
        else:
            encoder_value = str(encoder).lower().strip()
            valid_encoders = {e.value for e in Encoder}
            if encoder_value not in valid_encoders:
                raise ValueError(f"Invalid encoder={encoder!r}. Choose one of {sorted(valid_encoders)}.")
            self.encoder = encoder_value

        if isinstance(preset, Preset):
            self.preset = preset.value
        else:
            preset_value = str(preset).lower().strip()
            valid_presets = {p.value for p in Preset}
            if preset_value not in valid_presets:
                raise ValueError(f"Invalid preset={preset!r}. Choose one of {sorted(valid_presets)}.")
            self.preset = preset_value

        self.crf = crf
        self.pix_fmt = pix_fmt
        self.compress_message = compress_message
        self.nodraw_empty = nodraw_empty

    @staticmethod
    def _normalize_color(color: object, default: tuple[int, int, int] = (0, 255, 255)) -> tuple[int, int, int]:
        """Normalize color input to BGR tuple.

        Parameters
        ----------
        color : object
            Color input (tuple, list, ndarray, or None).
        default : tuple[int, int, int]
            Default color if input is invalid. Default is cyan (0, 255, 255).

        Returns
        -------
        tuple[int, int, int]
            Normalized BGR color tuple.

        """
        if color is None:
            return default
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
            return (int(color[0]), int(color[1]), int(color[2]))
        return default

    @staticmethod
    def _ensure_draw_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has all required draw columns with defaults.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with all required DRAW_COLUMNS in correct order.

        """
        if "fill" not in df.columns:
            df["fill"] = False
        if "alpha" not in df.columns:
            df["alpha"] = 0.0
        return df[DRAW_COLUMNS]

    @staticmethod
    def _draw_element(frame: np.ndarray, element: dict[str, str | int | float | list | bool]) -> None:
        """Draw a single element on a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame (BGR image array).
        element : dict
            Element specification with keys: type, coords, color, size, thick, desc, fill, alpha.
            - type: ElementType value (txt, line, box, bbox, circle, polygon, polylines)
            - coords: List of coordinate tuples
            - color: RGB/BGR color tuple
            - size: Font size or radius (float)
            - thick: Line thickness (-1 for filled)
            - desc: Text description
            - fill: Whether to fill shapes (bool or str)
            - alpha: Overlay strength 0.0-1.0
              (0=invisible overlay, 1=fully visible overlay)

        """
        element_type = str(element["type"]).lower()
        coords = element["coords"]
        color = Labeler._normalize_color(element.get("color"))
        size = float(element.get("size", 1))
        thick = int(element.get("thick", 1))
        desc = str(element.get("desc", ""))
        fill_raw = element.get("fill", False)
        if isinstance(fill_raw, str):
            fill = fill_raw.strip().lower() in {"1", "true", "yes", "y", "t"}
        else:
            fill = bool(fill_raw)
        alpha = float(element.get("alpha", 0.0))
        alpha = min(max(alpha, 0.0), 1.0)

        if element_type == ElementType.TXT.value:
            cv2.putText(frame, desc, tuple(map(int, coords[0])), 0, size, color, thick)
        elif element_type == ElementType.LINE.value:
            cv2.line(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
        elif element_type == ElementType.BOX.value:
            if fill and alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.rectangle(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
        elif element_type == ElementType.BBOX.value:
            if fill and alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.rectangle(frame, tuple(map(int, coords[0])), tuple(map(int, coords[1])), color, thick)
            cv2.putText(
                frame,
                desc,
                (int(coords[0][0]), int(coords[0][1] - int(10 * size))),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                color,
                thick,
            )
        elif element_type == ElementType.CIRCLE.value:
            if fill and alpha > 0:
                overlay = frame.copy()
                cv2.circle(overlay, tuple(map(int, coords[0])), radius=max(1, int(size)), color=color, thickness=-1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.circle(frame, tuple(map(int, coords[0])), radius=max(1, int(size)), color=color, thickness=thick)
        elif element_type == ElementType.POLYGON.value:
            if fill and alpha > 0:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [np.array(coords, dtype=np.int32)], color=color)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.polylines(frame, [np.array(coords, dtype=np.int32)], isClosed=True, color=color, thickness=thick)
        elif element_type == ElementType.POLYLINES.value:
            cv2.polylines(frame, [np.array(coords, dtype=np.int32)], isClosed=False, color=color, thickness=thick)

    def draw_shapes(
        self,
        draw_file: str,
        output_file: str | None,
        shapes: list[dict] | dict,
        base_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add shapes to a draw file in DRAW_COLUMNS format.

        Parameters
        ----------
        draw_file : str
            Path to a draw CSV file. Used for reading existing data and as the
            default write target.
        output_file : str, optional
            Path to write the combined result. If None, writes to *draw_file*.
        shapes : list[dict] | dict
            One or more shape specifications. Each dict may contain:

            - ``type`` - ``"line"``, ``"polyline"``, ``"circle"``,
              ``"rectangle"``, or ``"polygon"``
            - ``geometry`` - coordinate pairs, a Shapely geometry, or a WKT string
            - ``color`` - BGR tuple, e.g. ``(0, 255, 0)`` *(optional)*
            - ``fill`` - whether to fill the shape *(optional)*
            - ``alpha`` - overlay strength 0.0-1.0; aliases: ``transparent``,
              ``transparant`` *(optional)*
            - ``size`` - font size or radius *(optional)*
            - ``thick`` - line thickness *(optional)*
            - ``desc`` - text description *(optional)*
        base_df : pd.DataFrame, optional
            Existing draw DataFrame to append to. If None and *draw_file*
            exists, that file is loaded instead.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with DRAW_COLUMNS.

        Raises
        ------
        ValueError
            If a shape specification is invalid (bad type, insufficient points,
            or unsupported geometry).

        """
        if isinstance(shapes, dict):
            shapes = [shapes]

        if base_df is not None:
            base_df = self._ensure_draw_columns(base_df.copy())
        elif os.path.exists(draw_file):
            base_df = pd.read_csv(
                draw_file,
                dtype={"frame": int, "type": str, "size": float, "desc": str, "thick": int},
                converters={"coords": lambda x: list(literal_eval(x)), "color": lambda x: literal_eval(x)},
            )
            base_df = self._ensure_draw_columns(base_df)
        else:
            base_df = pd.DataFrame(columns=DRAW_COLUMNS)

        def _parse_geom(geom: object) -> object:
            if isinstance(geom, str):
                try:
                    from shapely import wkt
                except ImportError as exc:
                    raise ValueError("WKT geometry requires shapely to be installed.") from exc
                return wkt.loads(geom)
            return geom

        def _as_pairs(geom: object) -> list[tuple[float, float]]:
            if hasattr(geom, "coords"):
                return [(float(x), float(y)) for x, y in geom.coords]
            if isinstance(geom, (list, tuple, np.ndarray)):
                vals = list(geom)
                if len(vals) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                    return [(float(vals[0]), float(vals[1]))]
                if len(vals) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                    return [(float(vals[0]), float(vals[1])), (float(vals[2]), float(vals[3]))]
                if (
                    len(vals) >= 2
                    and len(vals) % 2 == 0
                    and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals)
                ):
                    return [(float(vals[i]), float(vals[i + 1])) for i in range(0, len(vals), 2)]
                pairs: list[tuple[float, float]] = []
                for p in vals:
                    if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                        pairs.append((float(p[0]), float(p[1])))
                    else:
                        raise ValueError(f"Unsupported geometry point format: {p!r}")
                return pairs
            raise ValueError(f"Unsupported geometry coordinates: {type(geom)!r}")

        default_frames = sorted(base_df["frame"].astype(int).unique().tolist()) if len(base_df) > 0 else [0]

        if not shapes:
            target_file = output_file or draw_file
            if target_file:
                base_df.to_csv(target_file, index=False)
            return base_df

        rows: list[list[object]] = []
        for idx, shape in enumerate(shapes):
            if isinstance(shape, dict):
                shape_type = str(shape.get("type", "")).lower().strip()
                geom = _parse_geom(shape.get("geometry"))
                color = self._normalize_color(shape.get("color"), _bright_color(idx))
                fill = bool(shape.get("fill", False))
                alpha = float(shape.get("alpha", shape.get("transparent", shape.get("transparant", 0.0))))
                size = float(shape.get("size", 4.0))
                thick = int(shape.get("thick", 2))
                desc = str(shape.get("desc", ""))
            else:
                geom = _parse_geom(shape)
                shape_type = ""
                color = _bright_color(idx)
                fill = False
                alpha = 0.0
                size = 4.0
                thick = 2
                desc = ""

            alpha = min(max(alpha, 0.0), 1.0)
            target_frames = default_frames

            if not shape_type:
                if hasattr(geom, "geom_type"):
                    geom_type = str(geom.geom_type).lower()
                    if geom_type in {"linestring", "linearring"}:
                        shape_type = "polyline"
                    elif geom_type == "polygon":
                        shape_type = "polygon"
                    elif geom_type == "point":
                        shape_type = "circle"
                elif isinstance(geom, (list, tuple)) and len(geom) == 4:
                    shape_type = "rectangle"

            if shape_type == "line":
                coords = _as_pairs(geom)
                if len(coords) != 2:
                    raise ValueError("line requires exactly 2 points.")
                elem_type = ElementType.LINE.value
            elif shape_type in {"polyline", "polylines"}:
                coords = _as_pairs(geom)
                if len(coords) < 2:
                    raise ValueError("polyline requires at least 2 points.")
                elem_type = ElementType.POLYLINES.value
            elif shape_type == "circle":
                if hasattr(geom, "geom_type") and str(geom.geom_type).lower() == "point":
                    coords = [(float(geom.x), float(geom.y))]
                else:
                    pts = _as_pairs(geom)
                    if len(pts) != 1:
                        raise ValueError("circle requires a center point.")
                    coords = [pts[0]]
                elem_type = ElementType.CIRCLE.value
            elif shape_type in {"rectangle", "rect", "box"}:
                if hasattr(geom, "bounds"):
                    minx, miny, maxx, maxy = geom.bounds
                    coords = [(float(minx), float(miny)), (float(maxx), float(maxy))]
                else:
                    pts = _as_pairs(geom)
                    if len(pts) != 2:
                        raise ValueError("rectangle requires 2 corner points.")
                    coords = [pts[0], pts[1]]
                elem_type = ElementType.BOX.value
            elif shape_type == "polygon":
                if hasattr(geom, "geom_type") and str(geom.geom_type).lower() == "polygon":
                    coords = [(float(x), float(y)) for x, y in geom.exterior.coords]
                else:
                    coords = _as_pairs(geom)
                if len(coords) < 3:
                    raise ValueError("polygon requires at least 3 points.")
                if coords[0] != coords[-1]:
                    coords = [*coords, coords[0]]
                elem_type = ElementType.POLYGON.value
            else:
                raise ValueError(f"Unsupported shape type: {shape_type!r}")

            for frame in target_frames:
                rows.append([frame, elem_type, coords, color, size, thick, desc, fill, alpha])

        shape_df = pd.DataFrame(rows, columns=DRAW_COLUMNS)
        out_df = pd.concat([base_df, shape_df], ignore_index=True)

        out_df.sort_values(by="frame", inplace=True)
        target_file = output_file or draw_file
        if target_file:
            out_df.to_csv(target_file, index=False)

        return out_df

    def _apply_shapes_with_draw_shapes(self, df: pd.DataFrame, shapes: list[dict] | None) -> pd.DataFrame:
        """Apply shape overlays to an in-memory draw DataFrame via ``draw_shapes``.

        Parameters
        ----------
        df : pd.DataFrame
            Existing draw DataFrame.
        shapes : list[dict], optional
            Shape specifications (see ``draw_shapes`` for format). If None or
            empty, *df* is returned unchanged.

        Returns
        -------
        pd.DataFrame
            DataFrame with shape rows appended.

        """
        if not shapes:
            return df
        return self.draw_shapes(draw_file="", output_file=None, shapes=shapes, base_df=df)

    def draw(
        self,
        input_video: str,
        output_video: str,
        draws: pd.DataFrame | None = None,
        draw_file: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        verbose: bool = True,
    ):
        """Draw labels on video.

        Parameters
        ----------
        input_video : str
            Path to raw video file.
        output_video : str
            Path to output labeled video file.
        draws : pd.DataFrame, optional
            A DataFrame containing labeling information. If None, reads from draw_file.
        draw_file : str, optional
            A txt/csv file with header:
            ['frame','type','coords','color','size','thick','desc','fill','alpha']
        start_frame : int, optional
            Starting frame number. If None, defaults to 0.
        end_frame : int, optional
            Ending frame number. If None, defaults to last frame.
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.
        verbose : bool, optional
            Whether to show progress. Default is True.

        Raises
        ------
        OSError
            If the input video cannot be opened.
        RuntimeError
            If FFmpeg fails (when using ffmpeg or chrome_safe method).

        """
        if draws is not None:
            data = draws
        else:
            data = pd.read_csv(
                draw_file,
                dtype={"frame": int, "type": str, "size": float, "desc": str, "thick": int},
                converters={"coords": lambda x: list(literal_eval(x)), "color": lambda x: literal_eval(x)},
            )
        data = self._ensure_draw_columns(data)

        output_dir = os.path.dirname(output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        tot_frames = end_frame - start_frame + 1
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_to_elements = {k: g for k, g in data.groupby("frame", sort=False)}
        use_ffmpeg = self.method in {"ffmpeg", "chrome_safe"}
        process = None
        writer = None

        if self.method == "ffmpeg":
            # FFmpeg command to write H.265 encoded video
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                self.pix_fmt,
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",  # Read input from stdin
                "-c:v",
                self.encoder,  # H.265 codec
                "-preset",
                self.preset,  # Adjust preset as needed (ultrafast, fast, medium, slow, etc.)
                "-crf",
                str(self.crf),  # Constant Rate Factor (higher = more compression, lower = better quality)
                output_video,
            ]

            # Start FFmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        elif self.method == "chrome_safe":
            # FFmpeg command for browser-safe H.264 output.
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                self.pix_fmt,
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",  # Read input from stdin
                "-c:v",
                "libx264",  # H.264 codec
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                output_video,
            ]

            # Start FFmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        if verbose:
            pbar = tqdm(total=tot_frames, unit=" frames")
            if self.compress_message:
                pbar.set_description_str("Labeling")
            else:
                if video_index and video_tot:
                    pbar.set_description_str(f"Labeling {video_index} of {video_tot}")
                else:
                    pbar.set_description_str(f"Labeling {input_video} ")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame > end_frame):
                break

            elements = frame_to_elements.get(pos_frame)
            if elements is None:
                if use_ffmpeg:
                    try:
                        process.stdin.write(frame.tobytes())
                    except BrokenPipeError as exc:
                        ffmpeg_error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                        raise RuntimeError(
                            f"FFmpeg pipe closed early while writing frame {pos_frame}.\n{ffmpeg_error}"
                        ) from exc
                else:
                    writer.write(frame)
                if verbose:
                    pbar.update()
                continue

            for _, element in elements.iterrows():
                self._draw_element(
                    frame,
                    {
                        "type": element["type"],
                        "coords": element["coords"],
                        "color": element["color"],
                        "size": element["size"],
                        "thick": element["thick"],
                        "desc": element["desc"],
                        "fill": element["fill"],
                        "alpha": element["alpha"],
                    },
                )

            if use_ffmpeg:
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError as exc:
                    ffmpeg_error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                    raise RuntimeError(
                        f"FFmpeg pipe closed early while writing frame {pos_frame}.\n{ffmpeg_error}"
                    ) from exc
            else:
                writer.write(frame)

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()
        # cv2.destroyAllWindows()
        cap.release()
        if use_ffmpeg:
            if process.stdin:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                ffmpeg_error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(f"FFmpeg failed with exit code {return_code}.\n{ffmpeg_error}")
        else:
            writer.release()

    def draw_track_clips(
        self,
        input_video: str,
        output_path: str,
        tracks: pd.DataFrame | None = None,
        track_file: str | None = None,
        method: TrackClipMethod | str = TrackClipMethod.ALL,
        random_number: int = 10,
        track_ids: list | None = None,
        start_frame_offset: int = 0,
        end_frame_offset: int = 0,
        tail_length: int = 0,
        label_prefix: bool = False,
        label_class: bool = False,
        shapes: list[dict] | None = None,
        color: tuple[int, int, int] | str | None = None,
        fill: bool = False,
        alpha: float = 0.0,
        size: int = 1,
        thick: int = 1,
        tail_size: int | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        verbose: bool = True,
    ):
        """Draw track clips from video.

        Parameters
        ----------
        input_video : str
            The raw video file.
        output_path : str
            The folder for outputting track clips.
        tracks : pd.DataFrame, optional
            The dataframe of tracks. If None, reads from track_file.
        track_file : str, optional
            The track file if tracks is None.
        method : TrackClipMethod | str
            'all' (default) - all tracks
            'random' - random select tracks
            'specify' - specify track ids
        random_number : int
            The number of track ids if method == 'random'. Default is 10.
        track_ids : list, optional
            The list of track ids if method == 'specify'.
        start_frame_offset : int
            The offset of start frame. Default is 0.
        end_frame_offset : int
            The offset of end frame. Default is 0.
        tail_length : int
            The tail length. Default is 0.
            Use `-1` to draw full history since the first frame of each track.
        label_prefix : bool
            If True, add the video file name as the prefix in output file names. Default is False.
        size : int
            Font size. Default is 1.
        thick : int
            Line thickness. Default is 1.
        tail_size : int, optional
            Radius of tail dots. If None, an auto-scaled visible radius is used.
        label_class : bool
            Whether to include class in track labels. Default is False.
        shapes : list[dict], optional
            Shape overlays to add before rendering. See ``draw_shapes`` for
            the expected dict format.
        color : tuple[int, int, int] | str, optional
            Clip color control:
            - None: default per-track color scheme
            - "random": random BGR color per clip
            - (b, g, r): fixed BGR color for all clips
        fill : bool
            Whether to fill generated track boxes. Default is False.
        alpha : float
            Overlay strength for fill in [0, 1]. Default is 0.0.
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.
        verbose : bool
            If True, show progress bar. Default is True.

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        if tracks is None:
            tracks = pd.read_csv(
                track_file,
                header=None,
                dtype={0: int, 1: int, 2: int, 3: int, 4: int, 5: int, 6: float, 7: int, 8: int, 9: int},
            )
        tracks.columns = TRACK_COLUMNS

        if isinstance(method, TrackClipMethod):
            track_clip_method = method.value
        else:
            track_clip_method = str(method).lower().strip()
            valid_methods = {m.value for m in TrackClipMethod}
            if track_clip_method not in valid_methods:
                raise ValueError(f"Invalid method={method!r}. Choose one of {sorted(valid_methods)}.")

        if track_clip_method == TrackClipMethod.RANDOM.value:
            track_ids = tracks["track"].unique().tolist()
            if random_number <= 0:
                random_number = 10
            random_number = min(random_number, len(track_ids))
            track_ids = random.sample(track_ids, random_number)
        elif track_clip_method == TrackClipMethod.SPECIFY.value:
            if (track_ids is None) or (len(track_ids) == 0):
                print("No tracks are provided!")
                return pd.DataFrame()
        else:
            track_ids = tracks["track"].unique().tolist()

        # pbar = tqdm(total=len(track_ids), desc='Labeling tracks ', unit='clips')
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        for id in track_ids:
            selected_tracks = tracks[tracks["track"] == id].copy()
            start_frame = max(selected_tracks["frame"].min() - start_frame_offset, 0)
            end_frame = min(
                selected_tracks["frame"].max() + end_frame_offset,
                frame_count - 1,
            )
            if label_prefix:
                out_video = os.path.join(
                    output_path, os.path.splitext(os.path.basename(input_video))[0] + "_" + str(id) + ".mp4"
                )
            else:
                out_video = os.path.join(output_path, str(id) + ".mp4")

            if isinstance(color, str):
                color_mode = color.lower().strip()
                if color_mode == "random":
                    clip_color = random.choice(BRIGHT_COLORS_BGR)
                elif color_mode in {"none", "default"}:
                    clip_color = None
                else:
                    raise ValueError("color must be None, 'random', or a BGR tuple like (0, 255, 0).")
            elif color is None:
                clip_color = None
            else:
                clip_color = self._normalize_color(color)

            self.draw_tracks(
                input_video=input_video,
                output_video=out_video,
                tracks=selected_tracks,
                color=clip_color,
                start_frame=start_frame,
                end_frame=end_frame,
                verbose=verbose,
                tail_length=tail_length,
                thick=thick,
                size=size,
                tail_size=tail_size,
                label_class=label_class,
                shapes=shapes,
                fill=fill,
                alpha=alpha,
                video_index=video_index,
                video_tot=video_tot,
            )

            # pbar.update()
        # pbar.close()

    def draw_tracks(
        self,
        input_video: str,
        output_video: str,
        tracks: pd.DataFrame | None = None,
        track_file: str | None = None,
        label_file: str | None = None,
        color: tuple[int, int, int] | None = None,
        thick: int = 2,
        size: int = 1,
        tail_length: int = 0,
        tail_size: int | None = None,
        label_class: bool = False,
        shapes: list[dict] | None = None,
        fill: bool = False,
        alpha: float = 0.0,
        start_frame: int | None = None,
        end_frame: int | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
        verbose: bool = True,
    ):
        """Draw tracks on video.

        Parameters
        ----------
        input_video : str
            Path to raw video file.
        output_video : str
            Path to output labeled video file.
        tracks : pd.DataFrame, optional
            DataFrame containing track data. If None, reads from track_file.
        track_file : str, optional
            Path to track file if tracks is None.
        label_file : str, optional
            Path to save label output.
        color : tuple[int, int, int], optional
            Custom color for drawing. If None, uses default colormap.
        tail_length : int
            Length of tail to display. Default is 0.
            Use `-1` to draw full history since the first frame of each track.
        tail_size : int, optional
            Radius of tail dots. If None, an auto-scaled visible radius is used.
        thick : int
            Line thickness. Default is 2.
        size : int
            Font size. Default is 1.
        label_class : bool
            Whether to display class names. Default is False.
        shapes : list[dict], optional
            Shape overlays to add before rendering. See ``draw_shapes`` for
            the expected dict format.
        fill : bool
            Whether to fill generated track boxes. Default is False.
        alpha : float
            Overlay strength for fill in [0, 1]. Default is 0.0.
        start_frame : int, optional
            Starting frame. If None, defaults to 0.
        end_frame : int, optional
            Ending frame. If None, defaults to last frame.
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.
        verbose : bool
            Whether to show progress. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing generated draw elements (with DRAW_COLUMNS).

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        if tracks is None:
            tracks = pd.read_csv(
                track_file,
                header=None,
                dtype={0: int, 1: int, 2: int, 3: int, 4: int, 5: int, 6: float, 7: int, 8: int, 9: int},
            )
        tracks.columns = TRACK_COLUMNS

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        selected_tracks = tracks.loc[(tracks["frame"] >= start_frame) & (tracks["frame"] <= end_frame)].copy()

        pbar_desc = ""
        if self.compress_message:
            pbar_desc = "Generating labels"
        else:
            if video_index and video_tot:
                pbar_desc = f"Generating labels {video_index} of {video_tot}"
            else:
                pbar_desc = f"Generating labels {input_video} "

        names = load_classes()
        pbar = tqdm(total=len(selected_tracks), unit=" frames", desc=pbar_desc, disable=not verbose)
        tail_radius = max(1, int((size) * 1)) if tail_size is None else max(1, int(tail_size))
        alpha = min(max(float(alpha), 0.0), 1.0)
        results = []
        for _, track in selected_tracks.iterrows():
            final_color = _bright_color(int(track["track"])) if color is None else color

            track_id = int(track["track"])
            cls_desc = names[int(track["cls"])] if label_class else ""
            desc = f"{track_id} {cls_desc}"
            results.append([
                track["frame"],
                ElementType.BBOX.value,
                [(track["x"], track["y"]), (track["x"] + track["w"], track["y"] + track["h"])],
                final_color,
                size,
                thick,
                desc,
                fill,
                alpha,
            ])
            if tail_length != 0:
                current_frame = int(track["frame"])
                if tail_length == -1:
                    pre_boxes = tracks.loc[
                        (tracks["track"] == track["track"]) & (tracks["frame"] < current_frame)
                    ].values.tolist()
                elif tail_length > 0:
                    frames = [*range(current_frame - tail_length, current_frame)]
                    pre_boxes = tracks.loc[
                        (tracks["frame"].isin(frames)) & (tracks["track"] == track["track"])
                    ].values.tolist()
                else:
                    pre_boxes = []

                if len(pre_boxes) > 0:
                    for pre_box in pre_boxes:
                        xc = int(pre_box[2]) + int(pre_box[4] / 2)
                        yc = int(pre_box[3]) + int(pre_box[5] / 2)
                        results.append([
                            track["frame"],
                            ElementType.CIRCLE.value,
                            [(xc, yc)],
                            final_color,
                            tail_radius,
                            -1,
                            "",
                            fill,
                            alpha,
                        ])

            pbar.update()

        pbar.close()

        results.sort()
        results = list(results for results, _ in itertools.groupby(results))
        df = pd.DataFrame(results, columns=DRAW_COLUMNS)
        df = self._apply_shapes_with_draw_shapes(df, shapes)
        df.sort_values(by="frame", inplace=True)

        if output_video:
            self.draw(
                input_video=input_video,
                output_video=output_video,
                draws=df,
                start_frame=start_frame,
                end_frame=end_frame,
                video_index=video_index,
                video_tot=video_tot,
                verbose=verbose,
            )

        if label_file:
            df.to_csv(label_file, index=False)

        return df

    def draw_dets(
        self,
        input_video: str,
        output_video: str,
        dets: pd.DataFrame | None = None,
        det_file: str | None = None,
        label_file: str | None = None,
        color: tuple[int, int, int] | None = None,
        thick: int = 2,
        size: int = 1,
        label_score: bool = True,
        shapes: list[dict] | None = None,
        fill: bool = False,
        alpha: float = 0.0,
        start_frame: int | None = None,
        end_frame: int | None = None,
        video_index: int | None = None,
        video_tot: int | None = None,
    ):
        """Draw detections on video.

        Parameters
        ----------
        input_video : str
            Path to raw video file.
        output_video : str
            Path to output labeled video file.
        dets : pd.DataFrame, optional
            DataFrame containing detection data. If None, reads from det_file.
        det_file : str, optional
            Path to detection file if dets is None.
        label_file : str, optional
            Path to save label output.
        color : tuple[int, int, int], optional
            Custom color for drawing. If None, uses default colormap.
        thick : int
            Line thickness. Default is 2.
        size : int
            Font size. Default is 1.
        label_score : bool
            Whether to display detection scores. Default is True.
        shapes : list[dict], optional
            Shape overlays to add before rendering. See ``draw_shapes`` for
            the expected dict format.
        fill : bool
            Whether to fill generated detection boxes. Default is False.
        alpha : float
            Overlay strength for fill in [0, 1]. Default is 0.0.
        start_frame : int, optional
            Starting frame. If None, defaults to 0.
        end_frame : int, optional
            Ending frame. If None, defaults to last frame.
        video_index : int, optional
            Video index for batch processing display.
        video_tot : int, optional
            Total video count for batch processing display.

        Returns
        -------
        pd.DataFrame
            DataFrame containing generated draw elements (with DRAW_COLUMNS).

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        if dets is None:
            dets = pd.read_csv(det_file, header=None)

        names = load_classes()

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        selected_dets = dets.loc[(dets[0] >= start_frame) & (dets[0] <= end_frame)].copy()

        pbar = tqdm(total=len(selected_dets), unit=" dets")
        if self.compress_message:
            pbar.set_description_str("Generating labels")
        else:
            if video_index and video_tot:
                pbar.set_description_str(f"Generating labels {video_index} of {video_tot}")
            else:
                pbar.set_description_str(f"Generating labels {input_video} ")

        results = []
        alpha = min(max(float(alpha), 0.0), 1.0)
        for _, det in selected_dets.iterrows():
            final_color = _bright_color(int(det[7])) if color is None else color

            desc = f"{names[int(det[7])]} {det[6]:.1f}" if label_score else str(int(det[7]))

            results.append([
                det[0],
                ElementType.BBOX.value,
                [(det[2], det[3]), (det[2] + det[4], det[3] + det[5])],
                final_color,
                size,
                thick,
                desc,
                fill,
                alpha,
            ])
            pbar.update()

        results.sort()
        results = list(results for results, _ in itertools.groupby(results))
        df = pd.DataFrame(results, columns=DRAW_COLUMNS)
        df = self._apply_shapes_with_draw_shapes(df, shapes)
        df.sort_values(by="frame", inplace=True)

        if output_video:
            self.draw(
                input_video=input_video,
                output_video=output_video,
                draws=df,
                start_frame=start_frame,
                end_frame=end_frame,
                video_index=video_index,
                video_tot=video_tot,
            )

        if label_file:
            df.to_csv(label_file, index=False)

        return df

    def clip(
        self,
        input_video: str,
        output_video: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        method: LabelMethod | str | None = None,
    ):
        """Extract a clip from the video.

        Parameters
        ----------
        input_video : str
            Path to input video file.
        output_video : str
            Path to output clipped video file.
        start_frame : int, optional
            Starting frame. If None, defaults to 0.
        end_frame : int, optional
            Ending frame. If None, defaults to last frame.
        method : LabelMethod | str, optional
            Output backend for clipping. If None, uses the instance method.
            Supported: 'opencv', 'ffmpeg', 'chrome_safe'.

        Raises
        ------
        OSError
            If the input video cannot be opened.
        ValueError
            If frame range is invalid or video has no frames.
        RuntimeError
            If FFmpeg fails (when using ffmpeg or chrome_safe method).

        """
        if method is None:
            clip_method = self.method
        elif isinstance(method, LabelMethod):
            clip_method = method.value
        else:
            clip_method = str(method).lower().strip()
            valid_methods = {m.value for m in LabelMethod}
            if clip_method not in valid_methods:
                raise ValueError(f"Invalid method={method!r}. Choose one of {sorted(valid_methods)}.")

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"No frames available in video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS ({fps}) for video: {input_video}")

        if start_frame is None:
            start_frame = 0

        if end_frame is None:
            end_frame = total_frames - 1

        start_frame = max(0, min(int(start_frame), total_frames - 1))
        end_frame = max(0, min(int(end_frame), total_frames - 1))
        if end_frame < start_frame:
            end_frame = start_frame

        tot_frames = end_frame - start_frame + 1
        fps = int(fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_dir = os.path.dirname(output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Keep clip output bitrate aligned with source when possible.
        source_bitrate_bps: int | None = None
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=bit_rate",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    input_video,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                value = probe.stdout.strip()
                if value.isdigit():
                    source_bitrate_bps = int(value)
        except FileNotFoundError:
            source_bitrate_bps = None

        if source_bitrate_bps is None:
            cap_bitrate_kbps = cap.get(cv2.CAP_PROP_BITRATE)
            if cap_bitrate_kbps and cap_bitrate_kbps > 0:
                source_bitrate_bps = int(cap_bitrate_kbps * 1000)

        bitrate_args: list[str] = []
        if source_bitrate_bps and source_bitrate_bps > 0:
            bitrate_args = [
                "-b:v",
                str(source_bitrate_bps),
                "-minrate",
                str(source_bitrate_bps),
                "-maxrate",
                str(source_bitrate_bps),
                "-bufsize",
                str(source_bitrate_bps * 2),
            ]

        use_ffmpeg = clip_method in {LabelMethod.FFMPEG.value, LabelMethod.CHROME_SAFE.value}
        process = None
        writer = None

        if clip_method == LabelMethod.FFMPEG.value:
            rate_control_args = bitrate_args if bitrate_args else ["-crf", str(self.crf)]
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                self.pix_fmt,
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                self.encoder,
                "-preset",
                self.preset,
                *rate_control_args,
                output_video,
            ]
            process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        elif clip_method == LabelMethod.CHROME_SAFE.value:
            rate_control_args = bitrate_args if bitrate_args else ["-crf", "23"]
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                self.pix_fmt,
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-preset",
                "medium",
                *rate_control_args,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                output_video,
            ]
            process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        pbar = tqdm(total=tot_frames, unit=" frames")
        if self.compress_message:
            pbar.set_description_str("Cutting")
        else:
            pbar.set_description_str(f"Cutting {input_video} ")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while cap.isOpened():
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if (not ret) or (pos_frame > end_frame):
                break

            if use_ffmpeg:
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError as exc:
                    ffmpeg_error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                    raise RuntimeError(
                        f"FFmpeg pipe closed early while clipping frame {pos_frame}.\n{ffmpeg_error}"
                    ) from exc
            else:
                writer.write(frame)

            pbar.update()

        pbar.close()
        cap.release()
        if use_ffmpeg:
            if process.stdin:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                ffmpeg_error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(f"FFmpeg failed with exit code {return_code}.\n{ffmpeg_error}")
        else:
            writer.release()

    def clip_by_time(
        self,
        input_video: str,
        output_video: str,
        start_sec: float = 0.0,
        clip_len_sec: float | None = None,
        method: LabelMethod | str | None = None,
    ):
        """Extract a clip by time range.

        Parameters
        ----------
        input_video : str
            Path to input video file.
        output_video : str
            Path to output clipped video file.
        start_sec : float
            Starting second. Negative values are treated as 0.
        clip_len_sec : float, optional
            Clip length in seconds. If None, clip runs to end of video.
        method : LabelMethod | str, optional
            Output backend for clipping. If None, uses the instance method.
            Supported: 'opencv', 'ffmpeg', 'chrome_safe'.

        Raises
        ------
        OSError
            If the input video cannot be opened.
        ValueError
            If the video has no frames, FPS is invalid, or clip_len_sec <= 0.

        """
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"No frames available in video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS ({fps}) for video: {input_video}")

        start_frame = int(max(0.0, start_sec) * fps)
        if clip_len_sec is None:
            end_frame = total_frames - 1
        else:
            if clip_len_sec <= 0:
                cap.release()
                raise ValueError(f"clip_len_sec must be > 0, got {clip_len_sec}.")
            clip_len_frames = max(1, round(clip_len_sec * fps))
            end_frame = start_frame + clip_len_frames - 1

        cap.release()

        self.clip(
            input_video=input_video,
            output_video=output_video,
            start_frame=start_frame,
            end_frame=end_frame,
            method=method,
        )

    @staticmethod
    def export_frames(input_video: str, frames: list[int], output_path: str, prefix: str | None = None):
        """Extract specific frames from video.

        Parameters
        ----------
        input_video : str
            Path to input video file.
        frames : list[int]
            List of frame numbers to extract.
        output_path : str
            Path to output directory for extracted frames.
        prefix : str, optional
            Prefix to add to output filenames. If None, no prefix is used.

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        pbar = tqdm(total=len(frames), unit=" frames")
        pbar.set_description_str("Extracting frame")

        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_read = cap.read()

            if prefix is None:
                frame_file = os.path.join(output_path, str(frame) + ".jpg")
            else:
                frame_file = os.path.join(output_path, prefix + "-" + str(frame) + ".jpg")

            if ret:
                cv2.imwrite(frame_file, frame_read)
            else:
                break

            pbar.update()

        pbar.close()
        cap.release()

        print(f"Writing frames to {output_path}")

    @staticmethod
    def export_track_frames(
        input_video: str,
        tracks: pd.DataFrame,
        output_path: str,
        bbox: bool = True,
        prefix: str | None = None,
        thick: int = 2,
    ):
        """Extract frames for each track from video.

        Parameters
        ----------
        input_video : str
            Path to input video file.
        tracks : pd.DataFrame
            DataFrame containing track data with TRACK_COLUMNS.
        output_path : str
            Path to output directory for extracted frames.
        bbox : bool
            Whether to draw bounding boxes. Default is True.
        prefix : str, optional
            Prefix to add to output filenames. If None, no prefix is used.
        thick : int
            Line thickness for bounding boxes. Default is 2.

        Raises
        ------
        OSError
            If the input video cannot be opened.
        Exception
            If tracks DataFrame has invalid format.

        """
        if (tracks is None) or (len(tracks.columns) < 10):
            raise ValueError("Invalid tracks: DataFrame must have at least 10 columns.")
        tracks.columns = TRACK_COLUMNS
        ids = tracks["track"].unique()
        os.makedirs(output_path, exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        pbar = tqdm(total=len(ids), unit=" frame")
        for id in ids:
            pbar.set_description_str("Extracting track: " + str(id))
            selected = tracks[tracks["track"] == id]
            if len(selected) > 0:
                for _, track in selected.iterrows():
                    frame = track["frame"]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, img = cap.read()

                    if ret:
                        if bbox:
                            x1 = track["x"]
                            y1 = track["y"]
                            x2 = track["x"] + track["w"]
                            y2 = track["y"] + track["h"]
                            final_color = _bright_color(int(id))
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), final_color, thick)

                            if prefix is None:
                                frame_file = os.path.join(output_path, str(id) + "_" + str(frame) + ".jpg")
                            else:
                                frame_file = os.path.join(
                                    output_path, prefix + "-" + str(id) + "_" + str(frame) + ".jpg"
                                )

                            cv2.imwrite(frame_file, img)
                    else:
                        break

                pbar.update()

        pbar.close()
        cap.release()

        print(f"Writing frames to {output_path}")

    @staticmethod
    def time2frame(input_video: str, time: float):
        """Convert time in seconds to frame number.

        Parameters
        ----------
        input_video : str
            Path to input video file.
        time : float
            Time in seconds.

        Returns
        -------
        int
            Frame number corresponding to the given time.

        Raises
        ------
        OSError
            If the input video cannot be opened.

        """
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise OSError("Couldn't open webcam or video")

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))  # original fps
        frame = int(video_fps * time)
        cap.release()
        return frame


if __name__ == "__main__":
    pass
