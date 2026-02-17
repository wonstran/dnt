import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pytz import timezone
from tqdm.auto import tqdm

from dnt.detect import Detector


class Synchronizer:
    """
    Synchronizes multiple videos based on reference frame/time and offsets.
    """

    def __init__(
        self,
        videos: List[str],
        ref_frame: int,
        ref_time: int,  # unix ms
        ref_timezone: str = "US/Eastern",
        offsets: Optional[List[int]] = None,  # offsets in FRAMES
    ) -> None:
        """
        Args:
            videos: List of video file paths.
            ref_frame: Reference frame number.
            ref_time: Reference time in unix ms.
            ref_timezone: Timezone string for reference time.
            offsets: Optional list of frame offsets for each video.
        """
        self.videos = videos
        self.ref_frame = int(ref_frame)
        self.ref_time = int(ref_time)
        self.ref_tz_str = ref_timezone
        self.ref_tz = timezone(ref_timezone)
        self.offsets = offsets

    def process(self, output_path: Optional[str] = None, local: bool = False, message: bool = False) -> pd.DataFrame:
        """
        Process all videos, synchronizing frames to unix time and optionally saving results.

        Args:
            output_path: Optional directory to save CSVs.
            local: If True, add local time column.
            message: If True, print progress messages.
        Returns:
            DataFrame with frame, unix_time, video, and optionally local_time.
        """
        if not self.videos:
            return pd.DataFrame(columns=["frame", "unix_time", "video"] + (["local_time"] if local else []))

        offsets = self.offsets if self.offsets is not None else [0] * len(self.videos)
        if len(offsets) != len(self.videos):
            raise ValueError(f"offsets length ({len(offsets)}) must match videos length ({len(self.videos)})")

        results = []
        ref_frame = self.ref_frame
        ref_time = self.ref_time  # ms

        for i, video in enumerate(self.videos):
            df, fps = self.add_unix_time(
                video, ref_frame, ref_time, message=message, video_index=i + 1, video_tot=len(self.videos)
            )

            if local:
                # unix_time is epoch ms -> convert (vectorized)
                dt = pd.to_datetime(df["unix_time"], unit="ms", utc=True).dt.tz_convert(self.ref_tz_str)
                df["local_time"] = dt.astype(str)

            results.append(df)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                basename = os.path.splitext(os.path.basename(video))[0]
                df.to_csv(os.path.join(output_path, f"{basename}_time.csv"), index=False)

            # Prepare next video's reference time
            if i < len(self.videos) - 1:
                next_video = self.videos[i + 1]
                next_fps = Detector.get_fps(next_video)
                if next_fps <= 0:
                    raise ValueError(f"fps is invalid for {next_video}")

                if df.empty:
                    raise ValueError(f"DataFrame for video {video} is empty; cannot determine last unix_time.")

                ref_frame = 0

        if not results:
            return pd.DataFrame(columns=["frame", "unix_time", "video"] + (["local_time"] if local else []))

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def add_unix_time(
        video: str,
        ref_frame: int,
        ref_time: int,
        video_index: Optional[int] = None,
        video_tot: Optional[int] = None,
        message: bool = False,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Compute unix time for each frame in a video based on a reference frame and time.

        Parameters
        ----------
        video : str
            Video file path.
        ref_frame : int
            Reference frame number.
        ref_time : int
            Reference time in unix milliseconds.
        video_index : int, optional
            Index of the video for progress messages (default is None).
        video_tot : int, optional
            Total number of videos for progress messages (default is None).
        message : bool, optional
            If True, print progress message (default is False).

        Returns
        -------
        tuple of (pd.DataFrame, float)
            DataFrame with columns [frame, unix_time, video], and the video's frames-per-second (fps).

        Raises
        ------
        ValueError
            If fps is zero or negative, or if video frame count is invalid.
        """
        fps = Detector.get_fps(video)
        ms_per_frame = 1000.0 / fps
        n_frames = int(Detector.get_frames(video))
        frames = np.arange(n_frames, dtype=np.int64)
        unix_time = np.round(ref_time + (frames - int(ref_frame)) * ms_per_frame).astype(np.int64)
        df = pd.DataFrame({"frame": frames, "unix_time": unix_time, "video": video})

        if message:
            label = f"{video_index} of {video_tot}" if (video_index and video_tot) else video
            tqdm.write(f"Synced {video} ({n_frames} frames @ {fps:.3f} fps) [{label}]")

        return df, fps

    @staticmethod
    def convert_unix_local(unix_time: int, ref_timezone: str = "US/Eastern") -> str:
        """
        Convert unix time in ms to ISO8601 string in the given timezone.
        """
        tz = timezone(ref_timezone)
        return datetime.fromtimestamp(unix_time / 1000.0, tz).isoformat()
