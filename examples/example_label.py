"""Example script for labeling video tracks using the Labeler class."""

import os
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from dnt.label import Encoder, Labeler, LabelMethod, TrackClipMethod


def main() -> None:
    """Label video tracks using the Labeler class with FFMPEG method."""
    input_video = "/mnt/d/videos/sample/traffic.mp4"
    det_file = "/mnt/d/videos/sample/dets/traffic_det.txt"
    track_file = "/mnt/d/videos/sample/tracks/traffic_track.txt"
    track_label = "/mnt/d/videos/sample/tracks/traffic_label.txt"
    det_video = "/mnt/d/videos/sample/labels/traffic_det.mp4"
    track_video = "/mnt/d/videos/sample/labels/traffic_track.mp4"
    clip_video = "/mnt/d/videos/sample/traffic_clip.mp4"

    track_clip_output = "/mnt/d/videos/sample/labels/track_clips"
    os.makedirs(track_clip_output, exist_ok=True)

    labeler = Labeler(method=LabelMethod.CHROME_SAFE, encoder=Encoder.LIBX265)
    """
    labeler.draw_tracks(
        input_video=input_video,
        output_video=track_video,
        track_file=track_file,
        label_class_name=True,
        thick=1,
        size=0.8,
    )
    """
    # labeler.draw_dets(input_video, det_video, det_file=det_file, label_score=True, thick=1, size=0.8)
    shapes = [
        {
            "type": "polygon",
            "geometry": [(200, 200), (400, 200), (400, 400)],
            "fill": True,
            "color": (255, 0, 0),
            "alpha": 0.2,
            "size": 20,
            "thick": 1,
        },
    ]

    labeler.draw_track_clips(
        input_video=input_video,
        output_path=track_clip_output,
        track_file=track_file,
        method=TrackClipMethod.RANDOM,
        random_number=5,
        tail_length=0,
        label_class=True,
        label_prefix=True,
        thick=1,
        size=0.8,
        fill=True,
        alpha=0.2,
        color="random",
    )

    # labeler.clip_by_time(input_video=input_video, output_video=clip_video, clip_len_sec=5.0, method=LabelMethod.FFMPEG)
    # frames = range(800,1000)
    # labeler.export_frames(input_video, frames, "/mnt/d/videos/ped2stage/frames")
    print("ok")


if __name__ == "__main__":
    main()
