"""Per-stage throughput on a clip (spec §5.4). Prints one markdown table row.

    .venv/bin/python tools/bench.py --video CLIP [--stages detect,track,post] [--dets D] [--tracks T] [--label X]
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_cases as gc  # noqa: E402

STAGES = ("detect", "track", "post")


def plan_stages(stages: list[str], dets: Path | None, tracks: Path | None) -> None:
    """Raise ValueError unless every selected stage has its input (review item 6)."""
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {STAGES}")
    if "track" in stages and "detect" not in stages and dets is None:
        raise ValueError("--stages track without detect needs --dets")
    if "post" in stages and "track" not in stages and tracks is None:
        raise ValueError("--stages post without track needs --tracks")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--stages", default="detect,track,post")
    p.add_argument("--dets", type=Path, help="detection file when 'detect' is not selected")
    p.add_argument("--tracks", type=Path, help="ByteTrack track file when 'post' runs without 'track'")
    p.add_argument("--label", default="")
    a = p.parse_args()
    stages = [s for s in a.stages.split(",") if s]
    try:
        plan_stages(stages, a.dets, a.tracks)
    except ValueError as exc:
        p.error(str(exc))

    import dnt
    import torch

    cap = cv2.VideoCapture(str(a.video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    work = Path(tempfile.mkdtemp())
    fps, t_total = {}, time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dets = a.dets
    if "detect" in stages:
        dets = work / "dets.txt"
        t = time.perf_counter()
        gc.run_detect(a.video, dets)
        fps["detect"] = frames / (time.perf_counter() - t)
    tracks = a.tracks
    if "track" in stages:
        for case in ("bytetrack", "botsort"):
            t = time.perf_counter()
            gc.run_track(gc.CONFIG_NAMES[case], {}, dets, a.video, work / f"track_{case}.txt")
            fps[f"track:{case}"] = frames / (time.perf_counter() - t)
        tracks = work / "track_bytetrack.txt"
    if "post" in stages:
        t = time.perf_counter()
        gc.run_post(tracks, work, "bytetrack")
        fps["post"] = frames / (time.perf_counter() - t)

    mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    cells = " | ".join(f"{k} {v:.1f}" for k, v in fps.items())
    print(f"| {a.label or dnt.__version__} | {cells} | {mem:.0f} MiB | {time.perf_counter() - t_total:.1f} s |")


if __name__ == "__main__":
    main()
