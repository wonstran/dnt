"""Command-line entry point ``dnt-refine`` (spec 2.4)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .refiner import TrackRefiner


def build_parser() -> argparse.ArgumentParser:
    """Return the ``dnt-refine`` argument parser."""
    p = argparse.ArgumentParser(
        prog="dnt-refine",
        description="Refine a track file: split ID switches, screen false tracks, link "
        "fragments, drop orphans, and fill gaps.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="refine a track file")
    run.add_argument("tracks", help="headerless dnt (10-column) or MOT track file")
    run.add_argument("--video", help="source video (frame rate, frame size, and later evidence)")
    run.add_argument("--fps", type=float, help="frame rate; required when there is no video")
    run.add_argument("--context", help="context file: dnt tracks (10 columns) or detections (8)")
    run.add_argument("--reclass-hints", dest="reclass_hints", help="ReClass output CSV")
    run.add_argument("--config", required=True, help="RefineConfig YAML")
    run.add_argument("--out", required=True, help="output track file")
    run.add_argument("--format", choices=("dnt", "mot"), default="dnt", help="input format")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run ``dnt-refine``; return the process exit code.

    Returns 0 on success (a JSON summary goes to stdout) and 2 when the inputs are invalid
    (``ValueError``, or ``OSError`` such as ``FileNotFoundError`` or a Hugging Face Hub
    download that fails offline) or the encoder's package is missing (``ImportError``); the
    message goes to stderr.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        refiner = TrackRefiner(config_yaml=args.config)
        refiner.refine(
            args.tracks,
            args.out,
            video_file=args.video,
            context_file=args.context,
            reclass_file=args.reclass_hints,
            fps=args.fps,
            fmt=args.format,
            verbose=False,
        )
        res = refiner.last_result
    except (ValueError, OSError, ImportError) as exc:
        print(f"dnt-refine: error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"out": args.out, "ledger": str(res.ledger_path), "summary": res.summary},
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
