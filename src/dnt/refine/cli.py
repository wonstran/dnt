"""Command-line entry point ``dnt-refine`` (spec 2.4)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import BACKENDS, RefineConfig
from .refiner import TrackRefiner

#: The ``vlm`` config fields that a ``--vlm-*`` option overrides (``--vlm-base-url``: base_url).
_VLM_FIELDS = ("backend", "model", "base_url", "api_key_env", "api_key_file")


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
    vlm = run.add_argument_group(
        "VLM verification",
        "override the config's vlm block; there is no option for the key itself (it would be "
        "kept in the shell history and shown in the process list)",
    )
    vlm.add_argument("--vlm-backend", choices=BACKENDS, help="vlm.backend")
    vlm.add_argument("--vlm-model", help="vlm.model")
    vlm.add_argument(
        "--vlm-base-url", help="vlm.base_url: the endpoint (a local server, a proxy, a gateway)"
    )
    vlm.add_argument(
        "--vlm-api-key-env", help="vlm.api_key_env: the NAME of the variable that holds the key"
    )
    vlm.add_argument("--vlm-api-key-file", help="vlm.api_key_file: a file that holds the key")
    return p


def load_config(args: argparse.Namespace) -> RefineConfig:
    """Return the config of ``args.config`` with the ``--vlm-*`` options applied and validated.

    An empty value (``--vlm-base-url ""``) resets the field to its default.
    """
    cfg = RefineConfig.from_yaml(args.config)
    for name in _VLM_FIELDS:
        value = getattr(args, f"vlm_{name}", None)
        if value is not None:
            setattr(cfg.vlm, name, value)
    cfg.validate()
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Run ``dnt-refine``; return the process exit code.

    Returns 0 on success (a JSON summary goes to stdout) and 2 when the inputs are invalid
    (``ValueError``, or ``OSError`` such as ``FileNotFoundError`` or a Hugging Face Hub
    download that fails offline) or the encoder's package is missing (``ImportError``); the
    message goes to stderr. That includes a bad ``--vlm-base-url``, an unreadable
    ``--vlm-api-key-file``, and a missing key; no message holds a key.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        refiner = TrackRefiner(config=load_config(args))
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
            {
                "out": args.out,
                "ledger": str(res.ledger_path),
                "review": None if res.review_path is None else str(res.review_path),
                "summary": res.summary,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
