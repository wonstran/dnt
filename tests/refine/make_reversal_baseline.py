"""Write the reversal-stage baseline: the refine output of the code BEFORE the stage existed.

Run ONCE, before any reversal code is written (plan Task 1):

    .venv/bin/python tests/refine/make_reversal_baseline.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _fixtures import DATA, reversal_scene  # noqa: E402

from dnt.refine import RefineConfig, TrackRefiner  # noqa: E402

OUT = DATA / "reversal_baseline"


def main() -> None:
    """Refine the scene with the vehicle defaults, motion only, at 10 fps."""
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw.csv"
    reversal_scene().to_csv(raw, index=False, header=False)
    cfg = RefineConfig.defaults("vehicle")
    cfg.encoder.kind = "none"
    refiner = TrackRefiner(cfg)
    refiner.refine(raw, OUT / "tracks.csv", fps=10, verbose=False)
    shutil.move(str(refiner.last_result.ledger_path), OUT / "ledger.jsonl")


if __name__ == "__main__":
    main()
