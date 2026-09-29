"""Write characterization baselines for the dnt.refine move.

Run ONCE, before moving any code (plan Task 1), against the pre-move dnt.track.post_process:

    .venv/bin/python tests/refine/make_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _fixtures import DATA, load_raw, random_tracks  # noqa: E402

from dnt.track.post_process import interpolate_tracks_rts, link_tracklets  # noqa: E402


def main() -> None:
    """Write raw inputs and the current interpolate/link outputs for three seeds."""
    DATA.mkdir(exist_ok=True)
    for seed in (0, 1, 2):
        random_tracks(seed=seed).to_csv(DATA / f"raw_{seed}.csv", index=False, header=False)
        raw = load_raw(seed)
        interpolate_tracks_rts(
            raw.copy(), output_file=str(DATA / f"interp_{seed}.csv"), verbose=False
        )
        interpolate_tracks_rts(
            raw.copy(),
            output_file=str(DATA / f"interp_smooth_{seed}.csv"),
            smooth_existing=True,
            verbose=False,
        )
        link_tracklets(
            raw.copy(), output_file=str(DATA / f"link_{seed}.csv"), max_gap=20, verbose=False
        )


if __name__ == "__main__":
    main()
