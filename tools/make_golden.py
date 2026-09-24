"""Generate golden outputs with dnt 0.3.2.4 in the reference environment. Fails closed (spec §5.3).

    DNT_REF_CLIP=... DNT_GOLDEN_DIR=... ../dnt-baseline-venv/bin/python tools/make_golden.py
"""

import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_cases as gc

ROOT = Path(__file__).resolve().parents[1]
DEFECTS = ROOT / "tests" / "golden" / "known_baseline_defects.yaml"


def main() -> int:
    import dnt

    if dnt.__version__ != "0.3.2.4":
        sys.exit(f"make_golden.py must run on dnt 0.3.2.4, found {dnt.__version__}")
    clip = Path(os.environ["DNT_REF_CLIP"])
    out = Path(os.environ["DNT_GOLDEN_DIR"])

    # ---- pre-flight: nothing is written if any check fails --------------------------
    env = gc.check_env(ROOT / "tests" / "golden" / "reference-env.txt")
    runtime = gc.runtime_info()
    defects = gc.load_defects(DEFECTS, ROOT)
    cap = cv2.VideoCapture(str(clip))
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if frames <= 0 or fps <= 0:
        sys.exit(f"reference clip unreadable: frames={frames} fps={fps}")
    weights = Path(dnt.__file__).parent / "track" / "reid_weights" / "osnet_x1_0_msmt17.pt"
    expected = (ROOT / "tests" / "golden" / "weights.sha256").read_text().split()[0]
    if not weights.exists() or gc.sha256(weights) != expected:
        sys.exit(f"ReID weights missing or wrong hash: {weights}")
    out.mkdir(parents=True, exist_ok=True)
    (out / ".write_test").write_text("ok")
    (out / ".write_test").unlink()

    # ---- detection, twice ----------------------------------------------------------
    gc.run_detect(clip, out / "det_run1.txt")
    gc.run_detect(clip, out / "det_run2.txt")
    if os.path.getsize(out / "det_run1.txt") == 0:
        sys.exit("baseline detection produced no rows")
    deterministic = gc.sha256(out / "det_run1.txt") == gc.sha256(out / "det_run2.txt")
    (out / "dets.txt").write_bytes((out / "det_run1.txt").read_bytes())

    # ---- tracking (fails closed) and post-processing (successful cases only) -------
    statuses = gc.run_track_cases(
        gc.CASES, defects,
        lambda cls_name, kwargs, target: gc.run_track(cls_name, kwargs, out / "dets.txt", clip, target), out)
    post = gc.run_post_cases(statuses, gc.run_post, out)

    manifest = {
        "dnt": dnt.__version__, **runtime, "env": env,
        "clip": {"path": str(clip), "sha256": gc.sha256(clip), "frames": frames, "fps": fps},
        "detect_deterministic": deterministic, "cases": statuses, "post": post,
        "defects_file_sha256": gc.sha256(DEFECTS),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"cases": statuses, "post": post}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
