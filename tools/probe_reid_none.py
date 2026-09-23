"""Run ByteTrackConfig(extra_kwargs={'tracker_type': T}) on 0.3.2.4 for each ReID target T.

    ../dnt-baseline-venv/bin/python tools/probe_reid_none.py tests/data
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "support"))
from synthetic import StubDetector, make_synthetic_video  # noqa: E402

import dnt  # noqa: E402
from dnt.track import ByteTrackConfig, Tracker  # noqa: E402

assert dnt.__version__ == "0.3.2.4"
out = Path(sys.argv[1])
work = out / "tracker_type_reid_none"
work.mkdir(parents=True, exist_ok=True)
video = work / "scene.mp4"
dets = work / "dets.txt"
StubDetector(make_synthetic_video(video)).detect(video, iou_file=dets)
outcomes = {}
for target in ("botsort", "strongsort", "deepocsort", "hybridsort", "boosttrack"):
    try:
        cfg = ByteTrackConfig(extra_kwargs={"tracker_type": target})
        Tracker(config=cfg, device="cpu").track(str(dets), str(work / f"{target}.txt"), str(video))
        outcomes[target] = "ok"
    except Exception:
        outcomes[target] = "error"
        (work / f"{target}_traceback.txt").write_text(traceback.format_exc())
video.unlink()
dets.unlink()
for p in work.glob("*.txt"):
    if not p.name.endswith("_traceback.txt"):
        p.unlink()
(out / "tracker_type_reid_none.json").write_text(json.dumps(outcomes, indent=2, sort_keys=True))
print(outcomes)
