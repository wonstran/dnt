import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import golden_cases as gc  # noqa: E402

pytestmark = pytest.mark.golden
if not (os.environ.get("DNT_REF_CLIP") and os.environ.get("DNT_GOLDEN_DIR")):
    pytest.skip("DNT_REF_CLIP and DNT_GOLDEN_DIR not set", allow_module_level=True)
CLIP, GOLD = Path(os.environ["DNT_REF_CLIP"]), Path(os.environ["DNT_GOLDEN_DIR"])
MANIFEST = json.loads((GOLD / "manifest.json").read_text())
DEFECTS_PATH = ROOT / "tests" / "golden" / "known_baseline_defects.yaml"
DEFECTS = {d["id"]: d for d in gc.load_defects(DEFECTS_PATH, ROOT)}


def test_environment_matches_recorded_environment():
    assert gc.check_env(ROOT / "tests" / "golden" / "reference-env.txt") == MANIFEST["env"]
    runtime = gc.runtime_info()
    for key in ("python", "torch", "torch_build"):
        assert runtime[key] == MANIFEST[key], (key, runtime[key], MANIFEST[key])
    assert gc.sha256(CLIP) == MANIFEST["clip"]["sha256"]
    gc.check_manifest_defects(MANIFEST, DEFECTS_PATH)  # the reviewed defects file is the one used (all fields)


def test_detection(tmp_path):
    gc.run_detect(CLIP, tmp_path / "d.txt")
    if MANIFEST["detect_deterministic"]:
        # a deterministic GPU result is only claimed on the recorded GPU and driver
        assert gc.runtime_info()["gpu"] == MANIFEST["gpu"], "run on the GPU/driver recorded in the manifest"
        assert (tmp_path / "d.txt").read_bytes() == (GOLD / "det_run1.txt").read_bytes()
        return
    new, old = (pd.read_csv(p, header=None) for p in (tmp_path / "d.txt", GOLD / "det_run1.txt"))
    assert len(new) == len(old)
    for col in (0, 1, 7):  # frame, res, class: exact (review item 4)
        assert new[col].tolist() == old[col].tolist(), f"column {col} differs"
    assert (new[[2, 3, 4, 5]] - old[[2, 3, 4, 5]]).abs().max().max() <= 1
    assert (new[6] - old[6]).abs().max() <= 0.01


@pytest.mark.parametrize(("case_id", "cls_name", "kwargs"), gc.CASES, ids=[c[0] for c in gc.CASES])
def test_tracks_byte_identical(case_id, cls_name, kwargs, tmp_path):
    out = tmp_path / f"track_{case_id}.txt"
    status = MANIFEST["cases"][case_id]
    if status["status"] == "known_defect":
        gc.check_manifest_defects(MANIFEST, DEFECTS_PATH)  # no field of any entry changed since generation
        entry = DEFECTS[status["defect_id"]]  # the exemption still exists and its evidence is unchanged
        assert entry["evidence_sha256"] == status["evidence_sha256"]
        assert gc.sha256(GOLD / f"traceback_{case_id}.txt") == status["traceback_sha256"]
        try:
            gc.run_track(cls_name, kwargs, GOLD / "dets.txt", CLIP, out)
        except Exception as exc:  # the reviewed upstream defect must reproduce identically
            assert gc.match_defect(entry, exc), f"{case_id}: failure differs from reviewed defect {entry['id']}: {exc!r}"
            print(f"EXEMPT {case_id}: reproduces reviewed baseline defect {status['defect_id']}")
            return
        print(f"EXEMPT {case_id}: reviewed baseline defect {status['defect_id']} no longer reproduces on 0.3.3")
        assert len(pd.read_csv(out, header=None).columns) == 10
        return
    gc.run_track(cls_name, kwargs, GOLD / "dets.txt", CLIP, out)
    assert out.read_bytes() == (GOLD / f"track_{case_id}.txt").read_bytes()


@pytest.mark.parametrize("case_id", gc.POST_CASES)
def test_post_processing_byte_identical(case_id, tmp_path):
    if MANIFEST["post"][case_id] != "ok":
        pytest.skip(f"{case_id}: {MANIFEST['post'][case_id]}")
    gc.run_post(GOLD / f"track_{case_id}.txt", tmp_path, case_id)
    for suffix in ("interp", "linked"):
        assert (tmp_path / f"{case_id}_{suffix}.txt").read_bytes() == (GOLD / f"{case_id}_{suffix}.txt").read_bytes()
