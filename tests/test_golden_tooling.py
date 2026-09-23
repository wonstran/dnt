import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bench  # noqa: E402
import golden_cases as gc  # noqa: E402


# ---- defect file validation (review item 1, 5) ------------------------------------
def _entry(**over):
    base = {"id": "D1", "tracker": "bytetrack", "exception_type": "RuntimeError",
            "message_regex": r"boom: .*", "raised_at": "test_golden_tooling.py:build",
            "root_cause": "0.3.2.4 passes X", "reviewed_by": "maintainer", "reviewed_on": "2026-09-24",
            "evidence": "tests/golden/defects/D1.txt"}
    return {**base, **over}


def _write(tmp_path, entries, evidence=True):
    (tmp_path / "tests" / "golden" / "defects").mkdir(parents=True)
    (tmp_path / "README.md").write_text("not a traceback")  # exists, but outside the evidence directory
    if evidence:
        (tmp_path / "tests/golden/defects/D1.txt").write_text("Traceback ...")
    path = tmp_path / "defects.yaml"
    path.write_text(yaml.safe_dump({"defects": entries}))
    return path


def test_repository_defects_file_is_valid():
    gc.load_defects(ROOT / "tests" / "golden" / "known_baseline_defects.yaml", ROOT)


def test_valid_entry_gets_evidence_hash(tmp_path):
    [entry] = gc.load_defects(_write(tmp_path, [_entry()]), tmp_path)
    assert entry["evidence_sha256"] == gc.sha256(tmp_path / "tests/golden/defects/D1.txt")


@pytest.mark.parametrize(("entries", "evidence", "match"), [
    ([{k: v for k, v in _entry().items() if k != "root_cause"}], True, "root_cause"),
    ([_entry()], False, "evidence"),
    ([_entry(), _entry(tracker="ocsort")], True, "duplicate id"),
    ([_entry(tracker="sort")], True, "unknown case"),
    ([_entry(reviewed_on="yesterday")], True, "reviewed_on"),
    ([_entry(evidence="README.md")], True, "inside tests/golden/defects"),
    ([_entry(evidence="tests/golden/defects/../../../README.md")], True, "inside tests/golden/defects"),
    ([_entry(evidence="/etc/hostname")], True, "inside tests/golden/defects"),
])
def test_invalid_defect_entries_rejected(tmp_path, entries, evidence, match):
    with pytest.raises(ValueError, match=match):
        gc.load_defects(_write(tmp_path, entries, evidence), tmp_path)


def test_symlink_escaping_evidence_dir_rejected(tmp_path):
    path = _write(tmp_path, [_entry(evidence="tests/golden/defects/link.txt")])
    (tmp_path / "tests/golden/defects/link.txt").symlink_to(tmp_path / "README.md")
    with pytest.raises(ValueError, match="inside tests/golden/defects"):
        gc.load_defects(path, tmp_path)


def test_manifest_detects_metadata_only_change(tmp_path):
    path = _write(tmp_path, [_entry()])
    manifest = {"defects_file_sha256": gc.sha256(path)}
    gc.check_manifest_defects(manifest, path)  # unchanged: passes
    data = yaml.safe_load(path.read_text())
    data["defects"][0]["reviewed_by"] = "someone else"  # evidence file untouched
    path.write_text(yaml.safe_dump(data))
    gc.load_defects(path, tmp_path)  # still a valid file ...
    with pytest.raises(ValueError, match="changed since the golden manifest"):
        gc.check_manifest_defects(manifest, path)  # ... but no longer the one the goldens were made with


# ---- defect matching ----------------------------------------------------------------
def build():
    raise RuntimeError("boom: x.pt")


def _exc():
    try:
        build()
    except RuntimeError as exc:
        return exc


def test_defect_match_requires_type_full_message_and_site():
    exc = _exc()
    assert gc.match_defect(_entry(), exc)
    assert not gc.match_defect(_entry(exception_type="ValueError"), exc)
    assert not gc.match_defect(_entry(message_regex="boom"), exc)
    assert not gc.match_defect(_entry(raised_at="other.py:build"), exc)


# ---- exemption paths (review item 1) --------------------------------------------
CASES = [("bytetrack", "ByteTrackConfig", {}), ("botsort", "BoTSORTConfig", {}), ("ocsort", "OCSORTConfig", {})]


def _track_fn(failing):
    def track_fn(cls_name, kwargs, out_file):
        if cls_name in failing:
            build()
        Path(out_file).write_text("0,1,1,1,1,1,0.9,2,-1,-1\n")
    return track_fn


def test_exempted_case_recorded_with_evidence(tmp_path):
    entry = {**_entry(), "evidence_sha256": "e" * 64}
    statuses = gc.run_track_cases(CASES, [entry], _track_fn({"ByteTrackConfig"}), tmp_path)
    assert statuses["bytetrack"]["status"] == "known_defect"
    assert statuses["bytetrack"]["defect_id"] == "D1"
    assert statuses["bytetrack"]["evidence_sha256"] == "e" * 64
    assert statuses["bytetrack"]["traceback_sha256"] == gc.sha256(tmp_path / "traceback_bytetrack.txt")
    assert not (tmp_path / "track_bytetrack.txt").exists()
    assert statuses["botsort"]["status"] == statuses["ocsort"]["status"] == "ok"


def test_unmatched_error_aborts_and_saves_traceback(tmp_path):
    with pytest.raises(SystemExit, match="unexpected baseline error in botsort"):
        gc.run_track_cases(CASES, [], _track_fn({"BoTSORTConfig"}), tmp_path)
    assert "boom" in (tmp_path / "unexpected_botsort.txt").read_text()


def test_stale_entry_aborts(tmp_path):
    entry = {**_entry(tracker="ocsort"), "evidence_sha256": "e" * 64}
    with pytest.raises(SystemExit, match="stale"):
        gc.run_track_cases(CASES, [entry], _track_fn(set()), tmp_path)


def test_post_skipped_for_exempted_post_case(tmp_path):
    calls = []
    statuses = {"bytetrack": {"status": "known_defect"}, "botsort": {"status": "ok"}}
    post = gc.run_post_cases(statuses, lambda tf, od, cid: calls.append(cid), tmp_path)
    assert calls == ["botsort"]
    assert post == {"bytetrack": "skipped: known_defect", "botsort": "ok"}


# ---- environment (review item 2) -----------------------------------------------------
def test_check_env_reports_mismatch(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("# torch build: cpu\npytest==0.0.1 \\\n    --hash=sha256:abc\n")
    with pytest.raises(SystemExit, match="pytest"):
        gc.check_env(lock)


def test_check_env_distinguishes_local_builds(tmp_path, monkeypatch):
    lock = tmp_path / "lock.txt"
    lock.write_text("torch==2.10.0\n")
    monkeypatch.setattr(gc.importlib.metadata, "version", lambda name: "2.10.0+cpu")
    with pytest.raises(SystemExit, match="torch"):
        gc.check_env(lock)


def test_runtime_info_keys():
    assert set(gc.runtime_info()) == {"python", "torch", "torch_build", "gpu"}


def test_cases_cover_all_trackers_and_passthrough():
    assert [c[0] for c in gc.CASES] == ["bytetrack", "botsort", "ocsort", "deepocsort", "strongsort",
                                        "hybridsort", "boosttrack", "sfsort", "bytetrack_evolve",
                                        "bytetrack_to_ocsort"]


# ---- benchmark stage planning (review item 6) ---------------------------------------
def test_bench_full_pipeline_ok():
    bench.plan_stages(["detect", "track", "post"], None, None)


@pytest.mark.parametrize(("stages", "dets", "tracks", "match"), [
    (["track"], None, None, "--dets"),
    (["post"], None, None, "--tracks"),
    (["detect", "post"], None, None, "--tracks"),
    (["detect", "fly"], None, None, "unknown stage"),
])
def test_bench_rejects_unsatisfiable_stages(stages, dets, tracks, match):
    with pytest.raises(ValueError, match=match):
        bench.plan_stages(stages, dets, tracks)


def test_bench_isolated_stages_with_inputs(tmp_path):
    bench.plan_stages(["track"], tmp_path / "d.txt", None)
    bench.plan_stages(["post"], None, tmp_path / "t.txt")
