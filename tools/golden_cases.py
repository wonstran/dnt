"""Golden cases shared by tools/make_golden.py (0.3.2.4) and tests/golden/test_golden.py (0.3.3). Spec §5."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import platform
import re
import subprocess
import traceback
import warnings
from pathlib import Path

import pandas as pd
import yaml

TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort", "strongsort", "hybridsort", "boosttrack", "sfsort"]
CONFIG_NAMES = {"bytetrack": "ByteTrackConfig", "botsort": "BoTSORTConfig", "ocsort": "OCSORTConfig",
                "deepocsort": "DeepOCSORTConfig", "strongsort": "StrongSORTConfig",
                "hybridsort": "HybridSORTConfig", "boosttrack": "BoostTrackConfig", "sfsort": "SFSORTConfig"}
CASES: list[tuple[str, str, dict]] = [(t, CONFIG_NAMES[t], {}) for t in TRACKERS] + [
    ("bytetrack_evolve", "ByteTrackConfig", {"extra_kwargs": {"evolve_param_dict": {"track_thresh": 0.7}}}),
    ("bytetrack_to_ocsort", "ByteTrackConfig", {"extra_kwargs": {"tracker_type": "ocsort"}}),
]
POST_CASES = ("bytetrack", "botsort")
REQUIRED_DEFECT_KEYS = ("id", "tracker", "exception_type", "message_regex", "raised_at",
                        "root_cause", "reviewed_by", "reviewed_on", "evidence")
DEFECTS_DIR = Path("tests") / "golden" / "defects"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_env(lock: Path) -> dict[str, str]:
    """Compare installed versions with the lock's `name==version` pins, including local suffixes (+cpu)."""
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s\;]+)", Path(lock).read_text(), flags=re.M))
    installed, bad = {}, []
    for name, want in pins.items():
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have = "missing"
        installed[name] = have
        if have != want:
            bad.append(f"{name} (lock {want}, installed {have})")
    if bad:
        raise SystemExit("environment differs from the reference lock: " + ", ".join(bad))
    return installed


def runtime_info() -> dict[str, str]:
    """Python, full torch version, torch build and GPU name/driver ("none" without CUDA)."""
    import torch

    gpu = "none"
    if torch.cuda.is_available():
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torch_build": torch.version.cuda or "cpu", "gpu": gpu}


def load_defects(path: Path, root: Path) -> list[dict]:
    """Load and validate reviewed baseline defects; add each entry's evidence_sha256 (spec §5.3)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    entries = data.get("defects")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a mapping with a 'defects' list")
    case_ids = {c[0] for c in CASES}
    seen_ids, seen_cases, out = set(), set(), []
    for entry in entries:
        missing = [k for k in REQUIRED_DEFECT_KEYS if not entry.get(k)]
        if missing:
            raise ValueError(f"{path}: defect {entry.get('id')!r} is missing {missing}")
        if entry["id"] in seen_ids:
            raise ValueError(f"{path}: duplicate id {entry['id']!r}")
        if entry["tracker"] not in case_ids:
            raise ValueError(f"{path}: defect {entry['id']!r} names unknown case {entry['tracker']!r}")
        if entry["tracker"] in seen_cases:
            raise ValueError(f"{path}: more than one defect for case {entry['tracker']!r}")
        try:
            dt.date.fromisoformat(str(entry["reviewed_on"]))
        except ValueError as exc:
            raise ValueError(f"{path}: defect {entry['id']!r} reviewed_on must be YYYY-MM-DD") from exc
        raw = Path(str(entry["evidence"]))
        allowed = (Path(root) / DEFECTS_DIR).resolve()
        evidence = (Path(root) / raw).resolve()  # resolves '..' and symlinks
        if raw.is_absolute() or not evidence.is_relative_to(allowed):
            raise ValueError(f"{path}: defect {entry['id']!r} evidence must be a repo-relative path inside "
                             f"{DEFECTS_DIR}/, got {entry['evidence']!r}")
        if not evidence.is_file():
            raise ValueError(f"{path}: defect {entry['id']!r} evidence file not found: {evidence}")
        seen_ids.add(entry["id"])
        seen_cases.add(entry["tracker"])
        out.append({**entry, "evidence_sha256": sha256(evidence)})
    return out


def check_manifest_defects(manifest: dict, defects_path: Path) -> None:
    """Raise ValueError if the defects file changed since the manifest was generated (any field)."""
    current = sha256(defects_path)
    if current != manifest["defects_file_sha256"]:
        raise ValueError(f"{defects_path} changed since the golden manifest was generated "
                         f"(recorded {manifest['defects_file_sha256'][:12]}, now {current[:12]}); regenerate the golden "
                         "outputs with tools/make_golden.py after the change is reviewed")


def match_defect(entry: dict, exc: BaseException) -> bool:
    """True only if type, full message and innermost raise site all match a reviewed defect entry."""
    frames = traceback.extract_tb(exc.__traceback__)
    site = f"{Path(frames[-1].filename).name}:{frames[-1].name}" if frames else ""
    return (type(exc).__name__ == entry["exception_type"]
            and re.fullmatch(entry["message_regex"], str(exc)) is not None
            and site == entry["raised_at"])


def run_track_cases(cases, defects: list[dict], track_fn, out: Path) -> dict[str, dict]:
    """Run every case; only a matching reviewed defect may exempt one. Fails closed otherwise."""
    by_case = {d["tracker"]: d for d in defects}
    statuses: dict[str, dict] = {}
    for case_id, cls_name, kwargs in cases:
        entry = by_case.get(case_id)
        target = out / f"track_{case_id}.txt"
        try:
            track_fn(cls_name, kwargs, target)
        except Exception as exc:
            text = "".join(traceback.format_exception(exc))
            if entry is None or not match_defect(entry, exc):
                (out / f"unexpected_{case_id}.txt").write_text(text)
                raise SystemExit(f"unexpected baseline error in {case_id}: {exc!r} "
                                 f"(traceback saved to unexpected_{case_id}.txt)") from exc
            (out / f"traceback_{case_id}.txt").write_text(text)
            target.unlink(missing_ok=True)
            statuses[case_id] = {"status": "known_defect", "defect_id": entry["id"],
                                 "traceback_sha256": sha256(out / f"traceback_{case_id}.txt"),
                                 "evidence_sha256": entry["evidence_sha256"]}
            continue
        if entry is not None:
            raise SystemExit(f"stale known_baseline_defects entry {entry['id']}: {case_id} now succeeds on 0.3.2.4")
        statuses[case_id] = {"status": "ok", "sha256": sha256(target)}
    return statuses


def run_post_cases(statuses: dict[str, dict], post_fn, out: Path) -> dict[str, str]:
    """Post-process only cases whose baseline tracks exist (status ok)."""
    result = {}
    for case_id in POST_CASES:
        status = statuses[case_id]["status"]
        if status != "ok":
            result[case_id] = f"skipped: {status}"
            continue
        post_fn(out / f"track_{case_id}.txt", out, case_id)
        result[case_id] = "ok"
    return result


def run_detect(clip: Path, out: Path) -> None:
    from dnt.detect import Detector

    Detector(device="auto", half=False).detect(str(clip), iou_file=str(out), verbose=False)


def run_track(cls_name: str, kwargs: dict, dets: Path, video: Path, out: Path) -> None:
    from dnt.track import tracker as tr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = getattr(tr, cls_name)(**kwargs)
        tr.Tracker(config=cfg, device="cpu", half=False).track(str(dets), str(out), str(video))


def run_post(track_file: Path, out_dir: Path, case_id: str) -> None:
    from dnt.track.post_process import interpolate_tracks_rts, link_tracklets

    tracks = pd.read_csv(track_file, header=None)  # DataFrame path: 0.3.2.4's file path has bug B3
    interp = interpolate_tracks_rts(tracks=tracks, output_file=str(out_dir / f"{case_id}_interp.txt"), verbose=False)
    link_tracklets(tracks=interp, output_file=str(out_dir / f"{case_id}_linked.txt"), verbose=False)
