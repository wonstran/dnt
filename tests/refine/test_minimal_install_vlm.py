import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import logging, sys
    for name in ("transformers", "torchreid", "openai", "anthropic"):
        sys.modules[name] = None  # a minimal install
    from pathlib import Path

    from dnt.refine import RefineConfig, TrackRefiner

    # tests/ has no __init__.py and the same package name pytest uses here is `refine`; importing
    # `tests.refine` instead fails when another distribution installs a top-level `tests`
    sys.path.insert(0, sys.argv[2])
    from refine._video import takeover_scene

    out = Path(sys.argv[1])
    src, vid = takeover_scene(out)
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"
    cfg.link.enabled = False
    cfg.vlm.backend, cfg.vlm.model = "openai_compat", "m"
    refiner = TrackRefiner(cfg)
    try:
        refiner.refine(src, out / "a.txt", video_file=vid, verbose=False)
    except ImportError as err:
        assert "refine-vlm" in str(err), err
    else:
        raise SystemExit("expected an ImportError for a video with a VLM backend")
    assert not (out / "a.txt").exists() and not (out / "a.ledger.jsonl").exists()
    refiner.refine(src, out / "b.txt", fps=10, verbose=False)  # no video: the backend is ignored
    assert (out / "b.txt").is_file() and refiner.last_result.summary["vlm"]["calls"] == 0
    cfg.vlm.backend, cfg.vlm.model = "none", None
    refiner.refine(src, out / "c.txt", video_file=vid, verbose=False)
    assert (out / "c.txt").is_file()
    print("OK")
    """
)


def test_a_minimal_install_runs_without_a_vlm_and_fails_early_with_one_requested(tmp_path):
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), tests_dir], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout
