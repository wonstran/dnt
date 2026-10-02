import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import sys
    for name in ("transformers", "torchreid", "openai", "anthropic"):
        sys.modules[name] = None  # a minimal install: none of the optional packages
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    from dnt.refine import RefineConfig, TrackRefiner

    out = Path(sys.argv[1])
    cfg = RefineConfig.defaults()  # encoder.kind is "dino", but the package is missing
    rows = [[f, 1, 100.0 + 2 * f, 100.0, 30.0, 60.0, 0.9, 0, -1, -1] for f in range(60)]
    src = out / "t.txt"
    pd.DataFrame(rows).to_csv(src, index=False, header=False)
    vid = out / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(60):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()

    refiner = TrackRefiner(cfg)
    refiner.refine(src, out / "a.txt", fps=10, verbose=False)  # no video: no package needed
    assert (out / "a.txt").is_file()
    try:
        refiner.refine(src, out / "b.txt", video_file=vid, verbose=False)
    except ImportError as err:
        assert "refine-dino" in str(err), err
    else:
        raise SystemExit("expected an ImportError for a video with encoder.kind dino")
    assert not (out / "b.txt").exists() and not (out / "b.ledger.jsonl").exists()
    cfg.encoder.kind = "none"
    refiner.refine(src, out / "c.txt", video_file=vid, verbose=False)
    assert (out / "c.txt").is_file() and not (out / "c.features.npz").exists()
    print("OK")
    """
)


def test_a_minimal_install_runs_without_an_encoder_and_fails_early_with_one_requested(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout
