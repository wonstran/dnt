import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(ROOT)], check=True)
    return next(out.glob("dnt-0.3.3-*.whl"))


BUNDLED = ("dnt/track/reid_weights/osnet_x1_0_msmt17.pt", "dnt/detect/signal/weights/ped_signal.pt")


def test_wheel_contains_class_names(wheel):
    names = zipfile.ZipFile(wheel).namelist()
    for stem in ("coco", "openimages", "voc"):
        assert f"dnt/shared/data/{stem}.names" in names


def test_wheel_bundles_referenced_weights_only(wheel):
    names = zipfile.ZipFile(wheel).namelist()
    for path in BUNDLED:
        assert path in names, path
    assert "dnt/detect/signal/weights/wb_ped_signal.pt" not in names  # unreferenced
    assert not [n for n in names if "Zone.Identifier" in n]


def _python311():
    if os.environ.get("DNT_SMOKE_PYTHON"):
        return os.environ["DNT_SMOKE_PYTHON"]
    if sys.version_info[:2] == (3, 11):
        return sys.executable
    return shutil.which("python3.11")


def test_smoke_pipeline_in_clean_311_venv(wheel, tmp_path):
    py = _python311()
    if not py:
        pytest.skip("no Python 3.11 interpreter found; set DNT_SMOKE_PYTHON")
    venv = tmp_path / "venv"
    subprocess.run([py, "-m", "venv", str(venv)], check=True)
    vpy = venv / "bin" / "python"
    subprocess.run([str(vpy), "-m", "pip", "install", "--quiet", str(wheel),
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    work = tmp_path / "work"
    work.mkdir()
    for name in ("synthetic.py", "smoke_pipeline.py"):
        shutil.copy(ROOT / "tests" / "support" / name, work / name)
    shas = [hashlib.sha256((ROOT / "src" / p).read_bytes()).hexdigest() for p in BUNDLED]
    result = subprocess.run([str(vpy), "smoke_pipeline.py", str(work), *shas], cwd=work, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE OK" in result.stdout
