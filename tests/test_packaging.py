import importlib.metadata
import tomllib
from pathlib import Path

import dnt

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_version_consistent():
    assert dnt.__version__ == PROJECT["version"] == "0.3.4"
    assert importlib.metadata.version("dnt") == dnt.__version__


def test_metadata_constraints():
    assert PROJECT["requires-python"] == ">=3.11"
    deps = PROJECT["dependencies"]
    assert "boxmot==16.0.11" in deps
    assert "ultralytics>=8.4.14,<9" in deps
    assert not any(d.startswith(("opencv-contrib-python", "faiss", "torchaudio", "thop")) for d in deps)
    assert PROJECT["license"] == "MIT"
