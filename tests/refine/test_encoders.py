import importlib.machinery
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from dnt.refine.config import EncoderConfig
from dnt.refine.encoders import (
    check_encoder_dependencies,
    default_reid_weights,
    make_encoder,
    parameters_digest,
    weights_digest,
    weights_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def _present(monkeypatch, name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    monkeypatch.setitem(sys.modules, name, mod)


def test_missing_dino_package_names_the_extra_and_the_alternative(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-dino\]") as err:
        check_encoder_dependencies(EncoderConfig(kind="dino"))
    assert "encoder.kind: none" in str(err.value)


def test_missing_reid_package_names_its_own_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchreid", None)
    with pytest.raises(ImportError, match=r"dnt\[refine-reid\]"):
        check_encoder_dependencies(EncoderConfig(kind="reid"))


def test_kind_none_needs_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torchreid", None)
    check_encoder_dependencies(EncoderConfig(kind="none"))


def test_an_installed_package_passes(monkeypatch):
    _present(monkeypatch, "transformers")
    check_encoder_dependencies(EncoderConfig(kind="dino"))


def test_a_module_without_a_spec_still_counts_as_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    check_encoder_dependencies(EncoderConfig(kind="dino"))  # find_spec would raise ValueError


def test_parameters_digest_follows_the_weights_not_the_name():
    import torch

    def model(seed):
        torch.manual_seed(seed)
        return torch.nn.Linear(3, 4)

    assert parameters_digest(model(0)) == parameters_digest(model(0))
    assert len(parameters_digest(model(0))) == 64
    assert parameters_digest(model(0)) != parameters_digest(model(1))


def test_weights_identity_is_the_digest_of_the_local_file_or_none(tmp_path):
    w = tmp_path / "w.pt"
    w.write_bytes(b"one")
    assert weights_identity(EncoderConfig(kind="reid", weights=str(w))) == weights_digest(w)
    w.write_bytes(b"two")  # replaced in place
    assert weights_identity(EncoderConfig(kind="reid", weights=str(w))) == weights_digest(w)
    assert weights_identity(EncoderConfig(kind="reid")) == weights_digest(default_reid_weights())
    assert weights_identity(EncoderConfig(kind="reid"), "vehicle") is None
    assert weights_identity(EncoderConfig(kind="dino")) is None
    assert weights_identity(EncoderConfig(kind="none")) is None
    with pytest.raises(ValueError, match="not found"):
        weights_identity(EncoderConfig(kind="dino", weights=str(tmp_path / "gone")))


def test_make_encoder_rejects_none_and_vehicle_reid_without_weights():
    with pytest.raises(ValueError, match="kind='none'"):
        make_encoder(EncoderConfig(kind="none"))
    with pytest.raises(ValueError, match="vehicle"):
        make_encoder(EncoderConfig(kind="reid"), "vehicle")


def test_the_shipped_osnet_weights_are_found():
    assert default_reid_weights().is_file()


def test_weights_digest_hashes_files_and_directories(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    d = tmp_path / "model"
    d.mkdir()
    (d / "a.bin").write_bytes(b"1")
    (d / "b.bin").write_bytes(b"2")
    assert weights_digest(f) == weights_digest(f) and len(weights_digest(f)) == 64
    first = weights_digest(d)
    assert first == weights_digest(d)
    (d / "b.bin").write_bytes(b"3")
    assert weights_digest(d) != first
    with pytest.raises(ValueError, match="not found"):
        weights_digest(tmp_path / "missing")


def test_extras_are_declared_and_not_required():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = meta["optional-dependencies"]
    assert extras["refine-dino"] == ["transformers>=4.40"]
    assert extras["refine-reid"] == ["torchreid", "tensorboard"]
    assert extras["refine-vlm"] == ["openai>=1.40", "anthropic>=0.40"]
    assert set(extras["refine"]) == {
        "transformers>=4.40",
        "torchreid",
        "tensorboard",
        "openai>=1.40",
        "anthropic>=0.40",
    }
    required = " ".join(meta["dependencies"])
    names = ("transformers", "torchreid", "tensorboard", "openai", "anthropic")
    assert not any(n in required for n in names)


def test_importing_the_package_loads_no_encoder_library():
    code = (
        "import sys, dnt.refine, dnt.refine.encoders; "
        "bad = [m for m in ('transformers', 'torchreid') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
