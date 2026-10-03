"""Tests for .env loading used by the refiner."""

import os

from dnt.refine.config import load_env_file


def test_load_env_file_sets_missing_only(tmp_path, monkeypatch):
    """Shell variables win; quotes, comments and ``export`` are handled."""
    f = tmp_path / ".env"
    f.write_text('# c\nexport A_KEY="abc"\nB_KEY=two\nC_KEY=file\n')
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "shell")
    assert load_env_file(f)
    assert os.environ["A_KEY"] == "abc"
    assert os.environ["B_KEY"] == "two"
    assert os.environ["C_KEY"] == "shell"
    monkeypatch.delenv("A_KEY")
    monkeypatch.delenv("B_KEY")


def test_load_env_file_missing(tmp_path):
    """A missing file is ignored."""
    assert not load_env_file(tmp_path / ".env")


def test_vlm_endpoints_select_and_validate():
    """``vlm.use`` picks one named endpoint; bad names and keys are rejected."""
    import pytest

    from dnt.refine.config import RefineConfig

    eps = {
        "qwen": {"backend": "openai_compat", "base_url": "http://g/v1", "model": "q"},
        "claude": {"backend": "anthropic"},
    }
    cfg = RefineConfig.from_dict({"vlm": {"use": "qwen", "endpoints": eps, "votes": 3}})
    cfg.validate()
    got = cfg.vlm.resolve()
    assert (got.base_url, got.model, got.votes) == ("http://g/v1", "q", 3)
    assert RefineConfig.from_dict(cfg.to_dict()).vlm.use == "qwen"
    for bad in ({"use": "nope", "endpoints": eps}, {"endpoints": {"a": {"foo": 1}}}):
        with pytest.raises(ValueError):
            RefineConfig.from_dict({"vlm": bad}).validate()
