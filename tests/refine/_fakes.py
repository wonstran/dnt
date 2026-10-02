"""Fake ``transformers`` and ``torchreid`` modules for the encoder tests."""

from __future__ import annotations

import importlib.machinery
import sys
import types

import torch


def _module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec works on it
    return mod


class _FakeDino(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.config = types.SimpleNamespace(hidden_size=8)
        self.proj = torch.nn.Linear(3, 8, bias=False)

    def forward(self, pixel_values):
        feats = self.proj(pixel_values.mean(dim=(2, 3)))
        tokens = feats[:, None, :].repeat(1, 5, 1)
        return types.SimpleNamespace(last_hidden_state=tokens)


def install_fake_transformers(monkeypatch):
    """Install a fake ``transformers``; set ``seen["seed"]`` to change what the model name loads."""
    seen = {"seed": 0, "loads": 0}

    class AutoModel:
        @staticmethod
        def from_pretrained(source):
            seen["source"] = source
            seen["loads"] += 1
            return _FakeDino(seen["seed"])

    mod = _module("transformers")
    mod.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return seen


def install_fake_torchreid(monkeypatch):
    """Install a fake ``torchreid`` whose extractor embeds a crop by its mean color."""
    seen = {"n": 0, "loads": 0}

    class FeatureExtractor:
        def __init__(self, model_name, model_path, device, verbose=True):
            seen.update(model_name=model_name, model_path=model_path, device=device)
            seen["loads"] += 1

        def __call__(self, images):
            seen["n"] += len(images)
            return torch.tensor(
                [[float(im[..., c].mean()) + 1.0 for c in range(3)] + [1.0] for im in images]
            )

    utils = _module("torchreid.utils")
    utils.FeatureExtractor = FeatureExtractor
    top = _module("torchreid")
    top.utils = utils
    monkeypatch.setitem(sys.modules, "torchreid", top)
    monkeypatch.setitem(sys.modules, "torchreid.utils", utils)
    return seen
