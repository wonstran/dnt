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
    def __init__(self, seed, batches):
        super().__init__()
        torch.manual_seed(seed)
        self._batches = batches
        self.config = types.SimpleNamespace(hidden_size=8)
        self.proj = torch.nn.Linear(3, 8, bias=False)

    def forward(self, pixel_values):
        self._batches.append(int(pixel_values.shape[0]))
        feats = self.proj(pixel_values.mean(dim=(2, 3)))
        tokens = -feats[:, None, :].repeat(1, 5, 1)  # the patch tokens differ from ...
        tokens[:, 0] = feats  # ... the CLS token at position 0
        return types.SimpleNamespace(last_hidden_state=tokens)


def install_fake_transformers(monkeypatch):
    """Install a fake ``transformers``; set ``seen["seed"]`` to change what the model name loads.

    ``seen["batches"]`` collects the batch size of every forward pass; ``seen["half"]`` makes
    the loaded model half precision.
    """
    seen = {"seed": 0, "loads": 0, "batches": [], "half": False}

    class AutoModel:
        @staticmethod
        def from_pretrained(source):
            seen["source"] = source
            seen["loads"] += 1
            model = _FakeDino(seen["seed"], seen["batches"])
            return model.half() if seen["half"] else model  # a half-precision checkpoint

    mod = _module("transformers")
    mod.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return seen


def install_fake_torchreid(monkeypatch, layout="torchreid.utils"):
    """Install a fake ``torchreid`` whose extractor embeds a crop by its mean color.

    Like the real ``FeatureExtractor`` (which converts an ndarray with ``T.ToPILImage()``) it
    treats ndarray inputs as RGB, and like it, it prints a weights message to stdout when
    built. ``seen["batches"]`` collects the number of images of every call.

    ``layout`` is the module that provides ``FeatureExtractor``: ``"torchreid.utils"``
    (KaiyangZhou's deep-person-reid), ``"torchreid.reid.utils"`` (the PyPI ``torchreid``
    package), or ``None`` (a ``torchreid`` without it).
    """
    seen = {"n": 0, "loads": 0, "batches": []}

    class FeatureExtractor:
        def __init__(self, model_name, model_path, device, verbose=True):
            seen.update(model_name=model_name, model_path=model_path, device=device)
            seen["loads"] += 1
            print(f"Successfully loaded pretrained weights from {model_path}")

        def __call__(self, images):
            seen["n"] += len(images)
            seen["batches"].append(len(images))
            return torch.tensor(
                [[float(im[..., c].mean()) + 1.0 for c in range(3)] + [1.0] for im in images]
            )

    top = _module("torchreid")
    monkeypatch.setitem(sys.modules, "torchreid", top)
    if layout is not None:
        utils = _module(layout)
        utils.FeatureExtractor = FeatureExtractor
        monkeypatch.setitem(sys.modules, layout, utils)
        if layout == "torchreid.utils":
            top.utils = utils
        else:
            parent = _module("torchreid.reid")
            parent.utils = utils
            top.reid = parent
            monkeypatch.setitem(sys.modules, "torchreid.reid", parent)
    return seen
