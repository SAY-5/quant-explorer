"""Unit tests for the lower-level helpers in ``train.py`` and ``data.py``.

These avoid CIFAR-10 download by exercising only the pure-tensor code paths.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quant_explorer.data import _eval_transform, _train_transform
from quant_explorer.train import _evaluate, _set_seed


def test_set_seed_makes_torch_deterministic() -> None:
    _set_seed(1234)
    a = torch.randn(8)
    _set_seed(1234)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_eval_transform_shape_and_normalisation() -> None:
    """The eval transform should ToTensor + Normalize a 32x32 RGB PIL image."""
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(255, 128, 0))
    out = _eval_transform()(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    # After Normalize the values should not all be in [0, 1] anymore.
    assert (out < 0).any() or (out > 1).any()


def test_train_transform_shape() -> None:
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(0, 64, 200))
    out = _train_transform()(img)
    assert out.shape == (3, 32, 32)


class _IdentityClassifier(nn.Module):
    """Maps each integer input row-id to a one-hot logit row."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x.long().view(-1)]


def test_evaluate_returns_top1_fraction() -> None:
    n_classes = 4
    n = 6
    # Logits row i picks class i % n_classes (so 4 of 6 are "correct" if
    # labels match input index modulo n_classes).
    logits = torch.zeros(n, n_classes)
    for i in range(n):
        logits[i, i % n_classes] = 5.0
    inputs = torch.arange(n, dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    model = _IdentityClassifier(logits)
    ds = TensorDataset(inputs, labels)
    loader = DataLoader(ds, batch_size=2)
    acc = _evaluate(model, loader, torch.device("cpu"))
    # input indices 0..5 → predicted classes 0,1,2,3,0,1; matches labels exactly
    assert acc == 1.0


def test_evaluate_zero_when_all_wrong() -> None:
    logits = torch.tensor([[5.0, 0.0, 0.0]] * 4)
    inputs = torch.arange(4, dtype=torch.long)
    labels = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    model = _IdentityClassifier(logits)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    acc = _evaluate(model, loader, torch.device("cpu"))
    assert acc == 0.0
