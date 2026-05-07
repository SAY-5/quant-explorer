"""Per-class aggregator + top-1/top-5 accuracy with synthetic logits."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quant_explorer.eval.accuracy import PerClassAggregator, evaluate_accuracy


def test_per_class_aggregator_counts_correctly() -> None:
    agg = PerClassAggregator(n_classes=3)
    preds = torch.tensor([0, 1, 1, 2, 2, 0])
    labels = torch.tensor([0, 1, 0, 2, 1, 0])
    agg.update(preds, labels)
    pc = agg.per_class_accuracy()
    # class 0: 3 examples, correct = preds[0]==0, preds[5]==0; preds[2]!=0 -> 2/3
    assert pc[0] == pytest.approx(2.0 / 3.0)
    # class 1: 2 examples, correct = preds[1]==1; preds[4]=2!=1 -> 1/2
    assert pc[1] == pytest.approx(0.5)
    # class 2: 1 example, preds[3]=2 -> 1/1
    assert pc[2] == pytest.approx(1.0)


def test_per_class_aggregator_handles_class_with_no_examples() -> None:
    agg = PerClassAggregator(n_classes=4)
    preds = torch.tensor([0])
    labels = torch.tensor([0])
    agg.update(preds, labels)
    pc = agg.per_class_accuracy()
    assert pc[0] == pytest.approx(1.0)
    assert pc[1] == pytest.approx(0.0)
    assert pc[2] == pytest.approx(0.0)
    assert pc[3] == pytest.approx(0.0)


def test_per_class_aggregator_rejects_shape_mismatch() -> None:
    agg = PerClassAggregator(n_classes=2)
    with pytest.raises(ValueError):
        agg.update(torch.tensor([0, 1, 0]), torch.tensor([0, 1]))


def test_per_class_aggregator_rejects_oob_label() -> None:
    agg = PerClassAggregator(n_classes=2)
    with pytest.raises(ValueError):
        agg.update(torch.tensor([0]), torch.tensor([5]))


class _DeterministicHead(nn.Module):
    """Returns a fixed logit table indexed by the input scalar."""

    def __init__(self, logit_table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logit_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.long().view(-1)
        return self.logits[idx]


def test_evaluate_accuracy_with_synthetic_predictions() -> None:
    """3-class problem with hand-built logits."""
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],  # predicts 0
            [0.0, 3.0, 0.0],  # predicts 1
            [0.0, 0.0, 3.0],  # predicts 2
            [3.0, 0.0, 0.0],  # predicts 0
        ]
    )
    inputs = torch.tensor([0, 1, 2, 3])  # row indices
    labels = torch.tensor([0, 1, 0, 0])
    model = _DeterministicHead(logits)
    ds = TensorDataset(inputs, labels)
    loader = DataLoader(ds, batch_size=2)
    result = evaluate_accuracy(model, loader, class_names=("a", "b", "c"))
    assert result.n_samples == 4
    assert result.top1 == pytest.approx(0.75)
    # top-5 with only 3 classes: every label appears in the top-3, so 100%
    assert result.top5 == pytest.approx(1.0)
    assert result.per_class["a"] == pytest.approx(2.0 / 3.0)
    assert result.per_class["b"] == pytest.approx(1.0)
    assert result.per_class["c"] == pytest.approx(0.0)
