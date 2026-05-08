"""Hypothesis property tests for the accuracy aggregator + top-1 / top-5 numerics.

Properties:

1. ``PerClassAggregator``: per-class correct counts sum to total correct;
   per-class total counts sum to ``n``.
2. Top-1 from ``evaluate_accuracy`` matches a hand-rolled NumPy reference.
3. Top-5 from ``evaluate_accuracy`` matches a hand-rolled NumPy reference.
4. Top-1 <= top-5 always (top-1 is a special case of top-k).
5. Per-class breakdown sum-of-corrects equals overall top-1 numerator.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quant_explorer.eval.accuracy import PerClassAggregator, evaluate_accuracy

# Keep dimensions small but non-trivial so Hypothesis explores corner cases.
_n_classes_strategy = st.integers(min_value=2, max_value=6)
_n_samples_strategy = st.integers(min_value=1, max_value=24)


@st.composite
def _preds_labels(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor, int]:
    n_classes = draw(_n_classes_strategy)
    n = draw(_n_samples_strategy)
    preds_list = draw(st.lists(st.integers(0, n_classes - 1), min_size=n, max_size=n))
    labels_list = draw(st.lists(st.integers(0, n_classes - 1), min_size=n, max_size=n))
    return (
        torch.tensor(preds_list, dtype=torch.long),
        torch.tensor(labels_list, dtype=torch.long),
        n_classes,
    )


@settings(max_examples=200, deadline=None)
@given(_preds_labels())
def test_aggregator_sums_match_total(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    preds, labels, n_classes = data
    agg = PerClassAggregator(n_classes=n_classes)
    agg.update(preds, labels)

    n_correct = int((preds == labels).sum().item())
    n_total = labels.numel()

    assert sum(agg.correct) == n_correct
    assert sum(agg.total) == n_total


@settings(max_examples=100, deadline=None)
@given(_preds_labels())
def test_aggregator_per_class_accuracy_matches_reference(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    preds, labels, n_classes = data
    agg = PerClassAggregator(n_classes=n_classes)
    agg.update(preds, labels)
    pc = agg.per_class_accuracy()

    for cls in range(n_classes):
        mask = labels == cls
        total = int(mask.sum().item())
        if total == 0:
            assert pc[cls] == 0.0
        else:
            correct = int(((preds == labels) & mask).sum().item())
            assert pc[cls] == correct / total


class _LogitTable(nn.Module):
    """Maps each integer input row-index to a fixed logit row."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x.long().view(-1)]


@st.composite
def _logits_and_labels(
    draw: st.DrawFn,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    n_classes = draw(st.integers(min_value=2, max_value=8))
    n = draw(st.integers(min_value=1, max_value=20))
    # Use floats with a coarse grid so ties happen rarely (when they do, both
    # the reference and the implementation behave consistently — ``argmax``
    # picks the first index).
    flat = draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
            min_size=n * n_classes,
            max_size=n * n_classes,
        )
    )
    logits = torch.tensor(flat, dtype=torch.float32).reshape(n, n_classes)
    labels = torch.tensor(
        draw(st.lists(st.integers(0, n_classes - 1), min_size=n, max_size=n)),
        dtype=torch.long,
    )
    return logits, labels, n_classes


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_logits_and_labels())
def test_evaluate_accuracy_top1_matches_reference(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    logits, labels, n_classes = data
    n = logits.shape[0]
    inputs = torch.arange(n, dtype=torch.long)
    model = _LogitTable(logits)
    ds = TensorDataset(inputs, labels)
    loader = DataLoader(ds, batch_size=4)
    class_names = tuple(f"c{i}" for i in range(n_classes))
    result = evaluate_accuracy(model, loader, class_names=class_names)

    # Reference top-1 via numpy.
    np_logits = logits.numpy()
    np_labels = labels.numpy()
    pred = np_logits.argmax(axis=1)
    expected_top1 = float((pred == np_labels).mean())

    assert result.n_samples == n
    assert abs(result.top1 - expected_top1) < 1e-6


@st.composite
def _logits_no_ties_and_labels(
    draw: st.DrawFn,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Same as ``_logits_and_labels`` but each row's logits are unique.

    Top-k under ties is implementation-defined (``torch.topk`` doesn't
    promise lower-index wins), so testing top-5 needs unique logits per row.
    """
    n_classes = draw(st.integers(min_value=2, max_value=8))
    n = draw(st.integers(min_value=1, max_value=12))
    rows: list[list[float]] = []
    for _ in range(n):
        # A permutation of n_classes distinct values guarantees no ties.
        base = list(range(n_classes))
        perm = draw(st.permutations(base))
        # Float-ify with arbitrary offset/scale so numerics aren't trivial.
        scale = draw(st.floats(min_value=0.5, max_value=5.0, allow_nan=False))
        offset = draw(st.floats(min_value=-3.0, max_value=3.0, allow_nan=False))
        rows.append([offset + scale * v for v in perm])
    logits = torch.tensor(rows, dtype=torch.float32)
    labels = torch.tensor(
        draw(st.lists(st.integers(0, n_classes - 1), min_size=n, max_size=n)),
        dtype=torch.long,
    )
    return logits, labels, n_classes


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_logits_no_ties_and_labels())
def test_evaluate_accuracy_top5_matches_reference(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    logits, labels, n_classes = data
    n = logits.shape[0]
    inputs = torch.arange(n, dtype=torch.long)
    model = _LogitTable(logits)
    ds = TensorDataset(inputs, labels)
    loader = DataLoader(ds, batch_size=4)
    class_names = tuple(f"c{i}" for i in range(n_classes))
    result = evaluate_accuracy(model, loader, class_names=class_names)

    k = min(5, n_classes)
    np_logits = logits.numpy()
    np_labels = labels.numpy()
    # No ties guaranteed by the strategy, so argpartition gives a canonical
    # top-k set.
    top_k_idx = np.argpartition(-np_logits, kth=k - 1, axis=1)[:, :k]
    hit = np.array([np_labels[i] in top_k_idx[i] for i in range(n)])
    expected_top5 = float(hit.mean())

    assert abs(result.top5 - expected_top5) < 1e-6


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_logits_and_labels())
def test_top1_le_top5(data: tuple[torch.Tensor, torch.Tensor, int]) -> None:
    logits, labels, n_classes = data
    n = logits.shape[0]
    inputs = torch.arange(n, dtype=torch.long)
    model = _LogitTable(logits)
    ds = TensorDataset(inputs, labels)
    loader = DataLoader(ds, batch_size=4)
    class_names = tuple(f"c{i}" for i in range(n_classes))
    result = evaluate_accuracy(model, loader, class_names=class_names)
    assert result.top1 <= result.top5 + 1e-9


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_logits_and_labels())
def test_per_class_breakdown_consistent_with_overall(
    data: tuple[torch.Tensor, torch.Tensor, int],
) -> None:
    """Sum of per-class corrects should equal overall top-1 numerator."""
    logits, labels, n_classes = data
    assume(n_classes >= 2)
    n = logits.shape[0]
    inputs = torch.arange(n, dtype=torch.long)
    model = _LogitTable(logits)
    ds = TensorDataset(inputs, labels)
    loader = DataLoader(ds, batch_size=4)
    class_names = tuple(f"c{i}" for i in range(n_classes))
    result = evaluate_accuracy(model, loader, class_names=class_names)

    # Hand-rolled per-class via numpy.
    np_logits = logits.numpy()
    np_labels = labels.numpy()
    pred = np_logits.argmax(axis=1)

    correct_per_class = 0
    total_per_class = 0
    for cls in range(n_classes):
        mask = np_labels == cls
        cls_total = int(mask.sum())
        if cls_total == 0:
            # accuracy undefined — implementation reports 0.0
            assert result.per_class[class_names[cls]] == 0.0
            continue
        cls_correct = int(((pred == np_labels) & mask).sum())
        correct_per_class += cls_correct
        total_per_class += cls_total
        assert abs(result.per_class[class_names[cls]] - cls_correct / cls_total) < 1e-6

    # Sum of per-class corrects equals overall correct count.
    overall_correct = int((pred == np_labels).sum())
    assert correct_per_class == overall_correct
    assert total_per_class == n
