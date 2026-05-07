"""Top-1 / Top-5 accuracy on the CIFAR-10 test split, with per-class breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..settings import CIFAR10_CLASSES


@dataclass
class PerClassAggregator:
    """Running per-class top-1 counts. Tested directly in unit tests."""

    n_classes: int
    correct: list[int] = field(default_factory=list)
    total: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.correct:
            self.correct = [0] * self.n_classes
        if not self.total:
            self.total = [0] * self.n_classes

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        if preds.shape != labels.shape:
            raise ValueError(f"shape mismatch: preds={preds.shape} labels={labels.shape}")
        for pred, label in zip(preds.tolist(), labels.tolist(), strict=True):
            if not (0 <= int(label) < self.n_classes):
                raise ValueError(f"label {label} out of range [0, {self.n_classes})")
            self.total[int(label)] += 1
            if int(pred) == int(label):
                self.correct[int(label)] += 1

    def per_class_accuracy(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for i in range(self.n_classes):
            out[i] = self.correct[i] / self.total[i] if self.total[i] > 0 else 0.0
        return out


@dataclass(frozen=True)
class AccuracyResult:
    top1: float
    top5: float
    per_class: dict[str, float]
    n_samples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "top1": self.top1,
            "top5": self.top5,
            "per_class": self.per_class,
            "n_samples": self.n_samples,
        }


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    *,
    class_names: tuple[str, ...] = CIFAR10_CLASSES,
) -> AccuracyResult:
    """Top-1 + top-5 accuracy plus per-class top-1 breakdown."""
    model.eval()
    n_classes = len(class_names)
    agg = PerClassAggregator(n_classes=n_classes)
    top5_correct = 0
    n = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            top1 = logits.argmax(dim=1)
            agg.update(top1, labels)

            k = min(5, logits.shape[1])
            _, top5 = logits.topk(k, dim=1)
            match = (top5 == labels.unsqueeze(1)).any(dim=1)
            top5_correct += int(match.sum().item())
            n += labels.size(0)

    top1_overall = sum(agg.correct) / max(n, 1)
    top5_overall = top5_correct / max(n, 1)

    per_class = agg.per_class_accuracy()
    per_class_named = {class_names[i]: per_class[i] for i in range(n_classes)}

    return AccuracyResult(
        top1=top1_overall,
        top5=top5_overall,
        per_class=per_class_named,
        n_samples=n,
    )
