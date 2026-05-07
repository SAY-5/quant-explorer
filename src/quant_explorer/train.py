"""CIFAR-10 training loop. SGD + momentum, cosine LR, 5 epochs by default."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import get_test_loader, get_train_loader
from .model import CifarCNN, count_parameters
from .settings import TrainConfig


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return correct / max(total, 1)


def train_model(
    cfg: TrainConfig,
    *,
    data_dir: Path,
    out_path: Path,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> dict[str, float | int]:
    _set_seed(cfg.seed)
    device = torch.device("cpu")

    train_loader = get_train_loader(
        data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        subset_size=train_subset,
    )
    test_loader = get_test_loader(
        data_dir,
        batch_size=256,
        num_workers=cfg.num_workers,
        subset_size=test_subset,
    )

    model = CifarCNN(num_classes=10, quantizable=True).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        test_acc = _evaluate(model, test_loader, device)
        print(
            f"epoch {epoch + 1}/{cfg.epochs} "
            f"loss={train_loss:.4f} test_acc={test_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.4f}"
        )
    wall = time.perf_counter() - start

    final_acc = _evaluate(model, test_loader, device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    return {
        "n_params": n_params,
        "final_test_acc": final_acc,
        "wall_seconds": wall,
        "epochs": cfg.epochs,
    }
