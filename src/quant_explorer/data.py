"""CIFAR-10 dataset loaders, transforms, and calibration sampling."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .settings import CIFAR10_MEAN, CIFAR10_STD


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_train_dataset(data_dir: Path, *, augment: bool = True) -> Dataset[tuple[torch.Tensor, int]]:
    transform = _train_transform() if augment else _eval_transform()
    ds: Dataset[tuple[torch.Tensor, int]] = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    return ds


def get_test_dataset(data_dir: Path) -> Dataset[tuple[torch.Tensor, int]]:
    ds: Dataset[tuple[torch.Tensor, int]] = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=_eval_transform(),
    )
    return ds


def get_train_loader(
    data_dir: Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    subset_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, int]]:
    ds: Dataset[tuple[torch.Tensor, int]] = get_train_dataset(data_dir)
    if subset_size is not None:
        ds = Subset(ds, list(range(min(subset_size, len(ds)))))  # type: ignore[arg-type]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_loader(
    data_dir: Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    subset_size: int | None = None,
) -> DataLoader[tuple[torch.Tensor, int]]:
    ds: Dataset[tuple[torch.Tensor, int]] = get_test_dataset(data_dir)
    if subset_size is not None:
        ds = Subset(ds, list(range(min(subset_size, len(ds)))))  # type: ignore[arg-type]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_calibration_loader(
    data_dir: Path,
    *,
    n_images: int,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Deterministic calibration loader (no augmentation, no shuffle).

    Calibration must be run on un-augmented images so the activation
    statistics observed match what the model sees at inference time.
    """
    ds: Dataset[tuple[torch.Tensor, int]] = get_train_dataset(data_dir, augment=False)
    ds = Subset(ds, list(range(min(n_images, len(ds)))))  # type: ignore[arg-type]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def iter_calibration_batches(
    loader: DataLoader[tuple[torch.Tensor, int]],
) -> Iterator[torch.Tensor]:
    for images, _labels in loader:
        yield images
