"""Project-wide settings, paths, and quantization-engine selection."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
WEIGHTS_DIR: Path = ARTIFACTS_DIR / "weights"
RESULTS_DIR: Path = ARTIFACTS_DIR / "results"

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# CIFAR-10 channel statistics (computed on the train split).
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def select_quantization_engine() -> str:
    """Pick the best quantization backend available for the host.

    On x86 Linux ``fbgemm`` is available; on Apple Silicon / arm the only
    option is ``qnnpack``. Both are compatible with the eager-mode
    quantization APIs used here.
    """
    supported = list(torch.backends.quantized.supported_engines)
    machine = platform.machine().lower()
    system = platform.system().lower()
    if "fbgemm" in supported and (system == "linux" or machine in {"x86_64", "amd64"}):
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"
    # last-resort: whatever isn't 'none'
    for engine in supported:
        if engine != "none":
            return engine
    raise RuntimeError("no usable quantization engine available")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 1729
    num_workers: int = 0


@dataclass(frozen=True)
class BenchConfig:
    warmup_iters: int = 10
    measure_iters: int = 200
    batch_sizes: tuple[int, ...] = (1, 8, 32)


@dataclass(frozen=True)
class CalibrationConfig:
    n_images: int = 256
    batch_size: int = 32


@dataclass(frozen=True)
class TinyConfig:
    """Sub-scale config for the integration / smoke pipeline."""

    epochs: int = 1
    train_subset: int = 200
    test_subset: int = 200
    batch_size: int = 64
    bench_warmup: int = 2
    bench_iters: int = 10
    calibration_n: int = 32


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, WEIGHTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    weights_dir: Path = field(default_factory=lambda: WEIGHTS_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
