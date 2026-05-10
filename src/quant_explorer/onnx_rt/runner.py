"""End-to-end orchestration for the cross-runtime comparison.

Holds the glue between the PyTorch side (loading a quant config's
in-runtime module + reading PT-side bench results) and the ONNX side
(export, quantize, ORT inference). Kept separate from ``compare.py``
so the comparison data type stays import-light (no torch).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from ..bench.latency import benchmark_latency
from ..eval.accuracy import evaluate_accuracy
from .bench import bench_onnx_latency, onnx_top1_accuracy
from .compare import CrossRuntimeResult
from .export import export_fp32_onnx
from .quantize import quantize_dynamic_int8_onnx, quantize_static_int8_onnx

# Configs that the cross-runtime path supports. ``qat_int8`` is omitted
# (QAT export to ONNX needs a different code path than PTQ; tracked as
# follow-up). The static_int8_per_channel + static_int8_per_tensor pair
# is the headline comparison.
CROSS_RUNTIME_CONFIGS: tuple[str, ...] = (
    "fp32_baseline",
    "dynamic_int8",
    "static_int8_per_tensor",
    "static_int8_per_channel",
)


@dataclass(frozen=True)
class _ONNXArtifact:
    """The ONNX file for one config and its sidecar size in KB."""

    path: Path
    size_kb: float


def _file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def _calibration_numpy_batches(
    loader: DataLoader[Any],
) -> Iterable[NDArray[np.float32]]:
    """Stream ``(images,)`` batches from a DataLoader as float32 NumPy arrays."""
    for images, _labels in loader:
        if isinstance(images, torch.Tensor):
            yield images.detach().cpu().numpy().astype(np.float32)
        else:  # pragma: no cover - DataLoader yields tensors in practice
            yield np.asarray(images, dtype=np.float32)


def _labelled_numpy_batches(
    loader: DataLoader[Any],
) -> Iterable[tuple[NDArray[np.float32], NDArray[np.int64]]]:
    for images, labels in loader:
        x = images.detach().cpu().numpy().astype(np.float32)
        y = labels.detach().cpu().numpy().astype(np.int64)
        yield x, y


def build_onnx_artifacts(
    *,
    fp32_model: nn.Module,
    out_dir: Path,
    calibration_loader: DataLoader[Any],
    configs: Iterable[str] = CROSS_RUNTIME_CONFIGS,
) -> dict[str, _ONNXArtifact]:
    """Materialize one ``.onnx`` file per config under ``out_dir``.

    ``fp32_model`` must already have the trained FP32 weights loaded; the
    function takes care of exporting once and reusing the FP32 file as
    the source for the INT8 paths. Returns a mapping of
    ``config_name -> (.onnx path, size_kb)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "fp32_baseline.onnx"
    export_fp32_onnx(fp32_model, fp32_onnx)

    # Materialize calibration batches once (the static quantizer consumes
    # the iterator twice for per-tensor + per-channel — collecting up
    # front keeps both passes deterministic).
    calibration_arrays = list(_calibration_numpy_batches(calibration_loader))

    artifacts: dict[str, _ONNXArtifact] = {
        "fp32_baseline": _ONNXArtifact(path=fp32_onnx, size_kb=_file_size_kb(fp32_onnx)),
    }
    for name in configs:
        if name == "fp32_baseline":
            continue
        out_path = out_dir / f"{name}.onnx"
        if name == "dynamic_int8":
            quantize_dynamic_int8_onnx(fp32_onnx, out_path)
        elif name == "static_int8_per_tensor":
            quantize_static_int8_onnx(
                fp32_onnx,
                out_path,
                calibration_batches=iter(calibration_arrays),
                per_channel=False,
            )
        elif name == "static_int8_per_channel":
            quantize_static_int8_onnx(
                fp32_onnx,
                out_path,
                calibration_batches=iter(calibration_arrays),
                per_channel=True,
            )
        else:
            raise ValueError(f"unsupported cross-runtime config: {name!r}")
        artifacts[name] = _ONNXArtifact(path=out_path, size_kb=_file_size_kb(out_path))
    return artifacts


@dataclass(frozen=True)
class PyTorchSideMeasurement:
    """PT-side numbers fed into the cross-runtime comparison."""

    top1: float
    p50_ms_b1: float
    size_kb: float
    n_samples: int


def measure_pytorch_side(
    model_builder: Callable[[], nn.Module],
    *,
    weights_path: Path,
    test_loader: DataLoader[Any],
    bench_warmup: int,
    bench_iters: int,
) -> PyTorchSideMeasurement:
    """Bench latency + accuracy + on-disk size for one PT config.

    ``model_builder`` is invoked twice (once for the accuracy pass, once
    for the latency pass) so the same module isn't re-used with cached
    state across runs.
    """
    acc_model = model_builder()
    acc = evaluate_accuracy(acc_model, test_loader)

    lat_model = model_builder()
    lat_result = benchmark_latency(
        lat_model,
        batch_size=1,
        n_warmup=bench_warmup,
        n_measure=bench_iters,
    )

    size_kb = weights_path.stat().st_size / 1024.0 if weights_path.exists() else 0.0
    return PyTorchSideMeasurement(
        top1=float(acc.top1),
        p50_ms_b1=float(lat_result.p50_ms),
        size_kb=size_kb,
        n_samples=int(acc.n_samples),
    )


def measure_onnx_side(
    *,
    onnx_artifact: _ONNXArtifact,
    test_loader: DataLoader[Any],
    bench_warmup: int,
    bench_iters: int,
) -> tuple[float, float, int]:
    """Bench latency + accuracy under ORT CPU EP. Returns (top1, p50_ms, n)."""
    top1, n = onnx_top1_accuracy(
        str(onnx_artifact.path),
        _labelled_numpy_batches(test_loader),
    )
    lat = bench_onnx_latency(
        str(onnx_artifact.path),
        batch_size=1,
        n_warmup=bench_warmup,
        n_measure=bench_iters,
    )
    return top1, float(lat.p50_ms), n


def assemble_row(
    *,
    config: str,
    pt: PyTorchSideMeasurement,
    onnx_top1: float,
    onnx_p50_ms: float,
    onnx_size_kb: float,
    n_samples_onnx: int,
) -> CrossRuntimeResult:
    """Materialize one comparison row.

    The number of samples reported is the minimum of the two — if the
    runtimes were fed different-size loaders we want the cross-section
    they share, not the larger of the two.
    """
    return CrossRuntimeResult(
        config=config,
        pt_top1=pt.top1,
        onnx_top1=onnx_top1,
        pt_p50_ms=pt.p50_ms_b1,
        onnx_p50_ms=onnx_p50_ms,
        pt_size_kb=pt.size_kb,
        onnx_size_kb=onnx_size_kb,
        n_samples=min(pt.n_samples, n_samples_onnx) if n_samples_onnx > 0 else pt.n_samples,
    )
