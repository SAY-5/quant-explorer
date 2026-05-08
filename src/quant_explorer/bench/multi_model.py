"""Multi-model bench harness.

Runs the 4 quantization configs against the 3 registered models and
emits a 12-cell Pareto-style report. Latency + on-disk size are measured
for every cell; accuracy is measured only for ``small_cnn`` (the only
model trained on CIFAR-10). For the torchvision models accuracy is left
as ``None`` and explicitly labelled "not measured" in the report.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .. import quant as quant_pkg
from ..models import REGISTRY as MODEL_REGISTRY
from ..models import ModelSpec, get_model_spec
from .latency import LatencyResult, benchmark_latency
from .size import file_size

ALL_QUANT_CONFIGS = (
    "fp32_baseline",
    "dynamic_int8",
    "static_int8_per_tensor",
    "static_int8_per_channel",
)


@dataclass(frozen=True)
class CellResult:
    """One (model, quant_config) cell of the multi-model grid."""

    model: str
    quant_config: str
    size_kb: float
    latency: LatencyResult
    accuracy_top1: float | None  # None -> not measured for this model

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "quant_config": self.quant_config,
            "size_kb": self.size_kb,
            "latency": self.latency.as_dict(),
            "accuracy_top1": self.accuracy_top1,
        }


def _synthetic_calibration(
    spec: ModelSpec, *, batch_size: int = 4, n_batches: int = 4
) -> Iterator[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    c, h, w = spec.input_shape
    for _ in range(n_batches):
        yield torch.randn(batch_size, c, h, w, generator=g)


def _build_quantized(
    spec: ModelSpec,
    quant_config: str,
    *,
    weight_loader: Callable[[ModelSpec], nn.Module] | None = None,
) -> nn.Module:
    """Build a quantized variant of ``spec`` for the given config.

    If ``weight_loader`` is provided it's called instead of ``spec.builder()``
    to obtain the FP32 model (the small_cnn cell uses this to load the
    trained baseline weights so reported accuracy reflects the actual
    network rather than random init).
    """
    model = weight_loader(spec) if weight_loader is not None else spec.builder()
    model.eval()
    if quant_config == "fp32_baseline":
        return model
    cfg = quant_pkg.get_config(quant_config)
    if cfg.needs_calibration:
        return cfg.apply(model, _synthetic_calibration(spec))
    return cfg.apply(model, None)


def bench_cell(
    model_name: str,
    quant_config: str,
    *,
    bench_batch_size: int = 1,
    n_warmup: int = 2,
    n_measure: int = 20,
    accuracy_fn: Callable[[nn.Module], float] | None = None,
    weight_loaders: dict[str, Callable[[ModelSpec], nn.Module]] | None = None,
) -> CellResult:
    """Bench one (model, quant_config) cell.

    ``accuracy_fn`` is invoked on the quantized model only for models
    where ``measures_accuracy`` is True; for others the cell's accuracy
    is left as ``None``.

    ``weight_loaders`` maps model names to callables that return an FP32
    module with trained weights loaded; falls back to ``spec.builder()``
    (random init) for any model not in the dict.
    """
    spec = get_model_spec(model_name)
    weight_loader = (weight_loaders or {}).get(model_name)
    qm = _build_quantized(spec, quant_config, weight_loader=weight_loader)

    # On-disk size — saved through a temp file so the size test
    # mirrors what would land in ``artifacts/weights/`` without
    # littering the workspace.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        torch.save(qm.state_dict(), tmp_path)
        size_kb = file_size(tmp_path).kb
    finally:
        tmp_path.unlink(missing_ok=True)

    # Latency at the given batch size, with the model's native input shape.
    lat = benchmark_latency(
        qm,
        batch_size=bench_batch_size,
        input_shape=spec.input_shape,
        n_warmup=n_warmup,
        n_measure=n_measure,
    )

    acc: float | None = None
    if spec.measures_accuracy and accuracy_fn is not None:
        acc = float(accuracy_fn(qm))

    return CellResult(
        model=model_name,
        quant_config=quant_config,
        size_kb=size_kb,
        latency=lat,
        accuracy_top1=acc,
    )


def bench_grid(
    *,
    models: tuple[str, ...] = tuple(MODEL_REGISTRY),
    quant_configs: tuple[str, ...] = ALL_QUANT_CONFIGS,
    bench_batch_size: int = 1,
    n_warmup: int = 2,
    n_measure: int = 20,
    accuracy_fn: Callable[[nn.Module], float] | None = None,
    weight_loaders: dict[str, Callable[[ModelSpec], nn.Module]] | None = None,
) -> list[CellResult]:
    """Run the full ``len(models) * len(quant_configs)`` grid."""
    cells: list[CellResult] = []
    for m in models:
        for c in quant_configs:
            cells.append(
                bench_cell(
                    m,
                    c,
                    bench_batch_size=bench_batch_size,
                    n_warmup=n_warmup,
                    n_measure=n_measure,
                    accuracy_fn=accuracy_fn,
                    weight_loaders=weight_loaders,
                )
            )
    return cells


def grid_to_pareto_rows(cells: list[CellResult]) -> list[dict[str, Any]]:
    """Group cells into one Pareto computation per model.

    Pareto comparisons must stay within a model — comparing latency
    between a 32x32 small CNN and a 224x224 VGG would be meaningless.
    For models without measured accuracy the size/latency frontier
    is computed assuming a fixed reference accuracy of 1.0 (so the
    accuracy axis collapses); the report calls this out explicitly.
    """
    by_model: dict[str, list[CellResult]] = {}
    for cell in cells:
        by_model.setdefault(cell.model, []).append(cell)
    rows: list[dict[str, Any]] = []
    for model_name, model_cells in by_model.items():
        for cell in model_cells:
            top1 = cell.accuracy_top1 if cell.accuracy_top1 is not None else 1.0
            rows.append(
                {
                    "name": f"{model_name}/{cell.quant_config}",
                    "model": model_name,
                    "quant_config": cell.quant_config,
                    "size_kb": cell.size_kb,
                    "p50_lat_ms_b1": cell.latency.p50_ms,
                    "top1_acc": top1,
                    "accuracy_measured": cell.accuracy_top1 is not None,
                }
            )
    return rows


def render_multi_model_markdown(cells: list[CellResult]) -> str:
    """Markdown table grouped by model with a per-model frontier marker.

    Frontier within a model: a config is dominated only by configs of
    the *same* model that are strictly better on every measured axis.
    Cross-model comparisons are deliberately not made.
    """
    from ..report.pareto import ParetoPoint, pareto_frontier

    by_model: dict[str, list[CellResult]] = {}
    for cell in cells:
        by_model.setdefault(cell.model, []).append(cell)

    lines = [
        "# Multi-model quantization Pareto",
        "",
        "Each model has its own frontier (cross-model latency / size "
        "comparisons aren't meaningful — different input shapes, "
        "parameter counts, and intended deployment targets).",
        "",
    ]

    # Per-model fp32 reference for ratio columns.
    for model_name in sorted(by_model):
        model_cells = by_model[model_name]
        fp32 = next((c for c in model_cells if c.quant_config == "fp32_baseline"), None)
        if fp32 is None:
            continue
        base_size = fp32.size_kb if fp32.size_kb > 0 else 1.0
        base_lat = fp32.latency.p50_ms if fp32.latency.p50_ms > 0 else 1.0
        spec = get_model_spec(model_name)

        # Frontier within this model only.
        points = [
            ParetoPoint(
                name=c.quant_config,
                size_kb=c.size_kb,
                p50_lat_ms_b1=c.latency.p50_ms,
                top1_acc=(c.accuracy_top1 if c.accuracy_top1 is not None else 0.0),
            )
            for c in model_cells
        ]
        frontier = pareto_frontier(points)

        lines.append(f"## {model_name}")
        lines.append(
            f"Input shape: {spec.input_shape}. "
            + (
                "Top-1 measured on the CIFAR-10 test split."
                if spec.measures_accuracy
                else "Top-1 not measured (ImageNet domain — see README)."
            )
        )
        lines.append("")
        lines.append("| quant_config | size_kb | size_ratio | p50_ms | speedup | top1 | pareto |")
        lines.append("|---|---:|---:|---:|---:|---:|:---:|")
        for c in model_cells:
            ratio = c.size_kb / base_size
            speedup = base_lat / c.latency.p50_ms if c.latency.p50_ms > 0 else 0.0
            top1_str = f"{c.accuracy_top1 * 100:.1f}%" if c.accuracy_top1 is not None else "n/a"
            mark = "yes" if c.quant_config in frontier else "no"
            lines.append(
                f"| {c.quant_config} | {c.size_kb:.0f} | {ratio:.2f}x | "
                f"{c.latency.p50_ms:.2f} | {speedup:.2f}x | {top1_str} | {mark} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def emit_multi_model_results(cells: list[CellResult], out_dir: Path) -> tuple[Path, Path]:
    """Write multi_model.json + multi_pareto.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "multi_model.json"
    md_path = out_dir / "multi_pareto.md"
    payload = {
        "cells": [c.as_dict() for c in cells],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_multi_model_markdown(cells), encoding="utf-8")
    return json_path, md_path
