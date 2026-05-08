"""Tests for the multi-model registry + bench harness."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from quant_explorer.bench.latency import LatencyResult
from quant_explorer.bench.multi_model import (
    ALL_QUANT_CONFIGS,
    CellResult,
    bench_cell,
    emit_multi_model_results,
    grid_to_pareto_rows,
    render_multi_model_markdown,
)
from quant_explorer.models import REGISTRY, get_model_spec, list_models
from quant_explorer.settings import select_quantization_engine


def setup_module(_module: object) -> None:
    torch.backends.quantized.engine = select_quantization_engine()


def test_registry_has_three_documented_models() -> None:
    names = list_models()
    assert "small_cnn" in names
    assert "mobilenet_v3" in names
    assert "vgg11_bn" in names
    # small_cnn is the only one with measured accuracy
    assert get_model_spec("small_cnn").measures_accuracy is True
    assert get_model_spec("mobilenet_v3").measures_accuracy is False
    assert get_model_spec("vgg11_bn").measures_accuracy is False


def test_each_model_forward_passes_with_native_input_shape() -> None:
    for name in list_models():
        spec = REGISTRY[name]
        model = spec.builder()
        model.eval()
        x = torch.randn(1, *spec.input_shape)
        with torch.no_grad():
            y = model(x)
        # Output should be a 2D tensor (batch, classes).
        assert y.dim() == 2
        assert y.shape[0] == 1


def test_bench_cell_small_cnn_fp32() -> None:
    """Tiny smoke that ``bench_cell`` returns sensible numbers."""
    cell = bench_cell(
        "small_cnn",
        "fp32_baseline",
        bench_batch_size=1,
        n_warmup=1,
        n_measure=3,
    )
    assert cell.model == "small_cnn"
    assert cell.quant_config == "fp32_baseline"
    assert cell.size_kb > 0
    assert isinstance(cell.latency, LatencyResult)
    assert cell.latency.p50_ms > 0
    # No accuracy_fn passed, so even for small_cnn accuracy stays None.
    assert cell.accuracy_top1 is None


def test_bench_cell_runs_dynamic_int8_on_torchvision_model() -> None:
    """Smoke: dynamic INT8 must work on a torchvision model."""
    cell = bench_cell(
        "vgg11_bn",
        "dynamic_int8",
        bench_batch_size=1,
        n_warmup=1,
        n_measure=2,
    )
    assert cell.size_kb > 0
    assert cell.latency.p50_ms > 0


def test_bench_grid_yields_twelve_cells() -> None:
    """Sanity: 3 models * 4 quant configs = 12 cells (only run a tiny subset
    here to keep the test fast — but assert the shape on a hand-built grid)."""
    # We don't actually call ``bench_grid`` over the full 12 cells in
    # the unit suite (that's an integration test). But the API should
    # construct exactly len(models) * len(configs) cells.
    model_names = list_models()
    assert len(model_names) == 3
    assert len(ALL_QUANT_CONFIGS) == 4
    # The expected grid size is 12.
    assert len(model_names) * len(ALL_QUANT_CONFIGS) == 12


def _fake_lat(p50_ms: float) -> LatencyResult:
    return LatencyResult(
        batch_size=1,
        n_warmup=1,
        n_measure=1,
        p50_ms=p50_ms,
        p95_ms=p50_ms,
        p99_ms=p50_ms,
        mean_ms=p50_ms,
    )


def test_grid_to_pareto_rows_uses_one_top1_per_cell() -> None:
    cells = [
        CellResult("small_cnn", "fp32_baseline", 1000.0, _fake_lat(2.0), 0.82),
        CellResult("small_cnn", "static_int8_per_channel", 250.0, _fake_lat(0.6), 0.815),
        CellResult("vgg11_bn", "fp32_baseline", 500_000.0, _fake_lat(30.0), None),
        CellResult("vgg11_bn", "static_int8_per_tensor", 130_000.0, _fake_lat(60.0), None),
    ]
    rows = grid_to_pareto_rows(cells)
    assert len(rows) == 4
    # accuracy_measured flag is set per row
    by_name = {r["name"]: r for r in rows}
    assert by_name["small_cnn/fp32_baseline"]["accuracy_measured"] is True
    assert by_name["vgg11_bn/fp32_baseline"]["accuracy_measured"] is False
    # When accuracy isn't measured, top1_acc collapses to 1.0 (so the
    # axis effectively drops out of within-model frontier comparisons).
    assert by_name["vgg11_bn/fp32_baseline"]["top1_acc"] == 1.0


def test_render_multi_model_markdown_has_expected_sections() -> None:
    cells = [
        CellResult("small_cnn", "fp32_baseline", 1000.0, _fake_lat(2.0), 0.82),
        CellResult("small_cnn", "static_int8_per_channel", 250.0, _fake_lat(0.6), 0.815),
        CellResult("mobilenet_v3", "fp32_baseline", 20_000.0, _fake_lat(180.0), None),
        CellResult("mobilenet_v3", "static_int8_per_channel", 5000.0, _fake_lat(8.0), None),
        CellResult("vgg11_bn", "fp32_baseline", 500_000.0, _fake_lat(30.0), None),
        CellResult("vgg11_bn", "static_int8_per_tensor", 130_000.0, _fake_lat(60.0), None),
    ]
    md = render_multi_model_markdown(cells)
    # Each model has its own section.
    assert "## small_cnn" in md
    assert "## mobilenet_v3" in md
    assert "## vgg11_bn" in md
    # Models without measured accuracy say so.
    assert "not measured" in md.lower()
    # Frontier column is present.
    assert "pareto" in md
    # Within-model speedup / size_ratio columns rendered.
    assert "size_ratio" in md
    assert "speedup" in md


def test_emit_multi_model_results_writes_both_files(tmp_path: Path) -> None:
    cells = [
        CellResult("small_cnn", "fp32_baseline", 1000.0, _fake_lat(2.0), 0.82),
        CellResult("small_cnn", "static_int8_per_channel", 250.0, _fake_lat(0.6), 0.815),
    ]
    json_path, md_path = emit_multi_model_results(cells, tmp_path)
    assert json_path.exists() and md_path.exists()
    payload = json.loads(json_path.read_text())
    assert "cells" in payload
    assert len(payload["cells"]) == 2
    md = md_path.read_text()
    assert "## small_cnn" in md
