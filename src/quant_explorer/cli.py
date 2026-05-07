"""Command-line interface for quant-explorer.

Subcommands: train, quantize, bench, evaluate, report, pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import torch
from torch import nn

from . import quant as quant_pkg
from .bench.latency import benchmark_latency
from .bench.memory import benchmark_memory
from .bench.size import file_size
from .data import get_calibration_loader, get_test_loader, iter_calibration_batches
from .eval.accuracy import evaluate_accuracy
from .model import CifarCNN
from .report.json_emit import emit_full_results
from .report.pareto import render_pareto_markdown
from .settings import (
    DATA_DIR,
    RESULTS_DIR,
    WEIGHTS_DIR,
    BenchConfig,
    CalibrationConfig,
    Paths,
    TinyConfig,
    TrainConfig,
    ensure_dirs,
    select_quantization_engine,
)
from .train import train_model

ALL_CONFIGS = ("fp32_baseline", "dynamic_int8", "static_int8_per_tensor", "static_int8_per_channel")


def _set_quant_engine() -> str:
    engine = select_quantization_engine()
    torch.backends.quantized.engine = engine
    return engine


def _baseline_path() -> Path:
    return WEIGHTS_DIR / "fp32_baseline.pt"


def _quantized_path(name: str) -> Path:
    return WEIGHTS_DIR / f"{name}.pt"


def _result_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def _load_baseline_model() -> nn.Module:
    model = CifarCNN(num_classes=10, quantizable=True)
    state = torch.load(_baseline_path(), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_quantized_model(
    name: str, *, calibration_n: int = 256, batch_size: int = 32
) -> nn.Module:
    """Apply the named quant config to a fresh copy of the baseline."""
    if name == "fp32_baseline":
        return _load_baseline_model()
    cfg = quant_pkg.get_config(name)
    base = _load_baseline_model()
    if cfg.needs_calibration:
        loader = get_calibration_loader(DATA_DIR, n_images=calibration_n, batch_size=batch_size)
        return cfg.apply(base, iter_calibration_batches(loader))
    return cfg.apply(base, None)


@click.group()
def main() -> None:
    """Quant-explorer CLI."""


@main.command()
@click.option("--epochs", type=int, default=5)
@click.option("--batch-size", type=int, default=128)
@click.option("--train-subset", type=int, default=None)
@click.option("--test-subset", type=int, default=None)
def train(epochs: int, batch_size: int, train_subset: int | None, test_subset: int | None) -> None:
    """Train the FP32 baseline and save weights to artifacts/weights/."""
    ensure_dirs()
    cfg = TrainConfig(epochs=epochs, batch_size=batch_size)
    out = _baseline_path()
    info = train_model(
        cfg,
        data_dir=DATA_DIR,
        out_path=out,
        train_subset=train_subset,
        test_subset=test_subset,
    )
    click.echo(f"saved baseline to {out}")
    click.echo(json.dumps(info, indent=2))


@main.command()
@click.option("--config", "config_name", required=True, type=click.Choice(quant_pkg.list_configs()))
@click.option("--calibration-n", type=int, default=256)
def quantize(config_name: str, calibration_n: int) -> None:
    """Apply a quant config to the baseline weights and save the quantized state_dict."""
    ensure_dirs()
    engine = _set_quant_engine()
    click.echo(f"quantization engine: {engine}")
    qm = _build_quantized_model(config_name, calibration_n=calibration_n)
    out = _quantized_path(config_name)
    torch.save(qm.state_dict(), out)
    click.echo(f"saved quantized state_dict to {out}")


def _bench_one_config(
    name: str,
    *,
    bench_cfg: BenchConfig,
    calibration_n: int,
    paths: Paths,
) -> dict[str, Any]:
    if name == "fp32_baseline":
        model = _load_baseline_model()
        weights_path = _baseline_path()
    else:
        # Re-build the quantized graph from scratch (don't try to load
        # a quantized state_dict into an FP32 module — the keys won't
        # line up because static quant rewrites the graph).
        model = _build_quantized_model(name, calibration_n=calibration_n)
        weights_path = _quantized_path(name)

    latencies = []
    for bs in bench_cfg.batch_sizes:
        r = benchmark_latency(
            model,
            batch_size=bs,
            n_warmup=bench_cfg.warmup_iters,
            n_measure=bench_cfg.measure_iters,
        )
        latencies.append(r.as_dict())

    mem = benchmark_memory(model, batch_size=32, n_iters=16).as_dict()
    size = file_size(weights_path).as_dict() if weights_path.exists() else {"bytes": 0, "kb": 0.0}

    return {
        "config": name,
        "size": size,
        "latency": latencies,
        "memory": mem,
    }


@main.command()
@click.option("--config", "config_name", required=True, type=click.Choice(ALL_CONFIGS))
@click.option("--warmup", type=int, default=10)
@click.option("--iters", type=int, default=200)
@click.option("--calibration-n", type=int, default=256)
def bench(config_name: str, warmup: int, iters: int, calibration_n: int) -> None:
    """Bench a single config (latency + memory + size); merges into per-config result JSON."""
    ensure_dirs()
    _set_quant_engine()
    bench_cfg = BenchConfig(warmup_iters=warmup, measure_iters=iters)
    out = _bench_one_config(
        config_name, bench_cfg=bench_cfg, calibration_n=calibration_n, paths=Paths()
    )

    path = _result_path(config_name)
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(out)
    emit_full_results(existing, path)
    click.echo(f"wrote {path}")


@main.command()
@click.option("--config", "config_name", required=True, type=click.Choice(ALL_CONFIGS))
@click.option("--test-subset", type=int, default=None)
@click.option("--batch-size", type=int, default=128)
@click.option("--calibration-n", type=int, default=256)
def evaluate(
    config_name: str, test_subset: int | None, batch_size: int, calibration_n: int
) -> None:
    """Evaluate top-1 / top-5 / per-class accuracy for a single config."""
    ensure_dirs()
    _set_quant_engine()
    if config_name == "fp32_baseline":
        model = _load_baseline_model()
    else:
        model = _build_quantized_model(config_name, calibration_n=calibration_n)
    loader = get_test_loader(DATA_DIR, batch_size=batch_size, subset_size=test_subset)
    acc = evaluate_accuracy(model, loader)
    path = _result_path(config_name)
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing["accuracy"] = acc.as_dict()
    existing.setdefault("config", config_name)
    emit_full_results(existing, path)
    click.echo(json.dumps({"config": config_name, **acc.as_dict()}, indent=2))


def _aggregate_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for name in ALL_CONFIGS:
        r = results.get(name)
        if r is None:
            continue
        size_kb = float(r.get("size", {}).get("kb", 0.0))
        latencies = r.get("latency", [])
        # p50 at batch 1
        b1 = next((lat for lat in latencies if lat.get("batch_size") == 1), None)
        p50 = float(b1["p50_ms"]) if b1 else float("nan")
        acc = r.get("accuracy", {})
        top1 = float(acc.get("top1", 0.0))
        mem = r.get("memory", {})
        mem_peak = float(mem.get("rss_peak_mb", 0.0))
        rows.append(
            {
                "name": name,
                "size_kb": size_kb,
                "p50_lat_ms_b1": p50,
                "top1_acc": top1,
                "mem_peak_mb": mem_peak,
            }
        )
    return rows


@main.command()
def report() -> None:
    """Aggregate per-config result JSONs into full_results.json + pareto.md."""
    ensure_dirs()
    results: dict[str, Any] = {}
    for name in ALL_CONFIGS:
        path = _result_path(name)
        if path.exists():
            results[name] = json.loads(path.read_text())
    if not results:
        raise click.UsageError("no per-config result JSONs found; run bench + evaluate first")

    full_path = RESULTS_DIR / "full_results.json"
    emit_full_results({"configs": results}, full_path)

    rows = _aggregate_rows(results)
    if not rows:
        raise click.UsageError("no rows to aggregate")
    md = render_pareto_markdown(rows)
    pareto_path = RESULTS_DIR / "pareto.md"
    pareto_path.write_text(md, encoding="utf-8")
    click.echo(f"wrote {full_path}")
    click.echo(f"wrote {pareto_path}")


@main.command()
@click.option("--tiny", is_flag=True, help="run a sub-scale pipeline (smoke / CI)")
@click.option("--epochs", type=int, default=5)
def pipeline(tiny: bool, epochs: int) -> None:
    """End-to-end: train, quantize all configs, bench, evaluate, report."""
    ensure_dirs()
    engine = _set_quant_engine()
    click.echo(f"quantization engine: {engine}")

    if tiny:
        tcfg = TinyConfig()
        train_cfg = TrainConfig(epochs=tcfg.epochs, batch_size=tcfg.batch_size)
        bench_cfg = BenchConfig(
            warmup_iters=tcfg.bench_warmup,
            measure_iters=tcfg.bench_iters,
            batch_sizes=(1, 8),
        )
        cal_n = tcfg.calibration_n
        train_subset: int | None = tcfg.train_subset
        test_subset: int | None = tcfg.test_subset
    else:
        train_cfg = TrainConfig(epochs=epochs)
        bench_cfg = BenchConfig()
        cal_n = CalibrationConfig().n_images
        train_subset = None
        test_subset = None

    info = train_model(
        train_cfg,
        data_dir=DATA_DIR,
        out_path=_baseline_path(),
        train_subset=train_subset,
        test_subset=test_subset,
    )
    click.echo(f"baseline trained: {json.dumps(info)}")

    for cfg_name in ALL_CONFIGS:
        if cfg_name != "fp32_baseline":
            qm = _build_quantized_model(cfg_name, calibration_n=cal_n)
            torch.save(qm.state_dict(), _quantized_path(cfg_name))
            click.echo(f"quantized {cfg_name}")

        out = _bench_one_config(cfg_name, bench_cfg=bench_cfg, calibration_n=cal_n, paths=Paths())
        if cfg_name == "fp32_baseline":
            model = _load_baseline_model()
        else:
            model = _build_quantized_model(cfg_name, calibration_n=cal_n)
        loader = get_test_loader(DATA_DIR, batch_size=128, subset_size=test_subset)
        acc = evaluate_accuracy(model, loader)
        out["accuracy"] = acc.as_dict()
        emit_full_results(out, _result_path(cfg_name))
        click.echo(f"benched + evaluated {cfg_name}: top1={acc.top1:.4f}")

    # Roll up the report.
    results = {
        n: json.loads(_result_path(n).read_text()) for n in ALL_CONFIGS if _result_path(n).exists()
    }
    full_path = RESULTS_DIR / "full_results.json"
    emit_full_results({"configs": results}, full_path)
    rows = _aggregate_rows(results)
    md = render_pareto_markdown(rows)
    (RESULTS_DIR / "pareto.md").write_text(md, encoding="utf-8")
    click.echo("pipeline done")


if __name__ == "__main__":
    main()
