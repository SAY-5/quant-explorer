"""Command-line interface for quant-explorer.

Subcommands: train, quantize, bench, evaluate, report, pipeline,
qat-finetune, multi-bench, cross-runtime.
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
from .bench.multi_model import bench_grid, emit_multi_model_results
from .bench.size import file_size
from .data import (
    get_calibration_loader,
    get_test_loader,
    get_train_loader,
    iter_calibration_batches,
)
from .eval.accuracy import evaluate_accuracy
from .model import CifarCNN
from .onnx_rt import (
    CROSS_RUNTIME_CONFIGS,
    assemble_row,
    build_cross_runtime_table,
    build_onnx_artifacts,
    measure_onnx_side,
    measure_pytorch_side,
    render_cross_runtime_markdown,
)
from .quant.qat import build_qat_for_eval, run_qat_finetune
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

ALL_CONFIGS = (
    "fp32_baseline",
    "dynamic_int8",
    "static_int8_per_tensor",
    "static_int8_per_channel",
    "qat_int8",
)


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
    """Apply the named quant config to a fresh copy of the baseline.

    Special-cased configs:
      * ``fp32_baseline``: just load the FP32 weights.
      * ``qat_int8``: rebuild the QAT-converted graph from the baseline
        weights (the QAT-finetuned weights live in ``qat_int8.pt`` but the
        prepare-then-convert structural transform is regenerated here so
        the converted state_dict's keys can be loaded). See
        ``quant_explorer.quant.qat`` for the full pipeline.
    """
    if name == "fp32_baseline":
        return _load_baseline_model()
    if name == "qat_int8":
        # Build the converted graph and then load the QAT state_dict on top.
        model = build_qat_for_eval(baseline_path=_baseline_path())
        qat_path = _quantized_path("qat_int8")
        if qat_path.exists():
            model.load_state_dict(torch.load(qat_path, map_location="cpu"))
        return model
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


@main.command("qat-finetune")
@click.option("--epochs", type=int, default=1, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option(
    "--train-subset",
    type=int,
    default=None,
    help="Use only the first N training images (faster, for tiny / CI runs).",
)
def qat_finetune(epochs: int, batch_size: int, lr: float, train_subset: int | None) -> None:
    """Run QAT fine-tuning starting from the FP32 baseline.

    Saves the converted (real INT8) state_dict to
    ``artifacts/weights/qat_int8.pt``. The CLI ``evaluate --config qat_int8``
    + ``bench --config qat_int8`` pick up that file the same way the PTQ
    configs do.
    """
    ensure_dirs()
    _set_quant_engine()
    train_loader = get_train_loader(DATA_DIR, batch_size=batch_size, subset_size=train_subset)
    out_path = _quantized_path("qat_int8")
    info = run_qat_finetune(
        baseline_path=_baseline_path(),
        out_path=out_path,
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
    )
    click.echo(f"saved QAT state_dict to {out_path}")
    click.echo(json.dumps(info, indent=2, default=str))


@main.command("multi-bench")
@click.option("--warmup", type=int, default=2, show_default=True)
@click.option("--iters", type=int, default=20, show_default=True)
@click.option(
    "--bench-batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for the latency measurement.",
)
@click.option(
    "--measure-accuracy/--no-measure-accuracy",
    default=False,
    show_default=True,
    help=(
        "If set, run CIFAR-10 top-1 accuracy on small_cnn cells. Requires "
        "the dataset to be present in ./data and the baseline weights in "
        "artifacts/weights/fp32_baseline.pt."
    ),
)
@click.option(
    "--accuracy-subset",
    type=int,
    default=None,
    help="Use only the first N test images for accuracy (faster, lower fidelity).",
)
def multi_bench(
    warmup: int,
    iters: int,
    bench_batch_size: int,
    measure_accuracy: bool,
    accuracy_subset: int | None,
) -> None:
    """Bench every (model, quant_config) cell of the multi-model grid.

    Outputs ``artifacts/results/multi_model.json`` and
    ``artifacts/results/multi_pareto.md``. Accuracy is measured only for
    ``small_cnn`` (the only model trained on CIFAR-10); for other models
    the cell's top-1 is reported as ``n/a`` and explicitly labelled
    "not measured" in the report.
    """
    ensure_dirs()
    _set_quant_engine()

    from collections.abc import Callable

    from .models import ModelSpec

    accuracy_fn: Callable[[nn.Module], float] | None = None
    weight_loaders: dict[str, Callable[[ModelSpec], nn.Module]] | None = None
    if measure_accuracy:
        loader = get_test_loader(DATA_DIR, batch_size=128, subset_size=accuracy_subset)

        def _accuracy(model: nn.Module) -> float:
            return float(evaluate_accuracy(model, loader).top1)

        accuracy_fn = _accuracy

        # For accuracy to be meaningful on small_cnn we need the trained
        # baseline weights — bench_grid otherwise hands the configs a
        # randomly-initialised CifarCNN.
        def _load_small_cnn_trained(_spec: ModelSpec) -> nn.Module:
            return _load_baseline_model()

        weight_loaders = {"small_cnn": _load_small_cnn_trained}

    cells = bench_grid(
        bench_batch_size=bench_batch_size,
        n_warmup=warmup,
        n_measure=iters,
        accuracy_fn=accuracy_fn,
        weight_loaders=weight_loaders,
    )
    json_path, md_path = emit_multi_model_results(cells, RESULTS_DIR)
    click.echo(f"wrote {json_path}")
    click.echo(f"wrote {md_path}")


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


@main.command("cross-runtime")
@click.option(
    "--config",
    "config_names",
    type=click.Choice(CROSS_RUNTIME_CONFIGS),
    multiple=True,
    help=(
        "Restrict the comparison to the named configs (repeatable). "
        "Defaults to all four PTQ configs."
    ),
)
@click.option(
    "--calibration-n",
    type=int,
    default=128,
    show_default=True,
    help="Number of training images used to calibrate the static-INT8 ONNX models.",
)
@click.option(
    "--accuracy-subset",
    type=int,
    default=None,
    help="Use only the first N test images for accuracy (faster, lower fidelity).",
)
@click.option(
    "--warmup",
    type=int,
    default=5,
    show_default=True,
    help="Latency benchmark warmup iterations (each runtime).",
)
@click.option(
    "--iters",
    type=int,
    default=50,
    show_default=True,
    help="Latency benchmark measure iterations (each runtime).",
)
def cross_runtime(
    config_names: tuple[str, ...],
    calibration_n: int,
    accuracy_subset: int | None,
    warmup: int,
    iters: int,
) -> None:
    """Compare PyTorch quantized inference vs ONNX Runtime quantized inference.

    For each config the command exports the FP32 baseline to ONNX, then
    derives the INT8 variant from that file (dynamic via
    ``onnxruntime.quantization.quantize_dynamic``, static via
    ``quantize_static`` with real CIFAR-10 calibration). Both runtimes
    are then benched on the same test loader; the per-config rows are
    written to ``artifacts/results/cross_runtime.{json,md}``. See
    ``docs/cross_runtime.md`` for the methodology.
    """
    ensure_dirs()
    _set_quant_engine()

    configs: tuple[str, ...] = config_names or CROSS_RUNTIME_CONFIGS

    fp32 = _load_baseline_model()
    onnx_dir = WEIGHTS_DIR / "onnx"
    cal_loader = get_calibration_loader(DATA_DIR, n_images=calibration_n, batch_size=32)
    artifacts = build_onnx_artifacts(
        fp32_model=fp32,
        out_dir=onnx_dir,
        calibration_loader=cal_loader,
        configs=configs,
    )

    test_loader = get_test_loader(DATA_DIR, batch_size=128, subset_size=accuracy_subset)
    rows = []
    for name in configs:
        # PyTorch side: rebuild the quantized graph fresh each time so
        # neither runtime sees a state cached from the other's pass.
        def _builder(cfg_name: str = name) -> nn.Module:
            if cfg_name == "fp32_baseline":
                return _load_baseline_model()
            return _build_quantized_model(cfg_name, calibration_n=calibration_n)

        pt_weights = _baseline_path() if name == "fp32_baseline" else _quantized_path(name)
        pt = measure_pytorch_side(
            _builder,
            weights_path=pt_weights,
            test_loader=test_loader,
            bench_warmup=warmup,
            bench_iters=iters,
        )
        click.echo(
            f"[pt]   {name}: top1={pt.top1:.4f} p50={pt.p50_ms_b1:.2f}ms size={pt.size_kb:.0f}kb"
        )

        # ONNX side.
        onnx_top1, onnx_p50, n_onnx = measure_onnx_side(
            onnx_artifact=artifacts[name],
            test_loader=test_loader,
            bench_warmup=warmup,
            bench_iters=iters,
        )
        click.echo(
            f"[onnx] {name}: top1={onnx_top1:.4f} p50={onnx_p50:.2f}ms size={artifacts[name].size_kb:.0f}kb"
        )

        rows.append(
            assemble_row(
                config=name,
                pt=pt,
                onnx_top1=onnx_top1,
                onnx_p50_ms=onnx_p50,
                onnx_size_kb=artifacts[name].size_kb,
                n_samples_onnx=n_onnx,
            )
        )

    json_path = RESULTS_DIR / "cross_runtime.json"
    md_path = RESULTS_DIR / "cross_runtime.md"
    emit_full_results(build_cross_runtime_table(rows), json_path)
    md_path.write_text(render_cross_runtime_markdown(rows), encoding="utf-8")
    click.echo(f"wrote {json_path}")
    click.echo(f"wrote {md_path}")


if __name__ == "__main__":
    main()
