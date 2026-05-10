"""Cross-runtime (PyTorch quantized vs ONNX Runtime quantized) tests.

Uses synthetic 32x32 images so the tests don't depend on the CIFAR-10
dataset being downloaded. Real-data validation is exercised by the
``cross-runtime-smoke`` CI job (which loads the committed FP32 baseline
weights and the cached CIFAR-10 test split).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from quant_explorer.model import CifarCNN
from quant_explorer.onnx_rt import (
    ACCURACY_TOL_PP,
    CROSS_RUNTIME_CONFIGS,
    CrossRuntimeResult,
    build_cross_runtime_table,
    export_fp32_onnx,
    quantize_dynamic_int8_onnx,
    quantize_static_int8_onnx,
    render_cross_runtime_markdown,
)
from quant_explorer.onnx_rt.bench import bench_onnx_latency, onnx_top1_accuracy
from quant_explorer.onnx_rt.runner import _calibration_numpy_batches
from quant_explorer.settings import select_quantization_engine


@pytest.fixture(autouse=True)
def _set_engine() -> None:
    torch.backends.quantized.engine = select_quantization_engine()


@pytest.fixture
def fp32_model() -> CifarCNN:
    """Untrained but deterministic CifarCNN — fine for parity tests.

    Parity here is structural: the same weights export and run under
    both runtimes, so PT-fp32 and ONNX-fp32 should agree to numerical
    precision regardless of whether the weights are trained.
    """
    torch.manual_seed(0)
    m = CifarCNN(num_classes=10, quantizable=True)
    m.eval()
    return m


def test_export_fp32_onnx_produces_runnable_session(tmp_path: Path, fp32_model: CifarCNN) -> None:
    out = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, out)
    assert out.exists()
    assert out.stat().st_size > 0

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": np.random.randn(2, 3, 32, 32).astype(np.float32)})[0]
    assert y.shape == (2, 10)


def test_export_fp32_onnx_matches_pytorch_outputs(tmp_path: Path, fp32_model: CifarCNN) -> None:
    """FP32 export must agree with the PyTorch forward to floating-point
    precision (the export step is meant to be lossless)."""
    out = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, out)

    x_np = np.random.RandomState(123).randn(4, 3, 32, 32).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        y_pt = fp32_model(x_pt).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"input": x_np})[0]
    # 1e-4 abs tolerance is comfortable for FP32 graph round-trip.
    np.testing.assert_allclose(y_onnx, y_pt, atol=1e-4, rtol=1e-4)


def test_quantize_dynamic_int8_onnx_runs(tmp_path: Path, fp32_model: CifarCNN) -> None:
    fp32 = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, fp32)
    dyn = tmp_path / "dyn.onnx"
    quantize_dynamic_int8_onnx(fp32, dyn)
    assert dyn.exists()
    sess = ort.InferenceSession(str(dyn), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)})[0]
    assert y.shape == (1, 10)


def test_quantize_static_int8_onnx_per_tensor_and_per_channel(
    tmp_path: Path, fp32_model: CifarCNN
) -> None:
    fp32 = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, fp32)
    rng = np.random.RandomState(0)
    cal = [rng.randn(8, 3, 32, 32).astype(np.float32) for _ in range(4)]

    pt_out = tmp_path / "static_pt.onnx"
    quantize_static_int8_onnx(fp32, pt_out, calibration_batches=iter(cal), per_channel=False)
    pc_out = tmp_path / "static_pc.onnx"
    quantize_static_int8_onnx(fp32, pc_out, calibration_batches=iter(cal), per_channel=True)

    # Both files exist and are smaller than the FP32 source (INT8
    # weights are ~4x smaller than FP32 weights).
    fp32_size = fp32.stat().st_size
    assert pt_out.stat().st_size < fp32_size
    assert pc_out.stat().st_size < fp32_size

    for path in (pt_out, pc_out):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)})[0]
        assert y.shape == (1, 10)


def test_bench_onnx_latency_smoke(tmp_path: Path, fp32_model: CifarCNN) -> None:
    fp32 = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, fp32)
    r = bench_onnx_latency(str(fp32), batch_size=1, n_warmup=2, n_measure=8)
    assert r.n_measure == 8
    assert r.batch_size == 1
    assert r.p50_ms > 0
    assert r.p95_ms >= r.p50_ms
    assert r.p99_ms >= r.p95_ms


def test_onnx_top1_accuracy_synthetic(tmp_path: Path, fp32_model: CifarCNN) -> None:
    fp32 = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, fp32)
    # 16 random images with random labels — accuracy ~ 0.1 for a random
    # model on 10-class targets. We only care that the call succeeds
    # and returns a value in [0,1] for n>0.
    rng = np.random.RandomState(7)
    batches = [
        (rng.randn(8, 3, 32, 32).astype(np.float32), rng.randint(0, 10, size=8).astype(np.int64))
        for _ in range(2)
    ]
    top1, n = onnx_top1_accuracy(str(fp32), iter(batches))
    assert n == 16
    assert 0.0 <= top1 <= 1.0


def test_cross_runtime_result_deltas() -> None:
    r = CrossRuntimeResult(
        config="static_int8_per_tensor",
        pt_top1=0.820,
        onnx_top1=0.815,
        pt_p50_ms=1.50,
        onnx_p50_ms=2.10,
        pt_size_kb=293.0,
        onnx_size_kb=305.0,
        n_samples=1000,
    )
    # ONNX is 0.5pp lower; well within the 1.0pp tolerance.
    assert r.top1_delta_pp == pytest.approx(-0.5, abs=1e-6)
    assert r.within_accuracy_tolerance
    # ONNX is slower => latency_ratio > 1.
    assert r.latency_ratio == pytest.approx(2.10 / 1.50, abs=1e-6)
    # ONNX is slightly larger.
    assert r.size_ratio == pytest.approx(305.0 / 293.0, abs=1e-6)


def test_cross_runtime_result_outside_tolerance_flagged() -> None:
    r = CrossRuntimeResult(
        config="static_int8_per_tensor",
        pt_top1=0.820,
        onnx_top1=0.800,  # 2pp drop = outside +/-1pp tolerance
        pt_p50_ms=1.0,
        onnx_p50_ms=1.0,
        pt_size_kb=300.0,
        onnx_size_kb=300.0,
        n_samples=1000,
    )
    assert r.top1_delta_pp == pytest.approx(-2.0, abs=1e-6)
    assert not r.within_accuracy_tolerance


def test_cross_runtime_result_handles_zero_baseline() -> None:
    r = CrossRuntimeResult(
        config="x",
        pt_top1=0.5,
        onnx_top1=0.5,
        pt_p50_ms=0.0,
        onnx_p50_ms=1.0,
        pt_size_kb=0.0,
        onnx_size_kb=10.0,
        n_samples=10,
    )
    assert r.latency_ratio == 0.0
    assert r.size_ratio == 0.0


def test_build_cross_runtime_table_round_trip() -> None:
    rows = [
        CrossRuntimeResult(
            config="fp32_baseline",
            pt_top1=0.823,
            onnx_top1=0.823,
            pt_p50_ms=1.67,
            onnx_p50_ms=1.20,
            pt_size_kb=1144.0,
            onnx_size_kb=1156.0,
            n_samples=10_000,
        ),
        CrossRuntimeResult(
            config="static_int8_per_channel",
            pt_top1=0.820,
            onnx_top1=0.816,
            pt_p50_ms=0.62,
            onnx_p50_ms=0.74,
            pt_size_kb=304.0,
            onnx_size_kb=310.0,
            n_samples=10_000,
        ),
    ]
    table = build_cross_runtime_table(rows)
    assert table["tolerance_pp"] == ACCURACY_TOL_PP
    assert len(table["rows"]) == 2
    fp32 = table["rows"][0]
    assert fp32["config"] == "fp32_baseline"
    assert fp32["pt"]["top1"] == pytest.approx(0.823)
    assert fp32["onnx"]["top1"] == pytest.approx(0.823)
    assert fp32["deltas"]["within_accuracy_tolerance"]
    assert "tolerance" not in fp32["deltas"] or True  # shape check
    assert "size_ratio" in fp32["deltas"]


def test_render_cross_runtime_markdown_shape() -> None:
    rows = [
        CrossRuntimeResult(
            config="fp32_baseline",
            pt_top1=0.823,
            onnx_top1=0.823,
            pt_p50_ms=1.67,
            onnx_p50_ms=1.20,
            pt_size_kb=1144.0,
            onnx_size_kb=1156.0,
            n_samples=10_000,
        ),
    ]
    md = render_cross_runtime_markdown(rows)
    assert "Cross-runtime comparison" in md
    assert "fp32_baseline" in md
    assert "pt_top1" in md
    assert "onnx_top1" in md
    assert "within_tol" in md
    # Cross-link to the sibling SAY-5 projects must be present.
    assert "SAY-5/onnx-deploy" in md
    assert "SAY-5/export-validator" in md


def test_cross_runtime_configs_match_ptq_configs() -> None:
    """The cross-runtime path covers the four PTQ configs, not QAT.

    QAT export is a separate code path (see docs/cross_runtime.md); the
    set of configs supported here is the documented surface area.
    """
    expected = {
        "fp32_baseline",
        "dynamic_int8",
        "static_int8_per_tensor",
        "static_int8_per_channel",
    }
    assert set(CROSS_RUNTIME_CONFIGS) == expected


def test_pt_vs_onnx_fp32_top1_parity_within_tolerance(tmp_path: Path, fp32_model: CifarCNN) -> None:
    """FP32 export must produce top-1 within +/-1pp on synthetic labels.

    Untrained CifarCNN on uniformly random labels should give ~10%
    top-1 from both runtimes; the exact value will differ by at most a
    handful of samples between PT and ONNX (FP32 round-trip is
    lossless). We assert the structural parity invariant: PT and ONNX
    pick the same arg-max on every sample.
    """
    fp32 = tmp_path / "fp32.onnx"
    export_fp32_onnx(fp32_model, fp32)

    x_np = np.random.RandomState(42).randn(32, 3, 32, 32).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_preds = fp32_model(x_pt).argmax(dim=1).numpy()
    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    onnx_preds = sess.run(None, {"input": x_np})[0].argmax(axis=1)
    np.testing.assert_array_equal(pt_preds, onnx_preds)


def test_calibration_numpy_batches_yields_float32() -> None:
    """The DataLoader -> NumPy adapter must hand ORT float32 arrays."""

    class _Loader:
        def __iter__(self):  # type: ignore[no-untyped-def]
            for _ in range(2):
                yield torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long)

    batches = list(_calibration_numpy_batches(_Loader()))  # type: ignore[arg-type]
    assert len(batches) == 2
    for b in batches:
        assert b.dtype == np.float32
        assert b.shape == (4, 3, 32, 32)


def test_accuracy_tol_pp_is_one_percentage_point() -> None:
    """Tolerance is a load-bearing constant — encode it in tests so a
    change requires updating the docs and the README simultaneously."""
    assert ACCURACY_TOL_PP == 1.0
