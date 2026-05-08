"""Unit tests for the QAT pipeline.

These tests use synthetic data so they don't require CIFAR-10 to be
present. End-to-end QAT on real data is exercised by the CLI smoke
job in CI.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from quant_explorer.model import CifarCNN
from quant_explorer.quant import REGISTRY, get_config
from quant_explorer.quant.qat import build_qat_for_eval, run_qat_finetune
from quant_explorer.settings import select_quantization_engine


def setup_module(_module: object) -> None:
    torch.backends.quantized.engine = select_quantization_engine()


def test_qat_is_registered() -> None:
    assert "qat_int8" in REGISTRY
    cfg = get_config("qat_int8")
    assert cfg.needs_calibration is False
    assert "QAT" in cfg.description.upper() or "quantization-aware" in cfg.description.lower()


def test_qat_apply_raises_with_clear_message() -> None:
    """``QuantConfig.apply`` for QAT must reject the calibration-shaped API."""
    import pytest

    cfg = get_config("qat_int8")
    with pytest.raises(NotImplementedError, match="run_qat_finetune"):
        cfg.apply(CifarCNN(), None)


def _save_dummy_baseline(path: Path) -> None:
    model = CifarCNN(num_classes=10, quantizable=True)
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def test_run_qat_finetune_produces_loadable_state_dict(tmp_path: Path) -> None:
    """End-to-end QAT smoke on synthetic data: prepare, fine-tune one
    pass, convert, save. Then re-load and check forward pass works."""
    baseline = tmp_path / "fp32.pt"
    qat_out = tmp_path / "qat.pt"
    _save_dummy_baseline(baseline)

    g = torch.Generator().manual_seed(0)
    images = torch.randn(32, 3, 32, 32, generator=g)
    labels = torch.randint(0, 10, (32,), generator=g)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    info = run_qat_finetune(
        baseline_path=baseline,
        out_path=qat_out,
        train_loader=loader,
        epochs=1,
        lr=1e-4,
        log_every=0,
    )
    assert info["epochs"] == 1
    assert info["n_batches"] == 4  # 32 samples / batch 8 = 4 batches
    assert qat_out.exists()
    assert qat_out.stat().st_size > 0

    # Re-construct the converted graph and load the saved state.
    reloaded = build_qat_for_eval(baseline_path=baseline)
    reloaded.load_state_dict(torch.load(qat_out, map_location="cpu"))
    reloaded.eval()
    y = reloaded(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 10)


def test_build_qat_for_eval_works_without_qat_state(tmp_path: Path) -> None:
    """``build_qat_for_eval`` should produce a valid (if untrained) INT8
    graph from just the FP32 baseline weights."""
    baseline = tmp_path / "fp32.pt"
    _save_dummy_baseline(baseline)
    m = build_qat_for_eval(baseline_path=baseline)
    m.eval()
    y = m(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)
