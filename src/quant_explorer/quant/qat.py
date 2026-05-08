"""Quantization-aware training (QAT) variant.

Where PTQ (post-training quantization) calibrates a fixed FP32 model
once and then converts, QAT inserts fake-quant ops *during* fine-tuning
so the optimiser can compensate for quantization noise. On well-behaved
small models the gap is tiny (fractions of a percentage point); on
larger models or aggressive quant configs (e.g. INT4) QAT routinely
recovers several points of top-1.

The pipeline here is deliberately minimal:

  1. Load the FP32 baseline weights into a fresh ``CifarCNN``.
  2. ``fuse_modules()`` for Conv-BN-ReLU triples.
  3. Set the qconfig and call ``prepare_qat`` (inserts fake-quant ops
     and freezes batch-norm statistics after a few hundred batches).
  4. Run 1 epoch of fine-tuning at a small learning rate.
  5. ``eval()`` + ``convert()`` to a real INT8 graph.
  6. Save the converted state_dict to ``artifacts/weights/qat_int8.pt``.

Honest reporting note: on this small CNN the static-INT8 per-channel
PTQ already costs only ~0.3 percentage points of top-1; QAT may or may
not improve on that. The CLI command writes the actual measured delta
to ``artifacts/results/qat_int8.json`` so the README reflects the run,
not a hand-set number.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.ao.quantization import (
    convert,
    get_default_qat_qconfig,
    prepare_qat,
)
from torch.utils.data import DataLoader

from ..model import CifarCNN
from ._base import QuantConfig, register_quant_config


def _apply_unsupported(_model: nn.Module, _calibration: Iterable[torch.Tensor] | None) -> nn.Module:
    """Placeholder ``apply`` for the registry.

    QAT needs gradient updates, not just calibration, so it doesn't fit
    the existing ``apply(model, calibration)`` shape. The CLI calls
    ``run_qat_finetune`` directly. The registry entry exists so listing
    configs surfaces it alongside the PTQ variants.
    """
    raise NotImplementedError(
        "QAT requires fine-tuning data and an optimiser; call "
        "`quant_explorer.quant.qat.run_qat_finetune` instead of `apply`."
    )


CONFIG = register_quant_config(
    QuantConfig(
        name="qat_int8",
        needs_calibration=False,  # QAT uses training data, not calibration
        apply=_apply_unsupported,
        description=(
            "quantization-aware training (1 epoch fine-tune); per-channel "
            "weights, fake-quant ops in the forward pass during fine-tune"
        ),
    )
)


def _build_qat_model(baseline_state: dict[str, torch.Tensor], engine: str) -> nn.Module:
    """Reload baseline weights, fuse, then prepare for QAT."""
    model = CifarCNN(num_classes=10, quantizable=True)
    model.load_state_dict(baseline_state)
    # ``prepare_qat`` requires train mode (BN must update during the
    # initial part of fine-tuning). Fusion in ``CifarCNN`` walks
    # ``fuse_conv_bn_eval`` though, which asserts ``not training``;
    # fuse first under eval, then switch back to train.
    model.eval()
    model.fuse_modules()
    model.train()
    model.qconfig = get_default_qat_qconfig(engine)  # type: ignore[no-untyped-call]
    prepared: nn.Module = prepare_qat(model, inplace=False)  # type: ignore[no-untyped-call]
    return prepared


def run_qat_finetune(
    *,
    baseline_path: Path,
    out_path: Path,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    epochs: int = 1,
    lr: float = 1e-4,
    momentum: float = 0.9,
    engine: str | None = None,
    log_every: int = 50,
) -> dict[str, Any]:
    """Run QAT fine-tuning and save the converted INT8 state_dict.

    Returns a small info dict (epochs, train_loss, wall_seconds, n_batches).
    """
    if engine is None:
        engine = torch.backends.quantized.engine
    state = torch.load(baseline_path, map_location="cpu")
    prepared = _build_qat_model(state, engine)

    optimizer = torch.optim.SGD(
        prepared.parameters(),
        lr=lr,
        momentum=momentum,
    )
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    info: dict[str, Any] = {"epochs": epochs, "lr": lr, "engine": engine}
    n_batches_total = 0
    final_loss = 0.0
    for epoch in range(epochs):
        running = 0.0
        n_b = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = prepared(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_b += 1
            n_batches_total += 1
            if log_every and n_b % log_every == 0:
                print(f"qat epoch {epoch + 1} batch {n_b} loss={running / n_b:.4f}")
        final_loss = running / max(n_b, 1)

    info["train_loss"] = final_loss
    info["n_batches"] = n_batches_total

    # Switch to eval and convert. ``convert`` rewrites the graph to use
    # quantized ops with the per-channel int8 weights baked in.
    prepared.eval()
    converted = convert(prepared.cpu(), inplace=False)  # type: ignore[no-untyped-call]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted.state_dict(), out_path)
    info["wall_seconds"] = time.perf_counter() - start
    info["out_path"] = str(out_path)
    return info


def build_qat_for_eval(*, baseline_path: Path, engine: str | None = None) -> nn.Module:
    """Construct a converted QAT model for benching / evaluation.

    The weights file contains a *converted* state_dict (post-``convert``).
    To load it we have to repeat the prepare-then-convert structural
    transformation on a fresh ``CifarCNN`` so the keys line up.
    """
    if engine is None:
        engine = torch.backends.quantized.engine
    base_state = torch.load(baseline_path, map_location="cpu")
    prepared = _build_qat_model(base_state, engine)
    prepared.eval()
    converted: nn.Module = convert(prepared, inplace=False)  # type: ignore[no-untyped-call]
    return converted
