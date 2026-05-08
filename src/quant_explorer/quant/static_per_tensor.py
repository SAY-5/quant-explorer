"""Static INT8 quantization with per-tensor weight observers.

The full graph is quantized: weights and activations are mapped to INT8
using one scale + zero-point per tensor. Calibration is required to fit
activation observers to representative inputs (256 unaugmented training
images by default).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.ao.quantization import (
    QConfig,
    convert,
    default_observer,
    default_weight_observer,
    prepare,
)

from ._base import QuantConfig, register_quant_config


def _apply(model: nn.Module, calibration: Iterable[torch.Tensor] | None) -> nn.Module:
    if calibration is None:
        raise ValueError("static_int8_per_tensor requires calibration data")
    model.eval()

    # Fuse Conv-BN-ReLU triples in place. Required so quantization sees
    # the fused operator (ConvReLU2d) instead of unfused BN+ReLU.
    if hasattr(model, "fuse_modules"):
        model.fuse_modules()

    # Per-tensor weight + activation observers. If the model already has a
    # qconfig set (some torchvision quantizable models prefer
    # engine-specific configs), respect it; otherwise use the per-tensor
    # default.
    if getattr(model, "qconfig", None) is None:
        qcfg = QConfig(  # type: ignore[no-untyped-call]
            activation=default_observer,
            weight=default_weight_observer,
        )
        model.qconfig = qcfg  # type: ignore[assignment]
    prepared = prepare(model, inplace=False)  # type: ignore[no-untyped-call]

    n_batches = 0
    with torch.no_grad():
        for batch in calibration:
            prepared(batch)
            n_batches += 1
    if n_batches == 0:
        raise ValueError("calibration iterator yielded zero batches")

    converted: nn.Module = convert(prepared, inplace=False)  # type: ignore[no-untyped-call]
    return converted


CONFIG = register_quant_config(
    QuantConfig(
        name="static_int8_per_tensor",
        needs_calibration=True,
        apply=_apply,
        description="static INT8, per-tensor weight + activation observers, calibrated",
    )
)
