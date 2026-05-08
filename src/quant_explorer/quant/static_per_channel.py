"""Static INT8 with per-channel weight observers.

Weights get one scale + zero-point per output channel (rather than one
per tensor). This is the recommended default for static quantization in
PyTorch — the per-channel resolution typically recovers most of the
accuracy lost to per-tensor quantization at near-identical inference
cost.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.ao.quantization import (
    QConfig,
    convert,
    default_observer,
    default_per_channel_weight_observer,
    prepare,
)

from ._base import QuantConfig, register_quant_config


def _apply(model: nn.Module, calibration: Iterable[torch.Tensor] | None) -> nn.Module:
    if calibration is None:
        raise ValueError("static_int8_per_channel requires calibration data")
    model.eval()

    if hasattr(model, "fuse_modules"):
        model.fuse_modules()

    if getattr(model, "qconfig", None) is None:
        qcfg = QConfig(  # type: ignore[no-untyped-call]
            activation=default_observer,
            weight=default_per_channel_weight_observer,
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
        name="static_int8_per_channel",
        needs_calibration=True,
        apply=_apply,
        description="static INT8, per-channel weight observers, calibrated",
    )
)
