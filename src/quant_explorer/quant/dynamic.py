"""Dynamic INT8 quantization over ``nn.Linear`` modules.

Dynamic quantization stores weights as INT8 and quantizes activations on
the fly per inference call. No calibration data is required. For this CNN
only the final ``fc`` layer is quantized, so the size + speed gains are
modest. Included for comparison.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from ._base import QuantConfig, register_quant_config


def _apply(model: nn.Module, _calibration: Iterable[torch.Tensor] | None) -> nn.Module:
    model.eval()
    qm: nn.Module = torch.ao.quantization.quantize_dynamic(  # type: ignore[no-untyped-call]
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return qm


CONFIG = register_quant_config(
    QuantConfig(
        name="dynamic_int8",
        needs_calibration=False,
        apply=_apply,
        description="dynamic INT8 over nn.Linear (weights INT8, activations quantized at runtime)",
    )
)
