"""Export the FP32 baseline to ONNX.

The FP32 export uses ``torch.onnx.export`` with a dynamic batch axis so
the same model can be benched at any batch size. ONNX Runtime's static
INT8 path then re-uses this FP32 file as its input, so the file is the
single source of truth for the cross-runtime comparison.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_fp32_onnx(
    model: nn.Module,
    out_path: Path,
    *,
    input_shape: tuple[int, int, int] = (3, 32, 32),
    opset: int = 13,
    input_name: str = "input",
    output_name: str = "logits",
) -> Path:
    """Export ``model`` to ONNX with a dynamic batch axis.

    The model is expected to be in eval mode; we set it here defensively.
    A single example tensor of shape ``(1, *input_shape)`` is used to
    trace the graph; the batch dimension is then marked dynamic so the
    exported file can be run at any batch size at inference time.
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        example,
        str(out_path),
        opset_version=opset,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        do_constant_folding=True,
    )
    return out_path
