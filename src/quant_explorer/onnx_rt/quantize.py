"""ONNX-side quantization helpers (dynamic + static INT8).

The static path mirrors PyTorch's static INT8 PTQ (per-tensor and
per-channel weight observers, real calibration data). The dynamic path
restricts quantization to ``MatMul``/``Gemm`` operators to mirror
PyTorch's dynamic INT8 PTQ, which only quantizes ``nn.Linear``. ONNX
Runtime CPU EP doesn't ship a kernel for ``ConvInteger`` (the op
``quantize_dynamic`` emits for convolutions by default), so quantizing
the full graph dynamically would produce a model that can't run; the
restriction keeps the comparison meaningful and runnable.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class _IterDataReader(CalibrationDataReader):  # type: ignore[misc]
    """Adapter: NumPy batches -> ORT calibration data reader."""

    def __init__(self, batches: Iterable[NDArray[np.float32]], input_name: str) -> None:
        self._iter = iter(batches)
        self._input_name = input_name

    def get_next(self) -> dict[str, NDArray[np.float32]] | None:
        try:
            arr = next(self._iter)
        except StopIteration:
            return None
        return {self._input_name: arr}

    def rewind(self) -> None:
        # The base class doesn't require this, but some quantize_* paths
        # call it; we don't support re-iteration here (the caller passes
        # a fresh iterator per quantize call).
        return None

    def __iter__(self) -> Any:  # pragma: no cover - parent compatibility
        return self


def _preprocess(fp32_path: Path) -> Path:
    """Run ORT shape inference + symbolic-shape preprocess.

    ``quantize_static`` and ``quantize_dynamic`` both recommend this step
    (they log a warning when skipped). The preprocessed file is written
    next to the input with a ``.preproc.onnx`` suffix.
    """
    pp_path = fp32_path.with_suffix(".preproc.onnx")
    quant_pre_process(str(fp32_path), str(pp_path))
    return pp_path


def quantize_dynamic_int8_onnx(
    fp32_path: Path,
    out_path: Path,
    *,
    op_types_to_quantize: tuple[str, ...] = ("MatMul", "Gemm"),
) -> Path:
    """Apply ORT dynamic INT8 quantization, restricted to linear ops.

    Mirrors PyTorch's ``quantize_dynamic`` (which quantizes only
    ``nn.Linear``); see the module docstring for why convolutions are
    excluded.
    """
    pp = _preprocess(fp32_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(pp),
        str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=list(op_types_to_quantize),
    )
    return out_path


def quantize_static_int8_onnx(
    fp32_path: Path,
    out_path: Path,
    *,
    calibration_batches: Iterable[NDArray[np.float32]],
    input_name: str = "input",
    per_channel: bool,
) -> Path:
    """Apply ORT static INT8 quantization (QDQ format).

    ``per_channel=False`` corresponds to PyTorch's
    ``static_int8_per_tensor`` config; ``True`` corresponds to
    ``static_int8_per_channel``. Activations are always per-tensor INT8
    (ORT QDQ doesn't support per-channel activations, matching PT).
    """
    pp = _preprocess(fp32_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = _IterDataReader(calibration_batches, input_name=input_name)
    quantize_static(
        str(pp),
        str(out_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    return out_path
