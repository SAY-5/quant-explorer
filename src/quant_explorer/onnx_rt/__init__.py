"""Cross-runtime ONNX quantization + benchmarking.

This sub-package exports each PTQ config to ONNX (with quantization
preserved) and benchmarks inference under ONNX Runtime's CPU EP, so the
same model can be compared head-to-head against PyTorch's quantized
runtime on three axes: top-1 accuracy, latency, and on-disk size.

See ``docs/cross_runtime.md`` for the methodology and the
``cross-runtime`` CLI command for the orchestration.
"""

from .bench import bench_onnx_latency, onnx_top1_accuracy
from .compare import (
    ACCURACY_TOL_PP,
    CrossRuntimeResult,
    build_cross_runtime_table,
    render_cross_runtime_markdown,
)
from .export import export_fp32_onnx
from .quantize import quantize_dynamic_int8_onnx, quantize_static_int8_onnx
from .runner import (
    CROSS_RUNTIME_CONFIGS,
    PyTorchSideMeasurement,
    assemble_row,
    build_onnx_artifacts,
    measure_onnx_side,
    measure_pytorch_side,
)

__all__ = [
    "ACCURACY_TOL_PP",
    "CROSS_RUNTIME_CONFIGS",
    "CrossRuntimeResult",
    "PyTorchSideMeasurement",
    "assemble_row",
    "bench_onnx_latency",
    "build_cross_runtime_table",
    "build_onnx_artifacts",
    "export_fp32_onnx",
    "measure_onnx_side",
    "measure_pytorch_side",
    "onnx_top1_accuracy",
    "quantize_dynamic_int8_onnx",
    "quantize_static_int8_onnx",
    "render_cross_runtime_markdown",
]
