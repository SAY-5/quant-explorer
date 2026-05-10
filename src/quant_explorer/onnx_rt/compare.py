"""Build the PyTorch vs ONNX Runtime comparison table.

For each config the comparison records: top-1 accuracy under both
runtimes (with their absolute delta in percentage points), p50 latency
at batch 1 under both runtimes (with the speedup ratio), and on-disk
size of the serialized weights. The rendered Markdown is the authoring
artifact; the JSON is the machine-readable source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# A 1pp tolerance window for accuracy parity. Static-INT8 in PyTorch
# (FBGEMM/QNNPACK eager-mode) and in ONNX Runtime (QDQ format) differ on
# small numerical details — calibrator algorithm, fold ordering, the
# exact quantization formula for activations — so exact bit-parity is
# unrealistic. 1pp is the structural-parity assertion: both backends
# should land in the same neighbourhood for a well-trained CIFAR-10 CNN.
ACCURACY_TOL_PP: float = 1.0


@dataclass(frozen=True)
class CrossRuntimeResult:
    """One row of the cross-runtime comparison table."""

    config: str
    pt_top1: float
    onnx_top1: float
    pt_p50_ms: float
    onnx_p50_ms: float
    pt_size_kb: float
    onnx_size_kb: float
    n_samples: int

    @property
    def top1_delta_pp(self) -> float:
        """ONNX minus PT, in percentage points. Positive = ONNX higher."""
        return (self.onnx_top1 - self.pt_top1) * 100.0

    @property
    def within_accuracy_tolerance(self) -> bool:
        return abs(self.top1_delta_pp) <= ACCURACY_TOL_PP

    @property
    def latency_ratio(self) -> float:
        """ONNX p50 / PT p50. <1.0 = ONNX is faster."""
        if self.pt_p50_ms <= 0.0:
            return 0.0
        return self.onnx_p50_ms / self.pt_p50_ms

    @property
    def size_ratio(self) -> float:
        """ONNX size / PT size."""
        if self.pt_size_kb <= 0.0:
            return 0.0
        return self.onnx_size_kb / self.pt_size_kb

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "n_samples": self.n_samples,
            "pt": {
                "top1": self.pt_top1,
                "p50_ms_b1": self.pt_p50_ms,
                "size_kb": self.pt_size_kb,
            },
            "onnx": {
                "top1": self.onnx_top1,
                "p50_ms_b1": self.onnx_p50_ms,
                "size_kb": self.onnx_size_kb,
            },
            "deltas": {
                "top1_pp": self.top1_delta_pp,
                "latency_ratio": self.latency_ratio,
                "size_ratio": self.size_ratio,
                "within_accuracy_tol_pp": ACCURACY_TOL_PP,
                "within_accuracy_tolerance": self.within_accuracy_tolerance,
            },
        }


def build_cross_runtime_table(rows: list[CrossRuntimeResult]) -> dict[str, Any]:
    """Pack rows into the on-disk JSON shape."""
    return {
        "tolerance_pp": ACCURACY_TOL_PP,
        "rows": [r.as_dict() for r in rows],
    }


def render_cross_runtime_markdown(rows: list[CrossRuntimeResult]) -> str:
    """Render the cross-runtime comparison as a Markdown table."""
    lines = [
        "# Cross-runtime comparison: PyTorch quantized vs ONNX Runtime quantized",
        "",
        (
            f"Top-1 accuracy parity is asserted within +/-{ACCURACY_TOL_PP:.1f}pp; "
            "static INT8 in PyTorch (eager-mode FBGEMM/QNNPACK) and ONNX Runtime "
            "(QDQ format) differ on small numerical details, so exact bit-parity "
            "is not the goal. Latency is p50 at batch 1; size is the on-disk "
            "state_dict (PT) or `.onnx` file (ONNX)."
        ),
        "",
        "| config | pt_top1 | onnx_top1 | top1_delta_pp | pt_p50_ms | onnx_p50_ms | latency_ratio | pt_size_kb | onnx_size_kb | size_ratio | within_tol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        sign = "+" if r.top1_delta_pp > 0 else ""
        within = "yes" if r.within_accuracy_tolerance else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    r.config,
                    f"{r.pt_top1 * 100:.1f}%",
                    f"{r.onnx_top1 * 100:.1f}%",
                    f"{sign}{r.top1_delta_pp:.2f}",
                    f"{r.pt_p50_ms:.2f}",
                    f"{r.onnx_p50_ms:.2f}",
                    f"{r.latency_ratio:.2f}x",
                    f"{r.pt_size_kb:.0f}",
                    f"{r.onnx_size_kb:.0f}",
                    f"{r.size_ratio:.2f}x",
                    within,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Cross-links:",
            "- `SAY-5/onnx-deploy` consumes the ONNX files produced here as its",
            "  deployment artifact (CPU EP target).",
            "- `SAY-5/export-validator` re-uses the parity assertion above as a",
            "  generic export-quality gate (top-1 within +/-1pp = pass).",
        ]
    )
    return "\n".join(lines) + "\n"
