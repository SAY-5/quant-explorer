"""Bench-regress gate.

Reads ``artifacts/results/multi_model.json`` and asserts a small set of
*structural* invariants that quantization must always satisfy on a
healthy build:

* INT8 size shrink: every static-INT8 cell must be <= 50% of its model's
  FP32 size. Failing this means the converter silently fell back to FP32
  weights for some layers.
* No NaN / negative latency: the percentile reporter should never emit a
  non-positive number.
* Each (model, quant_config) cell from the registered grid is present.

We deliberately do *not* gate on absolute latency — that's noisy on
shared CI runners. We gate on the structural property (INT8 size
shrink), which is repeatable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_MODELS = ("small_cnn", "mobilenet_v3", "vgg11_bn")
EXPECTED_QUANT_CONFIGS = (
    "fp32_baseline",
    "dynamic_int8",
    "static_int8_per_tensor",
    "static_int8_per_channel",
)
SIZE_SHRINK_MAX_RATIO = 0.50  # static INT8 must be <= half of FP32


def main(json_path: str = "artifacts/results/multi_model.json") -> int:
    p = Path(json_path)
    if not p.exists():
        print(f"ERROR: {p} does not exist", file=sys.stderr)
        return 2
    payload = json.loads(p.read_text())
    cells = payload["cells"]

    failures: list[str] = []

    # Index by (model, quant_config) for property checks.
    by_key = {(c["model"], c["quant_config"]): c for c in cells}

    # Coverage of the grid.
    for m in EXPECTED_MODELS:
        for q in EXPECTED_QUANT_CONFIGS:
            if (m, q) not in by_key:
                failures.append(f"missing cell: ({m}, {q})")

    # Structural invariants.
    for m in EXPECTED_MODELS:
        fp32 = by_key.get((m, "fp32_baseline"))
        if fp32 is None:
            continue
        base_size = fp32["size_kb"]
        for static_q in ("static_int8_per_tensor", "static_int8_per_channel"):
            cell = by_key.get((m, static_q))
            if cell is None:
                continue
            ratio = cell["size_kb"] / base_size if base_size > 0 else 1.0
            if ratio > SIZE_SHRINK_MAX_RATIO:
                failures.append(
                    f"{m}/{static_q} size_ratio={ratio:.2f}x of FP32, "
                    f"expected <= {SIZE_SHRINK_MAX_RATIO:.2f}x"
                )

    # Latency sanity.
    for c in cells:
        p50 = c["latency"]["p50_ms"]
        if not (p50 > 0):
            failures.append(f"{c['model']}/{c['quant_config']} non-positive p50_ms={p50}")

    if failures:
        print("BENCH-REGRESS FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"bench-regress OK: {len(cells)} cells passed structural checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
