#!/usr/bin/env bash
# End-to-end: run the full pipeline locally and assert pareto.md is produced.
# This is *not* run in CI (too slow); it's the script you run on your laptop.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ ! -f .venv/bin/quant-explorer ]; then
    echo "venv missing; run 'make install' first" >&2
    exit 1
fi

echo "=== train ==="
.venv/bin/quant-explorer train --epochs 5

echo "=== quantize all ==="
for cfg in dynamic_int8 static_int8_per_tensor static_int8_per_channel; do
    .venv/bin/quant-explorer quantize --config "$cfg"
done

echo "=== bench all ==="
for cfg in fp32_baseline dynamic_int8 static_int8_per_tensor static_int8_per_channel; do
    .venv/bin/quant-explorer bench --config "$cfg"
done

echo "=== evaluate all ==="
for cfg in fp32_baseline dynamic_int8 static_int8_per_tensor static_int8_per_channel; do
    .venv/bin/quant-explorer evaluate --config "$cfg"
done

echo "=== report ==="
.venv/bin/quant-explorer report

echo "=== pareto ==="
cat artifacts/results/pareto.md
