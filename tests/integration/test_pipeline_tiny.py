"""Tiny end-to-end pipeline. Gated on RUN_INTEGRATION=1."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REQUIRES_INT = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run integration tests",
)


@REQUIRES_INT
def test_tiny_pipeline_produces_all_artifacts() -> None:
    """Run ``quant-explorer pipeline --tiny`` from the repo root.

    The CLI uses paths relative to the package's ``settings.py``, not the
    cwd, so this test runs in-place. We just assert artifacts exist after.
    """
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_results = repo_root / "artifacts" / "results"
    artifacts_weights = repo_root / "artifacts" / "weights"

    # Run the pipeline via the console-script entrypoint.
    subprocess.run(
        ["quant-explorer", "pipeline", "--tiny"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        timeout=600,
    )

    # All four configs should have a per-config results JSON.
    for cfg in (
        "fp32_baseline",
        "dynamic_int8",
        "static_int8_per_tensor",
        "static_int8_per_channel",
    ):
        result_path = artifacts_results / f"{cfg}.json"
        assert result_path.exists(), f"missing {result_path}"
        result = json.loads(result_path.read_text())
        assert "size" in result
        assert "latency" in result and len(result["latency"]) > 0
        assert "memory" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"]["top1"] <= 1.0

    # Aggregate report exists.
    assert (artifacts_results / "full_results.json").exists()
    assert (artifacts_results / "pareto.md").exists()

    # Quantized weights for non-baseline configs.
    for cfg in ("dynamic_int8", "static_int8_per_tensor", "static_int8_per_channel"):
        assert (artifacts_weights / f"{cfg}.pt").exists(), f"missing weights for {cfg}"
