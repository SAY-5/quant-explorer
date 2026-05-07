"""Memory bench harness sanity."""

from __future__ import annotations

from torch import nn

from quant_explorer.bench.memory import benchmark_memory


def test_benchmark_memory_returns_well_formed_result() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10)
    )
    r = benchmark_memory(model, batch_size=2, n_iters=4)
    assert r.rss_baseline_mb > 0.0
    assert r.rss_peak_mb >= r.rss_baseline_mb
    assert r.rss_delta_mb == r.rss_peak_mb - r.rss_baseline_mb
    assert r.tracemalloc_peak_mb >= 0.0
