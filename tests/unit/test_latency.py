"""Latency percentile math + bench harness sanity."""

from __future__ import annotations

import math

import pytest
from torch import nn

from quant_explorer.bench.latency import benchmark_latency, percentile


def test_percentile_p50_of_sorted_odd_list() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == pytest.approx(3.0)


def test_percentile_p50_of_even_list_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)


def test_percentile_p100_is_max() -> None:
    assert percentile([3.0, 1.0, 2.0], 100.0) == pytest.approx(3.0)


def test_percentile_p0_is_min() -> None:
    assert percentile([3.0, 1.0, 2.0], 0.0) == pytest.approx(1.0)


def test_percentile_p99_close_to_max() -> None:
    samples = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(samples, 99.0) == pytest.approx(99.01)


def test_percentile_unsorted_input_is_handled() -> None:
    samples = [5.0, 2.0, 8.0, 1.0, 3.0]
    assert percentile(samples, 50.0) == pytest.approx(3.0)


def test_percentile_singleton_returns_value() -> None:
    assert percentile([4.2], 50.0) == pytest.approx(4.2)
    assert percentile([4.2], 99.0) == pytest.approx(4.2)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError):
        percentile([], 50.0)


def test_percentile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101.0)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -0.1)


def test_benchmark_latency_returns_well_formed_result() -> None:
    """Run the bench against a trivial net so we're not dependent on
    quantization or training. Just check the harness produces sane numbers.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10)
    )
    r = benchmark_latency(model, batch_size=2, n_warmup=2, n_measure=8)
    assert r.batch_size == 2
    assert r.n_measure == 8
    assert math.isfinite(r.p50_ms) and r.p50_ms > 0.0
    assert r.p95_ms >= r.p50_ms
    assert r.p99_ms >= r.p95_ms


def test_benchmark_latency_rejects_zero_iters() -> None:
    model = nn.Linear(3, 1)
    with pytest.raises(ValueError):
        benchmark_latency(model, batch_size=1, n_warmup=0, n_measure=0)
