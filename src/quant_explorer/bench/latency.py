"""Latency benchmark: warmup + measured iterations + percentile reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatencyResult:
    batch_size: int
    n_warmup: int
    n_measure: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "batch_size": self.batch_size,
            "n_warmup": self.n_warmup,
            "n_measure": self.n_measure,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
        }


def percentile(samples_ms: list[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default).

    ``q`` is in [0, 100]. The samples list does not need to be sorted.
    """
    if not samples_ms:
        raise ValueError("samples_ms is empty")
    if q < 0.0 or q > 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    sorted_samples = sorted(samples_ms)
    n = len(sorted_samples)
    if n == 1:
        return sorted_samples[0]
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac


def benchmark_latency(
    model: nn.Module,
    *,
    batch_size: int,
    input_shape: tuple[int, int, int] = (3, 32, 32),
    n_warmup: int = 10,
    n_measure: int = 200,
    device: torch.device | None = None,
    seed: int = 0,
) -> LatencyResult:
    """Time ``model(x)`` with ``batch_size`` inputs over ``n_measure`` runs.

    A fresh tensor is timed each iteration so that any caching the model
    does on input identity is not measured.
    """
    if n_measure < 1:
        raise ValueError("n_measure must be >= 1")
    device = device or torch.device("cpu")
    model.eval()

    g = torch.Generator(device=device).manual_seed(seed)
    c, h, w = input_shape

    with torch.no_grad():
        for _ in range(n_warmup):
            x = torch.randn(batch_size, c, h, w, generator=g, device=device)
            model(x)

    samples_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_measure):
            x = torch.randn(batch_size, c, h, w, generator=g, device=device)
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1000.0)

    return LatencyResult(
        batch_size=batch_size,
        n_warmup=n_warmup,
        n_measure=n_measure,
        p50_ms=percentile(samples_ms, 50.0),
        p95_ms=percentile(samples_ms, 95.0),
        p99_ms=percentile(samples_ms, 99.0),
        mean_ms=sum(samples_ms) / len(samples_ms),
    )
