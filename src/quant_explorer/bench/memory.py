"""Memory bench: process RSS delta + tracemalloc peak.

RSS captures the OS-level resident set size — what the system sees as
process memory, including PyTorch's own native allocations. Tracemalloc
captures only Python-level allocations and is reported as a complementary
signal; for tensor-heavy workloads the RSS delta is the more meaningful
number, but tracemalloc rules out runaway Python overhead.
"""

from __future__ import annotations

import gc
import tracemalloc
from dataclasses import dataclass

import psutil
import torch
from torch import nn


@dataclass(frozen=True)
class MemoryResult:
    rss_baseline_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    tracemalloc_peak_mb: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rss_baseline_mb": self.rss_baseline_mb,
            "rss_peak_mb": self.rss_peak_mb,
            "rss_delta_mb": self.rss_delta_mb,
            "tracemalloc_peak_mb": self.tracemalloc_peak_mb,
        }


def benchmark_memory(
    model: nn.Module,
    *,
    batch_size: int,
    input_shape: tuple[int, int, int] = (3, 32, 32),
    n_iters: int = 32,
    seed: int = 0,
) -> MemoryResult:
    model.eval()
    proc = psutil.Process()

    gc.collect()
    rss_baseline = proc.memory_info().rss
    rss_peak = rss_baseline

    tracemalloc.start()
    g = torch.Generator(device="cpu").manual_seed(seed)
    c, h, w = input_shape

    with torch.no_grad():
        for _ in range(n_iters):
            x = torch.randn(batch_size, c, h, w, generator=g)
            model(x)
            current_rss = proc.memory_info().rss
            if current_rss > rss_peak:
                rss_peak = current_rss

    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mb = 1024.0 * 1024.0
    return MemoryResult(
        rss_baseline_mb=rss_baseline / mb,
        rss_peak_mb=rss_peak / mb,
        rss_delta_mb=(rss_peak - rss_baseline) / mb,
        tracemalloc_peak_mb=tm_peak / mb,
    )
