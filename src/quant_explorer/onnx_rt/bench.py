"""ONNX Runtime CPU inference + latency + accuracy helpers.

The latency methodology mirrors ``quant_explorer.bench.latency``: warmup
iterations are discarded, measure iterations are timed via
``time.perf_counter``, and p50/p95/p99 are reported. A fresh input
tensor is created per iteration to keep the comparison apples-to-apples
with the PyTorch benchmark.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..bench.latency import LatencyResult, percentile


def _make_session(model_path: str, intra_op_threads: int | None = None) -> ort.InferenceSession:
    """Build a deterministic ORT inference session on the CPU EP."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is not None:
        opts.intra_op_num_threads = intra_op_threads
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def bench_onnx_latency(
    model_path: str,
    *,
    batch_size: int,
    input_shape: tuple[int, int, int] = (3, 32, 32),
    n_warmup: int = 10,
    n_measure: int = 200,
    input_name: str = "input",
    seed: int = 0,
    intra_op_threads: int | None = None,
) -> LatencyResult:
    """Time ``InferenceSession.run`` on ``n_measure`` random inputs."""
    if n_measure < 1:
        raise ValueError("n_measure must be >= 1")
    rng = np.random.default_rng(seed)
    c, h, w = input_shape
    sess = _make_session(model_path, intra_op_threads=intra_op_threads)

    for _ in range(n_warmup):
        x = rng.standard_normal((batch_size, c, h, w)).astype(np.float32)
        sess.run(None, {input_name: x})

    samples_ms: list[float] = []
    for _ in range(n_measure):
        x = rng.standard_normal((batch_size, c, h, w)).astype(np.float32)
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
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


def onnx_top1_accuracy(
    model_path: str,
    batches: Iterable[tuple[NDArray[np.float32], NDArray[np.int64]]],
    *,
    input_name: str = "input",
    intra_op_threads: int | None = None,
) -> tuple[float, int]:
    """Compute top-1 accuracy under ORT CPU EP.

    Returns ``(top1, n_samples)``. ``batches`` yields ``(images, labels)``
    pairs where ``images`` is float32 and ``labels`` is int64.
    """
    sess = _make_session(model_path, intra_op_threads=intra_op_threads)
    correct = 0
    total = 0
    for images, labels in batches:
        if images.dtype != np.float32:
            images = images.astype(np.float32)
        logits = sess.run(None, {input_name: images})[0]
        preds = logits.argmax(axis=1)
        correct += int((preds == labels).sum())
        total += int(labels.shape[0])
    if total == 0:
        return 0.0, 0
    return correct / total, total
