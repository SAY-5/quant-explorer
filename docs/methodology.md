# Bench methodology

## Latency

For every config, every batch size in `{1, 8, 32}`:

1. **Warmup**: 10 forward passes (default; configurable via
   `--warmup`). Discarded.
2. **Measure**: 200 forward passes (default; configurable via
   `--iters`). Each pass times `model(x)` with `time.perf_counter()`
   on a freshly drawn random input tensor (so any input-identity
   caching is not measured).
3. **Aggregate**: report p50, p95, p99, mean. Linear-interpolation
   percentiles, matching `numpy.percentile` defaults.

### Caveats

- **CPU only**. There's no GPU code path here. PyTorch quantized
  ops on CPU fall back to one of two backends: `qnnpack` on
  Apple Silicon, `fbgemm` on x86 Linux. The CI runner is x86 Linux
  and uses `fbgemm`.
- **Single-process timing**. PyTorch's CPU ops use
  `torch.get_num_threads()` threads by default, which on a multi-core
  machine is set to the core count. On a 4-core laptop you'll see
  better latency than on the 2-core CI runner. The *ratios* between
  configs are stable; the absolute milliseconds are not portable.
- **No `numactl` pinning**. For paper-grade numbers you'd pin to one
  NUMA node and one thread (`numactl --cpunodebind=0 --membind=0
  taskset -c 0 …`). The current bench doesn't, so very-small batches
  show context-switching jitter at the p99 tail.
- **No profile-guided rerun**. The bench doesn't re-run on outliers; if
  your machine is doing other work, p99 reflects that.

## Memory

Two complementary signals:

1. `psutil.Process().memory_info().rss` — process RSS sampled before
   inference and after every iteration; we keep the max. This is the
   OS-level resident memory, including PyTorch's native allocations.
   For tensor-heavy workloads it's the right number.
2. `tracemalloc` peak — Python-level allocations only. Useful for
   spotting Python overhead. For PyTorch-dominated workloads it's
   small relative to RSS.

## Size

`bytes(state_dict)` after `torch.save` to an in-memory buffer. We
measure the state_dict, not the whole module, because state_dict is
what actually ships at deploy time — module scaffolding (observer
metadata, etc.) is recipient-side.

## Accuracy

Top-1 and top-5 accuracy on the full CIFAR-10 test set (10 000 images),
plus per-class top-1. Per-class breakdown is reported because static
INT8 quantization sometimes regresses disproportionately on a single
class (when that class's activation distribution sits in the tails of
the per-tensor scale). The aggregate top-1 hides that; the per-class
table surfaces it.

Inference for accuracy uses batch size 128 and the standard CIFAR-10
eval transform (normalize only — no crop, no flip). Augmentation at
eval would be a methodology error.

## Reproducibility

`TrainConfig.seed` (default 1729) is used for `torch.manual_seed` and
`torch.cuda.manual_seed_all` before training. With a fixed seed, no
GPU, and Python 3.11 + PyTorch 2.2.2 + numpy<2, the FP32 baseline is
deterministic to within FP32 rounding across runs on the same machine.
*Across machines* (different BLAS, different CPU vector width)
non-bitwise differences are possible.

Quantization steps consume the calibration loader in a fixed order
(`shuffle=False`, `num_workers=0`), so quantized state_dicts are also
reproducible.
