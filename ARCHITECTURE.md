# Architecture

## Pipeline

```
CIFAR-10 (50k train / 10k test)
   |
   v
[ train.py ]  SGD + cosine LR, 5 epochs, batch 128
   |
   v
fp32_baseline.pt  ---+ committed under artifacts/weights/
                     |
                     +--> [ dynamic_int8       ]  -> dynamic_int8.pt
                     +--> [ static_int8_per_tensor (calib) ] -> static_int8_per_tensor.pt
                     +--> [ static_int8_per_channel (calib) ] -> static_int8_per_channel.pt
                                                  |
                                                  v
                                       [ bench/ ]  latency, memory, size
                                       [ eval/ ]   top-1 / top-5 / per-class
                                                  |
                                                  v
                                       [ report/pareto.py ] full_results.json + pareto.md
```

## The four configurations

See [`docs/quantization.md`](docs/quantization.md) for the per-config
recipe. Summary of what each does at the numerical level:

### `fp32_baseline`
No-op. The trained CNN at IEEE 754 single-precision. Reference for every
ratio in the report.

### `dynamic_int8`
`torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=qint8)`.
Weights of `nn.Linear` modules are pre-quantized to INT8; activations are
quantized at runtime per call (cheap, but adds a per-call overhead).
Conv layers are untouched, so for this CNN only the final classifier is
changed. No calibration needed.

### `static_int8_per_tensor`
Full-graph static PTQ. Conv-BN-ReLU triples are fused. `QuantStub` /
`DeQuantStub` mark the quantization boundary. `default_observer`
(MinMaxObserver) records activation ranges over a 256-image
unaugmented calibration loader; `default_weight_observer` records
per-tensor weight ranges. After `convert`, all conv kernels are
replaced with `quantized::conv2d_relu` and similar. One scale + one
zero-point per weight tensor.

### `static_int8_per_channel`
Same as above except weights use `default_per_channel_weight_observer`
— one scale + one zero-point *per output channel* of each conv weight
tensor. Outliers in one output channel no longer pollute the scale of
the rest. Activations remain per-tensor (PyTorch CPU kernels expect
per-tensor activation quantization).

## Why static needs calibration but dynamic doesn't

Dynamic quantization computes activation scales at inference time from
the actual input tensor — it sees the real data so it doesn't need a
preview. The cost is that it does this work on every call.

Static quantization bakes the activation scales into the graph at
convert time so the per-call cost is zero. The price is that the scales
have to be set *before* the model sees real data, and they're set by
running the model on a representative calibration set first. The
calibration set is usually a small slice of unaugmented training
images (256 by default here).

If the calibration set is too small or unrepresentative, the activation
scales miss the actual dynamic range — outlier inputs at inference time
will saturate to the INT8 max/min, and accuracy will drop. 256 is
empirically fine for a small CNN on CIFAR-10; larger calibration sets
help marginally.

## Per-channel vs per-tensor

A weight tensor for `Conv2d(in=64, out=128, k=3)` has 128 output
channels — each output channel has its own filter. Per-tensor
quantization picks one scale for all 128 filters; per-channel picks
128 scales, one each.

Per-channel matters when the filter magnitudes differ a lot. A common
case: deeper-network conv layers where some output channels are doing
"detection" (high-magnitude weights for important features) and others
are doing "background filtering" (small weights). Per-tensor scaling
to the loud channels' range gives the quiet channels essentially
0-bit precision; per-channel preserves both.

The runtime cost difference is small: per-channel needs a vector of
scales rather than a scalar, but the kernel is otherwise identical and
both are well-optimized in `fbgemm` and `qnnpack`.

## Bench discipline

Detailed in [`docs/methodology.md`](docs/methodology.md). The short
version:

- 10 warmup forward passes, 200 measured forward passes per
  (config, batch_size). Discard warmup.
- Linear-interpolation percentiles (numpy default). Report p50, p95,
  p99.
- Memory: max-RSS sampling between baseline and per-iteration,
  plus tracemalloc peak.
- Size: `bytes(state_dict)`, not whole-module size.

## Pareto frontier

Each config is a point in 3D `(size, latency, accuracy)` space. A point
is on the Pareto frontier if no other point dominates it on all three
axes. "Dominate" means: strictly better on at least one axis, and not
worse on any other axis.

The frontier algorithm is a straightforward O(n^2) double-loop in
[`src/quant_explorer/report/pareto.py`](src/quant_explorer/report/pareto.py),
fine for n=4 and tested in
[`tests/unit/test_pareto.py`](tests/unit/test_pareto.py).

## What's deliberately not here

- **No QAT (quantization-aware training).** QAT inserts fake-quant
  ops during fine-tuning so weights adapt to the rounding. It usually
  beats PTQ on accuracy but takes another full training run. PTQ is
  this project's scope.
- **No INT4 / INT2.** PyTorch CPU has limited support for sub-INT8
  weights at the kernel level. They're available via
  `torch.ao.quantization.fx` and bitsandbytes for GPU, neither of
  which fits the CPU-only scope.
- **No GPU quantization.** PyTorch's quantization APIs target CPU
  inference; GPU PTQ is a different stack (TensorRT, FasterTransformer,
  ONNX-Runtime CUDA).
- **No transformer / NLP models.** The bench is shaped for a CNN.
  Transformers want very different quantization (key/value caches,
  attention scaling, GroupNorm vs BatchNorm) and the kernel-level
  tradeoffs are different enough that pretending they're the same study
  would be misleading.
