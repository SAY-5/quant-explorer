# quant-explorer

PyTorch post-training quantization explorer for a small CIFAR-10 CNN.
Trains an FP32 baseline, applies four quantization configurations
(none, dynamic INT8, static INT8 per-tensor, static INT8 per-channel),
and compares them on size, latency, peak memory, and top-1 / top-5
accuracy. The output is a Pareto-frontier table you can use to pick the
right tradeoff.

## What this studies

- The **cost of quantization**: how much accuracy you give up for how
  much size and latency you gain.
- **Per-tensor vs per-channel** weight quantization — the most-asked
  question in PyTorch eager-mode PTQ. Per-channel resolution typically
  recovers most of the accuracy loss at near-zero runtime cost; this
  project quantifies the gap on a real (small) network.
- **Pareto-frontier framing**: instead of picking a winner, surface
  which configs aren't dominated and let the reader choose based on
  their tolerance for accuracy loss.

## Pareto table

Below are the numbers from a real run on a 4-core Apple M-series CPU
(committed under
[`artifacts/results/pareto.md`](artifacts/results/pareto.md)):

| config | size_kb | size_ratio | p50_lat_ms_b1 | latency_speedup | top1_acc | acc_drop_pp | mem_peak_mb | pareto_optimal |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| fp32_baseline | 1144 | 1.00x | 1.67 | 1.00x | 82.3% | 0.0pp | 585.8 | no |
| dynamic_int8 | 1141 | 1.00x | 0.70 | 2.38x | 82.3% | -0.0pp | 280.2 | yes |
| static_int8_per_tensor | 293 | 0.26x | 0.67 | 2.51x | 82.1% | -0.2pp | 642.3 | yes |
| static_int8_per_channel | 304 | 0.27x | 0.62 | 2.72x | 82.0% | -0.3pp | 622.3 | yes |
| qat_int8 | 293 | 0.26x | 0.94 | 1.78x | 82.4% | +0.1pp | 222.8 | yes |

Pareto frontier picks:
- minimum size: `qat_int8` (0.26x of FP32)
- highest accuracy: `qat_int8` (top-1 82.4%, slightly above FP32)
- lowest latency: `static_int8_per_channel` (p50 0.62ms at batch 1)

What this says: **static INT8 quantization (per-channel) cuts model
size to ~27% and runs 2.7x faster than FP32 with a 0.3-percentage-point
top-1 accuracy drop**. Dynamic INT8 (which only quantizes `nn.Linear`)
still gets a meaningful 2.4x speedup with essentially zero accuracy
loss because it leaves the conv layers in FP32. Per-tensor static is
slightly smaller than per-channel and the accuracy gap is modest at
this scale; on bigger models the per-channel advantage usually grows.

**QAT (quantization-aware training)** closes the accuracy gap from
PTQ entirely on this network and even slightly exceeds the FP32
baseline (+0.07pp), at the cost of 1 epoch of fine-tuning. The
converted INT8 graph from QAT is the same size as PTQ per-tensor but
its p50 latency lands between FP32 and PTQ static — slightly slower
than PTQ on this CPU. See [QAT vs PTQ](#qat-vs-ptq) below.

Full per-config measurements (latency at batch sizes 1, 8, 32; memory;
per-class accuracy) live in
[`artifacts/results/full_results.json`](artifacts/results/full_results.json).

## Multi-model bench

The same 4 quant configs applied to two larger torchvision networks
gives a 12-cell grid (3 models x 4 configs). Latency + on-disk size are
measured for every cell; **top-1 accuracy is only measured for
`small_cnn`** because it's the only model trained on CIFAR-10 — the
torchvision models are random-init at 224x224 inputs (a different
domain). Within-model frontier picks live in
[`artifacts/results/multi_pareto.md`](artifacts/results/multi_pareto.md);
representative numbers from a recent run on a 4-core M-series CPU:

| model | quant_config | size_kb | size_ratio | p50_ms (b=1) | top1 |
|---|---|---:|---:|---:|---:|
| small_cnn | fp32_baseline | 1144 | 1.00x | 8.12 | (see single-model table) |
| small_cnn | static_int8_per_tensor | 293 | 0.26x | 1.69 | (see single-model table) |
| mobilenet_v3 | fp32_baseline | 21622 | 1.00x | 1279.5 | n/a |
| mobilenet_v3 | static_int8_per_channel | 5520 | 0.26x | 22.6 | n/a |
| vgg11_bn | fp32_baseline | 519061 | 1.00x | 81.6 | n/a |
| vgg11_bn | static_int8_per_tensor | 129800 | 0.25x | 132.6 | n/a |

Read the full 12-cell table in
[`multi_pareto.md`](artifacts/results/multi_pareto.md). Two honest
caveats with this grid:

- **VGG11 INT8 is slower than its FP32 baseline in this measurement**
  (~0.5x speedup). VGG11 has no Conv-BN-ReLU runs that *don't* fuse,
  so static-INT8 should be faster — but qnnpack on random-init weights
  produces extreme activations and triggers fallbacks, and on macOS the
  CPU GEMM kernels for INT8 large convolutions are mature on x86 but
  not on Apple Silicon. The size shrink (4x) is real and structural;
  the latency speedup isn't transferable from this measurement.
- **MobileNetV3 shows the largest INT8 speedup** (50x+) — but the
  baseline is also slow on random init because depthwise convs hit
  unoptimised paths. The INT8 speedup vs FP32 is genuine but should
  not be read as a deployment number.

The CI `multi-bench-regress` job re-runs this grid on every push and
asserts a structural invariant: **every static-INT8 cell must be
<= 50% of its model's FP32 size**. This catches regressions in the
quantization converter (e.g. a layer silently keeping FP32 weights)
without depending on noisy absolute latency numbers. See
[`scripts/bench_regress_check.py`](scripts/bench_regress_check.py).

## Quickstart

```bash
make install        # CPU PyTorch + dev tooling
make pipeline       # train + quantize + bench + evaluate + report (~10 min)
```

For a smoke run that finishes in under a minute (200 train images,
1 epoch, no useful accuracy):

```bash
make pipeline-tiny
```

You can run individual steps too:

```bash
quant-explorer train --epochs 5
quant-explorer quantize --config static_int8_per_channel
quant-explorer bench --config static_int8_per_channel
quant-explorer evaluate --config static_int8_per_channel
quant-explorer report
```

## Architecture

```
CIFAR-10 (50k train / 10k test)
    |
    v
[ train.py ]  SGD + cosine LR, 5 epochs
    |
    v
fp32_baseline.pt
    |
    +--> dynamic_int8        (no calibration)
    +--> static_int8_per_tensor   (256-image calibration)
    +--> static_int8_per_channel  (256-image calibration)
                |
                v
        bench/  latency, memory, size
        eval/   top-1 / top-5 / per-class
                |
                +--> onnx_rt/ (FP32 export + ONNX-side INT8 quantization)
                |            -> ORT CPU EP inference: top-1 + latency
                |            -> cross_runtime.{json,md}
                v
        report/ full_results.json + pareto.md
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for what each config does
numerically and why.

## Configurations

| name | what it does | needs calibration |
|---|---|:---:|
| `fp32_baseline` | reference; no quantization | — |
| `dynamic_int8` | INT8 weights for `nn.Linear`, runtime activation quantization | no |
| `static_int8_per_tensor` | full-graph INT8, one scale per weight tensor | yes |
| `static_int8_per_channel` | full-graph INT8, one scale per weight output channel | yes |

Detail: [`docs/quantization.md`](docs/quantization.md).
Bench discipline: [`docs/methodology.md`](docs/methodology.md).
Adding a new config: [`docs/README.md`](docs/README.md).

## QAT vs PTQ

[`quant_explorer.quant.qat`](src/quant_explorer/quant/qat.py) implements
quantization-aware training: prepare the model with fake-quant ops
inserted, fine-tune for one epoch at a small learning rate, then convert
to a real INT8 graph. The `quant-explorer qat-finetune` CLI command
runs this pipeline starting from `fp32_baseline.pt` and writes
`artifacts/weights/qat_int8.pt`.

Measured comparison on the full CIFAR-10 test set (10k images):

| variant | top1 | acc delta vs FP32 | size_kb | p50_b1 |
|---|---:|---:|---:|---:|
| fp32_baseline | 82.34% | 0.0pp | 1144 | 1.67ms |
| static_int8_per_channel (PTQ) | 82.00% | -0.34pp | 304 | 0.62ms |
| **qat_int8 (1 epoch fine-tune)** | **82.41%** | **+0.07pp** | 293 | 0.94ms |

What this says: on this small CNN, **QAT recovers the entire
accuracy drop from static PTQ** and lands fractionally above the FP32
baseline (the fake-quant noise during training acts as a mild
regulariser). The cost is a few minutes of fine-tuning. PTQ remains
the right answer when retraining isn't an option (no labelled data,
no pipeline, or the trained model is a black box); QAT is the right
answer when you control the training pipeline and care about every
fraction of a point of accuracy.

Honest caveat: this is a small, well-behaved network where PTQ already
gets to within 0.34pp of FP32. QAT's relative win usually grows with
network size and quantization aggressiveness — INT4 weight-only QAT
on a transformer can recover several percentage points where PTQ falls
off a cliff.

## Cross-runtime: PyTorch quantized vs ONNX Runtime quantized

The same four PTQ configs can be exported to ONNX and benched under
ONNX Runtime's CPU EP for a head-to-head with the PyTorch quantized
runtime. `quant-explorer cross-runtime` runs the comparison and writes
[`artifacts/results/cross_runtime.md`](artifacts/results/cross_runtime.md)
+ `cross_runtime.json`. Numbers from a recent run on a 4-core
M-series CPU (full 10 000-image test split, 256-image calibration):

| config | pt_top1 | onnx_top1 | top1_delta_pp | pt_p50_ms | onnx_p50_ms | latency_ratio | pt_size_kb | onnx_size_kb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32_baseline | 82.3% | 82.3% | 0.00 | 1.83 | 0.83 | 0.46x | 1144 | 1129 |
| dynamic_int8 | 82.3% | 82.3% | 0.00 | 1.14 | 0.38 | 0.33x | 1141 | 1128 |
| static_int8_per_tensor | 82.1% | 82.1% | -0.05 | 1.77 | 0.18 | 0.10x | 293 | 297 |
| static_int8_per_channel | 82.0% | 82.3% | +0.27 | 1.27 | 0.18 | 0.14x | 304 | 303 |

What this says: **every config's top-1 agrees across runtimes within
+/-0.3pp** (well inside the +/-1pp structural-parity tolerance we
assert; static INT8 is lossy by definition so exact bit-parity isn't
the goal). On-disk size matches to within ~1% per config. Latency is
where the two runtimes diverge: ORT CPU EP is consistently faster on
this network (4-10x at INT8) because the ORT CPU INT8 kernels for
small convolutions are more mature on x86 Linux than PyTorch's
eager-mode quantized ops.

Methodology + per-runtime export plumbing:
[`docs/cross_runtime.md`](docs/cross_runtime.md). Cross-linked from
`SAY-5/onnx-deploy` (consumer of the `.onnx` files) and
`SAY-5/export-validator` (re-uses the +/-1pp parity gate).

## What this is not

- **Not INT4 / INT2.** PyTorch's CPU backends don't have first-class
  kernels for sub-INT8 weights. INT4 lives in different stacks
  (bitsandbytes for GPU, GGML, ONNX-Runtime).
- **Not GPU quantization.** PyTorch eager-mode quantization targets
  CPU. GPU quantization (TensorRT, FasterTransformer) is a separate
  toolchain.
- **Not transformer / NLP.** Quantizing transformers (KV-cache,
  attention masks, LayerNorm) is shaped differently enough that
  bundling it with vision PTQ would be misleading. Vision only.

## Repo layout

```
src/quant_explorer/
  cli.py              Click entrypoint (train, quantize, bench, evaluate, report, pipeline)
  model.py            small CNN (~290k params)
  train.py            CIFAR-10 training loop
  data.py             dataset + transforms + calibration loader
  quant/              one module per quantization config; auto-registered
  bench/              latency / memory / size measurement
  eval/               top-1 / top-5 / per-class accuracy
  onnx_rt/            FP32 export, ONNX-side INT8 quantization, ORT CPU EP bench
  report/             pareto frontier + JSON / Markdown emit
  settings.py         paths, dataclasses, engine selection
artifacts/
  weights/            committed fp32_baseline.pt + per-config quantized state_dicts
  results/            committed full_results.json + pareto.md
docs/                 quantization.md, methodology.md, README.md (extension guide)
tests/
  unit/               pareto algorithm, percentile math, accuracy aggregator, etc.
  integration/        tiny end-to-end pipeline (RUN_INTEGRATION=1)
  e2e/                full-pipeline shell script (laptop)
```

## License

MIT. See [`LICENSE`](LICENSE).
