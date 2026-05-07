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
| fp32_baseline | 1144 | 1.00x | 1.67 | 1.00x | 82.3% | 0.0pp | 585.8 | yes |
| dynamic_int8 | 1141 | 1.00x | 0.70 | 2.38x | 82.3% | -0.0pp | 280.2 | yes |
| static_int8_per_tensor | 293 | 0.26x | 0.67 | 2.51x | 82.1% | -0.2pp | 642.3 | yes |
| static_int8_per_channel | 304 | 0.27x | 0.62 | 2.72x | 82.0% | -0.3pp | 622.3 | yes |

Pareto frontier picks:
- minimum size: `static_int8_per_tensor` (0.26x of FP32)
- highest accuracy: `fp32_baseline` (top-1 82.3%)
- lowest latency: `static_int8_per_channel` (p50 0.62ms at batch 1)

What this says: **static INT8 quantization (per-channel) cuts model
size to ~27% and runs 2.7x faster than FP32 with a 0.3-percentage-point
top-1 accuracy drop**. Dynamic INT8 (which only quantizes `nn.Linear`)
still gets a meaningful 2.4x speedup with essentially zero accuracy
loss because it leaves the conv layers in FP32. Per-tensor static is
slightly smaller than per-channel and the accuracy gap is modest at
this scale; on bigger models the per-channel advantage usually grows.

Full per-config measurements (latency at batch sizes 1, 8, 32; memory;
per-class accuracy) live in
[`artifacts/results/full_results.json`](artifacts/results/full_results.json).

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

## What this is not

- **Not QAT.** Quantization-aware training inserts fake-quant ops
  during fine-tuning. It usually beats PTQ but takes another training
  run. Documented as future work.
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
