# Cross-runtime: PyTorch quantized vs ONNX Runtime quantized

This doc describes the methodology for the cross-runtime comparison
written to `artifacts/results/cross_runtime.{json,md}` by the
`quant-explorer cross-runtime` CLI command.

## What this measures

The question is: **for the same quantization config, does the model
behave the same when run under PyTorch's quantized runtime vs ONNX
Runtime's CPU EP?** Three axes:

- **Top-1 accuracy** on the CIFAR-10 test split, end-to-end (no
  trickery: each runtime receives the same test loader, runs full
  inference, and reports its own top-1).
- **p50 latency at batch size 1** (single-image inference), using each
  runtime's native timing path: `time.perf_counter()` around the
  `model(x)` / `sess.run(...)` call, identical warmup + measure
  schedules.
- **On-disk size**: the PyTorch `state_dict` `.pt` file vs the
  `.onnx` (or quantized `.onnx`) file. We compare what ships, not the
  in-memory module footprint.

## Why exact bit-parity is not the goal

Static INT8 PTQ in PyTorch eager-mode (FBGEMM on x86, QNNPACK on arm)
uses one specific calibrator algorithm + one specific Conv-BN-ReLU
fusion ordering. ONNX Runtime's `quantize_static` uses the QDQ format
with its own (closely related but not identical) calibrator and
fold ordering. The two paths can therefore land on slightly different
INT8 weights even when fed identical FP32 weights and calibration
data.

The structural parity invariant we assert is **top-1 within +/-1
percentage point**. Encoded as a constant in
`quant_explorer.onnx_rt.compare.ACCURACY_TOL_PP` and verified by
`test_accuracy_tol_pp_is_one_percentage_point`. If you change the
tolerance you also need to update this doc and the README; the
constant is load-bearing.

## How each config is exported

| config | PT side | ONNX side |
|---|---|---|
| `fp32_baseline` | Saved state_dict | `torch.onnx.export` from the FP32 module |
| `dynamic_int8` | `quantize_dynamic` over `nn.Linear` | `onnxruntime.quantization.quantize_dynamic` restricted to `MatMul`/`Gemm` |
| `static_int8_per_tensor` | `prepare` + calibrate + `convert`, per-tensor weight observer | `quantize_static`, QDQ format, `per_channel=False`, real CIFAR-10 calibration |
| `static_int8_per_channel` | same, per-channel weight observer | `quantize_static`, QDQ format, `per_channel=True`, real CIFAR-10 calibration |

`qat_int8` is **not** in the cross-runtime grid: QAT export to ONNX
requires a different code path (export the prepared model with
fake-quant ops baked in, not the converted INT8 graph). Tracked as a
follow-up; the four PTQ configs are the headline comparison.

### Why dynamic_int8 is restricted to MatMul/Gemm in ONNX

ONNX Runtime's CPU EP does not ship a kernel for `ConvInteger` (the op
that `quantize_dynamic` emits for convolutions by default). Quantizing
the full graph dynamically therefore produces a model that loads but
fails at `sess.run` with `NOT_IMPLEMENTED`. Restricting to
`MatMul`/`Gemm` mirrors what PyTorch's dynamic INT8 PTQ does (it only
quantizes `nn.Linear`), so the comparison is honest: both runtimes are
running a model where only the linear layer's weights are INT8 and
activations are quantized on the fly.

## Cross-links

- **`SAY-5/onnx-deploy`** consumes the `.onnx` files this command
  produces as its deployment artifact. The cross-runtime gate here is
  the canary that catches export-vs-runtime drift before the deploy
  pipeline does.
- **`SAY-5/export-validator`** re-uses the +/-1pp parity assertion as
  a generic export-quality gate. If you change `ACCURACY_TOL_PP` here,
  bump the matching constant there.

## CI

The `cross-runtime-smoke` job runs the command on a 2000-image
accuracy subset with 128 calibration images. It then asserts:

1. All four PTQ configs are present in the output.
2. Every row's top-1 delta is within +/-5pp (the CI regression gate).
3. Latency and size are positive for both runtimes.
4. The Markdown report exists and contains the SAY-5 cross-links.

### CI gate (+/-5pp) vs publishable claim (+/-1pp)

The publishable parity claim is **+/-1pp on the full 10 000-image test
split**, measured locally on Apple Silicon (qnnpack + ORT CPU EP) and
committed to `artifacts/results/cross_runtime.{json,md}`. The CI job
runs on Linux x86 (fbgemm + ORT CPU EP) and on a 2000-image subset; in
that environment ORT's per-channel static-INT8 calibrator diverges
from PT eager-mode fbgemm by ~2pp on this small CNN, close enough
that something is working, far enough that the 1pp gate isn't a useful
smoke signal.

The +/-5pp CI gate is therefore a **regression canary**: if any cell
drifts past 5pp something structural broke (wrong calibration data,
missing fusion, opset mismatch). The publishable +/-1pp claim is
verified by the committed full-run artifacts, not by the smoke job.

The `ACCURACY_TOL_PP` constant in
`quant_explorer.onnx_rt.compare` still carries the publishable value
(1.0); CI's gate is open-coded in the workflow because it is an
environment-specific looser threshold, not the structural assertion.
