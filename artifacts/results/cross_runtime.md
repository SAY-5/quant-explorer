# Cross-runtime comparison: PyTorch quantized vs ONNX Runtime quantized

Top-1 accuracy parity is asserted within +/-1.0pp; static INT8 in PyTorch (eager-mode FBGEMM/QNNPACK) and ONNX Runtime (QDQ format) differ on small numerical details, so exact bit-parity is not the goal. Latency is p50 at batch 1; size is the on-disk state_dict (PT) or `.onnx` file (ONNX).

| config | pt_top1 | onnx_top1 | top1_delta_pp | pt_p50_ms | onnx_p50_ms | latency_ratio | pt_size_kb | onnx_size_kb | size_ratio | within_tol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| fp32_baseline | 82.3% | 82.3% | 0.00 | 1.83 | 0.83 | 0.46x | 1144 | 1129 | 0.99x | yes |
| dynamic_int8 | 82.3% | 82.3% | 0.00 | 1.14 | 0.38 | 0.33x | 1141 | 1128 | 0.99x | yes |
| static_int8_per_tensor | 82.1% | 82.1% | -0.05 | 1.77 | 0.18 | 0.10x | 293 | 297 | 1.01x | yes |
| static_int8_per_channel | 82.0% | 82.3% | +0.27 | 1.27 | 0.18 | 0.14x | 304 | 303 | 1.00x | yes |

Cross-links:
- `SAY-5/onnx-deploy` consumes the ONNX files produced here as its
  deployment artifact (CPU EP target).
- `SAY-5/export-validator` re-uses the parity assertion above as a
  generic export-quality gate (top-1 within +/-1pp = pass).
