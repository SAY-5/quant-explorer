# Quantization tradeoff Pareto

| config | size_kb | size_ratio | p50_lat_ms_b1 | latency_speedup | top1_acc | acc_drop_pp | mem_peak_mb | pareto_optimal |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| fp32_baseline | 1144 | 1.00x | 1.67 | 1.00x | 82.3% | 0.0pp | 585.8 | no |
| dynamic_int8 | 1141 | 1.00x | 0.70 | 2.38x | 82.3% | -0.0pp | 280.2 | yes |
| static_int8_per_tensor | 293 | 0.26x | 0.67 | 2.51x | 82.1% | -0.2pp | 642.3 | yes |
| static_int8_per_channel | 304 | 0.27x | 0.62 | 2.72x | 82.0% | -0.3pp | 622.3 | yes |
| qat_int8 | 293 | 0.26x | 0.94 | 1.78x | 82.4% | +0.1pp | 222.8 | yes |

Pareto frontier picks:
- minimum size: `qat_int8` (0.26x of FP32)
- highest accuracy: `qat_int8` (top-1 82.4%)
- lowest latency: `static_int8_per_channel` (p50 0.62ms at batch 1)
