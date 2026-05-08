# Multi-model quantization Pareto

Each model has its own frontier (cross-model latency / size comparisons aren't meaningful — different input shapes, parameter counts, and intended deployment targets).

## mobilenet_v3
Input shape: (3, 224, 224). Top-1 not measured (ImageNet domain — see README).

| quant_config | size_kb | size_ratio | p50_ms | speedup | top1 | pareto |
|---|---:|---:|---:|---:|---:|:---:|
| fp32_baseline | 21622 | 1.00x | 1279.48 | 1.00x | n/a | no |
| dynamic_int8 | 14273 | 0.66x | 649.60 | 1.97x | n/a | no |
| static_int8_per_tensor | 5520 | 0.26x | 26.10 | 49.02x | n/a | no |
| static_int8_per_channel | 5520 | 0.26x | 22.63 | 56.55x | n/a | yes |

## small_cnn
Input shape: (3, 32, 32). Top-1 measured on the CIFAR-10 test split.

| quant_config | size_kb | size_ratio | p50_ms | speedup | top1 | pareto |
|---|---:|---:|---:|---:|---:|:---:|
| fp32_baseline | 1144 | 1.00x | 8.12 | 1.00x | n/a | no |
| dynamic_int8 | 1141 | 1.00x | 12.28 | 0.66x | n/a | no |
| static_int8_per_tensor | 293 | 0.26x | 1.69 | 4.81x | n/a | yes |
| static_int8_per_channel | 304 | 0.27x | 1.82 | 4.45x | n/a | no |

## vgg11_bn
Input shape: (3, 224, 224). Top-1 not measured (ImageNet domain — see README).

| quant_config | size_kb | size_ratio | p50_ms | speedup | top1 | pareto |
|---|---:|---:|---:|---:|---:|:---:|
| fp32_baseline | 519061 | 1.00x | 81.62 | 1.00x | n/a | yes |
| dynamic_int8 | 156855 | 0.30x | 89.77 | 0.91x | n/a | yes |
| static_int8_per_tensor | 129800 | 0.25x | 132.60 | 0.62x | n/a | yes |
| static_int8_per_channel | 129993 | 0.25x | 149.03 | 0.55x | n/a | no |
