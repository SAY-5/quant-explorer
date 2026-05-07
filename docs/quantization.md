# Quantization configurations

Each entry below is a config name as it appears in
`artifacts/results/full_results.json` and as a CLI option to
`quant-explorer quantize --config <name>`.

---

## `fp32_baseline`

The trained CIFAR-10 model in IEEE 754 single-precision floats. This is
the reference point — every other config is reported as a delta against
this one. No quantization is applied.

The weights file (`fp32_baseline.pt`) is committed and re-used for every
quantization run, so the quantization comparisons are deterministic with
respect to the trained weights.

---

## `dynamic_int8`

Dynamic post-training quantization over `nn.Linear` modules only:

```python
torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

Weights are pre-quantized to INT8 once. Activations are observed and
quantized on the fly per inference call (this is what "dynamic" means).
No calibration data is required, which makes this the cheapest config to
apply by orders of magnitude.

For a CNN like this, `nn.Linear` is just the final classifier head
(`Linear(128, 10)`). Conv layers are not touched. Expect a small size
reduction and roughly baseline accuracy. Included as the simplest
PyTorch quantization recipe and as a control: if your model is mostly
conv, dynamic quantization barely moves the needle, and that's the
point to demonstrate.

---

## `static_int8_per_tensor`

Full-graph post-training static quantization with per-tensor weight and
activation observers:

```python
qconfig = QConfig(
    activation=default_observer,
    weight=default_weight_observer,  # per-tensor MinMax
)
```

Steps:

1. Fuse `Conv2d -> BatchNorm2d -> ReLU` triples (so the quantizer sees
   the fused op).
2. Insert `QuantStub` / `DeQuantStub` at module entry / exit (already
   baked into `CifarCNN(quantizable=True)`).
3. Set `model.qconfig = qconfig` and call
   `torch.ao.quantization.prepare(model)`.
4. Run inference on 256 unaugmented training images so the observers can
   record activation statistics.
5. Call `torch.ao.quantization.convert(model)` — observers are replaced
   with quantized ops using the recorded ranges.

Per-tensor means each weight tensor gets exactly one scale and one
zero-point. This is the simplest mapping and historically the
fastest, but it loses accuracy on layers whose output channels have
very different dynamic ranges (one outlier pegs the scale and shrinks
the resolution everyone else sees).

Expect substantial size reduction (~4x for the conv-heavy parts)
and a moderate latency speedup (depending on backend kernel
availability). Accuracy may drop more than per-channel.

---

## `static_int8_per_channel`

Same as above except weights are observed per output channel:

```python
qconfig = QConfig(
    activation=default_observer,
    weight=default_per_channel_weight_observer,
)
```

Each output channel of each conv weight tensor gets its own scale.
Outlier channels no longer pollute the scale of the rest. Activations
remain per-tensor (only weights gain channel resolution).

This is the recommended default for static PTQ in PyTorch. The accuracy
recovery vs per-tensor is usually meaningful with negligible inference
overhead.

---

## When to pick which

| If you care about… | Pick |
|---|---|
| Drop-in for a model that's mostly `Linear` | `dynamic_int8` |
| Smallest deployable artifact | `static_int8_per_tensor` |
| Best accuracy retention at INT8 | `static_int8_per_channel` |
| Reference number / debugging | `fp32_baseline` |

The actual numbers from a run on this CNN are in
[`artifacts/results/pareto.md`](../artifacts/results/pareto.md).

---

## Calibration set size

Static quantization uses 256 unaugmented training images by default
(`--calibration-n 256`). Activation observers (the `MinMaxObserver`
variant `default_observer` uses) just need enough samples to see a
representative range. 256 is a common default; smaller (32) is fine for
a smoke test, larger (1024+) doesn't usually help past a point of
diminishing returns for this kind of small CNN.

The calibration loader explicitly disables augmentation (random crops
and flips) so the observer sees the same statistics the model will see
at inference time.
