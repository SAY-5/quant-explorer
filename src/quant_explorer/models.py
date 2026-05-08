"""Multi-model registry for the quantization study.

Each registered model exposes:
  * ``build()`` — returns an FP32 ``nn.Module`` ready for quantization. The
    module is wrapped in a ``QuantStub`` / ``DeQuantStub`` when it isn't
    already a ``CifarCNN``.
  * ``input_shape`` — a ``(C, H, W)`` tuple for synthetic-input benches.
  * ``measures_accuracy`` — whether ``evaluate_accuracy`` is meaningful
    for this model on the CIFAR-10 test split. Only ``small_cnn`` is
    CIFAR-10-trained. The torchvision models are pre-trained on
    ImageNet and operate on different image dimensions, so accuracy on
    CIFAR-10 would be uninformative; we still bench latency and on-disk
    size at CIFAR-10-ish input shape (32x32 for ``small_cnn``,
    224x224 for the torchvision models).

This intentionally keeps the registry tiny — three models, with
explicit, documented limits on what each one says about quantization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision import models as tv_models

from .model import CifarCNN


class _QuantWrappedTV(nn.Module):
    """Wraps a torchvision model with ``QuantStub`` / ``DeQuantStub``.

    ``inner.fuse_modules`` (if present, e.g. from
    ``torchvision.models.mobilenet_v3_small().fuse_model()``) is exposed
    via a ``fuse_modules`` method so the existing static-quant configs
    can call it uniformly.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.quant = QuantStub()  # type: ignore[no-untyped-call]
        self.inner = inner
        self.dequant = DeQuantStub()  # type: ignore[no-untyped-call]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.inner(x)
        return self.dequant(x)  # type: ignore[no-any-return]

    def fuse_modules(self) -> _QuantWrappedTV:
        # Torchvision provides a ``fuse_model()`` on quantizable variants.
        if hasattr(self.inner, "fuse_model"):
            self.inner.fuse_model()
        return self


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], nn.Module]
    input_shape: tuple[int, int, int]
    measures_accuracy: bool
    description: str


def _build_small_cnn() -> nn.Module:
    return CifarCNN(num_classes=10, quantizable=True)


class _MobileNetV3Adapter(nn.Module):
    """Adapter so ``torchvision.models.quantization.mobilenet_v3_large`` plugs
    into the existing ``apply()`` pipeline.

    The quantizable variant already has its own ``QuantStub`` / ``DeQuantStub``
    and a ``fuse_model()``. We expose ``fuse_modules()`` (matching the
    ``CifarCNN`` interface) and pre-set a qnnpack-compatible qconfig
    because MobileNetV3's HardSwish + SE blocks won't quantize correctly
    under the generic ``default_observer`` qconfig.
    """

    def __init__(self) -> None:
        super().__init__()
        inner = tv_models.quantization.mobilenet_v3_large(weights=None, quantize=False)
        inner.eval()
        self.inner = inner
        # Pre-set the qnnpack qconfig at construction so the existing
        # static configs respect it (they only set their own qconfig if
        # one isn't set).
        self.qconfig = torch.ao.quantization.get_default_qconfig(  # type: ignore[no-untyped-call]
            "qnnpack"
        )
        self.inner.qconfig = self.qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)  # type: ignore[no-any-return]

    def fuse_modules(self) -> _MobileNetV3Adapter:
        self.inner.fuse_model()
        return self


def _build_mobilenet_v3() -> nn.Module:
    return _MobileNetV3Adapter()


def _build_vgg11_bn() -> nn.Module:
    # ``vgg11_bn`` doesn't have an official quantizable variant in
    # torchvision, but Conv-BN-ReLU triples in ``features`` fuse cleanly
    # with the eager-mode quantizer. Random init.
    inner = tv_models.vgg11_bn(weights=None)
    inner.eval()
    wrapped = _QuantWrappedTV(inner)

    # Patch a fuse_modules onto the wrapper that walks ``features`` and
    # fuses Conv-BN-ReLU triples wherever present.
    def _fuse() -> _QuantWrappedTV:
        from torch.ao.quantization import fuse_modules

        feats = inner.features
        # Walk the Sequential and find runs of (Conv2d, BatchNorm2d, ReLU).
        triples: list[list[str]] = []
        children = list(feats.named_children())
        i = 0
        while i + 2 < len(children):
            n0, m0 = children[i]
            n1, m1 = children[i + 1]
            n2, m2 = children[i + 2]
            if (
                isinstance(m0, nn.Conv2d)
                and isinstance(m1, nn.BatchNorm2d)
                and isinstance(m2, nn.ReLU)
            ):
                triples.append([f"features.{n0}", f"features.{n1}", f"features.{n2}"])
                i += 3
            else:
                i += 1
        if triples:
            fuse_modules(inner, triples, inplace=True)  # type: ignore[no-untyped-call]
        return wrapped

    wrapped.fuse_modules = _fuse  # type: ignore[method-assign]
    return wrapped


REGISTRY: dict[str, ModelSpec] = {
    "small_cnn": ModelSpec(
        name="small_cnn",
        builder=_build_small_cnn,
        input_shape=(3, 32, 32),
        measures_accuracy=True,
        description="The compact CIFAR-10 CNN trained from scratch (~290k params).",
    ),
    "mobilenet_v3": ModelSpec(
        name="mobilenet_v3",
        # torchvision 0.17.x ships ``mobilenet_v3_large`` under the
        # ``quantization`` namespace; ``small`` is only added in newer
        # versions. ``large`` is the canonical depthwise-separable
        # mobile net for this family and serves the same study point —
        # show that quantization shrinks an inverted-residual model
        # dramatically vs the small custom CNN.
        builder=_build_mobilenet_v3,
        input_shape=(3, 224, 224),
        measures_accuracy=False,
        description=(
            "torchvision MobileNetV3-Large (quantizable family). Benched "
            "for latency and on-disk size at 224x224; CIFAR-10 accuracy "
            "is not measured (ImageNet domain)."
        ),
    ),
    "vgg11_bn": ModelSpec(
        name="vgg11_bn",
        builder=_build_vgg11_bn,
        input_shape=(3, 224, 224),
        measures_accuracy=False,
        description=(
            "torchvision VGG-11 with batch-norm. Benched for latency "
            "and on-disk size at 224x224; CIFAR-10 accuracy is not "
            "measured (ImageNet domain)."
        ),
    ),
}


def list_models() -> list[str]:
    return sorted(REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown model: {name!r}; known={list_models()}")
    return REGISTRY[name]
