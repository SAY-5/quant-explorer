"""Quantization config protocol + registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QuantConfig:
    """A named quantization configuration.

    ``apply`` takes a trained FP32 model and returns a quantized module
    suitable for inference. For static configs the calibration iterator is
    consumed during ``apply``; for dynamic configs it is ignored.
    """

    name: str
    needs_calibration: bool
    apply: Callable[[nn.Module, Iterable[torch.Tensor] | None], nn.Module]
    description: str


REGISTRY: dict[str, QuantConfig] = {}


def register_quant_config(cfg: QuantConfig) -> QuantConfig:
    if cfg.name in REGISTRY:
        raise ValueError(f"quant config {cfg.name!r} already registered")
    REGISTRY[cfg.name] = cfg
    return cfg


def get_config(name: str) -> QuantConfig:
    if name not in REGISTRY:
        raise KeyError(f"unknown quant config: {name!r}; known={sorted(REGISTRY)}")
    return REGISTRY[name]


def list_configs() -> list[str]:
    return sorted(REGISTRY)
