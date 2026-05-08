"""Quantization configurations and registry."""

from . import (  # noqa: F401  side-effect import
    dynamic,
    qat,
    static_per_channel,
    static_per_tensor,
)
from ._base import REGISTRY, QuantConfig, get_config, list_configs

__all__ = ["REGISTRY", "QuantConfig", "get_config", "list_configs"]
