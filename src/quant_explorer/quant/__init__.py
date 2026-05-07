"""Quantization configurations and registry."""

from . import dynamic, static_per_channel, static_per_tensor  # noqa: F401  side-effect import
from ._base import REGISTRY, QuantConfig, get_config, list_configs

__all__ = ["REGISTRY", "QuantConfig", "get_config", "list_configs"]
