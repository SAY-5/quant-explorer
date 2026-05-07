"""Quantization registry: presence, dynamic apply, calibration requirement."""

from __future__ import annotations

import pytest
import torch

from quant_explorer import quant as quant_pkg
from quant_explorer.model import CifarCNN
from quant_explorer.settings import select_quantization_engine


@pytest.fixture(autouse=True)
def _set_engine() -> None:
    torch.backends.quantized.engine = select_quantization_engine()


def test_known_configs_are_registered() -> None:
    names = set(quant_pkg.list_configs())
    assert {"dynamic_int8", "static_int8_per_tensor", "static_int8_per_channel"}.issubset(names)


def test_dynamic_int8_does_not_need_calibration() -> None:
    cfg = quant_pkg.get_config("dynamic_int8")
    assert cfg.needs_calibration is False


def test_static_configs_need_calibration() -> None:
    for name in ("static_int8_per_tensor", "static_int8_per_channel"):
        cfg = quant_pkg.get_config(name)
        assert cfg.needs_calibration is True


def test_dynamic_int8_apply_runs_inference() -> None:
    cfg = quant_pkg.get_config("dynamic_int8")
    model = CifarCNN().eval()
    qm = cfg.apply(model, None)
    out = qm(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_static_apply_rejects_no_calibration() -> None:
    cfg = quant_pkg.get_config("static_int8_per_tensor")
    model = CifarCNN().eval()
    with pytest.raises(ValueError):
        cfg.apply(model, None)


def test_get_config_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        quant_pkg.get_config("nonexistent_config")
