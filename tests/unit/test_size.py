"""Size measurement and ratio math."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from quant_explorer.bench.size import size_ratio, state_dict_size


def test_state_dict_size_grows_with_params() -> None:
    small = nn.Linear(8, 8)
    big = nn.Linear(64, 64)
    s_small = state_dict_size(small)
    s_big = state_dict_size(big)
    assert s_big.bytes > s_small.bytes
    assert s_small.kb == pytest.approx(s_small.bytes / 1024.0)


def test_size_ratio_divides() -> None:
    assert size_ratio(50.0, 100.0) == pytest.approx(0.5)
    assert size_ratio(150.0, 100.0) == pytest.approx(1.5)


def test_size_ratio_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError):
        size_ratio(50.0, 0.0)


def test_state_dict_size_round_trip_matches() -> None:
    """Saving via the API and saving manually should produce identical sizes."""
    layer = nn.Linear(16, 32)
    s = state_dict_size(layer)
    buf = io.BytesIO()
    torch.save(layer.state_dict(), buf)
    assert s.bytes == buf.tell()
