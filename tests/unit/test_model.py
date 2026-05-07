"""Model architecture sanity checks."""

from __future__ import annotations

import torch

from quant_explorer.model import CifarCNN, count_parameters


def test_forward_shape_matches_num_classes() -> None:
    model = CifarCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10)


def test_param_count_within_expected_range() -> None:
    """Model is sized to be CPU-friendly. Pin a wide range so unrelated
    refactors don't break the test, but catch a 10x size regression.
    """
    model = CifarCNN()
    n = count_parameters(model)
    assert 100_000 < n < 600_000


def test_quantizable_flag_inserts_stubs() -> None:
    qm = CifarCNN(quantizable=True)
    nqm = CifarCNN(quantizable=False)
    assert qm(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert nqm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_fuse_modules_returns_self() -> None:
    """fuse_modules is in-place and returns the module so it can chain.

    Fusion requires eval mode (see torch.nn.utils.fusion.fuse_conv_bn_eval).
    """
    model = CifarCNN().eval()
    out = model.fuse_modules()
    assert out is model
