"""Small CNN for CIFAR-10.

Three Conv-BN-ReLU x2 blocks (32, 64, 128 channels) with MaxPool between
blocks, followed by adaptive average pooling and a single linear classifier.

The model is intentionally small (~290k params) so it (a) trains in a few
minutes on a laptop CPU and (b) the per-config quantization tradeoffs are
visible. The structure (Conv2d, BatchNorm2d, ReLU) folds cleanly during
static quantization preparation.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub


class ConvBlock(nn.Module):
    """Conv-BN-ReLU x2 with a MaxPool tail.

    Sequence is ordered so ``torch.ao.quantization.fuse_modules`` can fuse
    each Conv-BN-ReLU triple in place.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return cast(torch.Tensor, self.pool(x))


class CifarCNN(nn.Module):
    """Compact CNN for CIFAR-10 sized for CPU inference.

    With ``quantizable=True``, ``QuantStub``/``DeQuantStub`` are inserted
    around the forward pass so the module is ready for static quantization.
    For dynamic quantization (which only quantizes ``nn.Linear``) the stubs
    are inert.
    """

    def __init__(self, num_classes: int = 10, quantizable: bool = True) -> None:
        super().__init__()
        self.quantizable = quantizable
        self.quant = QuantStub()  # type: ignore[no-untyped-call]
        self.dequant = DeQuantStub()  # type: ignore[no-untyped-call]
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantizable:
            x = self.quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        if self.quantizable:
            x = self.dequant(x)
        return x

    def fuse_modules(self) -> CifarCNN:
        """Fuse Conv-BN-ReLU triples for static quantization. In-place.

        Requires eval mode (``torch.nn.utils.fusion.fuse_conv_bn_eval``
        asserts ``not training``).
        """
        from torch.ao.quantization import fuse_modules

        for block_name in ("block1", "block2", "block3"):
            block = getattr(self, block_name)
            fuse_modules(  # type: ignore[no-untyped-call]
                block,
                [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]],
                inplace=True,
            )
        return self


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
