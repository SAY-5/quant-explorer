"""On-disk model size: bytes of the saved ``state_dict``.

We measure the size of what actually ships (the state_dict, in PyTorch's
serialization format) rather than the in-memory module footprint, which
includes scaffolding the receiver doesn't need.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class SizeResult:
    bytes: int
    kb: float

    def as_dict(self) -> dict[str, float | int]:
        return {"bytes": self.bytes, "kb": self.kb}


def state_dict_size(model: nn.Module) -> SizeResult:
    """Serialize the state_dict to an in-memory buffer and report size."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    n = buf.tell()
    return SizeResult(bytes=n, kb=n / 1024.0)


def file_size(path: Path) -> SizeResult:
    n = path.stat().st_size
    return SizeResult(bytes=n, kb=n / 1024.0)


def size_ratio(target_kb: float, baseline_kb: float) -> float:
    if baseline_kb <= 0.0:
        raise ValueError("baseline_kb must be > 0")
    return target_kb / baseline_kb
