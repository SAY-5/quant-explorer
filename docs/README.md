# Adding a new quantization config

1. Create a new module under `src/quant_explorer/quant/`. Pattern:

   ```python
   # src/quant_explorer/quant/my_new_config.py
   from collections.abc import Iterable
   import torch
   from torch import nn
   from ._base import QuantConfig, register_quant_config

   def _apply(model: nn.Module, calibration: Iterable[torch.Tensor] | None) -> nn.Module:
       # Apply your quantization recipe here.
       return ...

   register_quant_config(QuantConfig(
       name="my_new_config",
       needs_calibration=True,  # or False
       apply=_apply,
       description="one-line summary",
   ))
   ```

2. Import the module in `src/quant_explorer/quant/__init__.py` so
   registration runs at import time:

   ```python
   from . import my_new_config  # noqa: F401
   ```

3. Add it to the `ALL_CONFIGS` tuple in `src/quant_explorer/cli.py`
   (this is the source of truth for the bench-all and report
   aggregation).

4. Add it to the `Makefile` `quantize-all`, `bench-all`, `evaluate-all`
   targets.

5. Run `make pipeline` locally. The new config is picked up
   automatically by `report` so the Pareto markdown will include it.

6. Update [`docs/quantization.md`](quantization.md) with a per-config
   description.

The registry is import-time side-effecting, which keeps `cli.py`'s
`click.Choice` arguments in sync with what's actually available without
a second source of truth.
