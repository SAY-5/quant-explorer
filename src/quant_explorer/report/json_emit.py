"""Full-results JSON emitter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def emit_full_results(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(o: Any) -> Any:
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(f"unserializable: {type(o).__name__}")
