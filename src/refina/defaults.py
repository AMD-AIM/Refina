"""Built-in engine defaults for image-only command execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .engines.registry import specs


def engine_defaults(engine: str) -> dict[str, Any]:
    spec = specs().get(engine)
    return deepcopy(dict(spec.defaults)) if spec else {}
