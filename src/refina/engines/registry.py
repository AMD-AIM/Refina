"""Engine registry backed by the ``refina.engines`` entry point group.

Each entry point resolves to an :class:`EngineSpec`. Engine modules import heavy
frameworks inside their functions, so reading specs stays CPU-only and a broken
trl install cannot stop a swift run from starting. A plugin whose spec fails to
import is reported on its own and does not hide the others.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from typing import TYPE_CHECKING

from ..errors import UsageError
from ..params import ENGINES
from .spec import ENTRY_POINT_GROUP, EngineSpec

if TYPE_CHECKING:
    from .base import Engine

_broken: dict[str, str] = {}


@functools.cache
def specs() -> dict[str, EngineSpec]:
    entry_points = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
    if not entry_points:
        raise UsageError(
            f"no engines registered under the '{ENTRY_POINT_GROUP}' entry point group; "
            "install refina with `pip install -e .` so its entry points exist"
        )
    found: dict[str, EngineSpec] = {}
    _broken.clear()
    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
        except Exception as exc:  # one bad plugin must not hide the rest
            _broken[entry_point.name] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(loaded, EngineSpec):
            _broken[entry_point.name] = f"entry point does not resolve to an EngineSpec: {loaded!r}"
            continue
        if loaded.name != entry_point.name:
            raise UsageError(
                f"engine entry point '{entry_point.name}' publishes a spec named '{loaded.name}'"
            )
        found[loaded.name] = loaded
    return found


def broken() -> dict[str, str]:
    """Plugins whose spec could not be loaded, with the reason."""
    specs()
    return dict(_broken)


def available() -> tuple[str, ...]:
    registered = specs()
    builtin = tuple(name for name in ENGINES if name in registered)
    return builtin + tuple(sorted(set(registered) - set(builtin)))


def spec(name: str) -> EngineSpec:
    registered = specs()
    if name in registered:
        return registered[name]
    if name in _broken:
        raise UsageError(f"engine '{name}' failed to load: {_broken[name]}")
    raise UsageError(f"unknown engine '{name}'; expected one of {', '.join(available())}")


def load(name: str) -> Engine:
    module_name, _, attribute = spec(name).factory.partition(":")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory()
