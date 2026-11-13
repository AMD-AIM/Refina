"""Engine metadata that plugins publish through the ``refina.engines`` entry point group.

Reading a spec must stay cheap: engine modules define ``SPEC`` at top level and
import their framework only inside functions, so listing engines never pulls in
torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

ENTRY_POINT_GROUP: Final = "refina.engines"


@dataclass(frozen=True)
class DoctorCheck:
    module: str
    package: str
    executable: str | None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    factory: str  # "package.module:Class", imported only when the engine runs
    framework: str
    mode: str
    input: str
    defaults: Mapping[str, Any]
    env: str | None = None
    doctor: DoctorCheck | None = None
    dataset_purpose: str | None = None
    optional: bool = False
    internal: bool = False

    @property
    def plugin(self) -> str:
        if self.internal:
            return "internal"
        return "optional" if self.optional else "core"
