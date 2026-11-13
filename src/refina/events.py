"""Machine-readable progress lines for the service layer.

Each event is one log line ending in ``REFINA_EVENT <json>``; the JSON always
has ``event`` and ``time`` (UTC, RFC 3339)::

    REFINA_EVENT {"event": "stage", "stage": "downloading", "time": "..."}
    REFINA_EVENT {"event": "exit", "exit_code": 5, "code": "IntegrityCheckFailed",
                  "stage": "downloading", ...}

Stages, in order: ``downloading``, ``training``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

MARKER = "REFINA_EVENT"
STAGES = ("downloading", "training")

logger = logging.getLogger("refina.events")
# The service parses these lines, so LOG_LEVEL=WARNING must not hide them.
logger.setLevel(logging.INFO)

_current: dict[str, str | None] = {"stage": None}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def emit(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields, "time": utc_now()}
    logger.info("%s %s", MARKER, json.dumps(payload, ensure_ascii=False, default=str))


def reset() -> None:
    _current["stage"] = None


def stage(name: str) -> None:
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}")
    if _current["stage"] == name:
        return
    _current["stage"] = name
    emit("stage", stage=name)


def current_stage() -> str | None:
    return _current["stage"]


def finish(exit_code: int, code: str | None = None) -> None:
    emit("exit", exit_code=exit_code, code=code, stage=_current["stage"])


def parse(line: str) -> dict[str, Any] | None:
    """Inverse of :func:`emit`, for the service layer and tests."""
    _, sep, rest = line.partition(f"{MARKER} ")
    if not sep:
        return None
    try:
        payload = json.loads(rest)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) and "event" in payload else None
