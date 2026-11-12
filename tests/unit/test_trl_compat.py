"""Pin the TRL interface Refina relies on. Skipped when trl is not installed."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest
import yaml

trl = pytest.importorskip("trl")

from packaging.version import Version  # noqa: E402

from refina.engines import trl as trl_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (ROOT / "configs/trl/trl_distill.yaml", ROOT / "configs/smoke/trl_distill.yaml")


def _config_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(trl.DistillationConfig)}


def test_trl_version_in_supported_range():
    version = Version(trl.__version__)
    assert Version("1.17") <= version < Version("1.18")


def test_distillation_config_accepts_engine_fields():
    fields = _config_fields()
    assert trl_engine.ENGINE_CONFIG_FIELDS <= fields
    for path in CONFIGS:
        section = yaml.safe_load(path.read_text(encoding="utf-8"))["trl"]
        assert set(section) <= fields, path


def test_prepare_fsdp_still_importable():
    from trl.models.utils import prepare_fsdp

    assert list(inspect.signature(prepare_fsdp).parameters)[:2] == ["model", "accelerator"]


def test_generation_hook_still_exists():
    # The engine wraps this method in accelerator.autocast(); a rename would silently drop that.
    assert callable(getattr(trl.DistillationTrainer, "_generate_single_turn", None))
