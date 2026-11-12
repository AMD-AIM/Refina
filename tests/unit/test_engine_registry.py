from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from refina import params as P
from refina.cli import main
from refina.engines import registry
from refina.engines.base import Engine, RunResult
from refina.engines.spec import ENTRY_POINT_GROUP, DoctorCheck, EngineSpec
from refina.errors import ExitCode, UsageError

BUILTIN = {"easydistill", "swift", "trl", "easydistill-swift", "noop"}


class DummyEngine(Engine):
    name = "dummy"

    def preflight(self, ctx):
        pass

    def run(self, ctx):
        return RunResult()


SPEC = EngineSpec(
    name="dummy",
    factory="tests.unit.test_engine_registry:DummyEngine",
    framework="test",
    mode="test",
    input="none",
    defaults={},
)


class FakeEntryPoint:
    def __init__(self, name, target):
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


@pytest.fixture(autouse=True)
def _fresh_registry():
    registry.specs.cache_clear()
    yield
    registry.specs.cache_clear()


def _install(monkeypatch, *extra, keep_builtin=True):
    real = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)) if keep_builtin else []

    def fake(**kwargs):
        assert kwargs == {"group": ENTRY_POINT_GROUP}
        return [*real, *extra]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)


def test_builtin_engines_come_from_entry_points():
    assert set(registry.specs()) == BUILTIN


def test_builtin_engines_match_params():
    assert registry.available() == P.ENGINES


def test_optional_plugins_are_marked():
    optional = {name for name, spec in registry.specs().items() if spec.optional}
    assert optional == {"swift", "easydistill", "easydistill-swift"}
    assert registry.spec("noop").internal is True


def test_specs_do_not_import_frameworks():
    code = (
        "import sys, refina.engines.registry as r; r.specs(); "
        "assert not {'torch', 'trl', 'swift', 'easydistill'} & set(sys.modules), "
        "sorted({'torch', 'trl', 'swift', 'easydistill'} & set(sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_spec_tables_match_previous_values():
    assert registry.spec("trl").doctor == DoctorCheck("trl", "trl", "accelerate")
    assert registry.spec("swift").doctor == DoctorCheck("swift", "ms-swift", "swift")
    assert registry.spec("easydistill").doctor == DoctorCheck("easydistill", "easydistill", None)
    assert registry.spec("easydistill-swift").env == "easydistill"
    assert registry.spec("swift").dataset_purpose == "sft"
    assert registry.spec("trl").dataset_purpose == "prompt"
    assert registry.spec("easydistill").dataset_purpose == "seed"
    assert registry.spec("easydistill").framework == "EasyDistill"
    assert registry.spec("noop").env is None


def test_unknown_engine_lists_choices():
    with pytest.raises(UsageError, match=r"unknown engine 'nope'; expected one of easydistill"):
        registry.spec("nope")


def test_third_party_plugin_is_listed(monkeypatch):
    _install(monkeypatch, FakeEntryPoint("dummy", SPEC))
    assert registry.available() == (*P.ENGINES, "dummy")
    assert isinstance(registry.load("dummy"), DummyEngine)


def test_spec_name_must_match_entry_point(monkeypatch):
    _install(monkeypatch, FakeEntryPoint("a", replace(SPEC, name="b")))
    with pytest.raises(UsageError, match="'a'.*'b'"):
        registry.specs()


def test_missing_entry_points_explains_install(monkeypatch):
    _install(monkeypatch, keep_builtin=False)
    with pytest.raises(UsageError, match=r"pip install -e \."):
        registry.specs()


def test_broken_plugin_does_not_hide_others(monkeypatch, capsys):
    _install(monkeypatch, FakeEntryPoint("broken", ImportError("no module named 'missing_dep'")))
    assert BUILTIN <= set(registry.available())
    with pytest.raises(UsageError, match="missing_dep"):
        registry.spec("broken")

    assert main(["--engines"]) == ExitCode.OK
    engines = yaml.safe_load(capsys.readouterr().out)["engines"]
    assert engines["broken"]["plugin"] == "error"
    assert "missing_dep" in engines["broken"]["error"]
    assert engines["trl"]["plugin"] == "core"


def test_pyproject_pins_trl_and_easydistill():
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")
    root = Path(__file__).resolve().parents[2]
    extras = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert extras["trl"] == ["trl>=1.17,<1.18"]
    assert extras["easydistill"] == [
        "easydistill @ git+https://github.com/modelscope/easydistill@v2.0.0"
    ]


def test_doctor_reports_no_submodules(tmp_path, monkeypatch):
    from refina import doctor

    monkeypatch.setenv("REFINA_ENVS_ROOT", str(tmp_path))
    names = [check.name for check in doctor.collect()]
    assert not [name for name in names if name.startswith("submodule:")]
