"""scripts/image_smoke.py: the check list and the summary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENGINES = ("swift", "trl", "easydistill")


@pytest.fixture(scope="module")
def smoke():
    path = REPO / "scripts" / "image_smoke.py"
    spec = importlib.util.spec_from_file_location("image_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _checks(smoke, tmp_path: Path, device: str = "0"):
    checks = smoke.build_checks(device, tmp_path)
    return {check.name: check for check in checks}, checks


def test_build_checks_uses_the_required_values(smoke, tmp_path):
    by_name, checks = _checks(smoke, tmp_path)
    assert [check.name for check in checks] == [
        "shell-tools",
        "doctor",
        "torch:core",
        "torch:swift",
        "torch:trl",
        "torch:easydistill",
    ]
    doctor = ["refina", "doctor"]
    for engine in ENGINES:
        doctor += ["--engine", engine]
    assert by_name["doctor"].argv == doctor

    assert "from swift.pipelines import sft_main" in by_name["torch:swift"].argv[-1]
    assert "from trl import DistillationConfig" in by_name["torch:trl"].argv[-1]
    assert "from easydistill.cli.main import" in by_name["torch:easydistill"].argv[-1]
    for name in ("torch:core", "torch:swift"):
        assert "assert torch.cuda.is_available()" in by_name[name].argv[-1]
    for name in (f"torch:{venv}" for venv in ("core", *ENGINES)):
        assert "import refina.cli" in by_name[name].argv[-1]
    assert by_name["torch:trl"].argv[0] == str(tmp_path / "trl" / "bin" / "python")


def test_cpu_checks_do_not_require_a_gpu(smoke, tmp_path):
    by_name, _ = _checks(smoke, tmp_path, device="cpu")
    assert "assert torch.cuda.is_available()" not in by_name["torch:core"].argv[-1]


def test_run_checks_reports_each_check_and_fails_if_any_fails(smoke, capsys):
    result = smoke.run_checks([smoke.Check("a", ["true"]), smoke.Check("b", ["false"])])
    out = capsys.readouterr().out
    assert result == 1
    assert "IMAGE_SMOKE a exit=0" in out
    assert "IMAGE_SMOKE b exit=1" in out
    assert "IMAGE_SMOKE result=FAIL" in out


def test_missing_artifact_fails_a_zero_exit(smoke, capsys):
    result = smoke.run_checks([smoke.Check("c", ["true"], expect=lambda: False)])
    out = capsys.readouterr().out
    assert result == 1
    assert "IMAGE_SMOKE c exit=0 artifact=missing" in out
    assert out.splitlines()[-1] == "IMAGE_SMOKE result=FAIL"


def test_all_passing_checks_pass(smoke, capsys):
    assert smoke.run_checks([smoke.Check("d", ["true"], expect=lambda: True)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "IMAGE_SMOKE result=PASS"
