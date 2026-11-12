#!/usr/bin/env python3
"""In-container smoke for the refina image.

Runs, in order: the shell tools the RC wrapper needs, ``refina doctor`` for every
engine, then ROCm Torch, the GPU (unless ``--device cpu``), the engine's framework
imports and an importable refina CLI in every venv. Each check prints
``IMAGE_SMOKE <name> exit=<n> seconds=<s>``; the last line is
``IMAGE_SMOKE result=PASS`` or ``IMAGE_SMOKE result=FAIL``.

    python scripts/image_smoke.py --device 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ENGINES = ("swift", "trl", "easydistill")
VENVS = ("core", *ENGINES)
SHELL_TOOLS = ("setsid", "find", "head")


@dataclass
class Check:
    name: str
    argv: list[str]
    expect: Callable[[], bool] | None = None


# The imports each engine performs at run time; doctor does not exercise them, and
# ms-swift is installed --no-deps, so a missing dependency only shows up here.
FRAMEWORK_IMPORTS = {
    "swift": ("from swift.pipelines import sft_main", "from swift.arguments import SftArguments"),
    "trl": ("from trl import DistillationConfig, DistillationTrainer",),
    "easydistill": ("from easydistill.cli.main import _JOB_DISPATCH",),
}


def _torch_code(venv: str, device: str) -> str:
    lines = ["import torch", "import refina.cli", *FRAMEWORK_IMPORTS.get(venv, ())]
    lines += [
        "assert torch.version.hip, f'not a ROCm build: {torch.__version__}'",
        "ok = torch.cuda.is_available()",
        "name = torch.cuda.get_device_name(0) if ok else '-'",
        f"print({venv!r}, torch.__version__, torch.version.hip, ok, name)",
    ]
    if device != "cpu":
        lines += [
            "assert torch.cuda.is_available(), 'no GPU visible'",
            "assert torch.ones(4, device='cuda').sum().item() == 4",
        ]
    return "\n".join(lines)


def build_checks(device: str, envs_root: Path) -> list[Check]:
    doctor = ["refina", "doctor"]
    for engine in ENGINES:
        doctor += ["--engine", engine]
    checks = [
        Check(
            "shell-tools",
            ["sh", "-c", f'for t in {" ".join(SHELL_TOOLS)}; do command -v "$t" || exit 1; done'],
        ),
        Check("doctor", doctor),
    ]
    for venv in VENVS:
        python = envs_root / venv / "bin" / "python"
        checks.append(Check(f"torch:{venv}", [str(python), "-c", _torch_code(venv, device)]))
    return checks


def run_checks(checks: list[Check]) -> int:
    failed = False
    for check in checks:
        print("+", " ".join(check.argv), flush=True)
        started = time.monotonic()
        try:
            code = subprocess.run(check.argv).returncode  # noqa: S603
        except OSError as exc:
            print(f"IMAGE_SMOKE {check.name} {exc}", flush=True)
            code = 127
        seconds = f"seconds={time.monotonic() - started:.1f}"
        if code == 0 and check.expect is not None and not check.expect():
            print(f"IMAGE_SMOKE {check.name} exit=0 artifact=missing {seconds}", flush=True)
            failed = True
            continue
        print(f"IMAGE_SMOKE {check.name} exit={code} {seconds}", flush=True)
        failed = failed or code != 0
    print(f"IMAGE_SMOKE result={'FAIL' if failed else 'PASS'}", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    args = parser.parse_args(argv)
    envs_root = Path(os.environ.get("REFINA_ENVS_ROOT", "/opt/refina/.venvs"))
    return run_checks(build_checks(args.device, envs_root))


if __name__ == "__main__":
    raise SystemExit(main())
