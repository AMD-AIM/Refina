# Independent installation

`refina` declares its framework dependencies as versioned extras and does not
import repositories next to its checkout.

## Development checkout

```bash
git clone <repository>
cd refina
make dev PROFILE=rocm714
```

The framework versions are pinned as extras in `pyproject.toml`:

- `trl`: `trl>=1.17,<1.18` from PyPI
- `easydistill`: the upstream `v2.0.0` tag, installed from Git because the
  `easydistill` name on PyPI is a 0.0.1 placeholder

`make dev` delegates environment creation to `scripts/setup_envs.py`, which
installs `.[trl]` into the TRL environment and `.[easydistill]` into the
EasyDistill environment. The script creates a lightweight core environment and
one environment per heavy engine:

```text
.venv/
.venvs/swift/
.venvs/trl/
.venvs/easydistill/
```

The container wrapper uses `.venvs/core` for its core interpreter so a
container-created environment never overwrites a host developer's `.venv`.

The environments use `--system-site-packages` so the ROCm Torch supplied by the
base image is reused. A generated constraint file prevents pip from replacing
Torch or vLLM while resolving engine dependencies.

Install only selected engines with:

```bash
make dev PROFILE=rocm714 ENGINES=trl
```

For an offline installation, build the direct dependency wheelhouse on a
networked machine with the same Python version, then invoke:

```bash
make wheelhouse PROFILE=rocm714
python3 scripts/setup_envs.py --profile rocm714 --offline
```

The wheelhouse intentionally excludes Torch, vLLM, ROCm libraries, and
transitive packages already supplied by the base image.

## Updating framework dependencies

Updates are explicit. Change the `trl` or `easydistill` extra in
`pyproject.toml` (a version range for TRL, a reviewed tag for EasyDistill),
then run `make check` and the multi-GPU smoke suite before accepting it.

## Runtime selection

The public CLI detects `.venvs/<engine>/bin/python` and relaunches the complete
command once in that environment. This preserves the existing CLI and
Accelerate/torchrun behavior while isolating conflicting TRL and ms-swift
dependency graphs. Set `REFINA_ENVS_ROOT` when the environments are mounted
at a different path.

Run diagnostics with:

```bash
refina doctor
refina doctor --engine trl --json
```
