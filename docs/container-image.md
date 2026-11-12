# Container image

## What it is

`ccr.ccs.tencentyun.com/radeon/refina` is the gfx1100 (Radeon PRO W7900 /
W7900D) training image for `refina`. It is built from `docker/Dockerfile`
by `scripts/build-image.sh`, from a clean checkout, so every tag maps to one
commit of this repository and one base image digest. The image holds the core
CLI and one venv per engine:

```text
/opt/refina/.venvs/core          refina CLI (on PATH)
/opt/refina/.venvs/swift         ms-swift
/opt/refina/.venvs/trl           TRL (trl>=1.17,<1.18)
/opt/refina/.venvs/easydistill   EasyDistill (v2.0.0 Git tag)
```

It is a single-GPU image: one container, one visible GPU.

## Base

```text
docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
(rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0)
```

Ubuntu 24.04, Python 3.12, ROCm 7.2.4 (HIP 7.2.53211), and in `/opt/venv`:
torch `2.10.0+rocm7.2.4.lw.git3d3aa833`, torchvision
`0.25.0+rocm7.2.4.git82df5f59`, torchaudio `2.10.0+rocm7.2.4.git5047768f`,
triton `3.6.0+rocm7.2.4.git4ed88892`. The base is pinned by digest in
`ARG BASE_IMAGE`; override it with `BASE_IMAGE=... scripts/build-image.sh`.

The previous image was built on a personal-registry vLLM image with a ROCm
7.14 nightly and torch `2.13.0a0`. That base has no build recipe of its own,
and refina does not use vLLM, so this image uses AMD's released ROCm
PyTorch image instead.

`/opt/venv` is itself a venv. `scripts/setup_envs.py` writes
`_refina_base.pth` into every engine venv so that the engine venvs see
`/opt/venv` (and its ROCm Torch) and not only the system interpreter, and pins
torch, torchvision, torchaudio and triton in `.distill/base-constraints.txt`,
so pip never replaces the image's Torch.

## Engines and versions

Each venv is installed from its requirements file:

| venv | requirements |
|---|---|
| core | `pyproject.toml` (`.[dev,dataset,hub]`) |
| swift | `requirements/rocm714-swift.txt` + `ms-swift==4.5.2` (no deps) |
| trl | `requirements/rocm714-trl.txt` + `.[trl]` |
| easydistill | `requirements/rocm714-easydistill.txt` + `.[easydistill]` |

The exact versions are frozen into the image at build time:

```bash
IMG=ccr.ccs.tencentyun.com/radeon/refina:<tag>
docker run --rm --entrypoint cat $IMG /opt/refina/.distill/freeze-<venv>.txt
docker run --rm --entrypoint cat $IMG /opt/refina/.distill/base-constraints.txt
docker run --rm $IMG doctor
```

`<venv>` is one of `core`, `swift`, `trl`, `easydistill`.

## Runtime contract

| | |
|---|---|
| `Entrypoint` | `["refina"]` |
| `Cmd` | `["--help"]` |
| `WorkingDir` | `/opt/refina` |
| `Env` | `REFINA_ENVS_ROOT=/opt/refina/.venvs`, `PATH=/opt/refina/.venvs/core/bin:/opt/venv/bin:/opt/rocm/bin:...`, `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, `PYTHONUNBUFFERED=1` |
| Labels | `io.amd.rocm.arch=gfx1100`, `refina.gpu-mode=single-gpu`, `org.opencontainers.image.revision` (full commit), `org.opencontainers.image.base.name` |

The venvs record their own path in `.refina-runtime`; they only work at
`/opt/refina/.venvs`, so do not move them.

The image deliberately sets no `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
`CUDA_VISIBLE_DEVICES`, `HOME`, `WORK_DIR` or `OUTPUT_DIR`. Which GPU a
container sees is decided by whoever starts it (the Kubernetes device plugin,
or `docker run --device`); a device mask baked into the image hides that GPU
whenever it is not the one the image was built next to. Home, work and output
directories belong to the caller as well.

## Tag rule

```text
<git rev-parse --short=12 HEAD>-gfx1100
```

There is no `latest` tag. Deploy by digest
(`ccr.ccs.tencentyun.com/radeon/refina@sha256:...`); the tag only tells
you which commit the image came from.

## Build and push

From a clean checkout:

```bash
scripts/build-image.sh             # build, tag, push, record the digest
scripts/build-image.sh --no-push   # build and tag only
```

The script refuses to build with uncommitted changes or untracked files (exit
1). On success the last lines of stdout are:

```text
IMAGE=ccr.ccs.tencentyun.com/radeon/refina:<tag>
DIGEST=ccr.ccs.tencentyun.com/radeon/refina@sha256:<digest>
```

and a line `time  tag  commit  digest  base` is appended to
`build/image-digests.tsv`.

Registry login: the script reuses an existing `docker login`. Otherwise set
`CCR_PASSWORD` in the environment (the CCR username defaults to
`100053684421`, override with `CCR_USERNAME`); it is passed only through
`docker login --password-stdin`, never as an argument or build arg. The
password itself is kept in the team's cluster notes, not in this repository.

Other knobs: `IMAGE_REPO`, `BASE_IMAGE`, `ENGINES`, `PIP_INDEX_URL` (default
`https://mirrors.aliyun.com/pypi/simple`).

## Pull

The `radeon` namespace on CCR allows anonymous pulls, so no `docker login` is needed:

```bash
docker pull ccr.ccs.tencentyun.com/radeon/refina:<tag>
```

Deployments should reference the digest printed by `scripts/build-image.sh` (also in
`build/image-digests.tsv` on the build host), e.g.
`ccr.ccs.tencentyun.com/radeon/refina@sha256:<digest>`.

| Tag | Digest |
|---|---|
| `d15c0acb08bb-gfx1100` | `sha256:471e8b6a1e10e66e7f50b67bce4ede360ad5166e9fb02d32ddafcca0fe1d42c1` |

## Smoke

`scripts/image_smoke.py` runs inside the image and checks: the shell tools the
platform wrapper uses (`setsid`, `find`, `head`); `refina doctor` for every
engine; ROCm Torch, the GPU, the engine's framework imports and an importable
CLI in every venv.

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  --entrypoint /opt/refina/.venvs/core/bin/python $IMG \
  /opt/refina/scripts/image_smoke.py --device 0
```

Each check prints `IMAGE_SMOKE <name> exit=<n> seconds=<s>`; the last line is
`IMAGE_SMOKE result=PASS` (exit 0) or `IMAGE_SMOKE result=FAIL` (exit 1). Use
`--device cpu` on a machine without an AMD GPU.

## Licenses

refina is Apache-2.0. The engines keep their own licenses:

| package | license |
|---|---|
| TRL | Apache-2.0, <https://github.com/huggingface/trl/blob/main/LICENSE> |
| EasyDistill | Apache-2.0, <https://github.com/modelscope/easydistill/blob/main/LICENSE> |
| ms-swift | Apache-2.0, <https://github.com/modelscope/ms-swift/blob/main/LICENSE> |

The base image carries PyTorch (BSD-3-Clause) and ROCm under their own licenses.

## Differences from `amd-radeon/refina:v0.1`

| | `ccr.ccs.tencentyun.com/amd-radeon/refina:v0.1` | this image |
|---|---|---|
| how it was made | `docker commit` of a dev container, no recipe | `docker/Dockerfile` + `scripts/build-image.sh` |
| base | `vllm-rocm-w7900-modelscope:1.36.3` (personal Aliyun registry), Debian bookworm | `rocm/pytorch` by digest, Ubuntu 24.04 |
| ROCm / torch | 7.14 nightly / `2.13.0a0+rocm7.14.0a20260612` | 7.2.4 / `2.10.0+rocm7.2.4` |
| vLLM | `0.1.dev1+g3775d5fca.rocm714` | none (refina does not use it) |
| engines | swift, trl, easydistill | swift, trl, easydistill |
| Env | `ROCR_VISIBLE_DEVICES=2`, `HOME=/work/users/1001/home`, `WORK_DIR=/work/users/1001`, `OUTPUT_DIR=/output`, ROCm SDK paths | none of these |
| package lists | none | `/opt/refina/.distill/freeze-<venv>.txt` |
| TRL / EasyDistill versions | not recorded | `freeze-trl.txt` / `freeze-easydistill.txt` |

The core and engine venvs no longer contain what the vLLM base brought in
(vLLM and its serving stack, `transformers`/`peft` in the core venv,
`transformers`/`datasets` in the easydistill venv). EasyDistill needs
`transformers` only for its `local` (vLLM) extra.
