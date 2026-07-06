# RunPod Deployment Design

**Date:** 2026-07-05  
**Status:** Approved  
**Scope:** Remote GPU training on RunPod.io with artifact retrieval

---

## Overview

Add a `runpod/` module to MLProjects that makes it easy to provision RunPod GPU pods,
push code and data, run training, and retrieve results. Designed as modular primitives
that compose into full automation or can be called individually for fine-grained control.

---

## Module Structure

New top-level `runpod/` package alongside `time_series/` and `common/`:

```
runpod/
    __init__.py
    config.py          # RunPodConfig dataclass; loaded from YAML + env
    pod.py             # pod lifecycle: create, status, stop, terminate, list
    sync.py            # rsync primitives: push-code, push-data, pull results
    remote.py          # SSH command execution on a pod
    environment.py     # bootstrap: pip install remote deps, start/stop MLflow server
    cli.py             # Typer CLI entry point
```

New config files:
- `runpod_config.yaml.example` — committed template
- `runpod_config.yaml` — actual config, gitignored (API key never stored here)

---

## CLI

Invoked as `python -m runpod <subcommand>` (or `uv run python -m runpod` locally).

### Pod lifecycle
```
runpod pod create [--gpu-type STR] [--gpu-count N] [--image STR]  # prints pod-id
runpod pod list
runpod pod status <pod-id>
runpod pod stop <pod-id>
runpod pod terminate <pod-id>
```

### Sync
```
runpod sync push-code <pod-id>                      # rsync source files only
runpod sync push-data <pod-id> --dataset <name>     # rsync data/<name>/ to pod
runpod sync pull <pod-id>                           # rsync mlruns/ back to local
```

`push-code` syncs: `runpod/`, `time_series/`, `common/`, `pyproject.toml`.
Excludes: `data/`, `mlruns/`, `.venv/`, `__pycache__/`, `.git/`.

rsync uses incremental transfer — unchanged files are not re-sent. Re-syncing a
dataset already on the pod is cheap.

### Environment (idempotent)
```
runpod env setup <pod-id>          # pip install remote deps into system Python
runpod env mlflow start <pod-id>   # start MLflow server on pod (localhost:5000)
runpod env mlflow stop <pod-id>
```

### Training
```
runpod train <pod-id> -- <train-command>
```

The CLI strips a leading `uv run` from the train command (a no-op on the pod) and
warns if `uv run` appears elsewhere. The command is executed as:
```
cd /workspace/mlprojects && MLFLOW_TRACKING_URI=http://localhost:5000 <train-command>
```

### Full pipeline — Mode A (single run)
```
runpod run [--dataset <name>] [--on-complete terminate|stop|keep] -- <train-command>
```
Executes: `pod create → push-code → push-data → env setup → mlflow start → train → pull → on-complete action`

Pod is left running if `pull` or `train` fails, with a warning message showing the pod ID.

### Sweep mode — Mode A (parallel runs)
```
runpod sweep create-mlflow-pod                               # persistent CPU pod for MLflow
runpod sweep run --mlflow-pod <id> [--n-pods N] [--dataset <name>] -- <train-command>
runpod sweep pull <mlflow-pod-id>                            # pull mlruns/ from CPU pod
runpod sweep teardown <mlflow-pod-id>
```

---

## Remote Environment

**Base image:** RunPod PyTorch image (CUDA + PyTorch pre-installed in system Python).
We do **not** create a venv or use uv on the pod — this avoids reinstalling the
CUDA-linked PyTorch from PyPI.

**Bootstrap steps** (`env setup`):
1. Verify SSH connectivity
2. `pip install` the `remote` dependency group (all non-torch deps)
3. Verify CUDA is visible: `python -c "import torch; assert torch.cuda.is_available()"`

**Remote dependency group** in `pyproject.toml`:
```toml
[dependency-groups]
remote = [
    "mlflow>=3.12.0",
    "optuna>=4.9.0",
    "pandas>=2.2.0,<3",
    "numpy>=2.4.6",
    "scikit-learn>=1.8.0",
    "scipy>=1.17.1",
    "tqdm>=4.67.3",
    "kaggle>=2.1.2",
]
# torch / torchvision / torchaudio intentionally excluded — provided by pod image
```

This group must be kept in sync with `pyproject.toml` main deps when non-torch
packages are added.

---

## MLflow Modes

### Mode 1 — Simple training (single GPU pod)

MLflow server runs on `localhost:5000` on the GPU pod itself. Training logs to it
via `MLFLOW_TRACKING_URI=http://localhost:5000`. After training, `sync pull` rsyncs
`mlruns/` back to the local machine for viewing in the local MLflow UI.

No ports exposed publicly. No changes to `runners.py` or any training code —
`MLFLOW_TRACKING_URI` is injected as an env var by the remote runner.

### Mode 2 — Sweeps / parallel runs (CPU MLflow pod + N GPU pods)

A persistent cheap CPU pod runs MLflow server. Its port 5000 is exposed via
RunPod's HTTP proxy (`https://<pod-id>-5000.proxy.runpod.net`). GPU training pods
set `MLFLOW_TRACKING_URI` to that URL and log in real time.

After the sweep, `runpod sweep pull <mlflow-pod-id>` rsyncs `mlruns/` from the CPU
pod to local. The CPU pod can be left running between sweeps to accumulate results,
then torn down when no longer needed.

---

## Config Schema

**`runpod_config.yaml.example`:**
```yaml
# API key: always from env var RUNPOD_API_KEY — never put it here

gpu_type: "NVIDIA GeForce RTX 4090"
gpu_count: 1
docker_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
mlflow_cpu_image: "runpod/base:0.6.2-cpu"

pod_name_prefix: "mlprojects"
remote_project_dir: "/workspace/mlprojects"
remote_data_dir: "/workspace/data"
remote_mlruns_dir: "/workspace/mlruns"
local_mlruns_dir: "mlruns"

default_datasets: []  # e.g. ["store-sales-time-series-forecasting"]

on_complete: "terminate"   # terminate | stop | keep
mlflow_port: 5000
```

**Config layering** (later layers override earlier):
1. `runpod_config.yaml` (base defaults)
2. Alternate YAML via `--config <path>`
3. Individual CLI flags (`--gpu-type`, `--on-complete`, `--dataset`, etc.)

`RunPodConfig` is a dataclass in `config.py`. `RUNPOD_API_KEY` is read from env at
load time and fails fast with a clear message if missing.

---

## Error Handling

- **Fail fast at each primitive.** Each operation raises immediately on failure with
  the RunPod API error or rsync/SSH exit code + stderr.
- **No automatic retries.** These are expensive operations; silent retries would mask
  real problems.
- **Pod safety on failure:** in `runpod run` / `runpod sweep run`, the `on_complete`
  action (terminate/stop) only fires if *all* preceding steps succeed. If `train` or
  `pull` fails, the pod is left running and the pod ID is printed so you can SSH in
  to debug.
- **`uv run` guard:** if the train command contains `uv run`, the CLI strips it from
  the leading position and emits a warning if found elsewhere.

---

## Files Changed

### New
- `runpod/__init__.py`
- `runpod/config.py`
- `runpod/pod.py`
- `runpod/sync.py`
- `runpod/remote.py`
- `runpod/environment.py`
- `runpod/cli.py`
- `runpod_config.yaml.example`

### Modified
- `pyproject.toml` — add `remote` dependency group; add `runpod` SDK + `typer` + `pyyaml` to main deps
- `.gitignore` (or equivalent) — add `runpod_config.yaml`
- `docs/store_sales_project.md` — add remote training section

### Not changed
- `time_series/store_sales/runners.py` — no modifications needed
- Any training entrypoint scripts — no modifications needed
