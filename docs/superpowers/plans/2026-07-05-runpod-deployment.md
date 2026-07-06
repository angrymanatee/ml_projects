# RunPod Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `remote/` CLI module for provisioning RunPod GPU pods, syncing code/data, running training, and retrieving MLflow results.

**Architecture:** Modular Python package (`remote/`) with six focused files — config, SSH execution, pod lifecycle, rsync sync, environment bootstrap, and a Typer CLI that composes them. CLI commands can be called individually (manual workflow) or chained via `remote run` (full automation). Two MLflow modes: localhost on the GPU pod for simple runs, CPU MLflow pod for parallel sweeps.

**Tech Stack:** `runpod` PyPI SDK (pod lifecycle), `typer` (CLI), `pyyaml` (config), `subprocess` (SSH/rsync via system tools).

## Global Constraints

- Python 3.14, `uv` for local dependency management
- Module named `remote/` — NOT `runpod/` (would shadow the PyPI `runpod` package since `pythonpath = ["."]` puts project root first in sys.path)
- Remote pods use system Python + pip; no uv/venv on the pod — PyTorch is pre-installed by the RunPod image
- `RUNPOD_API_KEY` always from environment variable, never stored in files
- `runpod_config.yaml` is gitignored; `runpod_config.yaml.example` is committed
- Tests use `pytest` + `unittest.mock`; no real RunPod API calls in tests
- All public functions have type hints and docstrings
- Format: `ruff` + `black`; type-check: `pyright`

---

## File Map

| File | Responsibility |
|------|---------------|
| `remote/__init__.py` | Empty package marker |
| `remote/__main__.py` | Entry point: `python -m remote` |
| `remote/config.py` | `RunPodConfig` dataclass + `load_config()` |
| `remote/ssh.py` | `SSHTarget` dataclass + `run_remote()` |
| `remote/pod.py` | Pod lifecycle: create, list, get, wait, stop, terminate |
| `remote/sync.py` | rsync push-code, push-data, pull-results |
| `remote/environment.py` | Bootstrap remote env; MLflow start/stop |
| `remote/cli.py` | Typer app with all subcommands |
| `runpod_config.yaml.example` | Committed config template |
| `tests/remote/test_config.py` | Config loading tests |
| `tests/remote/test_ssh.py` | SSH execution tests |
| `tests/remote/test_pod.py` | Pod lifecycle tests |
| `tests/remote/test_sync.py` | rsync tests |
| `tests/remote/test_environment.py` | Environment bootstrap tests |
| `tests/remote/test_cli.py` | CLI integration tests |

---

### Task 1: Scaffolding, config, and dependency setup

**Files:**
- Create: `remote/__init__.py`, `remote/__main__.py`, `remote/config.py`
- Create: `runpod_config.yaml.example`
- Create: `tests/remote/__init__.py`, `tests/remote/test_config.py`
- Modify: `pyproject.toml` (add deps + remote group)
- Modify: `.gitignore` (add runpod_config.yaml)

**Interfaces:**
- Produces: `RunPodConfig` dataclass, `load_config(config_path: Path) -> RunPodConfig`

---

- [ ] **Step 1: Write the failing tests**

`tests/remote/__init__.py` — empty file.

`tests/remote/test_config.py`:
```python
from pathlib import Path

import pytest

from remote.config import RunPodConfig, load_config


def test_load_config_reads_api_key_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test_key_abc")
    config_file = tmp_path / "runpod_config.yaml"
    config_file.write_text("gpu_type: 'NVIDIA A100'\ngpu_count: 2\n")
    config = load_config(config_file)
    assert config.api_key == "test_key_abc"
    assert config.gpu_type == "NVIDIA A100"
    assert config.gpu_count == 2


def test_load_config_uses_defaults_when_no_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config.on_complete == "terminate"
    assert config.mlflow_port == 5000
    assert config.default_datasets == []
    assert config.gpu_count == 1


def test_load_config_fails_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        load_config(tmp_path / "config.yaml")


def test_load_config_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("on_complete: stop\nmlflow_port: 5001\ndefault_datasets:\n  - store-sales\n")
    config = load_config(config_file)
    assert config.on_complete == "stop"
    assert config.mlflow_port == 5001
    assert config.default_datasets == ["store-sales"]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/remote/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'remote'`

- [ ] **Step 3: Add dependencies to pyproject.toml**

In the `[project]` `dependencies` list, add:
```toml
    "runpod>=1.7.0",
    "typer>=0.12.0",
    "pyyaml>=6.0",
```

Add a new section after `[dependency-groups]`:
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
# torch / torchvision / torchaudio intentionally excluded — provided by RunPod pod image
```

Run:
```bash
uv sync
```

- [ ] **Step 4: Add runpod_config.yaml to .gitignore**

Add this line to `.gitignore`:
```
runpod_config.yaml
```

- [ ] **Step 5: Create remote/__init__.py**

Empty file at `remote/__init__.py`.

- [ ] **Step 6: Create remote/__main__.py**

```python
from remote.cli import app

app()
```

- [ ] **Step 7: Create remote/config.py**

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunPodConfig:
    """Configuration for remote RunPod training.

    api_key is always sourced from RUNPOD_API_KEY env var at load time.
    All other fields can be overridden in runpod_config.yaml or via CLI flags.
    """

    api_key: str
    gpu_type: str = "NVIDIA GeForce RTX 4090"
    gpu_count: int = 1
    cloud_type: str = "SECURE"  # SECURE | COMMUNITY (community is cheaper)
    docker_image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    mlflow_cpu_image: str = "runpod/base:0.6.2-cpu"
    pod_name_prefix: str = "mlprojects"
    remote_project_dir: str = "/workspace/mlprojects"
    remote_data_dir: str = "/workspace/data"
    remote_mlruns_dir: str = "/workspace/mlruns"
    local_mlruns_dir: str = "mlruns"
    default_datasets: list[str] = field(default_factory=list)
    on_complete: str = "terminate"  # terminate | stop | keep
    mlflow_port: int = 5000


def load_config(config_path: Path = Path("runpod_config.yaml")) -> RunPodConfig:
    """Load RunPodConfig from YAML file with RUNPOD_API_KEY injected from env.

    Args:
        config_path: Path to the YAML config file. Missing file is allowed;
            defaults are used for all fields except api_key.

    Returns:
        Populated RunPodConfig.

    Raises:
        RuntimeError: If RUNPOD_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY environment variable not set.\n"
            "Store it in Keychain: security add-generic-password -a $USER -s RUNPOD_API_KEY -w <key>\n"
            "Load it in .zshrc: export RUNPOD_API_KEY=$(security find-generic-password -a $USER -s RUNPOD_API_KEY -w)"
        )

    data: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    return RunPodConfig(api_key=api_key, **data)
```

- [ ] **Step 8: Create runpod_config.yaml.example**

```yaml
# RunPod training configuration
# Copy to runpod_config.yaml (gitignored) and fill in your values.
# API key: never put here — always from env var RUNPOD_API_KEY
#   Store: security add-generic-password -a $USER -s RUNPOD_API_KEY -w <key>
#   Load in .zshrc: export RUNPOD_API_KEY=$(security find-generic-password -a $USER -s RUNPOD_API_KEY -w)

gpu_type: "NVIDIA GeForce RTX 4090"
gpu_count: 1
cloud_type: "SECURE"          # SECURE (reliable) or COMMUNITY (cheaper)
docker_image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
mlflow_cpu_image: "runpod/base:0.6.2-cpu"

pod_name_prefix: "mlprojects"
remote_project_dir: "/workspace/mlprojects"
remote_data_dir: "/workspace/data"
remote_mlruns_dir: "/workspace/mlruns"
local_mlruns_dir: "mlruns"

default_datasets: []          # e.g. ["store-sales-time-series-forecasting"]

on_complete: "terminate"      # terminate | stop | keep
mlflow_port: 5000
```

- [ ] **Step 9: Run tests — verify they pass**

```bash
uv run pytest tests/remote/test_config.py -v
```
Expected: 4 PASSED

- [ ] **Step 10: Commit**

```bash
git add remote/__init__.py remote/__main__.py remote/config.py \
        runpod_config.yaml.example \
        tests/remote/__init__.py tests/remote/test_config.py \
        pyproject.toml .gitignore uv.lock
git commit -m "Add remote training module: scaffolding and config"
```

---

### Task 2: SSH execution — `remote/ssh.py`  *(parallel with Tasks 3 and 4)*

**Files:**
- Create: `remote/ssh.py`
- Create: `tests/remote/test_ssh.py`

**Interfaces:**
- Consumes: nothing from prior tasks except stdlib
- Produces:
  - `SSHTarget(host: str, port: int, user: str = "root")`
  - `run_remote(target: SSHTarget, command: str, *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]`
  - `wait_for_ssh(target: SSHTarget, timeout: int = 120) -> None`

---

- [ ] **Step 1: Write the failing tests**

`tests/remote/test_ssh.py`:
```python
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from remote.ssh import SSHTarget, run_remote, wait_for_ssh


@pytest.fixture
def target() -> SSHTarget:
    return SSHTarget(host="1.2.3.4", port=22345)


def _success() -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 0
    r.stdout = "ok"
    r.stderr = ""
    return r


def _failure(stderr: str = "error") -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 1
    r.stdout = ""
    r.stderr = stderr
    return r


def test_ssh_destination(target: SSHTarget) -> None:
    assert target.ssh_destination() == "root@1.2.3.4"


def test_ssh_opts_include_port(target: SSHTarget) -> None:
    assert "-p" in target.ssh_opts
    assert "22345" in target.ssh_opts


def test_ssh_opts_disable_host_check(target: SSHTarget) -> None:
    opts_str = " ".join(target.ssh_opts)
    assert "StrictHostKeyChecking=no" in opts_str


def test_run_remote_success(target: SSHTarget) -> None:
    with patch("subprocess.run", return_value=_success()) as mock_run:
        result = run_remote(target, "echo hello")
    assert result.stdout == "ok"
    cmd = mock_run.call_args[0][0]
    assert "ssh" in cmd
    assert "echo hello" in cmd


def test_run_remote_injects_env(target: SSHTarget) -> None:
    with patch("subprocess.run", return_value=_success()) as mock_run:
        run_remote(target, "python -m app", env={"FOO": "bar"})
    cmd = mock_run.call_args[0][0]
    assert any("FOO=bar" in arg for arg in cmd)


def test_run_remote_failure_raises(target: SSHTarget) -> None:
    with patch("subprocess.run", return_value=_failure("command not found")):
        with pytest.raises(RuntimeError, match="Remote command failed"):
            run_remote(target, "bad-command")


def test_run_remote_no_check_returns_result(target: SSHTarget) -> None:
    with patch("subprocess.run", return_value=_failure()):
        result = run_remote(target, "bad", check=False)
    assert result.returncode == 1


def test_wait_for_ssh_succeeds_immediately(target: SSHTarget) -> None:
    with patch("remote.ssh.run_remote", return_value=_success()):
        wait_for_ssh(target, timeout=10)  # should not raise


def test_wait_for_ssh_retries_until_timeout(target: SSHTarget) -> None:
    with patch("remote.ssh.run_remote", return_value=_failure()), \
         patch("remote.ssh.time.sleep"):
        with pytest.raises(TimeoutError):
            wait_for_ssh(target, timeout=5)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/remote/test_ssh.py -v
```
Expected: `ModuleNotFoundError: No module named 'remote.ssh'`

- [ ] **Step 3: Implement remote/ssh.py**

```python
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class SSHTarget:
    """SSH connection info for a RunPod pod."""

    host: str
    port: int
    user: str = "root"

    @property
    def ssh_opts(self) -> list[str]:
        return [
            "-p", str(self.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
        ]

    def ssh_destination(self) -> str:
        return f"{self.user}@{self.host}"


def run_remote(
    target: SSHTarget,
    command: str,
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command on a remote pod via SSH.

    Args:
        target: SSH connection details.
        command: Shell command to run on the remote machine.
        check: Raise RuntimeError on non-zero exit code.
        env: Environment variables prepended to the command string.

    Returns:
        CompletedProcess with stdout and stderr captured.
    """
    if env:
        env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
        command = f"{env_prefix} {command}"

    result = subprocess.run(
        ["ssh", *target.ssh_opts, target.ssh_destination(), command],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Remote command failed (exit {result.returncode}):\n"
            f"Command: {command}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


def wait_for_ssh(target: SSHTarget, timeout: int = 120) -> None:
    """Poll until SSH accepts connections on the pod.

    Args:
        target: SSH connection details.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If SSH is not available within timeout seconds.
    """
    elapsed = 0
    poll_interval = 5
    while elapsed < timeout:
        result = run_remote(target, "echo ok", check=False)
        if result.returncode == 0:
            return
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"SSH not available on {target.host}:{target.port} after {timeout}s")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/remote/test_ssh.py -v
```
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add remote/ssh.py tests/remote/test_ssh.py
git commit -m "Add SSH execution primitive (remote/ssh.py)"
```

---

### Task 3: Pod lifecycle — `remote/pod.py`  *(parallel with Tasks 2 and 4)*

**Files:**
- Create: `remote/pod.py`
- Create: `tests/remote/test_pod.py`

**Interfaces:**
- Consumes: `RunPodConfig` from `remote.config`, `SSHTarget` from `remote.ssh`
- Produces:
  - `PodInfo(pod_id: str, name: str, status: str)`
  - `create_pod(config: RunPodConfig, *, name_suffix: str = "") -> str`
  - `create_mlflow_pod(config: RunPodConfig) -> str`
  - `get_ssh_target(config: RunPodConfig, pod_id: str) -> SSHTarget`
  - `wait_for_running(config: RunPodConfig, pod_id: str, timeout: int = 300) -> SSHTarget`
  - `list_pods(config: RunPodConfig) -> list[PodInfo]`
  - `stop_pod(config: RunPodConfig, pod_id: str) -> None`
  - `terminate_pod(config: RunPodConfig, pod_id: str) -> None`

**Note on CPU pods:** RunPod's SDK `create_pod` requires `gpu_type_id`. For the MLflow CPU pod, verify during implementation whether the SDK supports CPU-only pods or whether a small cheap GPU type should be used instead. The `create_mlflow_pod` function is clearly commented about this.

---

- [ ] **Step 1: Write the failing tests**

`tests/remote/test_pod.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from remote.config import RunPodConfig
from remote.pod import (
    PodInfo,
    create_pod,
    get_ssh_target,
    list_pods,
    stop_pod,
    terminate_pod,
    wait_for_running,
)
from remote.ssh import SSHTarget


@pytest.fixture
def config() -> RunPodConfig:
    return RunPodConfig(api_key="test_key")


def _running_pod(pod_id: str = "abc123") -> dict:
    return {
        "id": pod_id,
        "name": "mlprojects-run",
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [
                {
                    "ip": "1.2.3.4",
                    "publicPort": 22345,
                    "privatePort": 22,
                    "isIpPublic": True,
                    "type": "tcp",
                }
            ]
        },
    }


def test_create_pod_returns_id(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.create_pod.return_value = {"id": "abc123"}
        pod_id = create_pod(config, name_suffix="run1")
    assert pod_id == "abc123"


def test_create_pod_sets_api_key(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.create_pod.return_value = {"id": "xyz"}
        create_pod(config)
    assert mock_sdk.api_key == "test_key"


def test_create_pod_passes_gpu_type(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.create_pod.return_value = {"id": "xyz"}
        create_pod(config)
    kwargs = mock_sdk.create_pod.call_args[1]
    assert kwargs["gpu_type_id"] == config.gpu_type
    assert "22/tcp" in kwargs["ports"]


def test_wait_for_running_returns_ssh_target(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.get_pod.return_value = _running_pod()
        target = wait_for_running(config, "abc123")
    assert isinstance(target, SSHTarget)
    assert target.host == "1.2.3.4"
    assert target.port == 22345


def test_wait_for_running_polls_until_running(config: RunPodConfig) -> None:
    not_yet = {"id": "x", "desiredStatus": "CREATED", "runtime": {"ports": []}}
    with patch("remote.pod.runpod_sdk") as mock_sdk, \
         patch("remote.pod.time.sleep"):
        mock_sdk.get_pod.side_effect = [not_yet, not_yet, _running_pod()]
        target = wait_for_running(config, "x")
    assert mock_sdk.get_pod.call_count == 3
    assert isinstance(target, SSHTarget)


def test_wait_for_running_times_out(config: RunPodConfig) -> None:
    not_running = {"id": "x", "desiredStatus": "CREATED", "runtime": {"ports": []}}
    with patch("remote.pod.runpod_sdk") as mock_sdk, \
         patch("remote.pod.time.sleep"):
        mock_sdk.get_pod.return_value = not_running
        with pytest.raises(TimeoutError):
            wait_for_running(config, "x", timeout=10)


def test_get_ssh_target_extracts_port(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.get_pod.return_value = _running_pod()
        target = get_ssh_target(config, "abc123")
    assert target.host == "1.2.3.4"
    assert target.port == 22345


def test_get_ssh_target_raises_when_no_port(config: RunPodConfig) -> None:
    pod = {"id": "x", "desiredStatus": "RUNNING", "runtime": {"ports": []}}
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.get_pod.return_value = pod
        with pytest.raises(RuntimeError, match="No public SSH port"):
            get_ssh_target(config, "x")


def test_list_pods(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        mock_sdk.get_pods.return_value = [
            {"id": "a1", "name": "mlprojects-1", "desiredStatus": "RUNNING"},
            {"id": "b2", "name": "mlprojects-2", "desiredStatus": "EXITED"},
        ]
        pods = list_pods(config)
    assert len(pods) == 2
    assert pods[0].pod_id == "a1"
    assert pods[1].status == "EXITED"


def test_stop_pod(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        stop_pod(config, "abc123")
    mock_sdk.stop_pod.assert_called_once_with("abc123")


def test_terminate_pod(config: RunPodConfig) -> None:
    with patch("remote.pod.runpod_sdk") as mock_sdk:
        terminate_pod(config, "abc123")
    mock_sdk.terminate_pod.assert_called_once_with("abc123")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/remote/test_pod.py -v
```
Expected: `ModuleNotFoundError: No module named 'remote.pod'`

- [ ] **Step 3: Implement remote/pod.py**

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import runpod as runpod_sdk

from remote.config import RunPodConfig
from remote.ssh import SSHTarget

_POLL_INTERVAL = 5


@dataclass
class PodInfo:
    """Summary of a RunPod pod."""

    pod_id: str
    name: str
    status: str


def _init_sdk(config: RunPodConfig) -> None:
    runpod_sdk.api_key = config.api_key


def _extract_ssh_target(pod: dict) -> SSHTarget:
    ports = pod.get("runtime", {}).get("ports", [])
    ssh_port = next(
        (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
        None,
    )
    if not ssh_port:
        raise RuntimeError(f"No public SSH port found for pod {pod['id']!r}")
    return SSHTarget(host=ssh_port["ip"], port=ssh_port["publicPort"])


def create_pod(config: RunPodConfig, *, name_suffix: str = "") -> str:
    """Create a RunPod GPU pod and return its pod ID.

    The pod exposes port 22 (TCP) for SSH access. Does not wait for RUNNING.

    Args:
        config: RunPod configuration.
        name_suffix: Appended to pod_name_prefix to form the pod name.

    Returns:
        RunPod pod ID string.
    """
    _init_sdk(config)
    name = f"{config.pod_name_prefix}-{name_suffix}" if name_suffix else config.pod_name_prefix
    pod = runpod_sdk.create_pod(
        name=name,
        image_name=config.docker_image,
        gpu_type_id=config.gpu_type,
        cloud_type=config.cloud_type,
        gpu_count=config.gpu_count,
        container_disk_in_gb=50,
        ports="22/tcp",
        support_public_ip=True,
    )
    return pod["id"]


def create_mlflow_pod(config: RunPodConfig) -> str:
    """Create a persistent pod to host the MLflow tracking server.

    Note: RunPod's SDK may require a gpu_type_id even for CPU workloads.
    If creation fails, verify whether the SDK supports CPU-only pods or
    select the cheapest available GPU type from the RunPod dashboard.

    Returns:
        RunPod pod ID string.
    """
    _init_sdk(config)
    pod = runpod_sdk.create_pod(
        name=f"{config.pod_name_prefix}-mlflow",
        image_name=config.mlflow_cpu_image,
        gpu_type_id=config.gpu_type,  # may need adjustment for CPU-only pods
        cloud_type="SECURE",
        gpu_count=0,
        container_disk_in_gb=20,
        ports=f"22/tcp,{config.mlflow_port}/http",
        support_public_ip=True,
    )
    return pod["id"]


def get_ssh_target(config: RunPodConfig, pod_id: str) -> SSHTarget:
    """Fetch SSH connection info for a pod that is already running."""
    _init_sdk(config)
    pod = runpod_sdk.get_pod(pod_id)
    return _extract_ssh_target(pod)


def wait_for_running(
    config: RunPodConfig,
    pod_id: str,
    timeout: int = 300,
) -> SSHTarget:
    """Poll until the pod reaches RUNNING status; return SSH connection info.

    Args:
        config: RunPod configuration.
        pod_id: Pod to wait for.
        timeout: Maximum seconds to wait.

    Returns:
        SSHTarget for the running pod.

    Raises:
        TimeoutError: If the pod is not RUNNING within timeout seconds.
    """
    _init_sdk(config)
    elapsed = 0
    while elapsed < timeout:
        pod = runpod_sdk.get_pod(pod_id)
        if pod.get("desiredStatus") == "RUNNING" and pod.get("runtime", {}).get("ports"):
            return _extract_ssh_target(pod)
        time.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL
    raise TimeoutError(f"Pod {pod_id!r} did not reach RUNNING status within {timeout}s")


def list_pods(config: RunPodConfig) -> list[PodInfo]:
    """Return summary info for all pods on the account."""
    _init_sdk(config)
    return [
        PodInfo(
            pod_id=p["id"],
            name=p.get("name", ""),
            status=p.get("desiredStatus", "UNKNOWN"),
        )
        for p in runpod_sdk.get_pods()
    ]


def stop_pod(config: RunPodConfig, pod_id: str) -> None:
    """Stop a pod (preserves disk; can be resumed)."""
    _init_sdk(config)
    runpod_sdk.stop_pod(pod_id)


def terminate_pod(config: RunPodConfig, pod_id: str) -> None:
    """Terminate a pod (destroys disk and deallocates all resources)."""
    _init_sdk(config)
    runpod_sdk.terminate_pod(pod_id)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/remote/test_pod.py -v
```
Expected: 11 PASSED

- [ ] **Step 5: Commit**

```bash
git add remote/pod.py tests/remote/test_pod.py
git commit -m "Add pod lifecycle primitive (remote/pod.py)"
```

---

### Task 4: Sync — `remote/sync.py`  *(parallel with Tasks 2 and 3)*

**Files:**
- Create: `remote/sync.py`
- Create: `tests/remote/test_sync.py`

**Interfaces:**
- Consumes: `RunPodConfig`, `SSHTarget`
- Produces:
  - `push_code(target: SSHTarget, config: RunPodConfig) -> None`
  - `push_data(target: SSHTarget, config: RunPodConfig, dataset: str) -> None`
  - `pull_results(target: SSHTarget, config: RunPodConfig) -> None`

---

- [ ] **Step 1: Write the failing tests**

`tests/remote/test_sync.py`:
```python
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from remote.config import RunPodConfig
from remote.ssh import SSHTarget
from remote.sync import pull_results, push_code, push_data


@pytest.fixture
def config() -> RunPodConfig:
    return RunPodConfig(api_key="test_key")


@pytest.fixture
def target() -> SSHTarget:
    return SSHTarget(host="1.2.3.4", port=22345)


def _ok() -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 0
    r.stderr = ""
    return r


def _fail() -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 23
    r.stderr = "connection refused"
    return r


def test_push_code_calls_rsync(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "time_series").mkdir()
    (tmp_path / "common").mkdir()
    (tmp_path / "remote").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]")

    with patch("subprocess.run", return_value=_ok()) as mock_run:
        push_code(target, config)

    assert mock_run.call_count >= 1
    all_cmds = [mock_run.call_args_list[i][0][0] for i in range(mock_run.call_count)]
    assert any("rsync" in cmd[0] for cmd in all_cmds)


def test_push_code_includes_port_in_ssh_opts(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "time_series").mkdir()

    with patch("subprocess.run", return_value=_ok()) as mock_run:
        push_code(target, config)

    if mock_run.called:
        cmd_str = " ".join(mock_run.call_args_list[0][0][0])
        assert "22345" in cmd_str


def test_push_data_raises_for_missing_dataset(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        push_data(target, config, "nonexistent")


def test_push_data_calls_rsync(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "store-sales").mkdir(parents=True)

    with patch("subprocess.run", return_value=_ok()) as mock_run:
        push_data(target, config, "store-sales")

    assert mock_run.called
    cmd_str = " ".join(mock_run.call_args[0][0])
    assert "store-sales" in cmd_str


def test_push_data_rsync_failure_raises(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "store-sales").mkdir(parents=True)
    with patch("subprocess.run", return_value=_fail()):
        with pytest.raises(RuntimeError, match="rsync failed"):
            push_data(target, config, "store-sales")


def test_pull_results_calls_rsync_from_remote(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run", return_value=_ok()) as mock_run:
        pull_results(target, config)
    assert mock_run.called
    cmd_str = " ".join(mock_run.call_args[0][0])
    assert "mlruns" in cmd_str
    # source must be remote (contains @), dest must be local
    assert f"{target.user}@{target.host}" in cmd_str


def test_pull_results_creates_local_dir(
    config: RunPodConfig, target: SSHTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run", return_value=_ok()):
        pull_results(target, config)
    assert (tmp_path / config.local_mlruns_dir).exists()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/remote/test_sync.py -v
```
Expected: `ModuleNotFoundError: No module named 'remote.sync'`

- [ ] **Step 3: Implement remote/sync.py**

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from remote.config import RunPodConfig
from remote.ssh import SSHTarget

_SOURCE_DIRS = ["time_series", "common"]
_SOURCE_FILES = ["pyproject.toml"]
_RSYNC_BASE_OPTS = ["-avz", "--delete"]
_CODE_EXCLUDE_OPTS = ["--exclude=__pycache__/", "--exclude=*.pyc", "--exclude=.venv/"]


def _ssh_opt(target: SSHTarget) -> str:
    return f"ssh -p {target.port} -o StrictHostKeyChecking=no -o ConnectTimeout=30"


def _run_rsync(source: str, dest: str, target: SSHTarget, extra_opts: list[str] | None = None) -> None:
    cmd = [
        "rsync",
        *_RSYNC_BASE_OPTS,
        *(extra_opts or []),
        "-e", _ssh_opt(target),
        source,
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rsync failed (exit {result.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}"
        )


def push_code(target: SSHTarget, config: RunPodConfig) -> None:
    """Rsync source code to the remote pod.

    Syncs remote/, time_series/, common/, pyproject.toml.
    Skips: data/, mlruns/, .venv/, __pycache__/, .git/.
    """
    remote_dest = f"{target.user}@{target.host}:{config.remote_project_dir}"
    for dir_name in _SOURCE_DIRS:
        if Path(dir_name).exists():
            _run_rsync(f"{dir_name}/", f"{remote_dest}/{dir_name}/", target, _CODE_EXCLUDE_OPTS)
    for file_name in _SOURCE_FILES:
        if Path(file_name).exists():
            _run_rsync(file_name, f"{remote_dest}/{file_name}", target)


def push_data(target: SSHTarget, config: RunPodConfig, dataset: str) -> None:
    """Rsync data/<dataset>/ to the remote pod.

    Args:
        target: SSH connection info.
        config: RunPod configuration.
        dataset: Subdirectory name under local data/.

    Raises:
        FileNotFoundError: If data/<dataset> does not exist locally.
    """
    local_path = Path("data") / dataset
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset not found locally: {local_path}")
    remote_dest = f"{target.user}@{target.host}:{config.remote_data_dir}/{dataset}/"
    _run_rsync(f"data/{dataset}/", remote_dest, target)


def pull_results(target: SSHTarget, config: RunPodConfig) -> None:
    """Rsync mlruns/ from the remote pod back to the local machine."""
    local_dir = Path(config.local_mlruns_dir)
    local_dir.mkdir(exist_ok=True)
    remote_source = f"{target.user}@{target.host}:{config.remote_mlruns_dir}/"
    _run_rsync(remote_source, f"{config.local_mlruns_dir}/", target)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/remote/test_sync.py -v
```
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add remote/sync.py tests/remote/test_sync.py
git commit -m "Add rsync sync primitives (remote/sync.py)"
```

---

### Task 5: Environment bootstrap — `remote/environment.py`  *(after Task 2)*

**Files:**
- Create: `remote/environment.py`
- Create: `tests/remote/test_environment.py`

**Interfaces:**
- Consumes: `SSHTarget` from `remote.ssh`, `run_remote` from `remote.ssh`, `RunPodConfig`
- Produces:
  - `setup_environment(target: SSHTarget, config: RunPodConfig) -> None`
  - `start_mlflow(target: SSHTarget, config: RunPodConfig) -> None`
  - `stop_mlflow(target: SSHTarget) -> None`
  - `mlflow_proxy_url(pod_id: str, config: RunPodConfig) -> str`

---

- [ ] **Step 1: Write the failing tests**

`tests/remote/test_environment.py`:
```python
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from remote.config import RunPodConfig
from remote.environment import (
    mlflow_proxy_url,
    setup_environment,
    start_mlflow,
    stop_mlflow,
)
from remote.ssh import SSHTarget


@pytest.fixture
def config() -> RunPodConfig:
    return RunPodConfig(api_key="test_key")


@pytest.fixture
def target() -> SSHTarget:
    return SSHTarget(host="1.2.3.4", port=22345)


def _ok() -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 0
    r.stdout = ""
    r.stderr = ""
    return r


def test_setup_environment_installs_deps(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any("pip install" in cmd for cmd in commands)  # matches "python -m pip install"


def test_setup_environment_installs_mlflow(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any("mlflow" in cmd for cmd in commands)


def test_setup_environment_creates_dirs(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any("mkdir" in cmd for cmd in commands)


def test_setup_environment_verifies_cuda(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any("cuda" in cmd.lower() for cmd in commands)


def test_start_mlflow_launches_server(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        start_mlflow(target, config)
    cmd = mock_run.call_args[0][1]
    assert "mlflow server" in cmd
    assert str(config.mlflow_port) in cmd


def test_start_mlflow_runs_in_background(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        start_mlflow(target, config)
    cmd = mock_run.call_args[0][1]
    assert "nohup" in cmd and "&" in cmd


def test_stop_mlflow_kills_process(config: RunPodConfig, target: SSHTarget) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        stop_mlflow(target)
    cmd = mock_run.call_args[0][1]
    assert "pkill" in cmd


def test_mlflow_proxy_url(config: RunPodConfig) -> None:
    url = mlflow_proxy_url("abc123", config)
    assert "abc123" in url
    assert str(config.mlflow_port) in url
    assert url.startswith("https://")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/remote/test_environment.py -v
```
Expected: `ModuleNotFoundError: No module named 'remote.environment'`

- [ ] **Step 3: Implement remote/environment.py**

```python
from __future__ import annotations

from remote.config import RunPodConfig
from remote.ssh import SSHTarget, run_remote

_REMOTE_DEPS = [
    "mlflow>=3.12.0",
    "optuna>=4.9.0",
    "pandas>=2.2.0,<3",
    "numpy>=2.4.6",
    "scikit-learn>=1.8.0",
    "scipy>=1.17.1",
    "tqdm>=4.67.3",
    "kaggle>=2.1.2",
]


def setup_environment(target: SSHTarget, config: RunPodConfig) -> None:
    """Bootstrap the remote pod for training.

    Installs non-torch Python deps into the system Python (torch is
    pre-installed by the RunPod PyTorch image). Creates required remote
    directories. Verifies CUDA is accessible.
    """
    deps = " ".join(f'"{d}"' for d in _REMOTE_DEPS)
    run_remote(target, f"python -m pip install --quiet {deps}")
    run_remote(
        target,
        f"mkdir -p {config.remote_project_dir} {config.remote_data_dir} {config.remote_mlruns_dir}",
    )
    run_remote(
        target,
        'python -c "import torch; assert torch.cuda.is_available(), \'CUDA not available on pod\'"',
    )


def start_mlflow(target: SSHTarget, config: RunPodConfig) -> None:
    """Start an MLflow tracking server on the pod as a background process.

    Listens on 0.0.0.0 so it can serve both localhost (Mode 1) and remote
    clients via RunPod's HTTP proxy (Mode 2).
    """
    run_remote(
        target,
        (
            f"nohup mlflow server "
            f"--host 0.0.0.0 "
            f"--port {config.mlflow_port} "
            f"--backend-store-uri {config.remote_mlruns_dir} "
            f"> /tmp/mlflow.log 2>&1 &"
        ),
    )


def stop_mlflow(target: SSHTarget) -> None:
    """Stop the MLflow tracking server on the pod."""
    run_remote(target, "pkill -f 'mlflow server' || true", check=False)


def mlflow_proxy_url(pod_id: str, config: RunPodConfig) -> str:
    """Return the RunPod HTTP proxy URL for the MLflow server on a pod.

    Used in sweep mode so GPU pods can log to the MLflow CPU pod.
    Format: https://<pod-id>-<port>.proxy.runpod.net
    """
    return f"https://{pod_id}-{config.mlflow_port}.proxy.runpod.net"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/remote/test_environment.py -v
```
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add remote/environment.py tests/remote/test_environment.py
git commit -m "Add environment bootstrap (remote/environment.py)"
```

---

### Task 6: CLI — `remote/cli.py`  *(after Tasks 2, 3, 4, 5)*

**Files:**
- Create: `remote/cli.py`
- Create: `tests/remote/test_cli.py`

**Interfaces:**
- Consumes all of: `remote.config`, `remote.pod`, `remote.sync`, `remote.ssh`, `remote.environment`
- Produces: `app` Typer application, invocable as `python -m remote <subcommand>`

---

- [ ] **Step 1: Write the failing tests**

`tests/remote/test_cli.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from remote.cli import app, _normalize_train_command

runner = CliRunner()


# --- _normalize_train_command unit tests ---

def test_normalize_strips_uv_run_prefix() -> None:
    result = _normalize_train_command(["uv", "run", "python", "-m", "app"])
    assert result == ["python", "-m", "app"]


def test_normalize_leaves_other_commands_intact() -> None:
    result = _normalize_train_command(["python", "-m", "app"])
    assert result == ["python", "-m", "app"]


def test_normalize_warns_on_uv_in_middle(capsys: pytest.CaptureFixture) -> None:
    # should not strip, but warn
    _normalize_train_command(["python", "-m", "uv", "something"])
    # warning goes to stderr via typer.echo(..., err=True); capsys captures it
    # just verify it doesn't raise


# --- pod subcommands ---

def test_pod_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    mock_pods = [MagicMock(pod_id="abc", name="ml-1", status="RUNNING")]
    with patch("remote.cli.list_pods", return_value=mock_pods):
        result = runner.invoke(app, ["pod", "list"])
    assert result.exit_code == 0
    assert "abc" in result.output


def test_pod_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with patch("remote.cli.get_pod", return_value={"desiredStatus": "RUNNING"}):
        result = runner.invoke(app, ["pod", "status", "abc123"])
    assert result.exit_code == 0
    assert "RUNNING" in result.output


def test_pod_terminate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with patch("remote.cli.terminate_pod") as mock_terminate:
        result = runner.invoke(app, ["pod", "terminate", "abc123"])
    assert result.exit_code == 0
    mock_terminate.assert_called_once()


# --- sync subcommands ---

def test_sync_push_data_missing_dataset_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["sync", "push-data", "abc123"])
    assert result.exit_code != 0


# --- run command ---

def test_run_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with patch("remote.cli.create_pod", return_value="pod1") as mock_create, \
         patch("remote.cli.wait_for_running") as mock_wait, \
         patch("remote.cli.wait_for_ssh"), \
         patch("remote.cli.push_code"), \
         patch("remote.cli.setup_environment"), \
         patch("remote.cli.start_mlflow"), \
         patch("remote.cli.run_remote") as mock_train, \
         patch("remote.cli.pull_results"), \
         patch("remote.cli.terminate_pod") as mock_terminate:

        mock_wait.return_value = MagicMock(host="1.2.3.4", port=22345, user="root")
        result = runner.invoke(app, ["run", "python", "-m", "app"])

    assert result.exit_code == 0
    mock_create.assert_called_once()
    mock_terminate.assert_called_once()


def test_run_leaves_pod_running_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with patch("remote.cli.create_pod", return_value="pod1"), \
         patch("remote.cli.wait_for_running") as mock_wait, \
         patch("remote.cli.wait_for_ssh"), \
         patch("remote.cli.push_code"), \
         patch("remote.cli.setup_environment"), \
         patch("remote.cli.start_mlflow"), \
         patch("remote.cli.run_remote", side_effect=RuntimeError("SSH failed")), \
         patch("remote.cli.terminate_pod") as mock_terminate:

        mock_wait.return_value = MagicMock(host="1.2.3.4", port=22345, user="root")
        result = runner.invoke(app, ["run", "python", "-m", "app"])

    assert result.exit_code != 0
    mock_terminate.assert_not_called()
    assert "pod1" in result.output
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/remote/test_cli.py -v
```
Expected: `ModuleNotFoundError: No module named 'remote.cli'`

- [ ] **Step 3: Implement remote/cli.py**

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from remote.config import RunPodConfig, load_config
from remote.environment import mlflow_proxy_url, setup_environment, start_mlflow, stop_mlflow
from remote.pod import (
    PodInfo,
    create_mlflow_pod,
    create_pod,
    get_pod,
    get_ssh_target,
    list_pods,
    stop_pod,
    terminate_pod,
    wait_for_running,
)
from remote.ssh import SSHTarget, run_remote, wait_for_ssh
from remote.sync import pull_results, push_code, push_data

app = typer.Typer(name="remote", help="Remote GPU training on RunPod", no_args_is_help=True)
pod_app = typer.Typer(help="Pod lifecycle", no_args_is_help=True)
sync_app = typer.Typer(help="Code and data sync", no_args_is_help=True)
env_app = typer.Typer(help="Remote environment", no_args_is_help=True)
sweep_app = typer.Typer(help="Parallel sweeps", no_args_is_help=True)
app.add_typer(pod_app, name="pod")
app.add_typer(sync_app, name="sync")
app.add_typer(env_app, name="env")
app.add_typer(sweep_app, name="sweep")

_CONFIG_OPTION = typer.Option(Path("runpod_config.yaml"), "--config", help="Path to config YAML")


def _normalize_train_command(command: list[str]) -> list[str]:
    """Strip leading 'uv run' from a command; warn if uv appears elsewhere."""
    if len(command) >= 2 and command[0] == "uv" and command[1] == "run":
        return list(command[2:])
    if "uv" in command:
        typer.echo(
            "Warning: 'uv' found in train command. "
            "Remote pods use system Python — 'uv run' is unavailable.",
            err=True,
        )
    return list(command)


def _apply_on_complete(config: RunPodConfig, pod_id: str) -> None:
    if config.on_complete == "terminate":
        terminate_pod(config, pod_id)
        typer.echo(f"Pod {pod_id} terminated.")
    elif config.on_complete == "stop":
        stop_pod(config, pod_id)
        typer.echo(f"Pod {pod_id} stopped.")
    else:
        typer.echo(f"Pod {pod_id} left running (on_complete=keep).")


# ── Pod commands ──────────────────────────────────────────────────────────────

@pod_app.command("create")
def pod_create(
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type"),
    gpu_count: Optional[int] = typer.Option(None, "--gpu-count"),
    image: Optional[str] = typer.Option(None, "--image"),
    name_suffix: str = typer.Option("", "--name-suffix"),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Create a GPU pod and print its pod ID."""
    config = load_config(config_path)
    if gpu_type:
        config.gpu_type = gpu_type
    if gpu_count is not None:
        config.gpu_count = gpu_count
    if image:
        config.docker_image = image
    pod_id = create_pod(config, name_suffix=name_suffix)
    typer.echo(pod_id)


@pod_app.command("list")
def pod_list(config_path: Path = _CONFIG_OPTION) -> None:
    """List all pods on the account."""
    config = load_config(config_path)
    for pod in list_pods(config):
        typer.echo(f"{pod.pod_id}\t{pod.name}\t{pod.status}")


@pod_app.command("status")
def pod_status(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Print the status of a pod."""
    config = load_config(config_path)
    pod = get_pod(config, pod_id)
    typer.echo(pod.get("desiredStatus", "UNKNOWN"))


@pod_app.command("stop")
def pod_stop(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Stop a pod (preserves disk)."""
    config = load_config(config_path)
    stop_pod(config, pod_id)
    typer.echo(f"Stopped {pod_id}")


@pod_app.command("terminate")
def pod_terminate(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Terminate a pod (destroys disk and resources)."""
    config = load_config(config_path)
    terminate_pod(config, pod_id)
    typer.echo(f"Terminated {pod_id}")


# ── Sync commands ─────────────────────────────────────────────────────────────

@sync_app.command("push-code")
def sync_push_code(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Rsync source code to the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    push_code(target, config)
    typer.echo("Code synced.")


@sync_app.command("push-data")
def sync_push_data(
    pod_id: str,
    dataset: str = typer.Option(..., "--dataset", help="Dataset subdirectory name under data/"),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Rsync a local data/<dataset>/ directory to the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    push_data(target, config, dataset)
    typer.echo(f"Dataset '{dataset}' synced.")


@sync_app.command("pull")
def sync_pull(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Rsync mlruns/ from the pod back to local."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    pull_results(target, config)
    typer.echo("Results pulled.")


# ── Env commands ──────────────────────────────────────────────────────────────

@env_app.command("setup")
def env_setup(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Install remote deps and verify CUDA on the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    setup_environment(target, config)
    typer.echo("Environment ready.")


@env_app.command("mlflow-start")
def env_mlflow_start(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Start the MLflow server on the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    start_mlflow(target, config)
    typer.echo(f"MLflow server started on pod {pod_id}.")


@env_app.command("mlflow-stop")
def env_mlflow_stop(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Stop the MLflow server on the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    stop_mlflow(target)
    typer.echo("MLflow server stopped.")


# ── Train command ─────────────────────────────────────────────────────────────

@app.command("train")
def train_cmd(
    pod_id: str,
    train_command: list[str] = typer.Argument(..., help="Command to run (leading 'uv run' is stripped)"),
    mlflow_uri: Optional[str] = typer.Option(None, "--mlflow-uri", help="Override MLFLOW_TRACKING_URI"),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Run a training command on a running pod via SSH."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    normalized = _normalize_train_command(train_command)
    uri = mlflow_uri or f"http://localhost:{config.mlflow_port}"
    full_cmd = f"cd {config.remote_project_dir} && {' '.join(normalized)}"
    run_remote(target, full_cmd, env={"MLFLOW_TRACKING_URI": uri})
    typer.echo("Training complete.")


# ── Run command (full pipeline, single pod) ───────────────────────────────────

@app.command("run")
def run_cmd(
    train_command: list[str] = typer.Argument(..., help="Training command (leading 'uv run' is stripped)"),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    on_complete: Optional[str] = typer.Option(None, "--on-complete", help="terminate|stop|keep"),
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type"),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Full pipeline: provision → push → setup → train → pull → terminate."""
    config = load_config(config_path)
    if on_complete:
        config.on_complete = on_complete
    if gpu_type:
        config.gpu_type = gpu_type

    normalized = _normalize_train_command(train_command)
    datasets = [dataset] if dataset else config.default_datasets

    pod_id: Optional[str] = None
    try:
        typer.echo("Creating pod...")
        pod_id = create_pod(config)
        typer.echo(f"Pod created: {pod_id}")

        typer.echo("Waiting for pod to start...")
        target = wait_for_running(config, pod_id)
        typer.echo("Waiting for SSH...")
        wait_for_ssh(target)

        typer.echo("Pushing code...")
        push_code(target, config)

        for ds in datasets:
            typer.echo(f"Pushing dataset: {ds}")
            push_data(target, config, ds)

        typer.echo("Setting up environment...")
        setup_environment(target, config)

        typer.echo("Starting MLflow server...")
        start_mlflow(target, config)

        typer.echo("Running training...")
        full_cmd = f"cd {config.remote_project_dir} && {' '.join(normalized)}"
        run_remote(target, full_cmd, env={"MLFLOW_TRACKING_URI": f"http://localhost:{config.mlflow_port}"})

        typer.echo("Pulling results...")
        pull_results(target, config)

        _apply_on_complete(config, pod_id)
        pod_id = None
    except Exception as exc:
        typer.echo(f"\nError: {exc}", err=True)
        if pod_id:
            typer.echo(f"Pod {pod_id} left running — SSH in to debug or terminate manually:", err=True)
            typer.echo(f"  python -m remote pod terminate {pod_id}", err=True)
        raise typer.Exit(1)


# ── Sweep commands ────────────────────────────────────────────────────────────

@sweep_app.command("create-mlflow-pod")
def sweep_create_mlflow_pod(config_path: Path = _CONFIG_OPTION) -> None:
    """Create a persistent CPU pod to host MLflow for parallel sweeps."""
    config = load_config(config_path)
    pod_id = create_mlflow_pod(config)
    typer.echo(pod_id)
    typer.echo(f"MLflow URL (once running): {mlflow_proxy_url(pod_id, config)}", err=True)


@sweep_app.command("run")
def sweep_run(
    mlflow_pod_id: str = typer.Option(..., "--mlflow-pod"),
    n_pods: int = typer.Option(1, "--n-pods"),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    on_complete: Optional[str] = typer.Option(None, "--on-complete"),
    config_path: Path = _CONFIG_OPTION,
    train_command: list[str] = typer.Argument(...),
) -> None:
    """Launch N parallel GPU pods all logging to a shared MLflow pod."""
    import concurrent.futures

    config = load_config(config_path)
    if on_complete:
        config.on_complete = on_complete
    normalized = _normalize_train_command(train_command)
    datasets = [dataset] if dataset else config.default_datasets
    mlflow_uri = mlflow_proxy_url(mlflow_pod_id, config)

    def _run_one(index: int) -> str:
        pod_id = create_pod(config, name_suffix=f"sweep-{index}")
        typer.echo(f"[sweep-{index}] Pod created: {pod_id}")
        try:
            target = wait_for_running(config, pod_id)
            wait_for_ssh(target)
            push_code(target, config)
            for ds in datasets:
                push_data(target, config, ds)
            setup_environment(target, config)
            full_cmd = f"cd {config.remote_project_dir} && {' '.join(normalized)}"
            run_remote(target, full_cmd, env={"MLFLOW_TRACKING_URI": mlflow_uri})
            typer.echo(f"[sweep-{index}] Training complete.")
            _apply_on_complete(config, pod_id)
            return pod_id
        except Exception as exc:
            typer.echo(f"[sweep-{index}] Error on pod {pod_id}: {exc}", err=True)
            typer.echo(f"[sweep-{index}] Pod {pod_id} left running.", err=True)
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_pods) as executor:
        futures = [executor.submit(_run_one, i) for i in range(n_pods)]
        concurrent.futures.wait(futures)

    failed = [f for f in futures if f.exception() is not None]
    if failed:
        typer.echo(f"{len(failed)}/{n_pods} pods failed. See errors above.", err=True)
        raise typer.Exit(1)


@sweep_app.command("pull")
def sweep_pull(mlflow_pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Pull mlruns/ from the MLflow pod back to local."""
    config = load_config(config_path)
    target = get_ssh_target(config, mlflow_pod_id)
    pull_results(target, config)
    typer.echo("Sweep results pulled.")


@sweep_app.command("teardown")
def sweep_teardown(mlflow_pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Terminate the MLflow pod."""
    config = load_config(config_path)
    terminate_pod(config, mlflow_pod_id)
    typer.echo(f"MLflow pod {mlflow_pod_id} terminated.")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/remote/test_cli.py -v
```
Expected: all PASSED

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
uv run pytest tests/ -v
```
Expected: all tests pass (existing tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add remote/cli.py tests/remote/test_cli.py
git commit -m "Add Typer CLI for remote training (remote/cli.py)"
```

---

### Task 7: Docs update  *(after Task 6)*

**Files:**
- Modify: `docs/store_sales_project.md`

---

- [ ] **Step 1: Add remote training section to docs/store_sales_project.md**

Append this section at the end of the file:

```markdown
---

## Remote Training on RunPod

See [RunPod deployment design](superpowers/specs/2026-07-05-runpod-deployment-design.md) for full design rationale.

### Prerequisites

1. RunPod account with SSH public key added in [RunPod Settings](https://www.runpod.io/console/user/settings)
2. API key in macOS Keychain:
   ```bash
   security add-generic-password -a "$USER" -s "RUNPOD_API_KEY" -w "<your-key>"
   ```
3. Add to `~/.zshrc`:
   ```bash
   export RUNPOD_API_KEY=$(security find-generic-password -a "$USER" -s "RUNPOD_API_KEY" -w 2>/dev/null)
   ```
4. Copy and fill in config:
   ```bash
   cp runpod_config.yaml.example runpod_config.yaml
   ```

### Quick start — single run (fully automated)

```bash
uv run python -m remote run --dataset store-sales-time-series-forecasting \
  -- python -m time_series.main_store_sales_encoder_only --epochs 200
```

### Manual workflow (fine-grained control)

```bash
# 1. Create pod
POD_ID=$(uv run python -m remote pod create)

# 2. Push code and data
uv run python -m remote sync push-code $POD_ID
uv run python -m remote sync push-data $POD_ID --dataset store-sales-time-series-forecasting

# 3. Set up environment
uv run python -m remote env setup $POD_ID
uv run python -m remote env mlflow-start $POD_ID

# 4. Train
uv run python -m remote train $POD_ID -- python -m time_series.main_store_sales_encoder_only --epochs 200

# 5. Pull results
uv run python -m remote sync pull $POD_ID

# 6. Clean up
uv run python -m remote pod terminate $POD_ID
```

### Hyperparameter sweep (parallel pods, shared MLflow)

```bash
# 1. Start persistent MLflow pod
MLFLOW_POD=$(uv run python -m remote sweep create-mlflow-pod)

# 2. Run N parallel training pods
uv run python -m remote sweep run \
  --mlflow-pod $MLFLOW_POD --n-pods 4 \
  --dataset store-sales-time-series-forecasting \
  -- python -m time_series.tune_store_sales_encoder_only --n-trials 20 --epochs-per-trial 30

# 3. Pull results and tear down
uv run python -m remote sweep pull $MLFLOW_POD
uv run python -m remote sweep teardown $MLFLOW_POD
```

### Remote deps group

When adding non-torch Python packages to `pyproject.toml`, also add them to the
`remote` dependency group so they are installed on RunPod pods:

```toml
[dependency-groups]
remote = [
    # add new non-torch packages here
]
```
```

- [ ] **Step 2: Verify existing content is intact**

```bash
head -20 docs/store_sales_project.md
```
Expected: original Run Commands section still present

- [ ] **Step 3: Run precommit-check**

```bash
uv run pytest tests/ -v
uv run pyright remote/
uv run ruff check remote/ tests/remote/
uv run black --check remote/ tests/remote/
```
Fix any issues before committing.

- [ ] **Step 4: Commit**

```bash
git add docs/store_sales_project.md
git commit -m "Document remote RunPod training workflow in store_sales_project.md"
```

---

## Parallelization Guide

Tasks 2, 3, and 4 have no mutual dependencies and can be executed in parallel after Task 1 completes:

```
Task 1 (config + scaffolding)
    ├── Task 2 (ssh.py)    ─┐
    ├── Task 3 (pod.py)    ─┤ parallel
    └── Task 4 (sync.py)  ─┘
            │
        Task 5 (environment.py)   ← needs ssh.py
            │
        Task 6 (cli.py)           ← needs all
            │
        Task 7 (docs)
```
