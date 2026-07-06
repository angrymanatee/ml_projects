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


def _run_rsync(
    source: str, dest: str, target: SSHTarget, extra_opts: list[str] | None = None
) -> None:
    cmd = [
        "rsync",
        *_RSYNC_BASE_OPTS,
        *(extra_opts or []),
        "-e",
        _ssh_opt(target),
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

    Syncs time_series/, common/, pyproject.toml.
    Skips: data/, mlruns/, .venv/, __pycache__/, .git/.
    """
    remote_dest = f"{target.user}@{target.host}:{config.remote_project_dir}"
    for dir_name in _SOURCE_DIRS:
        if Path(dir_name).exists():
            _run_rsync(
                f"{dir_name}/", f"{remote_dest}/{dir_name}/", target, _CODE_EXCLUDE_OPTS
            )
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
