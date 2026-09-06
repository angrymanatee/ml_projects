from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from remote.config import RunPodConfig
from remote.ssh import SSHTarget, run_remote

_SOURCE_DIRS = ["time_series", "common", "rl_godot"]
_SOURCE_FILES = ["pyproject.toml"]
_RSYNC_BASE_OPTS = ["-avz", "--delete"]
_CODE_EXCLUDE_OPTS = ["--exclude=__pycache__/", "--exclude=*.pyc", "--exclude=.venv/"]


def _ssh_opt(target: SSHTarget) -> str:
    # Reuses SSHTarget.ssh_opts rather than building options separately, so
    # rsync picks up identity_file the same way run_remote()/wait_for_ssh() do —
    # this previously omitted -i entirely, silently breaking every sync
    # operation whenever ssh_key_path is set (which raw-IP RunPod connections
    # require; see configs/runpod_config.yaml.example).
    return "ssh " + " ".join(target.ssh_opts)


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
    run_remote(
        target, "which rsync || (apt-get update -qq && apt-get install -y -qq rsync)"
    )
    run_remote(target, f"mkdir -p {config.remote_project_dir}")
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
    remote_dataset_dir = f"{config.remote_data_dir}/{dataset}"
    run_remote(target, f"mkdir -p {remote_dataset_dir}")
    _run_rsync(
        f"data/{dataset}/", f"{target.user}@{target.host}:{remote_dataset_dir}/", target
    )


def push_godot_build(
    target: SSHTarget, config: RunPodConfig, maze_bots_path: Path
) -> None:
    """Rsync the exported Linux Godot binary to the remote pod.

    Args:
        target: SSH connection info.
        config: RunPod configuration.
        maze_bots_path: Local path to the maze_bots checkout.

    Raises:
        FileNotFoundError: If <maze_bots_path>/build/linux does not exist locally
            (run maze_bots's `scripts/export.sh <type> linux` first — the export
            defaults to the host platform only).
    """
    local_build_dir = maze_bots_path / "build" / "linux"
    if not local_build_dir.exists():
        raise FileNotFoundError(
            f"Linux export not found: {local_build_dir}\n"
            "Run `scripts/export.sh debug linux` in the maze_bots repo first."
        )
    run_remote(target, f"mkdir -p {config.remote_godot_build_dir}")
    _run_rsync(
        f"{local_build_dir}/",
        f"{target.user}@{target.host}:{config.remote_godot_build_dir}/",
        target,
    )
    run_remote(target, f"chmod +x {config.remote_godot_build_dir}/MazeBots.x86_64")


def pull_results(target: SSHTarget, config: RunPodConfig) -> None:
    """Export MLflow runs from the remote pod and import them locally.

    Uses mlflow-export-import to export all experiments from the remote
    tracking server, rsync the bundle locally, then import into the local
    file-store. Run IDs are remapped on import so there are no collisions
    with existing local runs.
    """
    remote_export_dir = "/tmp/mlflow-export"
    remote_mlflow_uri = f"http://localhost:{config.mlflow_port}"

    run_remote(
        target,
        f"export-experiments --experiments '*' --output-dir {remote_export_dir}",
        env={"MLFLOW_TRACKING_URI": remote_mlflow_uri},
    )

    local_mlruns = Path(config.local_mlruns_dir)
    local_mlruns.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mlflow-export-") as local_export_dir:
        remote_source = f"{target.user}@{target.host}:{remote_export_dir}/"
        _run_rsync(remote_source, f"{local_export_dir}/", target)

        local_tracking_uri = local_mlruns.resolve().as_uri()
        result = subprocess.run(
            ["import-experiments", "--input-dir", local_export_dir],
            capture_output=True,
            text=True,
            env={**os.environ, "MLFLOW_TRACKING_URI": local_tracking_uri},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mlflow-export-import import failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr}"
            )
