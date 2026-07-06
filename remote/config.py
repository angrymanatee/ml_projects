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
