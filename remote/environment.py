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
        "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available on pod'\"",
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
