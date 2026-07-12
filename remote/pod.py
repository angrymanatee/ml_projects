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


def _extract_ssh_target(pod: dict, config: RunPodConfig) -> SSHTarget:
    ports = (pod.get("runtime") or {}).get("ports", [])
    ssh_port = next(
        (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
        None,
    )
    if not ssh_port:
        raise RuntimeError(f"No public SSH port found for pod {pod['id']!r}")
    return SSHTarget(
        host=ssh_port["ip"],
        port=ssh_port["publicPort"],
        identity_file=config.ssh_key_path,
    )


def create_pod(config: RunPodConfig, *, name_suffix: str = "") -> str:
    """Create a RunPod GPU pod and return its pod ID.

    Exposes port 22 (TCP) for SSH access. Does not wait for RUNNING state.

    Args:
        config: RunPod configuration.
        name_suffix: Appended to pod_name_prefix (with a hyphen) to form the pod name.

    Returns:
        RunPod pod ID string.
    """
    _init_sdk(config)
    name = (
        f"{config.pod_name_prefix}-{name_suffix}"
        if name_suffix
        else config.pod_name_prefix
    )
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

    Note: RunPod's SDK requires a gpu_type_id even for CPU workloads. If
    creation fails, check whether the SDK supports cpu_only pods or pick
    the cheapest GPU type from the RunPod dashboard and set it in config.

    Returns:
        RunPod pod ID string.
    """
    _init_sdk(config)
    pod = runpod_sdk.create_pod(
        name=f"{config.pod_name_prefix}-mlflow",
        image_name=config.mlflow_cpu_image,
        gpu_type_id=config.gpu_type,  # may need a cheap GPU type for CPU-only workloads
        cloud_type="SECURE",
        gpu_count=0,
        container_disk_in_gb=20,
        ports=f"22/tcp,{config.mlflow_port}/http",
        support_public_ip=True,
    )
    return pod["id"]


def get_pod(config: RunPodConfig, pod_id: str) -> dict:
    """Fetch raw pod data from the RunPod API."""
    _init_sdk(config)
    return runpod_sdk.get_pod(pod_id)


def get_ssh_target(config: RunPodConfig, pod_id: str) -> SSHTarget:
    """Fetch SSH connection info for a pod that is already running."""
    _init_sdk(config)
    pod = runpod_sdk.get_pod(pod_id)
    if pod is None:
        raise RuntimeError(f"Pod {pod_id!r} not found")
    return _extract_ssh_target(pod, config)


def wait_for_running(
    config: RunPodConfig,
    pod_id: str,
    timeout: int = 300,
) -> SSHTarget:
    """Poll until the pod reaches RUNNING status; return SSH connection info.

    Args:
        config: RunPod configuration.
        pod_id: Pod to wait for.
        timeout: Maximum seconds to wait before raising TimeoutError.

    Returns:
        SSHTarget for the running pod.

    Raises:
        TimeoutError: If the pod is not RUNNING within timeout seconds.
    """
    _init_sdk(config)
    elapsed = 0
    while elapsed < timeout:
        pod = runpod_sdk.get_pod(pod_id)
        if (
            pod is not None
            and pod.get("desiredStatus") == "RUNNING"
            and (pod.get("runtime") or {}).get("ports")
        ):
            return _extract_ssh_target(pod, config)
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
