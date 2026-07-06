from unittest.mock import patch

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
