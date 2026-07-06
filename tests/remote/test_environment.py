import subprocess
from unittest.mock import MagicMock, patch

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


def test_setup_environment_installs_deps(
    config: RunPodConfig, target: SSHTarget
) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any(
        "pip install" in cmd for cmd in commands
    )  # matches "python -m pip install"


def test_setup_environment_installs_mlflow(
    config: RunPodConfig, target: SSHTarget
) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any("mlflow" in cmd for cmd in commands)


def test_setup_environment_creates_dirs(
    config: RunPodConfig, target: SSHTarget
) -> None:
    with patch("remote.environment.run_remote", return_value=_ok()) as mock_run:
        setup_environment(target, config)
    commands = [c[0][1] for c in mock_run.call_args_list]
    assert any("mkdir" in cmd for cmd in commands)


def test_setup_environment_verifies_cuda(
    config: RunPodConfig, target: SSHTarget
) -> None:
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


def test_start_mlflow_runs_in_background(
    config: RunPodConfig, target: SSHTarget
) -> None:
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
