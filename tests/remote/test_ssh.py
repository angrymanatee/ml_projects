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
    assert any("export FOO=bar" in arg for arg in cmd)


def test_run_remote_env_exported_before_compound_command(target: SSHTarget) -> None:
    with patch("subprocess.run", return_value=_success()) as mock_run:
        run_remote(target, "cd /tmp && echo hi", env={"FOO": "bar"})
    cmd = mock_run.call_args[0][0]
    full_command = cmd[-1]
    assert "export FOO=" in full_command
    assert full_command.index("export FOO=") < full_command.index("cd /tmp")


def test_run_remote_failure_raises(target: SSHTarget) -> None:
    with (
        patch("subprocess.run", return_value=_failure("command not found")),
        pytest.raises(RuntimeError, match="Remote command failed"),
    ):
        run_remote(target, "bad-command")


def test_run_remote_no_check_returns_result(target: SSHTarget) -> None:
    with patch("subprocess.run", return_value=_failure()):
        result = run_remote(target, "bad", check=False)
    assert result.returncode == 1


def test_wait_for_ssh_succeeds_immediately(target: SSHTarget) -> None:
    with patch("remote.ssh.run_remote", return_value=_success()):
        wait_for_ssh(target, timeout=10)  # should not raise


def test_wait_for_ssh_retries_until_timeout(target: SSHTarget) -> None:
    with (
        patch("remote.ssh.run_remote", return_value=_failure()),
        patch("remote.ssh.time.sleep"),
        pytest.raises(TimeoutError),
    ):
        wait_for_ssh(target, timeout=5)
