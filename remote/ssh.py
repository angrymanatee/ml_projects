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
            "-p",
            str(self.port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=30",
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
    raise TimeoutError(
        f"SSH not available on {target.host}:{target.port} after {timeout}s"
    )
