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
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "time_series").mkdir()

    with patch("subprocess.run", return_value=_ok()) as mock_run:
        push_code(target, config)

    if mock_run.called:
        cmd_str = " ".join(mock_run.call_args_list[0][0][0])
        assert "22345" in cmd_str


def test_push_data_raises_for_missing_dataset(
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        push_data(target, config, "nonexistent")


def test_push_data_calls_rsync(
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "store-sales").mkdir(parents=True)

    with patch("subprocess.run", return_value=_ok()) as mock_run:
        push_data(target, config, "store-sales")

    assert mock_run.called
    cmd_str = " ".join(mock_run.call_args[0][0])
    assert "store-sales" in cmd_str


def test_push_data_rsync_failure_raises(
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "store-sales").mkdir(parents=True)
    with (
        patch("subprocess.run", return_value=_fail()),
        pytest.raises(RuntimeError, match="rsync failed"),
    ):
        push_data(target, config, "store-sales")


def test_pull_results_calls_rsync_from_remote(
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    config: RunPodConfig,
    target: SSHTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run", return_value=_ok()):
        pull_results(target, config)
    assert (tmp_path / config.local_mlruns_dir).exists()
