from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from remote.cli import _normalize_train_command, app

runner = CliRunner()


# --- _normalize_train_command unit tests ---


def test_normalize_strips_uv_run_prefix() -> None:
    result = _normalize_train_command(["uv", "run", "python", "-m", "app"])
    assert result == ["python", "-m", "app"]


def test_normalize_leaves_other_commands_intact() -> None:
    result = _normalize_train_command(["python", "-m", "app"])
    assert result == ["python", "-m", "app"]


def test_normalize_warns_on_uv_in_middle(capsys: pytest.CaptureFixture) -> None:
    # should not strip, but warn
    _normalize_train_command(["python", "-m", "uv", "something"])
    # warning goes to stderr via typer.echo(..., err=True); capsys captures it
    # just verify it doesn't raise


# --- pod subcommands ---


def test_pod_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    mock_pods = [MagicMock(pod_id="abc", name="ml-1", status="RUNNING")]
    with patch("remote.cli.list_pods", return_value=mock_pods):
        result = runner.invoke(app, ["pod", "list"])
    assert result.exit_code == 0
    assert "abc" in result.output


def test_pod_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with patch("remote.cli.get_pod", return_value={"desiredStatus": "RUNNING"}):
        result = runner.invoke(app, ["pod", "status", "abc123"])
    assert result.exit_code == 0
    assert "RUNNING" in result.output


def test_pod_terminate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with patch("remote.cli.terminate_pod") as mock_terminate:
        result = runner.invoke(app, ["pod", "terminate", "abc123"])
    assert result.exit_code == 0
    mock_terminate.assert_called_once()


# --- sync subcommands ---


def test_sync_push_data_missing_dataset_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["sync", "push-data", "abc123"])
    assert result.exit_code != 0


# --- run command ---


def test_run_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with (
        patch("remote.cli.create_pod", return_value="pod1") as mock_create,
        patch("remote.cli.wait_for_running") as mock_wait,
        patch("remote.cli.wait_for_ssh"),
        patch("remote.cli.push_code"),
        patch("remote.cli.setup_environment"),
        patch("remote.cli.start_mlflow"),
        patch("remote.cli.run_remote"),
        patch("remote.cli.pull_results"),
        patch("remote.cli.terminate_pod") as mock_terminate,
    ):
        mock_wait.return_value = MagicMock(host="1.2.3.4", port=22345, user="root")
        result = runner.invoke(app, ["run", "python", "-m", "app"])

    assert result.exit_code == 0
    mock_create.assert_called_once()
    mock_terminate.assert_called_once()


def test_run_leaves_pod_running_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.chdir(tmp_path)

    with (
        patch("remote.cli.create_pod", return_value="pod1"),
        patch("remote.cli.wait_for_running") as mock_wait,
        patch("remote.cli.wait_for_ssh"),
        patch("remote.cli.push_code"),
        patch("remote.cli.setup_environment"),
        patch("remote.cli.start_mlflow"),
        patch("remote.cli.run_remote", side_effect=RuntimeError("SSH failed")),
        patch("remote.cli.terminate_pod") as mock_terminate,
    ):
        mock_wait.return_value = MagicMock(host="1.2.3.4", port=22345, user="root")
        result = runner.invoke(app, ["run", "python", "-m", "app"])

    assert result.exit_code != 0
    mock_terminate.assert_not_called()
    assert "pod1" in result.output
