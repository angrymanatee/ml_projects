from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from rl_godot.play_model import app, resolve_model_path, resolve_run_name

runner = CliRunner()


# --- resolve_model_path unit tests ---


EXPERIMENT = "GodotMazeBots_PPO"


def make_run(run_id: str):
    """Minimal stand-in for an MLflow Run — only `.info.run_id` is read."""
    run = SimpleNamespace()
    run.info = SimpleNamespace(run_id=run_id)
    return run


@pytest.mark.parametrize(
    "model_path,run_id,run_name",
    [
        (Path("model.zip"), "run123", None),
        (Path("model.zip"), None, "unique-ram-468"),
        (None, "run123", "unique-ram-468"),
        (Path("model.zip"), "run123", "unique-ram-468"),
    ],
)
def test_rejects_more_than_one_model_source(
    model_path: Path | None, run_id: str | None, run_name: str | None
) -> None:
    with pytest.raises(typer.BadParameter):
        resolve_model_path(model_path, run_id, run_name, None, EXPERIMENT)


def test_rejects_no_model_source() -> None:
    with pytest.raises(typer.BadParameter):
        resolve_model_path(None, None, None, None, EXPERIMENT)


def test_rejects_artifact_path_with_model_path() -> None:
    with pytest.raises(typer.BadParameter):
        resolve_model_path(
            Path("model.zip"),
            None,
            None,
            "checkpoints/step_000005000/ppo_model.zip",
            EXPERIMENT,
        )


def test_model_path_returned_unchanged() -> None:
    assert resolve_model_path(Path("model.zip"), None, None, None, EXPERIMENT) == Path(
        "model.zip"
    )


def test_run_id_downloads_default_artifact_path() -> None:
    with patch("rl_godot.play_model.MlflowClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.download_artifacts.return_value = (
            "/tmp/play_model_xyz/ppo_model.zip"
        )

        result = resolve_model_path(None, "run123", None, None, EXPERIMENT)

    mock_client.download_artifacts.assert_called_once()
    call_args = mock_client.download_artifacts.call_args
    assert call_args.args[0] == "run123"
    assert call_args.args[1] == "model/ppo_model.zip"
    assert result == Path("/tmp/play_model_xyz/ppo_model.zip")


def test_run_id_downloads_custom_artifact_path() -> None:
    with patch("rl_godot.play_model.MlflowClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.download_artifacts.return_value = (
            "/tmp/play_model_xyz/ppo_model.zip"
        )

        resolve_model_path(
            None,
            "run123",
            None,
            "checkpoints/step_000005000/ppo_model.zip",
            EXPERIMENT,
        )

    call_args = mock_client.download_artifacts.call_args
    assert call_args.args[1] == "checkpoints/step_000005000/ppo_model.zip"


# --- resolve_run_name unit tests ---


def test_run_name_resolves_to_run_id() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="0")
    client.search_runs.return_value = [make_run("run123")]

    assert resolve_run_name(client, "unique-ram-468", EXPERIMENT) == "run123"
    filter_string = client.search_runs.call_args.kwargs["filter_string"]
    assert filter_string == "attributes.run_name = 'unique-ram-468'"


def test_run_name_picks_most_recent_of_several() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="0")
    client.search_runs.return_value = [make_run("newest"), make_run("older")]

    assert resolve_run_name(client, "retry", EXPERIMENT) == "newest"
    # Ordering is the server's job, not ours — assert we asked for it.
    assert client.search_runs.call_args.kwargs["order_by"] == [
        "attributes.start_time DESC"
    ]


def test_run_name_rejects_unknown_experiment() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None

    with pytest.raises(typer.BadParameter):
        resolve_run_name(client, "unique-ram-468", "NoSuchExperiment")


def test_run_name_rejects_no_match() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="0")
    client.search_runs.return_value = []

    with pytest.raises(typer.BadParameter):
        resolve_run_name(client, "nope", EXPERIMENT)


def test_run_name_downloads_via_resolved_run_id() -> None:
    with patch("rl_godot.play_model.MlflowClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="0"
        )
        mock_client.search_runs.return_value = [make_run("run123")]
        mock_client.download_artifacts.return_value = (
            "/tmp/play_model_xyz/ppo_model.zip"
        )

        result = resolve_model_path(None, None, "unique-ram-468", None, EXPERIMENT)

    assert mock_client.download_artifacts.call_args.args[0] == "run123"
    assert result == Path("/tmp/play_model_xyz/ppo_model.zip")


# --- CLI validation ---


def test_rl_config_requires_env_path() -> None:
    result = runner.invoke(app, ["--model-path", "model.zip", "--rl-config", "no_rays"])
    assert result.exit_code != 0


def test_map_requires_env_path() -> None:
    result = runner.invoke(app, ["--model-path", "model.zip", "--map", "GoStraight"])
    assert result.exit_code != 0


def test_missing_model_source_rejected() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_both_model_path_and_run_id_rejected() -> None:
    result = runner.invoke(app, ["--model-path", "model.zip", "--run-id", "run123"])
    assert result.exit_code != 0


# --- env construction wiring ---


class _FakeGodotVecEnv:
    """Minimal stand-in for StableBaselinesGodotEnv satisfying the VecEnv interface
    that Float32ObsVecEnvWrapper/VecMonitor actually touch, so main() can run its
    rollout loop end to end without a real Godot process."""

    def __init__(self) -> None:
        self.num_envs = 1
        self._step_count = 0
        # godot_rl always exposes a Dict observation space (MultiInputPolicy) and a
        # continuous action space; the exact shapes don't matter to this test.
        self.observation_space = gym.spaces.Dict(
            {"goalbearing": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def get_attr(self, attr_name: str, indices=None):
        if attr_name == "render_mode":
            return [None]
        raise AttributeError(attr_name)

    def reset(self):
        return {"goalbearing": np.zeros((1, 3), dtype=np.float64)}

    def step_async(self, actions) -> None:
        self._last_actions = actions

    def step_wait(self):
        self._step_count += 1
        obs = {"goalbearing": np.zeros((1, 3), dtype=np.float64)}
        rewards = np.array([1.0], dtype=np.float32)
        dones = np.array([True])
        infos = [{}]
        return obs, rewards, dones, infos

    def close(self) -> None:
        pass


class _FakeModel:
    def predict(self, obs, deterministic: bool = True):
        return np.zeros((1, 3), dtype=np.float32), None


def test_show_window_and_n_parallel_reach_env_constructor(tmp_path: Path) -> None:
    fake_env = _FakeGodotVecEnv()
    with (
        patch("rl_godot.play_model.StableBaselinesGodotEnv") as mock_env_cls,
        patch("rl_godot.play_model.PPO") as mock_ppo,
    ):
        mock_env_cls.return_value = fake_env
        mock_ppo.load.return_value = _FakeModel()

        result = runner.invoke(
            app,
            [
                "--model-path",
                str(tmp_path / "ppo_model.zip"),
                "--env-path",
                "/fake/MazeBots",
                "--no-build",
                "--n-steps",
                "1",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_env_cls.assert_called_once()
    _, kwargs = mock_env_cls.call_args
    assert kwargs["show_window"] is True
    assert kwargs["n_parallel"] == 1
    assert "episodes completed: 1" in result.output


def test_speedup_passed_through_to_env(tmp_path: Path) -> None:
    fake_env = _FakeGodotVecEnv()
    with (
        patch("rl_godot.play_model.StableBaselinesGodotEnv") as mock_env_cls,
        patch("rl_godot.play_model.PPO") as mock_ppo,
    ):
        mock_env_cls.return_value = fake_env
        mock_ppo.load.return_value = _FakeModel()

        result = runner.invoke(
            app,
            [
                "--model-path",
                str(tmp_path / "ppo_model.zip"),
                "--env-path",
                "/fake/MazeBots",
                "--no-build",
                "--n-steps",
                "1",
                "--speedup",
                "4",
            ],
        )

    assert result.exit_code == 0, result.output
    _, kwargs = mock_env_cls.call_args
    assert kwargs["speedup"] == 4


def test_device_passed_through_to_ppo_load(tmp_path: Path) -> None:
    fake_env = _FakeGodotVecEnv()
    with (
        patch("rl_godot.play_model.StableBaselinesGodotEnv") as mock_env_cls,
        patch("rl_godot.play_model.PPO") as mock_ppo,
    ):
        mock_env_cls.return_value = fake_env
        mock_ppo.load.return_value = _FakeModel()
        model_path = tmp_path / "ppo_model.zip"

        result = runner.invoke(
            app,
            [
                "--model-path",
                str(model_path),
                "--env-path",
                "/fake/MazeBots",
                "--no-build",
                "--n-steps",
                "1",
                "--device",
                "mps",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_ppo.load.assert_called_once_with(model_path, device="mps")
