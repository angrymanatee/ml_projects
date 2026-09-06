from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import mlflow
import numpy as np
import pytest
from typer.testing import CliRunner

from rl_godot.simple_ppo_train import (
    MLflowCheckpointCallback,
    MLflowWriter,
    app,
    log_model_to_mlflow,
)

runner = CliRunner()


@pytest.fixture
def logged_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[dict[str, float], int]]:
    """Capture mlflow.log_metrics calls as (metrics, step) pairs."""
    calls: list[tuple[dict[str, float], int]] = []

    def fake_log_metrics(metrics: dict[str, float], step: int = 0) -> None:
        calls.append((metrics, step))

    monkeypatch.setattr(mlflow, "log_metrics", fake_log_metrics)
    return calls


def test_forwards_scalar_metrics(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    MLflowWriter().write({"rollout/ep_rew_mean": 12.5, "train/loss": 3}, {}, step=64)

    assert logged_metrics == [({"rollout/ep_rew_mean": 12.5, "train/loss": 3.0}, 64)]


def test_casts_numpy_scalars_to_float(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    key_values: dict[str, Any] = {
        "train/std": np.float32(1.5),
        "time/iterations": np.int64(4),
    }

    MLflowWriter().write(key_values, {}, step=1)

    metrics, _ = logged_metrics[0]
    assert metrics == {"train/std": 1.5, "time/iterations": 4.0}
    assert all(type(value) is float for value in metrics.values())


def test_drops_non_numeric_values(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    key_values: dict[str, Any] = {
        "train/loss": 0.5,
        "some/label": "text",
        "some/missing": None,
        "some/array": np.zeros(3),
    }

    MLflowWriter().write(key_values, {}, step=0)

    assert logged_metrics == [({"train/loss": 0.5}, 0)]


def test_honors_mlflow_exclusion(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    key_values: dict[str, Any] = {"train/loss": 0.5, "train/secret": 9.0}
    key_excluded = {"train/secret": ("mlflow", "csv")}

    MLflowWriter().write(key_values, key_excluded, step=0)

    assert logged_metrics == [({"train/loss": 0.5}, 0)]


def test_keeps_keys_excluded_from_other_sinks(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    # SB3 excludes plenty of keys from stdout/tensorboard that MLflow should keep.
    MLflowWriter().write({"train/loss": 0.5}, {"train/loss": ("stdout", "csv")}, step=0)

    assert logged_metrics == [({"train/loss": 0.5}, 0)]


def test_tolerates_none_valued_exclusion(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    # SB3's Logger stores None rather than an empty tuple for unexcluded keys.
    MLflowWriter().write({"train/loss": 0.5}, {"train/loss": None}, step=0)  # type: ignore[dict-item]

    assert logged_metrics == [({"train/loss": 0.5}, 0)]


def test_skips_call_when_nothing_loggable(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    MLflowWriter().write({"some/label": "text"}, {}, step=0)

    assert logged_metrics == []


def test_step_defaults_to_zero(
    logged_metrics: list[tuple[dict[str, float], int]],
) -> None:
    MLflowWriter().write({"train/loss": 0.5}, {})

    assert logged_metrics[0][1] == 0


@pytest.fixture
def logged_artifacts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Capture mlflow.log_artifact calls as (local_filename, artifact_path) pairs."""
    calls: list[tuple[str, str]] = []

    def fake_log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        assert Path(local_path).exists(), "artifact must still exist when logged"
        calls.append((Path(local_path).name, artifact_path or ""))

    monkeypatch.setattr(mlflow, "log_artifact", fake_log_artifact)
    return calls


class FakeSavableModel:
    """Stands in for an SB3 algorithm: `save` is all `log_model_to_mlflow` uses."""

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"fake-sb3-zip")


def test_log_model_to_mlflow_uploads_saved_zip(
    logged_artifacts: list[tuple[str, str]],
) -> None:
    log_model_to_mlflow(cast(Any, FakeSavableModel()), "model")

    assert logged_artifacts == [("ppo_model.zip", "model")]


def _run_callback(
    callback: MLflowCheckpointCallback, n_calls: int, num_envs: int
) -> None:
    """Drive `_on_step` the way SB3 does: one call per vec-env step."""
    callback.model = cast(Any, FakeSavableModel())
    for _ in range(n_calls):
        callback.n_calls += 1
        callback.num_timesteps += num_envs
        callback._on_step()


def test_checkpoint_callback_logs_on_frequency(
    logged_artifacts: list[tuple[str, str]],
) -> None:
    callback = MLflowCheckpointCallback(checkpoint_freq=4, num_envs=1)
    _run_callback(callback, n_calls=9, num_envs=1)

    assert [path for _, path in logged_artifacts] == [
        "checkpoints/step_000000004",
        "checkpoints/step_000000008",
    ]


def test_checkpoint_callback_scales_frequency_by_num_envs() -> None:
    # 4 envs step together, so 1000 timesteps is 250 _on_step calls.
    assert MLflowCheckpointCallback(checkpoint_freq=1000, num_envs=4).call_freq == 250


def test_checkpoint_callback_freq_never_drops_below_one() -> None:
    assert MLflowCheckpointCallback(checkpoint_freq=2, num_envs=8).call_freq == 1


# --- CLI validation ---


def test_layout_config_requires_env_path() -> None:
    result = runner.invoke(app, ["--layout-config", "layout_GoStraightTrap"])
    assert result.exit_code != 0


def test_seed_reaches_the_env_constructor(tmp_path: Path) -> None:
    with patch("rl_godot.simple_ppo_train.StableBaselinesGodotEnv") as mock_env_cls:
        mock_env_cls.side_effect = RuntimeError("stop after construction")
        runner.invoke(
            app,
            [
                "--env-path",
                str(tmp_path / "MazeBots"),
                "--no-build",
                "--seed",
                "10000",
                "--layout-config",
                "layout_GoStraightTrap",
            ],
        )
    kwargs = mock_env_cls.call_args.kwargs
    assert kwargs["seed"] == 10000
    assert kwargs["layout_config"] == "layout_GoStraightTrap"
