from typing import Any

import numpy as np
import pytest

import mlflow
from rl_godot.simple_ppo_train import MLflowWriter


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
