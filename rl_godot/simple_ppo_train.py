"""Train PPO on the maze_bots Godot env, logging to MLflow.

Run with:
    uv run python -m rl_godot.simple_ppo_train
"""

from __future__ import annotations

import sys
from typing import Any, cast

import numpy as np
import typer
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecMonitor

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI

app = typer.Typer(add_completion=False)


class MLflowWriter(KVWriter):
    """SB3 log sink that forwards scalar metrics to the active MLflow run.

    SB3 writes metrics to its own Logger rather than MLflow, so training curves
    (`rollout/ep_rew_mean`, `train/loss`, ...) are invisible to MLflow without a
    bridge. Hooking SB3's dump cadence here means step numbering matches what SB3
    reports, instead of being re-derived from a callback.
    """

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        metrics = {
            key: float(value)
            for key, value in key_values.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and "mlflow" not in (key_excluded.get(key) or ())
        }
        if metrics:
            mlflow.log_metrics(metrics, step=step)


@app.command()
def main(
    total_timesteps: int = typer.Option(10_000, "--total-timesteps"),
    n_steps: int = typer.Option(
        64,
        "--n-steps",
        help="Env steps collected per env per PPO update (rollout buffer is "
        "n_steps * num_envs).",
    ),
    eval_steps: int = typer.Option(200, "--eval-steps"),
    port: int = typer.Option(11008, "--port"),
    env_path: str | None = typer.Option(
        None,
        "--env-path",
        help="Path (no platform suffix) to an exported maze_bots binary",
    ),
    n_parallel: int = typer.Option(
        1,
        "--parallel",
        min=1,
        help="Number of parallel envs. Requires --env-path; binds ports "
        "[--port, --port + parallel - 1].",
    ),
    experiment: str = typer.Option("GodotMazeBots_PPO", "--experiment"),
    run_name: str | None = typer.Option(None, "--run-name"),
) -> None:
    """Train PPO on a maze_bots Godot instance, then roll out the learned policy.

    With no --env-path: connects to an already-running `godot-mono` process on a scene
    with `EnableRlSync = true` (e.g. `maze_bots/maps/GoStraight.tscn`) — you launch Godot
    yourself in a second terminal.

    With --env-path: launches and owns the Godot process itself (an exported binary, no
    platform suffix — godot_rl appends .app/.x86_64/.exe based on the host platform), no
    second terminal needed.
    """
    # VecMonitor is what populates SB3's ep_info_buffer; without it the `rollout/`
    # block (ep_rew_mean) never prints, since SB3 only auto-wraps non-VecEnvs.
    env = VecMonitor(
        StableBaselinesGodotEnv(env_path=env_path, port=port, n_parallel=n_parallel)
    )
    # MultiInputPolicy, not MlpPolicy: godot_rl always exposes a Dict observation space.
    model = PPO("MultiInputPolicy", env, n_steps=n_steps, verbose=1)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(
        run_name=run_name,
        tags={
            "architecture": "ppo-godot-maze-bots",
            "git_branch": get_branch(),
            "git_sha": get_sha(),
        },
    ):
        mlflow.log_params(
            {
                "total_timesteps": total_timesteps,
                "eval_steps": eval_steps,
                "n_parallel": n_parallel,
                "num_envs": env.num_envs,
                "env_path": env_path or "editor",
                "policy": "MultiInputPolicy",
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "n_epochs": model.n_epochs,
                "learning_rate": model.learning_rate,
                # PPO declares clip_range as float | Schedule and swaps in a
                # schedule at setup; normalize the union back to a scalar.
                "clip_range": get_schedule_fn(model.clip_range)(1.0),
                "gamma": model.gamma,
                "gae_lambda": model.gae_lambda,
                "ent_coef": model.ent_coef,
            }
        )

        # Replacing the logger drops SB3's default console output, so
        # HumanOutputFormat goes back in explicitly to keep the verbose=1 table.
        model.set_logger(
            Logger(
                folder=None,
                output_formats=[MLflowWriter(), HumanOutputFormat(sys.stdout)],
            )
        )
        model.learn(total_timesteps=total_timesteps)

        obs = env.reset()
        print(f"reset obs: {obs}")
        episode_returns: list[float] = []
        for step in range(eval_steps):
            # VecEnvObs includes a tuple arm that godot_rl never returns —
            # its observation space is always a Dict.
            actions, _ = model.predict(
                cast(dict[str, np.ndarray], obs), deterministic=True
            )
            obs, rewards, dones, infos = env.step(actions)
            # VecMonitor stamps info["episode"] on the step an env terminates, so
            # returns stay correct across the auto-reset a VecEnv hides.
            episode_returns.extend(
                float(info["episode"]["r"]) for info in infos if "episode" in info
            )
            print(f"step {step}: reward={rewards} done={dones}")

        if episode_returns:
            mean_episode_return = float(np.mean(episode_returns))
            mlflow.log_metrics(
                {
                    "eval/mean_episode_return": mean_episode_return,
                    "eval/num_episodes": len(episode_returns),
                }
            )
            print(
                f"mean return over {len(episode_returns)} episodes: "
                f"{mean_episode_return:.3f}"
            )
        else:
            print(f"no episode completed within {eval_steps} steps; nothing logged")

    env.close()


if __name__ == "__main__":
    app()
