"""Train PPO on the maze_bots Godot env, logging to MLflow.

Run with:
    uv run python -m rl_godot.simple_ppo_train
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import typer
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import rl_godot.env_launch  # noqa: F401 — patches StableBaselinesGodotEnv for user args
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from rl_godot.export_env import MAZE_BOTS_PATH_OPTION, ExportType, build_env

app = typer.Typer(add_completion=False)


class Float32ObsVecEnvWrapper(VecEnvWrapper):
    """Casts Dict-obs entries to float32.

    `StableBaselinesGodotEnv` builds obs arrays with plain `np.array(v)` on
    JSON-decoded Python floats, so they come back float64 regardless of the
    declared (float32) observation_space dtype. PyTorch's MPS backend rejects
    float64 tensors outright (`obs_as_tensor` does a bare `th.as_tensor`, no
    dtype cast), so training on `--device mps` crashes on the first rollout
    step without this.
    """

    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)

    def reset(self) -> VecEnvObs:
        return self._cast(self.venv.reset())

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._cast(obs), rewards, dones, infos

    @staticmethod
    def _cast(obs: VecEnvObs) -> VecEnvObs:
        assert isinstance(obs, dict)
        return {key: value.astype(np.float32) for key, value in obs.items()}


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


def log_model_to_mlflow(model: BaseAlgorithm, artifact_path: str) -> None:
    """Log an SB3 model snapshot to the active MLflow run under `artifact_path`.

    SB3 has no MLflow flavor, so this stores its native `.zip` (policy weights,
    optimizer state, and hyperparameters) as a plain artifact. Reload it with
    `PPO.load(<downloaded path>)`; the env is not part of the zip, so a reloaded
    model needs `set_env()` before further training.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ppo_model.zip"
        model.save(path)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _layout_config_digest(maze_bots_path: Path, layout_config: str | None) -> str:
    """SHA-256 of the layout config's contents, or "none" when scene defaults are used.

    The layout distribution depends on this file's contents, so the seed alone does not
    identify a training distribution -- without the digest a checkpoint exists whose
    layouts cannot be reconstructed.
    """
    if layout_config is None:
        return "none"
    path = maze_bots_path / "configs" / f"{layout_config}.cfg"
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MLflowCheckpointCallback(BaseCallback):
    """Logs a PPO snapshot to MLflow every `checkpoint_freq` env timesteps.

    SB3's own `CheckpointCallback` writes to a local directory, which is useless
    on an ephemeral RunPod pod — this uploads instead. `_on_step` fires once per
    vec-env step (i.e. `num_envs` timesteps), so the call cadence is scaled down
    by `num_envs` to keep the flag in timestep units regardless of --parallel.
    """

    def __init__(self, checkpoint_freq: int, num_envs: int) -> None:
        super().__init__()
        self.call_freq = max(checkpoint_freq // num_envs, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.call_freq == 0:
            log_model_to_mlflow(
                self.model, f"checkpoints/step_{self.num_timesteps:09d}"
            )
        return True


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
    rl_config: str | None = typer.Option(
        None,
        "--rl-config",
        help="Config name under maze_bots/configs/, no .cfg extension (e.g. 'no_rays'), "
        "passed to the launched Godot process as -- --rl-config=res://configs/"
        "<name>.cfg. Requires --env-path — there's no process to pass it to otherwise.",
    ),
    map_name: str | None = typer.Option(
        None,
        "--map",
        help="Map to load: a scene name under maze_bots/maps/ with no .tscn extension "
        "(e.g. 'GoStraightTrap') or a full res:// path, passed to the launched Godot "
        "process as -- --map=<name>. Requires --env-path — there's no process to pass "
        "it to otherwise. Defaults to maze_bots' own default map (GoStraight).",
    ),
    layout_config: str | None = typer.Option(
        None,
        "--layout-config",
        help="Layout randomization config under maze_bots/configs/, no .cfg extension "
        "(e.g. 'layout_GoStraightTrap'). Its sensor-independent ranges must suit the "
        "--map being loaded. Requires --env-path — there's no process to pass it to "
        "otherwise.",
    ),
    layout_seed: int | None = typer.Option(
        None,
        "--layout-seed",
        help="Pin every episode to one fixed layout instead of deriving a new one per "
        "episode from --seed. For debugging a specific layout and for deterministic "
        "playback. Requires --env-path.",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Base env seed. Worker p launches with seed + p, and each episode's layout "
        "is derived from it, so this selects the training layout distribution. Use a "
        "disjoint band (e.g. 10000+) for held-out evaluation.",
    ),
    build: bool = typer.Option(
        True,
        "--build/--no-build",
        help="Export a fresh maze_bots binary before launching (only when --env-path "
        "is given). Pass --no-build when running externally against a binary that "
        "was already built and pushed separately (e.g. on a RunPod pod, which has no "
        "Godot editor to export with).",
    ),
    maze_bots_path: Path = MAZE_BOTS_PATH_OPTION,
    export_type: ExportType = ExportType.DEBUG,
    experiment: str = typer.Option("GodotMazeBots_PPO", "--experiment"),
    run_name: str | None = typer.Option(None, "--run-name"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Torch device for the policy net ('auto', 'cpu', 'cuda', 'mps', ...). "
        "'auto' only checks for CUDA, so it won't pick up MPS on Apple Silicon — "
        "pass --device mps explicitly to use it.",
    ),
    learning_rate: float = typer.Option(
        3e-4,
        "--learning-rate",
        help="PPO Adam learning rate (SB3 default: 3e-4).",
    ),
    checkpoint_freq: int = typer.Option(
        5_000,
        "--checkpoint-freq",
        min=0,
        help="Log an intermediate model snapshot to MLflow every N env timesteps "
        "(summed across parallel envs). 0 disables checkpoints; the final model is "
        "logged either way.",
    ),
) -> None:
    """Train PPO on a maze_bots Godot instance, then roll out the learned policy.

    With no --env-path: connects to an already-running `godot-mono` process on a scene
    with `EnableRlSync = true` (e.g. `maze_bots/maps/GoStraight.tscn`) — you launch Godot
    yourself in a second terminal.

    With --env-path: launches and owns the Godot process itself (an exported binary, no
    platform suffix — godot_rl appends .app/.x86_64/.exe based on the host platform), no
    second terminal needed. By default this also re-exports maze_bots from
    --maze-bots-path first (--build/--no-build); pass --no-build to train against an
    already-built binary as-is.

    --map selects which scene under maze_bots/maps/ to load (default GoStraight);
    like --rl-config it only applies when this process launches Godot (--env-path).
    """
    if rl_config is not None and env_path is None:
        raise typer.BadParameter("--rl-config requires --env-path")
    if map_name is not None and env_path is None:
        raise typer.BadParameter("--map requires --env-path")
    if layout_config is not None and env_path is None:
        raise typer.BadParameter("--layout-config requires --env-path")
    if layout_seed is not None and env_path is None:
        raise typer.BadParameter("--layout-seed requires --env-path")

    if build and env_path is not None:
        build_env(maze_bots_path, export_type)

    # VecMonitor is what populates SB3's ep_info_buffer; without it the `rollout/`
    # block (ep_rew_mean) never prints, since SB3 only auto-wraps non-VecEnvs.
    env = VecMonitor(
        Float32ObsVecEnvWrapper(
            StableBaselinesGodotEnv(
                env_path=env_path,
                port=port,
                n_parallel=n_parallel,
                rl_config=rl_config,
                map_name=map_name,
                seed=seed,
                layout_config=layout_config,
                layout_seed=layout_seed,
            )
        )
    )
    # MultiInputPolicy, not MlpPolicy: godot_rl always exposes a Dict observation space.
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=n_steps,
        device=device,
        learning_rate=learning_rate,
        verbose=1,
    )

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
                "seed": seed,
                "layout_config": layout_config or "scene-defaults",
                "layout_config_sha256": _layout_config_digest(
                    maze_bots_path, layout_config
                ),
                "layout_seed": (
                    layout_seed if layout_seed is not None else "per-episode"
                ),
                "policy": "MultiInputPolicy",
                # Resolved device, not the raw CLI value — 'auto' logs as whatever
                # get_device() actually picked (e.g. 'cpu' when 'auto' misses MPS).
                "device": str(model.device),
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
                "checkpoint_freq": checkpoint_freq,
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
        model.learn(
            total_timesteps=total_timesteps,
            callback=(
                MLflowCheckpointCallback(checkpoint_freq, env.num_envs)
                if checkpoint_freq > 0
                else None
            ),
        )
        # Logged before the eval rollout so a crash there doesn't cost the
        # trained weights.
        log_model_to_mlflow(model, "model")

        obs = env.reset()
        print(f"reset obs: {obs}")
        episode_returns: list[float] = []
        step_rewards: list[float] = []
        for step in range(eval_steps):
            # VecEnvObs includes a tuple arm that godot_rl never returns —
            # its observation space is always a Dict.
            actions, _ = model.predict(
                cast(dict[str, np.ndarray], obs), deterministic=True
            )
            obs, rewards, dones, infos = env.step(actions)
            step_rewards.extend(float(r) for r in np.asarray(rewards).reshape(-1))
            # VecMonitor stamps info["episode"] on the step an env terminates, so
            # returns stay correct across the auto-reset a VecEnv hides.
            episode_returns.extend(
                float(info["episode"]["r"]) for info in infos if "episode" in info
            )
            print(f"step {step}: reward={rewards} done={dones}")

        # eval/mean_step_reward and eval/num_episodes always log, even when no
        # episode completes in eval_steps (e.g. a short smoke run) — otherwise a
        # run like that leaves MLflow with no eval signal at all.
        eval_metrics = {
            "eval/mean_step_reward": float(np.mean(step_rewards)),
            "eval/num_episodes": len(episode_returns),
        }
        if episode_returns:
            mean_episode_return = float(np.mean(episode_returns))
            eval_metrics["eval/mean_episode_return"] = mean_episode_return
            print(
                f"mean return over {len(episode_returns)} episodes: "
                f"{mean_episode_return:.3f}"
            )
        else:
            print(f"no episode completed within {eval_steps} steps")
        mlflow.log_metrics(eval_metrics)

    env.close()


if __name__ == "__main__":
    app()
