"""Watch a trained SB3 PPO policy play maze_bots in a visible Godot window.

Run with:
    uv run python -m rl_godot.play_model --model-path path/to/ppo_model.zip \
        --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import typer
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from mlflow.tracking import MlflowClient
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

import rl_godot.env_launch  # noqa: F401 — patches StableBaselinesGodotEnv for user args
from common.model_registry import TRACKING_URI
from rl_godot.export_env import MAZE_BOTS_PATH_OPTION, ExportType, build_env
from rl_godot.simple_ppo_train import Float32ObsVecEnvWrapper

app = typer.Typer(add_completion=False)

DEFAULT_ARTIFACT_PATH = "model/ppo_model.zip"

MODEL_PATH_OPTION = typer.Option(
    None,
    "--model-path",
    help="Local path to a ppo_model.zip. Mutually exclusive with --run-id.",
)


def resolve_model_path(
    model_path: Path | None,
    run_id: str | None,
    artifact_path: str | None,
) -> Path:
    """Resolve the --model-path/--run-id CLI options to a local ppo_model.zip path.

    Downloads from MLflow into a fresh temp dir when --run-id is given. The temp
    dir is intentionally never cleaned up — this is a short-lived CLI process, and
    the OS reclaims /tmp eventually, so it's not worth the teardown bookkeeping.
    """
    if model_path is not None and run_id is not None:
        raise typer.BadParameter("pass either --model-path or --run-id, not both")
    if model_path is not None:
        if artifact_path is not None:
            raise typer.BadParameter("--artifact-path requires --run-id")
        return model_path
    # Ordered after the --model-path return rather than paired with the both-given
    # check above, so the type checker narrows run_id to str for the download below.
    if run_id is None:
        raise typer.BadParameter("pass one of --model-path or --run-id")

    client = MlflowClient(tracking_uri=TRACKING_URI)
    dst_dir = tempfile.mkdtemp(prefix="play_model_")
    downloaded = client.download_artifacts(
        run_id, artifact_path or DEFAULT_ARTIFACT_PATH, dst_path=dst_dir
    )
    return Path(downloaded)


@app.command()
def main(
    model_path: Path | None = MODEL_PATH_OPTION,
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="MLflow run id to download the model from. Mutually exclusive with "
        "--model-path.",
    ),
    artifact_path: str | None = typer.Option(
        None,
        "--artifact-path",
        help=f"Artifact path under --run-id (default: {DEFAULT_ARTIFACT_PATH!r}). "
        "Requires --run-id.",
    ),
    env_path: str | None = typer.Option(
        None,
        "--env-path",
        help="Path (no platform suffix) to an exported maze_bots binary",
    ),
    build: bool = typer.Option(
        True,
        "--build/--no-build",
        help="Export a fresh maze_bots binary before launching (only when --env-path "
        "is given). Pass --no-build when running externally against a binary that "
        "was already built and pushed separately.",
    ),
    maze_bots_path: Path = MAZE_BOTS_PATH_OPTION,
    export_type: ExportType = ExportType.DEBUG,
    port: int = typer.Option(11008, "--port"),
    rl_config: str | None = typer.Option(
        None,
        "--rl-config",
        help="Config name under maze_bots/configs/, no .cfg extension (e.g. "
        "'watch' for the ray-fan debug overlay). Requires --env-path — there's no "
        "process to pass it to otherwise. Its sensor settings must match the config "
        "the policy was TRAINED under: they determine the observation vector's "
        "shape and meaning, so a mismatched config either crashes on the first "
        "predict() or silently feeds the policy garbage.",
    ),
    map_name: str | None = typer.Option(
        None,
        "--map",
        help="Scene name under maze_bots/maps/, no .tscn extension. Requires "
        "--env-path.",
    ),
    speedup: int = typer.Option(
        1,
        "--speedup",
        help="Godot engine speedup multiplier. Default 1 (real time) so a human "
        "can actually watch; simple_ppo_train.py leaves this unset to run as fast "
        "as possible instead.",
    ),
    n_steps: int = typer.Option(
        1000,
        "--n-steps",
        help="Env steps to roll out. The loop always runs for exactly this many "
        "steps regardless of how many episodes complete inside them.",
    ),
    stochastic: bool = typer.Option(
        False,
        "--stochastic",
        help="Sample actions from the policy distribution instead of taking the "
        "deterministic (argmax/mean) action.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Torch device for the policy net ('auto', 'cpu', 'cuda', 'mps', ...). "
        "'auto' only checks for CUDA, so it won't pick up MPS on Apple Silicon — "
        "pass --device mps explicitly to use it.",
    ),
) -> None:
    """Load a trained PPO policy and run it against a visible maze_bots window.

    Model source is either a local --model-path or an MLflow --run-id (downloaded
    via the MLflow client into a temp dir); exactly one is required. Env
    construction mirrors simple_ppo_train.py but forces n_parallel=1 (watching N
    windows defeats the point) and show_window=True. This tool does not start or
    log to an MLflow run — --run-id only reads a model from the existing run.
    """
    if rl_config is not None and env_path is None:
        raise typer.BadParameter("--rl-config requires --env-path")
    if map_name is not None and env_path is None:
        raise typer.BadParameter("--map requires --env-path")

    resolved_model_path = resolve_model_path(model_path, run_id, artifact_path)

    if build and env_path is not None:
        build_env(maze_bots_path, export_type)

    env = VecMonitor(
        Float32ObsVecEnvWrapper(
            StableBaselinesGodotEnv(
                env_path=env_path,
                port=port,
                n_parallel=1,
                rl_config=rl_config,
                map_name=map_name,
                show_window=True,
                speedup=speedup,
            )
        )
    )
    model = PPO.load(resolved_model_path, device=device)

    obs = env.reset()
    episode_returns: list[float] = []
    step_rewards: list[float] = []
    for step in range(n_steps):
        # VecEnvObs includes a tuple arm that godot_rl never returns — its
        # observation space is always a Dict.
        actions, _ = model.predict(
            cast(dict[str, np.ndarray], obs), deterministic=not stochastic
        )
        obs, rewards, dones, infos = env.step(actions)
        step_rewards.extend(float(r) for r in np.asarray(rewards).reshape(-1))
        # VecMonitor stamps info["episode"] on the step an env terminates, so
        # returns stay correct across the auto-reset a VecEnv hides.
        episode_returns.extend(
            float(info["episode"]["r"]) for info in infos if "episode" in info
        )
        print(f"step {step}: reward={rewards} done={dones}")

    print(f"episodes completed: {len(episode_returns)}")
    if episode_returns:
        print(f"mean episode return: {np.mean(episode_returns):.3f}")
    else:
        print(f"no episode completed within {n_steps} steps")
    print(f"mean step reward: {np.mean(step_rewards):.3f}")

    env.close()


if __name__ == "__main__":
    app()
