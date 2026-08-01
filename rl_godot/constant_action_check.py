from __future__ import annotations

import numpy as np
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n_steps: int = typer.Option(200, "--n-steps"),
    port: int = typer.Option(11008, "--port"),
    env_path: str | None = typer.Option(
        None,
        "--env-path",
        help="Path (no platform suffix) to an exported maze_bots binary",
    ),
) -> None:
    """Step a maze_bots Godot instance with a constant action to verify the interface.

    With no --env-path: connects to an already-running `godot-mono` process on a scene
    with `EnableRlSync = true` (e.g. `maze_bots/maps/GoStraight.tscn`) — you launch Godot
    yourself in a second terminal.

    With --env-path: launches and owns the Godot process itself (an exported binary, no
    platform suffix — godot_rl appends .app/.x86_64/.exe based on the host platform), no
    second terminal needed.
    """
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path=env_path, port=port)
    # One row per agent/env instance; rl_agent.gd's action space is a single
    # 2-float continuous "direction" key, and this scene has one agent.
    action = np.array([[0.0, -1.0]], dtype=np.float32)

    obs = env.reset()
    print(f"reset obs: {obs}")
    for step in range(n_steps):
        obs, reward, done, info = env.step(action)
        print(f"step {step}: obs={obs} reward={reward} done={done}")

    env.close()


if __name__ == "__main__":
    app()
