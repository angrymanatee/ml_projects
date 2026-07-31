from __future__ import annotations

import numpy as np
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n_steps: int = typer.Option(200, "--n-steps"),
    port: int = typer.Option(11008, "--port"),
) -> None:
    """Step a live maze_bots Godot instance with a constant action to verify the interface.

    Requires a `godot-mono` process already running a scene with `EnableRlSync = true`
    (e.g. `maze_bots/maps/GoStraightTraining.tscn`) — this script connects to it rather
    than launching it (`env_path=None`).
    """
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path=None, port=port)
    # One row per agent/env instance; rl_agent.gd's action space is a single
    # 2-float continuous "direction" key, and this scene has one agent.
    action = np.array([[1.0, 0.0]], dtype=np.float32)

    obs = env.reset()
    print(f"reset obs: {obs}")
    for step in range(n_steps):
        obs, reward, done, info = env.step(action)
        print(f"step {step}: obs={obs} reward={reward} done={done}")

    env.close()


if __name__ == "__main__":
    app()
