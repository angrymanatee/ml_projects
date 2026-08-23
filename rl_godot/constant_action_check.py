from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import typer

if TYPE_CHECKING:
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

app = typer.Typer(add_completion=False)


def _format_obs(obs: dict[str, np.ndarray]) -> str:
    parts = []
    for key, value in obs.items():
        vec = np.asarray(value).reshape(-1)
        parts.append(f"{key}=[{', '.join(f'{v:.3f}' for v in vec)}]")
    return " ".join(parts)


def _run_gui(
    env: StableBaselinesGodotEnv, action: np.ndarray, action_step: float
) -> None:
    """Drive the step loop from tkinter's event loop, single-threaded.

    `env.step()` runs on the Tk main thread via `after()` polling rather than a
    background thread, so there's no need to reason about whether the Godot socket
    wrapper is thread-safe.
    """
    import tkinter as tk

    root = tk.Tk()
    root.title("constant_action_check")

    state = {"step": 0}

    action_label = tk.Label(root, text="action: -", font=("Menlo", 12))
    action_label.pack(anchor="w", padx=8, pady=(8, 0))

    obs_label = tk.Label(root, text="obs: -", font=("Menlo", 12), justify="left")
    obs_label.pack(anchor="w", padx=8)

    status_label = tk.Label(root, text="reward: - done: - step: 0", font=("Menlo", 12))
    status_label.pack(anchor="w", padx=8, pady=(0, 8))

    entry_frame = tk.Frame(root)
    entry_frame.pack(anchor="w", padx=8, pady=(0, 8))
    tk.Label(entry_frame, text="x:").pack(side="left")
    x_entry = tk.Entry(entry_frame, width=8)
    x_entry.insert(0, str(action[0, 0]))
    x_entry.pack(side="left")
    tk.Label(entry_frame, text="y:").pack(side="left")
    y_entry = tk.Entry(entry_frame, width=8)
    y_entry.insert(0, str(action[0, 1]))
    y_entry.pack(side="left")

    def apply_entry() -> None:
        try:
            action[0, 0] = float(x_entry.get())
            action[0, 1] = float(y_entry.get())
        except ValueError:
            pass

    tk.Button(entry_frame, text="Set", command=apply_entry).pack(
        side="left", padx=(4, 0)
    )

    help_label = tk.Label(
        root,
        text="arrow keys nudge action, space zeros it",
        font=("Menlo", 10),
        fg="gray40",
    )
    help_label.pack(anchor="w", padx=8, pady=(0, 8))

    def nudge(dx: float, dy: float) -> None:
        action[0, 0] += dx
        action[0, 1] += dy

    def zero_action(_event=None) -> None:
        action[0, 0] = 0.0
        action[0, 1] = 0.0

    root.bind("<Up>", lambda _e: nudge(0.0, -action_step))
    root.bind("<Down>", lambda _e: nudge(0.0, action_step))
    root.bind("<Left>", lambda _e: nudge(-action_step, 0.0))
    root.bind("<Right>", lambda _e: nudge(action_step, 0.0))
    root.bind("<space>", zero_action)

    def tick() -> None:
        obs, reward, done, info = env.step(action)
        state["step"] += 1
        line = f"step {state['step']}: obs={obs} reward={reward} done={done}"
        print(line)

        action_label.config(text=f"action: x={action[0, 0]:.3f} y={action[0, 1]:.3f}")
        obs_label.config(text=f"obs: {_format_obs(obs)}")
        status_label.config(text=f"reward: {reward} done: {done} step: {state['step']}")

        root.after(100, tick)

    def on_close() -> None:
        env.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, tick)
    root.mainloop()


@app.command()
def main(
    n_steps: int = typer.Option(
        200, "--n-steps", help="Ignored in --gui mode; runs until the window closes."
    ),
    port: int = typer.Option(11008, "--port"),
    env_path: str | None = typer.Option(
        None,
        "--env-path",
        help="Path (no platform suffix) to an exported maze_bots binary",
    ),
    action_x: float = typer.Option(0.0, "--action-x", help="Initial direction.x"),
    action_y: float = typer.Option(-1.0, "--action-y", help="Initial direction.y"),
    action_step: float = typer.Option(
        0.1, "--action-step", help="Keyboard nudge size in --gui mode"
    ),
    gui: bool = typer.Option(
        False, "--gui", help="Open a live tkinter window to watch/set the action"
    ),
) -> None:
    """Step a maze_bots Godot instance with a constant action to verify the interface.

    With no --env-path: connects to an already-running `godot-mono` process on a scene
    with `EnableRlSync = true` (e.g. `maze_bots/maps/GoStraight.tscn`) — you launch Godot
    yourself in a second terminal.

    With --env-path: launches and owns the Godot process itself (an exported binary, no
    platform suffix — godot_rl appends .app/.x86_64/.exe based on the host platform), no
    second terminal needed.

    --action-x/--action-y set the starting action (defaults reproduce the original
    hardcoded straight-to-goal action). --gui opens a window that also lets you edit the
    action live via Entry fields or arrow keys/space, alongside the normal terminal
    printout.
    """
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path=env_path, port=port)
    # One row per agent/env instance; rl_agent.gd's action space is a single
    # 2-float continuous "direction" key, and this scene has one agent.
    action = np.array([[action_x, action_y]], dtype=np.float32)

    obs = env.reset()
    print(f"reset obs: {obs}")

    if gui:
        _run_gui(env, action, action_step)
        return

    for step in range(n_steps):
        obs, reward, done, info = env.step(action)
        print(f"step {step}: obs={obs} reward={reward} done={done}")

    env.close()


if __name__ == "__main__":
    app()
