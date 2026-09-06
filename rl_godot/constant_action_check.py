from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import typer

from rl_godot.export_env import MAZE_BOTS_PATH_OPTION, ExportType, build_env

if TYPE_CHECKING:
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

app = typer.Typer(add_completion=False)


def _format_obs(obs: dict[str, np.ndarray]) -> str:
    parts = []
    for key, value in obs.items():
        vec = np.asarray(value).reshape(-1)
        parts.append(f"{key}=[{', '.join(f'{v:.3f}' for v in vec)}]")
    return " ".join(parts)


def _format_obs_lines(obs: dict[str, np.ndarray], values_per_row: int = 8) -> str:
    """Same content as `_format_obs`, one key per line for GUI display.

    Long vectors (e.g. a 48-element ray fan) are wrapped across multiple rows,
    `values_per_row` values each, so a single key never produces a giant line.
    `rays` is special-cased to 3 values per row: `RayFanSensor` packs one
    float per class layer (world/enemy/trap) per ray, so 3-per-row puts one
    ray on each line regardless of `values_per_row`.
    """
    lines = []
    for key, value in obs.items():
        vec = np.asarray(value).reshape(-1)
        formatted = [f"{v:.3f}" for v in vec]
        row_size = 3 if key == "rays" else values_per_row
        rows = [formatted[i : i + row_size] for i in range(0, len(formatted), row_size)]
        indent = " " * (len(key) + 2)
        for row_idx, row in enumerate(rows):
            prefix = f"{key}: [" if row_idx == 0 else f"{indent} "
            suffix = "]" if row_idx == len(rows) - 1 else ","
            lines.append(f"{prefix}{', '.join(row)}{suffix}")
    return "\n".join(lines)


def _format_step(step: int, obs: dict[str, np.ndarray], reward, done) -> str:
    reward_value = float(np.asarray(reward).reshape(-1)[0])
    done_value = bool(np.asarray(done).reshape(-1)[0])
    return (
        f"step {step:>4} | reward={reward_value:+.3f} done={done_value!s:<5} | "
        f"{_format_obs(obs)}"
    )


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

    mono = ("Menlo", 12)
    mono_bold = ("Menlo", 12, "bold")

    state = {"step": 0}

    status_frame = tk.Frame(root)
    status_frame.pack(anchor="w", padx=8, pady=(8, 4))

    tk.Label(status_frame, text="step:", font=mono_bold).grid(
        row=0, column=0, sticky="w"
    )
    step_value = tk.Label(status_frame, text="0", font=mono)
    step_value.grid(row=0, column=1, sticky="w", padx=(4, 16))

    tk.Label(status_frame, text="reward:", font=mono_bold).grid(
        row=0, column=2, sticky="w"
    )
    reward_value_label = tk.Label(status_frame, text="-", font=mono)
    reward_value_label.grid(row=0, column=3, sticky="w", padx=(4, 16))

    tk.Label(status_frame, text="done:", font=mono_bold).grid(
        row=0, column=4, sticky="w"
    )
    done_value_label = tk.Label(status_frame, text="-", font=mono)
    done_value_label.grid(row=0, column=5, sticky="w")

    tk.Label(root, text="action:", font=mono_bold).pack(anchor="w", padx=8)
    action_label = tk.Label(root, text="x=- y=- rotation=-", font=mono, justify="left")
    action_label.pack(anchor="w", padx=16)

    tk.Label(root, text="obs:", font=mono_bold).pack(anchor="w", padx=8, pady=(4, 0))
    obs_label = tk.Label(root, text="-", font=mono, justify="left")
    obs_label.pack(anchor="w", padx=16, pady=(0, 8))

    entry_frame = tk.Frame(root)
    entry_frame.pack(anchor="w", padx=8, pady=(0, 8))
    tk.Label(entry_frame, text="x:").pack(side="left")
    x_entry = tk.Entry(entry_frame, width=8)
    x_entry.insert(0, str(action[0, 0]))
    x_entry.pack(side="left")
    tk.Label(entry_frame, text="y:").pack(side="left", padx=(8, 0))
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
        side="left", padx=(8, 0)
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
        print(_format_step(state["step"], obs, reward, done))

        reward_value = float(np.asarray(reward).reshape(-1)[0])
        done_value = bool(np.asarray(done).reshape(-1)[0])
        step_value.config(text=str(state["step"]))
        reward_value_label.config(text=f"{reward_value:+.3f}")
        done_value_label.config(text=str(done_value))
        action_label.config(
            text=f"x={action[0, 0]:.3f} y={action[0, 1]:.3f} rotation={action[0, 2]:.3f}"
        )
        obs_label.config(text=_format_obs_lines(obs))

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
    action_x: float = typer.Option(1.0, "--action-x", help="Initial direction.x"),
    action_y: float = typer.Option(0.0, "--action-y", help="Initial direction.y"),
    action_rotation: float = typer.Option(
        0.0, "--action-rotation", help="Constant turn value"
    ),
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
    second terminal needed. By default this also re-exports maze_bots from
    --maze-bots-path first (--build/--no-build); pass --no-build to launch an
    already-built binary as-is.

    --map selects which scene under maze_bots/maps/ to load (default GoStraight);
    like --rl-config it only applies when this process launches Godot (--env-path).

    --action-x/--action-y set the starting movement action (defaults reproduce the
    original hardcoded straight-to-goal action); --action-rotation sets the constant
    turn value. --gui opens a window that also lets you edit the movement action live
    via Entry fields or arrow keys/space (rotation stays fixed), alongside the normal
    terminal printout.
    """
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    import rl_godot.env_launch  # noqa: F401 — patches StableBaselinesGodotEnv for user args

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

    env = StableBaselinesGodotEnv(
        env_path=env_path,
        port=port,
        rl_config=rl_config,
        map_name=map_name,
        layout_config=layout_config,
        layout_seed=layout_seed,
    )
    # One row per agent/env instance; rl_agent.gd's action space is a 2-float
    # continuous "movement" key plus a 1-float "rotation" key, and this scene
    # has one agent.
    action = np.array([[action_x, action_y, action_rotation]], dtype=np.float32)

    obs = env.reset()
    print(f"reset obs: {_format_obs(obs)}")

    if gui:
        _run_gui(env, action, action_step)
        return

    for step in range(n_steps):
        obs, reward, done, info = env.step(action)
        print(_format_step(step, obs, reward, done))

    env.close()


if __name__ == "__main__":
    app()
