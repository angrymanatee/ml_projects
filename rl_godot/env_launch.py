"""Godot process launching with support for maze_bots' post-`--` user args.

`godot_rl.core.godot_env.GodotEnv._launch_env` appends extra kwargs as bare
`--key=value` engine flags, with no `--` separator before them. maze_bots reads its
config override, map selection, and layout randomization overrides via
`OS.GetCmdlineUserArgs()` (`maze_bots/player/controllers/RLConfig.cs`, `maze_bots/
maps/MapSelection.cs`, `maze_bots/maps/MapBase.cs`), which only sees args placed
after Godot's own `--` separator — so the stock kwargs mechanism can never reach
them. Importing this module patches in a `GodotEnv` subclass (used by
`StableBaselinesGodotEnv`) that inserts that separator when an `rl_config`,
`map_name`, `layout_config`, or `layout_seed` kwarg is present.
"""

from __future__ import annotations

import subprocess
from sys import platform

import godot_rl.wrappers.stable_baselines_wrapper as stable_baselines_wrapper
from godot_rl.core.godot_env import GodotEnv
from godot_rl.core.utils import convert_macos_path


def build_launch_cmd(
    env_path: str,
    port: int,
    seed: int,
    show_window: bool,
    framerate: int | None,
    action_repeat: int | None,
    speedup: int | None,
    rl_config: str | None,
    map_name: str | None = None,
    layout_config: str | None = None,
    layout_seed: int | None = None,
    **kwargs: object,
) -> str:
    """Build the Godot command line, mirroring `GodotEnv._launch_env`.

    `rl_config`, if given, is a config name under `maze_bots/configs/` with no `.cfg`
    extension (e.g. `"no_rays"`). `map_name`, if given, is a scene name under
    `maze_bots/maps/` with no `.tscn` extension (e.g. `"GoStraightTrap"`) or a full
    `res://` scene path — `MapSelection.cs` accepts either, so it is passed through
    verbatim. `layout_config`, if given, is a layout randomization config name under
    `maze_bots/configs/` with no `.cfg` extension (e.g. `"layout_GoStraightTrap"`).
    `layout_seed`, if given, pins every episode to one fixed layout instead of
    letting `MapBase` derive a per-episode seed from `--env_seed`. All four are
    appended after a `--` separator so maze_bots picks them up via
    `OS.GetCmdlineUserArgs()`.
    """
    path = convert_macos_path(env_path) if platform == "darwin" else env_path
    launch_cmd = f"{path} --port={port} --env_seed={seed}"
    if show_window is False:
        launch_cmd += " --disable-render-loop --headless"
    if framerate is not None:
        launch_cmd += f" --fixed-fps {framerate}"
    if action_repeat is not None:
        launch_cmd += f" --action_repeat={action_repeat}"
    if speedup is not None:
        launch_cmd += f" --speedup={speedup}"
    for key, value in kwargs.items():
        launch_cmd += f" --{key}={value}"

    user_args = []
    if rl_config is not None:
        user_args.append(f"--rl-config=res://configs/{rl_config}.cfg")
    if map_name is not None:
        user_args.append(f"--map={map_name}")
    if layout_config is not None:
        user_args.append(f"--layout-config=res://configs/{layout_config}.cfg")
    if layout_seed is not None:
        user_args.append(f"--layout-seed={layout_seed}")
    if user_args:
        launch_cmd += " -- " + " ".join(user_args)
    return launch_cmd


class _GodotEnvWithUserArgs(GodotEnv):
    def _launch_env(
        self,
        env_path,
        port,
        show_window,
        framerate,
        seed,
        action_repeat,
        speedup,
        **kwargs,
    ) -> None:
        rl_config = kwargs.pop("rl_config", None)
        map_name = kwargs.pop("map_name", None)
        layout_config = kwargs.pop("layout_config", None)
        layout_seed = kwargs.pop("layout_seed", None)
        launch_cmd = build_launch_cmd(
            env_path,
            port,
            seed,
            show_window,
            framerate,
            action_repeat,
            speedup,
            rl_config,
            map_name,
            layout_config,
            layout_seed,
            **kwargs,
        )
        self.proc = subprocess.Popen(launch_cmd.split(" "), start_new_session=True)


stable_baselines_wrapper.GodotEnv = _GodotEnvWithUserArgs
