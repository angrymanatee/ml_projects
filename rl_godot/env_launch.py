"""Godot process launching with support for maze_bots' `--rl-config` pass-through.

`godot_rl.core.godot_env.GodotEnv._launch_env` appends extra kwargs as bare
`--key=value` engine flags, with no `--` separator before them. maze_bots reads its
config override via `OS.GetCmdlineUserArgs()` (`maze_bots/player/controllers/
RLConfig.cs`), which only sees args placed after Godot's own `--` separator — so the
stock kwargs mechanism can never reach it. Importing this module patches in a
`GodotEnv` subclass (used by `StableBaselinesGodotEnv`) that inserts that separator
when an `rl_config` kwarg is present.
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
    **kwargs: object,
) -> str:
    """Build the Godot command line, mirroring `GodotEnv._launch_env`.

    `rl_config`, if given, is a config name under `maze_bots/configs/` with no `.cfg`
    extension (e.g. `"no_rays"`) and is appended after a `--` separator so maze_bots'
    `RLConfig.cs` picks it up via `OS.GetCmdlineUserArgs()`.
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
    if rl_config is not None:
        launch_cmd += f" -- --rl-config=res://configs/{rl_config}.cfg"
    return launch_cmd


class _GodotEnvWithRlConfig(GodotEnv):
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
        launch_cmd = build_launch_cmd(
            env_path,
            port,
            seed,
            show_window,
            framerate,
            action_repeat,
            speedup,
            rl_config,
            **kwargs,
        )
        self.proc = subprocess.Popen(launch_cmd.split(" "), start_new_session=True)


stable_baselines_wrapper.GodotEnv = _GodotEnvWithRlConfig
