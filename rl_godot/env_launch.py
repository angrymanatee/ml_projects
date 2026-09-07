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

It also fixes a process leak in the upstream launcher: `GodotEnv._launch_env`
Popens the Godot binary with `start_new_session=True` and upstream's own
`GodotEnv.close()` never kills `self.proc` at all — it only sends a graceful
"close" message over the wire socket and trusts Godot to exit on its own. If the
Python process dies before that happens (crash, Ctrl-C, `kill`), the detached
Godot child is reparented to init/launchd and runs forever, holding its port and
RAM. `start_new_session=True` is kept here rather than dropped — upstream never
had a kill path for it to be "protecting", and keeping it means a terminal's
Ctrl-C (SIGINT to the foreground process group only) does *not* reach Godot
directly; only Python does, which is what lets the cleanup below run a graceful
terminate()-then-kill() instead of Godot dying mid-handshake in whatever way its
default signal disposition happens to do. It also lets `os.killpg` below reach
any child processes Godot itself spawns, which a shared-process-group child
would not (killpg would then hit Python too). What was missing is the actual
cleanup: every process this module launches is tracked in `_LAUNCHED_PROCESSES`
and reaped by `_cleanup_launched_processes`, wired to both `atexit` (covers
normal exit, an unhandled exception, and SIGINT once Python turns it into a
`KeyboardInterrupt`) and an explicit `SIGTERM` handler (Python's default SIGTERM
disposition kills the process immediately *without* running atexit handlers, so
atexit alone would miss a plain `kill <pid>`).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import sys
from sys import platform

import godot_rl.wrappers.stable_baselines_wrapper as stable_baselines_wrapper
from godot_rl.core.godot_env import GodotEnv
from godot_rl.core.utils import convert_macos_path

_LAUNCHED_PROCESSES: list[subprocess.Popen] = []
_TERMINATE_TIMEOUT_SECONDS = 5.0


def _terminate_launched_process(proc: subprocess.Popen) -> None:
    """Terminate one launched Godot process (and its process group), if still alive.

    `proc.poll()` reflects real OS process state regardless of whether
    `GodotEnv.close()`'s graceful "close" message already ran, so a process that
    exited cleanly on its own is simply skipped here rather than double-killed.
    Sends SIGTERM to the whole process group first (`start_new_session=True` put
    Godot in its own group, catching any children it spawned too) and only
    escalates to SIGKILL if it hasn't exited within `_TERMINATE_TIMEOUT_SECONDS`.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)


def _cleanup_launched_processes() -> None:
    """Terminate every Godot process `_launch_env` has ever launched, if still alive."""
    while _LAUNCHED_PROCESSES:
        _terminate_launched_process(_LAUNCHED_PROCESSES.pop())


def _handle_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into a normal interpreter exit so the atexit cleanup runs.

    Unlike SIGINT (which Python's default handler turns into a catchable
    `KeyboardInterrupt`, unwinding normally into atexit), SIGTERM's default
    disposition kills the process immediately with no atexit pass at all — so a
    `kill <pid>`'d training run would otherwise leak its Godot children.
    """
    sys.exit(128 + signum)


atexit.register(_cleanup_launched_processes)
signal.signal(signal.SIGTERM, _handle_sigterm)


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
        _LAUNCHED_PROCESSES.append(self.proc)


stable_baselines_wrapper.GodotEnv = _GodotEnvWithUserArgs
