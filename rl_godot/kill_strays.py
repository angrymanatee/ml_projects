"""Recovery for leaked maze_bots RL processes that outlived their Python parent.

`rl_godot.env_launch` cleans up normal exits, unhandled exceptions, SIGINT, and
SIGTERM via a live-process registry (see its module docstring), but nothing can
catch a SIGKILL or a hard crash of the Python parent — the Godot child, already
detached into its own session (`start_new_session=True`), is simply reparented to
launchd and keeps running, holding its port and RAM. This module finds and
terminates those strays after the fact, by command line.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

import typer

app = typer.Typer(add_completion=False)

# Never match the user's Godot *editor* -- it is launched by hand and must survive
# this tool. `Godot_mono.app` is the editor's bundle name on this machine;
# maze_bots' exported binary is a bare executable (macOS: an .app it did NOT get
# from the editor install; Linux: a raw x86_64 file), never this path.
_EDITOR_MARKER = "Godot_mono.app"

_GRACE_PERIOD_SECONDS = 5.0


def _is_stray_command(command: str) -> bool:
    """True if `command` is a launched (non-editor) maze_bots RL process.

    `env_launch.build_launch_cmd` always emits `--port=<n> --env_seed=<n>` on the
    exported binary's own command line -- that pair, together, is specific to a
    process this codebase launched (either flag alone could plausibly appear on
    something unrelated). The editor never carries either flag, but it is still
    excluded explicitly by name as a second, independent guard.
    """
    if _EDITOR_MARKER in command:
        return False
    return "--port=" in command and "--env_seed=" in command


def find_strays() -> list[tuple[int, str]]:
    """Return (pid, command) for every live stray-matching process."""
    result = subprocess.run(
        ["ps", "-eo", "pid,command", "-ww"],
        capture_output=True,
        text=True,
        check=True,
    )
    strays = []
    for line in result.stdout.splitlines()[1:]:  # header row
        line = line.strip()
        if not line:
            continue
        pid_str, _, command = line.partition(" ")
        if not pid_str.isdigit():
            continue
        if _is_stray_command(command):
            strays.append((int(pid_str), command))
    return strays


def kill_strays(
    strays: list[tuple[int, str]], grace_period: float = _GRACE_PERIOD_SECONDS
) -> None:
    """SIGTERM every stray, wait one shared grace period, then SIGKILL survivors.

    A shared grace period (rather than one per process) keeps this O(1) sleeps
    regardless of how many strays there are.
    """
    for pid, _ in strays:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    if strays and grace_period > 0:
        time.sleep(grace_period)
    for pid, _ in strays:
        try:
            os.kill(pid, 0)  # still alive?
        except ProcessLookupError:
            continue
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@app.command()
def main(
    grace_period: float = typer.Option(
        _GRACE_PERIOD_SECONDS,
        "--grace-period",
        help="Seconds to wait after SIGTERM before escalating to SIGKILL.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List strays that would be killed, without killing them.",
    ),
) -> None:
    """Find and terminate leftover maze_bots RL processes from a dead parent.

    Matches by command line (the exported binary path plus --port=/--env_seed=,
    which only an env_launch-launched process carries) and never touches the
    Godot editor.
    """
    strays = find_strays()
    if not strays:
        typer.echo("No stray processes found.")
        return
    for pid, command in strays:
        typer.echo(f"{'would kill' if dry_run else 'killing'} pid={pid}: {command}")
    if dry_run:
        return
    kill_strays(strays, grace_period=grace_period)
    typer.echo(f"Killed {len(strays)} stray process(es).")


if __name__ == "__main__":
    app()
