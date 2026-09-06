from __future__ import annotations

import subprocess
import sys
from enum import StrEnum, auto
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


class ExportType(StrEnum):
    DEBUG = auto()
    RELEASE = auto()


MAZE_BOTS_PATH_OPTION = typer.Option(
    Path("/Users/sauron/GodotProjects/maze_bots"),
    "--maze-bots-path",
    help="Path to the maze_bots Godot project checkout",
)


# Only the host platform's binary is exported: the launchers run the Godot process
# locally, and the pod-side (Linux) binary is cross-exported by hand and rsynced —
# see `remote/sync.py`.
_HOST_TARGETS = {
    "darwin": ("macos", Path("build") / "macos" / "MazeBots"),
    "linux": ("linux", Path("build") / "linux" / "MazeBots.x86_64"),
}


def build_env(maze_bots_path: Path, export_type: ExportType) -> Path:
    """Run maze_bots/scripts/export.sh for this machine, returning the binary path.

    Shared by this module's CLI and the godot_rl launchers' `--build` step.

    Raises:
        FileNotFoundError: If export.sh is missing under maze_bots_path.
        RuntimeError: If the host platform has no maze_bots export preset.
    """
    export_script = maze_bots_path / "scripts" / "export.sh"
    if not export_script.exists():
        raise FileNotFoundError(
            f"export.sh not found at {export_script} — check --maze-bots-path"
        )

    if sys.platform not in _HOST_TARGETS:
        raise RuntimeError(f"No maze_bots export target for platform {sys.platform}")
    target, relative_binary = _HOST_TARGETS[sys.platform]

    subprocess.run(
        [str(export_script), export_type, target], check=True, cwd=maze_bots_path
    )

    return maze_bots_path / relative_binary


@app.command()
def main(
    maze_bots_path: Path = MAZE_BOTS_PATH_OPTION,
    export_type: ExportType = ExportType.DEBUG,
) -> None:
    """Export the maze_bots Godot world to a headless-runnable binary for this machine.

    Thin wrapper around maze_bots/scripts/export.sh — runs it from this repo so you
    don't need to switch checkouts or remember the godot-mono export flags. On
    success, prints the exported binary path and the exact
    rl_godot.constant_action_check command to run against it. To cross-export the
    Linux binary the RunPod flow ships, run `scripts/export.sh <type> linux` in the
    maze_bots checkout.
    """
    try:
        env_path = build_env(maze_bots_path, export_type)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc

    print()
    print(f"Binary: {env_path}")
    print()
    print("Run it:")
    print(f"  uv run python -m rl_godot.constant_action_check --env-path {env_path}")


if __name__ == "__main__":
    app()
