from __future__ import annotations

import subprocess
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


def build_env(maze_bots_path: Path, export_type: ExportType) -> tuple[Path, Path]:
    """Run maze_bots/scripts/export.sh, returning the (macos, linux) binary paths.

    Shared by this module's CLI and the godot_rl launchers' `--build` step.
    """
    export_script = maze_bots_path / "scripts" / "export.sh"
    if not export_script.exists():
        raise FileNotFoundError(
            f"export.sh not found at {export_script} — check --maze-bots-path"
        )

    subprocess.run([str(export_script), export_type], check=True, cwd=maze_bots_path)

    return (
        maze_bots_path / "build" / "macos" / "MazeBots",
        maze_bots_path / "build" / "linux" / "MazeBots.x86_64",
    )


@app.command()
def main(
    maze_bots_path: Path = MAZE_BOTS_PATH_OPTION,
    export_type: ExportType = ExportType.DEBUG,
) -> None:
    """Export the maze_bots Godot world to headless-runnable macOS/Linux binaries.

    Thin wrapper around maze_bots/scripts/export.sh — runs it from this repo so you
    don't need to switch checkouts or remember the godot-mono export flags. On
    success, prints the exported binary paths and the exact
    rl_godot.constant_action_check command to run against them.
    """
    try:
        macos_env_path, linux_binary = build_env(maze_bots_path, export_type)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc

    print()
    print(f"macOS binary: {macos_env_path}.app")
    print(f"Linux binary: {linux_binary}")
    print()
    print("Run it:")
    print(
        f"  uv run python -m rl_godot.constant_action_check --env-path {macos_env_path}"
    )


if __name__ == "__main__":
    app()
