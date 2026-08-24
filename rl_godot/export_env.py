from __future__ import annotations

import subprocess
from enum import StrEnum, auto
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


class ExportType(StrEnum):
    DEBUG = auto()
    RELEASE = auto()


_MAZE_BOTS_PATH_OPTION = typer.Option(
    Path("/Users/sauron/GodotProjects/maze_bots"),
    "--maze-bots-path",
    help="Path to the maze_bots Godot project checkout",
)


@app.command()
def main(
    maze_bots_path: Path = _MAZE_BOTS_PATH_OPTION,
    export_type: ExportType = ExportType.DEBUG,
) -> None:
    """Export the maze_bots Godot world to headless-runnable macOS/Linux binaries.

    Thin wrapper around maze_bots/scripts/export.sh — runs it from this repo so you
    don't need to switch checkouts or remember the godot-mono export flags. On
    success, prints the exported binary paths and the exact
    rl_godot.constant_action_check command to run against them.
    """
    export_script = maze_bots_path / "scripts" / "export.sh"
    if not export_script.exists():
        raise typer.BadParameter(
            f"export.sh not found at {export_script} — check --maze-bots-path"
        )

    try:
        subprocess.run(
            [str(export_script), export_type], check=True, cwd=maze_bots_path
        )
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc

    macos_env_path = maze_bots_path / "build" / "macos" / "MazeBots"
    linux_binary = maze_bots_path / "build" / "linux" / "MazeBots.x86_64"

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
