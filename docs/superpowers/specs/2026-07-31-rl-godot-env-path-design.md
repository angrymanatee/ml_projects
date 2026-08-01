# rl_godot `--env-path` Option — Design

**Scope:** Let `constant_action_check.py` launch and manage the `maze_bots` Godot process itself,
instead of requiring a manually-started second terminal.

## Overview

Phase 1 required two terminals: one running `godot-mono --headless --path . maps/GoStraight.tscn`
by hand, one running this script. `godot_rl`'s `GodotEnv` already supports launching the game
itself when constructed with `env_path=<path to an exported binary>` (traced through
`core/godot_env.py._launch_env()` earlier this session) — it runs the binary as a subprocess with
`--port`/`--env_seed`/`--headless` etc. flags matching what `Sync.gd` expects, and tears it down on
`close()`. This spec wires that up, consuming the exported binary produced by maze_bots's own
`2026-07-31-godot-export-binary-design.md`.

Out of scope: the export build itself (maze_bots repo, separate spec); `n_parallel > 1` (unlocked
by this but not exercised here); anything RunPod-related (Phase 3).

## Change

`rl_godot/constant_action_check.py` gets a new Typer option:

```python
env_path: str | None = typer.Option(
    None, "--env-path", help="Path (no platform suffix) to an exported maze_bots binary"
)
```

Passed straight through: `StableBaselinesGodotEnv(env_path=env_path, port=port)`. When `None`
(the default), behavior is unchanged from Phase 1 — connect to a manually-launched instance.
When set, `godot_rl` launches and owns the process; the script no longer needs a second terminal
at all, and `env.close()` (already called at the end of the script) tears the launched process
down.

No path defaulting/discovery logic (e.g. auto-finding `../maze_bots/build/macos/MazeBots`) —
the two repos are independent, sibling directories with no fixed relative relationship assumed
elsewhere in this codebase; the caller passes an explicit path. YAGNI: add a config default later
if typing the full path repeatedly becomes annoying in practice.

## Testing

Still a manual smoke-test script, per Phase 1's rationale — no pytest coverage. Verification: once
maze_bots's export exists, run
`uv run python -m rl_godot.constant_action_check --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots`
from a single terminal and confirm it launches Godot, connects, steps `n_steps`, and both processes
exit cleanly with no manual intervention.
