# rl_godot `--env-path` Option — Implementation Plan

**Goal:** Let `constant_action_check.py` launch Godot itself via `env_path`, eliminating the
second terminal from Phase 1's manual two-terminal flow.

Full design rationale: `docs/superpowers/specs/2026-07-31-rl-godot-env-path-design.md`.

## Global Constraints

- Still a manual smoke-test script, no pytest coverage — same rationale as Phase 1.
- Depends on maze_bots's own `2026-07-31-godot-export-binary` plan being implemented first for
  end-to-end verification (Step 3 below), but the code change itself doesn't need the binary to
  exist to be written.

---

### Task 1: Add `--env-path` option

**Files:**
- Modify: `rl_godot/constant_action_check.py`

- [ ] **Step 1: Add the option and thread it through**
  ```python
  env_path: str | None = typer.Option(
      None, "--env-path", help="Path (no platform suffix) to an exported maze_bots binary"
  ),
  ```
  and change:
  ```python
  env = StableBaselinesGodotEnv(env_path=env_path, port=port)
  ```
  Also correct the docstring, which currently references the now-deleted
  `maps/GoStraightTraining.tscn` — `maps/GoStraight.tscn` itself has `EnableRlSync = true` now
  (per the "edit the normal map instead of duplicating it" decision made earlier), so the
  docstring's example scene name is stale.

- [ ] **Step 2: Lint**
  ```bash
  uv run black rl_godot/ && uv run ruff check --fix rl_godot/
  ```

- [ ] **Step 3: Manual end-to-end verification**
  Requires maze_bots's export binary to exist (separate repo, separate plan). Single terminal:
  ```bash
  uv run python -m rl_godot.constant_action_check \
    --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots
  ```
  Expected: Godot launches automatically (no second terminal needed), connects, steps `n_steps`,
  and both the script and the launched Godot process exit cleanly on their own.

- [ ] **Step 4: Commit**
  ```bash
  git add rl_godot/constant_action_check.py
  git commit -m "Add --env-path option to launch Godot directly"
  ```
