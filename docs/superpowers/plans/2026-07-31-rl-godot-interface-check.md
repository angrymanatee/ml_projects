# RL-Godot Interface Check — Implementation Plan

**Goal:** Add a `rl_godot/` project to MLProjects that connects to a manually-launched `maze_bots`
Godot instance and drives it with a constant action, proving the Python <-> Godot wire protocol and
the existing C#/GDScript bridge work together.

Full design rationale: `docs/superpowers/specs/2026-07-31-rl-godot-interface-check-design.md`.

## Global Constraints

- `rl_godot/constant_action_check.py` is a manual smoke-test script, not a pytest target — it
  requires a live `godot-mono` process with a `Sync` node listening. No `tests/rl_godot/` directory
  is created for it; this deviates from the repo's usual "every module gets a mirrored test" rule
  deliberately (see design spec's testing rationale).
- New directory is `rl_godot/`, not `godot_rl/` — `pyproject.toml`'s `pythonpath = ["."]` would
  otherwise shadow the installed `godot_rl` package.
- Depends on the `maze_bots` repo's own `2026-07-31-godot-rl-sync-node` plan being implemented
  first (a `Sync` node and a training-enabled map need to exist there before this can be run
  end-to-end) — but the Python code itself can be written and reviewed independently.

---

### Task 1: Add `rl_godot` dependency group

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency group**

```toml
[dependency-groups]
rl_godot = [
    "godot-rl==0.8.2",
]
```

- [ ] **Step 2: Sync and verify**

```bash
uv sync --group rl_godot
uv run python -c "import godot_rl; print(godot_rl.__file__)"
```
Expected: prints a path under `.venv/`, no `ModuleNotFoundError`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add rl_godot dependency group with godot-rl==0.8.2"
```

---

### Task 2: `rl_godot/constant_action_check.py`

**Files:**
- Create: `rl_godot/__init__.py` (empty)
- Create: `rl_godot/constant_action_check.py`

**Interfaces:**
- Consumes: `godot_rl.wrappers.stable_baselines_wrapper.StableBaselinesGodotEnv`.
- Produces: a runnable module `python -m rl_godot.constant_action_check` with `--n-steps` and
  `--port` CLI flags (default port 11008, matching `sync.gd`'s `DEFAULT_PORT`).

- [ ] **Step 1: Write the script**

```python
from __future__ import annotations

import typer
import numpy as np

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n_steps: int = typer.Option(200, "--n-steps"),
    port: int = typer.Option(11008, "--port"),
) -> None:
    """Step a live maze_bots Godot instance with a constant action to verify the interface.

    Requires a `godot-mono` process already running a scene with `EnableRlSync = true`
    (e.g. `maze_bots/maps/GoStraightTraining.tscn`) — this script connects to it rather
    than launching it.
    """
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    env = StableBaselinesGodotEnv(env_path=None, port=port)
    action = np.array([1.0, 0.0], dtype=np.float32)

    obs = env.reset()
    print(f"reset obs: {obs}")
    for step in range(n_steps):
        obs, reward, done, info = env.step([action])
        print(f"step {step}: obs={obs} reward={reward} done={done}")

    env.close()


if __name__ == "__main__":
    app()
```

Note: the exact `StableBaselinesGodotEnv` constructor kwargs and `step()` return shape (3-tuple vs.
gymnasium's 5-tuple `obs, reward, terminated, truncated, info`) depend on what `godot-rl==0.8.2`
actually exposes — confirm against the installed package's source
(`.venv/lib/python3.14/site-packages/godot_rl/wrappers/stable_baselines_wrapper.py`) in Step 2
below and adjust before treating this as final; this is exactly the kind of detail the "run it and
see" step exists to catch, same as the maze_bots plan's GdUnit4 API uncertainty.

- [ ] **Step 2: Confirm the wrapper's actual API**

```bash
uv run python -c "import inspect; from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv; print(inspect.signature(StableBaselinesGodotEnv.__init__)); print(inspect.signature(StableBaselinesGodotEnv.step))"
```
Adjust Step 1's script to match the real signature (env_path arg name, step() return arity) if it
differs from the sketch above.

- [ ] **Step 3: Manual end-to-end run**

Requires the `maze_bots` `Sync`-node plan (separate repo) already implemented. Two terminals:

```bash
# Terminal 1 (maze_bots repo)
godot-mono --headless --path . maps/GoStraightTraining.tscn

# Terminal 2 (MLProjects repo)
uv run python -m rl_godot.constant_action_check
```

Expected: the Godot console logs a completed handshake (not the "using human controls instead"
fallback), the Python script prints `n_steps` obs/reward lines with finite, changing values (the
player visibly moving in the `+X` direction if the editor window is open instead of `--headless`),
and both processes exit cleanly.

- [ ] **Step 4: Commit**

```bash
git add rl_godot/
git commit -m "Add rl_godot constant-action interface check script"
```

---

### Task 3: Docs

**Files:**
- Create: `docs/rl_godot_project.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/rl_godot_project.md`**

Mirror `docs/store_sales_project.md`'s structure: what it is, current status (interface
verification only, no real training yet), how to run the manual check, pointer to the maze_bots
repo and its own design docs.

- [ ] **Step 2: Add a row to `README.md`'s project table**

```
| `rl_godot/` | Godot RL Agents interface check against `maze_bots` | [`docs/rl_godot_project.md`](docs/rl_godot_project.md) |
```

- [ ] **Step 3: Commit**

```bash
git add docs/rl_godot_project.md README.md
git commit -m "Add rl_godot project docs"
```
