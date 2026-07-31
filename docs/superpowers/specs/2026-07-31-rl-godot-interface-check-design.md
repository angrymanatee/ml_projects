# RL-Godot Interface Check — Design

**Date:** 2026-07-31
**Status:** Approved
**Scope:** Python-side plumbing to drive the `maze_bots` Godot RL Agents bridge with a constant
action, proving the obs/reward/action round trip end to end.

---

## Overview

`maze_bots` (`/Users/sauron/GodotProjects/maze_bots`, a separate git repo) already has a working
C#/GDScript bridge (`RLController` ↔ `rl_agent.gd`, an `AIController3D`) that exposes a flattened
goal-position observation, a scalar reward, and a 2-float continuous `direction` action over the
vendored `godot_rl_agents` addon's `Sync`/TCP protocol. What's never existed is a `Sync` node
actually placed in a scene (opening the training port) or any Python-side code to talk to it —
both were explicitly out of scope in the prior maze_bots design.

This spec covers only the MLProjects (Python) half: a new `rl_godot/` project directory that
connects to a manually-launched, already-running `godot-mono` instance and drives it with a
hardcoded constant action for a fixed number of steps, to prove the wire protocol and the
maze_bots bridge work together before any real training or RunPod deployment is attempted. The
maze_bots-side `Sync` node placement is a separate spec, in that repo.

Out of scope here:
- Any real training algorithm/policy (stable-baselines3, PPO, etc.) — that's a follow-on project.
- RunPod deployment — deferred to a second phase/spec once this interface check passes locally.
- Anything inside `maze_bots` itself.

---

## Why a new top-level directory, not reusing `time_series`/`remote`

- `godot-rl` (the PyPI package) pulls in `gymnasium` and is unrelated to the torch/MLflow/RunPod
  stack `time_series` uses. It gets its own dependency group so it never lands on a `time_series`
  GPU pod's install list.
- The directory is named `rl_godot/`, **not** `godot_rl/` — `pyproject.toml` sets
  `pythonpath = ["."]`, so a top-level `godot_rl/` directory would shadow the installed `godot_rl`
  package on `sys.path` and break `import godot_rl` inside the script itself.

---

## Module structure

```
rl_godot/
    __init__.py
    constant_action_check.py   # manual smoke-test script (not a pytest)
```

New `pyproject.toml` dependency group:

```toml
[dependency-groups]
rl_godot = [
    "godot-rl==0.8.2",
]
```

`0.8.2` is confirmed still the current latest release on PyPI (last published Feb 2025) —
unrelated to the `.venv`/`requirements.txt` being removed from the `maze_bots` repo, which pinned
the same version by mistake (a stray Python virtualenv checked into a Godot project, not a
deliberate version choice).

---

## `constant_action_check.py`

Connects to an **already-running** `godot-mono` instance over TCP using `godot_rl`'s Stable
Baselines gym wrapper (`godot_rl.wrappers.stable_baselines_wrapper.StableBaselinesGodotEnv`) with
`env_path=None` — the library's documented mode for pointing at a live instance instead of
launching one itself, meant for exactly this kind of manual/interactive check.

Flow:
1. Construct the env (`env_path=None`, default port 11008).
2. `reset()`.
3. For a fixed `n_steps` (e.g. 200), step with a hardcoded constant action
   (`np.array([1.0, 0.0], dtype=np.float32)`, matching `rl_agent.gd`'s single `"direction"` action
   key) and print `obs`, `reward`, `done`/`terminated`, and step index.
4. Exit cleanly (`env.close()`) whether or not any episode ended, after `n_steps`.

No error handling beyond what the library already does — this is a throwaway diagnostic script,
not production training code. Success criteria (checked manually, not asserted in code):
- The script runs `n_steps` without an exception.
- Observation values are finite and change across steps (the player is actually moving in the
  commanded direction).
- No handshake/version errors beyond the expected harmless major/minor mismatch warning
  documented in the maze_bots spec (vendored addon pins protocol `0.7`, pip package is `0.8.2`).

This script requires a live Godot process with a `Sync` node enabled and listening — it cannot run
under `pytest` and won't be added to the test suite. It's invoked manually:

```bash
uv run python -m rl_godot.constant_action_check
```

---

## Docs

- `docs/rl_godot_project.md` — new project reference, mirroring the structure of
  `docs/store_sales_project.md` (what it is, how to run it, current status).
- `README.md` — add a row to the project table pointing at the new doc.
