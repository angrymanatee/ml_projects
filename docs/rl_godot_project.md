# RL-Godot — Project Reference

Reinforcement-learning interface between this repo and `maze_bots` — a separate Godot 4.7
(Mono/C#) project at `/Users/sauron/GodotProjects/maze_bots`, purpose-built as an RL training
environment. This project is Python-only; the game itself, its C#/GDScript RL bridge, and the
Godot-side `godot_rl_agents` addon all live in that other repo.

**Status: interface verification only.** No real training (PPO, or any actual policy) exists yet.
The current goal is proving the wire protocol works end to end — a constant action sent from
Python actually moves the player in `maze_bots` and observations/rewards flow back correctly —
before building anything on top of it.

**If you notice a discrepancy between this file and the code, update this file.**

---

## How the two repos fit together

- `maze_bots` has a `Sync` node (from the vendored `godot_rl_agents` addon) that opens a TCP
  server (default port `11008`) when a scene has `EnableRlSync = true` on its `GameRunnerBase`
  (see that repo's `docs/superpowers/specs/2026-07-31-godot-rl-sync-node-design.md`).
- That `Sync` node drives an `RLController`/`rl_agent.gd` bridge already built there: observations
  are the goal's flattened `[X, Y, Z]` position, actions are a single 2-float continuous
  `"direction"` vector, and reward is whatever `ControllerBase.reward` accumulates.
- This repo's `rl_godot/` is the Python side of that TCP connection — nothing more yet.

## `rl_godot/constant_action_check.py`

Connects to an **already-running** `godot-mono` process (it does not launch Godot itself —
`env_path=None`), resets, and steps a fixed number of times with a hardcoded constant action
(`direction=[1.0, 0.0]`), printing obs/reward/done each step. This is a manual smoke-test script,
not covered by `pytest` — it needs a live Godot process on the other end of the socket.

### Running it

Two terminals:

```bash
# Terminal 1 — in the maze_bots repo
godot-mono --headless --path . maps/GoStraightTraining.tscn

# Terminal 2 — in this repo
uv run python -m rl_godot.constant_action_check
```

Success looks like: the Godot console completes a handshake (not the "using human controls
instead" fallback), and the Python side prints `--n-steps` (default 200) lines of finite,
changing observation values with no exceptions.

You may see a one-time console warning about a `major`/`minor` version mismatch (the vendored
addon pins wire-protocol `0.7`, the `godot-rl` pip package here is `0.8.2`) — this is expected and
harmless; `Sync`'s handshake only warns on mismatch, it doesn't fail the connection.

## Dependencies

`godot-rl==0.8.2` lives in its own `rl_godot` dependency group in `pyproject.toml` (pulls in
`gymnasium`/`stable-baselines3`, unrelated to the `time_series` project's torch/MLflow stack):

```bash
uv sync --group rl_godot
```

## Naming note

The directory is `rl_godot/`, not `godot_rl/` — this repo's `pyproject.toml` sets
`pythonpath = ["."]`, so a top-level `godot_rl/` directory would shadow the installed `godot_rl`
package on `sys.path`.

## Not yet designed

RunPod deployment (running `maze_bots` headless on a RunPod pod alongside the training process)
is a deferred follow-on, once this interface check is proven locally. See
`docs/superpowers/specs/2026-07-31-rl-godot-interface-check-design.md` for what's explicitly out
of scope so far.
