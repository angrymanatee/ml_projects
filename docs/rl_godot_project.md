# RL-Godot — Project Reference

Reinforcement-learning interface between this repo and `maze_bots` — a separate Godot 4.7
(Mono/C#) project at `/Users/sauron/GodotProjects/maze_bots`, purpose-built as an RL training
environment. This project is Python-only; the game itself, its C#/GDScript RL bridge, and the
Godot-side `godot_rl_agents` addon all live in that other repo.

**Status: interface verified end to end, including on RunPod.** No real training (PPO, or any
actual policy) exists yet — the constant-action check is the whole surface. Three phases so far:
1. Local two-terminal wire-protocol check (Sync node + Python client).
2. Exported Godot binary, Python launches it directly (`--env-path`, one terminal).
3. Same check running entirely on a RunPod CPU pod (`remote godot run`).

**If you notice a discrepancy between this file and the code, update this file.**

---

## How the two repos fit together

- `maze_bots` has a `Sync` node (from the vendored `godot_rl_agents` addon) that opens a TCP
  server (default port `11008`) when a scene has `EnableRlSync = true` on its `GameRunnerBase`
  (see that repo's `docs/superpowers/specs/2026-07-31-godot-rl-sync-node-design.md`).
  `maps/GoStraight.tscn` has this on by default — it's also the normal manual-play map now, with
  its controller swapped to `RLController`.
- That `Sync` node drives an `RLController`/`rl_agent.gd` bridge: observations are a **dict**
  keyed by observation kind (`"goal"`, `"player"`), each a 6-float `[X, Y, Z, orientationX,
  orientationY, orientationZ]` vector; actions are a single 2-float continuous `"direction"` key;
  reward and an episode-`done` signal both flow through the same bridge (`done` fires when
  `GoalReachGame` ends a round — reaching the goal, dying, or timing out).
- `maze_bots/scripts/export.sh` produces headless-runnable macOS/Linux exports at
  `maze_bots/build/{macos,linux}/` — gitignored build artifacts, not committed. Either the Python
  side connects to a manually-launched `godot-mono` process, or (preferred) launches an exported
  binary itself via `--env-path`.
- This repo's `rl_godot/` is the Python side of that TCP connection, plus (via `remote/`) the
  RunPod deployment that runs both halves on a pod.

## `rl_godot/export_env.py`

Thin wrapper around `maze_bots/scripts/export.sh` so you don't have to switch checkouts to
produce a fresh binary:

```bash
uv run python -m rl_godot.export_env
```

Defaults `--maze-bots-path` to `/Users/sauron/GodotProjects/maze_bots`; override it if your
checkout lives elsewhere. Prints the exported macOS/Linux binary paths and the exact
`constant_action_check --env-path ...` command to run next. Same manual-tool status as
`constant_action_check.py` below — needs a real `godot-mono` on `PATH`, not covered by `pytest`.

## `rl_godot/constant_action_check.py`

Steps a Godot instance with a constant action (default `direction=[0.0, -1.0]`, which drives the
player straight toward the goal on `GoStraight.tscn`; override with `--action-x`/`--action-y`),
printing obs/reward/done each step. Manual smoke-test script, not covered by `pytest` — needs a
live Godot process on the other end.

`--gui` opens a live tkinter debug window alongside the terminal printout, showing the current
action and observation and letting you edit the action while the loop runs — Entry fields, arrow
keys (nudge by `--action-step`), or spacebar (zero it). In `--gui` mode the loop runs until the
window is closed rather than for `--n-steps`.

### Running it locally

**With `--env-path`** (preferred — one terminal, Python launches Godot itself):
```bash
uv run python -m rl_godot.constant_action_check \
  --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots
```
Requires the export to exist first: `uv run python -m rl_godot.export_env` (see above).

**Without `--env-path`** (two terminals, connects to an already-running instance):
```bash
# Terminal 1 — in the maze_bots repo
godot-mono --headless --path . maps/GoStraight.tscn

# Terminal 2 — in this repo
uv run python -m rl_godot.constant_action_check
```

Success looks like: the Godot console completes a handshake (not the "using human controls
instead" fallback), and the Python side prints obs/reward/done lines with finite, changing values
— `reward=1`/`done=True` firing when the player reaches the goal, followed by an automatic round
reset.

You may see a one-time console warning about a `major`/`minor` version mismatch (the vendored
addon pins wire-protocol `0.7`, the `godot-rl` pip package here is `0.8.2`) — expected and
harmless; the handshake only warns on mismatch, it doesn't fail the connection.

## Running it on RunPod

`remote/` (this repo's general RunPod CLI, otherwise used for `time_series` GPU training) has a
`godot` subcommand group for a CPU-only pod flavor, verified working end to end:

```bash
# Full pipeline: create CPU pod -> push code+binary -> install deps -> run the check -> terminate
uv run python -m remote godot run

# Decomposed primitives, for debugging against an already-created pod:
uv run python -m remote godot push <pod-id> [--maze-bots-path PATH]
uv run python -m remote godot setup <pod-id>
uv run python -m remote godot check <pod-id> [--n-steps N]
```

Since `Sync.gd` hardcodes `127.0.0.1` (loopback-only, not exposed over the network), the Python
process and the Godot binary must run on the *same* machine — there's no "train locally against a
remote game server" mode. `godot run` pushes both `rl_godot/` (Python) and the exported Linux
binary (`maze_bots/build/linux/`, built via that repo's `scripts/export.sh`) to the same pod and
runs the check there.

**Setup prerequisites beyond `time_series`'s existing RunPod setup** (`docs/store_sales_project.md`),
discovered getting this working — worth knowing since they affect any `remote` command, not just
`godot`:

- `runpod_config.yaml` **must** set `ssh_key_path` to an **unencrypted** private key. RunPod pods
  are reached by raw IP, which `~/.ssh/config` `Host` patterns (e.g. `Host *.runpod.io`) don't
  match, and the default SSH agent has no identities loaded — so without `ssh_key_path`, `-i` is
  never passed at all. An encrypted key just fails non-interactively with no clear error (this
  pipeline never prompts). See `configs/runpod_config.yaml.example`.
- Real CPU pods use a completely different RunPod API path than "a GPU pod with `gpu_count=0`" —
  the latter doesn't work at all (confirmed empirically: rejected as "no instances available"
  across every GPU type and cloud tier). The pip `runpod` SDK only creates a genuine CPU pod when
  `gpu_type_id=None`, optionally with `instance_id` for a specific flavor (e.g. `"cpu3c-2-4"` —
  see `runpod.list_cpu_types()`/`mcp__runpod__list-cpu-types`). CPU pods also cap
  `container_disk_in_gb` at 20 (GPU pods allow 50) — a hard platform limit, not configurable.
- `godot_docker_image` is the same PyTorch image as the main GPU flow
  (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`), not a lighter CPU-specific image —
  `runpod/base:0.6.2-cpu` was tried first (matching the sweep-mode MLflow pod's image) but its
  container never actually writes the injected SSH key into `authorized_keys` when deployed as a
  bare pod outside RunPod's "Projects" feature; only images RunPod calls "official templates"
  (PyTorch, Stable Diffusion, etc.) have that wired up. No CUDA is actually used — `godot-rl`
  pulls a CPU-only `stable-baselines3`/`torch` via pip regardless of the image's CUDA libraries.

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

Real training (an actual RL algorithm/policy, not a constant action) — everything so far is
interface plumbing.
