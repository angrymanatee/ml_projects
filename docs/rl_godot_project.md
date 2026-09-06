# RL-Godot — Project Reference

Reinforcement-learning interface between this repo and `maze_bots` — a separate Godot 4.7
(Mono/C#) project at `/Users/sauron/GodotProjects/maze_bots`, purpose-built as an RL training
environment. This project is Python-only; the game itself, its C#/GDScript RL bridge, and the
Godot-side `godot_rl_agents` addon all live in that other repo.

**Status: interface verified end to end, including on RunPod; PPO trains but doesn't learn
anything useful yet.** Three phases so far:
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
  keyed by lowercased `ObservationKind` — `"goalbearing"` (3-float body-frame `[cos, sin,
  log(1+range)]` to the goal) and `"rays"` (a 16-ray body-frame ray fan, 3 floats/ray — one
  per class layer: world/enemy/trap — so 48 floats total, each in `[0, 1]`; enemy/trap
  channels currently read zero, reserved for when those layers are populated). Actions are
  **body-frame**: a 2-float continuous `"movement"` key (`[forward, strafe]`) plus a 1-float
  continuous `"rotation"` key. Reward and an episode-`done` signal both flow through the same
  bridge (`done` fires when `GoalReachGame` ends a round — reaching the goal, dying, or timing
  out).
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

Steps a Godot instance with a constant action (default `movement=[1.0, 0.0]` — body-frame
forward — plus `rotation=0.0`, which drives the player straight toward the goal on
`GoStraight.tscn`; override with `--action-x`/`--action-y`/`--action-rotation`), printing
obs/reward/done each step. Manual smoke-test script, not covered by `pytest` — needs a live
Godot process on the other end.

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
`--build` (default) re-exports maze_bots from `--maze-bots-path` before launching, so
there's no need to run `rl_godot.export_env` separately first — it's the same
`build_env` helper, just invoked automatically. Pass `--no-build` to skip straight to
launching the binary already at `--env-path` as-is — e.g. when running externally
against a binary that was already built and pushed separately (there's no Godot editor
on a RunPod pod to export with; `remote/cli.py`'s pod-side invocation always passes
`--no-build`). `simple_ppo_train` takes the same `--build/--no-build`,
`--maze-bots-path`, and `--export-type` options.

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

### Selecting an RL config

`maze_bots/configs/rl_config.cfg` is the default; other configs (e.g. `no_rays.cfg`) can be
selected with `--rl-config <name>` (no `.cfg` extension) on `constant_action_check` /
`simple_ppo_train`, which requires `--env-path` (there's no running process to pass it to
otherwise):

```bash
uv run python -m rl_godot.constant_action_check \
  --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots \
  --rl-config no_rays
```

This works around `godot_rl.core.godot_env.GodotEnv`'s own kwargs pass-through, which appends
`--key=value` engine flags with no `--` separator — maze_bots' `RLConfig.cs` reads its override
via `OS.GetCmdlineUserArgs()`, which only sees args placed after that separator. `rl_godot/
env_launch.py` patches `StableBaselinesGodotEnv` to build `... -- --rl-config=res://configs/
<name>.cfg` instead. Without `--env-path` (two-terminal mode), pass it directly to `godot-mono`
yourself: `-- --rl-config=res://configs/no_rays.cfg` after your own flags.

Also note: `.cfg` files under `maze_bots/configs/` must be listed in `export_presets.cfg`'s
`include_filter` (e.g. `configs/*.cfg`) to actually end up in an export — Godot's
`export_filter="all_resources"` only packs files tracked by `ResourceLoader`, and a `ConfigFile`
loaded manually (as `RLConfig.cs` does) isn't one. Without that filter, `--rl-config` silently
falls back to scene defaults (`RLConfig: could not load ... (FileNotFound)` in the Godot log) —
this bit us once already.

### Selecting a map

`maps/GoStraight.tscn` is the default; other scenes under `maze_bots/maps/` are selected with
`--map <name>` (no `.tscn` extension, or a full `res://` path — `MapSelection.cs` accepts
either) on `constant_action_check` / `simple_ppo_train`. Like `--rl-config` it requires
`--env-path` and rides the same post-`--` user-args path built by `rl_godot/env_launch.py`;
both flags share a single `--` separator when passed together:

```bash
uv run python -m rl_godot.constant_action_check \
  --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots \
  --map GoStraightTrap --rl-config no_rays
```

`remote/cli.py`'s `godot check` / `godot run` take `--map` too and forward it to the pod-side
`constant_action_check`. In two-terminal mode, name the scene on the `godot-mono` command line
as usual (`godot-mono --headless --path . maps/GoStraightTrap.tscn`) or pass
`-- --map=GoStraightTrap` after your own flags.

## `rl_godot/simple_ppo_train.py`

Trains SB3 PPO (`MultiInputPolicy` — godot_rl always exposes a Dict obs space) against a
maze_bots instance, then rolls the learned policy out for `--eval-steps` steps. Takes the same
`--env-path` / `--build` / `--rl-config` / `--map` / `--parallel` options as
`constant_action_check`, plus `--total-timesteps`, `--n-steps`, `--learning-rate`, and
`--device` (pass `--device mps` explicitly on Apple Silicon — SB3's `auto` only looks for CUDA).

### Model artifacts in MLflow

Every run logs the trained policy to its MLflow run as SB3's native `.zip`:

- `model/ppo_model.zip` — final model, logged right after `learn()` (before the eval rollout, so
  a crash there doesn't cost the weights).
- `checkpoints/step_<timesteps>/ppo_model.zip` — intermediate snapshots every
  `--checkpoint-freq` env timesteps (default 5000, summed across parallel envs; `0` disables).

SB3 has no MLflow model flavor, so these are plain artifacts, not registered models — reload
with `PPO.load(<downloaded path>)`. The env isn't in the zip, so a reloaded model needs
`set_env()` before further training.

## `rl_godot/play_model.py`

Loads a trained policy and runs it against a **visible** Godot window, so you can watch what it
actually does rather than reading reward numbers. `simple_ppo_train` only evaluates in-process at
the end of a training run, always headless — this is the separate "watch it play" tool.

```bash
uv run python -m rl_godot.play_model \
  --env-path /Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots \
  --run-name unique-ram-468 \
  --map GoStraightTrap --rl-config watch
```

The model comes from exactly one of three sources:

- `--model-path` — a local `ppo_model.zip`.
- `--run-id` — an MLflow run id (the 32-char hex one), downloaded to a temp dir.
- `--run-name` — an MLflow run name, resolved to a run id within `--experiment` (default
  `GodotMazeBots_PPO`). Run names are **not** unique in MLflow — training twice under one
  `--run-name` is normal — so several matches resolve to the most recent, and the chosen run id is
  printed rather than picked silently.

`--artifact-path` selects an artifact other than the default `model/ppo_model.zip` within the run,
e.g. `checkpoints/step_000005000/ppo_model.zip` for an intermediate snapshot (the step number is
9-digit zero-padded — `MLflowCheckpointCallback` formats it as `step_{timesteps:09d}`). It requires
`--run-id` or `--run-name`; there is nothing to look it up in otherwise.

Takes the same `--env-path` / `--build` /
`--rl-config` / `--map` / `--device` options as the other launchers, plus `--n-steps`,
`--stochastic` (sample instead of taking the deterministic action), and `--speedup` (default `1`,
i.e. real time — `simple_ppo_train` leaves it unset so training runs as fast as the engine will
go). `--parallel` has no equivalent: it forces one env, since watching N windows defeats the point.
No MLflow run is started; `--run-id` only *reads* a model.

**The `--rl-config` you play under must have the same sensor settings as the config the policy was
trained under.** Those settings determine the observation vector's width and meaning, so a mismatch
either blows up on the first `predict()` or, worse, silently feeds the policy a differently-shaped
world. `configs/watch.cfg` in maze_bots exists for exactly this: a byte-for-byte copy of
`rl_config.cfg` with `debug_draw = true`, which turns on `RayFanSensor`'s debug overlay so you can
see the ray fan (white = world, orange = enemy, red = trap) while the policy drives. Play a
`no_rays`-trained policy under `no_rays`, not under `watch`.

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

Reward shaping and hyperparameter tuning that make PPO actually learn the task — `simple_ppo_train`
runs end to end and logs models, but the resulting policy is no good yet.
