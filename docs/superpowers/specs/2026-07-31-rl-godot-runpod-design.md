# RL-Godot on RunPod — Design (Phase 3)

**Date:** 2026-07-31
**Status:** Approved
**Scope:** Run the constant-action interface check on a live RunPod CPU pod — Python and the
exported Linux Godot binary both running on the pod, driven end-to-end from a single local
command.

---

## Overview

Phase 1 proved the wire protocol locally (two terminals, manually launched Godot). Phase 2
produced a self-contained exported binary and let Python launch it directly (`--env-path`, one
terminal). Phase 3 moves both halves onto a RunPod pod. Since `Sync.gd`'s `connect_to_server()`
hardcodes `127.0.0.1` (loopback only, not exposed over the network), Python and the Godot binary
must run on the *same* machine — there is no "train locally against a remote game server" mode.
So this is a same-machine deployment, just that machine is now a RunPod pod instead of this Mac.

This extends the existing `remote/` module (already handles time_series's GPU/PyTorch pods)
rather than building a parallel system — the pod lifecycle, SSH, and rsync machinery are already
proven; only the pod flavor (CPU, no CUDA) and what gets synced (a foreign-repo binary instead of
a `data/` dataset) are genuinely new.

Out of scope: real training (still just the constant-action check); a persistent/reusable pod
workflow (this is the same ephemeral create→run→terminate shape as `remote run`); anything in the
maze_bots repo (its Linux binary already exists from Phase 2 and needs no changes here).

---

## Config additions (`remote/config.py`)

```python
godot_docker_image: str = "runpod/base:0.6.2-cpu"  # same cheap image as mlflow_cpu_image
remote_godot_build_dir: str = "/workspace/maze_bots_build"
maze_bots_repo_path: str = "/Users/sauron/GodotProjects/maze_bots"  # overridable via CLI
```

`maze_bots_repo_path` gets a default matching the path already hardcoded throughout this
project's docs (single-user personal repo, consistent with existing convention) — but every CLI
command that needs it accepts a `--maze-bots-path` override rather than only reading the config
default, since it's inherently a local-machine path.

---

## New pieces

- **`remote/sync.py`**: `push_godot_build(target, config, maze_bots_path)` — rsyncs
  `<maze_bots_path>/build/linux/` to `config.remote_godot_build_dir` on the pod, then
  `chmod +x` the binary. Raises `FileNotFoundError` if the local build doesn't exist (same
  pattern as `push_data`'s missing-dataset check) — the error message should say to run
  maze_bots's `scripts/export.sh` first. `push_code()`'s `_SOURCE_DIRS` gets `"rl_godot"` added
  (alongside the existing `time_series`, `common`) so the Python side ships too.

- **`remote/environment.py`**: `setup_godot_environment(target, config)` — installs
  `godot-rl==0.8.2` via pip (pulls a CPU-only `stable-baselines3`/`torch` automatically — no
  `--index-url` needed, PyPI's default torch wheel is CPU-only). No CUDA verification (unlike
  `setup_environment()`, which is GPU-specific and stays as-is for time_series).

- **`remote/cli.py`**: new `godot` subcommand group, mirroring `sweep`'s structure:
  ```
  remote godot push <pod-id> [--maze-bots-path PATH]   # push_code (incl. rl_godot/) + push_godot_build
  remote godot setup <pod-id>                           # setup_godot_environment
  remote godot check <pod-id> [--n-steps N]              # run constant_action_check.py over SSH
  remote godot run [--maze-bots-path PATH] [--n-steps N] [--on-complete terminate|stop|keep]
                                                          # full pipeline: create -> push -> setup -> check -> terminate
  ```
  `run` is the actual deliverable; `push`/`setup`/`check` are decomposed primitives for
  fine-grained debugging, matching how `pod create`/`sync push-code`/`env setup`/`train` relate
  to `run` for the existing GPU flow.

`godot check`'s remote command:
```bash
cd {remote_project_dir} && python -m rl_godot.constant_action_check \
  --env-path {remote_godot_build_dir}/MazeBots --n-steps {n_steps}
```
(no platform suffix — `godot_rl` appends `.x86_64` itself based on the pod's `platform`, which
will correctly resolve to Linux).

---

## Real unknowns, verified empirically during implementation

Not assumed here, checked by actually running it (same approach that resolved Phase 2's export
issues — read the real error, fix, move on):

1. Whether `runpod/base:0.6.2-cpu` has Python 3 and pip preinstalled, and what version.
2. Whether the exported Linux binary needs any shared libraries installed on a minimal image
   even in `--headless` mode (Godot headless shouldn't need X11/GL, but may still dynamically
   link optional libs like audio that could be entirely absent from a minimal base image).

If either surfaces a real gap, the fix is a `apt-get install` line in `setup_godot_environment()`
or a base image swap — not a redesign.

---

## Testing

No automated tests — this is live infrastructure code, same convention as the rest of `remote/`
(none of `pod.py`/`sync.py`/`environment.py` have unit tests; they're verified by running them
against a real pod). Success is `remote godot run` completing end-to-end: pod created, code and
binary pushed, deps installed, the check script runs `n_steps` on the pod printing sane
obs/reward/done values (same output shape already verified locally in Phase 2), pod terminates
automatically.
