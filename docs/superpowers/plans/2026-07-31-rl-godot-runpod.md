# RL-Godot on RunPod — Implementation Plan (Phase 3)

**Goal:** `remote godot run` — an ephemeral RunPod CPU pod running both the Python trainer and the
exported Linux Godot binary, driven end-to-end from a single local command.

Full design rationale: `docs/superpowers/specs/2026-07-31-rl-godot-runpod-design.md`.

## Global Constraints

- No automated tests — matches `remote/`'s existing convention (none of `pod.py`/`sync.py`/
  `environment.py` have unit tests; verified by running against a real pod).
- Requires `maze_bots/build/linux/MazeBots.x86_64` to already exist (Phase 2, `scripts/export.sh`
  in the maze_bots repo) — if missing, fail with a clear message pointing at that script, same
  pattern as `push_data`'s missing-dataset `FileNotFoundError`.
- Two unknowns get verified empirically, not assumed (see design spec): Python/pip availability
  on `runpod/base:0.6.2-cpu`, and whether the exported binary needs any shared libs installed on
  a minimal image. Treat unexpected failures here as "read the error, add one `apt-get install`
  line or swap the base image" — not a redesign.

---

### Task 1: Config additions

**Files:**
- Modify: `remote/config.py`

- [ ] Add three fields to `RunPodConfig`:
  ```python
  godot_docker_image: str = "runpod/base:0.6.2-cpu"
  remote_godot_build_dir: str = "/workspace/maze_bots_build"
  maze_bots_repo_path: str = "/Users/sauron/GodotProjects/maze_bots"
  ```
- [ ] Commit: `git add remote/config.py && git commit -m "Add RunPod config for godot CPU pod flavor"`

---

### Task 2: `push_godot_build` + extend `push_code`

**Files:**
- Modify: `remote/sync.py`

- [ ] Add `"rl_godot"` to `_SOURCE_DIRS`:
  ```python
  _SOURCE_DIRS = ["time_series", "common", "rl_godot"]
  ```
- [ ] Add `push_godot_build`:
  ```python
  def push_godot_build(target: SSHTarget, config: RunPodConfig, maze_bots_path: Path) -> None:
      """Rsync the exported Linux Godot binary to the remote pod.

      Args:
          target: SSH connection info.
          config: RunPod configuration.
          maze_bots_path: Local path to the maze_bots checkout.

      Raises:
          FileNotFoundError: If <maze_bots_path>/build/linux does not exist locally
              (run maze_bots's scripts/export.sh first).
      """
      local_build_dir = maze_bots_path / "build" / "linux"
      if not local_build_dir.exists():
          raise FileNotFoundError(
              f"Linux export not found: {local_build_dir}\n"
              "Run scripts/export.sh in the maze_bots repo first."
          )
      run_remote(target, f"mkdir -p {config.remote_godot_build_dir}")
      _run_rsync(
          f"{local_build_dir}/",
          f"{target.user}@{target.host}:{config.remote_godot_build_dir}/",
          target,
      )
      run_remote(target, f"chmod +x {config.remote_godot_build_dir}/MazeBots.x86_64")
  ```
- [ ] Manual check: `uv run python -c "from remote.sync import push_godot_build"` — confirms it
      imports cleanly (no live pod to test against yet at this step).
- [ ] Commit: `git add remote/sync.py && git commit -m "Add push_godot_build, sync rl_godot/ in push_code"`

---

### Task 3: `setup_godot_environment`

**Files:**
- Modify: `remote/environment.py`

- [ ] Add:
  ```python
  _GODOT_REMOTE_DEPS = ["godot-rl==0.8.2"]


  def setup_godot_environment(target: SSHTarget, config: RunPodConfig) -> None:
      """Bootstrap the remote pod for the Godot RL interface check.

      Installs godot-rl (pulls a CPU-only stable-baselines3/torch automatically —
      no CUDA involved, unlike setup_environment()).
      """
      deps = " ".join(f'"{d}"' for d in _GODOT_REMOTE_DEPS)
      run_remote(target, f"python -m pip install --quiet --ignore-installed {deps}")
      run_remote(target, f"mkdir -p {config.remote_project_dir}")
      run_remote(target, "python -c \"import godot_rl; print('godot_rl OK')\"")
  ```
- [ ] Commit: `git add remote/environment.py && git commit -m "Add setup_godot_environment (CPU, no CUDA check)"`

---

### Task 4: `godot` CLI subcommand group

**Files:**
- Modify: `remote/cli.py`

- [ ] Add imports: `push_godot_build` (from `remote.sync`), `setup_godot_environment` (from
      `remote.environment`).
- [ ] Add the subcommand group, mirroring `sweep_app`'s registration:
  ```python
  godot_app = typer.Typer(help="Godot RL interface check on a CPU pod", no_args_is_help=True)
  app.add_typer(godot_app, name="godot")
  ```
- [ ] `godot push`:
  ```python
  @godot_app.command("push")
  def godot_push(
      pod_id: str,
      maze_bots_path: Path = typer.Option(None, "--maze-bots-path"),
      config_path: Path = _CONFIG_OPTION,
  ) -> None:
      """Push rl_godot/ source and the exported Linux binary to the pod."""
      config = load_config(config_path)
      target = get_ssh_target(config, pod_id)
      push_code(target, config)
      push_godot_build(target, config, maze_bots_path or Path(config.maze_bots_repo_path))
      typer.echo("Godot code and binary pushed.")
  ```
- [ ] `godot setup`:
  ```python
  @godot_app.command("setup")
  def godot_setup(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
      """Install godot-rl on the pod."""
      config = load_config(config_path)
      target = get_ssh_target(config, pod_id)
      setup_godot_environment(target, config)
      typer.echo("Godot environment ready.")
  ```
- [ ] `godot check`:
  ```python
  @godot_app.command("check")
  def godot_check(
      pod_id: str,
      n_steps: int = typer.Option(20, "--n-steps"),
      config_path: Path = _CONFIG_OPTION,
  ) -> None:
      """Run the constant-action interface check on the pod."""
      config = load_config(config_path)
      target = get_ssh_target(config, pod_id)
      full_cmd = (
          f"cd {config.remote_project_dir} && "
          f"python -m rl_godot.constant_action_check "
          f"--env-path {config.remote_godot_build_dir}/MazeBots --n-steps {n_steps}"
      )
      run_remote(target, full_cmd)
      typer.echo("Check complete.")
  ```
- [ ] `godot run` (the actual deliverable — full ephemeral pipeline, mirrors `run_cmd`'s
      structure/error handling exactly):
  ```python
  @godot_app.command("run")
  def godot_run(
      maze_bots_path: Path = typer.Option(None, "--maze-bots-path"),
      n_steps: int = typer.Option(20, "--n-steps"),
      on_complete: str | None = typer.Option(None, "--on-complete"),
      config_path: Path = _CONFIG_OPTION,
  ) -> None:
      """Full pipeline: provision CPU pod -> push -> setup -> check -> terminate."""
      config = load_config(config_path)
      if on_complete:
          config.on_complete = on_complete
      config.docker_image = config.godot_docker_image
      resolved_maze_bots_path = maze_bots_path or Path(config.maze_bots_repo_path)

      pod_id: str | None = None
      try:
          typer.echo("Creating pod...")
          pod_id = create_pod(config, name_suffix="godot")
          typer.echo(f"Pod created: {pod_id}")

          typer.echo("Waiting for pod to start...")
          target = wait_for_running(config, pod_id)
          typer.echo("Waiting for SSH...")
          wait_for_ssh(target)

          typer.echo("Pushing code and binary...")
          push_code(target, config)
          push_godot_build(target, config, resolved_maze_bots_path)

          typer.echo("Setting up environment...")
          setup_godot_environment(target, config)

          typer.echo("Running interface check...")
          full_cmd = (
              f"cd {config.remote_project_dir} && "
              f"python -m rl_godot.constant_action_check "
              f"--env-path {config.remote_godot_build_dir}/MazeBots --n-steps {n_steps}"
          )
          run_remote(target, full_cmd)

          _apply_on_complete(config, pod_id)
          pod_id = None
      except Exception as exc:
          typer.echo(f"\nError: {exc}", err=True)
          if pod_id:
              typer.echo(
                  f"Pod {pod_id} left running — SSH in to debug or terminate manually:",
                  err=True,
              )
              typer.echo(f"  python -m remote pod terminate {pod_id}", err=True)
          raise typer.Exit(1) from None
  ```
  Note: `create_pod()` currently always passes `gpu_type_id`/`gpu_count` from config
  (`remote/pod.py`'s `create_pod`, not `create_mlflow_pod`) — check whether it needs a
  `gpu_count=0` path for a true CPU pod (per `create_mlflow_pod`'s existing comment: "RunPod's
  SDK requires a gpu_type_id even for CPU workloads"). If `create_pod()` can't produce a CPU-only
  pod as-is, either add a `cpu_only: bool` parameter to it (small, targeted change — do not
  redesign pod creation) or reuse `create_mlflow_pod`'s pattern directly here. Resolve this
  empirically against the real RunPod API in Step "Run it for real" below, not by guessing.

- [ ] Commit: `git add remote/cli.py && git commit -m "Add remote godot CLI subcommand group"`

---

### Task 5: Run it for real

- [ ] `uv run python -m remote godot run --n-steps 20`
- [ ] Watch for and resolve, in order, treating each as a "read the real error, fix, continue"
      step (do not pre-solve these blind):
  1. Pod creation — does `create_pod()` need adjusting for `gpu_count=0`? (see Task 4's note)
  2. Does `runpod/base:0.6.2-cpu` have `python`/`pip` on `PATH`, and does `python -m pip install
     godot-rl==0.8.2` succeed?
  3. Does the pushed Linux binary run under `--headless` without missing shared libraries?
  4. Does the check script complete `n_steps` and print sane obs/reward/done (same shape already
     verified locally: `{'goal': [...6...], 'player': [...6...]}`, reward/done firing on goal
     reach)?
- [ ] Confirm the pod actually terminates on success (`remote pod list` shows it gone, or
      `desiredStatus` reflects termination).
- [ ] No commit for this step — it's verification, not a code change (unless Task 5 surfaces a
      real fix, in which case that fix gets its own commit per the constraint above).

---

### Task 6: Docs

**Files:**
- Modify: `docs/rl_godot_project.md`

- [ ] Add a "Phase 3: RunPod" section documenting `remote godot run` (and the decomposed
      `push`/`setup`/`check` commands for debugging), replacing the current "Not yet designed"
      closing section.
- [ ] Commit: `git add docs/rl_godot_project.md && git commit -m "Document RunPod deployment for rl_godot"`
