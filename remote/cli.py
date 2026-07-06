from __future__ import annotations

from pathlib import Path

import typer

from remote.config import RunPodConfig, load_config
from remote.environment import (
    mlflow_proxy_url,
    setup_environment,
    start_mlflow,
    stop_mlflow,
)
from remote.pod import (
    create_mlflow_pod,
    create_pod,
    get_pod,
    get_ssh_target,
    list_pods,
    stop_pod,
    terminate_pod,
    wait_for_running,
)
from remote.ssh import run_remote, wait_for_ssh
from remote.sync import pull_results, push_code, push_data

app = typer.Typer(
    name="remote", help="Remote GPU training on RunPod", no_args_is_help=True
)
pod_app = typer.Typer(help="Pod lifecycle", no_args_is_help=True)
sync_app = typer.Typer(help="Code and data sync", no_args_is_help=True)
env_app = typer.Typer(help="Remote environment", no_args_is_help=True)
sweep_app = typer.Typer(help="Parallel sweeps", no_args_is_help=True)
app.add_typer(pod_app, name="pod")
app.add_typer(sync_app, name="sync")
app.add_typer(env_app, name="env")
app.add_typer(sweep_app, name="sweep")

_CONFIG_OPTION = typer.Option(
    Path("runpod_config.yaml"), "--config", help="Path to config YAML"
)
_TRAIN_COMMAND_ARG = typer.Argument(
    ..., help="Command to run on the remote pod (leading 'uv run' is stripped)"
)
_SWEEP_COMMAND_ARG = typer.Argument(...)


def _normalize_train_command(command: list[str]) -> list[str]:
    """Strip leading 'uv run' from a command; warn if uv appears elsewhere.

    Remote pods use system Python, not uv. Leading 'uv run' is silently
    stripped. If 'uv' appears in any other position the caller likely made
    a mistake, so a warning is emitted but the command is left intact.
    """
    if len(command) >= 2 and command[0] == "uv" and command[1] == "run":
        return list(command[2:])
    if "uv" in command:
        typer.echo(
            "Warning: 'uv' found in train command. "
            "Remote pods use system Python — 'uv run' is unavailable.",
            err=True,
        )
    return list(command)


def _apply_on_complete(config: RunPodConfig, pod_id: str) -> None:
    """Terminate, stop, or keep the pod based on config.on_complete."""
    if config.on_complete == "terminate":
        terminate_pod(config, pod_id)
        typer.echo(f"Pod {pod_id} terminated.")
    elif config.on_complete == "stop":
        stop_pod(config, pod_id)
        typer.echo(f"Pod {pod_id} stopped.")
    else:
        typer.echo(f"Pod {pod_id} left running (on_complete=keep).")


# ── Pod commands ──────────────────────────────────────────────────────────────


@pod_app.command("create")
def pod_create(
    gpu_type: str | None = typer.Option(None, "--gpu-type"),
    gpu_count: int | None = typer.Option(None, "--gpu-count"),
    image: str | None = typer.Option(None, "--image"),
    name_suffix: str = typer.Option("", "--name-suffix"),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Create a GPU pod and print its pod ID."""
    config = load_config(config_path)
    if gpu_type:
        config.gpu_type = gpu_type
    if gpu_count is not None:
        config.gpu_count = gpu_count
    if image:
        config.docker_image = image
    pod_id = create_pod(config, name_suffix=name_suffix)
    typer.echo(pod_id)


@pod_app.command("list")
def pod_list(config_path: Path = _CONFIG_OPTION) -> None:
    """List all pods on the account."""
    config = load_config(config_path)
    for pod in list_pods(config):
        typer.echo(f"{pod.pod_id}\t{pod.name}\t{pod.status}")


@pod_app.command("status")
def pod_status(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Print the status of a pod."""
    config = load_config(config_path)
    pod = get_pod(config, pod_id)
    typer.echo(pod.get("desiredStatus", "UNKNOWN"))


@pod_app.command("stop")
def pod_stop(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Stop a pod (preserves disk)."""
    config = load_config(config_path)
    stop_pod(config, pod_id)
    typer.echo(f"Stopped {pod_id}")


@pod_app.command("terminate")
def pod_terminate(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Terminate a pod (destroys disk and resources)."""
    config = load_config(config_path)
    terminate_pod(config, pod_id)
    typer.echo(f"Terminated {pod_id}")


# ── Sync commands ─────────────────────────────────────────────────────────────


@sync_app.command("push-code")
def sync_push_code(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Rsync source code to the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    push_code(target, config)
    typer.echo("Code synced.")


@sync_app.command("push-data")
def sync_push_data(
    pod_id: str,
    dataset: str = typer.Option(
        ..., "--dataset", help="Dataset subdirectory name under data/"
    ),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Rsync a local data/<dataset>/ directory to the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    push_data(target, config, dataset)
    typer.echo(f"Dataset '{dataset}' synced.")


@sync_app.command("pull")
def sync_pull(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Rsync mlruns/ from the pod back to local."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    pull_results(target, config)
    typer.echo("Results pulled.")


# ── Env commands ──────────────────────────────────────────────────────────────


@env_app.command("setup")
def env_setup(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Install remote deps and verify CUDA on the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    setup_environment(target, config)
    typer.echo("Environment ready.")


@env_app.command("mlflow-start")
def env_mlflow_start(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Start the MLflow server on the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    start_mlflow(target, config)
    typer.echo(f"MLflow server started on pod {pod_id}.")


@env_app.command("mlflow-stop")
def env_mlflow_stop(pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Stop the MLflow server on the pod."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    stop_mlflow(target)
    typer.echo("MLflow server stopped.")


# ── Train command ─────────────────────────────────────────────────────────────


@app.command(
    "train", context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
def train_cmd(
    pod_id: str,
    train_command: list[str] = _TRAIN_COMMAND_ARG,
    mlflow_uri: str | None = typer.Option(
        None, "--mlflow-uri", help="Override MLFLOW_TRACKING_URI"
    ),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Run a training command on a running pod via SSH."""
    config = load_config(config_path)
    target = get_ssh_target(config, pod_id)
    normalized = _normalize_train_command(train_command)
    uri = mlflow_uri or f"http://localhost:{config.mlflow_port}"
    full_cmd = f"cd {config.remote_project_dir} && {' '.join(normalized)}"
    run_remote(target, full_cmd, env={"MLFLOW_TRACKING_URI": uri})
    typer.echo("Training complete.")


# ── Run command (full pipeline, single pod) ───────────────────────────────────


@app.command(
    "run", context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
def run_cmd(
    train_command: list[str] = _TRAIN_COMMAND_ARG,
    dataset: str | None = typer.Option(None, "--dataset"),
    on_complete: str | None = typer.Option(
        None, "--on-complete", help="terminate|stop|keep"
    ),
    gpu_type: str | None = typer.Option(None, "--gpu-type"),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Full pipeline: provision → push → setup → train → pull → terminate."""
    config = load_config(config_path)
    if on_complete:
        config.on_complete = on_complete
    if gpu_type:
        config.gpu_type = gpu_type

    normalized = _normalize_train_command(train_command)
    datasets = [dataset] if dataset else config.default_datasets

    pod_id: str | None = None
    try:
        typer.echo("Creating pod...")
        pod_id = create_pod(config)
        typer.echo(f"Pod created: {pod_id}")

        typer.echo("Waiting for pod to start...")
        target = wait_for_running(config, pod_id)
        typer.echo("Waiting for SSH...")
        wait_for_ssh(target)

        typer.echo("Pushing code...")
        push_code(target, config)

        for ds in datasets:
            typer.echo(f"Pushing dataset: {ds}")
            push_data(target, config, ds)

        typer.echo("Setting up environment...")
        setup_environment(target, config)

        typer.echo("Starting MLflow server...")
        start_mlflow(target, config)

        typer.echo("Running training...")
        full_cmd = f"cd {config.remote_project_dir} && {' '.join(normalized)}"
        run_remote(
            target,
            full_cmd,
            env={"MLFLOW_TRACKING_URI": f"http://localhost:{config.mlflow_port}"},
        )

        typer.echo("Pulling results...")
        pull_results(target, config)

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


# ── Sweep commands ────────────────────────────────────────────────────────────


@sweep_app.command("create-mlflow-pod")
def sweep_create_mlflow_pod(config_path: Path = _CONFIG_OPTION) -> None:
    """Create a persistent CPU pod to host MLflow for parallel sweeps."""
    config = load_config(config_path)
    pod_id = create_mlflow_pod(config)
    typer.echo(pod_id)
    typer.echo(
        f"MLflow URL (once running): {mlflow_proxy_url(pod_id, config)}", err=True
    )


@sweep_app.command(
    "run", context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
def sweep_run(
    mlflow_pod_id: str = typer.Option(..., "--mlflow-pod"),
    n_pods: int = typer.Option(1, "--n-pods"),
    dataset: str | None = typer.Option(None, "--dataset"),
    on_complete: str | None = typer.Option(None, "--on-complete"),
    config_path: Path = _CONFIG_OPTION,
    train_command: list[str] = _SWEEP_COMMAND_ARG,
) -> None:
    """Launch N parallel GPU pods all logging to a shared MLflow pod."""
    import concurrent.futures

    config = load_config(config_path)
    if on_complete:
        config.on_complete = on_complete
    normalized = _normalize_train_command(train_command)
    datasets = [dataset] if dataset else config.default_datasets
    mlflow_uri = mlflow_proxy_url(mlflow_pod_id, config)

    def _run_one(index: int) -> str:
        pod_id = create_pod(config, name_suffix=f"sweep-{index}")
        typer.echo(f"[sweep-{index}] Pod created: {pod_id}")
        try:
            target = wait_for_running(config, pod_id)
            wait_for_ssh(target)
            push_code(target, config)
            for ds in datasets:
                push_data(target, config, ds)
            setup_environment(target, config)
            full_cmd = f"cd {config.remote_project_dir} && {' '.join(normalized)}"
            run_remote(target, full_cmd, env={"MLFLOW_TRACKING_URI": mlflow_uri})
            typer.echo(f"[sweep-{index}] Training complete.")
            _apply_on_complete(config, pod_id)
            return pod_id
        except Exception as exc:
            typer.echo(f"[sweep-{index}] Error on pod {pod_id}: {exc}", err=True)
            typer.echo(f"[sweep-{index}] Pod {pod_id} left running.", err=True)
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_pods) as executor:
        futures = [executor.submit(_run_one, i) for i in range(n_pods)]
        concurrent.futures.wait(futures)

    failed = [f for f in futures if f.exception() is not None]
    if failed:
        typer.echo(f"{len(failed)}/{n_pods} pods failed. See errors above.", err=True)
        raise typer.Exit(1)


@sweep_app.command("pull")
def sweep_pull(mlflow_pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Pull mlruns/ from the MLflow pod back to local."""
    config = load_config(config_path)
    target = get_ssh_target(config, mlflow_pod_id)
    pull_results(target, config)
    typer.echo("Sweep results pulled.")


@sweep_app.command("teardown")
def sweep_teardown(mlflow_pod_id: str, config_path: Path = _CONFIG_OPTION) -> None:
    """Terminate the MLflow pod."""
    config = load_config(config_path)
    terminate_pod(config, mlflow_pod_id)
    typer.echo(f"MLflow pod {mlflow_pod_id} terminated.")
