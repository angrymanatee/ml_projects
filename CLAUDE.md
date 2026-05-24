# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

A personal ML learning and practice repository. Projects here explore ML techniques hands-on rather than production use.

Single top-level directory intentionally keeps one shared PyTorch installation across all projects.

## Environment

Managed with `uv`. Python 3.14, PyTorch 2.12 with MPS (Metal) backend on Apple Silicon.

```bash
uv run python   # run a script
uv add <pkg>    # add a dependency
uv sync         # sync the venv after pulling
```

MPS is the GPU backend — use `torch.device("mps")` for GPU acceleration.

## Stack

- Python 3.14, managed via `.python-version`
- PyTorch 2.12 + torchvision + torchaudio (MPS-enabled, no separate metal variant needed)
- Dependencies in `pyproject.toml`, locked in `uv.lock`

## Before Committing

Run both of these and fix all failures before committing:

```bash
uv run pytest          # unit tests with coverage
uv run pre-commit run --all-files  # lint, format, type check
```

Pre-commit hooks (configured in `.pre-commit-config.yaml`) run automatically on `git commit`. They enforce:

- **black** — code formatting (`line-length = 88`); auto-formats `.py` and `.ipynb` files
- **ruff** — linting + import sorting; auto-fixes safe issues, fails on remaining errors
- **nbstripout** — strips notebook output before committing
- **pyright** — static type checking (`typeCheckingMode = "standard"`); all `.py` files must pass

If a commit is rejected, the hooks will have auto-fixed what they can (black, ruff). Stage those changes and retry.

## Testing

Tests live in `tests/`, mirroring the source layout (e.g. `common/paths.py` → `tests/common/test_paths.py`). No `__init__.py` files needed — `pythonpath = ["."]` in `pyproject.toml` handles imports.

```bash
uv run pytest                        # all tests
uv run pytest tests/common/ -v       # specific directory
uv run pytest -k test_name           # specific test
```

Coverage is reported automatically. Hypothesis is available for property-based tests.
