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

Use the `precommit-check` skill.

## Testing

Tests live in `tests/`, mirroring the source layout (e.g. `common/paths.py` → `tests/common/test_paths.py`). No `__init__.py` files needed — `pythonpath = ["."]` in `pyproject.toml` handles imports.

```bash
uv run pytest                        # all tests
uv run pytest tests/common/ -v       # specific directory
uv run pytest -k test_name           # specific test
```

Coverage is reported automatically. Hypothesis is available for property-based tests.
