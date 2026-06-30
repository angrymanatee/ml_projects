# ML Projects

Personal ML learning and practice repo. Each top-level directory is a self-contained project sharing one PyTorch installation.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running Code

**Notebooks** — open in VS Code or JupyterLab:

```bash
uv run jupyter lab
```

Notebook kernels automatically `chdir` to the repo root on startup via `~/.ipython/profile_default/startup/00-repo-root.py` (a local machine config, not in this repo). It finds the git root and changes to it, so relative paths like `data/` resolve from the top level regardless of which subdirectory the notebook lives in.

If this file is missing on a new machine, create it to restore the behavior:

```python
# ~/.ipython/profile_default/startup/00-repo-root.py
import os, subprocess
try:
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
    ).decode().strip()
    if os.getcwd() != root:
        os.chdir(root)
except Exception:
    pass
```

**Scripts:**

```bash
uv run python some_script.py
```

## Testing

```bash
uv run pytest
```

Tests live in `tests/`, mirroring source layout. Coverage is printed automatically.

## Linting & Formatting

Auto-format and fix before committing:

```bash
uv run black .                  # reformat all .py and .ipynb files in place
uv run ruff check --fix .       # fix all auto-fixable lint issues
```

Then verify everything passes (including pyright and nbstripout):

```bash
uv run pre-commit run --all-files
```

| Hook | What it checks |
|------|---------------|
| black | Code formatting (`.py` and `.ipynb`) |
| ruff | Linting and import sorting |
| pyright | Static type checking |
| nbstripout | Strips notebook output before commit |

## Projects

| Directory | Topic | Reference |
|-----------|-------|-----------|
| `time_series/` | Kaggle Store Sales forecasting — Transformer-based RMSLE minimization | [`docs/store_sales_project.md`](docs/store_sales_project.md) |
| `intro/` | Introductory ML exercises | — |

## Stack

- Python 3.14, PyTorch 2.12, MPS backend (Apple Silicon)
- MLflow for experiment tracking (`mlflow ui` to browse runs)
- Optuna for hyperparameter search
- `uv` for dependency management — add packages with `uv add <pkg>`
