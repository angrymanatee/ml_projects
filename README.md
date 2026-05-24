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

## Stack

- Python 3.14, PyTorch 2.12, MPS backend (Apple Silicon)
- `uv` for dependency management — add packages with `uv add <pkg>`
