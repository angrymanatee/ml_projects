# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

A personal ML learning and practice repository. Projects here explore ML techniques hands-on rather than production use.

Single top-level directory intentionally keeps one shared PyTorch installation across all projects.

### Active Project: `time_series/` — Kaggle Store Sales Forecasting

**Competition:** Store Sales - Time Series Forecasting (Kaggle). Predict 16 days of sales across 54 Ecuadorian grocery stores × 33 product families. Metric: RMSLE (implemented as `MSLELoss` in `store_sales.py`).

**Data:** `data/store-sales-time-series-forecasting/` — `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `sample_submission.csv`. Not committed; must be downloaded from Kaggle.

**Core dataset class:** `time_series/store_sales.py::StoreData` — PyTorch `Dataset` that loads all CSVs, builds a `[T, 54, 33]` sales tensor, and yields sliding-window `(input, target)` pairs. Optional input features (all off by default except date/payday/earthquake):
- `date_features=True` — days-since-start, sin/cos day-of-week, sin/cos day-of-year (5 cols)
- `payday_features=True` — is_15th, is_month_end (2 cols)
- `earthquake_encoding` — proximity to 2016 Ecuador earthquake (1 col; DECAY or LINEAR)
- `include_oil=True` — WTI oil price, ffill/bfill for gaps (1 col)
- `include_onpromotion=True` — per-store-family promotion flag (33 cols)
- `store_feature_cols` — subset of `("city", "state", "type", "cluster")`; one-hot encoded (variable cols)
- `holiday_features` — subset of `HOLIDAY_FEATURE_COLS`; binary indicators, national ones broadcast, regional/local matched by state/city (variable cols)

**Input tensor shape:** `[window_lags, 54, n_families + n_date_features + n_oil + n_promo + n_store_features + n_holiday_features]`. Target always `[output_lags, 54, 33]` (sales only). Default: 60 days in, 16 days out.

**Models (in order of sophistication):**

| File | Class | MLflow experiment |
|------|-------|-------------------|
| `main_store_sales_baseline.py` | `HoldLastValue` — repeats last observed value | `StoreSales_Baseline` |
| `main_store_sales_transformer.py` | `StoreSalesTransformer` — encoder-decoder Transformer | `StoreSales_TransformerBasic` |
| `main_store_sales_encoder_only.py` | `StoreSalesEncoderOnly` — encoder-only, one-shot projection | `StoreSales_TransformerEncoderOnly` |

`StoreSalesEncoderOnly` is the current focus. Uses `nn.LazyLinear` so input feature count is inferred on first forward pass — adding new feature groups to `StoreData` doesn't require updating model constructors.

**Hyperparameter tuning:** `tune_store_sales_encoder_only.py` — Optuna search over lr, d_model, nhead, num_layers, dim_feedforward. Each trial is a nested MLflow run under a parent study run.

**Experiment tracking:** MLflow local server. `common/model_registry.py` holds `TRACKING_URI`. Each run logs git branch + SHA, device, architecture tags, and val loss.

**Run commands:**
```bash
uv run python -m time_series.main_store_sales_baseline
uv run python -m time_series.main_store_sales_encoder_only --epochs 200 --lr 1.75e-3 --d-model 64 --num-layers 2
uv run python -m time_series.main_store_sales_encoder_only --oil --onpromotion --store-features city state type --holiday-features national_holiday
uv run python -m time_series.tune_store_sales_encoder_only --n-trials 40 --epochs-per-trial 30
```

**Common module:** `common/` — shared utilities: `paths.py` (data dir resolution), `git.py` (branch/SHA), `model_registry.py` (MLflow URI), `modules.py` (PositionalEncoding, GetLastIndex).

## Environment

Managed with `uv`. Python 3.14, PyTorch 2.12 with MPS (Metal) backend on Apple Silicon.

```bash
uv run python   # run a script
uv add <pkg>    # add a dependency
uv sync         # sync the venv after pulling
```

MPS is the GPU backend — use `torch.device("mps")` for GPU acceleration.

## Notebook Working Directory

Notebook kernels auto-`chdir` to the repo root via `~/.ipython/profile_default/startup/00-repo-root.py` (machine-local, not committed). This means relative paths in notebooks resolve from the repo root. If a notebook seems to have the wrong CWD, that file is likely missing — recreate it:

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

## Stack

- Python 3.14, managed via `.python-version`
- PyTorch 2.12 + torchvision + torchaudio (MPS-enabled, no separate metal variant needed)
- Dependencies in `pyproject.toml`, locked in `uv.lock`

## Python Conventions

- Prefer `@dataclass` over `NamedTuple` for structured data.
- Add docstrings to all public classes and functions. Explain the non-obvious: contracts, units, shape conventions, caveats. Don't restate the signature.

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
