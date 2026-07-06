# Store Sales — Project Reference

Kaggle Store Sales - Time Series Forecasting. Predict 16 days of sales across
54 Ecuadorian grocery stores × 33 product families. Metric: RMSLE (implemented
as `MSLELoss` in `store_sales.py`).

**If you notice a discrepancy between this file and the code, update this file.**

---

## Data

`data/store-sales-time-series-forecasting/` — not committed; download from Kaggle.

Files: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`,
`sample_submission.csv`.

---

## Dataset: `StoreData`

`time_series/store_sales.py::StoreData` — PyTorch `Dataset` that loads all CSVs,
builds a `[T, 54, 33]` sales tensor, and yields sliding-window `(input, target)` pairs.

**Default window:** 60 days in (`window_lags`), 16 days out (`output_lags`).

**Target shape:** always `[output_lags, 54, 33]` (sales only).

**Input shape:** `[window_lags, 54, C]` where `C` is the sum of all enabled feature channels:

| Feature group | Flag | Cols | Notes |
|---------------|------|------|-------|
| Sales | always on | 33 | one per family |
| Date features | `date_features=True` (default) | 5 | days-since-start, sin/cos day-of-week, sin/cos day-of-year |
| Payday features | `payday_features=True` (default) | 2 | is_15th, is_month_end |
| Earthquake | `earthquake_encoding=DECAY` (default) | 1 | exp(-days/tau) since 2016 Ecuador quake; also LINEAR or None |
| Oil price | `include_oil=False` | 1 | WTI price, ffill/bfill for weekend/holiday gaps |
| On-promotion | `include_onpromotion=False` | 33 | per-store-family binary flag |
| Store features | `store_feature_cols=[]` | varies | subset of `("city", "state", "type", "cluster")`; categoricals one-hot encoded |
| Holiday features | `holiday_features=[]` | varies | subset of `HOLIDAY_FEATURE_COLS`; national features broadcast, regional/local matched by state/city |

`HOLIDAY_FEATURE_COLS`: `national_holiday`, `event`, `bridge`, `work_day`, `additional`,
`regional_holiday`, `local_holiday`.

Models use `nn.LazyLinear` for the input projection, so `C` is inferred on first forward
pass — adding new feature groups to `StoreData` does not require updating model constructors.

---

## Models

| File | Class | MLflow experiment | Status |
|------|-------|-------------------|--------|
| `main_store_sales_baseline.py` | `HoldLastValue` | `StoreSales_Baseline` | done — sets lower bound |
| `main_store_sales_transformer.py` | `StoreSalesTransformer` | `StoreSales_TransformerBasic` | superseded (decoder pathology, see `store_sales_transformer_notes.md`) |
| `main_store_sales_encoder_only.py` | `StoreSalesEncoderOnly` | `StoreSales_TransformerEncoderOnly` | **current focus** |

### `StoreSalesEncoderOnly`

Encoder-only Transformer: flatten → `LazyLinear` → sinusoidal PE → `TransformerEncoder`
→ pool → `LazyLinear` → unflatten → ReLU.

No causal masking. All encoder outputs projected to the full output horizon in one shot
(no autoregression). Pooling: `ALL` (flatten all timesteps) or `LAST` (take final token).

Key hyperparameters: `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `pooling_mode`.

---

## Hyperparameter Tuning

`time_series/tune_store_sales_encoder_only.py` — Optuna search over `lr`, `d_model`
(via `d_model_per_head`), `num_layers`, `dim_feedforward`. Each trial is a nested MLflow
run under a single parent study run.

---

## Experiment Tracking

MLflow local server. `TRACKING_URI` in `common/model_registry.py`. Each run logs:
- params: epochs, lr, batch_size, architecture hyperparams, feature flags
- tags: git branch + SHA, device, architecture name
- metrics: `train_loss`, `val_loss` (MSLE; take sqrt for RMSLE)
- artifacts: best model checkpoint, periodic checkpoints every N epochs

---

## Run Commands

```bash
# Baseline
uv run python -m time_series.main_store_sales_baseline

# Encoder-only (defaults)
uv run python -m time_series.main_store_sales_encoder_only

# Encoder-only with extra features
uv run python -m time_series.main_store_sales_encoder_only \
  --epochs 200 --lr 1.75e-3 --d-model 64 --num-layers 2 \
  --oil --onpromotion \
  --store-features city state type \
  --holiday-features national_holiday regional_holiday

# Hyperparameter search
uv run python -m time_series.tune_store_sales_encoder_only --n-trials 40 --epochs-per-trial 30
```

---

## Common Module

`common/` — shared utilities used across all projects:
- `paths.py` — data dir resolution
- `git.py` — branch/SHA for MLflow tags
- `model_registry.py` — `TRACKING_URI`
- `modules.py` — `PositionalEncoding`, `GetLastIndex`

---

## Remote Training on RunPod

See [RunPod deployment design](superpowers/specs/2026-07-05-runpod-deployment-design.md) for full design rationale.

### Prerequisites

1. RunPod account with SSH public key added in [RunPod Settings](https://www.runpod.io/console/user/settings)
2. API key in macOS Keychain:
   ```bash
   security add-generic-password -a "$USER" -s "RUNPOD_API_KEY" -w "<your-key>"
   ```
3. Add to `~/.zshrc`:
   ```bash
   export RUNPOD_API_KEY=$(security find-generic-password -a "$USER" -s "RUNPOD_API_KEY" -w 2>/dev/null)
   ```
4. Copy and fill in config:
   ```bash
   cp runpod_config.yaml.example runpod_config.yaml
   ```

### Quick start — single run (fully automated)

```bash
uv run python -m remote run --dataset store-sales-time-series-forecasting \
  -- python -m time_series.main_store_sales_encoder_only --epochs 200
```

### Manual workflow (fine-grained control)

```bash
# 1. Create pod
POD_ID=$(uv run python -m remote pod create)

# 2. Push code and data
uv run python -m remote sync push-code $POD_ID
uv run python -m remote sync push-data $POD_ID --dataset store-sales-time-series-forecasting

# 3. Set up environment
uv run python -m remote env setup $POD_ID
uv run python -m remote env mlflow-start $POD_ID

# 4. Train
uv run python -m remote train $POD_ID -- python -m time_series.main_store_sales_encoder_only --epochs 200

# 5. Pull results
uv run python -m remote sync pull $POD_ID

# 6. Clean up
uv run python -m remote pod terminate $POD_ID
```

### Hyperparameter sweep (parallel pods, shared MLflow)

```bash
# 1. Start persistent MLflow pod
MLFLOW_POD=$(uv run python -m remote sweep create-mlflow-pod)

# 2. Run N parallel training pods
uv run python -m remote sweep run \
  --mlflow-pod $MLFLOW_POD --n-pods 4 \
  --dataset store-sales-time-series-forecasting \
  -- python -m time_series.tune_store_sales_encoder_only --n-trials 20 --epochs-per-trial 30

# 3. Pull results and tear down
uv run python -m remote sweep pull $MLFLOW_POD
uv run python -m remote sweep teardown $MLFLOW_POD
```

### Remote deps group

When adding non-torch Python packages to `pyproject.toml`, also add them to the
`remote` dependency group so they are installed on RunPod pods:

```toml
[dependency-groups]
remote = [
    # add new non-torch packages here
]
```
