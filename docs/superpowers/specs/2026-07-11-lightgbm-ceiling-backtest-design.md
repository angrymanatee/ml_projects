# LightGBM Ceiling + Backtest Harness Design

**Date:** 2026-07-11
**Status:** Approved
**Scope:** A model-agnostic sliding-window backtest harness and a LightGBM global
ceiling model, to establish a trustworthy RMSLE upper bound before further deep-model work.

---

## Overview

The transformer models (encoder-only, hierarchical) have plateaued around MSLE ~0.89,
and every architecture comparison so far has been undermined by a single-split
evaluation whose rankings flipped three times across epoch budgets (see
`store_sales_transformer_notes.md`). This sub-project establishes two foundations:

1. **A sliding-window backtest harness** — model-agnostic, protocol-based, so every
   later model (iTransformer, TFT, N-HiTS, and the existing transformers) is evaluated
   the same trustworthy way: mean ± std RMSLE across multiple rolling-origin folds
   rather than one noisy split.

2. **A LightGBM global ceiling model** — the canonical Kaggle-competitive approach for
   this dataset (gradient-boosted trees with lag features, native categoricals, and
   known-future covariates). This gives a real RMSLE target to measure deep models
   against, and exploits an information source the transformers never used: promotions
   and holidays are *known in advance* for the forecast horizon.

This is sub-project 1 of a larger program ("establish the ceiling, then learn"). The
neural phase (iTransformer, TFT, N-HiTS, DLinear, RevIN) is a separate backlog, specced
later, and will reuse the harness built here.

---

## Motivating insight: known-future covariates

`onpromotion` is present in `test.csv` — the promotion schedule for all 16 forecast days
is known at prediction time. Holidays, paydays, and day-of-week are deterministic.
The transformer models only ever ingested these as *past* input channels; they were
never told "family X is on promotion on forecast day 12." Gradient-boosted trees use
future-known covariates natively. This is likely a real component of why deep models
plateau, and it is not an architecture problem — it is an information problem. The
LightGBM feature design below exploits it explicitly.

---

## Component A: Backtest harness

**Location:** `time_series/store_sales/backtest.py`

### Evaluation scheme

Rolling-origin (walk-forward) evaluation. `n_folds` (default 5) disjoint 16-day blocks
march backward from the end of the training data. For fold `i` with cutoff date `D_i`:

1. Fit the model on all data with date `<= D_i`.
2. Predict days `D_i + 1 … D_i + 16`.
3. Score RMSLE against actuals for that block.

Disjoint blocks (stride = horizon = 16) give `n_folds` independent RMSLE samples →
mean ± std. This directly addresses the single-split noise ball; a candidate is only
"better" if it wins beyond the fold-to-fold std.

### Interfaces

```python
from typing import Protocol
import numpy as np
import pandas as pd

class Forecaster(Protocol):
    """A model that can be fit on data up to a cutoff and predict the next horizon."""
    def fit(self, train_up_to: pd.Timestamp) -> None: ...
    def predict(self) -> np.ndarray:  # shape [horizon, n_stores, n_families], sales space
        ...

@dataclass
class BacktestConfig:
    n_folds: int = 5
    horizon: int = 16
    min_train_days: int = 365  # reject folds whose training slice is shorter than this

@dataclass
class BacktestResult:
    per_fold: pd.DataFrame       # columns: fold, cutoff, rmsle
    per_horizon: pd.DataFrame    # columns: fold, horizon_step (1..16), rmsle
    per_family: pd.DataFrame     # columns: fold, family, rmsle
    @property
    def mean_rmsle(self) -> float: ...
    @property
    def std_rmsle(self) -> float: ...
    def log_to_mlflow(self) -> None: ...

def backtest(
    forecaster_factory: Callable[[], Forecaster],
    data: StoreData,          # used only for fold-cutoff generation and actuals lookup
    config: BacktestConfig,
) -> BacktestResult: ...
```

**`forecaster_factory` builds a fresh model per fold** — no state or fitted parameters
leak across folds. Each fresh `Forecaster` closes over its own data source (the tabular
builder); the harness never hands it feature rows. The harness uses `data` (`StoreData`)
for exactly two things: generating fold cutoffs from the date range, and looking up
actuals to score against. The `fit(train_up_to)` signature takes only a timestamp
precisely because the model owns its data and slices internally at `<= train_up_to`.

### RMSLE

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_i \big(\log(1+\hat y_i) - \log(1+y_i)\big)^2}$$

Predictions clipped at 0 before scoring (sales are non-negative; `log1p` of a negative
prediction is undefined).

### Why model-agnostic matters

This is the reusable interface point. The neural models in the next sub-project implement
the same `Forecaster` protocol (wrapping their existing training loops) and reuse
`backtest()` verbatim. All cross-model comparisons then run through one evaluation path.

---

## Component B: LightGBM ceiling

**Locations:** `time_series/store_sales/tabular.py` (feature builder),
`time_series/store_sales/lgbm.py` (`LightGBMForecaster`),
`time_series/main_store_sales_lgbm.py` (entry point).

### Why LightGBM (not XGBoost / CatBoost)

Decided for this dataset specifically, not as a general claim:

1. **Native categorical handling.** `store` (54), `family` (33), `city`, `state`,
   `type`, `cluster` are categorical-heavy. LightGBM splits categoricals directly via
   Fisher's method (sort categories by $\sum g / \sum h$, optimal partition in
   $O(k \log k)$) with no one-hot blowup. XGBoost's native categorical support is newer
   and less battle-tested; CatBoost's is arguably best (ordered target statistics) but
   trains slower.
2. **Backtest throughput.** The harness does many fits; LightGBM's histogram + leaf-wise
   growth + GOSS/EFB is fastest per fit, which compounds across folds.

CatBoost is a legitimate swap if optimizing for "best number, least tuning" over
iteration speed; noted as a future option, not the first cut.

### Model structure: one global model, horizon as a feature

A single global LightGBM model over all (store, family) series and all 16 horizon steps,
with the horizon step `h` as an input feature — *not* 16 per-horizon models. Rationale:
pooling all horizons gives ~16× more rows per fit and shares day-of-week / seasonal
structure across horizons, at far lower fit/iteration cost (5 fits vs 80 across the
backtest). Per-horizon direct models specialize more (especially at long horizons) and
are the documented **squeeze-upgrade** if the pooled ceiling lands close enough to the
deep models to justify the last few percent.

Both are "direct" forecasting. Recursive forecasting (predict `D+1`, feed it back for
`D+2`…) is explicitly rejected: it accumulates error and cannot cleanly use
known-future-at-horizon covariates.

### Feature design (direct-multi-horizon correctness)

The correctness rule for direct forecasting drives the split between features evaluated
at the forecast origin `D` vs the target day `D + h`:

**Historical features — evaluated at origin `D` (all `<= D`, always valid for any `h`):**
- Sales lags at weekly multiples: `{7, 14, 21, 28, 35, 42, 49, 56, 63}` days.
- Rolling mean / std / min / max over windows `{7, 14, 28, 56}`.

**Known-future features — evaluated at target day `D + h`:**
- `onpromotion`, per store-family (the underused signal).
- Holiday flags (national / regional / local, event, bridge, work_day, additional).
- Calendar: day-of-week, day-of-month, month, is_15th, is_month_end, earthquake-decay.
- The horizon step `h` itself.

**Static categoricals (LightGBM native categorical):**
- `store`, `family`, `city`, `state`, `type`, `cluster`.

**Oil:** evaluated at origin `D` (future oil is not known; use last observed, ffill).

### Objective

- **Primary: L2 on `log1p(sales)`.** This is RMSLE per-sample —
  $(\log(1+\hat y) - \log(1+y))^2$ is exactly the RMSLE summand — so it directly targets
  the evaluation metric. Predict → `expm1` → clip at 0.
- **Challenger (optional, logged): Tweedie on raw sales** (`objective='tweedie'`). Sales
  are zero-inflated non-negative; Tweedie models the zero-mass + positive-continuous
  mixture explicitly. It is *not* a competing metric — it is a different training
  objective judged on the *same* backtest RMSLE. Kept only if it beats log1p-L2 on RMSLE;
  dropped otherwise.

Training objective and evaluation metric are distinct: both variants are scored by
backtest RMSLE regardless of what they optimize.

### Tuning

Modest only — the goal is a solid, trustworthy ceiling, not a Kaggle-winning squeeze.
`num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, and
`n_estimators` (via early stopping on a held-out validation fold). Deterministic with a
fixed seed.

### Runs locally

LightGBM is CPU-only and fast here — the full backtest runs on the Mac in minutes, no
RunPod needed. A deliberate contrast with the deep-model path.

---

## Data path

Reuse `StoreData`'s existing CSV loading and cleaning (oil ffill/bfill, holiday
state/city matching, promotion tensor construction) — do not duplicate that logic. Add
`tabular.py` that consumes the already-cleaned sources and emits a **long-form** training
frame, one row per `(origin_date D, store, family, horizon_step h)`, with target
`sales[D + h]`. So each `(origin, store, family)` fans out to 16 rows (one per
`h ∈ 1..16`); historical features are shared across those 16 rows (all evaluated at `D`),
known-future features and `h` vary per row (evaluated at `D + h`). LightGBM needs tabular
rows, not windowed tensors; the tensor `StoreData` path is untouched and continues to
serve the neural models.

---

## The comparison, made honest

The encoder-only best of **0.887 is MSLE** (mean squared log error, no square root);
the corresponding RMSLE is $\sqrt{0.887} \approx 0.94$. The entry point will confirm
exactly what `common.modules.MSLELoss` computes (whether it applies the sqrt) and report
the LightGBM ceiling in **both MSLE and RMSLE** so the deep-vs-GBM comparison is
apples-to-apples. This guards against an off-by-a-square-root error in the headline claim.

---

## MLflow logging

New experiment `StoreSales_LightGBM`. Per run:
- **params:** feature set, objective (log1p-L2 or tweedie), LightGBM hyperparameters,
  `n_folds`, horizon.
- **metrics:** backtest mean/std RMSLE, per-horizon RMSLE, per-family RMSLE, and MSLE
  (= RMSLE²) for direct comparison to the transformer experiments.
- **tags:** git branch + SHA, architecture = `lightgbm-global-direct`.

---

## File layout

| File | Purpose |
|------|---------|
| `time_series/store_sales/backtest.py` | `Forecaster` protocol, `BacktestConfig`, `BacktestResult`, `backtest()`, RMSLE |
| `time_series/store_sales/tabular.py` | long-form feature builder from cleaned raw data |
| `time_series/store_sales/lgbm.py` | `LightGBMForecaster` implementing `Forecaster` |
| `time_series/main_store_sales_lgbm.py` | entry point: build features, run backtest, log to MLflow |
| `tests/time_series/test_backtest.py` | harness tests |
| `tests/time_series/test_tabular.py` | feature-builder tests |
| `tests/time_series/test_lgbm.py` | forecaster + integration tests |

`lightgbm` added to `pyproject.toml` dependencies **and** the `remote` dependency group.

---

## Testing

- **Backtest harness:** fold-cutoff generation (correct dates, disjoint 16-day blocks,
  folds rejected when the training slice is shorter than `min_train_days`); RMSLE
  known-value test (hand-computed on a small array); aggregation (mean/std across folds);
  **leakage check** — a spy `Forecaster` asserts `fit` is only ever handed data `<= D_i`.
- **Feature builder:** output shape and column set; **no future leakage** — historical
  features at row `(D, h)` use only data `<= D`, known-future features use the value at
  `D + h`; lag correctness on a tiny synthetic frame with known values.
- **LightGBM forecaster:** fits and predicts the correct `[16, 54, 33]` shape; plugs
  into `backtest()` end-to-end; deterministic under a fixed seed.
- Reuse the existing `mock_data_dir` fixture pattern for integration tests.

---

## Out of scope (deferred to the neural sub-project)

- iTransformer, TFT, N-HiTS, PatchTST, DLinear.
- RevIN / per-series instance normalization.
- Wrapping the existing transformer models in the `Forecaster` protocol (straightforward
  once the harness exists; done when those models are re-evaluated).
- Ensembling LightGBM with a seasonal-naive baseline (a known ceiling-raiser, but the
  single-model ceiling comes first).
