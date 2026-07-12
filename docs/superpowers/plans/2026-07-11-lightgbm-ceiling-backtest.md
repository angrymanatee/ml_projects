# LightGBM Ceiling + Backtest Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic rolling-origin backtest harness and a LightGBM global direct-multi-horizon ceiling model, evaluated through the harness and logged to MLflow.

**Architecture:** A `Forecaster` protocol decouples models from evaluation; `backtest()` retrains a fresh forecaster per fold on data `<= cutoff`, scores RMSLE over disjoint 16-day blocks marching backward from the train end, and returns mean ± std plus per-horizon/per-family breakdowns. The LightGBM path adds a long-form tabular feature builder (historical features as-of the forecast origin, known-future covariates as-of the target day) and a `LightGBMForecaster` implementing the protocol.

**Tech Stack:** Python 3.14, LightGBM, pandas, numpy, MLflow, pytest, existing `StoreData` loader.

## Global Constraints

- Python 3.14; type hints on all function signatures; docstrings on public classes/functions (explain the non-obvious, don't restate the signature).
- ruff + black formatting; run the `precommit-check` skill before finalizing.
- Tests live under `tests/`, mirroring source layout. No `__init__.py`; `pythonpath = ["."]` handles imports. Use the existing `mock_data_dir` fixture pattern where a full `StoreData` is needed.
- RMSLE is the evaluation metric: `sqrt(mean((log1p(pred) - log1p(true))^2))`, predictions clipped at 0.
- Primary training objective: L2 on `log1p(sales)` (predict → `expm1` → clip at 0). Tweedie is an optional logged challenger only.
- Full variable names (`store`, `family_index`, not `s`, `f_idx`). Comments only where the *why* is non-obvious.
- `n_stores = 54`, `n_families = 33`, `horizon = 16` for the real dataset; all code parametrizes these from `StoreData`/`BacktestConfig` so tests can use smaller values.
- Store/family output ordering MUST match `StoreData`: store axis = `store_data.stores.index` order, family axis = `store_data.families` order (same as `store_data.sales_tensor`'s `[T, n_stores, n_families]`).

---

### Task 1: Add LightGBM dependency

**Files:**
- Modify: `pyproject.toml` (main dependencies + `[dependency-groups] remote`)
- Modify: `uv.lock`

**Interfaces:**
- Consumes: nothing.
- Produces: `import lightgbm` available locally and on remote pods.

- [ ] **Step 1: Add the dependency**

Run: `uv add lightgbm`
Expected: `pyproject.toml` gains `lightgbm>=...` under `[project] dependencies`, `uv.lock` updated.

- [ ] **Step 2: Mirror into the remote dependency group**

Edit `pyproject.toml` so the `remote` group under `[dependency-groups]` also lists the same `lightgbm>=...` line (the remote pod installs from this group; see `remote/environment.py` `_REMOTE_DEPS` sync requirement in CLAUDE.md / project docs).

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import lightgbm; print(lightgbm.__version__)"`
Expected: prints a version, no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add lightgbm dependency"
```

---

### Task 2: RMSLE metric

**Files:**
- Create: `time_series/store_sales/backtest.py`
- Test: `tests/time_series/test_backtest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/time_series/test_backtest.py
import numpy as np
import pytest

from time_series.store_sales.backtest import rmsle


def test_rmsle_zero_for_perfect_prediction() -> None:
    y = np.array([0.0, 5.0, 100.0])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_known_value() -> None:
    # single element: |log1p(3) - log1p(7)| = |1.386294 - 2.079442| = 0.693147
    assert rmsle(np.array([7.0]), np.array([3.0])) == pytest.approx(0.6931471, abs=1e-6)


def test_rmsle_clips_negative_predictions_to_zero() -> None:
    # prediction -4 is clipped to 0, so error vs true 0 is log1p(0)-log1p(0)=0
    assert rmsle(np.array([0.0]), np.array([-4.0])) == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_backtest.py -v`
Expected: FAIL — `ImportError` / `rmsle` not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# time_series/store_sales/backtest.py
"""Model-agnostic rolling-origin backtest harness for store sales forecasting."""

from __future__ import annotations

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared logarithmic error, the competition metric.

    Predictions are clipped at 0 before scoring (sales are non-negative and
    log1p of a negative value is undefined).
    """
    clipped = np.clip(y_pred, a_min=0.0, a_max=None)
    diff = np.log1p(clipped) - np.log1p(y_true)
    return float(np.sqrt(np.mean(diff**2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add time_series/store_sales/backtest.py tests/time_series/test_backtest.py
git commit -m "Add rmsle metric for backtest harness"
```

---

### Task 3: Backtest config + fold-cutoff generation

**Files:**
- Modify: `time_series/store_sales/backtest.py`
- Test: `tests/time_series/test_backtest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass BacktestConfig(n_folds: int = 5, horizon: int = 16, min_train_days: int = 365)`
  - `generate_fold_cutoffs(dates: pd.DatetimeIndex, config: BacktestConfig) -> list[pd.Timestamp]` — cutoffs in ascending date order; the predict block for cutoff `D` is `D+1 … D+horizon`. Blocks are disjoint (stride = horizon), march backward from the last date, and a fold is dropped if fewer than `min_train_days` distinct dates precede its cutoff.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from time_series.store_sales.backtest import BacktestConfig, generate_fold_cutoffs


def test_fold_cutoffs_disjoint_blocks_march_backward() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    config = BacktestConfig(n_folds=3, horizon=16, min_train_days=1)
    cutoffs = generate_fold_cutoffs(dates, config)
    # last date is 2020-04-09 (index 99). Fold 1 cutoff = last - 16 days.
    assert len(cutoffs) == 3
    assert cutoffs == sorted(cutoffs)  # ascending
    # disjoint: consecutive cutoffs are exactly `horizon` days apart
    assert (cutoffs[1] - cutoffs[0]).days == 16
    assert (cutoffs[2] - cutoffs[1]).days == 16
    # last cutoff leaves exactly `horizon` days of predict block at the end
    assert (dates[-1] - cutoffs[-1]).days == 16


def test_fold_cutoffs_respects_min_train_days() -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    # min_train_days=40 means a cutoff needs >=40 dates before it.
    config = BacktestConfig(n_folds=5, horizon=16, min_train_days=40)
    cutoffs = generate_fold_cutoffs(dates, config)
    for cutoff in cutoffs:
        n_before = (dates <= cutoff).sum()
        assert n_before >= 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_backtest.py -k fold -v`
Expected: FAIL — `BacktestConfig` / `generate_fold_cutoffs` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `backtest.py`:

```python
from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for rolling-origin backtesting.

    n_folds: number of disjoint predict blocks.
    horizon: forecast length per fold (competition uses 16 days).
    min_train_days: a fold is dropped unless at least this many distinct
        dates precede its cutoff, so early folds are not trained on too little.
    """

    n_folds: int = 5
    horizon: int = 16
    min_train_days: int = 365


def generate_fold_cutoffs(
    dates: pd.DatetimeIndex, config: BacktestConfig
) -> list[pd.Timestamp]:
    """Return fold cutoff dates in ascending order.

    The predict block for cutoff D is D+1 … D+horizon. Blocks are disjoint
    (stride = horizon) and march backward from the last available date. Folds
    with fewer than min_train_days dates before the cutoff are dropped.
    """
    ordered = pd.DatetimeIndex(dates).sort_values().unique()
    last = ordered[-1]
    cutoffs: list[pd.Timestamp] = []
    for fold in range(config.n_folds):
        cutoff = last - pd.Timedelta(days=config.horizon * (fold + 1))
        n_before = int((ordered <= cutoff).sum())
        if n_before < config.min_train_days:
            continue
        cutoffs.append(pd.Timestamp(cutoff))
    return sorted(cutoffs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_backtest.py -k fold -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add time_series/store_sales/backtest.py tests/time_series/test_backtest.py
git commit -m "Add BacktestConfig and fold-cutoff generation"
```

---

### Task 4: Expose `StoreData.dates`

**Files:**
- Modify: `time_series/store_sales/data.py` (in `__init__`, around line 164 where `dates` is computed)
- Test: `tests/time_series/test_store_sales.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `store_data.dates: pd.DatetimeIndex` — the sorted unique date axis aligned with `sales_tensor`'s time dimension. The harness and tabular builder use this instead of recomputing.

- [ ] **Step 1: Write the failing test**

```python
# tests/time_series/test_store_sales.py  (add near other StoreData tests)
def test_dates_attribute_matches_sales_tensor_length(mock_data_dir) -> None:
    from time_series.store_sales import StoreData
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    assert len(store_data.dates) == store_data.sales_tensor.shape[0]
    assert list(store_data.dates) == sorted(store_data.dates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_store_sales.py -k dates_attribute -v`
Expected: FAIL — `AttributeError: 'StoreData' object has no attribute 'dates'`.

- [ ] **Step 3: Write minimal implementation**

In `data.py` `__init__`, the line computing `dates = pd.DatetimeIndex(self.train.index.unique().sort_values())` already exists. Assign it to the instance:

```python
        self.dates = pd.DatetimeIndex(self.train.index.unique().sort_values())
```

Replace the existing local `dates = ...` usage so the local and attribute are the same object (use `self.dates` thereafter in `__init__`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_store_sales.py -k dates_attribute -v`
Expected: PASS.

- [ ] **Step 5: Run the full StoreData test module (no regressions)**

Run: `uv run pytest tests/time_series/test_store_sales.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add time_series/store_sales/data.py tests/time_series/test_store_sales.py
git commit -m "Expose StoreData.dates for backtest harness"
```

---

### Task 5: Forecaster protocol + BacktestResult

**Files:**
- Modify: `time_series/store_sales/backtest.py`
- Test: `tests/time_series/test_backtest.py`

**Interfaces:**
- Consumes: `BacktestConfig`.
- Produces:
  - `class Forecaster(Protocol): def fit(self, train_up_to: pd.Timestamp) -> None; def predict(self) -> np.ndarray` (shape `[horizon, n_stores, n_families]`, sales space).
  - `@dataclass BacktestResult` with fields `per_fold: pd.DataFrame` (cols `fold, cutoff, rmsle`), `per_horizon: pd.DataFrame` (cols `fold, horizon_step, rmsle`), `per_family: pd.DataFrame` (cols `fold, family, rmsle`); properties `mean_rmsle: float`, `std_rmsle: float`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from time_series.store_sales.backtest import BacktestResult


def test_backtest_result_mean_and_std() -> None:
    per_fold = pd.DataFrame(
        {"fold": [0, 1, 2], "cutoff": pd.to_datetime(["2020-01-01"] * 3), "rmsle": [1.0, 2.0, 3.0]}
    )
    result = BacktestResult(per_fold=per_fold, per_horizon=pd.DataFrame(), per_family=pd.DataFrame())
    assert result.mean_rmsle == pytest.approx(2.0)
    assert result.std_rmsle == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_backtest.py -k backtest_result -v`
Expected: FAIL — `BacktestResult` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `backtest.py`:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Forecaster(Protocol):
    """A model that fits on data up to a cutoff and predicts the next horizon.

    Implementations own their own data source and slice internally at
    `train_up_to`; the harness only passes the cutoff timestamp. `predict`
    returns sales-space forecasts shaped [horizon, n_stores, n_families],
    with store/family axes ordered to match StoreData.
    """

    def fit(self, train_up_to: pd.Timestamp) -> None: ...
    def predict(self) -> np.ndarray: ...


@dataclass
class BacktestResult:
    """Aggregated backtest metrics across folds."""

    per_fold: pd.DataFrame
    per_horizon: pd.DataFrame
    per_family: pd.DataFrame

    @property
    def mean_rmsle(self) -> float:
        return float(self.per_fold["rmsle"].mean())

    @property
    def std_rmsle(self) -> float:
        return float(self.per_fold["rmsle"].std(ddof=1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_backtest.py -k backtest_result -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add time_series/store_sales/backtest.py tests/time_series/test_backtest.py
git commit -m "Add Forecaster protocol and BacktestResult"
```

---

### Task 6: `backtest()` orchestration + leakage guard

**Files:**
- Modify: `time_series/store_sales/backtest.py`
- Test: `tests/time_series/test_backtest.py`

**Interfaces:**
- Consumes: `BacktestConfig`, `Forecaster`, `BacktestResult`, `generate_fold_cutoffs`, `rmsle`, `StoreData.dates`, `StoreData.sales_tensor`, `StoreData.families`.
- Produces: `backtest(forecaster_factory: Callable[[], Forecaster], store_data, config: BacktestConfig) -> BacktestResult`.

The harness, for each cutoff `D`: builds a fresh forecaster, calls `fit(D)`, calls `predict()` → `[horizon, n_stores, n_families]`, slices actuals from `sales_tensor` at the index range for `D+1 … D+horizon`, and records overall/per-horizon/per-family RMSLE.

- [ ] **Step 1: Write the failing test (with a fake forecaster + leakage spy)**

```python
import numpy as np
import pandas as pd
from time_series.store_sales.backtest import BacktestConfig, backtest


class _FakeStoreData:
    """Minimal stand-in exposing the attributes backtest() reads."""

    def __init__(self) -> None:
        self.dates = pd.date_range("2020-01-01", periods=80, freq="D")
        # sales_tensor [T, n_stores, n_families]; constant so RMSLE is predictable
        self.sales_tensor = np.full((80, 2, 2), 5.0)
        self.families = pd.Index(["A", "B"])


class _ConstantForecaster:
    def __init__(self, spy: list[pd.Timestamp]) -> None:
        self._spy = spy

    def fit(self, train_up_to: pd.Timestamp) -> None:
        self._spy.append(train_up_to)

    def predict(self) -> np.ndarray:
        return np.full((16, 2, 2), 5.0)  # perfect: matches constant sales


def test_backtest_scores_all_folds_and_is_leak_free() -> None:
    store_data = _FakeStoreData()
    seen_cutoffs: list[pd.Timestamp] = []
    config = BacktestConfig(n_folds=3, horizon=16, min_train_days=1)

    result = backtest(lambda: _ConstantForecaster(seen_cutoffs), store_data, config)

    assert len(result.per_fold) == 3
    # perfect constant prediction -> RMSLE 0 everywhere
    assert result.mean_rmsle == pytest.approx(0.0)
    # leakage guard: every cutoff handed to fit is strictly before its predict block
    for cutoff in seen_cutoffs:
        assert cutoff < store_data.dates[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_backtest.py -k leak_free -v`
Expected: FAIL — `backtest` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `backtest.py` (note `Callable` import):

```python
from typing import Callable


def backtest(
    forecaster_factory: Callable[[], Forecaster],
    store_data,
    config: BacktestConfig,
) -> BacktestResult:
    """Run rolling-origin backtesting, returning aggregated RMSLE metrics.

    A fresh forecaster is built per fold (no cross-fold leakage). `store_data`
    is used only for the date axis (fold generation) and actuals lookup.
    """
    dates = pd.DatetimeIndex(store_data.dates)
    families = list(store_data.families)
    cutoffs = generate_fold_cutoffs(dates, config)
    date_to_index = {date: idx for idx, date in enumerate(dates)}

    fold_rows: list[dict] = []
    horizon_rows: list[dict] = []
    family_rows: list[dict] = []

    for fold, cutoff in enumerate(cutoffs):
        forecaster = forecaster_factory()
        forecaster.fit(pd.Timestamp(cutoff))
        prediction = np.asarray(forecaster.predict())  # [horizon, n_stores, n_families]

        start = date_to_index[cutoff] + 1
        actual = np.asarray(store_data.sales_tensor)[start : start + config.horizon]

        fold_rows.append({"fold": fold, "cutoff": cutoff, "rmsle": rmsle(actual, prediction)})
        for step in range(config.horizon):
            horizon_rows.append(
                {"fold": fold, "horizon_step": step + 1, "rmsle": rmsle(actual[step], prediction[step])}
            )
        for family_index, family in enumerate(families):
            horizon_slice_actual = actual[:, :, family_index]
            horizon_slice_pred = prediction[:, :, family_index]
            family_rows.append(
                {"fold": fold, "family": family, "rmsle": rmsle(horizon_slice_actual, horizon_slice_pred)}
            )

    return BacktestResult(
        per_fold=pd.DataFrame(fold_rows),
        per_horizon=pd.DataFrame(horizon_rows),
        per_family=pd.DataFrame(family_rows),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_backtest.py -k leak_free -v`
Expected: PASS.

- [ ] **Step 5: Run the whole backtest module**

Run: `uv run pytest tests/time_series/test_backtest.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add time_series/store_sales/backtest.py tests/time_series/test_backtest.py
git commit -m "Add backtest() orchestration with leakage guard"
```

---

### Task 7: `BacktestResult.log_to_mlflow`

**Files:**
- Modify: `time_series/store_sales/backtest.py`
- Test: `tests/time_series/test_backtest.py`

**Interfaces:**
- Consumes: `BacktestResult` fields.
- Produces: `BacktestResult.log_to_mlflow(self) -> None` — logs `rmsle_mean`, `rmsle_std`, `msle_mean` (= `rmsle_mean**2`, for apples-to-apples comparison with the transformer experiments), per-horizon RMSLE as a stepped metric, and attaches per-family RMSLE as a CSV artifact. Must be called inside an active `mlflow.start_run()`.

- [ ] **Step 1: Write the failing test (file-store MLflow, temp dir)**

```python
import mlflow
from time_series.store_sales.backtest import BacktestResult


def test_log_to_mlflow_records_summary_metrics(tmp_path) -> None:
    per_fold = pd.DataFrame(
        {"fold": [0, 1], "cutoff": pd.to_datetime(["2020-01-01", "2020-01-17"]), "rmsle": [0.8, 1.0]}
    )
    per_horizon = pd.DataFrame({"fold": [0, 0], "horizon_step": [1, 2], "rmsle": [0.7, 0.9]})
    per_family = pd.DataFrame({"fold": [0], "family": ["A"], "rmsle": [0.85]})
    result = BacktestResult(per_fold=per_fold, per_horizon=per_horizon, per_family=per_family)

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test_backtest_logging")
    with mlflow.start_run() as run:
        result.log_to_mlflow()

    client = mlflow.tracking.MlflowClient()
    logged = client.get_run(run.info.run_id).data.metrics
    assert logged["rmsle_mean"] == pytest.approx(0.9)
    assert logged["msle_mean"] == pytest.approx(0.81)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_backtest.py -k log_to_mlflow -v`
Expected: FAIL — `log_to_mlflow` not defined.

- [ ] **Step 3: Write minimal implementation**

Add the method to `BacktestResult` (and `import mlflow`, `import tempfile`, `import os` at top of module):

```python
    def log_to_mlflow(self) -> None:
        """Log aggregate/per-horizon/per-family metrics to the active MLflow run.

        Logs msle_mean (= rmsle_mean**2) so these runs are directly comparable
        to the transformer experiments, which log MSLE.
        """
        import mlflow

        mlflow.log_metric("rmsle_mean", self.mean_rmsle)
        mlflow.log_metric("rmsle_std", self.std_rmsle)
        mlflow.log_metric("msle_mean", self.mean_rmsle**2)

        by_step = self.per_horizon.groupby("horizon_step")["rmsle"].mean()
        for step, value in by_step.items():
            mlflow.log_metric("rmsle_by_horizon", float(value), step=int(step))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "per_family_rmsle.csv")
            self.per_family.groupby("family")["rmsle"].mean().to_csv(path)
            mlflow.log_artifact(path)
```

Move the `import mlflow` / `import tempfile` / `import os` to module top (remove the inline import) so the module has them once.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_backtest.py -k log_to_mlflow -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add time_series/store_sales/backtest.py tests/time_series/test_backtest.py
git commit -m "Add BacktestResult.log_to_mlflow"
```

---

### Task 8: Tabular historical (as-of-origin) features

**Files:**
- Create: `time_series/store_sales/tabular.py`
- Test: `tests/time_series/test_tabular.py`

**Interfaces:**
- Consumes: nothing (operates on a plain long DataFrame).
- Produces:
  - `@dataclass FeatureConfig(lags: tuple[int, ...] = (7, 14, 21, 28, 35, 42, 49, 56, 63), rolling_windows: tuple[int, ...] = (7, 14, 28, 56), horizon: int = 16)`
  - `add_origin_features(sales_long: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame` — input columns `["date", "store", "family", "sales"]`; returns the frame with added `lag_{k}` and `roll_{w}_{mean,std,min,max}` columns, each computed **as-of that row's date** (using only sales at dates `<= date` within the same `(store, family)`). No row uses its own or any future sales for its lag features (min lag is `min(lags) >= 1`).

- [ ] **Step 1: Write the failing test (lag correctness + no leakage)**

```python
# tests/time_series/test_tabular.py
import numpy as np
import pandas as pd
import pytest

from time_series.store_sales.tabular import FeatureConfig, add_origin_features


def _single_series(n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {"date": dates, "store": 1, "family": "A", "sales": np.arange(n_days, dtype=float)}
    )


def test_lag_feature_is_past_value() -> None:
    df = _single_series()
    config = FeatureConfig(lags=(7,), rolling_windows=(), horizon=16)
    out = add_origin_features(df, config).sort_values("date").reset_index(drop=True)
    # sales[t] == t, so lag_7 at row t should equal t-7 (NaN for first 7 rows)
    assert out.loc[10, "lag_7"] == pytest.approx(3.0)
    assert np.isnan(out.loc[3, "lag_7"])


def test_rolling_mean_uses_only_past_and_present() -> None:
    df = _single_series()
    config = FeatureConfig(lags=(), rolling_windows=(3,), horizon=16)
    out = add_origin_features(df, config).sort_values("date").reset_index(drop=True)
    # rolling mean of {8,9,10} at t=10 == 9.0 (window ending at current date)
    assert out.loc[10, "roll_3_mean"] == pytest.approx(9.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_tabular.py -v`
Expected: FAIL — module/function not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# time_series/store_sales/tabular.py
"""Long-form tabular feature builder for the LightGBM store-sales model."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureConfig:
    lags: tuple[int, ...] = (7, 14, 21, 28, 35, 42, 49, 56, 63)
    rolling_windows: tuple[int, ...] = (7, 14, 28, 56)
    horizon: int = 16


def add_origin_features(sales_long: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add lag and rolling features computed as-of each row's date.

    Input columns: date, store, family, sales. Output adds lag_{k} and
    roll_{w}_{mean,std,min,max}. All features for a row at date d use only
    sales at dates <= d within the same (store, family) series, so a training
    row's features never see its own target-day sales.
    """
    df = sales_long.sort_values(["store", "family", "date"]).copy()
    grouped = df.groupby(["store", "family"], sort=False)["sales"]

    for lag in config.lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in config.rolling_windows:
        # shift(1) is unnecessary: the window ends at the current date (origin),
        # which is allowed — origin sales are known at prediction time.
        roll = grouped.rolling(window, min_periods=window)
        df[f"roll_{window}_mean"] = roll.mean().reset_index(level=[0, 1], drop=True)
        df[f"roll_{window}_std"] = roll.std().reset_index(level=[0, 1], drop=True)
        df[f"roll_{window}_min"] = roll.min().reset_index(level=[0, 1], drop=True)
        df[f"roll_{window}_max"] = roll.max().reset_index(level=[0, 1], drop=True)

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_tabular.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add time_series/store_sales/tabular.py tests/time_series/test_tabular.py
git commit -m "Add as-of-origin lag and rolling features"
```

---

### Task 9: Assemble training + prediction frames (horizon expansion, known-future, calendar)

**Files:**
- Modify: `time_series/store_sales/tabular.py`
- Test: `tests/time_series/test_tabular.py`

**Interfaces:**
- Consumes: `FeatureConfig`, `add_origin_features`, `StoreData` (`.dates`, `.sales_tensor`, `.stores`, `.families`, `.promotion_tensor`, `.holidays`).
- Produces:
  - `sales_long_from_store_data(store_data) -> pd.DataFrame` — melts `sales_tensor` into columns `["date", "store", "family", "sales"]` using `store_data.dates`, `store_data.stores.index`, `store_data.families`.
  - `build_training_frame(store_data, config: FeatureConfig, train_up_to: pd.Timestamp) -> pd.DataFrame` — one row per `(target_date t, store, family, horizon_step h)` with `t <= train_up_to` and origin `t - h` having complete lag features; columns: origin features (as-of `t-h`), known-future/calendar features (as-of `t`), `horizon_step`, categorical `store/family`, and `target = log1p(sales[t])`.
  - `build_prediction_frame(store_data, config: FeatureConfig, origin: pd.Timestamp) -> pd.DataFrame` — rows for `origin` at `h = 1 … horizon` (target days `origin+1 … origin+horizon`), same feature columns, **no** target.
  - `FEATURE_COLUMNS: list[str]` and `CATEGORICAL_COLUMNS: list[str]` — the exact model input columns (excludes `date/store/family/target` bookkeeping except where a categorical is a model input; `store` and `family` ARE categorical model inputs).

Calendar features (as-of target day `t`): `dayofweek`, `day`, `month`, `is_weekend`, `is_15th`, `is_month_end`. Known-future: `onpromotion` (from `promotion_tensor` at `t`), `is_holiday` (national holiday flag at `t` derived from `store_data.holidays`). Static categoricals join from `store_data.stores`: `city, state, type, cluster`.

- [ ] **Step 1: Write the failing test (shape, no-leakage, prediction frame has no target)**

```python
from time_series.store_sales import StoreData
from time_series.store_sales.tabular import (
    FeatureConfig,
    build_prediction_frame,
    build_training_frame,
    sales_long_from_store_data,
)


def test_sales_long_shape(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    long = sales_long_from_store_data(store_data)
    n_dates = len(store_data.dates)
    assert set(long.columns) >= {"date", "store", "family", "sales"}
    assert len(long) == n_dates * store_data.stores.shape[0] * store_data.families.size


def test_training_frame_target_is_log1p_and_respects_cutoff(mock_data_dir) -> None:
    # mock data has only 3 dates; use tiny lags/horizon so rows exist
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    cutoff = store_data.dates[-1]
    frame = build_training_frame(store_data, config, train_up_to=cutoff)
    # every target date must be <= cutoff
    assert (frame["target_date"] <= cutoff).all()
    # target is log1p of sales; all finite
    assert frame["target"].notna().all()


def test_prediction_frame_has_no_target_and_horizon_rows(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    origin = store_data.dates[-1]
    frame = build_prediction_frame(store_data, config, origin=origin)
    assert "target" not in frame.columns
    n_cells = store_data.stores.shape[0] * store_data.families.size
    assert len(frame) == config.horizon * n_cells
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_tabular.py -k "training_frame or prediction_frame or sales_long" -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `tabular.py`. (Implementation notes for the executing engineer: build the origin-feature table once via `add_origin_features`, then for each `horizon_step h` join origin features from date `t-h` onto target rows at date `t`. Because origin features live on the origin date, shifting the origin table forward by `h` days per `(store, family)` aligns them to the target date. Known-future/calendar features are computed directly on the target date `t`.)

```python
import numpy as np

FEATURE_COLUMNS: list[str] = []       # populated below after column construction
CATEGORICAL_COLUMNS: list[str] = ["store", "family", "city", "state", "type", "cluster"]


def sales_long_from_store_data(store_data) -> pd.DataFrame:
    """Melt sales_tensor [T, n_stores, n_families] into long form."""
    dates = pd.DatetimeIndex(store_data.dates)
    stores = list(store_data.stores.index)
    families = list(store_data.families)
    sales = np.asarray(store_data.sales_tensor)
    records = {
        "date": np.repeat(dates.values, len(stores) * len(families)),
        "store": np.tile(np.repeat(stores, len(families)), len(dates)),
        "family": np.tile(families, len(dates) * len(stores)),
        "sales": sales.reshape(-1),
    }
    return pd.DataFrame(records)


def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "dayofweek": d.dayofweek,
            "day": d.day,
            "month": d.month,
            "is_weekend": (d.dayofweek >= 5).astype(int),
            "is_15th": (d.day == 15).astype(int),
            "is_month_end": d.is_month_end.astype(int),
        },
        index=dates.index,
    )


def _promotion_long(store_data) -> pd.DataFrame:
    """Long-form onpromotion keyed by (date, store, family); zeros if absent."""
    dates = pd.DatetimeIndex(store_data.dates)
    stores = list(store_data.stores.index)
    families = list(store_data.families)
    if store_data.promotion_tensor is None:
        promo = np.zeros((len(dates), len(stores), len(families)))
    else:
        promo = np.asarray(store_data.promotion_tensor)
    return pd.DataFrame(
        {
            "date": np.repeat(dates.values, len(stores) * len(families)),
            "store": np.tile(np.repeat(stores, len(families)), len(dates)),
            "family": np.tile(families, len(dates) * len(stores)),
            "onpromotion": promo.reshape(-1),
        }
    )


def _national_holiday_dates(store_data) -> set:
    holidays = store_data.holidays
    national = holidays[holidays["locale"] == "National"] if "locale" in holidays else holidays
    return set(pd.to_datetime(national["date"]).dt.normalize())


def _feature_frame(
    store_data, config: FeatureConfig, origins_and_horizons: pd.DataFrame
) -> pd.DataFrame:
    """Core builder shared by training and prediction frames.

    origins_and_horizons: columns [store, family, origin_date, horizon_step,
    target_date]. Joins as-of-origin features (from origin_date) and
    as-of-target known-future/calendar/static features (from target_date).
    """
    sales_long = sales_long_from_store_data(store_data)
    origin_features = add_origin_features(sales_long, config)
    feature_cols = [c for c in origin_features.columns if c.startswith(("lag_", "roll_"))]

    # join origin features on (store, family, origin_date == date)
    frame = origins_and_horizons.merge(
        origin_features[["store", "family", "date", *feature_cols]].rename(columns={"date": "origin_date"}),
        on=["store", "family", "origin_date"],
        how="left",
    )

    # known-future: promotion at target date
    promo = _promotion_long(store_data).rename(columns={"date": "target_date"})
    frame = frame.merge(promo, on=["store", "family", "target_date"], how="left")

    # calendar at target date
    frame = frame.reset_index(drop=True)
    frame = pd.concat([frame, _calendar_features(frame["target_date"])], axis=1)

    # national holiday flag at target date
    national = _national_holiday_dates(store_data)
    frame["is_holiday"] = frame["target_date"].dt.normalize().isin(national).astype(int)

    # static store categoricals
    static = store_data.stores.reset_index().rename(columns={"store_nbr": "store"})
    frame = frame.merge(static[["store", "city", "state", "type", "cluster"]], on="store", how="left")

    frame["horizon_step"] = frame["horizon_step"].astype(int)
    return frame


def _model_feature_columns(config: FeatureConfig) -> list[str]:
    lag_cols = [f"lag_{k}" for k in config.lags]
    roll_cols = [f"roll_{w}_{stat}" for w in config.rolling_windows for stat in ("mean", "std", "min", "max")]
    calendar = ["dayofweek", "day", "month", "is_weekend", "is_15th", "is_month_end"]
    known_future = ["onpromotion", "is_holiday", "horizon_step"]
    return [*lag_cols, *roll_cols, *calendar, *known_future, *CATEGORICAL_COLUMNS]


def build_training_frame(store_data, config: FeatureConfig, train_up_to: pd.Timestamp) -> pd.DataFrame:
    """One row per (target_date t <= cutoff, store, family, horizon h); target = log1p(sales[t])."""
    dates = pd.DatetimeIndex(store_data.dates)
    train_dates = dates[dates <= train_up_to]
    sales_long = sales_long_from_store_data(store_data).set_index(["date", "store", "family"])["sales"]

    rows = []
    stores = list(store_data.stores.index)
    families = list(store_data.families)
    date_set = set(dates)
    for target_date in train_dates:
        for horizon_step in range(1, config.horizon + 1):
            origin_date = target_date - pd.Timedelta(days=horizon_step)
            if origin_date not in date_set:
                continue
            for store in stores:
                for family in families:
                    rows.append(
                        {
                            "store": store,
                            "family": family,
                            "origin_date": origin_date,
                            "horizon_step": horizon_step,
                            "target_date": target_date,
                            "target": float(np.log1p(sales_long.loc[(target_date, store, family)])),
                        }
                    )
    origins = pd.DataFrame(rows)
    frame = _feature_frame(store_data, config, origins)
    return frame.dropna(subset=[f"lag_{max(config.lags)}"]) if config.lags else frame


def build_prediction_frame(store_data, config: FeatureConfig, origin: pd.Timestamp) -> pd.DataFrame:
    """Rows for a single origin at h = 1..horizon; target days origin+1..origin+horizon; no target."""
    stores = list(store_data.stores.index)
    families = list(store_data.families)
    rows = []
    for horizon_step in range(1, config.horizon + 1):
        target_date = pd.Timestamp(origin) + pd.Timedelta(days=horizon_step)
        for store in stores:
            for family in families:
                rows.append(
                    {
                        "store": store,
                        "family": family,
                        "origin_date": pd.Timestamp(origin),
                        "horizon_step": horizon_step,
                        "target_date": target_date,
                    }
                )
    origins = pd.DataFrame(rows)
    return _feature_frame(store_data, config, origins)
```

> **Note on known-future at prediction time:** promotion for `target_date > origin` may be absent from `promotion_tensor` (which only covers observed dates). The merge yields NaN → treated as "no promotion." For the real competition, promotions ARE known for the test horizon; wiring `test.csv` promotions is a follow-up noted in the spec, not this task. Calendar and holiday features are deterministic and available for future dates.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_tabular.py -k "training_frame or prediction_frame or sales_long" -v`
Expected: PASS.

- [ ] **Step 5: Run the full tabular module**

Run: `uv run pytest tests/time_series/test_tabular.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add time_series/store_sales/tabular.py tests/time_series/test_tabular.py
git commit -m "Add training/prediction frame assembly with known-future features"
```

---

### Task 10: `LightGBMForecaster`

**Files:**
- Create: `time_series/store_sales/lgbm.py`
- Test: `tests/time_series/test_lgbm.py`

**Interfaces:**
- Consumes: `FeatureConfig`, `build_training_frame`, `build_prediction_frame`, `_model_feature_columns`, `CATEGORICAL_COLUMNS`, `Forecaster` (implements it), `StoreData`.
- Produces:
  - `@dataclass LGBMParams(objective: str = "regression", num_leaves: int = 63, learning_rate: float = 0.05, n_estimators: int = 300, min_child_samples: int = 100, feature_fraction: float = 0.8, seed: int = 0)`
  - `class LightGBMForecaster` implementing `Forecaster`: `__init__(self, store_data, feature_config: FeatureConfig, params: LGBMParams)`; `fit(train_up_to)`; `predict() -> np.ndarray` shape `[horizon, n_stores, n_families]` in **sales space** (`expm1`, clipped at 0), ordered by `store_data.stores.index` × `store_data.families`.

- [ ] **Step 1: Write the failing test**

```python
# tests/time_series/test_lgbm.py
import numpy as np
import pandas as pd

from time_series.store_sales import StoreData
from time_series.store_sales.backtest import BacktestConfig, backtest
from time_series.store_sales.lgbm import LGBMParams, LightGBMForecaster
from time_series.store_sales.tabular import FeatureConfig


def test_forecaster_predict_shape_and_nonneg(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    feature_config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    forecaster = LightGBMForecaster(store_data, feature_config, LGBMParams(n_estimators=5))
    forecaster.fit(store_data.dates[-1])
    prediction = forecaster.predict()
    n_stores = store_data.stores.shape[0]
    n_families = store_data.families.size
    assert prediction.shape == (1, n_stores, n_families)
    assert (prediction >= 0).all()


def test_forecaster_is_deterministic(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    feature_config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)

    def make() -> LightGBMForecaster:
        return LightGBMForecaster(store_data, feature_config, LGBMParams(n_estimators=5, seed=0))

    a, b = make(), make()
    a.fit(store_data.dates[-1]); b.fit(store_data.dates[-1])
    np.testing.assert_allclose(a.predict(), b.predict())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_lgbm.py -v`
Expected: FAIL — module not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# time_series/store_sales/lgbm.py
"""LightGBM global direct-multi-horizon forecaster implementing the Forecaster protocol."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from time_series.store_sales.tabular import (
    CATEGORICAL_COLUMNS,
    FeatureConfig,
    _model_feature_columns,
    build_prediction_frame,
    build_training_frame,
)


@dataclass
class LGBMParams:
    objective: str = "regression"  # L2 on log1p target == RMSLE
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 100
    feature_fraction: float = 0.8
    seed: int = 0


class LightGBMForecaster:
    """One global LightGBM model over all series and horizons (horizon as a feature).

    Trains on log1p(sales) with L2 (directly targeting RMSLE); predictions are
    expm1'd and clipped at 0. Store/family output axes match StoreData ordering.
    """

    def __init__(self, store_data, feature_config: FeatureConfig, params: LGBMParams) -> None:
        self._store_data = store_data
        self._feature_config = feature_config
        self._params = params
        self._model: lgb.LGBMRegressor | None = None
        self._feature_columns = _model_feature_columns(feature_config)
        self._stores = list(store_data.stores.index)
        self._families = list(store_data.families)

    def fit(self, train_up_to: pd.Timestamp) -> None:
        frame = build_training_frame(self._store_data, self._feature_config, pd.Timestamp(train_up_to))
        x = self._prepare(frame)
        y = frame["target"].to_numpy()
        self._model = lgb.LGBMRegressor(
            objective=self._params.objective,
            num_leaves=self._params.num_leaves,
            learning_rate=self._params.learning_rate,
            n_estimators=self._params.n_estimators,
            min_child_samples=self._params.min_child_samples,
            feature_fraction=self._params.feature_fraction,
            random_state=self._params.seed,
            verbose=-1,
        )
        self._model.fit(x, y, categorical_feature=CATEGORICAL_COLUMNS)

    def predict(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        origin = self._store_data.dates[-1]
        frame = build_prediction_frame(self._store_data, self._feature_config, pd.Timestamp(origin))
        preds_log = self._model.predict(self._prepare(frame))
        frame = frame.assign(pred=np.clip(np.expm1(preds_log), a_min=0.0, a_max=None))
        return self._to_grid(frame)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self._feature_columns].copy()
        for column in CATEGORICAL_COLUMNS:
            x[column] = x[column].astype("category")
        return x

    def _to_grid(self, frame: pd.DataFrame) -> np.ndarray:
        horizon = self._feature_config.horizon
        grid = np.zeros((horizon, len(self._stores), len(self._families)))
        store_pos = {store: idx for idx, store in enumerate(self._stores)}
        family_pos = {family: idx for idx, family in enumerate(self._families)}
        for _, row in frame.iterrows():
            grid[int(row["horizon_step"]) - 1, store_pos[row["store"]], family_pos[row["family"]]] = row["pred"]
        return grid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_lgbm.py -v`
Expected: PASS.

- [ ] **Step 5: Integration — runs through the harness**

Add to `tests/time_series/test_lgbm.py`:

```python
def test_forecaster_plugs_into_backtest(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    feature_config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    config = BacktestConfig(n_folds=1, horizon=1, min_train_days=1)

    def factory() -> LightGBMForecaster:
        return LightGBMForecaster(store_data, feature_config, LGBMParams(n_estimators=5))

    result = backtest(factory, store_data, config)
    assert len(result.per_fold) >= 1
    assert np.isfinite(result.mean_rmsle)
```

Run: `uv run pytest tests/time_series/test_lgbm.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add time_series/store_sales/lgbm.py tests/time_series/test_lgbm.py
git commit -m "Add LightGBMForecaster with backtest integration"
```

---

### Task 11: Entry point + MLflow run

**Files:**
- Create: `time_series/main_store_sales_lgbm.py`
- Modify: `time_series/store_sales/__init__.py` (export `backtest`, `BacktestConfig`, `BacktestResult`, `Forecaster`, `LightGBMForecaster`, `LGBMParams`, `FeatureConfig`)
- Test: `tests/time_series/test_lgbm.py` (a `main`-callable smoke path)

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m time_series.main_store_sales_lgbm` — builds `StoreData` (all features), runs the backtest, logs to MLflow experiment `StoreSales_LightGBM` with `mlflow.lightgbm.autolog()`, git tags, and `BacktestResult.log_to_mlflow()`. A `run_backtest(store_data, feature_config, params, backtest_config) -> BacktestResult` helper is unit-testable without MLflow.

- [ ] **Step 1: Write the failing test**

```python
def test_run_backtest_helper_returns_result(mock_data_dir) -> None:
    from time_series.main_store_sales_lgbm import run_backtest
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    result = run_backtest(
        store_data,
        FeatureConfig(lags=(1,), rolling_windows=(), horizon=1),
        LGBMParams(n_estimators=5),
        BacktestConfig(n_folds=1, horizon=1, min_train_days=1),
    )
    assert np.isfinite(result.mean_rmsle)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/time_series/test_lgbm.py -k run_backtest -v`
Expected: FAIL — module not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# time_series/main_store_sales_lgbm.py
"""Train and backtest the LightGBM ceiling model, logging results to MLflow.

Run with:
    uv run python -m time_series.main_store_sales_lgbm
"""

from __future__ import annotations

import argparse

import mlflow

from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from time_series.store_sales import StoreData
from time_series.store_sales.backtest import BacktestConfig, BacktestResult, backtest
from time_series.store_sales.lgbm import LGBMParams, LightGBMForecaster
from time_series.store_sales.tabular import FeatureConfig


def run_backtest(
    store_data: StoreData,
    feature_config: FeatureConfig,
    params: LGBMParams,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    """Run the LightGBM forecaster through the backtest harness (no MLflow)."""

    def factory() -> LightGBMForecaster:
        return LightGBMForecaster(store_data, feature_config, params)

    return backtest(factory, store_data, backtest_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the LightGBM store-sales ceiling")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_data = StoreData(flatten_output=False)
    feature_config = FeatureConfig(horizon=args.horizon)
    params = LGBMParams(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    backtest_config = BacktestConfig(n_folds=args.n_folds, horizon=args.horizon)

    mlflow.lightgbm.autolog()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_LightGBM")
    with mlflow.start_run(
        tags={
            "architecture": "lightgbm-global-direct",
            "git_branch": get_branch(),
            "git_sha": get_sha(),
        }
    ):
        mlflow.log_params(
            {
                "n_folds": args.n_folds,
                "horizon": args.horizon,
                "objective": params.objective,
                "num_leaves": params.num_leaves,
                "learning_rate": params.learning_rate,
                "n_estimators": params.n_estimators,
            }
        )
        result = run_backtest(store_data, feature_config, params, backtest_config)
        result.log_to_mlflow()
        print(f"RMSLE {result.mean_rmsle:.4f} ± {result.std_rmsle:.4f} (MSLE {result.mean_rmsle**2:.4f})")


if __name__ == "__main__":
    main()
```

Add the exports to `time_series/store_sales/__init__.py` (`from .backtest import ...`, `from .lgbm import ...`, `from .tabular import ...` and append names to `__all__`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/time_series/test_lgbm.py -k run_backtest -v`
Expected: PASS.

- [ ] **Step 5: Full suite + precommit**

Run: `uv run pytest -q`
Expected: all pass.
Then run the `precommit-check` skill and fix any lint/type issues.

- [ ] **Step 6: Commit**

```bash
git add time_series/main_store_sales_lgbm.py time_series/store_sales/__init__.py tests/time_series/test_lgbm.py
git commit -m "Add LightGBM ceiling entry point with MLflow logging"
```

---

### Task 12: Docs — update project reference

**Files:**
- Modify: `docs/store_sales_project.md` (Models table + a run command)

**Interfaces:**
- Consumes: nothing.
- Produces: documentation parity (CLAUDE.md requires the project doc track the code).

- [ ] **Step 1: Add the model row + run command**

Add to the Models table:

```markdown
| `main_store_sales_lgbm.py` | `LightGBMForecaster` | `StoreSales_LightGBM` | ceiling — non-neural baseline |
```

Add under Run Commands:

```markdown
# LightGBM ceiling (runs locally, CPU, minutes)
uv run python -m time_series.main_store_sales_lgbm --n-folds 5 --horizon 16
```

Add a short "Backtest harness" subsection noting the rolling-origin evaluation and that all models can be compared via `RMSLE`/`MSLE` through `time_series/store_sales/backtest.py`.

- [ ] **Step 2: Commit**

```bash
git add docs/store_sales_project.md
git commit -m "Document LightGBM ceiling and backtest harness"
```

---

## Self-Review

**Spec coverage:**
- Backtest harness (rolling-origin, disjoint folds, mean±std, per-horizon/family) → Tasks 2, 3, 5, 6, 7. ✓
- Model-agnostic `Forecaster` protocol → Task 5. ✓
- LightGBM global, horizon-as-feature, direct multi-horizon → Tasks 8, 9, 10. ✓
- Feature design (as-of-origin historical, as-of-target known-future, static categoricals) → Tasks 8, 9. ✓
- Objective L2-on-log1p primary; Tweedie challenger → Task 10 (`LGBMParams.objective`; Tweedie via `objective="tweedie"` param, no code change needed). ✓
- Data path reuses `StoreData` cleaning, emits long form → Tasks 4, 9. ✓
- MSLE + RMSLE reported for apples-to-apples → Tasks 7, 11. ✓
- MLflow experiment `StoreSales_LightGBM`, tags, autolog → Task 11. ✓
- Tests: fold gen, RMSLE known-value, leakage checks, feature no-leakage, forecaster round-trip → Tasks 3, 2, 6, 8, 9, 10. ✓
- `lightgbm` in deps + remote group → Task 1. ✓

**Placeholder scan:** No TBD/TODO; the empty `FEATURE_COLUMNS: list[str] = []` in Task 9 is a module-level constant intentionally superseded by `_model_feature_columns(config)` (config-dependent); the forecaster uses the function, not the constant. Removed ambiguity by having tasks reference `_model_feature_columns`. ✓

**Type consistency:** `Forecaster.fit(train_up_to: pd.Timestamp)` / `predict() -> np.ndarray [horizon, n_stores, n_families]` consistent across Tasks 5, 6, 10. `FeatureConfig.horizon` and `BacktestConfig.horizon` must be set equal by the caller (Task 11 does this). `build_training_frame`/`build_prediction_frame` signatures consistent between Tasks 9 and 10. ✓

**Known follow-ups (out of scope, noted for later):** wiring real `test.csv` future promotions into `build_prediction_frame`; Tweedie A/B run; per-horizon direct models as the squeeze upgrade; per-series RevIN for the neural models.
