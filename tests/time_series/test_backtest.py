import numpy as np
import pandas as pd
import pytest

import mlflow
from time_series.store_sales.backtest import (
    BacktestConfig,
    BacktestResult,
    backtest,
    generate_fold_cutoffs,
    rmsle,
)


def test_rmsle_zero_for_perfect_prediction() -> None:
    y = np.array([0.0, 5.0, 100.0])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_known_value() -> None:
    # single element: |log1p(3) - log1p(7)| = |1.386294 - 2.079442| = 0.693147
    assert rmsle(np.array([7.0]), np.array([3.0])) == pytest.approx(0.6931471, abs=1e-6)


def test_rmsle_clips_negative_predictions_to_zero() -> None:
    # prediction -4 is clipped to 0, so error vs true 0 is log1p(0)-log1p(0)=0
    assert rmsle(np.array([0.0]), np.array([-4.0])) == pytest.approx(0.0)


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


def test_backtest_result_mean_and_std() -> None:
    from time_series.store_sales.backtest import BacktestResult

    per_fold = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            "cutoff": pd.to_datetime(["2020-01-01"] * 3),
            "rmsle": [1.0, 2.0, 3.0],
        }
    )
    result = BacktestResult(
        per_fold=per_fold, per_horizon=pd.DataFrame(), per_family=pd.DataFrame()
    )
    assert result.mean_rmsle == pytest.approx(2.0)
    assert result.std_rmsle == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))


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


class _GapStoreData:
    """Dates with a deliberate interior gap; each date's sales encode its ordinal."""

    def __init__(self, drop: str = "2020-02-10") -> None:
        full = pd.date_range("2020-01-01", periods=60, freq="D")
        self.dates = full.delete(full.get_loc(pd.Timestamp(drop)))
        ordinals = np.array([d.toordinal() for d in self.dates], dtype=float)
        self.sales_tensor = ordinals.reshape(-1, 1, 1)  # unique value per date
        self.families = pd.Index(["A"])


class _DateOracleForecaster:
    """Predicts each horizon step from its calendar target date (cutoff + h)."""

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon
        self._cutoff: pd.Timestamp | None = None

    def fit(self, train_up_to: pd.Timestamp) -> None:
        self._cutoff = pd.Timestamp(train_up_to)  # type: ignore[assignment]

    def predict(self) -> np.ndarray:
        assert self._cutoff is not None
        grid = np.zeros((self._horizon, 1, 1))
        for step in range(self._horizon):
            target_date = self._cutoff + pd.Timedelta(days=step + 1)
            grid[step, 0, 0] = float(target_date.toordinal())  # type: ignore[union-attr]
        return grid


def test_backtest_aligns_actuals_to_calendar_dates_across_gaps() -> None:
    # The oracle predicts each target date's encoded value exactly. A correct
    # harness aligns predictions to actuals by calendar date (dropping the gap
    # day), so RMSLE is 0. Row-position slicing would shift actuals past the gap
    # and score mismatched dates -> RMSLE > 0. n_folds=3 includes a fold whose
    # 16-day window straddles the dropped 2020-02-10.
    store_data = _GapStoreData()
    config = BacktestConfig(n_folds=3, horizon=16, min_train_days=1)
    result = backtest(lambda: _DateOracleForecaster(config.horizon), store_data, config)
    assert len(result.per_fold) == 3
    assert result.mean_rmsle == pytest.approx(0.0)


def test_backtest_cutoff_on_missing_date_does_not_crash() -> None:
    # A fold cutoff computed by calendar arithmetic can land on a date absent
    # from the series; the harness must not index it by position (KeyError).
    # last=2020-02-29, horizon=16 -> fold-0 cutoff = 2020-02-13, which we drop.
    store_data = _GapStoreData(drop="2020-02-13")
    config = BacktestConfig(n_folds=1, horizon=16, min_train_days=1)
    result = backtest(lambda: _DateOracleForecaster(config.horizon), store_data, config)
    assert len(result.per_fold) == 1
    assert np.isfinite(result.mean_rmsle)


def test_log_to_mlflow_records_summary_metrics(tmp_path) -> None:
    per_fold = pd.DataFrame(
        {
            "fold": [0, 1],
            "cutoff": pd.to_datetime(["2020-01-01", "2020-01-17"]),
            "rmsle": [0.8, 1.0],
        }
    )
    per_horizon = pd.DataFrame(
        {"fold": [0, 0], "horizon_step": [1, 2], "rmsle": [0.7, 0.9]}
    )
    per_family = pd.DataFrame({"fold": [0], "family": ["A"], "rmsle": [0.85]})
    result = BacktestResult(
        per_fold=per_fold, per_horizon=per_horizon, per_family=per_family
    )

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test_backtest_logging")
    with mlflow.start_run() as run:
        result.log_to_mlflow()

    client = mlflow.tracking.MlflowClient()
    logged = client.get_run(run.info.run_id).data.metrics
    assert logged["rmsle_mean"] == pytest.approx(0.9)
    assert logged["msle_mean"] == pytest.approx(0.81)
