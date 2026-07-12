"""Tests for LightGBMForecaster: shape/nonneg, determinism, and backtest integration.

See the rootdir conftest.py for why `lightgbm` must be imported before `torch`
in this process (a segfault on this machine otherwise).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from time_series.store_sales import StoreData
from time_series.store_sales import lgbm as lgbm_module
from time_series.store_sales.backtest import BacktestConfig, backtest
from time_series.store_sales.lgbm import LGBMParams, LightGBMForecaster
from time_series.store_sales.tabular import FeatureConfig


@pytest.fixture(scope="module")
def long_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ~40-day single-series dataset, enough for a multi-fold backtest."""
    directory = tmp_path_factory.mktemp("store_sales_long")
    dates = pd.date_range("2013-01-01", periods=40, freq="D")
    rows = [
        {
            "date": date,
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": 10.0 + day_index + 3.0 * (day_index % 7),  # non-constant
            "onpromotion": 0,
        }
        for day_index, date in enumerate(dates)
    ]
    train = pd.DataFrame(rows)
    train.to_csv(directory / "train.csv", index=False)
    train.to_csv(directory / "test.csv", index=False)

    pd.DataFrame({"id": [0], "sales": [0.0]}).to_csv(
        directory / "sample_submission.csv", index=False
    )
    pd.DataFrame(
        {"city": ["Quito"], "state": ["Pichincha"], "type": ["D"], "cluster": [1]},
        index=pd.Index([1], name="store_nbr"),
    ).reset_index().to_csv(directory / "stores.csv", index=False)
    pd.DataFrame(
        {"date": dates.astype(str), "dcoilwtico": 90.0 + np.arange(len(dates)) * 0.1}
    ).to_csv(directory / "oil.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2013-01-01"],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New Year"],
            "transferred": [False],
        }
    ).to_csv(directory / "holidays_events.csv", index=False)
    return directory


def test_forecaster_predict_shape_and_nonneg(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    feature_config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    forecaster = LightGBMForecaster(
        store_data, feature_config, LGBMParams(n_estimators=5)
    )
    forecaster.fit(store_data.dates[-1])  # type: ignore[arg-type]
    prediction = forecaster.predict()
    n_stores = store_data.stores.shape[0]
    n_families = store_data.families.size
    assert prediction.shape == (1, n_stores, n_families)
    assert (prediction >= 0).all()


def test_forecaster_is_deterministic(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    feature_config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)

    def make() -> LightGBMForecaster:
        return LightGBMForecaster(
            store_data, feature_config, LGBMParams(n_estimators=5, seed=0)
        )

    a, b = make(), make()
    a.fit(store_data.dates[-1])  # type: ignore[arg-type]
    b.fit(store_data.dates[-1])  # type: ignore[arg-type]
    np.testing.assert_allclose(a.predict(), b.predict())


def test_forecaster_plugs_into_backtest(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    # lags=() (not (1,)): with only 3 mock dates, backtest's only feasible fold
    # cutoff (2013-01-02, one horizon day short of the last date) has no rows
    # with a fully-populated lag_1 (its own origin lacks a day-before), so a
    # dropna(subset=["lag_1"]) in build_training_frame would leave zero rows.
    feature_config = FeatureConfig(lags=(), rolling_windows=(), horizon=1)
    config = BacktestConfig(n_folds=1, horizon=1, min_train_days=1)

    def factory() -> LightGBMForecaster:
        return LightGBMForecaster(
            store_data, feature_config, LGBMParams(n_estimators=5)
        )

    result = backtest(factory, store_data, config)
    assert len(result.per_fold) >= 1
    assert np.isfinite(result.mean_rmsle)


def test_predict_forecasts_from_fit_cutoff(mock_data_dir, monkeypatch) -> None:
    # Regression guard: predict() must forecast the block after the fit cutoff,
    # not after the last observed date. Otherwise every fold predicts the same
    # trailing future window and the backtest scores mismatched date ranges.
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    feature_config = FeatureConfig(lags=(), rolling_windows=(), horizon=1)
    forecaster = LightGBMForecaster(
        store_data, feature_config, LGBMParams(n_estimators=5)
    )
    # mock_data_dir spans 2013-01-01..03; cutoff is the middle date, != last date
    cutoff = pd.Timestamp("2013-01-02")
    forecaster.fit(cutoff)  # type: ignore[arg-type]

    captured: dict[str, pd.Timestamp] = {}
    original = lgbm_module.build_prediction_frame

    def spy(store_data, config, origin):
        captured["origin"] = origin
        return original(store_data, config, origin)

    monkeypatch.setattr(lgbm_module, "build_prediction_frame", spy)
    forecaster.predict()

    assert captured["origin"] == cutoff
    assert captured["origin"] != pd.Timestamp("2013-01-03")


def test_multi_fold_backtest_runs_disjoint_folds(long_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=3, data_dir=long_data_dir)
    feature_config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=3)
    config = BacktestConfig(n_folds=2, horizon=3, min_train_days=1)

    def factory() -> LightGBMForecaster:
        return LightGBMForecaster(
            store_data, feature_config, LGBMParams(n_estimators=5)
        )

    result = backtest(factory, store_data, config)
    assert len(result.per_fold) == 2
    assert result.per_fold["cutoff"].nunique() == 2  # folds march to distinct cutoffs
    assert len(result.per_horizon) == 2 * 3
    assert np.isfinite(result.mean_rmsle)


def test_run_backtest_helper_returns_result(mock_data_dir) -> None:
    from time_series.main_store_sales_lgbm import run_backtest

    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    result = run_backtest(
        store_data,
        FeatureConfig(lags=(), rolling_windows=(), horizon=1),
        LGBMParams(n_estimators=5),
        BacktestConfig(n_folds=1, horizon=1, min_train_days=1),
    )
    assert np.isfinite(result.mean_rmsle)
