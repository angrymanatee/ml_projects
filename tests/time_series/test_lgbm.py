"""Tests for LightGBMForecaster: shape/nonneg, determinism, and backtest integration.

See the rootdir conftest.py for why `lightgbm` must be imported before `torch`
in this process (a segfault on this machine otherwise).
"""

import numpy as np

from time_series.store_sales import StoreData
from time_series.store_sales.backtest import BacktestConfig, backtest
from time_series.store_sales.lgbm import LGBMParams, LightGBMForecaster
from time_series.store_sales.tabular import FeatureConfig


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
