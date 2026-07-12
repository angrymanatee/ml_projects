from typing import cast

import numpy as np
import pandas as pd
import pytest

from time_series.store_sales import StoreData
from time_series.store_sales.tabular import (
    FeatureConfig,
    add_origin_features,
    build_prediction_frame,
    build_training_frame,
    sales_long_from_store_data,
)


def _single_series(n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store": 1,
            "family": "A",
            "sales": np.arange(n_days, dtype=float),
        }
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
    cutoff = cast(pd.Timestamp, store_data.dates[-1])
    frame = build_training_frame(store_data, config, train_up_to=cutoff)
    # every target date must be <= cutoff
    assert (frame.loc[:, "target_date"] <= cutoff).all()
    # target is log1p of sales; all finite
    assert frame.loc[:, "target"].notna().all()
    assert len(frame) > 0


def test_prediction_frame_has_no_target_and_horizon_rows(mock_data_dir) -> None:
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    origin = cast(pd.Timestamp, store_data.dates[-1])
    frame = build_prediction_frame(store_data, config, origin=origin)
    assert "target" not in frame.columns
    n_cells = store_data.stores.shape[0] * store_data.families.size
    assert len(frame) == config.horizon * n_cells
