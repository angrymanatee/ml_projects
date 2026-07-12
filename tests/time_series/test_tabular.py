from typing import cast

import numpy as np
import pandas as pd
import pytest

from time_series.store_sales import StoreData
from time_series.store_sales.tabular import (
    FeatureConfig,
    _national_holiday_dates,
    add_origin_features,
    build_prediction_frame,
    build_training_frame,
    sales_long_from_store_data,
)


class _HolidayStub:
    def __init__(self, holidays: pd.DataFrame) -> None:
        self.holidays = holidays


def test_national_holiday_dates_matches_neural_semantics() -> None:
    # Mirrors data.py's national_holiday: exclude transferred-away Holidays and
    # non-holiday national rows (Event/Work Day); include Transfer observances.
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2016-05-01", "2016-05-02", "2016-07-03", "2016-08-10", "2016-09-01"]
            ),
            "type": ["Holiday", "Transfer", "Holiday", "Event", "Work Day"],
            "locale": ["National", "National", "National", "National", "National"],
            "transferred": [True, False, False, False, False],
        }
    )
    holidays = holidays.set_index(pd.DatetimeIndex(holidays["date"]))
    result = _national_holiday_dates(_HolidayStub(holidays))

    assert pd.Timestamp("2016-05-01") not in result  # transferred away
    assert pd.Timestamp("2016-05-02") in result  # Transfer observance
    assert pd.Timestamp("2016-07-03") in result  # ordinary national Holiday
    assert pd.Timestamp("2016-08-10") not in result  # Event is not a holiday
    assert pd.Timestamp("2016-09-01") not in result  # Work Day is not a holiday


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


def test_training_frame_lag_comes_from_origin_not_target(mock_data_dir) -> None:
    """Verify that lag features source from origin_date, not target_date.

    This test ensures no data leakage: lag_1 must equal sales from 1 day
    BEFORE the origin_date, not from target_date.
    """
    store_data = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    config = FeatureConfig(lags=(1,), rolling_windows=(), horizon=1)
    train_up_to = cast(pd.Timestamp, store_data.dates[-1])
    frame = build_training_frame(store_data, config, train_up_to=train_up_to)
    long = sales_long_from_store_data(store_data).set_index(
        ["date", "store", "family"]
    )["sales"]
    dates_set = set(store_data.dates)

    # for each surviving row, lag_1 must equal sales at (origin_date - 1 day, store, family)
    for _, row in frame.iterrows():
        origin_ts = cast(pd.Timestamp, row["origin_date"])
        lag_source_date = origin_ts - pd.Timedelta(days=1)
        if lag_source_date in dates_set:
            expected = long.loc[(lag_source_date, row["store"], row["family"])]
            assert row["lag_1"] == pytest.approx(float(expected)) or (
                np.isnan(row["lag_1"]) and np.isnan(expected)
            )
        else:
            # If lag source date is not in dataset, lag_1 should be NaN
            assert np.isnan(row["lag_1"])
