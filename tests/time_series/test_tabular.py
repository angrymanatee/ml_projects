import numpy as np
import pandas as pd
import pytest

from time_series.store_sales.tabular import FeatureConfig, add_origin_features


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
