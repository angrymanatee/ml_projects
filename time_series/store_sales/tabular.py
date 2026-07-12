"""Long-form tabular feature builder for the LightGBM store-sales model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureConfig:
    lags: tuple[int, ...] = (7, 14, 21, 28, 35, 42, 49, 56, 63)
    rolling_windows: tuple[int, ...] = (7, 14, 28, 56)
    horizon: int = 16


def add_origin_features(
    sales_long: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
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
