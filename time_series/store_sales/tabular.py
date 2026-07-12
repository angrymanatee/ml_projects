"""Long-form tabular feature builder for the LightGBM store-sales model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS: list[str] = ["store", "family", "city", "state", "type", "cluster"]


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


def sales_long_from_store_data(store_data) -> pd.DataFrame:
    """Melt sales_tensor [T, n_stores, n_families] into long form.

    Column order in the tensor is (date, store, family) per StoreData's
    construction (stores sorted by store_nbr, families from the pivoted
    column index), so reshape(-1) with matching repeat/tile order preserves
    alignment.
    """
    dates = pd.DatetimeIndex(store_data.dates)
    stores = sorted(
        store_data.stores.index
    )  # ascending store_nbr to match sales_tensor's sort_index(axis="columns") column order
    families = list(store_data.families)
    sales = np.asarray(store_data.sales_tensor)
    return pd.DataFrame(
        {
            "date": np.repeat(dates.values, len(stores) * len(families)),
            "store": np.tile(np.repeat(stores, len(families)), len(dates)),
            "family": np.tile(families, len(dates) * len(stores)),
            "sales": sales.reshape(-1),
        }
    )


def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Deterministic day-of-{week,month,year} features for a Series of dates."""
    dt = dates.dt
    return pd.DataFrame(
        {
            "dayofweek": dt.dayofweek,
            "day": dt.day,
            "month": dt.month,
            "is_weekend": (dt.dayofweek >= 5).astype(int),
            "is_15th": (dt.day == 15).astype(int),
            "is_month_end": dt.is_month_end.astype(int),
        },
        index=dates.index,
    )


def _promotion_long(store_data) -> pd.DataFrame:
    """Long-form onpromotion keyed by (date, store, family); zeros if absent."""
    dates = pd.DatetimeIndex(store_data.dates)
    stores = sorted(
        store_data.stores.index
    )  # ascending store_nbr to match sales_tensor's sort_index(axis="columns") column order
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


def _national_holiday_dates(store_data) -> set[pd.Timestamp]:
    """Normalized dates of national holidays, matching StoreData's national_holiday.

    A date counts as a national holiday when it is a National `Holiday` that was
    not transferred away, or a National `Transfer` row (the relocated
    observance). Transferred-away originals and non-holiday national rows
    (`Event`, `Bridge`, `Work Day`, `Additional`) are excluded, so the LightGBM
    is_holiday flag agrees with the neural models' national_holiday channel
    (see data.py _setup_holiday_tensor). Filtering on locale alone would flag
    transferred-away days and compensatory work days as holidays.
    """
    holidays = store_data.holidays
    is_national = holidays["locale"] == "National"
    active = (
        (holidays["type"] == "Holiday") & is_national & ~holidays["transferred"]
    ) | ((holidays["type"] == "Transfer") & is_national)
    return set(pd.to_datetime(holidays.loc[active, "date"]).dt.normalize())


def _feature_frame(
    store_data, config: FeatureConfig, origins_and_horizons: pd.DataFrame
) -> pd.DataFrame:
    """Core builder shared by training and prediction frames.

    origins_and_horizons: columns [store, family, origin_date, horizon_step,
    target_date]. Joins as-of-origin lag/rolling features (from origin_date,
    strictly <= target_date - horizon_step days) and as-of-target
    known-future/calendar/static features (from target_date). No column from
    target_date's sales ever enters the origin-side join, so lag features
    never see the row's own target.
    """
    sales_long = sales_long_from_store_data(store_data)
    origin_features = add_origin_features(sales_long, config)
    feature_cols = [
        c for c in origin_features.columns if c.startswith(("lag_", "roll_"))
    ]

    origin_cols: list[str] = ["store", "family", "date", *feature_cols]
    origin_side = origin_features.loc[:, origin_cols].rename(
        columns={"date": "origin_date"}
    )
    frame = origins_and_horizons.merge(
        origin_side,
        on=["store", "family", "origin_date"],
        how="left",
    )

    # known-future: promotion at target date
    promo = _promotion_long(store_data).rename(columns={"date": "target_date"})
    frame = frame.merge(promo, on=["store", "family", "target_date"], how="left")

    # calendar at target date
    frame = frame.reset_index(drop=True)
    frame = pd.concat([frame, _calendar_features(frame.loc[:, "target_date"])], axis=1)

    # national holiday flag at target date
    national = _national_holiday_dates(store_data)
    frame["is_holiday"] = frame["target_date"].dt.normalize().isin(national).astype(int)

    # static store categoricals
    static = store_data.stores.reset_index().rename(columns={"store_nbr": "store"})
    frame = frame.merge(
        static[["store", "city", "state", "type", "cluster"]], on="store", how="left"
    )

    frame["horizon_step"] = frame["horizon_step"].astype(int)
    return frame


def _model_feature_columns(config: FeatureConfig) -> list[str]:
    """Ordered list of model input columns for a given FeatureConfig."""
    lag_cols = [f"lag_{k}" for k in config.lags]
    roll_cols = [
        f"roll_{w}_{stat}"
        for w in config.rolling_windows
        for stat in ("mean", "std", "min", "max")
    ]
    calendar = ["dayofweek", "day", "month", "is_weekend", "is_15th", "is_month_end"]
    known_future = ["onpromotion", "is_holiday", "horizon_step"]
    return [*lag_cols, *roll_cols, *calendar, *known_future, *CATEGORICAL_COLUMNS]


def build_training_frame(
    store_data, config: FeatureConfig, train_up_to: pd.Timestamp
) -> pd.DataFrame:
    """One row per (target_date t <= cutoff, store, family, horizon h).

    origin = t - h days; rows whose origin falls outside the observed date
    range are dropped (no lag features would be computable). target =
    log1p(sales[t]). Rows missing the deepest configured lag are dropped, since
    those rows have incomplete history and would otherwise silently train on NaN
    lag features.
    """
    dates = pd.DatetimeIndex(store_data.dates)
    train_dates = dates[dates <= train_up_to]
    sales_long = sales_long_from_store_data(store_data).set_index(
        ["date", "store", "family"]
    )["sales"]

    stores = list(store_data.stores.index)
    families = list(store_data.families)
    date_set = set(dates)

    rows = []
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
                            "target": float(
                                np.log1p(sales_long.loc[(target_date, store, family)])
                            ),
                        }
                    )
    origins = pd.DataFrame(rows)
    frame = _feature_frame(store_data, config, origins)
    if config.lags:
        frame = frame.dropna(subset=[f"lag_{max(config.lags)}"])
    return frame


def build_prediction_frame(
    store_data, config: FeatureConfig, origin: pd.Timestamp
) -> pd.DataFrame:
    """Rows for a single origin at h = 1..horizon; target days origin+1..origin+horizon.

    No target column: these target dates are unobserved (future).
    """
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
