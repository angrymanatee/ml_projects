"""Model-agnostic rolling-origin backtest harness for store sales forecasting."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

import mlflow


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared logarithmic error, the competition metric.

    Predictions are clipped at 0 before scoring (sales are non-negative and
    log1p of a negative value is undefined).
    """
    clipped = np.clip(y_pred, a_min=0.0, a_max=None)
    diff = np.log1p(clipped) - np.log1p(y_true)
    return float(np.sqrt(np.mean(diff**2)))


@dataclass
class BacktestConfig:
    """Configuration for rolling-origin backtesting.

    n_folds: number of disjoint predict blocks.
    horizon: forecast length per fold (competition uses 16 days).
    min_train_days: a fold is dropped unless at least this many distinct
        dates are available for training (dates <= cutoff, including the cutoff
        date itself, since the cutoff date's data is used for training).
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
    with fewer than min_train_days dates up to and including the cutoff
    (<=, since cutoff date's data is used for training) are dropped.
    """
    ordered = pd.DatetimeIndex(dates).sort_values().unique()
    last = ordered[-1]
    cutoffs: list[pd.Timestamp] = []
    for fold in range(config.n_folds):
        cutoff_ts: pd.Timestamp = last - pd.Timedelta(days=config.horizon * (fold + 1))
        n_before = int((ordered <= cutoff_ts).sum())
        if n_before < config.min_train_days:
            continue
        cutoffs.append(cutoff_ts)
    return sorted(cutoffs)


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

    def log_to_mlflow(self) -> None:
        """Log aggregate/per-horizon/per-family metrics to the active MLflow run.

        Logs msle_mean (= rmsle_mean**2) so these runs are directly comparable
        to the transformer experiments, which log MSLE.
        """
        mlflow.log_metric("rmsle_mean", self.mean_rmsle)
        mlflow.log_metric("rmsle_std", self.std_rmsle)
        mlflow.log_metric("msle_mean", self.mean_rmsle**2)

        by_step = self.per_horizon.groupby("horizon_step")["rmsle"].mean()
        for step, value in by_step.items():
            mlflow.log_metric(
                "rmsle_by_horizon",
                float(value),
                step=int(step),  # type: ignore[arg-type]
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "per_family_rmsle.csv")
            self.per_family.groupby("family")["rmsle"].mean().to_csv(path)
            mlflow.log_artifact(path)


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
        forecaster.fit(pd.Timestamp(cutoff))  # type: ignore[arg-type]
        prediction = np.asarray(forecaster.predict())  # [horizon, n_stores, n_families]
        if prediction.shape[0] != config.horizon:
            raise ValueError(
                f"forecaster returned {prediction.shape[0]} horizon steps but "
                f"BacktestConfig.horizon is {config.horizon}; the forecaster's "
                "feature horizon must equal the backtest horizon"
            )

        start = date_to_index[cutoff] + 1
        actual = np.asarray(store_data.sales_tensor)[start : start + config.horizon]

        fold_rows.append(
            {"fold": fold, "cutoff": cutoff, "rmsle": rmsle(actual, prediction)}
        )
        for step in range(config.horizon):
            horizon_rows.append(
                {
                    "fold": fold,
                    "horizon_step": step + 1,
                    "rmsle": rmsle(actual[step], prediction[step]),
                }
            )
        for family_index, family in enumerate(families):
            horizon_slice_actual = actual[:, :, family_index]
            horizon_slice_pred = prediction[:, :, family_index]
            family_rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "rmsle": rmsle(horizon_slice_actual, horizon_slice_pred),
                }
            )

    return BacktestResult(
        per_fold=pd.DataFrame(fold_rows),
        per_horizon=pd.DataFrame(horizon_rows),
        per_family=pd.DataFrame(family_rows),
    )
