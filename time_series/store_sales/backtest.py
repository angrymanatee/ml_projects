"""Model-agnostic rolling-origin backtest harness for store sales forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
