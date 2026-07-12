"""Model-agnostic rolling-origin backtest harness for store sales forecasting."""

from __future__ import annotations

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared logarithmic error, the competition metric.

    Predictions are clipped at 0 before scoring (sales are non-negative and
    log1p of a negative value is undefined).
    """
    clipped = np.clip(y_pred, a_min=0.0, a_max=None)
    diff = np.log1p(clipped) - np.log1p(y_true)
    return float(np.sqrt(np.mean(diff**2)))
