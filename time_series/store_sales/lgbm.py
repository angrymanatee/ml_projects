"""LightGBM global direct-multi-horizon forecaster implementing the Forecaster protocol."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from time_series.store_sales.tabular import (
    CATEGORICAL_COLUMNS,
    FeatureConfig,
    _model_feature_columns,
    build_prediction_frame,
    build_training_frame,
)


@dataclass
class LGBMParams:
    """Hyperparameters for the underlying `lgb.LGBMRegressor`.

    `feature_fraction` is the public knob name (matches LightGBM's native
    parameter naming) but is wired to the sklearn API's `colsample_bytree`
    internally — passing both `feature_fraction=` and `colsample_bytree=`
    to LGBMRegressor triggers an alias-conflict warning.
    """

    objective: str = "regression"  # L2 on log1p target == RMSLE
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 100
    feature_fraction: float = 0.8
    seed: int = 0


class LightGBMForecaster:
    """One global LightGBM model over all series and horizons (horizon as a feature).

    Trains on log1p(sales) with L2 (directly targeting RMSLE); predictions are
    expm1'd and clipped at 0. Store/family output axes match StoreData ordering
    (stores ascending by store_nbr, families = store_data.families).
    """

    def __init__(
        self, store_data, feature_config: FeatureConfig, params: LGBMParams
    ) -> None:
        self._store_data = store_data
        self._feature_config = feature_config
        self._params = params
        self._model: lgb.LGBMRegressor | None = None
        self._fit_cutoff: pd.Timestamp | None = None
        self._feature_columns = _model_feature_columns(feature_config)
        self._stores = sorted(store_data.stores.index)
        self._families = list(store_data.families)

    def fit(self, train_up_to: pd.Timestamp) -> None:
        """Build the training frame up to `train_up_to` and fit one global model."""
        cutoff: pd.Timestamp = pd.Timestamp(train_up_to)  # type: ignore[assignment]
        self._fit_cutoff = cutoff
        frame = build_training_frame(self._store_data, self._feature_config, cutoff)
        x = self._prepare(frame)
        y = frame["target"].to_numpy()
        self._model = lgb.LGBMRegressor(
            objective=self._params.objective,
            num_leaves=self._params.num_leaves,
            learning_rate=self._params.learning_rate,
            n_estimators=self._params.n_estimators,
            min_child_samples=self._params.min_child_samples,
            colsample_bytree=self._params.feature_fraction,
            random_state=self._params.seed,
            verbose=-1,
        )
        # categorical_feature is intentionally omitted: LightGBM auto-detects
        # pandas `category`-dtype columns (set in _prepare), and passing it
        # explicitly here as well triggers a redundant-argument warning.
        self._model.fit(x, y)

    def predict(self) -> np.ndarray:
        """Predict horizon days from the fit cutoff (origin), in sales space.

        The origin is the `train_up_to` cutoff passed to `fit`, not the last
        observed date: the backtest scores each fold's predictions against the
        actuals immediately following that fold's cutoff, so predicting from
        any other origin would compare mismatched date ranges. For a true
        out-of-sample forecast, call `fit(last_observed_date)` first.
        """
        if self._model is None or self._fit_cutoff is None:
            raise RuntimeError("fit must be called before predict")
        frame = build_prediction_frame(
            self._store_data, self._feature_config, self._fit_cutoff
        )
        preds_log = np.asarray(self._model.predict(self._prepare(frame)))
        frame = frame.assign(pred=np.clip(np.expm1(preds_log), a_min=0.0, a_max=None))
        return self._to_grid(frame)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self._feature_columns].copy()
        for column in CATEGORICAL_COLUMNS:
            x[column] = x[column].astype("category")
        return x  # type: ignore[return-value]

    def _to_grid(self, frame: pd.DataFrame) -> np.ndarray:
        horizon = self._feature_config.horizon
        grid = np.zeros((horizon, len(self._stores), len(self._families)))
        store_pos = {store: idx for idx, store in enumerate(self._stores)}
        family_pos = {family: idx for idx, family in enumerate(self._families)}
        step_idx = frame["horizon_step"].to_numpy(dtype=int) - 1
        store_idx = frame["store"].map(store_pos.get).to_numpy(dtype=int)
        family_idx = frame["family"].map(family_pos.get).to_numpy(dtype=int)
        grid[step_idx, store_idx, family_idx] = frame["pred"].to_numpy()
        return grid
