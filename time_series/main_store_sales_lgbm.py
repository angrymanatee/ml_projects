"""Train and backtest the LightGBM ceiling model, logging results to MLflow.

Run with:
    uv run python -m time_series.main_store_sales_lgbm
"""

from __future__ import annotations

# common.openmp_guard sets the OMP env vars that prevent a torch/lightgbm libomp
# deadlock; it must run before StoreData (torch) or lightgbm load, so it comes
# first and isort must not reorder it. See common/openmp_guard.py.
# isort: off
import common.openmp_guard  # noqa: F401
# isort: on

import argparse  # noqa: E402

import mlflow.lightgbm  # noqa: E402

import mlflow  # noqa: E402
from common.git import get_branch, get_sha  # noqa: E402
from common.model_registry import TRACKING_URI  # noqa: E402
from time_series.store_sales import StoreData  # noqa: E402
from time_series.store_sales.backtest import (  # noqa: E402
    BacktestConfig,
    BacktestResult,
    backtest,
)
from time_series.store_sales.lgbm import LGBMParams, LightGBMForecaster  # noqa: E402
from time_series.store_sales.tabular import FeatureConfig  # noqa: E402


def run_backtest(
    store_data: StoreData,
    feature_config: FeatureConfig,
    params: LGBMParams,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    """Run the LightGBM forecaster through the backtest harness (no MLflow)."""

    def factory() -> LightGBMForecaster:
        return LightGBMForecaster(store_data, feature_config, params)

    return backtest(factory, store_data, backtest_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the LightGBM store-sales ceiling"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_data = StoreData()
    feature_config = FeatureConfig(horizon=args.horizon)
    params = LGBMParams(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    backtest_config = BacktestConfig(n_folds=args.n_folds, horizon=args.horizon)

    mlflow.lightgbm.autolog()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_LightGBM")
    with mlflow.start_run(
        tags={
            "architecture": "lightgbm-global-direct",
            "git_branch": get_branch(),
            "git_sha": get_sha(),
        }
    ):
        mlflow.log_params(
            {
                "n_folds": args.n_folds,
                "horizon": args.horizon,
                "objective": params.objective,
                "num_leaves": params.num_leaves,
                "learning_rate": params.learning_rate,
                "n_estimators": params.n_estimators,
            }
        )
        result = run_backtest(store_data, feature_config, params, backtest_config)
        result.log_to_mlflow()
        print(
            f"RMSLE {result.mean_rmsle:.4f} ± {result.std_rmsle:.4f} "
            f"(MSLE {result.mean_rmsle**2:.4f})"
        )


if __name__ == "__main__":
    main()
