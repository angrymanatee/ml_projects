"""Optuna hyperparameter search for the encoder-only Store Sales Transformer.

Searches over lr, d_model, nhead, num_layers, batch_size, and pooling_mode.
Each trial runs as a nested MLflow run under a single parent study run, so
the full search is one collapsible entry in the MLflow UI.

Run with:
    uv run python -m time_series.tune_store_sales_encoder_only
    uv run python -m time_series.tune_store_sales_encoder_only --n-trials 40 --epochs-per-trial 30
"""

import argparse

import optuna
import torch

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from time_series.main_store_sales_encoder_only import train_and_eval
from time_series.store_sales import (
    HOLIDAY_FEATURE_COLS,
    STORE_FEATURE_COLS,
    PoolingMode,
    StoreData,
    get_device,
)


def build_config(trial: optuna.Trial) -> dict:
    """Sample a hyperparameter configuration from the Optuna trial.

    Core architecture (pooling=all, nhead=2, batch=64) is fixed from the prior
    sweep. Free parameters: lr, num_layers, d_model_per_head, dim_feedforward.

    Returns:
        Dict with keys: lr, d_model, nhead, num_layers, batch_size, pooling_mode, dim_feedforward.
    """
    d_model_per_head = trial.suggest_categorical("d_model_per_head", [32, 64])
    return {
        "lr": trial.suggest_float("lr", 1e-3, 2e-3, log=True),
        "d_model": 2 * d_model_per_head,
        "nhead": 2,
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "batch_size": 64,
        "pooling_mode": PoolingMode.ALL,
        "dim_feedforward": trial.suggest_categorical(
            "dim_feedforward", [64, 128, 256, 512]
        ),
    }


def objective(
    trial: optuna.Trial,
    store_data: StoreData,
    device: torch.device,
    epochs_per_trial: int,
    split: float,
) -> float:
    """Optuna objective: train one configuration and return best val MSLE.

    Starts a nested MLflow run so each trial appears as a child of the parent
    study run. Must be called with a parent run already active.

    Returns:
        Best validation MSLE across all epochs of the trial.
    """
    config = build_config(trial)
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(
            {
                "trial_number": trial.number,
                "lr": config["lr"],
                "d_model": config["d_model"],
                "nhead": config["nhead"],
                "num_layers": config["num_layers"],
                "batch_size": config["batch_size"],
                "pooling_mode": str(config["pooling_mode"]),
                "dim_feedforward": config["dim_feedforward"],
            }
        )
        val_loss, *_ = train_and_eval(
            config=config,
            store_data=store_data,
            device=device,
            epochs=epochs_per_trial,
            split=split,
            save_checkpoints=False,
            log_metrics=True,
        )
        mlflow.log_metric("best_val_loss", val_loss)
    return val_loss


def tune(
    n_trials: int,
    epochs_per_trial: int,
    split: float,
    study_name: str,
    store_feature_cols: list[str] | None = None,
    holiday_features: list[str] | None = None,
    include_oil: bool = False,
    include_onpromotion: bool = False,
) -> optuna.Study:
    """Run the full Optuna study and log results to MLflow.

    Loads StoreData once and reuses it across all trials. Creates a minimization
    study (objective is val MSLE). After all trials, logs the best val loss and
    best trial number to the parent run.

    Args:
        n_trials: total number of Optuna trials to run.
        epochs_per_trial: training epochs per trial; lower = faster search, noisier estimates.
        split: train/val split fraction.
        study_name: name passed to optuna.create_study and used as the MLflow parent run name.
        store_feature_cols: store metadata columns to include as input features.
            Passed through to StoreData unchanged.
        holiday_features: holiday/event features to include as binary input channels.
            Passed through to StoreData unchanged.
        include_oil: if True, append oil price as an extra input channel (broadcast across stores).
            Passed through to StoreData unchanged.
        include_onpromotion: if True, include onpromotion as an additional input feature.
            Passed through to StoreData unchanged.

    Returns:
        The completed Optuna study; caller can inspect study.best_trial.params for the
        winning hyperparameter configuration.
    """
    store_data = StoreData(
        dtype=torch.float32,
        store_feature_cols=store_feature_cols,
        holiday_features=holiday_features,
        include_oil=include_oil,
        include_onpromotion=include_onpromotion,
    )
    device = get_device()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_EncoderOnly_Tune")
    mlflow.set_experiment_tags(
        {
            "dataset": "store-sales-kaggle",
            "task": "time-series-forecasting",
            "loss": "RMSLE",
            "prediction_horizon_days": str(store_data.output_lags),
        }
    )

    study = optuna.create_study(direction="minimize", study_name=study_name)

    with mlflow.start_run(
        run_name=study_name,
        tags={
            "git_branch": get_branch(),
            "git_sha": get_sha(),
            "n_trials": str(n_trials),
            "epochs_per_trial": str(epochs_per_trial),
            "store_features": (
                ",".join(store_feature_cols) if store_feature_cols else "none"
            ),
            "holiday_features": (
                ",".join(holiday_features) if holiday_features else "none"
            ),
            "include_oil": str(include_oil),
            "include_onpromotion": str(include_onpromotion),
        },
    ):
        study.optimize(
            lambda trial: objective(trial, store_data, device, epochs_per_trial, split),
            n_trials=n_trials,
        )
        if study.best_trials:
            mlflow.log_metric("best_val_loss", study.best_value)
            mlflow.log_param("best_trial", study.best_trial.number)

    return study


def _positive_int(value: str) -> int:
    int_value = int(value)
    if int_value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {int_value}")
    return int_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for StoreSalesEncoderOnly"
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=_positive_int, default=50)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--study-name", type=str, default="store_sales_encoder_only")
    parser.add_argument(
        "--store-features",
        nargs="*",
        choices=list(STORE_FEATURE_COLS),
        default=[],
        metavar="COL",
        help=(
            f"store metadata columns to append as input features "
            f"(choices: {', '.join(STORE_FEATURE_COLS)})"
        ),
    )
    parser.add_argument(
        "--holiday-features",
        nargs="*",
        choices=list(HOLIDAY_FEATURE_COLS),
        default=[],
        metavar="FEAT",
        help=(
            f"holiday/event features to append as binary input channels "
            f"(choices: {', '.join(HOLIDAY_FEATURE_COLS)})"
        ),
    )
    parser.add_argument(
        "--oil",
        action="store_true",
        default=False,
        help="append oil price as an extra input channel (broadcast across stores)",
    )
    parser.add_argument(
        "--onpromotion",
        action="store_true",
        default=False,
        help="include onpromotion as an additional input feature (default: off)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run the tuning study."""
    args = parse_args()
    tune(
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        split=args.split,
        study_name=args.study_name,
        store_feature_cols=args.store_features,
        holiday_features=args.holiday_features,
        include_oil=args.oil,
        include_onpromotion=args.onpromotion,
    )


if __name__ == "__main__":
    main()
