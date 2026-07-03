"""Feature ablation study for the encoder-only Store Sales Transformer via Optuna.

Evaluates each new feature group in isolation, a no-features baseline, and all features
combined — 6 configurations total. Architecture is fixed; only dataset features vary.
Uses Optuna's GridSampler so each config runs exactly once per repeat.

Each trial is logged as a nested MLflow run under a single parent study run.

Run with:
    uv run python -m time_series.tune_store_sales_features
    uv run python -m time_series.tune_store_sales_features --epochs 50 --n-repeats 3
"""

import argparse
import dataclasses

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


@dataclasses.dataclass
class FeatureConfig:
    """One feature-set configuration to evaluate."""

    name: str
    include_oil: bool = False
    include_onpromotion: bool = False
    store_feature_cols: list[str] | None = None
    holiday_features: list[str] | None = None

    def as_mlflow_params(self) -> dict:
        return {
            "feature_config": self.name,
            "include_oil": self.include_oil,
            "include_onpromotion": self.include_onpromotion,
            "store_feature_cols": (
                ",".join(self.store_feature_cols) if self.store_feature_cols else "none"
            ),
            "holiday_features": (
                ",".join(self.holiday_features) if self.holiday_features else "none"
            ),
        }


# All feature combinations to evaluate.
FEATURE_CONFIGS: list[FeatureConfig] = [
    FeatureConfig(name="baseline"),
    FeatureConfig(name="oil", include_oil=True),
    FeatureConfig(name="onpromotion", include_onpromotion=True),
    FeatureConfig(name="store_features", store_feature_cols=list(STORE_FEATURE_COLS)),
    FeatureConfig(name="holiday_features", holiday_features=list(HOLIDAY_FEATURE_COLS)),
    FeatureConfig(
        name="all_features",
        include_oil=True,
        include_onpromotion=True,
        store_feature_cols=list(STORE_FEATURE_COLS),
        holiday_features=list(HOLIDAY_FEATURE_COLS),
    ),
]

# Fixed architecture — d64_l2_ff512, confirmed winner of the 300-epoch cosine-annealed
# robustness check (mean val MSLE 1.101 vs 1.299 for the runner-up); lr from its
# matching phase-1 tuning trial (trial_6, val_loss=1.151151).
_ARCH_CONFIG: dict = {
    "lr": 0.0011852762898126566,
    "d_model": 64,
    "nhead": 2,
    "num_layers": 2,
    "batch_size": 64,
    "pooling_mode": PoolingMode.ALL,
    "dim_feedforward": 512,
}


def objective(
    trial: optuna.Trial,
    device: torch.device,
    epochs: int,
    split: float,
    arch_config: dict,
    n_repeats: int,
) -> float:
    """Optuna objective: build dataset for one feature config, train, return best val MSLE.

    Must be called inside an active mlflow.start_run() context. Starts a nested child run
    so each trial appears as a collapsible entry under the parent study run.

    Args:
        trial: Optuna trial; samples ``feature_config_idx`` and ``repeat_idx`` from the grid.
        device: torch device to train on.
        epochs: training epochs for this trial.
        split: train/val fraction.
        arch_config: model architecture hyperparameters to use for every trial.
        n_repeats: size of the repeat_idx grid dimension. GridSampler treats
            (feature_config_idx, repeat_idx) pairs as distinct grid points; without this,
            GridSampler stops the study after visiting each feature_config_idx once,
            silently ignoring any requested repeats.

    Returns:
        Best validation MSLE across all epochs.
    """
    config_idx: int = trial.suggest_categorical(
        "feature_config_idx", list(range(len(FEATURE_CONFIGS)))
    )
    trial.suggest_categorical("repeat_idx", list(range(n_repeats)))
    feature_config = FEATURE_CONFIGS[config_idx]

    store_data = StoreData(
        dtype=torch.float32,
        include_oil=feature_config.include_oil,
        include_onpromotion=feature_config.include_onpromotion,
        store_feature_cols=feature_config.store_feature_cols,
        holiday_features=feature_config.holiday_features,
    )

    run_name = f"trial_{trial.number}_{feature_config.name}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_params(
            {
                **feature_config.as_mlflow_params(),
                "trial_number": trial.number,
                "epochs": epochs,
                "split": split,
            }
        )
        val_loss, *_ = train_and_eval(
            config=arch_config,
            store_data=store_data,
            device=device,
            epochs=epochs,
            split=split,
            save_checkpoints=False,
            log_metrics=True,
        )
        mlflow.log_metric("best_val_loss", val_loss)

    return val_loss


def run_study(
    epochs: int,
    split: float,
    n_repeats: int,
    study_name: str,
    arch_config: dict | None = None,
) -> None:
    """Run the feature ablation Optuna study and log everything to MLflow.

    Uses GridSampler to enumerate all FEATURE_CONFIGS exactly ``n_repeats`` times.
    Each repeat visits configs in the same order. After all trials, logs a summary
    of best val loss per feature config to the parent run.

    Args:
        epochs: training epochs per trial.
        split: train/val split fraction.
        n_repeats: how many times to run each feature config.
        study_name: Optuna study name and MLflow parent run name.
        arch_config: model architecture hyperparameters. Falls back to ``_ARCH_CONFIG``
            (lr=1.75e-3, d_model=64, nhead=2, num_layers=2, dim_feedforward=256) when None.
            Expected keys: lr, d_model, nhead, num_layers, batch_size, pooling_mode,
            dim_feedforward.
    """
    resolved_arch = arch_config if arch_config is not None else _ARCH_CONFIG

    device = get_device()

    n_configs = len(FEATURE_CONFIGS)
    # repeat_idx makes (feature_config_idx, repeat_idx) pairs distinct grid points;
    # GridSampler otherwise stops the study after visiting each feature_config_idx once,
    # silently ignoring any requested repeats.
    grid_search_space = {
        "feature_config_idx": list(range(n_configs)),
        "repeat_idx": list(range(n_repeats)),
    }
    sampler = optuna.samplers.GridSampler(grid_search_space, seed=42)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=study_name
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_FeatureAblation")
    mlflow.set_experiment_tags(
        {
            "dataset": "store-sales-kaggle",
            "task": "time-series-forecasting",
            "loss": "RMSLE",
        }
    )

    n_trials = n_configs * n_repeats

    with mlflow.start_run(
        run_name=study_name,
        tags={
            "git_branch": get_branch(),
            "git_sha": get_sha(),
            "n_trials": str(n_trials),
            "epochs_per_trial": str(epochs),
            "n_repeats": str(n_repeats),
            "feature_configs": ",".join(cfg.name for cfg in FEATURE_CONFIGS),
            "arch_source": "tuned" if arch_config is not None else "default",
        },
    ):
        mlflow.log_params({k: str(v) for k, v in resolved_arch.items()})
        study.optimize(
            lambda trial: objective(
                trial, device, epochs, split, resolved_arch, n_repeats
            ),
            n_trials=n_trials,
        )

        # Summarize best val loss per config in the parent run.
        config_best: dict[str, float] = {}
        for completed_trial in study.trials:
            if completed_trial.value is None:
                continue
            cfg_idx = completed_trial.params["feature_config_idx"]
            cfg_name = FEATURE_CONFIGS[cfg_idx].name
            if (
                cfg_name not in config_best
                or completed_trial.value < config_best[cfg_name]
            ):
                config_best[cfg_name] = completed_trial.value

        for cfg_name, best_loss in config_best.items():
            mlflow.log_metric(f"best_val_loss_{cfg_name}", best_loss)

        if study.best_trials:
            best_config_idx = study.best_trial.params["feature_config_idx"]
            mlflow.log_param(
                "best_feature_config", FEATURE_CONFIGS[best_config_idx].name
            )
            mlflow.log_metric("study_best_val_loss", study.best_value)

    print("\nFeature ablation results:")
    print(f"{'Config':<20} {'Best val MSLE':>15}")
    print("-" * 37)
    for cfg in FEATURE_CONFIGS:
        loss = config_best.get(cfg.name, float("nan"))
        print(f"{cfg.name:<20} {loss:>15.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature ablation study for StoreSalesEncoderOnly"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="training epochs per feature config (default: 50)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction (default: 0.9)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="number of times to repeat each feature config (default: 1)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="store_sales_feature_ablation",
        help="Optuna study name and MLflow parent run name",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run the feature ablation study."""
    args = parse_args()
    run_study(
        epochs=args.epochs,
        split=args.split,
        n_repeats=args.n_repeats,
        study_name=args.study_name,
    )


if __name__ == "__main__":
    main()
