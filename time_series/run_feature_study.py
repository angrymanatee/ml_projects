"""Two-phase feature study for the encoder-only Store Sales Transformer.

Phase 1 — architecture tuning: runs Optuna HPO on the full feature set
(all new features enabled) to find the best (lr, d_model, num_layers, dim_feedforward).

Phase 2 — feature ablation: fixes the best architecture from phase 1 and
evaluates each feature group in isolation, a no-features baseline, and all
features combined. Uses GridSampler so every config runs exactly n_repeats times.

All results are logged to MLflow; phase 1 under StoreSales_EncoderOnly_Tune,
phase 2 under StoreSales_FeatureAblation.

Run with:
    uv run python -m time_series.run_feature_study
    uv run python -m time_series.run_feature_study --tune-trials 30 --tune-epochs 50 --ablation-epochs 100 --ablation-repeats 3
"""

import argparse

import optuna

from time_series.store_sales import (
    HOLIDAY_FEATURE_COLS,
    STORE_FEATURE_COLS,
    PoolingMode,
)
from time_series.tune_store_sales_encoder_only import tune
from time_series.tune_store_sales_features import run_study


def best_arch_from_study(study: optuna.Study) -> dict:
    """Extract a full architecture config dict from the best trial of a tuning study.

    ``build_config`` parametrises d_model as d_model_per_head * 2 with nhead=2 fixed.
    The other fixed params (batch_size, pooling_mode) are added here so the result
    is a drop-in replacement for ``_ARCH_CONFIG`` in ``tune_store_sales_features``.

    Args:
        study: completed Optuna study from ``tune_store_sales_encoder_only.tune``.

    Returns:
        Dict with keys: lr, d_model, nhead, num_layers, batch_size, pooling_mode,
        dim_feedforward.
    """
    best_params = study.best_trial.params
    return {
        "lr": best_params["lr"],
        "d_model": 2 * best_params["d_model_per_head"],
        "nhead": 2,
        "num_layers": best_params["num_layers"],
        "batch_size": 64,
        "pooling_mode": PoolingMode.ALL,
        "dim_feedforward": best_params["dim_feedforward"],
    }


def run(
    tune_n_trials: int,
    tune_epochs: int,
    ablation_epochs: int,
    ablation_repeats: int,
    split: float,
    tune_study_name: str,
    ablation_study_name: str,
) -> None:
    """Run the full two-phase feature study.

    Phase 1 tunes architecture on all features; phase 2 ablates features with the
    resulting best config. Both phases log to MLflow independently.

    Args:
        tune_n_trials: number of Optuna trials for phase 1 architecture search.
        tune_epochs: training epochs per trial in phase 1.
        ablation_epochs: training epochs per feature config in phase 2.
        ablation_repeats: number of times each feature config is re-run in phase 2.
        split: train/val split fraction used in both phases.
        tune_study_name: Optuna study name and MLflow run name for phase 1.
        ablation_study_name: Optuna study name and MLflow run name for phase 2.
    """
    print("=" * 60)
    print("Phase 1: architecture tuning on all features")
    print(f"  {tune_n_trials} trials × {tune_epochs} epochs")
    print("=" * 60)

    tuning_study = tune(
        n_trials=tune_n_trials,
        epochs_per_trial=tune_epochs,
        split=split,
        study_name=tune_study_name,
        store_feature_cols=list(STORE_FEATURE_COLS),
        holiday_features=list(HOLIDAY_FEATURE_COLS),
        include_oil=True,
        include_onpromotion=True,
    )

    best_arch = best_arch_from_study(tuning_study)
    print(f"\nBest architecture: {best_arch}")
    print(f"Best val MSLE: {tuning_study.best_value:.6f}\n")

    print("=" * 60)
    print("Phase 2: feature ablation with tuned architecture")
    print(f"  6 configs × {ablation_repeats} repeats × {ablation_epochs} epochs")
    print("=" * 60)

    run_study(
        epochs=ablation_epochs,
        split=split,
        n_repeats=ablation_repeats,
        study_name=ablation_study_name,
        arch_config=best_arch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-phase feature study: tune arch on all features, then ablate"
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=20,
        help="number of Optuna trials for phase 1 architecture search (default: 20)",
    )
    parser.add_argument(
        "--tune-epochs",
        type=int,
        default=50,
        help="training epochs per trial in phase 1 (default: 50)",
    )
    parser.add_argument(
        "--ablation-epochs",
        type=int,
        default=100,
        help="training epochs per feature config in phase 2 (default: 100)",
    )
    parser.add_argument(
        "--ablation-repeats",
        type=int,
        default=3,
        help="repeats per feature config in phase 2 (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction for both phases (default: 0.9)",
    )
    parser.add_argument(
        "--tune-study-name",
        type=str,
        default="store_sales_allfeatures_tune",
    )
    parser.add_argument(
        "--ablation-study-name",
        type=str,
        default="store_sales_feature_ablation_tuned",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run the two-phase feature study."""
    args = parse_args()
    run(
        tune_n_trials=args.tune_trials,
        tune_epochs=args.tune_epochs,
        ablation_epochs=args.ablation_epochs,
        ablation_repeats=args.ablation_repeats,
        split=args.split,
        tune_study_name=args.tune_study_name,
        ablation_study_name=args.ablation_study_name,
    )


if __name__ == "__main__":
    main()
