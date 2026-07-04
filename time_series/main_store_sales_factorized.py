"""Train StoreSalesFactorizedEncoder on the Kaggle Store Sales dataset.

Architecture: factorized two-stage encoder.
  Stage 1 (time): process each (store, family) time series independently with a
    TransformerEncoder, then mean-pool over time to get one vector per (store, family).
  Stage 2 (store×family): run a TransformerEncoder across all (store, family) pairs,
    flatten, and project to the output horizon.
Input shape:  [batch, window_lags, n_stores, n_families, n_input_channels]
Output shape: [batch, output_lags, n_stores, n_families]
Loss: RMSLE (via MSLELoss), logged to MLflow.

Run with:
    uv run python -m time_series.main_store_sales_factorized
    uv run python -m time_series.main_store_sales_factorized --epochs 100 --d-model-time 128
"""

import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from common.modules import MSLELoss
from time_series.store_sales import (
    HOLIDAY_FEATURE_COLS,
    STORE_FEATURE_COLS,
    StoreData,
    StoreSalesFactorizedEncoder,
    Trainer,
    get_device,
    make_loaders,
)
from time_series.store_sales_viz import StoreSalesAnalyzer

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_eval(
    config: dict,
    store_data: StoreData,
    device: torch.device,
    epochs: int,
    split: float = 0.9,
    save_every: int | None = None,
    save_checkpoints: bool = True,
    log_metrics: bool = True,
) -> tuple[float, StoreSalesFactorizedEncoder, DataLoader[Tensor]]:
    """Build loaders and model from config, train, return results.

    Must be called inside an active mlflow.start_run() context.

    Args:
        config: keys lr (float), batch_size (int), d_model_time (int),
                nhead_time (int), dim_feedforward_time (int), num_layers_time (int),
                d_model_sf (int), nhead_sf (int), dim_feedforward_sf (int),
                num_layers_sf (int).
        store_data: pre-loaded dataset with flatten_output=False.
        device: torch device.
        epochs: number of training epochs.
        split: fraction used for training; must be in (0, 1).
        save_every: checkpoint every N epochs.
        save_checkpoints: if False, skips all checkpoint writes.
        log_metrics: if False, skips all mlflow.log_metrics calls.

    Returns:
        Tuple of (best_val_loss, trained_model, val_loader).
    """
    train_loader, val_loader = make_loaders(store_data, split, config["batch_size"])

    model = StoreSalesFactorizedEncoder(
        n_stores=store_data.stores.shape[0],
        n_families=store_data.families.size,
        n_output_steps=store_data.output_lags,
        d_model_time=config["d_model_time"],
        nhead_time=config["nhead_time"],
        dim_feedforward_time=config.get("dim_feedforward_time", 256),
        num_layers_time=config["num_layers_time"],
        d_model_sf=config["d_model_sf"],
        nhead_sf=config["nhead_sf"],
        dim_feedforward_sf=config.get("dim_feedforward_sf", 256),
        num_layers_sf=config["num_layers_sf"],
    )

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["lr"],
        loss_func=MSLELoss(),
        save_checkpoints=save_checkpoints,
        log_metrics=log_metrics,
    )
    best_val_loss = trainer.train(epochs, save_every_n_epochs=save_every)
    return best_val_loss, model, val_loader


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hyperparameters and dataset options."""
    parser = argparse.ArgumentParser(description="Train StoreSalesFactorizedEncoder")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        metavar="N",
        help="checkpoint every N epochs (default: 50)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction (default: 0.9)",
    )
    parser.add_argument("--batch-size", type=int, default=64)

    # Time encoder
    parser.add_argument("--d-model-time", type=int, default=64)
    parser.add_argument("--nhead-time", type=int, default=4)
    parser.add_argument("--num-layers-time", type=int, default=2)
    parser.add_argument("--dim-feedforward-time", type=int, default=256)

    # Store×family encoder
    parser.add_argument("--d-model-sf", type=int, default=64)
    parser.add_argument("--nhead-sf", type=int, default=4)
    parser.add_argument("--num-layers-sf", type=int, default=2)
    parser.add_argument("--dim-feedforward-sf", type=int, default=256)

    # Dataset features
    parser.add_argument(
        "--oil",
        action="store_true",
        default=True,
        help="include oil price (default: on)",
    )
    parser.add_argument("--no-oil", dest="oil", action="store_false")
    parser.add_argument(
        "--onpromotion",
        action="store_true",
        default=True,
        help="include onpromotion (default: on)",
    )
    parser.add_argument("--no-onpromotion", dest="onpromotion", action="store_false")
    parser.add_argument(
        "--store-features",
        nargs="*",
        choices=list(STORE_FEATURE_COLS),
        default=list(STORE_FEATURE_COLS),
        metavar="COL",
        help=f"store metadata columns (choices: {', '.join(STORE_FEATURE_COLS)}, default: all)",
    )
    parser.add_argument(
        "--holiday-features",
        nargs="*",
        choices=list(HOLIDAY_FEATURE_COLS),
        default=list(HOLIDAY_FEATURE_COLS),
        metavar="FEAT",
        help=f"holiday features (choices: {', '.join(HOLIDAY_FEATURE_COLS)}, default: all)",
    )
    return parser.parse_args()


def main() -> None:
    """Build dataset/model, run training, log results to MLflow."""
    args = parse_args()
    device = get_device()

    store_data = StoreData(
        dtype=torch.float32,
        include_oil=args.oil,
        include_onpromotion=args.onpromotion,
        store_feature_cols=args.store_features,
        holiday_features=args.holiday_features,
        flatten_output=False,
    )

    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "d_model_time": args.d_model_time,
        "nhead_time": args.nhead_time,
        "num_layers_time": args.num_layers_time,
        "dim_feedforward_time": args.dim_feedforward_time,
        "d_model_sf": args.d_model_sf,
        "nhead_sf": args.nhead_sf,
        "num_layers_sf": args.num_layers_sf,
        "dim_feedforward_sf": args.dim_feedforward_sf,
    }

    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_FactorizedEncoder")
    mlflow.set_experiment_tags(
        {
            "dataset": "store-sales-kaggle",
            "task": "time-series-forecasting",
            "loss": "RMSLE",
            "prediction_horizon_days": str(store_data.output_lags),
        }
    )

    with mlflow.start_run(
        tags={
            "architecture": "factorized-encoder",
            "input_window_days": str(store_data.window_lags),
            "output_window_days": str(store_data.output_lags),
            "n_input_channels": str(store_data.n_input_channels),
            "device": str(device),
            "git_branch": get_branch(),
            "git_sha": get_sha(),
        }
    ):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "split_fraction": args.split,
                "batch_size": args.batch_size,
                "d_model_time": args.d_model_time,
                "nhead_time": args.nhead_time,
                "num_layers_time": args.num_layers_time,
                "dim_feedforward_time": args.dim_feedforward_time,
                "d_model_sf": args.d_model_sf,
                "nhead_sf": args.nhead_sf,
                "num_layers_sf": args.num_layers_sf,
                "dim_feedforward_sf": args.dim_feedforward_sf,
                "include_oil": args.oil,
                "include_onpromotion": args.onpromotion,
                "n_input_channels": store_data.n_input_channels,
                "store_features": (
                    ",".join(args.store_features) if args.store_features else "none"
                ),
                "holiday_features": (
                    ",".join(args.holiday_features) if args.holiday_features else "none"
                ),
            }
        )
        _val_loss, model, val_loader = train_and_eval(
            config=config,
            store_data=store_data,
            device=device,
            epochs=args.epochs,
            split=args.split,
            save_every=args.save_every,
        )
        StoreSalesAnalyzer(
            model=model,
            data_loader=val_loader,
            stores=store_data.stores,
            families=store_data.families,
            device=device,
        ).run()


if __name__ == "__main__":
    main()
