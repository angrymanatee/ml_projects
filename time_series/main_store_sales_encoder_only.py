"""Train an encoder-only Transformer on the Kaggle Store Sales dataset.

Architecture: flatten → linear projection → sinusoidal PE → TransformerEncoder → pool → linear head → unflatten → ReLU.
No causal masking; the encoder attends over the full input window before projecting to the
output horizon. Loss is RMSLE (via MSLELoss), logged to MLflow.

Run with:
    uv run python -m time_series.main_store_sales_encoder_only
    uv run python -m time_series.main_store_sales_encoder_only --epochs 100 --lr 3e-4 --d-model 256
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
    PoolingMode,
    StoreData,
    StoreSalesEncoderOnly,
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
) -> tuple[float, StoreSalesEncoderOnly, DataLoader[Tensor]]:
    """Build loaders and model from config, train, return results.

    Must be called inside an active mlflow.start_run() context — Trainer
    logs metrics and checkpoints to whatever run is currently active.

    Args:
        config: keys lr (float), d_model (int), nhead (int), num_layers (int),
                batch_size (int), pooling_mode (PoolingMode),
                dim_feedforward (int, default 256).
        store_data: pre-loaded dataset; shared to avoid redundant CSV I/O.
        device: torch device.
        epochs: number of training epochs.
        split: fraction used for training; must be in (0, 1).
        save_every: checkpoint every N epochs.
        save_checkpoints: if False, skips all checkpoint writes. Set False during hyperparameter search.
        log_metrics: if False, skips all mlflow.log_metrics and mlflow.set_tag calls. Set False during hyperparameter search.

    Returns:
        Tuple of (best_val_loss, trained_model, val_loader).
    """
    train_loader, val_loader = make_loaders(store_data, split, config["batch_size"])

    model = StoreSalesEncoderOnly(
        n_stores=store_data.stores.shape[0],
        n_families=store_data.families.size,
        n_output_steps=store_data.output_lags,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        pooling_mode=config["pooling_mode"],
        dim_feedforward=config.get("dim_feedforward", 256),
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
    """Parse command-line arguments for training hyperparameters and model config."""
    parser = argparse.ArgumentParser(description="Train StoreSalesTransformer")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        metavar="N",
        help="save a checkpoint every N epochs (default: 50)",
    )
    parser.add_argument("--lr", type=float, default=0.0011852762898126566, metavar="LR")
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction (default: 0.9)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--pooling-mode",
        type=PoolingMode.parse,
        choices=[x.value for x in PoolingMode],
        default=PoolingMode.ALL,
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=512,
        help="FFN width inside each TransformerEncoderLayer (default: 512)",
    )
    parser.add_argument(
        "--oil",
        action="store_true",
        default=True,
        help="append oil price as an extra input channel (default: on)",
    )
    parser.add_argument(
        "--no-oil",
        dest="oil",
        action="store_false",
        help="disable oil price input channel",
    )
    parser.add_argument(
        "--onpromotion",
        action="store_true",
        default=True,
        help="include onpromotion as an additional input feature (default: on)",
    )
    parser.add_argument(
        "--no-onpromotion",
        dest="onpromotion",
        action="store_false",
        help="disable onpromotion input feature",
    )
    parser.add_argument(
        "--store-features",
        nargs="*",
        choices=list(STORE_FEATURE_COLS),
        default=list(STORE_FEATURE_COLS),
        metavar="COL",
        help=(
            f"store metadata columns to append as input features "
            f"(choices: {', '.join(STORE_FEATURE_COLS)}, default: all)"
        ),
    )
    parser.add_argument(
        "--holiday-features",
        nargs="*",
        choices=list(HOLIDAY_FEATURE_COLS),
        default=list(HOLIDAY_FEATURE_COLS),
        metavar="FEAT",
        help=(
            f"holiday/event features to append as binary input channels "
            f"(choices: {', '.join(HOLIDAY_FEATURE_COLS)}, default: all)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build dataset/model, run training, log results to MLflow, and produce validation plots."""
    args = parse_args()

    device = get_device()

    store_data = StoreData(
        dtype=torch.float32,
        include_oil=args.oil,
        include_onpromotion=args.onpromotion,
        store_feature_cols=args.store_features,
        holiday_features=args.holiday_features,
    )

    config = {
        "lr": args.lr,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "pooling_mode": args.pooling_mode,
        "dim_feedforward": args.dim_feedforward,
    }

    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_TransformerEncoderOnly")
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
            "architecture": "transformer",
            "input_window_days": str(store_data.window_lags),
            "output_window_days": str(store_data.output_lags),
            "device": str(device),
            "git_branch": get_branch(),
            "git_sha": get_sha(),
            "pool_type": str(args.pooling_mode),
        }
    ):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "split_fraction": args.split,
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "include_oil": args.oil,
                "include_onpromotion": args.onpromotion,
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
