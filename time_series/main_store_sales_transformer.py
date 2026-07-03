"""Train a Transformer model on the Kaggle Store Sales dataset.

Run with:
    uv run python -m time_series.main_store_sales_transformer
    uv run python -m time_series.main_store_sales_transformer --epochs 100 --lr 3e-4 --d-model 256
"""

import argparse

import torch

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from common.modules import MSLELoss
from time_series.store_sales import (
    StoreData,
    StoreSalesTransformer,
    Trainer,
    get_device,
    make_loaders,
)
from time_series.store_sales_viz import StoreSalesAnalyzer

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StoreSalesTransformer")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        metavar="N",
        help="save a checkpoint every N epochs (default: 50)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction (default: 0.9)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()
    store_data = StoreData(dtype=torch.float32)
    train_loader, val_loader = make_loaders(store_data, args.split, args.batch_size)

    model = StoreSalesTransformer(
        n_stores=store_data.stores.shape[0],
        n_families=store_data.families.size,
        n_output_steps=store_data.output_lags,
        d_model=args.d_model,
        nhead=args.nhead,
    )

    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_TransformerBasic")
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
            }
        )
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.lr,
            loss_func=MSLELoss(),
        )
        trainer.train(args.epochs, save_every_n_epochs=args.save_every)
        StoreSalesAnalyzer(
            model=model,
            data_loader=val_loader,
            stores=store_data.stores,
            families=store_data.families,
            device=device,
        ).run()


if __name__ == "__main__":
    main()
