"""Train a Transformer model on the Kaggle Store Sales dataset.

Run with:
    uv run python -m time_series.main_store_sales_transformer
    uv run python -m time_series.main_store_sales_transformer --epochs 100 --lr 3e-4 --d-model 256
"""

import argparse

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from time_series.store_sales import MSLELoss, StoreData, Trainer
from time_series.store_sales_viz import StoreSalesAnalyzer

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class StoreSalesTransformer(nn.Module):
    """Transformer encoder-decoder for multi-step store sales forecasting.

    Maps an input window of shape [batch, window_lags, n_stores, n_families]
    to a prediction window of shape [batch, output_lags, n_stores, n_families].
    The spatial dimensions (store x family) are flattened into a single feature
    vector per time step and projected into d_model before the transformer.
    """

    def __init__(
        self,
        n_stores: int,
        n_families: int,
        n_output_steps: int,
        d_model: int = 128,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        embedding_size = n_stores * n_families
        self.embedding_size = embedding_size
        self.n_output_steps = n_output_steps
        self.input_transform = nn.Linear(embedding_size, d_model)
        self.output_transform = nn.Linear(d_model, embedding_size)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.output_relu = nn.ReLU()
        self._n_stores = n_stores
        self._n_families = n_families

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: [batch, window_lags, n_stores, n_families]
        Returns:
            [batch, n_output_steps, n_stores, n_families]
        """
        input_internal = self.input_transform(input.flatten(-2))
        tgt_internal = torch.zeros(
            input_internal.shape[0],
            self.n_output_steps,
            self.transformer.d_model,
            dtype=input_internal.dtype,
            device=input_internal.device,
        )
        output_internal = self.transformer(input_internal, tgt_internal)
        out_shape = (-1, self.n_output_steps, self._n_stores, self._n_families)
        return self.output_relu(
            self.output_transform(output_internal).reshape(out_shape)
        )


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

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    store_data = StoreData(dtype=torch.float32)
    n_stores = store_data.stores.shape[0]
    n_families = store_data.families.size

    split_loc = int(len(store_data) * args.split)
    train_data = Subset(store_data, range(0, split_loc))
    val_data = Subset(store_data, range(split_loc, len(store_data)))
    train_loader = DataLoader[Tensor](
        train_data, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader[Tensor](val_data, batch_size=args.batch_size)

    model = StoreSalesTransformer(
        n_stores=n_stores,
        n_families=n_families,
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
