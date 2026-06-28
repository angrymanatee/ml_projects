"""Train an encoder-only Transformer on the Kaggle Store Sales dataset.

Architecture: linear projection → sinusoidal PE → TransformerEncoder → flatten → linear head.
No causal masking; the encoder attends over the full input window before projecting to the
output horizon. Loss is RMSLE (via MSLELoss), logged to MLflow.

Run with:
    uv run python -m time_series.main_store_sales_encoder_only
    uv run python -m time_series.main_store_sales_encoder_only --epochs 100 --lr 3e-4 --d-model 256
"""

import argparse
import enum
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from common.modules import GetLastIndex, PositionalEncoding
from time_series.store_sales import MSLELoss, StoreData, Trainer
from time_series.store_sales_viz import StoreSalesAnalyzer

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PoolingMode(enum.StrEnum):
    ALL = enum.auto()
    LAST = enum.auto()

    @staticmethod
    def get_module(pooling_mode: PoolingMode) -> nn.Module:
        match pooling_mode:
            case PoolingMode.ALL:
                return nn.Flatten(-2)
            case PoolingMode.LAST:
                return GetLastIndex()

    @staticmethod
    def parse(input_string: str) -> PoolingMode:
        return PoolingMode._value2member_map_[input_string]  # type: ignore


class StoreSalesEncoderOnly(nn.Module):
    """Encoder-only Transformer that maps a sales history window to a multi-step forecast.

    Input shape:  (batch, seq_len, n_stores, n_families)
    Output shape: (batch, n_output_steps, n_stores, n_families)

    The (n_stores x n_families) feature plane is flattened and linearly projected into
    d_model before being passed through the Transformer encoder. All encoder outputs are
    then flattened and projected to the full output horizon in one shot (no autoregression).
    """

    def __init__(
        self,
        n_stores: int,
        n_families: int,
        n_output_steps: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 512,
        pooling_mode: PoolingMode = PoolingMode.LAST,
    ) -> None:
        """Build the encoder-only model.

        Args:
            n_stores: Number of distinct stores in the dataset.
            n_families: Number of product families per store.
            n_output_steps: Forecast horizon (number of future time steps to predict).
            d_model: Transformer embedding dimension. Must be divisible by nhead.
            nhead: Number of attention heads.
            num_layers: Number of stacked TransformerEncoderLayer blocks.
            max_seq_length: Upper bound on input sequence length passed to PositionalEncoding.
        """
        super().__init__()
        self.n_stores = n_stores
        self.n_families = n_families
        self.n_output_steps = n_output_steps
        models = OrderedDict[str, nn.Module](
            [
                ("input_flatten", nn.Flatten(-2)),
                ("input_lin", nn.Linear(n_stores * n_families, d_model)),
                # Normalization?
                ("pos_enc", PositionalEncoding(d_model, max_length=max_seq_length)),
                (
                    "encoder",
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
                        num_layers=num_layers,
                    ),
                ),
                ("pooling", PoolingMode.get_module(pooling_mode)),
                ("output_lin", nn.LazyLinear(n_stores * n_families * n_output_steps)),
                (
                    "output_unflatten",
                    nn.Unflatten(-1, (n_output_steps, n_stores, n_families)),
                ),
                ("ReLU", nn.ReLU()),
            ]
        )
        self.sequence = nn.Sequential(models)

    def forward(self, input_sequence: Tensor) -> Tensor:
        """Run the full encoder-only forward pass.

        Args:
            input_sequence: Shape (batch, seq_len, n_stores, n_families).

        Returns:
            Forecast tensor of shape (batch, n_output_steps, n_stores, n_families).
        """
        return self.sequence(input_sequence)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training hyperparameters and model config."""
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
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument(
        "--pooling-mode",
        type=PoolingMode.parse,
        choices=[x.value for x in PoolingMode],
        default=PoolingMode.LAST,
    )
    return parser.parse_args()


def main() -> None:
    """Build dataset/model, run training, log results to MLflow, and produce validation plots."""
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

    model = StoreSalesEncoderOnly(
        n_stores=n_stores,
        n_families=n_families,
        n_output_steps=store_data.output_lags,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        pooling_mode=args.pooling_mode,
    )

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
