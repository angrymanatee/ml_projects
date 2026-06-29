"""Train an encoder-only Transformer on the Kaggle Store Sales dataset.

Architecture: flatten → linear projection → sinusoidal PE → TransformerEncoder → pool → linear head → unflatten → ReLU.
No causal masking; the encoder attends over the full input window before projecting to the
output horizon. Loss is RMSLE (via MSLELoss), logged to MLflow.

Run with:
    uv run python -m time_series.main_store_sales_encoder_only
    uv run python -m time_series.main_store_sales_encoder_only --epochs 100 --lr 3e-4 --d-model 256
"""

import argparse
import enum
from collections import OrderedDict

import mlflow
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from common.modules import (
    GetLastIndex,
    PositionalEncoding,
)
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
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        max_seq_length: int = 512,
        pooling_mode: PoolingMode = PoolingMode.ALL,
        dim_feedforward: int = 256,
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
            pooling_mode: How to collapse the sequence dimension after the encoder.
                ALL flattens all timestep embeddings; LAST takes only the final one.
            dim_feedforward: Width of the FFN sublayer inside each TransformerEncoderLayer.
                PyTorch default is 2048; for d_model=64 something in [128, 512] is typical.
        """
        super().__init__()
        self.n_stores = n_stores
        self.n_families = n_families
        self.n_output_steps = n_output_steps
        models = OrderedDict[str, nn.Module](
            [
                ("input_flatten", nn.Flatten(-2)),
                ("input_proj", nn.LazyLinear(d_model)),
                # Normalization?
                ("pos_enc", PositionalEncoding(d_model, max_length=max_seq_length)),
                (
                    "encoder",
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model,
                            nhead,
                            dim_feedforward=dim_feedforward,
                            batch_first=True,
                        ),
                        num_layers=num_layers,
                    ),
                ),
                ("pooling", PoolingMode.get_module(pooling_mode)),
                ("output_proj", nn.LazyLinear(n_stores * n_families * n_output_steps)),
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
    if not (0.0 < split < 1.0):
        raise ValueError(f"split must be in (0, 1), got {split}")
    split_loc = int(len(store_data) * split)
    train_loader = DataLoader[Tensor](
        Subset(store_data, range(0, split_loc)),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader[Tensor](
        Subset(store_data, range(split_loc, len(store_data))),
        batch_size=config["batch_size"],
    )

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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        metavar="N",
        help="save a checkpoint every N epochs (default: 50)",
    )
    parser.add_argument("--lr", type=float, default=1.75e-3, metavar="LR")
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
        default=256,
        help="FFN width inside each TransformerEncoderLayer (default: 256)",
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
