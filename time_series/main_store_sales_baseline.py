"""Hold-last-value baseline for the Kaggle Store Sales dataset.

The simplest possible forecast: repeat the final observed value across the
entire prediction horizon. This establishes a lower bound that learned models
should comfortably beat.

Run with:
    uv run python -m time_series.main_store_sales_baseline
    uv run python -m time_series.main_store_sales_baseline --split 0.8
"""

import argparse

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from time_series.store_sales import MSLELoss, StoreData
from time_series.store_sales_viz import StoreSalesAnalyzer

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HoldLastValue(nn.Module):
    """Baseline model that repeats the last observed time step as the forecast.

    Requires no training. For each sample in a batch, the final input step is
    tiled across all output steps. The resulting forecast is constant in time,
    making it a useful sanity-check lower bound for more complex models.

    Input shape:  (batch, seq_len, n_stores, n_families)
    Output shape: (batch, n_output_steps, n_stores, n_families)
    """

    def __init__(self, n_output_steps: int) -> None:
        """
        Args:
            n_output_steps: Number of future time steps to predict. Controls
                how many times the last observation is repeated.
        """
        super().__init__()
        self.n_output_steps = n_output_steps

    def forward(self, input_sequence: Tensor) -> Tensor:
        """Tile the last time step across the output horizon.

        Args:
            input_sequence: Shape (batch, seq_len, n_stores, n_families).

        Returns:
            Tensor of shape (batch, n_output_steps, n_stores, n_families)
            with the last observed value repeated at every output step.
        """
        last_step = input_sequence[:, -1:, :, :]
        return last_step.expand(-1, self.n_output_steps, -1, -1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compute_loss(
    model: nn.Module,
    data_loader: DataLoader,  # type: ignore[type-arg]
    loss_func: nn.Module,
) -> float:
    """Compute mean loss over all batches in a DataLoader.

    Args:
        model: Model to evaluate; called as model(batch_X).
        data_loader: Yields (input, target) pairs.
        loss_func: Reduction should be 'mean' so losses are comparable across batch sizes.

    Returns:
        Mean loss across all batches (simple average, not sample-weighted).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in data_loader:
        total_loss += loss_func(model(batch_x), batch_y).item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else float("nan")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run hold-last-value baseline")
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction (default: 0.9); val set is used for metrics and plots",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Evaluate the hold-last-value baseline and log metrics and plots to MLflow."""
    args = parse_args()

    store_data = StoreData(dtype=torch.float32)
    device = torch.device("cpu")  # no GPU ops needed for this baseline

    split_loc = int(len(store_data) * args.split)
    train_data = Subset(store_data, range(0, split_loc))
    val_data = Subset(store_data, range(split_loc, len(store_data)))
    train_loader: DataLoader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False
    )
    val_loader: DataLoader = DataLoader(val_data, batch_size=args.batch_size)

    model = HoldLastValue(n_output_steps=store_data.output_lags)
    loss_func = MSLELoss()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("StoreSales_Baseline")
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
            "architecture": "hold-last-value",
            "input_window_days": str(store_data.window_lags),
            "output_window_days": str(store_data.output_lags),
            "device": str(device),
            "git_branch": get_branch(),
            "git_sha": get_sha(),
        }
    ):
        mlflow.log_params(
            {
                "split_fraction": args.split,
                "batch_size": args.batch_size,
            }
        )

        train_loss = compute_loss(model, train_loader, loss_func)
        val_loss = compute_loss(model, val_loader, loss_func)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss})

        StoreSalesAnalyzer(
            model=model,
            data_loader=val_loader,
            stores=store_data.stores,
            families=store_data.families,
            device=device,
        ).run()


if __name__ == "__main__":
    main()
