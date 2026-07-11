import math
import tempfile
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import mlflow
from common.modules import MSLELoss

from .data import StoreData


def get_device() -> torch.device:
    """Return CUDA if available, MPS if on Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(
    store_data: StoreData,
    split: float,
    batch_size: int,
) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
    """Split StoreData into train/val DataLoaders.

    Args:
        store_data: dataset to split.
        split: fraction of windows used for training; must be in (0, 1).
        batch_size: batch size for both loaders.

    Returns:
        (train_loader, val_loader) — train loader shuffles, val does not.
    """
    if not (0.0 < split < 1.0):
        raise ValueError(f"split must be in (0, 1), got {split}")
    split_loc = int(len(store_data) * split)
    train_loader = DataLoader[Tensor](
        Subset(store_data, range(0, split_loc)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader[Tensor](
        Subset(store_data, range(split_loc, len(store_data))),
        batch_size=batch_size,
    )
    return train_loader, val_loader


class Trainer:
    """Generic supervised trainer with mlflow logging and checkpoint support.

    Runs train/val loops, logs metrics to the active mlflow run, and saves
    state-dict checkpoints as mlflow artifacts. Handles MPS, CUDA, and CPU
    devices; memory stats are reported only when the backend supports them.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader[Tensor],
        val_loader: DataLoader[Tensor],
        learning_rate: float = 1e-3,
        loss_func: nn.Module | None = None,
        save_checkpoints: bool = True,
        log_metrics: bool = True,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.device: torch.device = device
        self.model = model.to(device)
        self.optim = AdamW(model.parameters(), lr=learning_rate)
        self.loss_func = loss_func or MSLELoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_checkpoints = save_checkpoints
        self.log_metrics = log_metrics
        self.autocast_dtype = autocast_dtype

    def train(self, epochs: int, save_every_n_epochs: int | None = None) -> float:
        """Run the full training loop for `epochs` epochs.

        Learning rate follows a cosine annealing schedule from the initial
        learning_rate down to ~0 over the course of the run, so `epochs` sets the
        schedule's period in addition to the loop length.

        Args:
            epochs: total number of passes over the training set.
            save_every_n_epochs: if set, save a periodic checkpoint every N epochs
                in addition to the best-val checkpoint saved automatically.

        Returns:
            Best validation loss observed across all epochs.
        """
        scheduler = CosineAnnealingLR(self.optim, T_max=epochs)
        progress_bar = tqdm(range(epochs))
        digits = math.ceil(math.log10(epochs))
        train_loss = torch.nan
        val_loss = torch.nan
        best_val_loss = torch.inf
        for epoch_idx in progress_bar:
            progress_bar.set_description(f"T: {train_loss:.4f} | V: {val_loss:.4f}")
            train_loss = self.train_loop(epoch_idx).item()
            progress_bar.set_description(f"T: {train_loss:.4f} | V: {val_loss:.4f}")
            val_loss = self.val_loop(epoch_idx).item()
            progress_bar.set_description(f"T: {train_loss:.4f} | V: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.save_checkpoints:
                    progress_bar.set_description("saving best...")
                    self._checkpoint("best_model")
                if self.log_metrics:
                    mlflow.set_tag("best_epoch", epoch_idx)
            if (
                save_every_n_epochs
                and epoch_idx % save_every_n_epochs == 0
                and self.save_checkpoints
            ):
                progress_bar.set_description("saving periodic...")
                self._checkpoint(f"epoch_{epoch_idx:0{digits}}")
            if self.log_metrics:
                mlflow.log_metrics(
                    {"lr": float(scheduler.get_last_lr()[0])}, step=epoch_idx
                )
            scheduler.step()
        return best_val_loss

    def train_loop(self, epoch_idx: int) -> Tensor:
        """One pass over the training set; returns the loss of the last batch."""
        loss = torch.tensor(torch.nan, device=self.device)
        start_time = time.perf_counter()
        n_samples = 0
        for batch_X, batch_y in self.train_loader:
            self.optim.zero_grad()
            loss = self._run_loss(batch_X.to(self.device), batch_y.to(self.device))
            loss.backward()
            self.optim.step()
            n_samples += batch_X.shape[0]
        self._synchronize()
        elapsed = time.perf_counter() - start_time
        metrics: dict[str, float] = {
            "train_loss": loss.item(),
            "sample_rate": n_samples / elapsed,
        }
        mem = self._allocated_memory_gb()
        if mem is not None:
            metrics["mem_allocated_gb"] = mem
        if self.log_metrics:
            mlflow.log_metrics(metrics, step=epoch_idx)
        return loss

    @torch.inference_mode()
    def val_loop(self, epoch_idx: int) -> Tensor:
        """Average loss over the validation set."""
        loss = torch.tensor(0.0, device=self.device)
        n_batches = 0
        for batch_X, batch_y in self.val_loader:
            loss += self._run_loss(batch_X.to(self.device), batch_y.to(self.device))
            n_batches += 1
        loss /= n_batches
        if self.log_metrics:
            mlflow.log_metrics({"val_loss": loss.item()}, step=epoch_idx)
        return loss

    def _run_loss(self, batch_X: Tensor, batch_y: Tensor) -> Tensor:
        if self.autocast_dtype is not None:
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype
            ):
                return self.loss_func(self.model(batch_X), batch_y)
        return self.loss_func(self.model(batch_X), batch_y)

    def _checkpoint(self, name: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state_dict.pt"
            torch.save(self.model.state_dict(), path)
            mlflow.log_artifact(str(path), artifact_path=f"checkpoints/{name}")

    def _synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def _allocated_memory_gb(self) -> float | None:
        if self.device.type == "mps":
            return torch.mps.current_allocated_memory() / 1e9
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / 1e9
        return None


def run(
    model: nn.Module,
    dataset: StoreData,
    *,
    split: float = 0.8,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    epochs: int = 50,
    save_every_n_epochs: int | None = None,
    experiment_name: str = "store-sales",
    run_name: str | None = None,
) -> float:
    """Wire up loaders, MLflow, and Trainer, then train for `epochs` epochs.

    Args:
        model: Instantiated model to train. Must accept input of shape
            (batch, window_lags, n_stores, n_input_channels) and return
            (batch, output_lags, n_stores, n_families).
        dataset: Constructed StoreData instance. Query dataset.n_input_channels
            and dataset.sales_tensor.shape to size your model before calling.
        split: Fraction of windows used for training.
        batch_size: Batch size for both loaders.
        learning_rate: Initial AdamW learning rate (cosine-annealed to ~0).
        epochs: Number of training epochs.
        save_every_n_epochs: If set, save a periodic checkpoint every N epochs.
        experiment_name: MLflow experiment name.
        run_name: Optional MLflow run name.

    Returns:
        Best validation loss observed across all epochs.
    """
    train_loader, val_loader = make_loaders(dataset, split, batch_size)
    device = get_device()

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": type(model).__name__,
                "window_lags": dataset.window_lags,
                "output_lags": dataset.output_lags,
                "n_input_channels": dataset.n_input_channels,
                "split": split,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "device": device.type,
            }
        )
        trainer = Trainer(
            model,
            device,
            train_loader,
            val_loader,
            learning_rate=learning_rate,
        )
        return trainer.train(epochs, save_every_n_epochs=save_every_n_epochs)
