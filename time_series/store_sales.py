import math
import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.paths import get_data_dir


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(pd.to_datetime(df["date"]))


class StoreData(Dataset):
    """Store sales dataset for the Kaggle Store Sales forecasting competition.

    Loads all CSVs on construction and builds a [time, store, family] sales
    tensor. Acts as a sliding-window Dataset: each item is an (input, target)
    pair of consecutive windows drawn from the training series.

    Attributes:
        train, test, sample_submission, stores, oil, holidays: raw DataFrames.
        sales_tensor: float32 Tensor of shape [T, 54, 33].
        families: Index mapping column position -> product family name.
    """

    def __init__(
        self,
        window_lags: int = 60,
        output_lags: int = 16,
        data_dir: Path | None = None,
        copy: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Load data and build the sales tensor.

        Args:
            window_lags: length of the input window fed to the model.
            output_lags: length of the prediction horizon (competition uses 16).
            data_dir: directory containing the competition CSVs. Defaults to
                <repo_root>/data/store-sales-time-series-forecasting.
            copy: if True, copy the underlying numpy array so the tensor is
                writable; costs ~2× memory.
        """
        if data_dir is None:
            data_dir = get_data_dir() / "store-sales-time-series-forecasting"
        self.dtype = dtype
        self.train = self._load_train(data_dir)
        self.test = self._load_test(data_dir)
        self.sample_submission = self._load_sample_submission(data_dir)
        self.stores = self._load_stores(data_dir)
        self.oil = self._load_oil(data_dir)
        self.holidays = self._load_holidays(data_dir)

        self.window_lags = window_lags
        self.output_lags = output_lags
        self.sales_tensor, self.families = self._setup_tensor(
            self.train, self.stores, dtype, copy
        )
        self._len = self.sales_tensor.shape[0] - window_lags - output_lags

    @staticmethod
    def _load_train(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "train.csv"))

    @staticmethod
    def _load_test(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "test.csv"))

    @staticmethod
    def _load_sample_submission(data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir / "sample_submission.csv")

    @staticmethod
    def _load_stores(data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir / "stores.csv").set_index("store_nbr")

    @staticmethod
    def _load_oil(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "oil.csv"))

    @staticmethod
    def _load_holidays(data_dir: Path) -> pd.DataFrame:
        return _date_index(pd.read_csv(data_dir / "holidays_events.csv"))

    @staticmethod
    def _setup_tensor(
        train: pd.DataFrame,
        stores: pd.DataFrame,
        dtype: torch.dtype = torch.float32,
        copy: bool = False,
    ) -> tuple[Tensor, pd.Index]:
        num_stores = stores.shape[0]
        pivot = train.pivot(columns=("store_nbr", "family"), values="sales").sort_index(
            axis="columns"
        )
        families: pd.Index = pivot.columns.get_level_values("family").unique()
        num_families = len(families)
        arr = pivot.to_numpy().reshape(pivot.shape[0], num_stores, num_families)
        if copy:
            arr = arr.copy()
        return torch.from_numpy(arr).to(dtype), families  # type: ignore[attr-defined]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return (input, target) windows of shape [window_lags, 54, 33] and [output_lags, 54, 33]."""
        start = index
        mid = index + self.window_lags
        end = mid + self.output_lags
        return self.sales_tensor[start:mid], self.sales_tensor[mid:end]

    def __len__(self) -> int:
        """Number of stride-1 sliding windows; excludes the final output_lags days."""
        return self._len

    def __repr__(self) -> str:
        shape = self.sales_tensor.shape
        return f"StoreData(shape={shape}, window_lags={self.window_lags}, output_lags={self.output_lags})"


class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error loss.

    Computes MSE(log(1 + input), log(1 + target)), which is the competition
    metric (RMSLE) squared. Use sqrt on the output to recover RMSLE.
    Inputs must be non-negative; log1p is used for numerical stability near zero.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self._mse_loss(torch.log1p(input), torch.log1p(target))  # type: ignore[attr-defined]


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
    ) -> None:
        self.device: torch.device = device
        self.model = model.to(device)
        self.optim = AdamW(model.parameters(), lr=learning_rate)
        self.loss_func = loss_func or MSLELoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_checkpoints = save_checkpoints
        self.log_metrics = log_metrics

    def train(self, epochs: int, save_every_n_epochs: int | None = None) -> float:
        """Run the full training loop for `epochs` epochs.

        Args:
            epochs: total number of passes over the training set.
            save_every_n_epochs: if set, save a periodic checkpoint every N epochs
                in addition to the best-val checkpoint saved automatically.

        Returns:
            Best validation loss observed across all epochs.
        """
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
