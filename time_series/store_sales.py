import enum
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

# Date of the 7.8-magnitude earthquake that struck coastal Ecuador.
_ECUADOR_EARTHQUAKE_DATE = pd.Timestamp("2016-04-16")


class EarthquakeEncoding(enum.StrEnum):
    """How to encode elapsed time since the 2016 Ecuador earthquake.

    DECAY:  exp(-days_since / tau), equals 1.0 on the event day and decays toward 0.
            Requires an earthquake_tau parameter (days). Good default when you have
            a prior on the recovery timescale.
    LINEAR: days_since / 365, equals 0 on the event day and grows linearly.
            No hyperparameter; lets the model learn the decay shape via its FFN layers.
    """

    DECAY = enum.auto()
    LINEAR = enum.auto()


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(pd.to_datetime(df["date"]))


class StoreData(Dataset):
    """Store sales dataset for the Kaggle Store Sales forecasting competition.

    Loads all CSVs on construction and builds a [time, store, family] sales
    tensor and a [time, n_date_features] date features tensor. Acts as a
    sliding-window Dataset: each item is an (input, target) pair where input
    concatenates sales, broadcast date features, and optionally oil price
    along the family axis.

    Attributes:
        train, test, sample_submission, stores, oil, holidays: raw DataFrames.
        sales_tensor: float32 Tensor of shape [T, 54, 33].
        date_features_tensor: float32 Tensor of shape [T, n_date_features].
            When date_features=True (5 cols): days since first training date,
            sin/cos day-of-week (period 7), sin/cos day-of-year (period 365.25).
            When payday_features=True (2 cols): is_15th, is_month_end.
            When earthquake_encoding is set (1 col): earthquake proximity signal.
            Columns appear in the order listed above.
        n_date_features: total number of date feature columns (0–8).
        families: Index mapping column position -> product family name.
        oil_tensor: float32 Tensor of shape [T] when include_oil=True, else None.
            NaN gaps (weekends/holidays) are filled via forward-fill then back-fill.
        include_oil: whether oil price is appended to the input channel dimension.
    """

    def __init__(
        self,
        window_lags: int = 60,
        output_lags: int = 16,
        data_dir: Path | None = None,
        copy: bool = False,
        dtype: torch.dtype = torch.float32,
        date_features: bool = True,
        payday_features: bool = True,
        earthquake_encoding: EarthquakeEncoding | None = EarthquakeEncoding.DECAY,
        earthquake_tau: float = 30.0,
        include_oil: bool = False,
    ) -> None:
        """Load data and build the sales tensor.

        Args:
            window_lags: length of the input window fed to the model.
            output_lags: length of the prediction horizon (competition uses 16).
            data_dir: directory containing the competition CSVs. Defaults to
                <repo_root>/data/store-sales-time-series-forecasting.
            copy: if True, copy the underlying numpy array so the tensor is
                writable; costs ~2× memory.
            date_features: if True, append days-since-start, sin/cos day-of-week,
                and sin/cos day-of-year (5 columns) to each input timestep.
            payday_features: if True, append is_15th and is_month_end binary
                indicators (2 columns) to each input timestep.
            earthquake_encoding: how to encode time since the 2016 Ecuador earthquake
                (1 column). DECAY uses exp(-days_since / earthquake_tau); LINEAR uses
                days_since / 365. Set to None to omit the feature entirely.
            earthquake_tau: decay time constant in days for EarthquakeEncoding.DECAY.
                Ignored when earthquake_encoding is LINEAR or None.
            include_oil: if True, the oil price is appended as an extra channel
                in the input window. Each item's input becomes shape
                [window_lags, n_stores, n_families + n_date_features + 1]; targets
                are unchanged.
        """
        if data_dir is None:
            data_dir = get_data_dir() / "store-sales-time-series-forecasting"
        self.dtype = dtype
        self.include_oil = include_oil
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
        dates = pd.DatetimeIndex(self.train.index.unique().sort_values())
        self.date_features_tensor = self._setup_date_features(
            dates,
            date_features,
            payday_features,
            earthquake_encoding,
            earthquake_tau,
            dtype,
        )
        self.n_date_features: int = self.date_features_tensor.shape[1]
        self.oil_tensor: Tensor | None = (
            self._setup_oil_tensor(self.oil, self.train, dtype) if include_oil else None
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

    @staticmethod
    def _setup_date_features(
        dates: pd.DatetimeIndex,
        date_features: bool = True,
        payday_features: bool = True,
        earthquake_encoding: EarthquakeEncoding | None = EarthquakeEncoding.DECAY,
        earthquake_tau: float = 30.0,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Build a [T, n_date_features] tensor of enabled date features.

        date_features group (5 cols): days since first date, sin/cos day-of-week,
            sin/cos day-of-year.
        payday_features group (2 cols): is_15th, is_month_end.
        earthquake group (1 col): time-since-earthquake signal; see EarthquakeEncoding.
        Returns a [T, 0] empty tensor when all groups are disabled.
        """
        parts: list[Tensor] = []
        date_series = dates.to_series()
        if date_features:
            epoch = dates[0]
            days = torch.tensor(
                [(d - epoch).days for d in dates], dtype=dtype
            ).unsqueeze(1)
            two_pi = 2.0 * math.pi
            dow = torch.tensor(date_series.dt.dayofweek.to_numpy(), dtype=dtype)
            doy = torch.tensor(date_series.dt.dayofyear.to_numpy(), dtype=dtype)
            parts += [
                days,
                (two_pi * dow / 7).sin().unsqueeze(1),
                (two_pi * dow / 7).cos().unsqueeze(1),
                (two_pi * doy / 365.25).sin().unsqueeze(1),
                (two_pi * doy / 365.25).cos().unsqueeze(1),
            ]
        if payday_features:
            parts += [
                torch.tensor(
                    (date_series.dt.day == 15).to_numpy(), dtype=dtype
                ).unsqueeze(1),
                torch.tensor(
                    date_series.dt.is_month_end.to_numpy(), dtype=dtype
                ).unsqueeze(1),
            ]
        if earthquake_encoding is not None:
            days_since = torch.tensor(
                [(d - _ECUADOR_EARTHQUAKE_DATE).days for d in dates], dtype=dtype
            )
            if earthquake_encoding == EarthquakeEncoding.DECAY:
                feature = torch.where(
                    days_since >= 0,
                    torch.exp(-days_since.clamp(min=0) / earthquake_tau),
                    torch.zeros_like(days_since),
                )
            else:  # LINEAR
                feature = days_since.clamp(min=0) / 365.0
            parts.append(feature.unsqueeze(1))
        if not parts:
            return torch.zeros(len(dates), 0, dtype=dtype)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _setup_oil_tensor(
        oil: pd.DataFrame,
        train: pd.DataFrame,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Align oil prices to train dates and fill NaN gaps.

        Oil is reported only on trading days; weekends and holidays are NaN.
        Forward-fill propagates the last known price, then back-fill covers any
        leading NaNs at the start of the series.
        """
        train_dates = train.index.unique().sort_values()
        aligned = oil["dcoilwtico"].reindex(train_dates).ffill().bfill()
        arr = aligned.to_numpy(dtype="float32")
        return torch.from_numpy(arr).to(dtype)  # type: ignore[attr-defined]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return (input, target) windows.

        input shape:  [window_lags, n_stores, n_families + n_date_features + n_oil]
            Date features (if any) are broadcast across stores.
            Oil price (if enabled) is broadcast across stores as a final channel.
        target shape: [output_lags, n_stores, n_families] (sales only).
        """
        start = index
        mid = index + self.window_lags
        end = mid + self.output_lags
        sales_window = self.sales_tensor[start:mid]
        n_stores = sales_window.shape[1]
        parts: list[Tensor] = [sales_window]
        if self.n_date_features > 0:
            date_window = (
                self.date_features_tensor[start:mid].unsqueeze(1).expand(-1, n_stores, -1)
            )
            parts.append(date_window)
        if self.oil_tensor is not None:
            oil_channel = self.oil_tensor[start:mid].view(-1, 1, 1).expand(-1, n_stores, 1)
            parts.append(oil_channel)
        if len(parts) == 1:
            return parts[0], self.sales_tensor[mid:end]
        return (
            torch.cat(parts, dim=-1),  # type: ignore[attr-defined]
            self.sales_tensor[mid:end],
        )

    def __len__(self) -> int:
        """Number of stride-1 sliding windows; excludes the final output_lags days."""
        return self._len

    def __repr__(self) -> str:
        T, n_stores, n_families = self.sales_tensor.shape
        n_input = n_families + self.n_date_features + (1 if self.include_oil else 0)
        return (
            f"StoreData(T={T}, n_stores={n_stores}, "
            f"n_families={n_families}, n_date_features={self.n_date_features}, "
            f"include_oil={self.include_oil}, n_input_features={n_input}, "
            f"window_lags={self.window_lags}, output_lags={self.output_lags})"
        )


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
