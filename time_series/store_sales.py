from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

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
        sales_tensor: float64 Tensor of shape [T, 54, 33].
        families: Index mapping column position -> product family name.
    """

    def __init__(
        self,
        window_lags: int = 60,
        output_lags: int = 16,
        data_dir: Path | None = None,
        copy: bool = False,
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
        self.train = self._load_train(data_dir)
        self.test = self._load_test(data_dir)
        self.sample_submission = self._load_sample_submission(data_dir)
        self.stores = self._load_stores(data_dir)
        self.oil = self._load_oil(data_dir)
        self.holidays = self._load_holidays(data_dir)

        self.window_lags = window_lags
        self.output_lags = output_lags
        self.sales_tensor, self.families = self._setup_tensor(
            self.train, self.stores, copy
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
        train: pd.DataFrame, stores: pd.DataFrame, copy: bool = False
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
        return torch.from_numpy(arr), families  # type: ignore[attr-defined]

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
