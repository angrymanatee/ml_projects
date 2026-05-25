from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.paths import get_data_dir

_DATA_DIR = get_data_dir() / "store-sales-time-series-forecasting"


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


class StoreData(Dataset):
    def __init__(
        self,
        window_lags: int = 60,
        output_lags: int = 16,
        data_dir: Path = _DATA_DIR,
        copy: bool = False,
    ) -> None:
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
        families: pd.Index = pivot[1].columns
        num_families = len(families)
        arr = pivot.to_numpy().reshape(pivot.shape[0], num_stores, num_families)
        if copy:
            arr = arr.copy()
        return torch.from_numpy(arr), families  # type: ignore[attr-defined]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index
        mid = index + self.window_lags
        end = mid + self.output_lags
        return self.sales_tensor[start:mid], self.sales_tensor[mid:end]

    def __len__(self) -> int:
        return self._len

    def __repr__(self) -> str:
        shape = self.sales_tensor.shape
        return f"StoreData(shape={shape}, window_lags={self.window_lags}, output_lags={self.output_lags})"
