from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.paths import get_data_dir

_DATA_DIR = get_data_dir() / "store-sales-time-series-forecasting"


class StoreData(Dataset):
    def __init__(
        self,
        window_lags: int = 60,
        output_lags: int = 16,
        data_dir: Path = _DATA_DIR,
        copy: bool = False,
    ) -> None:
        def _date_index(df: pd.DataFrame) -> pd.DataFrame:
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")

        self.train = _date_index(pd.read_csv(data_dir / "train.csv"))
        self.test = _date_index(pd.read_csv(data_dir / "test.csv"))
        self.sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        self.stores = pd.read_csv(data_dir / "stores.csv").set_index("store_nbr")
        self.oil = _date_index(pd.read_csv(data_dir / "oil.csv"))
        self.holidays = _date_index(pd.read_csv(data_dir / "holidays_events.csv"))

        self.window_lags = window_lags
        self.output_lags = output_lags

        num_stores = self.stores.shape[0]
        pivot = self.train.pivot(
            columns=("store_nbr", "family"), values="sales"
        ).sort_index(axis="columns")
        self.families: pd.Index = pivot[1].columns
        num_families = len(self.families)
        arr = pivot.to_numpy().reshape(pivot.shape[0], num_stores, num_families)
        if copy:
            arr = arr.copy()
        self.sales_tensor: Tensor = torch.from_numpy(arr)  # type: ignore[attr-defined]

        self._len = self.sales_tensor.shape[0] - window_lags - output_lags

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
