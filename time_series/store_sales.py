from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.paths import get_data_dir

_DATA_DIR = get_data_dir() / "store-sales-time-series-forecasting"


class StoreSalesData(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame


def load_data(data_dir: Path = _DATA_DIR) -> StoreSalesData:
    def _set_date_index(df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    return StoreSalesData(
        train=_set_date_index(pd.read_csv(data_dir / "train.csv")),
        test=_set_date_index(pd.read_csv(data_dir / "test.csv")),
        sample_submission=pd.read_csv(data_dir / "sample_submission.csv"),
        stores=pd.read_csv(data_dir / "stores.csv").set_index("store_nbr"),
        oil=_set_date_index(pd.read_csv(data_dir / "oil.csv")),
        holidays=_set_date_index(pd.read_csv(data_dir / "holidays_events.csv")),
    )


def make_sales_tensor(
    train: pd.DataFrame, stores: pd.DataFrame, copy: bool = False
) -> tuple[Tensor, pd.Index]:
    """Convert train DataFrame to a [time, store, family] tensor.

    Returns (tensor, families_index) where families_index maps family name -> position.
    copy=True makes the tensor writable at the cost of doubled memory.
    """
    num_stores = stores.shape[0]
    pivot = train.pivot(columns=("store_nbr", "family"), values="sales").sort_index(
        axis="columns"
    )
    families = pivot[1].columns
    num_families = len(families)
    arr = pivot.to_numpy().reshape(pivot.shape[0], num_stores, num_families)
    if copy:
        arr = arr.copy()
    tensor = torch.from_numpy(arr)  # type: ignore[attr-defined]
    return tensor, families


class StoreData(Dataset):
    def __init__(
        self,
        sales_tensor: Tensor,
        window_lags: int = 60,
        output_lags: int = 16,
    ) -> None:
        self.sales_tensor = sales_tensor
        self.window_lags = window_lags
        self.output_lags = output_lags
        self._len = sales_tensor.shape[0] - window_lags - output_lags

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input_start = index
        input_end = index + self.window_lags
        output_end = input_end + self.output_lags
        return (
            self.sales_tensor[input_start:input_end],
            self.sales_tensor[input_end:output_end],
        )

    def __len__(self) -> int:
        return self._len

    def __repr__(self) -> str:
        shape = self.sales_tensor.shape
        return f"StoreData(shape={shape}, window_lags={self.window_lags}, output_lags={self.output_lags})"
