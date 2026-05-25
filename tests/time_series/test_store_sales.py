import pandas as pd
import pytest
import torch

from common.paths import get_data_dir
from time_series.store_sales import StoreData

_DATA_DIR = get_data_dir() / "store-sales-time-series-forecasting"


@pytest.fixture(scope="module")
def ds() -> StoreData:
    return StoreData()


# --- StoreData (integration) ---


def test_train_has_date_index(ds: StoreData) -> None:
    assert isinstance(ds.train.index, pd.DatetimeIndex)


def test_stores_indexed_by_store_nbr(ds: StoreData) -> None:
    assert ds.stores.index.name == "store_nbr"


def test_oil_has_date_index(ds: StoreData) -> None:
    assert isinstance(ds.oil.index, pd.DatetimeIndex)


def test_sales_tensor_shape(ds: StoreData) -> None:
    num_dates = ds.train.index.nunique()
    num_stores = ds.stores.shape[0]
    num_families = len(ds.families)
    assert ds.sales_tensor.shape == (num_dates, num_stores, num_families)


def test_sales_tensor_dtype(ds: StoreData) -> None:
    assert ds.sales_tensor.dtype == torch.float64  # type: ignore[attr-defined]


def test_len(ds: StoreData) -> None:
    assert len(ds) == ds.sales_tensor.shape[0] - ds.window_lags - ds.output_lags


def test_item_shapes(ds: StoreData) -> None:
    x, y = ds[0]
    _, num_stores, num_families = ds.sales_tensor.shape
    assert x.shape == (ds.window_lags, num_stores, num_families)
    assert y.shape == (ds.output_lags, num_stores, num_families)


def test_custom_lags() -> None:
    ds = StoreData(window_lags=30, output_lags=5)
    x, y = ds[0]
    assert x.shape[0] == 30
    assert y.shape[0] == 5


# --- individual loaders ---


def test_load_train_date_index() -> None:
    assert isinstance(StoreData._load_train(_DATA_DIR).index, pd.DatetimeIndex)


def test_load_stores_index() -> None:
    assert StoreData._load_stores(_DATA_DIR).index.name == "store_nbr"


# --- _setup_tensor ---


@pytest.fixture(scope="module")
def train_stores(ds: StoreData) -> tuple[pd.DataFrame, pd.DataFrame]:
    return ds.train, ds.stores


def test_setup_tensor_shape(train_stores: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, stores = train_stores
    tensor, families = StoreData._setup_tensor(train, stores)
    assert tensor.shape == (train.index.nunique(), stores.shape[0], len(families))


def test_setup_tensor_copy_is_writable(
    train_stores: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, stores = train_stores
    tensor, _ = StoreData._setup_tensor(train, stores, copy=True)
    tensor[0, 0, 0] = -1.0  # would raise or corrupt if non-writable
