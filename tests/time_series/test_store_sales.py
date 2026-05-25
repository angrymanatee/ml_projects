import pytest
import torch

from time_series.store_sales import (
    StoreData,
    StoreSalesData,
    load_data,
    make_sales_tensor,
)


@pytest.fixture(scope="module")
def data() -> StoreSalesData:
    return load_data()


def test_load_data_returns_named_tuple(data: StoreSalesData) -> None:
    assert isinstance(data, StoreSalesData)


def test_load_data_train_has_date_index(data: StoreSalesData) -> None:
    import pandas as pd

    assert isinstance(data.train.index, pd.DatetimeIndex)


def test_load_data_stores_indexed_by_store_nbr(data: StoreSalesData) -> None:
    assert data.stores.index.name == "store_nbr"


def test_load_data_oil_has_date_index(data: StoreSalesData) -> None:
    import pandas as pd

    assert isinstance(data.oil.index, pd.DatetimeIndex)


def test_make_sales_tensor_shape(data: StoreSalesData) -> None:
    tensor, families = make_sales_tensor(data.train, data.stores)
    num_dates = data.train.index.nunique()
    num_stores = data.stores.shape[0]
    num_families = len(families)
    assert tensor.shape == (num_dates, num_stores, num_families)


def test_make_sales_tensor_dtype(data: StoreSalesData) -> None:
    tensor, _ = make_sales_tensor(data.train, data.stores)
    assert tensor.dtype == torch.float64  # type: ignore[attr-defined]


def test_store_data_len(data: StoreSalesData) -> None:
    tensor, _ = make_sales_tensor(data.train, data.stores)
    window, output = 60, 16
    ds = StoreData(tensor, window_lags=window, output_lags=output)
    assert len(ds) == tensor.shape[0] - window - output


def test_store_data_item_shapes(data: StoreSalesData) -> None:
    tensor, _ = make_sales_tensor(data.train, data.stores)
    window, output = 30, 10
    ds = StoreData(tensor, window_lags=window, output_lags=output)
    x, y = ds[0]
    assert x.shape == (window, tensor.shape[1], tensor.shape[2])
    assert y.shape == (output, tensor.shape[1], tensor.shape[2])
