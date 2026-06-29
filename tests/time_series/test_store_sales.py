from pathlib import Path

import pandas as pd
import pytest
import torch

from time_series.store_sales import MSLELoss, StoreData

DATES = pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"])
STORE_NBRS = [1, 2]
FAMILIES = ["AUTOMOTIVE", "GROCERY I"]


# ---------------------------------------------------------------------------
# Integration tests — exercise the full constructor with real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds() -> StoreData:
    return StoreData()


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
    assert ds.sales_tensor.dtype == torch.float32  # type: ignore[attr-defined]


def test_len(ds: StoreData) -> None:
    assert len(ds) == ds.sales_tensor.shape[0] - ds.window_lags - ds.output_lags


def test_repr(ds: StoreData) -> None:
    r = repr(ds)
    assert "StoreData" in r
    assert str(ds.sales_tensor.shape) in r


def test_item_shapes(ds: StoreData) -> None:
    x, y = ds[0]
    _, num_stores, num_families = ds.sales_tensor.shape
    assert x.shape == (ds.window_lags, num_stores, num_families)
    assert y.shape == (ds.output_lags, num_stores, num_families)


# ---------------------------------------------------------------------------
# Unit tests — individual loaders, use synthetic data_dir
# ---------------------------------------------------------------------------


def test_load_train_date_index(mock_data_dir: Path) -> None:
    assert isinstance(StoreData._load_train(mock_data_dir).index, pd.DatetimeIndex)


def test_load_stores_index(mock_data_dir: Path) -> None:
    assert StoreData._load_stores(mock_data_dir).index.name == "store_nbr"


# ---------------------------------------------------------------------------
# Unit tests — _setup_tensor, use synthetic DataFrames
# ---------------------------------------------------------------------------


def test_setup_tensor_shape(
    mock_train: pd.DataFrame, mock_stores: pd.DataFrame
) -> None:
    tensor, families = StoreData._setup_tensor(mock_train, mock_stores)
    assert tensor.shape == (len(DATES), len(STORE_NBRS), len(FAMILIES))
    assert list(families) == sorted(FAMILIES)


def test_setup_tensor_copy_is_writable(
    mock_train: pd.DataFrame, mock_stores: pd.DataFrame
) -> None:
    tensor, _ = StoreData._setup_tensor(mock_train, mock_stores, copy=True)
    tensor[0, 0, 0] = -1.0  # would raise or corrupt if non-writable


def test_custom_lags(mock_data_dir: Path) -> None:
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    x, y = ds[0]
    assert x.shape[0] == 1
    assert y.shape[0] == 1


# ---------------------------------------------------------------------------
# Unit tests — MSLELoss
# ---------------------------------------------------------------------------


def test_msle_loss_zero_on_identical_inputs() -> None:
    loss = MSLELoss()
    x = torch.tensor([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    assert loss(x, x).item() == pytest.approx(0.0)


def test_msle_loss_matches_manual_formula() -> None:
    import math

    loss = MSLELoss()
    pred = torch.tensor([1.0, 2.0])  # type: ignore[attr-defined]
    target = torch.tensor([2.0, 3.0])  # type: ignore[attr-defined]
    pairs = [(1.0, 2.0), (2.0, 3.0)]
    expected = sum((math.log1p(p) - math.log1p(t)) ** 2 for p, t in pairs) / len(pairs)
    assert loss(pred, target).item() == pytest.approx(expected)


def test_msle_loss_reduction_sum() -> None:
    loss_mean = MSLELoss(reduction="mean")
    loss_sum = MSLELoss(reduction="sum")
    pred = torch.tensor([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    target = torch.tensor([2.0, 3.0, 4.0])  # type: ignore[attr-defined]
    assert loss_sum(pred, target).item() == pytest.approx(
        loss_mean(pred, target).item() * 3
    )
