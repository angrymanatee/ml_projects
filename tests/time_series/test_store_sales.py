import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from time_series.store_sales import EarthquakeEncoding, MSLELoss, StoreData

DATES = pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"])
STORE_NBRS = [1, 2]
FAMILIES = ["AUTOMOTIVE", "GROCERY I"]

# Dates spanning the Ecuador earthquake (2016-04-16) for spot-check tests.
_QUAKE_DATES = pd.DatetimeIndex(
    ["2016-04-14", "2016-04-16", "2016-04-17", "2016-05-16"]
)


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
    assert f"n_families={len(ds.families)}" in r
    assert f"window_lags={ds.window_lags}" in r


def test_item_shapes(ds: StoreData) -> None:
    x, y = ds[0]
    _, num_stores, num_families = ds.sales_tensor.shape
    assert x.shape == (ds.window_lags, num_stores, num_families + ds.n_date_features)
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


def test_setup_promotion_tensor_shape(
    mock_train: pd.DataFrame, mock_stores: pd.DataFrame
) -> None:
    tensor = StoreData._setup_promotion_tensor(mock_train, mock_stores)
    assert tensor.shape == (len(DATES), len(STORE_NBRS), len(FAMILIES))


def test_include_onpromotion_adds_families(mock_data_dir: Path) -> None:
    no_dates = {
        "date_features": False,
        "payday_features": False,
        "earthquake_encoding": None,
    }
    ds_base = StoreData(
        window_lags=1, output_lags=1, data_dir=mock_data_dir, **no_dates
    )
    ds_promo = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        include_onpromotion=True,
        **no_dates,
    )
    x_base, _ = ds_base[0]
    x_promo, y_promo = ds_promo[0]
    assert x_promo.shape[-1] == x_base.shape[-1] + len(FAMILIES)
    assert y_promo.shape[-1] == len(FAMILIES)


def test_exclude_onpromotion_default(mock_data_dir: Path) -> None:
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    assert ds.promotion_tensor is None
    x, _ = ds[0]
    assert x.shape[-1] == len(FAMILIES) + ds.n_date_features


# ---------------------------------------------------------------------------
# Unit tests — _setup_date_features
# ---------------------------------------------------------------------------


def test_setup_date_features_default_shape() -> None:
    # All three groups on: 5 (date) + 2 (payday) + 1 (earthquake) = 8
    tensor = StoreData._setup_date_features(_QUAKE_DATES)
    assert tensor.shape == (len(_QUAKE_DATES), 8)


def test_setup_date_features_all_disabled() -> None:
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES,
        date_features=False,
        payday_features=False,
        earthquake_encoding=None,
    )
    assert tensor.shape == (len(_QUAKE_DATES), 0)


def test_setup_date_features_date_only() -> None:
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES, payday_features=False, earthquake_encoding=None
    )
    assert tensor.shape == (len(_QUAKE_DATES), 5)


def test_setup_date_features_payday_only() -> None:
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES, date_features=False, earthquake_encoding=None
    )
    assert tensor.shape == (len(_QUAKE_DATES), 2)


def test_setup_date_features_earthquake_decay_only() -> None:
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES,
        date_features=False,
        payday_features=False,
        earthquake_encoding=EarthquakeEncoding.DECAY,
    )
    assert tensor.shape == (len(_QUAKE_DATES), 1)


def test_setup_date_features_earthquake_linear_only() -> None:
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES,
        date_features=False,
        payday_features=False,
        earthquake_encoding=EarthquakeEncoding.LINEAR,
    )
    assert tensor.shape == (len(_QUAKE_DATES), 1)


def test_setup_date_features_days_since_epoch() -> None:
    dates = pd.DatetimeIndex(["2013-01-01", "2013-01-02", "2013-01-05"])
    tensor = StoreData._setup_date_features(
        dates, payday_features=False, earthquake_encoding=None
    )
    # Col 0 is days since the first date in the index.
    assert tensor[0, 0].item() == pytest.approx(0.0)
    assert tensor[1, 0].item() == pytest.approx(1.0)
    assert tensor[2, 0].item() == pytest.approx(4.0)


def test_setup_date_features_dow_sincos() -> None:
    # 2013-01-01 is a Tuesday (dayofweek=1).
    dates = pd.DatetimeIndex(["2013-01-01"])
    tensor = StoreData._setup_date_features(
        dates, payday_features=False, earthquake_encoding=None
    )
    two_pi = 2.0 * math.pi
    assert tensor[0, 1].item() == pytest.approx(math.sin(two_pi * 1 / 7))
    assert tensor[0, 2].item() == pytest.approx(math.cos(two_pi * 1 / 7))


def test_setup_date_features_is_15th() -> None:
    dates = pd.DatetimeIndex(["2013-01-14", "2013-01-15", "2013-01-16"])
    tensor = StoreData._setup_date_features(
        dates, date_features=False, earthquake_encoding=None
    )
    assert tensor[0, 0].item() == pytest.approx(0.0)
    assert tensor[1, 0].item() == pytest.approx(1.0)
    assert tensor[2, 0].item() == pytest.approx(0.0)


def test_setup_date_features_is_month_end() -> None:
    dates = pd.DatetimeIndex(["2013-01-30", "2013-01-31", "2013-02-01"])
    tensor = StoreData._setup_date_features(
        dates, date_features=False, earthquake_encoding=None
    )
    assert tensor[0, 1].item() == pytest.approx(0.0)
    assert tensor[1, 1].item() == pytest.approx(1.0)
    assert tensor[2, 1].item() == pytest.approx(0.0)


def test_setup_date_features_earthquake_decay_values() -> None:
    tau = 30.0
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES,
        date_features=False,
        payday_features=False,
        earthquake_encoding=EarthquakeEncoding.DECAY,
        earthquake_tau=tau,
    )
    col = tensor[:, 0]
    assert col[0].item() == pytest.approx(0.0)  # 2 days before: 0
    assert col[1].item() == pytest.approx(1.0)  # earthquake day: exp(0)
    assert col[2].item() == pytest.approx(math.exp(-1 / tau))  # 1 day after
    assert col[3].item() == pytest.approx(math.exp(-30 / tau))  # 30 days after


def test_setup_date_features_earthquake_linear_values() -> None:
    tensor = StoreData._setup_date_features(
        _QUAKE_DATES,
        date_features=False,
        payday_features=False,
        earthquake_encoding=EarthquakeEncoding.LINEAR,
    )
    col = tensor[:, 0]
    assert col[0].item() == pytest.approx(0.0)  # before: 0
    assert col[1].item() == pytest.approx(0.0)  # earthquake day: 0 / 365
    assert col[2].item() == pytest.approx(1 / 365)  # 1 day after
    assert col[3].item() == pytest.approx(30 / 365)  # 30 days after


# ---------------------------------------------------------------------------
# Unit tests — StoreData date feature integration via mock_data_dir
# ---------------------------------------------------------------------------


def test_n_date_features_default(mock_data_dir: Path) -> None:
    # Default: all three groups enabled → 8 features.
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    assert ds.n_date_features == 8


def test_n_date_features_none(mock_data_dir: Path) -> None:
    ds = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        date_features=False,
        payday_features=False,
        earthquake_encoding=None,
    )
    assert ds.n_date_features == 0


def test_item_input_includes_date_features(mock_data_dir: Path) -> None:
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    x, y = ds[0]
    _, n_stores, n_families = ds.sales_tensor.shape
    assert x.shape == (1, n_stores, n_families + ds.n_date_features)
    assert y.shape == (1, n_stores, n_families)


def test_item_no_date_features_matches_sales_shape(mock_data_dir: Path) -> None:
    ds = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        date_features=False,
        payday_features=False,
        earthquake_encoding=None,
    )
    x, y = ds[0]
    _, n_stores, n_families = ds.sales_tensor.shape
    assert x.shape == (1, n_stores, n_families)
    assert y.shape == (1, n_stores, n_families)


def test_date_features_tensor_shape(mock_data_dir: Path) -> None:
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    T = ds.sales_tensor.shape[0]
    assert ds.date_features_tensor.shape == (T, ds.n_date_features)


# ---------------------------------------------------------------------------
# Unit tests — oil tensor setup and include_oil flag
# ---------------------------------------------------------------------------


def test_oil_tensor_none_by_default(ds: StoreData) -> None:
    assert ds.oil_tensor is None


def test_oil_tensor_shape() -> None:
    ds_oil = StoreData(include_oil=True)
    num_dates = ds_oil.train.index.nunique()
    assert ds_oil.oil_tensor is not None
    assert ds_oil.oil_tensor.shape == (num_dates,)


def test_oil_tensor_no_nans() -> None:
    ds_oil = StoreData(include_oil=True)
    assert ds_oil.oil_tensor is not None
    assert not torch.isnan(ds_oil.oil_tensor).any()  # type: ignore[attr-defined]


def test_item_shapes_with_oil_enabled(mock_data_dir: Path) -> None:
    ds_oil = StoreData(
        window_lags=1, output_lags=1, data_dir=mock_data_dir, include_oil=True
    )
    x, y = ds_oil[0]
    _, n_stores, n_families = ds_oil.sales_tensor.shape
    # input: sales + date_features + 1 oil channel; target unchanged
    assert x.shape == (1, n_stores, n_families + ds_oil.n_date_features + 1)
    assert y.shape == (1, n_stores, n_families)


def test_item_shapes_oil_no_date_features(mock_data_dir: Path) -> None:
    ds_oil = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        date_features=False,
        payday_features=False,
        earthquake_encoding=None,
        include_oil=True,
    )
    x, y = ds_oil[0]
    _, n_stores, n_families = ds_oil.sales_tensor.shape
    assert x.shape == (1, n_stores, n_families + 1)
    assert y.shape == (1, n_stores, n_families)


def test_setup_oil_tensor_ffill(mock_train: pd.DataFrame) -> None:
    """A leading NaN should be back-filled; mid-series NaN should be forward-filled."""
    oil_df = pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "dcoilwtico": [float("nan"), 93.14, float("nan")],
        },
    )
    oil_df = oil_df.set_index(pd.to_datetime(oil_df["date"]))
    oil_tensor = StoreData._setup_oil_tensor(oil_df, mock_train)
    assert oil_tensor.shape == (3,)
    assert not torch.isnan(oil_tensor).any()  # type: ignore[attr-defined]
    # 2013-01-01: bfilled from 2013-01-02 → 93.14
    assert oil_tensor[0].item() == pytest.approx(93.14)
    # 2013-01-03: ffilled from 2013-01-02 → 93.14
    assert oil_tensor[2].item() == pytest.approx(93.14)


# ---------------------------------------------------------------------------
# Unit tests — MSLELoss
# ---------------------------------------------------------------------------


def test_msle_loss_zero_on_identical_inputs() -> None:
    loss = MSLELoss()
    x = torch.tensor([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    assert loss(x, x).item() == pytest.approx(0.0)


def test_msle_loss_matches_manual_formula() -> None:
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
