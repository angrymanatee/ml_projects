import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from time_series.store_sales import (
    EarthquakeEncoding,
    MSLELoss,
    StoreData,
)

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
    n_time_steps = ds.sales_tensor.shape[0]
    assert ds.date_features_tensor.shape == (n_time_steps, ds.n_date_features)


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
# Unit tests — store feature tensor and __getitem__ integration
# ---------------------------------------------------------------------------


def test_store_feature_tensor_shape(mock_data_dir: Path) -> None:
    ds = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        date_features=False,
        payday_features=False,
        earthquake_encoding=None,
        store_feature_cols=["type"],
    )
    n_stores = ds.stores.shape[0]
    assert ds.store_feature_tensor is not None
    # type has 1 unique value ("D") → 1 one-hot column
    assert ds.store_feature_tensor.shape == (n_stores, 1)


def test_store_feature_tensor_none_by_default(mock_data_dir: Path) -> None:
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    assert ds.store_feature_tensor is None
    assert ds.n_store_features == 0


def test_item_shapes_with_store_features(mock_data_dir: Path) -> None:
    no_dates = {
        "date_features": False,
        "payday_features": False,
        "earthquake_encoding": None,
    }
    ds_base = StoreData(
        window_lags=1, output_lags=1, data_dir=mock_data_dir, **no_dates
    )
    ds_store = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        store_feature_cols=["type"],
        **no_dates,
    )
    x_base, _ = ds_base[0]
    x_store, y_store = ds_store[0]
    # type has 1 unique value → 1 one-hot column appended
    assert x_store.shape[-1] == x_base.shape[-1] + 1
    assert y_store.shape[-1] == len(FAMILIES)


def test_store_features_broadcast_across_time(mock_data_dir: Path) -> None:
    no_dates = {
        "date_features": False,
        "payday_features": False,
        "earthquake_encoding": None,
    }
    ds = StoreData(
        window_lags=2,
        output_lags=1,
        data_dir=mock_data_dir,
        store_feature_cols=["type"],
        **no_dates,
    )
    x, _ = ds[0]
    # store feature columns should be identical across time steps
    assert (x[0, :, -1] == x[1, :, -1]).all()


def test_n_store_features_property(mock_data_dir: Path) -> None:
    no_dates = {
        "date_features": False,
        "payday_features": False,
        "earthquake_encoding": None,
    }
    ds = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        store_feature_cols=["type"],
        **no_dates,
    )
    assert ds.n_store_features == 1


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


# ---------------------------------------------------------------------------
# Unit tests — _setup_holiday_tensor (static method)
# ---------------------------------------------------------------------------

# Five-day span used across holiday unit tests.
_H_DATES = pd.to_datetime(
    ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"]
)

# Store 1: Quito / Pichincha.  Store 2: Cuenca / Azuay.
_H_STORES = pd.DataFrame(
    {"city": ["Quito", "Cuenca"], "state": ["Pichincha", "Azuay"]},
    index=pd.Index([1, 2], name="store_nbr"),
)


def _make_train(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimal train DataFrame with a single store/family for a given date range."""
    rows = [(date, 1, "GROCERY I", 0.0, 0) for date in dates]
    df = pd.DataFrame(
        rows, columns=pd.Index(["date", "store_nbr", "family", "sales", "onpromotion"])
    )
    return df.set_index(pd.DatetimeIndex(df.pop("date"), name="date"))


def _make_holidays(rows: list[tuple]) -> pd.DataFrame:
    """Build a holidays DataFrame with a DatetimeIndex from (date, type, locale,
    locale_name, description, transferred) row tuples."""
    df = pd.DataFrame(
        rows,
        columns=pd.Index(
            ["date", "type", "locale", "locale_name", "description", "transferred"]
        ),
    )
    return df.set_index(pd.DatetimeIndex(pd.to_datetime(df.pop("date")), name="date"))


@pytest.fixture()
def h_train() -> pd.DataFrame:
    return _make_train(_H_DATES)


@pytest.fixture()
def h_holidays() -> pd.DataFrame:
    """Holidays covering every feature type plus a transferred/Transfer pair."""
    return _make_holidays(
        [
            # national_holiday (not transferred) — 2013-01-01
            ("2013-01-01", "Holiday", "National", "Ecuador", "New Year", False),
            # transferred Holiday — 2013-01-02 should NOT be flagged
            ("2013-01-02", "Holiday", "National", "Ecuador", "Moved Day", True),
            # Transfer row — 2013-01-03 IS the actual celebration
            ("2013-01-03", "Transfer", "National", "Ecuador", "Moved Day", False),
            # Event — 2013-01-04
            ("2013-01-04", "Event", "National", "Ecuador", "Election", False),
            # Bridge — 2013-01-05
            ("2013-01-05", "Bridge", "National", "Ecuador", "Bridge Day", False),
            # Work Day — 2013-01-05 (same date as bridge is fine for independent tests)
            ("2013-01-05", "Work Day", "National", "Ecuador", "Make-up Day", False),
            # Additional — 2013-01-01
            ("2013-01-01", "Additional", "National", "Ecuador", "Extra Holiday", False),
            # Regional holiday for Pichincha (store 1 only) — 2013-01-02
            ("2013-01-02", "Holiday", "Regional", "Pichincha", "Pichincha Day", False),
            # Local holiday for Cuenca (store 2 only) — 2013-01-03
            ("2013-01-03", "Holiday", "Local", "Cuenca", "Cuenca Day", False),
        ]
    )


def test_setup_holiday_tensor_empty_returns_none(
    h_train: pd.DataFrame,
) -> None:
    result = StoreData._setup_holiday_tensor(_make_holidays([]), h_train, _H_STORES, [])
    assert result is None


def test_holiday_tensor_none_by_default(mock_data_dir: Path) -> None:
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    assert ds.holiday_tensor is None
    assert ds.n_holiday_features == 0


def test_setup_holiday_tensor_shape(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    n_features = 3
    tensor = StoreData._setup_holiday_tensor(
        h_holidays,
        h_train,
        _H_STORES,
        ["national_holiday", "event", "bridge"],
    )
    assert tensor is not None
    assert tensor.shape == (len(_H_DATES), len(_H_STORES), n_features)


def test_setup_holiday_tensor_dtype(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["national_holiday"]
    )
    assert tensor is not None
    assert tensor.dtype == torch.float32  # type: ignore[attr-defined]


def test_national_holiday_active_on_non_transferred(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    # 2013-01-01: type=Holiday, national, not transferred → 1.0
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["national_holiday"]
    )
    assert tensor is not None
    assert tensor[0, :, 0].tolist() == [1.0, 1.0]


def test_national_holiday_inactive_on_transferred(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    # 2013-01-02: type=Holiday but transferred=True → 0.0
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["national_holiday"]
    )
    assert tensor is not None
    assert tensor[1, :, 0].tolist() == [0.0, 0.0]


def test_transfer_row_is_active_as_holiday(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    # 2013-01-03: type=Transfer, national → actual celebration date → 1.0
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["national_holiday"]
    )
    assert tensor is not None
    assert tensor[2, :, 0].tolist() == [1.0, 1.0]


def test_national_holiday_broadcast_across_stores(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["national_holiday"]
    )
    assert tensor is not None
    # All stores share the same value on every date.
    assert (tensor[:, 0, 0] == tensor[:, 1, 0]).all()


def test_event_feature(h_holidays: pd.DataFrame, h_train: pd.DataFrame) -> None:
    tensor = StoreData._setup_holiday_tensor(h_holidays, h_train, _H_STORES, ["event"])
    assert tensor is not None
    expected = [0.0, 0.0, 0.0, 1.0, 0.0]
    assert tensor[:, 0, 0].tolist() == expected


def test_bridge_feature(h_holidays: pd.DataFrame, h_train: pd.DataFrame) -> None:
    tensor = StoreData._setup_holiday_tensor(h_holidays, h_train, _H_STORES, ["bridge"])
    assert tensor is not None
    expected = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert tensor[:, 0, 0].tolist() == expected


def test_work_day_feature(h_holidays: pd.DataFrame, h_train: pd.DataFrame) -> None:
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["work_day"]
    )
    assert tensor is not None
    expected = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert tensor[:, 0, 0].tolist() == expected


def test_additional_feature(h_holidays: pd.DataFrame, h_train: pd.DataFrame) -> None:
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["additional"]
    )
    assert tensor is not None
    expected = [1.0, 0.0, 0.0, 0.0, 0.0]
    assert tensor[:, 0, 0].tolist() == expected


def test_regional_holiday_per_store(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    # 2013-01-02: regional holiday for Pichincha → store 1 (Pichincha) = 1.0, store 2 (Azuay) = 0.0
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["regional_holiday"]
    )
    assert tensor is not None
    assert tensor[1, 0, 0].item() == pytest.approx(1.0)
    assert tensor[1, 1, 0].item() == pytest.approx(0.0)


def test_regional_holiday_broadcast_check(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    # Dates with no regional holiday should be 0.0 for all stores.
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["regional_holiday"]
    )
    assert tensor is not None
    assert tensor[0, :, 0].tolist() == [0.0, 0.0]  # 2013-01-01: no regional holiday


def test_local_holiday_per_store(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    # 2013-01-03: local holiday for Cuenca → store 1 (Quito) = 0.0, store 2 (Cuenca) = 1.0
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["local_holiday"]
    )
    assert tensor is not None
    assert tensor[2, 0, 0].item() == pytest.approx(0.0)
    assert tensor[2, 1, 0].item() == pytest.approx(1.0)


def test_multiple_features_channel_order(
    h_holidays: pd.DataFrame, h_train: pd.DataFrame
) -> None:
    tensor = StoreData._setup_holiday_tensor(
        h_holidays, h_train, _H_STORES, ["national_holiday", "event", "bridge"]
    )
    assert tensor is not None
    # Channel 0: national_holiday — 1.0 on 2013-01-01
    assert tensor[0, 0, 0].item() == pytest.approx(1.0)
    # Channel 1: event — 1.0 on 2013-01-04
    assert tensor[3, 0, 1].item() == pytest.approx(1.0)
    # Channel 2: bridge — 1.0 on 2013-01-05
    assert tensor[4, 0, 2].item() == pytest.approx(1.0)


def test_n_holiday_features_property(mock_data_dir: Path) -> None:
    ds = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        holiday_features=["national_holiday", "event"],
    )
    assert ds.n_holiday_features == 2


def test_holiday_channels_added_to_item(mock_data_dir: Path) -> None:
    no_dates = {
        "date_features": False,
        "payday_features": False,
        "earthquake_encoding": None,
    }
    ds_base = StoreData(
        window_lags=1, output_lags=1, data_dir=mock_data_dir, **no_dates
    )
    ds_hol = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        holiday_features=["national_holiday", "event"],
        **no_dates,
    )
    x_base, _ = ds_base[0]
    x_hol, y_hol = ds_hol[0]
    assert x_hol.shape[-1] == x_base.shape[-1] + 2
    assert y_hol.shape == (
        1,
        ds_hol.sales_tensor.shape[1],
        ds_hol.sales_tensor.shape[2],
    )


def test_holiday_tensor_shape_full_dataset(mock_data_dir: Path) -> None:
    ds = StoreData(
        window_lags=1,
        output_lags=1,
        data_dir=mock_data_dir,
        holiday_features=["national_holiday"],
    )
    assert ds.holiday_tensor is not None
    n_time_steps = ds.sales_tensor.shape[0]
    n_stores = ds.stores.shape[0]
    assert ds.holiday_tensor.shape == (n_time_steps, n_stores, 1)


def test_invalid_holiday_feature_raises(mock_data_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unknown holiday feature"):
        StoreData(
            window_lags=1,
            output_lags=1,
            data_dir=mock_data_dir,
            holiday_features=["bogus"],
        )
