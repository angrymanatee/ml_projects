from pathlib import Path

import pandas as pd
import pytest
import torch

from time_series.store_sales import StoreData

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

DATES = pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"])
STORE_NBRS = [1, 2]
FAMILIES = ["AUTOMOTIVE", "GROCERY I"]


@pytest.fixture(scope="module")
def mock_train() -> pd.DataFrame:
    """Minimal train DataFrame: 3 dates × 2 stores × 2 families."""
    rows = [
        (date, store, family, float(i), 0)
        for i, (date, store, family) in enumerate(
            (d, s, f) for d in DATES for s in STORE_NBRS for f in FAMILIES
        )
    ]
    df = pd.DataFrame(
        rows, columns=["date", "store_nbr", "family", "sales", "onpromotion"]
    )
    return df.set_index(pd.DatetimeIndex(df.pop("date"), name="date"))


@pytest.fixture(scope="module")
def mock_stores() -> pd.DataFrame:
    return pd.DataFrame(
        {"city": ["Quito", "Guayaquil"], "type": ["D", "D"]},
        index=pd.Index(STORE_NBRS, name="store_nbr"),
    )


@pytest.fixture(scope="module")
def mock_data_dir(
    tmp_path_factory: pytest.TempPathFactory,
    mock_train: pd.DataFrame,
    mock_stores: pd.DataFrame,
) -> Path:
    """Temp directory populated with all CSVs StoreData.__init__ reads."""
    d = tmp_path_factory.mktemp("store_sales")

    mock_train.reset_index().to_csv(d / "train.csv", index=False)

    # test.csv shares the same schema as train
    mock_train.reset_index().to_csv(d / "test.csv", index=False)

    pd.DataFrame({"id": [0], "sales": [0.0]}).to_csv(
        d / "sample_submission.csv", index=False
    )

    mock_stores.reset_index().to_csv(d / "stores.csv", index=False)

    pd.DataFrame({"date": ["2013-01-01"], "dcoilwtico": [93.14]}).to_csv(
        d / "oil.csv", index=False
    )

    pd.DataFrame(
        {
            "date": ["2013-01-01"],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New Year"],
            "transferred": [False],
        }
    ).to_csv(d / "holidays_events.csv", index=False)

    return d


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
    assert ds.sales_tensor.dtype == torch.float64  # type: ignore[attr-defined]


def test_len(ds: StoreData) -> None:
    assert len(ds) == ds.sales_tensor.shape[0] - ds.window_lags - ds.output_lags


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
