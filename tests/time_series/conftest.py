"""Shared fixtures for time_series tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DATES = pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"])
_STORE_NBRS = [1, 2]
_FAMILIES = ["AUTOMOTIVE", "GROCERY I"]


def write_store_sales_csvs(
    directory: Path,
    *,
    train: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
) -> None:
    """Write the six CSVs StoreData.__init__ reads (single source for the file set).

    train/stores/oil/holidays are already column-shaped (no index written). test.csv
    reuses train so StoreData's future-frame load has the same schema.
    """
    train.to_csv(directory / "train.csv", index=False)
    train.to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"id": [0], "sales": [0.0]}).to_csv(
        directory / "sample_submission.csv", index=False
    )
    stores.to_csv(directory / "stores.csv", index=False)
    oil.to_csv(directory / "oil.csv", index=False)
    holidays.to_csv(directory / "holidays_events.csv", index=False)


def _national_new_year(first_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [first_date],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New Year"],
            "transferred": [False],
        }
    )


@pytest.fixture(scope="module")
def mock_train() -> pd.DataFrame:
    """Minimal train DataFrame: 3 dates × 2 stores × 2 families."""
    rows = [
        (date, store, family, float(i), 0)
        for i, (date, store, family) in enumerate(
            (d, s, f) for d in _DATES for s in _STORE_NBRS for f in _FAMILIES
        )
    ]
    df = pd.DataFrame(
        rows, columns=pd.Index(["date", "store_nbr", "family", "sales", "onpromotion"])
    )
    return df.set_index(pd.DatetimeIndex(df.pop("date"), name="date"))


@pytest.fixture(scope="module")
def mock_stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["D", "D"],
            "cluster": [1, 2],
        },
        index=pd.Index(_STORE_NBRS, name="store_nbr"),
    )


@pytest.fixture(scope="module")
def mock_data_dir(
    tmp_path_factory: pytest.TempPathFactory,
    mock_train: pd.DataFrame,
    mock_stores: pd.DataFrame,
) -> Path:
    """Temp directory populated with all CSVs StoreData.__init__ reads."""
    directory = tmp_path_factory.mktemp("store_sales")
    write_store_sales_csvs(
        directory,
        train=mock_train.reset_index(),
        stores=mock_stores.reset_index(),
        oil=pd.DataFrame(
            {
                "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
                "dcoilwtico": [93.14, 93.20, 93.08],
            }
        ),
        holidays=_national_new_year("2013-01-01"),
    )
    return directory


@pytest.fixture(scope="module")
def long_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ~40-day single-series dataset, enough for a multi-fold backtest."""
    directory = tmp_path_factory.mktemp("store_sales_long")
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    train = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": 10.0 + np.arange(40) + 3.0 * (np.arange(40) % 7),  # non-constant
            "onpromotion": 0,
        }
    )
    stores = pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["D"],
            "cluster": [1],
        }
    )
    write_store_sales_csvs(
        directory,
        train=train,
        stores=stores,
        oil=pd.DataFrame(
            {
                "date": dates.astype(str),
                "dcoilwtico": 90.0 + np.arange(len(dates)) * 0.1,
            }
        ),
        holidays=_national_new_year("2020-01-01"),
    )
    return directory
