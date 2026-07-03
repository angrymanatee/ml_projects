"""Shared fixtures for time_series tests."""

from pathlib import Path

import pandas as pd
import pytest

_DATES = pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"])
_STORE_NBRS = [1, 2]
_FAMILIES = ["AUTOMOTIVE", "GROCERY I"]


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
    d = tmp_path_factory.mktemp("store_sales")

    mock_train.reset_index().to_csv(d / "train.csv", index=False)
    mock_train.reset_index().to_csv(d / "test.csv", index=False)

    pd.DataFrame({"id": [0], "sales": [0.0]}).to_csv(
        d / "sample_submission.csv", index=False
    )

    mock_stores.reset_index().to_csv(d / "stores.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "dcoilwtico": [93.14, 93.20, 93.08],
        }
    ).to_csv(d / "oil.csv", index=False)

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
