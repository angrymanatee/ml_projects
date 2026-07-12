import numpy as np
import pandas as pd
import pytest

from time_series.store_sales.backtest import (
    BacktestConfig,
    generate_fold_cutoffs,
    rmsle,
)


def test_rmsle_zero_for_perfect_prediction() -> None:
    y = np.array([0.0, 5.0, 100.0])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_known_value() -> None:
    # single element: |log1p(3) - log1p(7)| = |1.386294 - 2.079442| = 0.693147
    assert rmsle(np.array([7.0]), np.array([3.0])) == pytest.approx(0.6931471, abs=1e-6)


def test_rmsle_clips_negative_predictions_to_zero() -> None:
    # prediction -4 is clipped to 0, so error vs true 0 is log1p(0)-log1p(0)=0
    assert rmsle(np.array([0.0]), np.array([-4.0])) == pytest.approx(0.0)


def test_fold_cutoffs_disjoint_blocks_march_backward() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    config = BacktestConfig(n_folds=3, horizon=16, min_train_days=1)
    cutoffs = generate_fold_cutoffs(dates, config)
    # last date is 2020-04-09 (index 99). Fold 1 cutoff = last - 16 days.
    assert len(cutoffs) == 3
    assert cutoffs == sorted(cutoffs)  # ascending
    # disjoint: consecutive cutoffs are exactly `horizon` days apart
    assert (cutoffs[1] - cutoffs[0]).days == 16
    assert (cutoffs[2] - cutoffs[1]).days == 16
    # last cutoff leaves exactly `horizon` days of predict block at the end
    assert (dates[-1] - cutoffs[-1]).days == 16


def test_fold_cutoffs_respects_min_train_days() -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    # min_train_days=40 means a cutoff needs >=40 dates before it.
    config = BacktestConfig(n_folds=5, horizon=16, min_train_days=40)
    cutoffs = generate_fold_cutoffs(dates, config)
    for cutoff in cutoffs:
        n_before = (dates <= cutoff).sum()
        assert n_before >= 40
