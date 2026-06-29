"""Tests for StoreSalesEncoderOnly, PoolingMode, and train_and_eval."""

import math
import unittest.mock
from pathlib import Path

import pytest
import torch
from torch import nn

from common.modules import GetLastIndex
from time_series.main_store_sales_encoder_only import (
    PoolingMode,
    StoreSalesEncoderOnly,
    train_and_eval,
)
from time_series.store_sales import StoreData, Trainer

N_STORES = 3
N_FAMILIES = 4
WINDOW_LAGS = 10
OUTPUT_LAGS = 2
BATCH = 6
D_MODEL = 16
NHEAD = 2
NUM_LAYERS = 1


# ---------------------------------------------------------------------------
# PoolingMode
# ---------------------------------------------------------------------------


class TestPoolingMode:
    def test_parse_all(self) -> None:
        assert PoolingMode.parse("all") == PoolingMode.ALL

    def test_parse_last(self) -> None:
        assert PoolingMode.parse("last") == PoolingMode.LAST

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(KeyError):
            PoolingMode.parse("invalid")

    def test_get_module_all_returns_flatten(self) -> None:
        module = PoolingMode.get_module(PoolingMode.ALL)
        assert isinstance(module, nn.Flatten)

    def test_get_module_last_returns_get_last_index(self) -> None:
        module = PoolingMode.get_module(PoolingMode.LAST)
        assert isinstance(module, GetLastIndex)


# ---------------------------------------------------------------------------
# StoreSalesEncoderOnly fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_last() -> StoreSalesEncoderOnly:
    return StoreSalesEncoderOnly(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        pooling_mode=PoolingMode.LAST,
    )


@pytest.fixture(scope="module")
def model_all() -> StoreSalesEncoderOnly:
    return StoreSalesEncoderOnly(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        pooling_mode=PoolingMode.ALL,
    )


def _rand_input(batch: int = BATCH, seq: int = WINDOW_LAGS) -> torch.Tensor:
    return torch.rand(batch, seq, N_STORES, N_FAMILIES)


# ---------------------------------------------------------------------------
# StoreSalesEncoderOnly — output shape
# ---------------------------------------------------------------------------


def test_output_shape_pooling_last(model_last: StoreSalesEncoderOnly) -> None:
    out = model_last(_rand_input())
    assert out.shape == (BATCH, OUTPUT_LAGS, N_STORES, N_FAMILIES)


def test_output_shape_pooling_all(model_all: StoreSalesEncoderOnly) -> None:
    out = model_all(_rand_input())
    assert out.shape == (BATCH, OUTPUT_LAGS, N_STORES, N_FAMILIES)


def test_output_shape_batch_size_one(model_last: StoreSalesEncoderOnly) -> None:
    out = model_last(_rand_input(batch=1))
    assert out.shape == (1, OUTPUT_LAGS, N_STORES, N_FAMILIES)


# ---------------------------------------------------------------------------
# StoreSalesEncoderOnly — value properties
# ---------------------------------------------------------------------------


def test_output_non_negative(model_last: StoreSalesEncoderOnly) -> None:
    """ReLU at the end must never produce negative values."""
    out = model_last(_rand_input())
    assert (out >= 0).all()


def test_no_nan_in_output(model_last: StoreSalesEncoderOnly) -> None:
    out = model_last(_rand_input())
    assert not torch.isnan(out).any()


def test_output_dtype_matches_input(model_last: StoreSalesEncoderOnly) -> None:
    x = _rand_input().to(torch.float32)
    assert model_last(x).dtype == torch.float32


def test_different_inputs_produce_different_outputs(
    model_last: StoreSalesEncoderOnly,
) -> None:
    x1 = _rand_input()
    x2 = _rand_input()
    assert not torch.equal(model_last(x1), model_last(x2))


# ---------------------------------------------------------------------------
# train_and_eval
# ---------------------------------------------------------------------------


def test_train_and_eval_returns_finite_loss(mock_data_dir: Path) -> None:
    store_data = StoreData(
        window_lags=1, output_lags=1, data_dir=mock_data_dir, dtype=torch.float32
    )
    config = {
        "lr": 1e-3,
        "d_model": 4,
        "nhead": 1,
        "num_layers": 1,
        "batch_size": 1,
        "pooling_mode": PoolingMode.LAST,
    }
    fixed_loss = torch.tensor(0.5)
    # DataLoader is mocked because any split of the 1-sample mock dataset produces
    # an empty Subset, which causes DataLoader to raise before mocked methods are reached.
    with (
        unittest.mock.patch("mlflow.log_metrics"),
        unittest.mock.patch("mlflow.set_tag"),
        unittest.mock.patch("mlflow.log_artifact"),
        unittest.mock.patch("mlflow.pytorch.autolog"),
        unittest.mock.patch("time_series.main_store_sales_encoder_only.DataLoader"),
        unittest.mock.patch.object(Trainer, "train_loop", return_value=fixed_loss),
        unittest.mock.patch.object(Trainer, "val_loop", return_value=fixed_loss),
        unittest.mock.patch.object(Trainer, "_checkpoint"),
    ):
        val_loss, model, val_loader = train_and_eval(
            config=config,
            store_data=store_data,
            device=torch.device("cpu"),
            epochs=1,
            split=0.5,
        )

    assert isinstance(val_loss, float)
    assert math.isfinite(val_loss)
    assert val_loss >= 0.0
