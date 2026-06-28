"""Tests for StoreSalesEncoderOnly and PoolingMode."""

import pytest
import torch
from torch import nn

from common.modules import GetLastIndex
from time_series.main_store_sales_encoder_only import PoolingMode, StoreSalesEncoderOnly

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
