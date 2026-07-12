"""Tests for StoreSalesHierarchicalEncoder and its component modules."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from time_series.store_sales import StoreData, StoreSalesHierarchicalEncoder
from time_series.store_sales.models import InputProjection, TimeToSF

N_STORES = 3
N_FAMILIES = 4
N_FEATURES = 5
WINDOW_LAGS = 8
OUTPUT_LAGS = 2
BATCH = 6
D_MODEL_TIME = 16
D_MODEL_SF = 16
NHEAD = 2
NUM_LAYERS = 1


def _det_input(
    batch: int = BATCH,
    seq: int = WINDOW_LAGS,
    n_features: int = N_FEATURES,
    offset: int = 0,
) -> torch.Tensor:
    """Return a deterministic tensor with sequential float values starting at offset."""
    total = batch * seq * N_STORES * N_FAMILIES * n_features
    return torch.arange(offset, offset + total, dtype=torch.float32).reshape(
        batch, seq, N_STORES, N_FAMILIES, n_features
    )


# ---------------------------------------------------------------------------
# InputTransform
# ---------------------------------------------------------------------------


def test_input_transform_output_shape() -> None:
    module = InputProjection(d_model=D_MODEL_TIME, max_length=512)
    x = _det_input()
    out = module(x)
    # [B, T, S, F, X] -> [B*S*F, T, d_model]
    assert out.shape == (BATCH * N_STORES * N_FAMILIES, WINDOW_LAGS, D_MODEL_TIME)


def test_input_transform_batch_size_one() -> None:
    module = InputProjection(d_model=D_MODEL_TIME, max_length=512)
    x = _det_input(batch=1)
    out = module(x)
    assert out.shape == (1 * N_STORES * N_FAMILIES, WINDOW_LAGS, D_MODEL_TIME)


# ---------------------------------------------------------------------------
# TimeToSF
# ---------------------------------------------------------------------------


def test_time_to_sf_output_shape() -> None:
    module = TimeToSF(
        n_stores=N_STORES, n_families=N_FAMILIES, d_time=D_MODEL_TIME, d_sf=D_MODEL_SF
    )
    # Input: [B*S*F, T, D_time]
    total = BATCH * N_STORES * N_FAMILIES * WINDOW_LAGS * D_MODEL_TIME
    x = torch.arange(total, dtype=torch.float32).reshape(
        BATCH * N_STORES * N_FAMILIES, WINDOW_LAGS, D_MODEL_TIME
    )
    out = module(x)
    # Expected: [B, S*F, d_sf]
    assert out.shape == (BATCH, N_STORES * N_FAMILIES, D_MODEL_SF)


# ---------------------------------------------------------------------------
# StoreSalesHierarchicalEncoder — output shape
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> StoreSalesHierarchicalEncoder:
    torch.manual_seed(0)
    return StoreSalesHierarchicalEncoder(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model_time=D_MODEL_TIME,
        nhead_time=NHEAD,
        num_layers_time=NUM_LAYERS,
        d_model_sf=D_MODEL_SF,
        nhead_sf=NHEAD,
        num_layers_sf=NUM_LAYERS,
        # Small kernel/stride so Conv1d fits within N_FEATURES=5 in unit tests.
        reduction_width=2,
        reduction_stride=1,
    )


def test_output_shape(model: StoreSalesHierarchicalEncoder) -> None:
    out = model(_det_input())
    assert out.shape == (BATCH, OUTPUT_LAGS, N_STORES, N_FAMILIES)


def test_output_shape_batch_size_one(model: StoreSalesHierarchicalEncoder) -> None:
    out = model(_det_input(batch=1))
    assert out.shape == (1, OUTPUT_LAGS, N_STORES, N_FAMILIES)


def test_output_non_negative(model: StoreSalesHierarchicalEncoder) -> None:
    assert (model(_det_input()) >= 0).all()


def test_no_nan_in_output(model: StoreSalesHierarchicalEncoder) -> None:
    assert not torch.isnan(model(_det_input())).any()


def test_output_dtype_matches_input(model: StoreSalesHierarchicalEncoder) -> None:
    x = _det_input().to(torch.float32)
    assert model(x).dtype == torch.float32


def test_different_inputs_produce_different_outputs(
    model: StoreSalesHierarchicalEncoder,
) -> None:
    was_training = model.training
    model.eval()
    try:
        total = BATCH * WINDOW_LAGS * N_STORES * N_FAMILIES * N_FEATURES
        with torch.no_grad():
            out_a = model(_det_input())
            out_b = model(_det_input(offset=total))
        assert not torch.equal(out_a, out_b)
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Integration — StoreData + DataLoader
# ---------------------------------------------------------------------------


def test_dataloader_compatible(mock_data_dir: Path) -> None:
    """Batches from StoreData should flow through the model without error."""
    ds = StoreData(window_lags=1, output_lags=1, data_dir=mock_data_dir)
    loader: DataLoader = DataLoader(ds, batch_size=2)
    x, _ = next(iter(loader))
    n_stores = ds.stores.shape[0]
    n_families = ds.families.size
    m = StoreSalesHierarchicalEncoder(
        n_stores=n_stores,
        n_families=n_families,
        n_output_steps=1,
        d_model_time=D_MODEL_TIME,
        nhead_time=NHEAD,
        num_layers_time=1,
        d_model_sf=D_MODEL_SF,
        nhead_sf=NHEAD,
        num_layers_sf=1,
        # reduction_width must be ≤ n_input_channels from StoreData (24 for mock data).
        # Use small values so the test is fast.
        reduction_width=2,
        reduction_stride=1,
    )
    out = m(x)
    assert out.shape == (x.shape[0], 1, n_stores, n_families)
