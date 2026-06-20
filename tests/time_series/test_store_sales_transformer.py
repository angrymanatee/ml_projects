"""Tests for StoreSalesTransformer and Trainer."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import mlflow
from time_series.main_store_sales_transformer import StoreSalesTransformer
from time_series.store_sales import MSLELoss, Trainer

CPU = torch.device("cpu")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_STORES = 3
N_FAMILIES = 4
WINDOW_LAGS = 5
OUTPUT_LAGS = 2
BATCH = 8


@pytest.fixture(scope="module")
def model() -> StoreSalesTransformer:
    return StoreSalesTransformer(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model=16,
        nhead=2,
    )


@pytest.fixture(scope="module")
def tiny_loaders() -> tuple[DataLoader, DataLoader]:
    """Two small synthetic DataLoaders with shape [T, n_stores, n_families]."""
    x = torch.rand(20, WINDOW_LAGS, N_STORES, N_FAMILIES)
    y = torch.rand(20, OUTPUT_LAGS, N_STORES, N_FAMILIES)
    train_loader: DataLoader = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=4)
    val_loader: DataLoader = DataLoader(TensorDataset(x[16:], y[16:]), batch_size=4)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# StoreSalesTransformer
# ---------------------------------------------------------------------------


def test_output_shape(model: StoreSalesTransformer) -> None:
    x = torch.rand(BATCH, WINDOW_LAGS, N_STORES, N_FAMILIES)
    out = model(x)
    assert out.shape == (BATCH, OUTPUT_LAGS, N_STORES, N_FAMILIES)


def test_output_non_negative(model: StoreSalesTransformer) -> None:
    """ReLU output gate must never produce negative values."""
    x = torch.rand(BATCH, WINDOW_LAGS, N_STORES, N_FAMILIES)
    out = model(x)
    assert (out >= 0).all()


def test_output_shape_batch_size_one(model: StoreSalesTransformer) -> None:
    x = torch.rand(1, WINDOW_LAGS, N_STORES, N_FAMILIES)
    out = model(x)
    assert out.shape == (1, OUTPUT_LAGS, N_STORES, N_FAMILIES)


def test_output_dtype_matches_input(model: StoreSalesTransformer) -> None:
    x = torch.rand(
        BATCH,
        WINDOW_LAGS,
        N_STORES,
        N_FAMILIES,
        dtype=torch.float32,
    )
    out = model(x)
    assert out.dtype == x.dtype


def test_no_nan_in_output(model: StoreSalesTransformer) -> None:
    x = torch.rand(BATCH, WINDOW_LAGS, N_STORES, N_FAMILIES)
    out = model(x)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer(
    model: StoreSalesTransformer,
    tiny_loaders: tuple[DataLoader, DataLoader],
    tmp_path: Path,
) -> Trainer:
    train_loader, val_loader = tiny_loaders
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    return Trainer(
        model=model,
        device=CPU,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        loss_func=MSLELoss(),
    )


def test_trainer_train_loop_returns_tensor(trainer: Trainer) -> None:
    with mlflow.start_run():
        loss = trainer.train_loop(epoch_idx=0)
    assert isinstance(loss, Tensor)
    assert loss.ndim == 0


def test_trainer_val_loop_returns_scalar(trainer: Trainer) -> None:
    with mlflow.start_run():
        loss = trainer.val_loop(epoch_idx=0)
    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_trainer_val_loss_finite(trainer: Trainer) -> None:
    with mlflow.start_run():
        loss = trainer.val_loop(epoch_idx=0)
    assert torch.isfinite(loss)


def test_trainer_smoke_two_epochs(
    tiny_loaders: tuple[DataLoader, DataLoader],
    tmp_path: Path,
) -> None:
    """Two full epochs must complete without error."""
    train_loader, val_loader = tiny_loaders
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    fresh_model = StoreSalesTransformer(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model=16,
        nhead=2,
    )
    t = Trainer(
        model=fresh_model,
        device=CPU,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    with mlflow.start_run(), patch.object(t, "_checkpoint"):
        t.train(epochs=2)


def test_trainer_default_loss_is_msle(
    tiny_loaders: tuple[DataLoader, DataLoader],
    tmp_path: Path,
) -> None:
    train_loader, val_loader = tiny_loaders
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    model = StoreSalesTransformer(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model=16,
        nhead=2,
    )
    t = Trainer(
        model=model, device=CPU, train_loader=train_loader, val_loader=val_loader
    )
    assert isinstance(t.loss_func, MSLELoss)


def test_trainer_custom_loss(
    tiny_loaders: tuple[DataLoader, DataLoader],
    tmp_path: Path,
) -> None:
    train_loader, val_loader = tiny_loaders
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    model = StoreSalesTransformer(
        n_stores=N_STORES,
        n_families=N_FAMILIES,
        n_output_steps=OUTPUT_LAGS,
        d_model=16,
        nhead=2,
    )
    custom_loss = nn.MSELoss()
    t = Trainer(
        model=model,
        device=CPU,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_func=custom_loss,
    )
    assert t.loss_func is custom_loss
