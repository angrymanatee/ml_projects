import numpy as np
import pandas as pd
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
import torch

from time_series.viz import (  # pyright: ignore[reportMissingImports]
    plot_error_distribution,
    plot_metric_grid,
    plot_scatter_pred_vs_actual,
    plot_series,
    plot_training_curve,
)

T, S, F = 10, 2, 3
FAMILIES = pd.Index(["AUTOMOTIVE", "GROCERY I", "PREPARED FOODS"])
STORES = pd.DataFrame(
    {"city": ["Quito", "Guayaquil"]},
    index=pd.Index([1, 2], name="store_nbr"),
)
DATES = pd.date_range("2023-01-01", periods=T)
TARGETS = torch.rand(T, S, F, dtype=torch.float64)  # type: ignore[attr-defined]
PREDS = torch.rand(T, S, F, dtype=torch.float64)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# plot_series
# ---------------------------------------------------------------------------


def test_plot_series_returns_figure() -> None:
    fig = plot_series(TARGETS, STORES, FAMILIES, store_nbr=1, family="AUTOMOTIVE")
    assert isinstance(fig, go.Figure)


def test_plot_series_one_trace_without_predictions() -> None:
    fig = plot_series(TARGETS, STORES, FAMILIES, store_nbr=1, family="AUTOMOTIVE")
    assert len(fig.data) == 1  # type: ignore[arg-type]


def test_plot_series_three_traces_with_predictions() -> None:
    fig = plot_series(TARGETS, STORES, FAMILIES, 1, "AUTOMOTIVE", predictions=PREDS)
    # actual + pred + error
    assert len(fig.data) == 3  # type: ignore[arg-type]


def test_plot_series_multi_overlay() -> None:
    fig = plot_series(TARGETS, STORES, FAMILIES, [1, 2], ["AUTOMOTIVE", "GROCERY I"])
    # 2 stores × 2 families = 4 actual traces
    assert len(fig.data) == 4  # type: ignore[arg-type]


def test_plot_series_with_dates() -> None:
    fig = plot_series(TARGETS, STORES, FAMILIES, 1, "AUTOMOTIVE", dates=DATES)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_metric_grid
# ---------------------------------------------------------------------------


def test_plot_metric_grid_tensor() -> None:
    metric = torch.rand(S, F)
    fig = plot_metric_grid(metric, STORES, FAMILIES, title="MSE")
    assert isinstance(fig, go.Figure)


def test_plot_metric_grid_ndarray() -> None:
    metric = np.random.rand(S, F)
    fig = plot_metric_grid(metric, STORES, FAMILIES)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_scatter_pred_vs_actual
# ---------------------------------------------------------------------------


def test_plot_scatter_returns_figure() -> None:
    fig = plot_scatter_pred_vs_actual(PREDS, TARGETS)
    assert isinstance(fig, go.Figure)


def test_plot_scatter_has_reference_line() -> None:
    fig = plot_scatter_pred_vs_actual(PREDS, TARGETS)
    # samples + y=x reference
    assert len(fig.data) == 2  # type: ignore[arg-type]


def test_plot_scatter_linear_scale() -> None:
    fig = plot_scatter_pred_vs_actual(PREDS, TARGETS, log_scale=False)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_error_distribution
# ---------------------------------------------------------------------------


def test_plot_error_distribution_returns_figure() -> None:
    fig = plot_error_distribution(PREDS, TARGETS)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_training_curve
# ---------------------------------------------------------------------------


def test_plot_training_curve_train_only() -> None:
    fig = plot_training_curve([1.0, 0.8, 0.6])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # type: ignore[arg-type]


def test_plot_training_curve_with_val() -> None:
    fig = plot_training_curve([1.0, 0.8], val_losses=[0.9, 0.7])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # type: ignore[arg-type]
