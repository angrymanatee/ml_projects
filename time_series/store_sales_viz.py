"""Visualization utilities for store sales forecasting.

All functions return ``go.Figure`` objects, which render interactively in Jupyter
notebooks and can be exported via ``fig.write_html`` / ``fig.write_image``.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import Tensor, nn
from torch.utils.data import DataLoader

_COLOR_CYCLE = plotly.colors.qualitative.Plotly


def _to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def plot_series(
    targets: Tensor,
    stores: pd.DataFrame,
    families: pd.Index,
    store_nbr: int | Sequence[int],
    family: str | Sequence[str],
    predictions: Tensor | None = None,
    dates: pd.DatetimeIndex | None = None,
    title: str | None = None,
) -> go.Figure:
    """Compare actual vs predicted sales, with an optional error panel below.

    Each (store_nbr, family) pair adds one actual trace (solid) and, if
    predictions are supplied, one predicted trace (dashed) and one error trace
    (filled) in a second subplot. Pass lists to overlay multiple series.

    Args:
        targets: sales values, shape [T, num_stores, num_families].
        stores: DataFrame indexed by store_nbr; used to locate each store in
            the tensor.
        families: Index mapping tensor column → family name.
        store_nbr: 1-based store number(s) matching the original data.
        family: product family name(s).
        predictions: optional tensor of the same shape as targets.
        dates: DatetimeIndex of length T; uses integer steps if None.
        title: figure title; auto-generated if None.

    Returns:
        Interactive Plotly figure (one or two rows).
    """
    if isinstance(store_nbr, int):
        store_nbr = [store_nbr]
    if isinstance(family, str):
        family = [family]

    has_pred = predictions is not None
    rows = 2 if has_pred else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Sales"] + (["Error (pred − actual)"] if has_pred else []),
        vertical_spacing=0.08,
    )
    x = np.array(dates) if dates is not None else np.arange(targets.shape[0])

    series_idx = 0
    for store in store_nbr:
        store_idx = cast(int, stores.index.get_loc(store))
        for family_name in family:
            family_idx = cast(int, families.get_loc(family_name))
            label = f"s{store}/{family_name}"
            color = _COLOR_CYCLE[series_idx % len(_COLOR_CYCLE)]
            series_idx += 1
            actual = _to_numpy(targets[:, store_idx, family_idx])

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=actual,
                    name=f"{label} actual",
                    mode="lines",
                    line=dict(color=color),
                ),
                row=1,
                col=1,
            )
            if has_pred:
                pred = _to_numpy(predictions[:, store_idx, family_idx])
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=pred,
                        name=f"{label} pred",
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=pred - actual,
                        name=f"{label} error",
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color=color),
                    ),
                    row=2,
                    col=1,
                )

    if title is None:
        title = f"store_nbr={list(store_nbr)}, family={list(family)}"
    fig.update_layout(title=title, height=600 if has_pred else 350)
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    if has_pred:
        fig.update_yaxes(title_text="Error", row=2, col=1)
    return fig


def plot_metric_grid(
    metric: Tensor | np.ndarray,
    stores: pd.DataFrame,
    families: pd.Index,
    title: str = "Metric",
    colorscale: str = "RdYlGn_r",
) -> go.Figure:
    """Heatmap of a scalar metric across the store × family grid.

    Args:
        metric: shape [num_stores, num_families]; rows are stores in the same
            order as stores.index, columns are families in families order.
        stores: DataFrame indexed by store_nbr; provides y-axis labels.
        families: Index of family names for x-axis labels.
        title: colorbar title and figure title.
        colorscale: Plotly colorscale name. ``RdYlGn_r`` maps high values to
            red (swap to ``RdYlGn`` for "higher is better" metrics).

    Returns:
        Interactive heatmap figure.
    """
    metric_vals = (
        _to_numpy(metric) if isinstance(metric, Tensor) else np.asarray(metric)
    )
    store_labels = [f"Store {n}" for n in stores.index]
    family_labels = list(families)

    fig = go.Figure(
        go.Heatmap(
            z=metric_vals,
            x=family_labels,
            y=store_labels,
            colorscale=colorscale,
            colorbar=dict(title=title),
            hovertemplate=(
                "Store: %{y}<br>Family: %{x}<br>" + title + ": %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45),
        height=max(400, 18 * len(store_labels)),
        margin=dict(b=160),
    )
    return fig


def plot_scatter_pred_vs_actual(
    predictions: Tensor,
    targets: Tensor,
    title: str = "Predicted vs Actual",
    log_scale: bool = True,
) -> go.Figure:
    """Scatter of predicted vs actual sales across all outputs.

    Points on the y = x diagonal indicate perfect calibration. Systematic
    offsets reveal bias; spread reveals variance.

    Args:
        predictions: any shape; flattened for the scatter.
        targets: same shape as predictions.
        title: figure title.
        log_scale: if True, apply log(1+x) to both axes to reduce dynamic range.

    Returns:
        Scatter figure with y = x reference line.
    """
    pred_vals = _to_numpy(predictions).ravel().astype(float)
    target_vals = _to_numpy(targets).ravel().astype(float)

    if log_scale:
        pred_vals = np.log1p(pred_vals)
        target_vals = np.log1p(target_vals)
        axis_label = "log(1 + sales)"
    else:
        axis_label = "sales"

    axis_lim = (
        float(min(pred_vals.min(), target_vals.min())),
        float(max(pred_vals.max(), target_vals.max())),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=target_vals,
            y=pred_vals,
            mode="markers",
            marker=dict(size=3, opacity=0.3),
            name="samples",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(axis_lim),
            y=list(axis_lim),
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="y = x",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"actual {axis_label}",
        yaxis_title=f"predicted {axis_label}",
    )
    return fig


def plot_error_distribution(
    predictions: Tensor,
    targets: Tensor,
    title: str = "Log-Space Error Distribution",
) -> go.Figure:
    """Histogram of signed log-space errors: log(1+pred) − log(1+actual).

    Centered near zero with small spread indicates a well-calibrated model.
    Systematic offsets reveal bias; heavy tails flag problematic outliers.

    Args:
        predictions: any shape; flattened. Must be non-negative — log1p(x) is
            undefined for x < -1 and will silently produce NaN.
        targets: same shape as predictions. Same non-negativity requirement.
        title: figure title.

    Returns:
        Histogram figure with a vertical line at zero.
    """
    errors = np.log1p(_to_numpy(predictions).ravel()) - np.log1p(
        _to_numpy(targets).ravel()
    )
    fig = go.Figure(go.Histogram(x=errors, nbinsx=100, name="error"))
    fig.add_vline(x=0.0, line_dash="dash", line_color="red", annotation_text="zero")
    fig.update_layout(
        title=title,
        xaxis_title="log(1 + pred) − log(1 + actual)",
        yaxis_title="count",
    )
    return fig


def plot_training_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    title: str = "Training Loss",
) -> go.Figure:
    """Line chart of training (and optional validation) loss over epochs.

    Args:
        train_losses: per-epoch training loss values.
        val_losses: optional per-epoch validation loss values.
        title: figure title.

    Returns:
        Line chart figure.
    """
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=epochs, y=list(train_losses), name="train", mode="lines+markers")
    )
    if val_losses is not None:
        fig.add_trace(
            go.Scatter(x=epochs, y=list(val_losses), name="val", mode="lines+markers")
        )
    fig.update_layout(title=title, xaxis_title="epoch", yaxis_title="loss")
    return fig


class StoreSalesAnalyzer:
    """Runs post-training analysis and logs plots as MLflow HTML artifacts.

    Collects model predictions over a DataLoader, then generates a
    predicted-vs-actual scatter, log-space error distribution, and a
    per-(store, family) RMSLE heatmap. All plots are logged to the active
    MLflow run under the ``analysis/`` artifact directory.

    Must be called within an active mlflow.start_run() context.

    Args:
        model: trained model to evaluate; will be set to eval mode.
        data_loader: DataLoader yielding (input, target) batches where
            target has shape [batch, output_lags, n_stores, n_families].
        stores: DataFrame indexed by store_nbr (from StoreData.stores).
        families: Index mapping column position to family name (from
            StoreData.families).
        device: device on which to run inference.
        autocast_dtype: if set, wraps inference in torch.autocast with this dtype.
            Must match the dtype training used — mixing a float16 input against
            plain float32 weights on MPS without autocast crashes the process.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,  # type: ignore[type-arg]
        stores: pd.DataFrame,
        families: pd.Index,
        device: torch.device,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.stores = stores
        self.families = families
        self.device = device
        self.autocast_dtype = autocast_dtype

    def run(self) -> None:
        """Generate all analysis plots and log them to the active MLflow run."""
        predictions, targets = self._collect_predictions()
        self._log_figure(
            plot_scatter_pred_vs_actual(predictions, targets),
            "analysis/scatter_pred_vs_actual",
        )
        self._log_figure(
            plot_error_distribution(predictions, targets),
            "analysis/error_distribution",
        )
        rmsle_grid = self._per_series_rmsle(predictions, targets)
        self._log_figure(
            plot_metric_grid(rmsle_grid, self.stores, self.families, title="RMSLE"),
            "analysis/rmsle_heatmap",
        )
        self._log_series_plots(predictions[0], targets[0])

    @torch.inference_mode()
    def _collect_predictions(self) -> tuple[Tensor, Tensor]:
        self.model.eval()
        all_predictions: list[Tensor] = []
        all_targets: list[Tensor] = []
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast:
            for batch_X, batch_y in self.data_loader:
                pred = self.model(batch_X.to(self.device))
                all_predictions.append(pred.cpu())
                all_targets.append(batch_y.cpu())
        return torch.cat(all_predictions), torch.cat(all_targets)

    @staticmethod
    def _per_series_rmsle(predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute RMSLE per (store, family) averaged over all windows and time steps.

        Args:
            predictions: shape [N, output_lags, n_stores, n_families].
            targets: same shape as predictions.

        Returns:
            Tensor of shape [n_stores, n_families].
        """
        squared_log_errors = (torch.log1p(predictions) - torch.log1p(targets)).pow(2)
        return squared_log_errors.mean(dim=(0, 1)).sqrt()

    def _log_series_plots(self, predictions: Tensor, targets: Tensor) -> None:
        """Log series plots per family: one ALL-stores overlay and one per-store plot.

        Directory structure: analysis/series/{family}/ALL.html and
        analysis/series/{family}/store_{n}.html for each store number.

        Args:
            predictions: shape [output_lags, n_stores, n_families] — first val sample.
            targets: same shape as predictions.
        """
        store_nbrs = list(self.stores.index)
        for family_name in self.families:
            slug = family_name.replace("/", "_").replace(",", "_").replace(" ", "_")
            self._log_figure(
                plot_series(
                    targets=targets,
                    stores=self.stores,
                    families=self.families,
                    store_nbr=store_nbrs,
                    family=family_name,
                    predictions=predictions,
                    title=f"{family_name} — all stores (first val window)",
                ),
                f"analysis/series/{slug}/ALL",
            )
            for store_nbr in store_nbrs:
                self._log_figure(
                    plot_series(
                        targets=targets,
                        stores=self.stores,
                        families=self.families,
                        store_nbr=store_nbr,
                        family=family_name,
                        predictions=predictions,
                        title=f"{family_name} — store {store_nbr} (first val window)",
                    ),
                    f"analysis/series/{slug}/store_{store_nbr}",
                )

    @staticmethod
    def _log_figure(fig: go.Figure, artifact_path: str) -> None:
        artifact_dir = str(Path(artifact_path).parent)
        stem = Path(artifact_path).name
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / f"{stem}.html"
            fig.write_html(str(html_path))
            mlflow.log_artifact(str(html_path), artifact_path=artifact_dir)
