import enum
from collections import OrderedDict

import torch
from torch import Tensor, nn

from common.modules import GetLastIndex, PositionalEncoding


class HoldLastValue(nn.Module):
    """Baseline model that repeats the last observed time step as the forecast.

    Requires no training. For each sample in a batch, the final input step is
    tiled across all output steps. The resulting forecast is constant in time,
    making it a useful sanity-check lower bound for more complex models.

    Input shape:  (batch, seq_len, n_stores, n_families)
    Output shape: (batch, n_output_steps, n_stores, n_families)
    """

    def __init__(self, n_output_steps: int) -> None:
        """
        Args:
            n_output_steps: Number of future time steps to predict.
        """
        super().__init__()
        self.n_output_steps = n_output_steps

    def forward(self, input_sequence: Tensor) -> Tensor:
        """Tile the last time step across the output horizon.

        Args:
            input_sequence: Shape (batch, seq_len, n_stores, n_families).

        Returns:
            Tensor of shape (batch, n_output_steps, n_stores, n_families)
            with the last observed value repeated at every output step.
        """
        last_step = input_sequence[:, -1:, :, :]
        return last_step.expand(-1, self.n_output_steps, -1, -1)


class StoreSalesTransformer(nn.Module):
    """Transformer encoder-decoder for multi-step store sales forecasting.

    Maps an input window of shape [batch, window_lags, n_stores, n_families]
    to a prediction window of shape [batch, output_lags, n_stores, n_families].
    The spatial dimensions (store x family) are flattened into a single feature
    vector per time step and projected into d_model before the transformer.
    """

    def __init__(
        self,
        n_stores: int,
        n_families: int,
        n_output_steps: int,
        d_model: int = 128,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        embedding_size = n_stores * n_families
        self.embedding_size = embedding_size
        self.n_output_steps = n_output_steps
        self.input_transform = nn.Linear(embedding_size, d_model)
        self.output_transform = nn.Linear(d_model, embedding_size)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.output_relu = nn.ReLU()
        self._n_stores = n_stores
        self._n_families = n_families

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: [batch, window_lags, n_stores, n_families]
        Returns:
            [batch, n_output_steps, n_stores, n_families]
        """
        input_internal = self.input_transform(input.flatten(-2))
        tgt_internal = torch.zeros(
            input_internal.shape[0],
            self.n_output_steps,
            self.transformer.d_model,
            dtype=input_internal.dtype,
            device=input_internal.device,
        )
        output_internal = self.transformer(input_internal, tgt_internal)
        out_shape = (-1, self.n_output_steps, self._n_stores, self._n_families)
        return self.output_relu(
            self.output_transform(output_internal).reshape(out_shape)
        )


class PoolingMode(enum.StrEnum):
    """How to collapse the sequence dimension after the TransformerEncoder.

    ALL:  flatten all timestep embeddings into one long vector.
    LAST: take only the final timestep embedding.
    """

    ALL = enum.auto()
    LAST = enum.auto()

    @staticmethod
    def get_module(pooling_mode: PoolingMode) -> nn.Module:
        match pooling_mode:
            case PoolingMode.ALL:
                return nn.Flatten(-2)
            case PoolingMode.LAST:
                return GetLastIndex()

    @staticmethod
    def parse(input_string: str) -> PoolingMode:
        return PoolingMode._value2member_map_[input_string]  # type: ignore


class StoreSalesEncoderOnly(nn.Module):
    """Encoder-only Transformer that maps a sales history window to a multi-step forecast.

    Input shape:  (batch, seq_len, n_stores, n_families * n_input_channels)
                  when used with StoreData(flatten_output=True)
    Output shape: (batch, n_output_steps, n_stores, n_families)

    The spatial-feature plane (n_stores × n_families × n_input_channels) is flattened
    and linearly projected into d_model before being passed through the Transformer encoder.
    All encoder outputs are then flattened and projected to the full output horizon in one
    shot (no autoregression).
    """

    def __init__(
        self,
        n_stores: int,
        n_families: int,
        n_output_steps: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        max_seq_length: int = 512,
        pooling_mode: PoolingMode = PoolingMode.ALL,
        dim_feedforward: int = 256,
    ) -> None:
        """Build the encoder-only model.

        Args:
            n_stores: Number of distinct stores in the dataset.
            n_families: Number of product families per store.
            n_output_steps: Forecast horizon (number of future time steps to predict).
            d_model: Transformer embedding dimension. Must be divisible by nhead.
            nhead: Number of attention heads.
            num_layers: Number of stacked TransformerEncoderLayer blocks.
            max_seq_length: Upper bound on input sequence length passed to PositionalEncoding.
            pooling_mode: How to collapse the sequence dimension after the encoder.
                ALL flattens all timestep embeddings; LAST takes only the final one.
            dim_feedforward: Width of the FFN sublayer inside each TransformerEncoderLayer.
                PyTorch default is 2048; for d_model=64 something in [128, 512] is typical.
        """
        super().__init__()
        self.n_stores = n_stores
        self.n_families = n_families
        self.n_output_steps = n_output_steps
        models = OrderedDict[str, nn.Module](
            [
                ("input_flatten", nn.Flatten(-2)),
                ("input_proj", nn.LazyLinear(d_model)),
                ("pos_enc", PositionalEncoding(d_model, max_length=max_seq_length)),
                (
                    "encoder",
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model,
                            nhead,
                            dim_feedforward=dim_feedforward,
                            batch_first=True,
                        ),
                        num_layers=num_layers,
                    ),
                ),
                ("pooling", PoolingMode.get_module(pooling_mode)),
                ("output_proj", nn.LazyLinear(n_stores * n_families * n_output_steps)),
                (
                    "output_unflatten",
                    nn.Unflatten(-1, (n_output_steps, n_stores, n_families)),
                ),
                ("ReLU", nn.ReLU()),
            ]
        )
        self.sequence = nn.Sequential(models)

    def forward(self, input_sequence: Tensor) -> Tensor:
        """Run the full encoder-only forward pass.

        Args:
            input_sequence: Shape (batch, seq_len, n_stores, n_families).

        Returns:
            Forecast tensor of shape (batch, n_output_steps, n_stores, n_families).
        """
        return self.sequence(input_sequence)


class TimeReduction(nn.Module):
    """Reduce the feature dimension via two Conv1d layers followed by average pooling.

    Treats the input as [B, T, S, F, X], transposes to [B, F, S, T, X], flattens the
    batch/spatial dims to [B*F*S, T, X], then applies Conv1d (which slides over X with
    T as channels). The result is unflattened and transposed back to [B, C_out, S, F, L_out]
    where C_out = out_channels and L_out is the reduced feature length.

    This gives the downstream InputProjection a fixed-length feature vector per (store, family)
    regardless of the original number of input channels.
    """

    def __init__(self, out_channels: int, kernel_size: int, stride: int = 7) -> None:
        """
        Args:
            out_channels: Number of output channels from the second Conv1d.
            kernel_size: Kernel width for both Conv1d layers. Must be ≤ n_input_channels.
            stride: Stride for the second Conv1d only (first Conv1d always has stride 1).
        """
        super().__init__()
        self.sequence = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.LazyConv1d(32, kernel_size)),
                    ("conv2", nn.LazyConv1d(out_channels, kernel_size, stride=stride)),
                    ("pool1", nn.AvgPool1d(2, 2)),
                ]
            )
        )

    def forward(self, input_sequence: Tensor) -> Tensor:
        """
        Args:
            input_sequence: Shape (batch, T, n_stores, n_families, n_input_channels).

        Returns:
            Tensor of shape (batch, out_channels, n_stores, n_families, L_out).
        """
        input_sequence = input_sequence.transpose(1, 3)
        input_shape = input_sequence.shape
        output_data: Tensor = self.sequence(input_sequence.flatten(0, -3))
        final_output = output_data.unflatten(0, (input_shape[:-2])).transpose(1, 3)
        return final_output


class InputProjection(nn.Module):
    """Project each (store, family) feature vector into d_model and add positional encoding.

    Accepts [B, seq, n_stores, n_families, X], transposes and flattens the batch and spatial
    dims to [B*n_families*n_stores, seq, X], applies a linear projection to d_model, then adds
    sinusoidal positional encodings over the sequence dimension.

    Output shape: (B*n_families*n_stores, seq, d_model).
    """

    def __init__(self, d_model: int, max_length: int) -> None:
        """
        Args:
            d_model: Projected embedding dimension.
            max_length: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.input_proj = nn.LazyLinear(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length=max_length)

    def forward(self, input_sequence: Tensor) -> Tensor:
        """
        Args:
            input_sequence: Shape (batch, seq, n_stores, n_families, X).

        Returns:
            Tensor of shape (batch*n_families*n_stores, seq, d_model).
        """
        transposed = input_sequence.transpose(1, 3).flatten(0, 2)
        projected = self.input_proj(transposed)
        encoded = self.pos_encoding(projected)
        return encoded


class TimeToSF(nn.Module):
    """Collapse the time/sequence dimension and reshape into (store × family) tokens.

    Mean-pools the sequence dimension from [B*n_families*n_stores, seq, d_time] to
    [B*n_families*n_stores, d_time], reshapes to [B, n_stores*n_families, d_time],
    then projects to d_sf. The result feeds the store×family TransformerEncoder.
    """

    def __init__(self, n_stores: int, n_families: int, d_time: int, d_sf: int) -> None:
        """
        Args:
            n_stores: Number of stores.
            n_families: Number of product families.
            d_time: Input feature dimension (output of the time TransformerEncoder).
            d_sf: Output feature dimension for the store×family encoder.
        """
        super().__init__()
        self.n_sf = n_stores * n_families
        self.d_time = d_time
        self.d_sf = d_sf
        self.proj = nn.Linear(d_time, d_sf)

    def forward(self, input_sequence: Tensor) -> Tensor:
        """
        Args:
            input_sequence: Shape (batch*n_families*n_stores, seq, d_time).

        Returns:
            Tensor of shape (batch, n_stores*n_families, d_sf).
        """
        mean_pooled = input_sequence.mean(1)
        reshaped = mean_pooled.reshape(-1, self.n_sf, self.d_time)
        return self.proj(reshaped)


class OutputProjection(nn.Module):
    """Project (store × family) token embeddings to the full output forecast horizon.

    Takes [B, n_stores*n_families, d_sf], projects each token to n_output_steps, then
    reshapes and permutes to [B, n_output_steps, n_stores, n_families].
    """

    def __init__(self, n_stores: int, n_families: int, n_output_steps: int) -> None:
        """
        Args:
            n_stores: Number of stores (used to unflatten the spatial dims).
            n_families: Number of product families.
            n_output_steps: Forecast horizon length.
        """
        super().__init__()
        self.n_stores = n_stores
        self.n_families = n_families
        self.n_output_steps = n_output_steps
        self.output_proj = nn.LazyLinear(n_output_steps)

    def forward(self, input_sequence: Tensor) -> Tensor:
        """
        Args:
            input_sequence: Shape (batch, n_stores*n_families, d_sf).

        Returns:
            Tensor of shape (batch, n_output_steps, n_stores, n_families).
        """
        projected: Tensor = self.output_proj(input_sequence)
        reshaped = projected.reshape(
            -1, self.n_stores, self.n_families, self.n_output_steps
        ).permute(0, 3, 1, 2)
        return reshaped


class StoreSalesFactorizedEncoder(nn.Module):
    """Two-stage factorized encoder for multi-step store sales forecasting.

    Stage 1 — time: each (store, family) time series is processed independently.
      TimeReduction compresses the feature dimension via Conv1d, then InputProjection
      projects to d_model_time and adds positional encoding. A TransformerEncoder
      attends over the reduced time tokens. TimeToSF mean-pools the sequence and
      reshapes to one embedding per (store, family) pair.

    Stage 2 — store×family: a second TransformerEncoder attends over all
      n_stores × n_families spatial tokens. OutputProjection then projects each token
      to n_output_steps and reshapes to the final forecast grid.

    Input shape:  (batch, window_lags, n_stores, n_families, n_input_channels)
    Output shape: (batch, n_output_steps, n_stores, n_families)
    """

    def __init__(
        self,
        n_stores: int,
        n_families: int,
        n_output_steps: int,
        d_model_time: int = 64,
        nhead_time: int = 2,
        dim_feedforward_time: int = 256,
        num_layers_time: int = 2,
        d_model_sf: int = 64,
        nhead_sf: int = 2,
        dim_feedforward_sf: int = 256,
        num_layers_sf: int = 2,
        max_seq_length: int = 128,
        reduction_channels: int = 12,
        reduction_width: int = 16,
        reduction_stride: int = 14,
    ) -> None:
        """
        Args:
            n_stores: Number of distinct stores.
            n_families: Number of product families per store.
            n_output_steps: Forecast horizon (future time steps to predict).
            d_model_time: Embedding dimension for the time-axis TransformerEncoder.
                Must be divisible by nhead_time.
            nhead_time: Attention heads in the time-axis encoder.
            dim_feedforward_time: FFN width inside each time-axis encoder layer.
            num_layers_time: Number of stacked time-axis encoder layers.
            d_model_sf: Embedding dimension for the store×family TransformerEncoder.
                Must be divisible by nhead_sf.
            nhead_sf: Attention heads in the store×family encoder.
            dim_feedforward_sf: FFN width inside each store×family encoder layer.
            num_layers_sf: Number of stacked store×family encoder layers.
            max_seq_length: Upper bound on sequence length passed to PositionalEncoding.
                Must be ≥ reduction_channels.
            reduction_channels: Output channels of TimeReduction (= sequence length seen
                by the time encoder).
            reduction_width: Conv1d kernel size in TimeReduction. Must be ≤ n_input_channels.
            reduction_stride: Stride for the second Conv1d in TimeReduction.
        """
        super().__init__()
        self.n_stores = n_stores
        self.n_families = n_families
        self.n_output_steps = n_output_steps
        models = OrderedDict[str, nn.Module](
            [
                # [B, T, S, F, X] -> [B, reduction_channels, S, F, L_out]
                (
                    "input_reduction",
                    TimeReduction(
                        reduction_channels, reduction_width, reduction_stride
                    ),
                ),
                # [B, reduction_channels, S, F, L_out] -> [B*F*S, reduction_channels, d_model_time]
                (
                    "input_proj",
                    InputProjection(d_model_time, max_length=max_seq_length),
                ),
                # [B*F*S, reduction_channels, d_model_time] (unchanged shape)
                (
                    "time_encoder",
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model_time,
                            nhead_time,
                            dim_feedforward=dim_feedforward_time,
                            batch_first=True,
                        ),
                        num_layers=num_layers_time,
                    ),
                ),
                # [B*F*S, reduction_channels, d_model_time] -> [B, S*F, d_model_sf]
                (
                    "time_to_storefam",
                    TimeToSF(n_stores, n_families, d_model_time, d_model_sf),
                ),
                (
                    "sf_encoder",
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model_sf,
                            nhead_sf,
                            dim_feedforward=dim_feedforward_sf,
                            batch_first=True,
                        ),
                        num_layers=num_layers_sf,
                    ),
                ),
                ("output_proj", OutputProjection(n_stores, n_families, n_output_steps)),
                ("ReLU", nn.ReLU()),
            ]
        )
        self.sequence = nn.Sequential(models)

    def forward(self, input_sequence: Tensor) -> Tensor:
        """Run the full encoder-only forward pass.

        Args:
            input_sequence: Shape (batch, seq_len, n_stores, n_families, n_features).

        Returns:
            Forecast tensor of shape (batch, n_output_steps, n_stores, n_families).
        """
        return self.sequence(input_sequence)
