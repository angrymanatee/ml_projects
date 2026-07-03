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

    Input shape:  (batch, seq_len, n_stores, n_families)
    Output shape: (batch, n_output_steps, n_stores, n_families)

    The (n_stores x n_families) feature plane is flattened and linearly projected into
    d_model before being passed through the Transformer encoder. All encoder outputs are
    then flattened and projected to the full output horizon in one shot (no autoregression).
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
