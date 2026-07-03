"""Personal Modules and utilities."""

import math
from collections import OrderedDict

import torch
from torch import Tensor, nn

SIN_POSITIONAL_SCALE = 10_000.0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from "Attention Is All You Need" (Vaswani et al., 2017).

    Precomputes a fixed (non-learned) encoding matrix of shape (1, max_length, d_model)
    using alternating sin/cos at geometrically spaced frequencies:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The buffer is added directly to the input embeddings in forward(), so the encoding
    is position-absolute and sequence-length-agnostic up to max_length.
    """

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        """Precompute and register the positional encoding buffer.

        Args:
            d_model: Embedding dimension. Must be even (sin fills even dims, cos fills odd).
            max_length: Maximum sequence length to support. Sequences longer than this will
                raise an index error in forward().
        """
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        position = torch.arange(max_length).unsqueeze(0).unsqueeze(-1)
        inv_divisor = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(SIN_POSITIONAL_SCALE) / d_model)
        )
        positional_encoding = torch.zeros(1, max_length, d_model)
        positional_encoding[0, :, 0::2] = torch.sin(position * inv_divisor)
        positional_encoding[0, :, 1::2] = torch.cos(position * inv_divisor)
        self.positional_encoding: Tensor  # Make typing happy
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Add positional encoding to input embeddings.

        Args:
            input_tensor: Shape (batch, seq_len, d_model). seq_len must be <= max_length.

        Returns:
            Tensor of same shape as input_tensor with positional encoding summed in.
        """
        return input_tensor + self.positional_encoding[:, : input_tensor.shape[1], :]


class LazyMLP(nn.Module):
    """MLP with lazy input inference and configurable depth.

    With depth=0, reduces to a single linear layer. With depth>0, inserts
    that many hidden layers of width features_per_layer before the output.
    Uses LazyLinear so the input dimension is inferred on the first forward pass.
    """

    def __init__(
        self,
        output_features: int = 16,
        depth: int = 0,
        features_per_layer: int = 16,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = activation or nn.ReLU()
        layers = OrderedDict[str, nn.Module]()
        for layer_i in range(depth):
            layers[f"linear_{layer_i}"] = nn.LazyLinear(features_per_layer)
            layers[f"activ_{layer_i}"] = activation
        layers["linear_output"] = nn.LazyLinear(output_features)
        self.sequence = nn.Sequential(layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.sequence(input)


class GetLastIndex(nn.Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor[..., -1]


class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error loss.

    Computes MSE(log(1 + input), log(1 + target)), which is the competition
    metric (RMSLE) squared. Use sqrt on the output to recover RMSLE.
    Inputs must be non-negative; log1p is used for numerical stability near zero.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self._mse_loss(torch.log1p(input), torch.log1p(target))  # type: ignore[attr-defined]
