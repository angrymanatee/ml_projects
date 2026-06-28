"""Tests for common/modules.py."""

import torch

from common.modules import GetLastIndex, PositionalEncoding

# ---------------------------------------------------------------------------
# GetLastIndex
# ---------------------------------------------------------------------------


class TestGetLastIndex:
    module = GetLastIndex()

    def test_1d(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])
        assert self.module(x).item() == 3.0

    def test_2d_shape(self) -> None:
        x = torch.rand(5, 4)
        out = self.module(x)
        assert out.shape == (5,)

    def test_2d_values(self) -> None:
        x = torch.arange(12.0).reshape(3, 4)
        out = self.module(x)
        # Last element of each row: 3, 7, 11
        assert torch.equal(out, torch.tensor([3.0, 7.0, 11.0]))

    def test_3d_shape(self) -> None:
        x = torch.rand(2, 5, 4)
        out = self.module(x)
        assert out.shape == (2, 5)

    def test_4d_shape(self) -> None:
        x = torch.rand(8, 10, 3, 4)
        out = self.module(x)
        assert out.shape == (8, 10, 3)

    def test_preserves_dtype(self) -> None:
        x = torch.rand(3, 4, dtype=torch.float64)
        assert self.module(x).dtype == torch.float64

    def test_size_one_last_dim(self) -> None:
        x = torch.rand(3, 1)
        out = self.module(x)
        assert out.shape == (3,)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    def test_output_shape_unchanged(self) -> None:
        pe = PositionalEncoding(d_model=16, max_length=32)
        x = torch.zeros(2, 10, 16)
        assert pe(x).shape == x.shape

    def test_zero_input_equals_encoding(self) -> None:
        pe = PositionalEncoding(d_model=16, max_length=32)
        x = torch.zeros(1, 8, 16)
        out = pe(x)
        assert torch.allclose(out, pe.positional_encoding[:, :8, :])

    def test_different_positions_differ(self) -> None:
        """Two distinct positions must have different encodings."""
        pe = PositionalEncoding(d_model=16, max_length=32)
        enc = pe.positional_encoding[0]  # (max_length, d_model)
        assert not torch.equal(enc[0], enc[1])

    def test_buffer_not_learned(self) -> None:
        pe = PositionalEncoding(d_model=16, max_length=32)
        assert list(pe.parameters()) == []

    def test_partial_sequence_length(self) -> None:
        pe = PositionalEncoding(d_model=8, max_length=64)
        x = torch.rand(3, 5, 8)
        assert pe(x).shape == (3, 5, 8)
