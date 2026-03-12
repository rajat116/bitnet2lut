"""Tests for weight extraction and absmean quantization."""

import numpy as np
import pytest
import torch

from bitnet2lut.extract import absmean_quantize, validate_ternary


class TestAbsmeanQuantize:
    """Test the BitNet b1.58 absmean quantization."""

    def test_simple_values(self):
        """Known values should quantize correctly."""
        # If alpha = mean(|W|) and we round(W/alpha), then:
        # W = [1, -1, 0, 0.5, -0.5] → alpha = 0.6
        # scaled = [1.67, -1.67, 0, 0.83, -0.83]
        # rounded = [2, -2, 0, 1, -1]
        # clamped = [1, -1, 0, 1, -1]
        w = torch.tensor([[1.0, -1.0, 0.0, 0.5, -0.5]])
        ternary, alpha = absmean_quantize(w)
        assert alpha == pytest.approx(0.6, abs=1e-6)
        expected = torch.tensor([[1, -1, 0, 1, -1]], dtype=torch.int8)
        assert torch.equal(ternary, expected)

    def test_all_zeros(self):
        """All-zero weights should produce all-zero ternary."""
        w = torch.zeros(4, 8)
        ternary, alpha = absmean_quantize(w)
        assert alpha == 0.0
        assert torch.all(ternary == 0)

    def test_output_range(self):
        """Output must only contain {-1, 0, +1}."""
        rng = torch.Generator().manual_seed(42)
        w = torch.randn(64, 128, generator=rng)
        ternary, alpha = absmean_quantize(w)
        unique = set(torch.unique(ternary).tolist())
        assert unique.issubset({-1, 0, 1})

    def test_output_dtype(self):
        """Output should be int8."""
        w = torch.randn(8, 16)
        ternary, alpha = absmean_quantize(w)
        assert ternary.dtype == torch.int8

    def test_large_values_clamped(self):
        """Very large values should be clamped to ±1."""
        w = torch.tensor([[100.0, -100.0, 0.01]])
        ternary, alpha = absmean_quantize(w)
        assert ternary[0, 0].item() == 1
        assert ternary[0, 1].item() == -1


class TestValidateTernary:
    """Test ternary validation and statistics."""

    def test_valid_tensor(self):
        """Valid ternary tensor should pass validation."""
        t = torch.tensor([[1, 0, -1], [0, 1, -1]], dtype=torch.int8)
        stats = validate_ternary(t, "test")
        assert stats.num_params == 6
        assert stats.count_neg1 == 2
        assert stats.count_zero == 2
        assert stats.count_pos1 == 2

    def test_invalid_tensor(self):
        """Tensor with value 2 should fail validation."""
        t = torch.tensor([[1, 2, -1]], dtype=torch.int8)
        with pytest.raises(ValueError, match="invalid values"):
            validate_ternary(t, "test")

    def test_all_zeros(self):
        """All-zero tensor should be valid with 100% sparsity."""
        t = torch.zeros(4, 8, dtype=torch.int8)
        stats = validate_ternary(t, "test")
        assert stats.frac_zero == 1.0
        assert stats.frac_neg1 == 0.0
        assert stats.frac_pos1 == 0.0
