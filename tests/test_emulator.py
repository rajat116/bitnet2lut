"""Tests for the LUT-based matrix-vector multiplication emulator."""

import numpy as np
import pytest

from bitnet2lut.emulator import (
    direct_ternary_matvec,
    lut_matvec,
    verify_tile_roundtrip,
)
from bitnet2lut.lut_gen import tile_to_lut_indices


class TestLutMatvec:
    """Test LUT-based matvec against direct ternary matvec."""

    def test_identity_like(self):
        """Diagonal-ish ternary matrix should select activation elements."""
        # 4×4 identity-like ternary matrix
        W = np.eye(4, dtype=np.int8)
        x = np.array([10, 20, 30, 40], dtype=np.int8)

        expected = direct_ternary_matvec(W, x)
        indices = tile_to_lut_indices(W, group_size=4)
        actual = lut_matvec(indices, x, group_size=4)

        np.testing.assert_array_equal(expected, actual)

    def test_random_small(self):
        """Random small matrix — LUT should match direct."""
        rng = np.random.default_rng(123)
        W = rng.choice([-1, 0, 1], size=(8, 16)).astype(np.int8)
        x = rng.integers(-128, 128, size=16, dtype=np.int8)

        expected = direct_ternary_matvec(W, x)
        indices = tile_to_lut_indices(W, group_size=4)
        actual = lut_matvec(indices, x, group_size=4)

        np.testing.assert_array_equal(expected, actual)

    def test_random_large(self):
        """Larger random matrix — stress test."""
        rng = np.random.default_rng(456)
        W = rng.choice([-1, 0, 1], size=(128, 128)).astype(np.int8)
        x = rng.integers(-128, 128, size=128, dtype=np.int8)

        expected = direct_ternary_matvec(W, x)
        indices = tile_to_lut_indices(W, group_size=4)
        actual = lut_matvec(indices, x, group_size=4)

        np.testing.assert_array_equal(expected, actual)

    def test_all_zero_weights(self):
        """All-zero weights should give all-zero output."""
        W = np.zeros((8, 16), dtype=np.int8)
        x = np.array([100] * 16, dtype=np.int8)

        indices = tile_to_lut_indices(W, group_size=4)
        result = lut_matvec(indices, x, group_size=4)

        np.testing.assert_array_equal(result, np.zeros(8))

    def test_all_ones_weights(self):
        """All +1 weights should sum all activations for each row."""
        W = np.ones((4, 8), dtype=np.int8)
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)

        expected = np.array([36, 36, 36, 36], dtype=np.int32)
        indices = tile_to_lut_indices(W, group_size=4)
        actual = lut_matvec(indices, x, group_size=4)

        np.testing.assert_array_equal(expected, actual)

    def test_multiple_random_vectors(self):
        """Same weight matrix, multiple activation vectors."""
        rng = np.random.default_rng(789)
        W = rng.choice([-1, 0, 1], size=(64, 64)).astype(np.int8)
        indices = tile_to_lut_indices(W, group_size=4)

        for _ in range(20):
            x = rng.integers(-128, 128, size=64, dtype=np.int8)
            expected = direct_ternary_matvec(W, x)
            actual = lut_matvec(indices, x, group_size=4)
            np.testing.assert_array_equal(expected, actual)

    def test_realistic_dimensions(self):
        """Test with dimensions matching BitNet 2B4T projections."""
        rng = np.random.default_rng(101)

        # Test q_proj dimensions: (2560, 2560) is too large for unit test
        # Use a scaled version: (128, 128) which has the same structure
        for M, K in [(128, 128), (64, 128), (128, 64)]:
            W = rng.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
            x = rng.integers(-128, 128, size=K, dtype=np.int8)

            expected = direct_ternary_matvec(W, x)
            indices = tile_to_lut_indices(W, group_size=4)
            actual = lut_matvec(indices, x, group_size=4)

            np.testing.assert_array_equal(
                expected, actual,
                err_msg=f"Failed for shape ({M}, {K})"
            )


class TestVerifyTileRoundtrip:
    """Test weight encoding roundtrip verification."""

    def test_roundtrip_passes(self):
        """Valid roundtrip should return True."""
        rng = np.random.default_rng(42)
        W = rng.choice([-1, 0, 1], size=(64, 64)).astype(np.int8)
        indices = tile_to_lut_indices(W, group_size=4)
        assert verify_tile_roundtrip(W, indices, group_size=4) is True

    def test_corrupted_index_fails(self):
        """Corrupted index should fail roundtrip."""
        W = np.ones((4, 4), dtype=np.int8)
        indices = tile_to_lut_indices(W, group_size=4)
        indices[0, 0] = 0  # corrupt: should be 80 (all +1), now 0 (all -1)
        assert verify_tile_roundtrip(W, indices, group_size=4) is False
