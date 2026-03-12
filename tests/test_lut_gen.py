"""Tests for T-MAC LUT generation."""

import itertools

import numpy as np
import pytest

from bitnet2lut.lut_gen import (
    compute_lut_entries,
    index_to_ternary,
    pack_ternary_to_index,
    ternary_configs,
    tile_to_lut_indices,
)


class TestTernaryConfigs:
    """Test ternary configuration enumeration."""

    def test_num_configs_g4(self):
        """group_size=4 should produce 81 configs."""
        configs = ternary_configs(4)
        assert configs.shape == (81, 4)

    def test_num_configs_g2(self):
        """group_size=2 should produce 9 configs."""
        configs = ternary_configs(2)
        assert configs.shape == (9, 2)

    def test_values_in_range(self):
        """All config values should be in {-1, 0, +1}."""
        configs = ternary_configs(4)
        unique = set(np.unique(configs).tolist())
        assert unique == {-1, 0, 1}

    def test_all_unique(self):
        """All configurations should be unique."""
        configs = ternary_configs(4)
        unique_rows = np.unique(configs, axis=0)
        assert unique_rows.shape[0] == 81


class TestPackTernaryToIndex:
    """Test ternary → index packing."""

    def test_all_zeros(self):
        """[0,0,0,0] should encode to index 40 (middle of 0..80)."""
        # {0,0,0,0} → digits {1,1,1,1} → 1*27 + 1*9 + 1*3 + 1 = 40
        w = np.array([[0, 0, 0, 0]], dtype=np.int8)
        idx = pack_ternary_to_index(w)
        assert idx[0] == 40

    def test_all_neg1(self):
        """[-1,-1,-1,-1] → digits [0,0,0,0] → index 0."""
        w = np.array([[-1, -1, -1, -1]], dtype=np.int8)
        idx = pack_ternary_to_index(w)
        assert idx[0] == 0

    def test_all_pos1(self):
        """[1,1,1,1] → digits [2,2,2,2] → 2*27+2*9+2*3+2 = 80."""
        w = np.array([[1, 1, 1, 1]], dtype=np.int8)
        idx = pack_ternary_to_index(w)
        assert idx[0] == 80

    def test_range(self):
        """All indices should be in [0, 80]."""
        configs = ternary_configs(4)
        indices = pack_ternary_to_index(configs)
        assert indices.min() == 0
        assert indices.max() == 80

    def test_unique_indices(self):
        """Each config should map to a unique index."""
        configs = ternary_configs(4)
        indices = pack_ternary_to_index(configs)
        assert len(np.unique(indices)) == 81

    def test_batch_packing(self):
        """Should work on batched inputs."""
        w = np.array([
            [0, 0, 0, 0],
            [-1, -1, -1, -1],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        indices = pack_ternary_to_index(w)
        assert indices.shape == (3,)
        assert indices[0] == 40
        assert indices[1] == 0
        assert indices[2] == 80


class TestIndexToTernary:
    """Test index → ternary decoding (inverse of packing)."""

    def test_roundtrip(self):
        """Pack → unpack should be identity."""
        configs = ternary_configs(4)
        indices = pack_ternary_to_index(configs)
        recovered = index_to_ternary(indices, group_size=4)
        np.testing.assert_array_equal(configs, recovered)

    def test_specific_values(self):
        """Check known index → ternary mappings."""
        # Index 0 → [-1,-1,-1,-1]
        result = index_to_ternary(np.array([0]), group_size=4)
        np.testing.assert_array_equal(result[0], [-1, -1, -1, -1])

        # Index 40 → [0,0,0,0]
        result = index_to_ternary(np.array([40]), group_size=4)
        np.testing.assert_array_equal(result[0], [0, 0, 0, 0])

        # Index 80 → [1,1,1,1]
        result = index_to_ternary(np.array([80]), group_size=4)
        np.testing.assert_array_equal(result[0], [1, 1, 1, 1])


class TestComputeLutEntries:
    """Test LUT entry computation."""

    def test_simple_case(self):
        """Manual computation for known values."""
        # activation = [10, 20, 30, 40]
        # config = [1, -1, 0, 1] → 10 - 20 + 0 + 40 = 30
        act = np.array([10, 20, 30, 40], dtype=np.int8)
        configs = np.array([[1, -1, 0, 1]], dtype=np.int8)
        result = compute_lut_entries(act, configs)
        assert result[0] == 30

    def test_all_zeros_config(self):
        """Config [0,0,0,0] should always produce 0."""
        act = np.array([100, 50, -30, 70], dtype=np.int8)
        configs = np.array([[0, 0, 0, 0]], dtype=np.int8)
        result = compute_lut_entries(act, configs)
        assert result[0] == 0

    def test_all_positive_config(self):
        """Config [1,1,1,1] should sum all activations."""
        act = np.array([10, 20, 30, 40], dtype=np.int8)
        configs = np.array([[1, 1, 1, 1]], dtype=np.int8)
        result = compute_lut_entries(act, configs)
        assert result[0] == 100

    def test_all_negative_config(self):
        """Config [-1,-1,-1,-1] should negate sum."""
        act = np.array([10, 20, 30, 40], dtype=np.int8)
        configs = np.array([[-1, -1, -1, -1]], dtype=np.int8)
        result = compute_lut_entries(act, configs)
        assert result[0] == -100

    def test_full_81_lut(self):
        """LUT with all 81 configs should have correct length."""
        act = np.array([10, 20, 30, 40], dtype=np.int8)
        configs = ternary_configs(4)
        result = compute_lut_entries(act, configs)
        assert result.shape == (81,)


class TestTileToLutIndices:
    """Test full tile → LUT index conversion."""

    def test_basic_conversion(self):
        """Convert a small ternary tile to LUT indices."""
        # 4×8 tile with g=4 → 4×2 indices
        tile = np.array([
            [1, 0, -1, 1, 0, 0, 0, 0],
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, -1, -1, -1, -1],
        ], dtype=np.int8)
        indices = tile_to_lut_indices(tile, group_size=4)
        assert indices.shape == (4, 2)
        # Second column of all-zeros row should be index 40
        assert indices[2, 0] == 40
        assert indices[2, 1] == 40

    def test_indivisible_cols_raises(self):
        """Columns not divisible by group_size should raise."""
        tile = np.zeros((4, 7), dtype=np.int8)  # 7 not divisible by 4
        with pytest.raises(ValueError, match="divisible"):
            tile_to_lut_indices(tile, group_size=4)

    def test_roundtrip_correctness(self):
        """Tile → indices → reconstruct should match original."""
        rng = np.random.default_rng(42)
        tile = rng.choice([-1, 0, 1], size=(128, 128)).astype(np.int8)
        indices = tile_to_lut_indices(tile, group_size=4)

        # Reconstruct
        from bitnet2lut.lut_gen import index_to_ternary
        reconstructed = index_to_ternary(indices, group_size=4)
        reconstructed = reconstructed.reshape(128, 128)
        np.testing.assert_array_equal(tile, reconstructed)
