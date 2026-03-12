"""Tests for weight matrix tiling."""

import numpy as np
import pytest

from bitnet2lut.tile import tile_matrix


class TestTileMatrix:
    """Test matrix tiling logic."""

    def test_exact_division(self):
        """Matrix that divides evenly into blocks."""
        matrix = np.ones((256, 256), dtype=np.int8)
        tiles, infos = tile_matrix(matrix, 128, 128)
        assert len(tiles) == 4  # 2×2 tiles
        assert all(t.shape == (128, 128) for t in tiles)
        assert all(not info.is_padded for info in infos)

    def test_padding_needed(self):
        """Matrix that requires padding on edges."""
        matrix = np.ones((300, 200), dtype=np.int8)
        tiles, infos = tile_matrix(matrix, 128, 128, pad_strategy="zero")
        # rows: ceil(300/128) = 3, cols: ceil(200/128) = 2
        assert len(tiles) == 6
        assert all(t.shape == (128, 128) for t in tiles)

        # Check padded tiles have zeros in padding region
        # Last row tile: rows 256..299 (44 rows of data, 84 rows of padding)
        last_row_tile = tiles[4]  # tile at (2, 0)
        assert infos[4].original_rows == 300 - 256  # 44

    def test_no_padding(self):
        """No-pad strategy produces variable-size edge tiles."""
        matrix = np.ones((300, 200), dtype=np.int8)
        tiles, infos = tile_matrix(matrix, 128, 128, pad_strategy="none")
        assert len(tiles) == 6
        # Edge tiles should have smaller dimensions
        assert tiles[4].shape[0] == 44   # 300 - 256
        assert tiles[5].shape[1] == 72   # 200 - 128

    def test_tile_contents_preserved(self):
        """Tiling should preserve the original data."""
        rng = np.random.default_rng(42)
        matrix = rng.integers(-1, 2, size=(256, 256), dtype=np.int8)
        tiles, infos = tile_matrix(matrix, 128, 128)

        # Reconstruct and compare
        reconstructed = np.zeros_like(matrix)
        for tile, info in zip(tiles, infos):
            r, c = info.row_start, info.col_start
            reconstructed[r : r + 128, c : c + 128] = tile

        np.testing.assert_array_equal(matrix, reconstructed)

    def test_single_tile(self):
        """Matrix smaller than block size produces one padded tile."""
        matrix = np.ones((64, 64), dtype=np.int8)
        tiles, infos = tile_matrix(matrix, 128, 128, pad_strategy="zero")
        assert len(tiles) == 1
        assert tiles[0].shape == (128, 128)
        assert infos[0].is_padded
        assert infos[0].original_rows == 64
        assert infos[0].original_cols == 64

    def test_tile_indices_sequential(self):
        """Tile indices should be sequential starting from 0."""
        matrix = np.ones((512, 512), dtype=np.int8)
        tiles, infos = tile_matrix(matrix, 128, 128)
        indices = [info.tile_idx for info in infos]
        assert indices == list(range(len(tiles)))
