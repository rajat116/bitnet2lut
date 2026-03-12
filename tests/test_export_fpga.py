"""Tests for FPGA BRAM init file export."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bitnet2lut.export_fpga import indices_to_coe, indices_to_mem


class TestIndicesExport:
    """Test .coe and .mem file generation."""

    def test_coe_format(self):
        """COE file should have correct header and data."""
        indices = np.array([0, 40, 80, 1, 2], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
            path = Path(f.name)
        indices_to_coe(indices, path, radix=16, bit_width=8)

        content = path.read_text()
        assert "memory_initialization_radix=16;" in content
        assert "memory_initialization_vector=" in content
        # Check hex values
        assert "00," in content    # index 0
        assert "28," in content    # index 40 = 0x28
        assert "50," in content    # index 80 = 0x50

        path.unlink()

    def test_mem_format(self):
        """MEM file should have @address hex_value format."""
        indices = np.array([10, 20, 30], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".mem", delete=False) as f:
            path = Path(f.name)
        indices_to_mem(indices, path, bit_width=8)

        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("@0000")
        assert "0A" in lines[0]  # 10 = 0x0A
        assert "14" in lines[1]  # 20 = 0x14
        assert "1E" in lines[2]  # 30 = 0x1E

        path.unlink()

    def test_coe_binary_radix(self):
        """COE file with binary radix."""
        indices = np.array([3], dtype=np.uint8)  # 3 = 00000011
        with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
            path = Path(f.name)
        indices_to_coe(indices, path, radix=2, bit_width=8)

        content = path.read_text()
        assert "memory_initialization_radix=2;" in content
        assert "00000011;" in content

        path.unlink()

    def test_empty_indices(self):
        """Empty index array should produce valid but empty-ish file."""
        indices = np.array([], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
            path = Path(f.name)
        indices_to_coe(indices, path)
        content = path.read_text()
        assert "memory_initialization_vector=" in content
        path.unlink()

    def test_large_tile(self):
        """Realistic tile size should produce valid file."""
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 81, size=(128, 32), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
            path = Path(f.name)
        indices_to_coe(indices, path)
        content = path.read_text()
        lines = content.strip().split("\n")
        # Header (2 lines) + data (128*32 = 4096 values)
        assert len(lines) == 2 + 128 * 32
        path.unlink()
