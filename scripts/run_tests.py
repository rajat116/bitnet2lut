#!/usr/bin/env python3
"""Standalone test runner for bitnet2lut — no external deps beyond numpy.

Runs all core algorithmic tests to verify the LUT pipeline correctness.
Does NOT test weight extraction (requires torch/safetensors) or CLI.

Usage:
    python3 scripts/run_tests.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = 0
FAIL = 0


def test(name):
    """Decorator to register and run a test."""
    def decorator(func):
        global PASS, FAIL
        try:
            func()
            print(f"  ✓ {name}")
            PASS += 1
        except Exception as e:
            print(f"  ✗ {name}")
            print(f"    {e}")
            traceback.print_exc(limit=2)
            FAIL += 1
        return func
    return decorator


# ======================================================================
# Model Config Tests
# ======================================================================
print("\n=== model_config ===")

from bitnet2lut.model_config import (
    BITLINEAR_PROJECTIONS,
    BitNetConfig,
    get_all_weight_names,
    get_weight_name,
    total_ternary_params,
)


@test("BitNetConfig defaults match 2B4T")
def _():
    c = BitNetConfig()
    assert c.hidden_size == 2560
    assert c.intermediate_size == 6912
    assert c.num_hidden_layers == 30
    assert c.num_attention_heads == 20
    assert c.num_key_value_heads == 5
    assert c.head_dim == 128
    assert c.num_gqa_groups == 4


@test("7 projections per layer")
def _():
    assert len(BITLINEAR_PROJECTIONS) == 7


@test("Weight naming pattern")
def _():
    name = get_weight_name(0, "self_attn.q_proj")
    assert name == "model.layers.0.self_attn.q_proj.weight"
    name = get_weight_name(29, "mlp.down_proj")
    assert name == "model.layers.29.mlp.down_proj.weight"


@test("Total ternary params ~2.08B")
def _():
    total = total_ternary_params()
    assert 2_000_000_000 < total < 2_200_000_000, f"Got {total}"


@test("All weight names count = 30 * 7 = 210")
def _():
    names = get_all_weight_names(30)
    assert len(names) == 210


# ======================================================================
# LUT Generation Tests
# ======================================================================
print("\n=== lut_gen ===")

from bitnet2lut.lut_gen import (
    compute_lut_entries,
    index_to_ternary,
    pack_ternary_to_index,
    ternary_configs,
    tile_to_lut_indices,
)


@test("ternary_configs(4) → 81 configs")
def _():
    configs = ternary_configs(4)
    assert configs.shape == (81, 4)
    assert set(np.unique(configs).tolist()) == {-1, 0, 1}
    assert np.unique(configs, axis=0).shape[0] == 81


@test("ternary_configs(2) → 9 configs")
def _():
    configs = ternary_configs(2)
    assert configs.shape == (9, 2)


@test("pack: [0,0,0,0] → 40")
def _():
    w = np.array([[0, 0, 0, 0]], dtype=np.int8)
    assert pack_ternary_to_index(w)[0] == 40


@test("pack: [-1,-1,-1,-1] → 0")
def _():
    w = np.array([[-1, -1, -1, -1]], dtype=np.int8)
    assert pack_ternary_to_index(w)[0] == 0


@test("pack: [1,1,1,1] → 80")
def _():
    w = np.array([[1, 1, 1, 1]], dtype=np.int8)
    assert pack_ternary_to_index(w)[0] == 80


@test("pack: all 81 indices unique in [0, 80]")
def _():
    configs = ternary_configs(4)
    indices = pack_ternary_to_index(configs)
    assert indices.min() == 0
    assert indices.max() == 80
    assert len(np.unique(indices)) == 81


@test("pack → unpack roundtrip is identity")
def _():
    configs = ternary_configs(4)
    indices = pack_ternary_to_index(configs)
    recovered = index_to_ternary(indices, group_size=4)
    np.testing.assert_array_equal(configs, recovered)


@test("unpack known indices")
def _():
    r = index_to_ternary(np.array([0]), group_size=4)
    np.testing.assert_array_equal(r[0], [-1, -1, -1, -1])
    r = index_to_ternary(np.array([40]), group_size=4)
    np.testing.assert_array_equal(r[0], [0, 0, 0, 0])
    r = index_to_ternary(np.array([80]), group_size=4)
    np.testing.assert_array_equal(r[0], [1, 1, 1, 1])


@test("compute_lut_entries: [1,-1,0,1] @ [10,20,30,40] = 30")
def _():
    act = np.array([10, 20, 30, 40], dtype=np.int8)
    configs = np.array([[1, -1, 0, 1]], dtype=np.int8)
    result = compute_lut_entries(act, configs)
    assert result[0] == 30


@test("compute_lut_entries: all-zero config → 0")
def _():
    act = np.array([100, 50, -30, 70], dtype=np.int8)
    configs = np.array([[0, 0, 0, 0]], dtype=np.int8)
    assert compute_lut_entries(act, configs)[0] == 0


@test("compute_lut_entries: all-ones config → sum")
def _():
    act = np.array([10, 20, 30, 40], dtype=np.int8)
    configs = np.array([[1, 1, 1, 1]], dtype=np.int8)
    assert compute_lut_entries(act, configs)[0] == 100


@test("compute_lut_entries: 81 entries for full config set")
def _():
    act = np.array([10, 20, 30, 40], dtype=np.int8)
    configs = ternary_configs(4)
    result = compute_lut_entries(act, configs)
    assert result.shape == (81,)


@test("tile_to_lut_indices: basic 4×8 tile")
def _():
    tile = np.array([
        [1, 0, -1, 1, 0, 0, 0, 0],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, -1, -1, -1, -1],
    ], dtype=np.int8)
    indices = tile_to_lut_indices(tile, group_size=4)
    assert indices.shape == (4, 2)
    assert indices[2, 0] == 40  # [0,0,0,0]
    assert indices[2, 1] == 40


@test("tile_to_lut_indices: roundtrip 128×128")
def _():
    rng = np.random.default_rng(42)
    tile = rng.choice([-1, 0, 1], size=(128, 128)).astype(np.int8)
    indices = tile_to_lut_indices(tile, group_size=4)
    reconstructed = index_to_ternary(indices, group_size=4).reshape(128, 128)
    np.testing.assert_array_equal(tile, reconstructed)


@test("tile_to_lut_indices: indivisible cols raises ValueError")
def _():
    tile = np.zeros((4, 7), dtype=np.int8)
    try:
        tile_to_lut_indices(tile, group_size=4)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ======================================================================
# Tiling Tests
# ======================================================================
print("\n=== tile ===")

from bitnet2lut.tile import tile_matrix


@test("exact division: 256×256 with 128×128 blocks → 4 tiles")
def _():
    matrix = np.ones((256, 256), dtype=np.int8)
    tiles, infos = tile_matrix(matrix, 128, 128)
    assert len(tiles) == 4
    assert all(t.shape == (128, 128) for t in tiles)
    assert all(not info.is_padded for info in infos)


@test("padding needed: 300×200 → 6 tiles, all 128×128")
def _():
    matrix = np.ones((300, 200), dtype=np.int8)
    tiles, infos = tile_matrix(matrix, 128, 128, pad_strategy="zero")
    assert len(tiles) == 6
    assert all(t.shape == (128, 128) for t in tiles)


@test("no padding: variable edge tile sizes")
def _():
    matrix = np.ones((300, 200), dtype=np.int8)
    tiles, infos = tile_matrix(matrix, 128, 128, pad_strategy="none")
    assert len(tiles) == 6
    assert tiles[4].shape[0] == 44   # 300 - 256
    assert tiles[5].shape[1] == 72   # 200 - 128


@test("tile contents preserved after tiling + reconstruction")
def _():
    rng = np.random.default_rng(42)
    matrix = rng.integers(-1, 2, size=(256, 256), dtype=np.int8)
    tiles, infos = tile_matrix(matrix, 128, 128)
    reconstructed = np.zeros_like(matrix)
    for tile, info in zip(tiles, infos):
        r, c = info.row_start, info.col_start
        reconstructed[r:r + 128, c:c + 128] = tile
    np.testing.assert_array_equal(matrix, reconstructed)


@test("single tile for small matrix")
def _():
    matrix = np.ones((64, 64), dtype=np.int8)
    tiles, infos = tile_matrix(matrix, 128, 128, pad_strategy="zero")
    assert len(tiles) == 1
    assert infos[0].is_padded
    assert infos[0].original_rows == 64


@test("sequential tile indices")
def _():
    matrix = np.ones((512, 512), dtype=np.int8)
    _, infos = tile_matrix(matrix, 128, 128)
    indices = [info.tile_idx for info in infos]
    assert indices == list(range(len(infos)))


# ======================================================================
# Emulator Tests
# ======================================================================
print("\n=== emulator ===")

from bitnet2lut.emulator import (
    direct_ternary_matvec,
    lut_matvec,
    verify_tile_roundtrip,
)


@test("LUT matvec matches direct: identity-like 4×4")
def _():
    W = np.eye(4, dtype=np.int8)
    x = np.array([10, 20, 30, 40], dtype=np.int8)
    expected = direct_ternary_matvec(W, x)
    indices = tile_to_lut_indices(W, group_size=4)
    actual = lut_matvec(indices, x, group_size=4)
    np.testing.assert_array_equal(expected, actual)


@test("LUT matvec matches direct: random 8×16")
def _():
    rng = np.random.default_rng(123)
    W = rng.choice([-1, 0, 1], size=(8, 16)).astype(np.int8)
    x = rng.integers(-128, 128, size=16, dtype=np.int8)
    expected = direct_ternary_matvec(W, x)
    indices = tile_to_lut_indices(W, group_size=4)
    actual = lut_matvec(indices, x, group_size=4)
    np.testing.assert_array_equal(expected, actual)


@test("LUT matvec matches direct: random 128×128")
def _():
    rng = np.random.default_rng(456)
    W = rng.choice([-1, 0, 1], size=(128, 128)).astype(np.int8)
    x = rng.integers(-128, 128, size=128, dtype=np.int8)
    expected = direct_ternary_matvec(W, x)
    indices = tile_to_lut_indices(W, group_size=4)
    actual = lut_matvec(indices, x, group_size=4)
    np.testing.assert_array_equal(expected, actual)


@test("LUT matvec: all-zero weights → zero output")
def _():
    W = np.zeros((8, 16), dtype=np.int8)
    x = np.full(16, 100, dtype=np.int8)
    indices = tile_to_lut_indices(W, group_size=4)
    result = lut_matvec(indices, x, group_size=4)
    np.testing.assert_array_equal(result, np.zeros(8))


@test("LUT matvec: all-ones weights sum activations")
def _():
    W = np.ones((4, 8), dtype=np.int8)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
    indices = tile_to_lut_indices(W, group_size=4)
    actual = lut_matvec(indices, x, group_size=4)
    np.testing.assert_array_equal(actual, np.array([36, 36, 36, 36]))


@test("LUT matvec: 20 random vectors, same weights")
def _():
    rng = np.random.default_rng(789)
    W = rng.choice([-1, 0, 1], size=(64, 64)).astype(np.int8)
    indices = tile_to_lut_indices(W, group_size=4)
    for _ in range(20):
        x = rng.integers(-128, 128, size=64, dtype=np.int8)
        expected = direct_ternary_matvec(W, x)
        actual = lut_matvec(indices, x, group_size=4)
        np.testing.assert_array_equal(expected, actual)


@test("LUT matvec: various realistic dimensions")
def _():
    rng = np.random.default_rng(101)
    for M, K in [(128, 128), (64, 128), (128, 64), (32, 256)]:
        W = rng.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
        x = rng.integers(-128, 128, size=K, dtype=np.int8)
        expected = direct_ternary_matvec(W, x)
        indices = tile_to_lut_indices(W, group_size=4)
        actual = lut_matvec(indices, x, group_size=4)
        np.testing.assert_array_equal(expected, actual)


@test("verify_tile_roundtrip: valid → True")
def _():
    rng = np.random.default_rng(42)
    W = rng.choice([-1, 0, 1], size=(64, 64)).astype(np.int8)
    indices = tile_to_lut_indices(W, group_size=4)
    assert verify_tile_roundtrip(W, indices, group_size=4) is True


@test("verify_tile_roundtrip: corrupted → False")
def _():
    W = np.ones((4, 4), dtype=np.int8)
    indices = tile_to_lut_indices(W, group_size=4)
    indices[0, 0] = 0  # corrupt
    assert verify_tile_roundtrip(W, indices, group_size=4) is False


# ======================================================================
# FPGA Export Tests
# ======================================================================
print("\n=== export_fpga ===")

import tempfile

from bitnet2lut.export_fpga import indices_to_coe, indices_to_mem


@test("COE file format: header + hex values")
def _():
    indices = np.array([0, 40, 80], dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
        path = Path(f.name)
    indices_to_coe(indices, path, radix=16, bit_width=8)
    content = path.read_text()
    assert "memory_initialization_radix=16;" in content
    assert "memory_initialization_vector=" in content
    assert "00," in content
    assert "28," in content   # 40 = 0x28
    assert "50;" in content   # 80 = 0x50 (last, ends with ;)
    path.unlink()


@test("MEM file format: @addr hex")
def _():
    indices = np.array([10, 20, 30], dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".mem", delete=False) as f:
        path = Path(f.name)
    indices_to_mem(indices, path, bit_width=8)
    content = path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 3
    assert "@0000" in lines[0]
    assert "0A" in lines[0]
    path.unlink()


@test("COE binary radix")
def _():
    indices = np.array([3], dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
        path = Path(f.name)
    indices_to_coe(indices, path, radix=2, bit_width=8)
    content = path.read_text()
    assert "radix=2;" in content
    assert "00000011;" in content
    path.unlink()


@test("Large tile export: 128×32 → 4096 values")
def _():
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 81, size=(128, 32), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".coe", delete=False) as f:
        path = Path(f.name)
    indices_to_coe(indices, path)
    content = path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2 + 128 * 32  # header + data
    path.unlink()


# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED ✓")
    sys.exit(0)
