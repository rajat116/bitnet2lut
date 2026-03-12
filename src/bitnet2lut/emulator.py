"""Software emulator for LUT-based ternary matrix-vector multiplication.

This module implements the exact same computation that will run on the FPGA,
but in Python/NumPy. Used for verification (Step 5) to ensure the LUT
conversion pipeline produces bit-exact results.

Three computation paths that must agree:
    1. PyTorch reference: output = W_bf16 @ x  (using dequantized ternary weights)
    2. Direct ternary:    output = W_ternary @ x_int8  (integer arithmetic)
    3. LUT emulator:      output = lut_matmul(lut_indices, x_int8, group_size)
"""

import logging
from pathlib import Path

import numpy as np

from .lut_gen import compute_lut_entries, index_to_ternary, ternary_configs

logger = logging.getLogger("bitnet2lut")


def lut_matvec(
    lut_indices: np.ndarray,
    activation: np.ndarray,
    group_size: int = 4,
) -> np.ndarray:
    """Perform matrix-vector multiplication using LUT indices.

    This exactly mimics what the FPGA does:
    1. For each group of `group_size` activation values, build a 3^g-entry LUT
    2. For each row, use the stored index to look up the partial sum
    3. Accumulate partial sums across groups

    Args:
        lut_indices: Array of shape (M, K // group_size) with uint8 indices in [0, 3^g-1]
        activation: Array of shape (K,) with INT8 activation values
        group_size: Number of weights per LUT group

    Returns:
        Array of shape (M,) — the matrix-vector product
    """
    M, num_groups = lut_indices.shape
    K = num_groups * group_size

    if activation.shape[0] != K:
        raise ValueError(
            f"Activation length ({activation.shape[0]}) != "
            f"expected K ({K} = {num_groups} groups × {group_size})"
        )

    num_configs = 3**group_size
    configs = ternary_configs(group_size)  # (num_configs, group_size)

    output = np.zeros(M, dtype=np.int32)

    for g_idx in range(num_groups):
        # Slice the activation for this group
        act_slice = activation[g_idx * group_size : (g_idx + 1) * group_size]

        # Build the LUT for this group position: all 81 possible partial sums
        lut = compute_lut_entries(act_slice, configs)  # (num_configs,)

        # For each row, look up the partial sum using the stored index
        row_indices = lut_indices[:, g_idx]  # (M,)
        output += lut[row_indices]

    return output


def lut_matvec_tiled(
    tile_indices_list: list[np.ndarray],
    tile_infos: list[dict],
    activation: np.ndarray,
    output_dim: int,
    group_size: int = 4,
) -> np.ndarray:
    """Perform tiled matrix-vector multiplication using LUT indices.

    Accumulates results across column tiles for each row tile.

    Args:
        tile_indices_list: List of LUT index arrays, one per tile
        tile_infos: List of tile metadata dicts (from tiling_map.json)
        activation: Full activation vector of shape (K,)
        output_dim: Output dimension M
        group_size: Weights per LUT group

    Returns:
        Array of shape (M,) — the full matrix-vector product
    """
    output = np.zeros(output_dim, dtype=np.int32)

    for tile_indices, info in zip(tile_indices_list, tile_infos):
        row_start = info["row_start"]
        col_start = info["col_start"]
        orig_rows = info["original_rows"]
        orig_cols = info["original_cols"]

        # Get the activation slice for this column range
        # Pad to match tile width if needed
        block_cols = tile_indices.shape[1] * group_size
        act_slice = np.zeros(block_cols, dtype=activation.dtype)
        act_end = min(col_start + orig_cols, len(activation))
        copy_len = act_end - col_start
        act_slice[:copy_len] = activation[col_start : col_start + copy_len]

        # Compute partial products for this tile
        tile_output = lut_matvec(tile_indices, act_slice, group_size)

        # Accumulate into the correct output rows
        output[row_start : row_start + orig_rows] += tile_output[:orig_rows]

    return output


def direct_ternary_matvec(
    weight_ternary: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    """Direct ternary matrix-vector multiply (no LUT, just int arithmetic).

    This is the ground truth reference: W_ternary @ activation using
    integer arithmetic. The LUT emulator must match this exactly.

    Args:
        weight_ternary: Array of shape (M, K) with int8 values in {-1, 0, +1}
        activation: Array of shape (K,) with int values

    Returns:
        Array of shape (M,) — the exact integer product
    """
    return weight_ternary.astype(np.int32) @ activation.astype(np.int32)


def verify_tile_roundtrip(
    weight_ternary: np.ndarray,
    lut_indices: np.ndarray,
    group_size: int = 4,
) -> bool:
    """Verify that LUT indices exactly reconstruct the original ternary weights.

    This checks the encoding/decoding roundtrip, independent of any activation.

    Args:
        weight_ternary: Original ternary weights (M, K) int8
        lut_indices: Packed indices (M, K // group_size) uint8

    Returns:
        True if roundtrip is exact
    """
    M, K = weight_ternary.shape
    reconstructed_groups = index_to_ternary(lut_indices, group_size)
    # Shape: (M, K // group_size, group_size)
    reconstructed = reconstructed_groups.reshape(M, K)

    match = np.array_equal(weight_ternary, reconstructed)
    if not match:
        mismatches = np.sum(weight_ternary != reconstructed)
        logger.error(
            f"Roundtrip FAILED: {mismatches} / {M * K} values differ"
        )
    return match
