"""Step 3: Generate T-MAC style lookup tables from tiled ternary weights.

T-MAC LUT methodology (arXiv:2407.00088):

Given a ternary weight matrix W (block_rows × block_cols) and an activation
vector x (block_cols,), the matmul y = W @ x can be computed via LUTs:

1. Along the K dimension (columns), group every `g` consecutive weights.
   Default g=4, giving 3^4 = 81 possible ternary configurations per group.

2. For each group of g weights from one row, the partial sum is:
       partial = w[0]*x[k] + w[1]*x[k+1] + ... + w[g-1]*x[k+g-1]
   Since w ∈ {-1, 0, +1}, this is just additions/subtractions of x values.

3. PRECOMPUTATION (done offline, this module):
   For each group position along K, enumerate all 81 weight configurations.
   For each configuration, record which activations to add/subtract/skip.
   This produces a "LUT recipe" — the FPGA just indexes into it at runtime.

4. AT RUNTIME on FPGA:
   - Pack the actual 4 ternary weights into an index (0..80)
   - Use that index to read the precomputed partial sum from BRAM
   - Accumulate across all groups to get the final output

The key insight: the LUT entries depend on BOTH the weights (known offline)
AND the activations (known only at runtime). So we don't store final sums —
we store the WEIGHT CONFIGURATION that tells the hardware what to do with
the activation values.

For FPGA BRAM, we store the weight configurations as compact indices.
At runtime, the FPGA precomputes the 81-entry LUT from the current
activation slice, then uses the stored weight index to look up the result.

Storage format per group:
    - weight_index: uint8 (0..80), the packed ternary configuration
    - This index maps to a specific add/subtract pattern of g activation values

Total groups per tile: (block_rows × block_cols) / g
Total indices per tile: block_rows × (block_cols / g)
"""

import itertools
import logging
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .utils import ensure_dir, load_json, save_json

logger = logging.getLogger("bitnet2lut")


def ternary_configs(group_size: int = 4) -> np.ndarray:
    """Generate all possible ternary weight configurations for a group.

    For group_size=4: 3^4 = 81 configurations.
    Each config is a tuple of g values from {-1, 0, +1}.

    Returns:
        Array of shape (3^g, g) with int8 values in {-1, 0, +1}
    """
    values = [-1, 0, 1]
    configs = list(itertools.product(values, repeat=group_size))
    return np.array(configs, dtype=np.int8)


def pack_ternary_to_index(weights: np.ndarray) -> np.ndarray:
    """Pack a group of g ternary weights into a LUT index.

    Encoding: map {-1, 0, +1} → {0, 1, 2}, then treat as base-3 number.
    For g=4: index = d[0]*27 + d[1]*9 + d[2]*3 + d[3]
    where d[i] = w[i] + 1 (shifting {-1,0,+1} to {0,1,2})

    Args:
        weights: Array of shape (..., g) with values in {-1, 0, +1}

    Returns:
        Array of shape (...) with uint8 indices in [0, 3^g - 1]
    """
    g = weights.shape[-1]
    digits = (weights + 1).astype(np.uint8)  # {-1,0,+1} → {0,1,2}

    # Base-3 encoding: most significant digit first
    powers = np.array([3 ** (g - 1 - i) for i in range(g)], dtype=np.uint8)
    index = np.sum(digits * powers, axis=-1).astype(np.uint8)
    return index


def index_to_ternary(index: np.ndarray, group_size: int = 4) -> np.ndarray:
    """Decode a LUT index back to ternary weights (inverse of pack_ternary_to_index).

    Args:
        index: Array of indices in [0, 3^g - 1]
        group_size: Number of weights per group

    Returns:
        Array of shape (*index.shape, group_size) with values in {-1, 0, +1}
    """
    result = np.zeros((*index.shape, group_size), dtype=np.int8)
    remainder = index.astype(np.int32).copy()

    for i in range(group_size):
        power = 3 ** (group_size - 1 - i)
        digit = remainder // power
        remainder = remainder % power
        result[..., i] = digit.astype(np.int8) - 1  # {0,1,2} → {-1,0,+1}

    return result


def compute_lut_entries(
    activation_slice: np.ndarray,
    configs: np.ndarray,
) -> np.ndarray:
    """Compute LUT entries for a group position given activation values.

    This is what happens AT RUNTIME on the FPGA. Included here for the
    software emulator and verification.

    Args:
        activation_slice: Array of shape (g,) — the g activation values
        configs: Array of shape (num_configs, g) — all weight configurations

    Returns:
        Array of shape (num_configs,) — partial sums for each configuration
    """
    # Each entry: sum of config[i] * activation[i] for i in 0..g-1
    # Since config values are {-1,0,+1}, this is add/skip/subtract
    return np.sum(configs.astype(np.int32) * activation_slice.astype(np.int32), axis=-1)


def tile_to_lut_indices(
    tile: np.ndarray,
    group_size: int = 4,
) -> np.ndarray:
    """Convert a tiled ternary weight block into packed LUT indices.

    Args:
        tile: Array of shape (block_rows, block_cols) with int8 ternary values
        group_size: Number of weights per LUT group

    Returns:
        Array of shape (block_rows, block_cols // group_size) with uint8 indices
    """
    block_rows, block_cols = tile.shape

    if block_cols % group_size != 0:
        raise ValueError(
            f"block_cols ({block_cols}) must be divisible by group_size ({group_size}). "
            f"Ensure tiling produces compatible dimensions."
        )

    num_groups = block_cols // group_size

    # Reshape to (block_rows, num_groups, group_size)
    grouped = tile.reshape(block_rows, num_groups, group_size)

    # Pack each group into a LUT index
    indices = pack_ternary_to_index(grouped)  # (block_rows, num_groups)

    return indices


def generate_luts_for_all_tiles(
    tiles_dir: str | Path,
    output_dir: str | Path,
    group_size: int = 4,
    lut_dtype: str = "int16",
    layers: list[int] | None = None,
) -> dict:
    """Generate LUT index arrays for all tiled weights.

    Args:
        tiles_dir: Directory containing tiled .npy files from Step 2
        output_dir: Where to save LUT index files
        group_size: Ternary weights per LUT group (default 4)
        lut_dtype: Data type for LUT entries (for future runtime use)
        layers: Specific layers (None = auto-detect)

    Returns:
        Summary dictionary
    """
    tiles_dir = Path(tiles_dir)
    output_dir = ensure_dir(output_dir)
    lut_dir = ensure_dir(output_dir / "lut_indices")

    num_configs = 3**group_size
    logger.info(
        f"Generating LUTs: group_size={group_size}, "
        f"num_configs={num_configs}, dtype={lut_dtype}"
    )

    # Pre-generate all ternary configurations (for reference/export)
    configs = ternary_configs(group_size)
    np.save(output_dir / "ternary_configs.npy", configs)

    # Auto-detect layers
    if layers is None:
        layer_dirs = sorted(
            [d for d in (tiles_dir / "tiles").iterdir() if d.is_dir()]
        )
    else:
        layer_dirs = [tiles_dir / "tiles" / f"layer_{l:03d}" for l in layers]

    total_groups = 0
    index_distribution = np.zeros(num_configs, dtype=np.int64)

    for layer_dir in tqdm(layer_dirs, desc="Generating LUT indices"):
        if not layer_dir.exists():
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

        layer_name = layer_dir.name
        layer_lut_dir = ensure_dir(lut_dir / layer_name)

        for proj_dir in sorted(layer_dir.iterdir()):
            if not proj_dir.is_dir():
                continue

            proj_lut_dir = ensure_dir(layer_lut_dir / proj_dir.name)

            for tile_path in sorted(proj_dir.glob("tile_*.npy")):
                tile = np.load(tile_path)

                # Convert tile to LUT indices
                indices = tile_to_lut_indices(tile, group_size)

                # Save indices
                out_path = proj_lut_dir / tile_path.name
                np.save(out_path, indices)

                # Track statistics
                total_groups += indices.size
                for idx_val in range(num_configs):
                    index_distribution[idx_val] += np.sum(indices == idx_val)

    # The all-zeros config (index for [0,0,0,0]) should be most common
    # due to weight sparsity
    zero_config_idx = pack_ternary_to_index(
        np.zeros(group_size, dtype=np.int8).reshape(1, -1)
    )[0]

    summary = {
        "group_size": group_size,
        "num_configs": num_configs,
        "total_groups": int(total_groups),
        "all_zero_config_index": int(zero_config_idx),
        "all_zero_config_fraction": float(
            index_distribution[zero_config_idx] / total_groups
        )
        if total_groups > 0
        else 0.0,
        "lut_dtype": lut_dtype,
        "num_layers_processed": len(layer_dirs),
    }

    # Save index distribution (useful for understanding sparsity patterns)
    np.save(output_dir / "index_distribution.npy", index_distribution)
    save_json(summary, output_dir / "lut_summary.json")

    logger.info(f"Total LUT groups: {total_groups:,}")
    logger.info(
        f"All-zero config (idx={zero_config_idx}) frequency: "
        f"{summary['all_zero_config_fraction']:.2%}"
    )
    logger.info(f"LUT indices saved to {lut_dir}")

    return summary
