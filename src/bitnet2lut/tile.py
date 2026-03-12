"""Step 2: Tile ternary weight matrices into configurable blocks.

Why tiling matters:
    FPGA BRAMs have fixed dimensions (e.g., 18Kb = 1024×18 bits for BRAM18).
    A weight matrix like (6912, 2560) doesn't fit in a single BRAM.
    We tile it into blocks (e.g., 128×128) so each block's LUT fits in BRAM.

    The tiling map records which block maps to which (row_start, col_start)
    in the original matrix, enabling the FPGA controller to orchestrate
    the block-sequential computation.

Tiling scheme:
    Given a weight matrix W of shape (M, K):
    - Tile into blocks of size (block_rows, block_cols)
    - If M or K is not divisible, pad with zeros (which are free in ternary)
    - Generate a JSON tiling map for the FPGA controller
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .model_config import BITLINEAR_PROJECTIONS, BitNetConfig, get_weight_name
from .utils import ensure_dir, load_json, save_json

logger = logging.getLogger("bitnet2lut")


@dataclass
class TileInfo:
    """Metadata for a single tile."""

    tile_idx: int
    row_start: int
    col_start: int
    row_end: int
    col_end: int
    # Whether this tile was padded (edge tile)
    is_padded: bool
    # Original (unpadded) dimensions within this tile
    original_rows: int
    original_cols: int


def tile_matrix(
    matrix: np.ndarray,
    block_rows: int = 128,
    block_cols: int = 128,
    pad_strategy: str = "zero",
) -> tuple[list[np.ndarray], list[TileInfo]]:
    """Tile a 2D matrix into fixed-size blocks.

    Args:
        matrix: 2D numpy array of shape (M, K) with int8 ternary values
        block_rows: Number of rows per tile
        block_cols: Number of columns per tile
        pad_strategy: "zero" to pad edge tiles, "none" for variable-size edge tiles

    Returns:
        tiles: List of 2D numpy arrays, each of shape (block_rows, block_cols)
        tile_infos: List of TileInfo metadata for each tile
    """
    M, K = matrix.shape
    tiles = []
    tile_infos = []
    tile_idx = 0

    num_row_tiles = (M + block_rows - 1) // block_rows
    num_col_tiles = (K + block_cols - 1) // block_cols

    for ri in range(num_row_tiles):
        row_start = ri * block_rows
        row_end = min(row_start + block_rows, M)
        orig_rows = row_end - row_start

        for ci in range(num_col_tiles):
            col_start = ci * block_cols
            col_end = min(col_start + block_cols, K)
            orig_cols = col_end - col_start

            block = matrix[row_start:row_end, col_start:col_end]
            is_padded = False

            if pad_strategy == "zero":
                if orig_rows < block_rows or orig_cols < block_cols:
                    padded = np.zeros((block_rows, block_cols), dtype=matrix.dtype)
                    padded[:orig_rows, :orig_cols] = block
                    block = padded
                    is_padded = True

            tiles.append(block)
            tile_infos.append(
                TileInfo(
                    tile_idx=tile_idx,
                    row_start=row_start,
                    col_start=col_start,
                    row_end=row_start + block_rows if pad_strategy == "zero" else row_end,
                    col_end=col_start + block_cols if pad_strategy == "zero" else col_end,
                    is_padded=is_padded,
                    original_rows=orig_rows,
                    original_cols=orig_cols,
                )
            )
            tile_idx += 1

    return tiles, tile_infos


def tile_all_weights(
    weights_dir: str | Path,
    output_dir: str | Path,
    block_rows: int = 128,
    block_cols: int = 128,
    pad_strategy: str = "zero",
    layers: list[int] | None = None,
) -> dict:
    """Tile all extracted ternary weights and generate addressing maps.

    Args:
        weights_dir: Directory containing layer_XXX.npz files from Step 1
        output_dir: Where to save tiled weights and tiling maps
        block_rows: Tile height
        block_cols: Tile width
        pad_strategy: Padding strategy for edge tiles
        layers: Specific layers to process (None = auto-detect)

    Returns:
        Summary dictionary
    """
    weights_dir = Path(weights_dir)
    output_dir = ensure_dir(output_dir)
    tiles_dir = ensure_dir(output_dir / "tiles")

    config = BitNetConfig()

    # Auto-detect available layers
    if layers is None:
        npz_files = sorted(weights_dir.glob("layer_*.npz"))
        layers = [int(f.stem.split("_")[1]) for f in npz_files]

    if not layers:
        raise FileNotFoundError(f"No layer files found in {weights_dir}")

    logger.info(f"Tiling {len(layers)} layers with block size ({block_rows}, {block_cols})")

    global_tiling_map = {}
    total_tiles = 0
    total_padded_tiles = 0

    proj_keys = [p[0].replace(".", "_") for p in BITLINEAR_PROJECTIONS]

    for layer_idx in tqdm(layers, desc="Tiling layers"):
        npz_path = weights_dir / f"layer_{layer_idx:03d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Weight file not found: {npz_path}")

        data = np.load(npz_path)
        layer_tiling = {}

        for proj_key in proj_keys:
            if proj_key not in data:
                raise KeyError(f"Projection '{proj_key}' not found in {npz_path}")

            matrix = data[proj_key]  # shape (out_dim, in_dim), int8
            tiles, tile_infos = tile_matrix(
                matrix, block_rows, block_cols, pad_strategy
            )

            # Save tiles for this projection
            proj_tiles_dir = ensure_dir(
                tiles_dir / f"layer_{layer_idx:03d}" / proj_key
            )
            for tile, info in zip(tiles, tile_infos):
                tile_path = proj_tiles_dir / f"tile_{info.tile_idx:04d}.npy"
                np.save(tile_path, tile)

            # Build tiling map entry
            weight_name = get_weight_name(
                layer_idx, proj_key.replace("_", ".", 1)
            )
            layer_tiling[proj_key] = {
                "weight_name": weight_name,
                "original_shape": list(matrix.shape),
                "block_rows": block_rows,
                "block_cols": block_cols,
                "num_row_tiles": (matrix.shape[0] + block_rows - 1) // block_rows,
                "num_col_tiles": (matrix.shape[1] + block_cols - 1) // block_cols,
                "num_tiles": len(tiles),
                "num_padded_tiles": sum(1 for t in tile_infos if t.is_padded),
                "tiles": [
                    {
                        "tile_idx": t.tile_idx,
                        "row_start": t.row_start,
                        "col_start": t.col_start,
                        "original_rows": t.original_rows,
                        "original_cols": t.original_cols,
                        "is_padded": t.is_padded,
                    }
                    for t in tile_infos
                ],
            }

            total_tiles += len(tiles)
            total_padded_tiles += sum(1 for t in tile_infos if t.is_padded)

        global_tiling_map[f"layer_{layer_idx:03d}"] = layer_tiling

    # Save global tiling map
    tiling_map_path = output_dir / "tiling_map.json"
    save_json(global_tiling_map, tiling_map_path)

    summary = {
        "block_rows": block_rows,
        "block_cols": block_cols,
        "pad_strategy": pad_strategy,
        "num_layers": len(layers),
        "total_tiles": total_tiles,
        "total_padded_tiles": total_padded_tiles,
        "tiles_per_layer": total_tiles // len(layers) if layers else 0,
    }

    save_json(summary, output_dir / "tiling_summary.json")
    logger.info(f"Total tiles: {total_tiles} ({total_padded_tiles} padded)")
    logger.info(f"Tiling map saved to {tiling_map_path}")

    return summary
