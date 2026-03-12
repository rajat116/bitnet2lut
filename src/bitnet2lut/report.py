"""Generate a consolidated statistics and FPGA resource estimation report.

Reads outputs from all pipeline steps and produces:
1. Per-layer weight statistics (sparsity, ternary distribution)
2. Tiling summary (blocks per projection, padding overhead)
3. LUT statistics (index distribution, all-zero fraction)
4. FPGA BRAM resource estimation for common Xilinx targets
5. Total storage requirements
"""

import logging
import math
from pathlib import Path

from .model_config import BITLINEAR_PROJECTIONS, BitNetConfig
from .utils import load_json, save_json

logger = logging.getLogger("bitnet2lut")

# FPGA targets for resource estimation
FPGA_TARGETS = {
    "xcvu9p (Alveo U280)": {
        "bram18": 4320,
        "bram36": 2160,
        "uram": 960,
        "uram_depth": 4096,
        "uram_width": 72,
        "dsp": 6840,
        "lut": 1182240,
    },
    "xcvu37p (Alveo U55C)": {
        "bram18": 3456,
        "bram36": 1728,
        "uram": 1280,
        "uram_depth": 4096,
        "uram_width": 72,
        "dsp": 9024,
        "lut": 1304640,
    },
    "xck26 (KV260)": {
        "bram18": 288,
        "bram36": 144,
        "uram": 64,
        "uram_depth": 4096,
        "uram_width": 72,
        "dsp": 1248,
        "lut": 117120,
    },
    "xcvu13p (VCU1525)": {
        "bram18": 5376,
        "bram36": 2688,
        "uram": 1280,
        "uram_depth": 4096,
        "uram_width": 72,
        "dsp": 12288,
        "lut": 1728000,
    },
}


def generate_report(
    output_dir: str | Path,
    group_size: int = 4,
    block_rows: int = 128,
    block_cols: int = 128,
) -> dict:
    """Generate the full statistics and resource estimation report.

    Args:
        output_dir: Pipeline output directory (must contain weight_stats.json, etc.)
        group_size: LUT group size used
        block_rows: Tile block height
        block_cols: Tile block width

    Returns:
        Report dictionary (also saved as report.json)
    """
    output_dir = Path(output_dir)
    config = BitNetConfig()
    report = {}

    # ================================================================
    # Section 1: Weight statistics
    # ================================================================
    stats_path = output_dir / "weight_stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        summary = stats.get("summary", {})
        per_weight = stats.get("per_weight", [])

        # Aggregate across all weights
        total_params = summary.get("total_ternary_params", 0)
        sparsity = summary.get("sparsity", 0)
        num_layers = summary.get("num_layers_extracted", 0)

        # Per-layer sparsity
        layer_sparsities = {}
        for w in per_weight:
            layer_name = w["name"]
            layer_idx = layer_name.split(".")[2]  # "model.layers.X...."
            if layer_idx not in layer_sparsities:
                layer_sparsities[layer_idx] = {"total": 0, "zeros": 0}
            layer_sparsities[layer_idx]["total"] += w["num_params"]
            layer_sparsities[layer_idx]["zeros"] += w["count_zero"]

        per_layer_sparsity = {
            f"layer_{k}": round(v["zeros"] / v["total"], 4) if v["total"] > 0 else 0
            for k, v in sorted(layer_sparsities.items())
        }

        report["weight_statistics"] = {
            "total_ternary_params": total_params,
            "num_layers": num_layers,
            "overall_sparsity": round(sparsity, 4),
            "per_layer_sparsity": per_layer_sparsity,
            "projections_per_layer": len(BITLINEAR_PROJECTIONS),
        }
    else:
        logger.warning(f"weight_stats.json not found at {stats_path}")
        report["weight_statistics"] = {"error": "weight_stats.json not found"}
        total_params = 0
        num_layers = 0

    # ================================================================
    # Section 2: Tiling summary
    # ================================================================
    tiling_summ_path = output_dir / "tiles" / "tiling_summary.json"
    if tiling_summ_path.exists():
        tiling = load_json(tiling_summ_path)
        report["tiling"] = tiling
    else:
        # Compute from architecture
        tiles_per_projection = {}
        total_tiles = 0
        for proj_name, out_dim, in_dim in BITLINEAR_PROJECTIONS:
            nr = math.ceil(out_dim / block_rows)
            nc = math.ceil(in_dim / block_cols)
            n = nr * nc
            tiles_per_projection[proj_name] = {
                "matrix_shape": [out_dim, in_dim],
                "num_row_tiles": nr,
                "num_col_tiles": nc,
                "total_tiles": n,
            }
            total_tiles += n

        report["tiling"] = {
            "block_rows": block_rows,
            "block_cols": block_cols,
            "tiles_per_projection": tiles_per_projection,
            "tiles_per_layer": total_tiles,
            "total_tiles_all_layers": total_tiles * (num_layers or 30),
        }

    # ================================================================
    # Section 3: LUT statistics
    # ================================================================
    lut_summ_path = output_dir / "luts" / "lut_summary.json"
    if lut_summ_path.exists():
        lut_summary = load_json(lut_summ_path)
        report["lut_statistics"] = lut_summary
    else:
        num_configs = 3**group_size
        bits_per_index = math.ceil(math.log2(num_configs))
        report["lut_statistics"] = {
            "group_size": group_size,
            "num_configs": num_configs,
            "bits_per_index": bits_per_index,
            "note": "computed from parameters, not from actual pipeline run",
        }

    # ================================================================
    # Section 4: FPGA resource estimation
    # ================================================================
    num_configs = 3**group_size
    bits_per_index = math.ceil(math.log2(num_configs))
    groups_per_tile_row = block_cols // group_size
    indices_per_tile = block_rows * groups_per_tile_row
    bits_per_tile = indices_per_tile * bits_per_index

    # BRAM18: 1024 addresses × 18 bits = 18,432 bits
    bram18_capacity_bits = 1024 * 18
    bram18_per_tile = math.ceil(bits_per_tile / bram18_capacity_bits)

    # How many tiles to store one full layer
    tiles_per_layer = 0
    for _, out_dim, in_dim in BITLINEAR_PROJECTIONS:
        nr = math.ceil(out_dim / block_rows)
        nc = math.ceil(in_dim / block_cols)
        tiles_per_layer += nr * nc

    bram18_per_layer = tiles_per_layer * bram18_per_tile
    bram18_all_layers = bram18_per_layer * (num_layers or 30)

    # Runtime LUT storage: at runtime, the FPGA needs to hold the 81-entry LUT
    # for the current group position. Each entry is a partial sum.
    max_partial_sum = group_size * 127  # max INT8 * group_size
    bits_per_lut_entry = math.ceil(math.log2(2 * max_partial_sum + 1))
    runtime_lut_bits = num_configs * bits_per_lut_entry
    runtime_lut_bram18 = math.ceil(runtime_lut_bits / bram18_capacity_bits)

    fpga_estimates = {}
    for target_name, resources in FPGA_TARGETS.items():
        avail_bram18 = resources["bram18"]
        # Tiles that fit simultaneously (for weight indices)
        tiles_on_chip = avail_bram18 // bram18_per_tile if bram18_per_tile > 0 else 0
        # Can we fit one full layer?
        fits_one_layer = avail_bram18 >= bram18_per_layer
        # How many layers fit?
        layers_on_chip = avail_bram18 // bram18_per_layer if bram18_per_layer > 0 else 0

        # URAM alternative: URAM is 4096×72 = 294,912 bits each
        uram_capacity_bits = resources["uram_depth"] * resources["uram_width"]
        uram_per_tile = math.ceil(bits_per_tile / uram_capacity_bits)
        tiles_in_uram = resources["uram"] // uram_per_tile if uram_per_tile > 0 else 0

        fpga_estimates[target_name] = {
            "available_bram18": avail_bram18,
            "available_uram": resources["uram"],
            "available_dsp": resources["dsp"],
            "bram18_per_tile": bram18_per_tile,
            "bram18_per_layer": bram18_per_layer,
            "bram18_all_30_layers": bram18_all_layers,
            "fits_one_layer_in_bram": fits_one_layer,
            "layers_fitting_in_bram": layers_on_chip,
            "tiles_fitting_simultaneously": tiles_on_chip,
            "uram_per_tile": uram_per_tile,
            "tiles_fitting_in_uram": tiles_in_uram,
            "dsp_used_for_linear_layers": 0,  # THE KEY CLAIM: zero DSP
            "note_dsp": "Zero DSP slices for weight-activation products (LUT-only)",
        }

    report["fpga_resource_estimation"] = {
        "parameters": {
            "block_size": [block_rows, block_cols],
            "group_size": group_size,
            "bits_per_index": bits_per_index,
            "indices_per_tile": indices_per_tile,
            "bits_per_tile": bits_per_tile,
            "bram18_per_tile": bram18_per_tile,
            "runtime_lut_entries": num_configs,
            "runtime_lut_entry_bits": bits_per_lut_entry,
            "runtime_lut_bram18": runtime_lut_bram18,
        },
        "per_layer": {
            "tiles_per_layer": tiles_per_layer,
            "bram18_per_layer": bram18_per_layer,
        },
        "full_model_30_layers": {
            "total_tiles": tiles_per_layer * 30,
            "total_bram18": bram18_all_layers,
        },
        "targets": fpga_estimates,
    }

    # ================================================================
    # Section 5: Storage summary
    # ================================================================
    # On-disk storage of LUT indices
    bytes_per_index = 1  # uint8
    total_indices = indices_per_tile * tiles_per_layer * (num_layers or 30)
    total_storage_bytes = total_indices * bytes_per_index

    # .coe / .mem file overhead (text encoding ~3-4x raw)
    estimated_coe_bytes = total_storage_bytes * 4

    report["storage"] = {
        "raw_lut_indices_bytes": total_storage_bytes,
        "raw_lut_indices_MB": round(total_storage_bytes / (1024 * 1024), 1),
        "estimated_coe_files_MB": round(estimated_coe_bytes / (1024 * 1024), 1),
        "ternary_weights_packed_bits": total_params * 2,  # 2 bits per trit
        "ternary_weights_packed_MB": round(total_params * 2 / 8 / (1024 * 1024), 1),
    }

    # ================================================================
    # Section 6: Key claims for paper
    # ================================================================
    report["key_claims"] = {
        "zero_dsp_for_linear_layers": True,
        "computation_method": "BRAM LUT lookup + fabric adder accumulation",
        "weight_representation": f"{bits_per_index}-bit LUT indices (encoding 3^{group_size}={num_configs} ternary configs)",
        "activation_precision": "INT8 (per-token absmax quantization)",
        "no_multiplication_in_linear_layers": True,
        "operations_per_output_element": f"{block_cols // group_size} LUT lookups + {block_cols // group_size - 1} additions per tile",
    }

    # Save report
    report_path = output_dir / "report.json"
    save_json(report, report_path)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    _print_report_summary(report)

    return report


def _print_report_summary(report: dict) -> None:
    """Print a human-readable summary of the report."""
    logger.info("=" * 60)
    logger.info("BITNET2LUT — RESOURCE ESTIMATION REPORT")
    logger.info("=" * 60)

    ws = report.get("weight_statistics", {})
    if "total_ternary_params" in ws:
        logger.info(f"Total ternary params: {ws['total_ternary_params']:,}")
        logger.info(f"Overall sparsity: {ws['overall_sparsity']:.1%}")

    fpga = report.get("fpga_resource_estimation", {})
    params = fpga.get("parameters", {})
    logger.info(f"Bits per LUT index: {params.get('bits_per_index', '?')}")
    logger.info(f"BRAM18 per tile: {params.get('bram18_per_tile', '?')}")

    per_layer = fpga.get("per_layer", {})
    logger.info(f"Tiles per layer: {per_layer.get('tiles_per_layer', '?')}")
    logger.info(f"BRAM18 per layer: {per_layer.get('bram18_per_layer', '?')}")

    full = fpga.get("full_model_30_layers", {})
    logger.info(f"Total BRAM18 (30 layers): {full.get('total_bram18', '?')}")

    targets = fpga.get("targets", {})
    for name, est in targets.items():
        fits = est.get("layers_fitting_in_bram", 0)
        avail = est.get("available_bram18", 0)
        logger.info(f"  {name}: {fits} layers fit ({avail} BRAM18 available)")

    storage = report.get("storage", {})
    logger.info(f"Raw LUT index storage: {storage.get('raw_lut_indices_MB', '?')} MB")
    logger.info(f"DSP slices for linear layers: 0")
    logger.info("=" * 60)
