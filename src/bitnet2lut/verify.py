"""Step 5: Verify numerical correctness of the LUT conversion pipeline.

Three levels of verification:

Level 1 — Weight roundtrip:
    Original ternary weights → pack to LUT indices → unpack back
    Must be bit-exact.

Level 2 — Single-layer matvec:
    For a random activation vector x:
        a) W_ternary @ x  (direct integer matmul)
        b) LUT emulator(lut_indices, x)  (our LUT-based computation)
    Must be bit-exact (both are integer arithmetic, no floating point).

Level 3 — Full model token match (optional, requires model download):
    Run the same prompt through:
        a) HuggingFace transformers (BF16 reference)
        b) Our LUT-based layer emulator
    Compare logits at each layer. Due to the quantize-on-the-fly nature
    of BitLinear, the ternary computation should match exactly for the
    linear layers. Differences may arise from RMSNorm / RoPE / attention
    which remain in floating point.
"""

import logging
from pathlib import Path

import numpy as np
try:
    from rich.table import Table
except ImportError:
    Table = None

from .emulator import (
    direct_ternary_matvec,
    lut_matvec,
    verify_tile_roundtrip,
)
from .lut_gen import tile_to_lut_indices
from .model_config import BITLINEAR_PROJECTIONS
from .utils import console, load_json, save_json

logger = logging.getLogger("bitnet2lut")


def verify_level1_roundtrip(
    weights_dir: str | Path,
    lut_dir: str | Path,
    group_size: int = 4,
    layers: list[int] | None = None,
) -> dict:
    """Level 1: Verify weight encoding roundtrip is exact.

    Loads original ternary weights and LUT indices, checks that
    decoding indices reproduces the original weights.
    """
    weights_dir = Path(weights_dir)
    lut_dir = Path(lut_dir)

    proj_keys = [p[0].replace(".", "_") for p in BITLINEAR_PROJECTIONS]

    if layers is None:
        npz_files = sorted(weights_dir.glob("layer_*.npz"))
        layers = [int(f.stem.split("_")[1]) for f in npz_files]

    results = {}
    all_pass = True

    for layer_idx in layers:
        npz_path = weights_dir / f"layer_{layer_idx:03d}.npz"
        data = np.load(npz_path)
        layer_pass = True

        for proj_key in proj_keys:
            weight = data[proj_key]

            # Re-generate LUT indices from weight (rather than loading saved ones)
            # to test the tile_to_lut_indices function
            M, K = weight.shape
            # Need to handle the case where K is not divisible by group_size
            pad_cols = (group_size - K % group_size) % group_size
            if pad_cols > 0:
                weight_padded = np.zeros((M, K + pad_cols), dtype=weight.dtype)
                weight_padded[:, :K] = weight
            else:
                weight_padded = weight

            indices = tile_to_lut_indices(weight_padded, group_size)
            passed = verify_tile_roundtrip(weight_padded, indices, group_size)

            if not passed:
                layer_pass = False
                all_pass = False

            results[f"layer_{layer_idx:03d}.{proj_key}"] = passed

    summary = {
        "level": 1,
        "test": "weight_roundtrip",
        "all_pass": all_pass,
        "num_checks": len(results),
        "num_passed": sum(results.values()),
        "num_failed": sum(not v for v in results.values()),
    }

    _print_verification_result("Level 1: Weight Roundtrip", summary)
    return summary


def verify_level2_matvec(
    weights_dir: str | Path,
    group_size: int = 4,
    num_random_vectors: int = 5,
    layers: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Level 2: Verify LUT-based matvec matches direct ternary matmul.

    For each projection in each layer, generate random INT8 activation
    vectors and verify that lut_matvec produces identical results to
    direct_ternary_matvec.
    """
    weights_dir = Path(weights_dir)
    rng = np.random.default_rng(seed)

    proj_keys = [p[0].replace(".", "_") for p in BITLINEAR_PROJECTIONS]

    if layers is None:
        npz_files = sorted(weights_dir.glob("layer_*.npz"))
        layers = [int(f.stem.split("_")[1]) for f in npz_files]

    results = {}
    all_pass = True
    max_abs_diff = 0

    for layer_idx in layers:
        npz_path = weights_dir / f"layer_{layer_idx:03d}.npz"
        data = np.load(npz_path)

        for proj_key in proj_keys:
            weight = data[proj_key]
            M, K = weight.shape

            # Pad if needed
            pad_cols = (group_size - K % group_size) % group_size
            if pad_cols > 0:
                weight_padded = np.zeros((M, K + pad_cols), dtype=weight.dtype)
                weight_padded[:, :K] = weight
                K_padded = K + pad_cols
            else:
                weight_padded = weight
                K_padded = K

            indices = tile_to_lut_indices(weight_padded, group_size)

            proj_pass = True
            for vec_idx in range(num_random_vectors):
                # Random INT8 activation
                activation = rng.integers(-128, 128, size=K_padded, dtype=np.int8)
                if pad_cols > 0:
                    activation[-pad_cols:] = 0  # zero-pad activations too

                # Direct ternary matmul
                expected = direct_ternary_matvec(weight_padded, activation)

                # LUT-based matmul
                actual = lut_matvec(indices, activation, group_size)

                if not np.array_equal(expected, actual):
                    diff = np.abs(expected - actual)
                    max_diff = diff.max()
                    max_abs_diff = max(max_abs_diff, max_diff)
                    logger.error(
                        f"MISMATCH: layer_{layer_idx:03d}.{proj_key} "
                        f"vec={vec_idx}, max_diff={max_diff}"
                    )
                    proj_pass = False
                    all_pass = False

            name = f"layer_{layer_idx:03d}.{proj_key}"
            results[name] = proj_pass

    summary = {
        "level": 2,
        "test": "lut_matvec_correctness",
        "all_pass": all_pass,
        "num_checks": len(results),
        "num_passed": sum(results.values()),
        "num_failed": sum(not v for v in results.values()),
        "num_random_vectors": num_random_vectors,
        "max_abs_diff": int(max_abs_diff),
        "seed": seed,
    }

    _print_verification_result("Level 2: LUT MatVec", summary)
    return summary


def run_all_verification(
    output_dir: str | Path,
    group_size: int = 4,
    num_random_vectors: int = 5,
    layers: list[int] | None = None,
) -> dict:
    """Run all verification levels.

    Args:
        output_dir: The main pipeline output directory
        group_size: LUT group size
        num_random_vectors: Number of random vectors for Level 2
        layers: Specific layers to verify (None = all)

    Returns:
        Combined verification results
    """
    output_dir = Path(output_dir)
    weights_dir = output_dir / "ternary_weights"

    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Ternary weights not found at {weights_dir}. Run 'extract' first."
        )

    logger.info("=" * 60)
    logger.info("VERIFICATION SUITE")
    logger.info("=" * 60)

    # Level 1
    lut_dir = output_dir / "lut_indices"
    l1 = verify_level1_roundtrip(weights_dir, lut_dir, group_size, layers)

    # Level 2
    l2 = verify_level2_matvec(weights_dir, group_size, num_random_vectors, layers)

    combined = {
        "level1_roundtrip": l1,
        "level2_matvec": l2,
        "all_pass": l1["all_pass"] and l2["all_pass"],
    }

    save_json(combined, output_dir / "verification_results.json")

    logger.info("=" * 60)
    status = "ALL PASSED" if combined["all_pass"] else "FAILURES DETECTED"
    logger.info(f"OVERALL: {status}")
    logger.info("=" * 60)

    return combined


def _print_verification_result(title: str, summary: dict) -> None:
    """Pretty-print verification results."""
    if Table is not None and console is not None:
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green" if summary["all_pass"] else "red")
        table.add_row("Status", "PASS" if summary["all_pass"] else "FAIL")
        table.add_row("Checks", str(summary["num_checks"]))
        table.add_row("Passed", str(summary["num_passed"]))
        table.add_row("Failed", str(summary["num_failed"]))
        console.print(table)
    else:
        status = "PASS" if summary["all_pass"] else "FAIL"
        logger.info(f"{title}: {status} ({summary['num_passed']}/{summary['num_checks']})")
