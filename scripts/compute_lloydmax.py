"""
Compute Lloyd-Max optimal quantization thresholds for BitNet 2B4T
down_proj activations.

Lloyd-Max finds thresholds and reconstruction levels that minimize
mean squared quantization error for a given distribution.

Algorithm (iterative):
    1. Start with uniform thresholds across the data range
    2. Reconstruction levels = conditional mean of data in each bin
    3. New thresholds = midpoints between adjacent reconstruction levels
    4. Repeat until convergence

Usage:
    python scripts/compute_lloydmax.py \
        --input outputs/down_proj_raw_activations.json \
        --bits 4 \
        --output outputs/lloydmax_thresholds.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lloydmax")


def lloyd_max(data: np.ndarray, n_levels: int, n_iter: int = 100) -> tuple:
    """
    Compute Lloyd-Max optimal quantization thresholds and levels.

    Args:
        data: 1D array of float values (the activation distribution)
        n_levels: Number of quantization levels (e.g., 16 for 4-bit, 256 for 8-bit)
        n_iter: Maximum iterations

    Returns:
        thresholds: (n_levels - 1,) array of decision boundaries
        levels: (n_levels,) array of reconstruction values
    """
    # Initialize: uniform thresholds across data range
    data_min = np.min(data)
    data_max = np.max(data)

    # Start with uniform levels
    levels = np.linspace(data_min, data_max, n_levels)

    prev_distortion = np.inf

    for iteration in range(n_iter):
        # Step 1: Thresholds = midpoints between adjacent levels
        thresholds = (levels[:-1] + levels[1:]) / 2.0

        # Step 2: Assign each data point to nearest level
        # np.digitize gives bin index
        bin_indices = np.digitize(data, thresholds)
        # bin_indices is in [0, n_levels-1]

        # Step 3: Reconstruction level = conditional mean of data in each bin
        new_levels = np.zeros(n_levels)
        for i in range(n_levels):
            mask = bin_indices == i
            if np.any(mask):
                new_levels[i] = np.mean(data[mask])
            else:
                # Empty bin — keep old level
                new_levels[i] = levels[i]

        # Check convergence: mean squared quantization error
        quantized = new_levels[bin_indices]
        distortion = np.mean((data - quantized) ** 2)

        levels = new_levels

        if iteration % 10 == 0:
            logger.info(f"  Iter {iteration}: MSE = {distortion:.6f}")

        # Converged if distortion change is tiny
        if abs(prev_distortion - distortion) < 1e-10:
            logger.info(f"  Converged at iteration {iteration}")
            break

        prev_distortion = distortion

    # Final thresholds
    thresholds = (levels[:-1] + levels[1:]) / 2.0

    return thresholds, levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/down_proj_raw_activations.json",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Number of bits (4 = 16 levels, 8 = 256 levels)",
    )
    parser.add_argument(
        "--output",
        default="outputs/lloydmax_thresholds.json",
    )
    args = parser.parse_args()

    n_levels = 2 ** args.bits
    logger.info(f"Computing Lloyd-Max with {args.bits} bits ({n_levels} levels)")

    # Load raw activations
    with open(args.input) as f:
        raw = json.load(f)

    logger.info(f"Loaded {len(raw)} projection keys")

    results = {}

    for key, values in raw.items():
        data = np.array(values, dtype=np.float32)
        logger.info(f"\n{key}: {len(data)} values, "
                    f"range=[{data.min():.3f}, {data.max():.3f}], "
                    f"std={data.std():.3f}")

        thresholds, levels = lloyd_max(data, n_levels)

        # Compare uniform vs Lloyd-Max distortion
        # Uniform: absmax quantization
        abs_max = np.max(np.abs(data))
        uniform_scale = abs_max / (n_levels // 2 - 1)
        uniform_quantized = np.clip(
            np.round(data / uniform_scale), -(n_levels // 2), n_levels // 2 - 1
        ) * uniform_scale
        uniform_mse = np.mean((data - uniform_quantized) ** 2)

        # Lloyd-Max distortion
        bin_indices = np.digitize(data, thresholds)
        lm_quantized = levels[bin_indices]
        lm_mse = np.mean((data - lm_quantized) ** 2)

        improvement = (uniform_mse - lm_mse) / uniform_mse * 100
        logger.info(f"  Uniform MSE:   {uniform_mse:.6f}")
        logger.info(f"  Lloyd-Max MSE: {lm_mse:.6f}")
        logger.info(f"  Improvement:   {improvement:.1f}%")

        results[key] = {
            "bits": args.bits,
            "n_levels": n_levels,
            "thresholds": thresholds.tolist(),
            "levels": levels.tolist(),
            "uniform_mse": float(uniform_mse),
            "lloydmax_mse": float(lm_mse),
            "improvement_pct": float(improvement),
        }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved thresholds to {out_path}")

    # Summary
    logger.info("\n--- Summary ---")
    for key, r in results.items():
        logger.info(
            f"{key}: {r['improvement_pct']:.1f}% MSE reduction vs uniform"
        )


if __name__ == "__main__":
    main()
