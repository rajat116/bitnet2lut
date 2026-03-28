"""
Measure activation distributions across all layers of BitNet 2B4T.
Runs a few prompts, collects raw pre-quantization activation values,
and saves histograms + statistics per projection per layer.

Usage:
    python scripts/measure_activations.py \
        --weights-dir outputs/ternary_weights \
        --model microsoft/bitnet-b1.58-2B-4T-bf16
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bitnet2lut.inference import BitNetEmulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("measure_activations")


# Use short prompts — we just need activation statistics, not good outputs
TEST_PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "The speed of light is",
    "Once upon a time there was",
    "The president of the United States is",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default="outputs/ternary_weights")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    parser.add_argument("--output", default="outputs/activation_stats.json")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build emulator with INT8 (baseline) — we want to measure
    # the raw activations before any quantization decision
    emulator = BitNetEmulator(
        weights_dir=args.weights_dir,
        model_path=args.model,
        group_size=4,
        use_lut=True,
        activation_bits=8,
    )

    # Enable collection
    collector = {}
    emulator.activation_collector = collector

    # Run prompts — just 1 new token each, we want activation stats
    for prompt in TEST_PROMPTS:
        logger.info(f"Processing: '{prompt}'")
        input_ids = tokenizer.encode(prompt)
        emulator.generate(input_ids, max_new_tokens=1, temperature=0.0)

    # Remove the internal key
    collector.pop("_current_key", None)

    logger.info(f"Collected activations for {len(collector)} projection keys")

    # Compute statistics per projection
    stats = {}
    for key, activation_list in collector.items():
        # Stack all collected activation vectors for this projection
        all_activations = np.concatenate(activation_list)

        abs_vals = np.abs(all_activations)

        stats[key] = {
            "count": int(len(all_activations)),
            "mean": float(np.mean(all_activations)),
            "std": float(np.std(all_activations)),
            "abs_mean": float(np.mean(abs_vals)),
            "abs_max": float(np.max(abs_vals)),
            "p50": float(np.percentile(abs_vals, 50)),
            "p90": float(np.percentile(abs_vals, 90)),
            "p95": float(np.percentile(abs_vals, 95)),
            "p99": float(np.percentile(abs_vals, 99)),
            "p999": float(np.percentile(abs_vals, 99.9)),
            # Fraction of values in the tails vs center
            "frac_in_top1pct": float(
                np.mean(abs_vals > np.percentile(abs_vals, 99))
            ),
        }

        logger.info(
            f"{key}: mean={stats[key]['mean']:.4f}, "
            f"std={stats[key]['std']:.4f}, "
            f"p99={stats[key]['p99']:.4f}, "
            f"abs_max={stats[key]['abs_max']:.4f}"
        )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved to {out_path}")

    # Print summary: how much of the dynamic range is in the tails?
    logger.info("\n--- Tail analysis (key question for Lloyd-Max) ---")
    logger.info("If p99/abs_max is small, most values cluster near zero")
    logger.info("and non-uniform quantization gives a big win.\n")

    # Save raw activation values for down_proj only (these are the outlier layers)
    raw_down_proj = {}
    for key, activation_list in collector.items():
        if "down_proj" in key:
            all_vals = np.concatenate(activation_list)
            raw_down_proj[key] = all_vals.tolist()

    raw_path = Path("outputs/down_proj_raw_activations.json")
    with open(raw_path, "w") as f:
        json.dump(raw_down_proj, f)
    logger.info(f"Saved raw down_proj activations to {raw_path}")

    for key, s in list(stats.items())[:7]:  # show first layer
        ratio = s["p99"] / s["abs_max"] if s["abs_max"] > 0 else 0
        logger.info(
            f"{key}: p99/abs_max = {ratio:.3f} "
            f"({'clustered — Lloyd-Max helps' if ratio < 0.5 else 'spread — uniform is okay'})"
        )


if __name__ == "__main__":
    main()
