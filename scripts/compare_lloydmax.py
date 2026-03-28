"""
Compare three inference paths for BitNet 2B4T down_proj activations:
  1. INT8 uniform absmax (current baseline)
  2. INT4 uniform absmax (naive low-bit, already shown to fail)
  3. Lloyd-Max 4-bit non-uniform (our contribution)
  4. Lloyd-Max 8-bit non-uniform (shows INT8 was also suboptimal)

Usage:
    python scripts/compare_lloydmax.py \
        --weights-dir outputs/ternary_weights \
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \
        --thresholds-4bit outputs/lloydmax_thresholds_4bit.json \
        --thresholds-8bit outputs/lloydmax_thresholds_8bit.json
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
logger = logging.getLogger("compare_lloydmax")


TEST_PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "The speed of light is approximately",
]


def build_emulator(weights_dir, model, activation_bits=8, thresholds_path=None):
    emulator = BitNetEmulator(
        weights_dir=weights_dir,
        model_path=model,
        group_size=4,
        use_lut=True,
        activation_bits=activation_bits,
    )
    if thresholds_path is not None:
        emulator.load_lloydmax_thresholds(thresholds_path)
    return emulator


def run_prompt(emulator, tokenizer, prompt, max_new_tokens=10):
    input_ids = tokenizer.encode(prompt)
    tokens = emulator.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens[len(input_ids):], text


def token_agreement(tokens_a, tokens_b):
    matches = sum(a == b for a, b in zip(tokens_a, tokens_b))
    return matches / max(len(tokens_a), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default="outputs/ternary_weights")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    parser.add_argument("--thresholds-4bit", default="outputs/lloydmax_thresholds_4bit.json")
    parser.add_argument("--thresholds-8bit", default="outputs/lloydmax_thresholds_8bit.json")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build all four emulators
    configs = [
        ("INT8 uniform (baseline)",   8, None),
        ("INT4 uniform (naive)",       4, None),
        ("Lloyd-Max 4-bit (ours)",     8, args.thresholds_4bit),
        ("Lloyd-Max 8-bit (ours)",     8, args.thresholds_8bit),
    ]

    all_results = []

    for prompt in TEST_PROMPTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"{'='*60}")

        prompt_results = {"prompt": prompt, "outputs": {}}
        baseline_tokens = None

        for name, bits, thresh_path in configs:
            logger.info(f"\nRunning: {name}")
            emulator = build_emulator(
                args.weights_dir, args.model, bits, thresh_path
            )
            tokens, text = run_prompt(
                emulator, tokenizer, prompt, args.max_new_tokens
            )
            logger.info(f"Output: '{text}'")

            agreement = (
                token_agreement(baseline_tokens, tokens)
                if baseline_tokens is not None
                else 1.0
            )

            if baseline_tokens is None:
                baseline_tokens = tokens

            prompt_results["outputs"][name] = {
                "text": text,
                "tokens": tokens,
                "agreement_with_baseline": round(agreement * 100, 1),
            }

        all_results.append(prompt_results)

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY — Token agreement with INT8 baseline")
    logger.info(f"{'='*60}")
    logger.info(f"{'Method':<30} " + "  ".join(
        f"P{i+1}" for i in range(len(TEST_PROMPTS))
    ))
    logger.info("-" * 60)

    method_names = [name for name, _, _ in configs]
    for name in method_names:
        agreements = []
        for r in all_results:
            ag = r["outputs"][name]["agreement_with_baseline"]
            agreements.append(f"{ag:>5.1f}%")
        logger.info(f"{name:<30} {'  '.join(agreements)}")

    # Save
    out_path = Path("outputs/lloydmax_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
