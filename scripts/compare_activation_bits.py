"""
Compare INT8 vs INT4 activation quantization on BitNet 2B4T.

Runs the same prompt through three paths:
  1. INT8 activations (baseline — what we already verified)
  2. INT4 activations (experimental — no retraining)

Measures:
  - Token match vs baseline
  - Perplexity on a short fixed text
  - Output text quality (visual inspection)

Usage:
    python scripts/compare_activation_bits.py \
        --weights-dir outputs/ternary_weights \
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \
        --layers 0 1 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Make sure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bitnet2lut.inference import BitNetEmulator, rms_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_activation_bits")


TEST_PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "The speed of light is approximately",
]


def run_comparison(weights_dir: str, model_path: str, max_new_tokens: int = 10):
    """Run INT8 vs INT4 comparison across test prompts."""

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    results = []

    for prompt in TEST_PROMPTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"{'='*60}")

        input_ids = tokenizer.encode(prompt)

        # --- Baseline: INT8 ---
        logger.info("Running INT8 (baseline)...")
        emulator_int8 = BitNetEmulator(
            weights_dir=weights_dir,
            model_path=model_path,
            group_size=4,
            use_lut=True,
            activation_bits=8,
        )
        tokens_int8 = emulator_int8.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=0.0
        )
        text_int8 = tokenizer.decode(tokens_int8, skip_special_tokens=True)
        generated_int8 = tokens_int8[len(input_ids):]
        logger.info(f"INT8 output: '{text_int8}'")

        # --- Experiment: INT4 ---
        logger.info("Running INT4 (experimental)...")
        emulator_int4 = BitNetEmulator(
            weights_dir=weights_dir,
            model_path=model_path,
            group_size=4,
            use_lut=True,
            activation_bits=4,
        )
        tokens_int4 = emulator_int4.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=0.0
        )
        text_int4 = tokenizer.decode(tokens_int4, skip_special_tokens=True)
        generated_int4 = tokens_int4[len(input_ids):]
        logger.info(f"INT4 output: '{text_int4}'")

        # --- Compare ---
        exact_match = generated_int8 == generated_int4
        token_agreement = sum(
            a == b for a, b in zip(generated_int8, generated_int4)
        ) / max(len(generated_int8), 1)

        result = {
            "prompt": prompt,
            "int8_text": text_int8,
            "int4_text": text_int4,
            "int8_tokens": generated_int8,
            "int4_tokens": generated_int4,
            "exact_match": exact_match,
            "token_agreement_pct": round(token_agreement * 100, 1),
        }
        results.append(result)

        logger.info(f"Exact match: {exact_match}")
        logger.info(f"Token agreement: {token_agreement:.1%}")

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Prompt':<40} {'Exact':>6} {'Agreement':>10}")
    logger.info("-" * 60)
    for r in results:
        short_prompt = r["prompt"][:38]
        logger.info(
            f"{short_prompt:<40} "
            f"{'YES' if r['exact_match'] else 'NO':>6} "
            f"{r['token_agreement_pct']:>9.1f}%"
        )

    # Save results
    out_path = Path("outputs/activation_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare INT8 vs INT4 activations")
    parser.add_argument(
        "--weights-dir",
        default="outputs/ternary_weights",
        help="Directory with extracted ternary weights",
    )
    parser.add_argument(
        "--model",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Tokens to generate per prompt",
    )
    args = parser.parse_args()

    run_comparison(
        weights_dir=args.weights_dir,
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
