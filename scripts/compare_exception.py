"""
Compare exception-aware INT4 vs INT8 baseline.

Usage:
    python scripts/compare_exception.py \
        --weights-dir outputs/ternary_weights \
        --model microsoft/bitnet-b1.58-2B-4T-bf16 \
        --stats outputs/activation_stats.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet2lut.inference import BitNetEmulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_exception")

TEST_PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "The speed of light is approximately",
]


def run(emulator, tokenizer, prompt, max_new_tokens=10):
    input_ids = tokenizer.encode(prompt)
    tokens = emulator.generate(
        input_ids, max_new_tokens=max_new_tokens, temperature=0.0
    )
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens[len(input_ids):], text


def agreement(a, b):
    return sum(x == y for x, y in zip(a, b)) / max(len(a), 1) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default="outputs/ternary_weights")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    parser.add_argument("--stats", default="outputs/activation_stats.json")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    configs = [
        ("INT8 uniform (baseline)", 8, False),
        ("INT4 uniform (naive)",    4, False),
        ("Exception-aware INT4",    8, True),
    ]

    all_results = []

    for prompt in TEST_PROMPTS:
        logger.info(f"\n{'='*55}")
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"{'='*55}")

        prompt_results = {"prompt": prompt, "outputs": {}}
        baseline_tokens = None

        for name, bits, use_exc in configs:
            logger.info(f"\nRunning: {name}")
            emulator = BitNetEmulator(
                weights_dir=args.weights_dir,
                model_path=args.model,
                group_size=4,
                use_lut=True,
                activation_bits=bits,
            )
            if use_exc:
                emulator.load_exception_thresholds(args.stats)

            tokens, text = run(emulator, tokenizer, prompt, args.max_new_tokens)
            logger.info(f"Output: '{text}'")

            ag = agreement(baseline_tokens, tokens) if baseline_tokens else 100.0
            if baseline_tokens is None:
                baseline_tokens = tokens

            prompt_results["outputs"][name] = {
                "text": text,
                "tokens": tokens,
                "agreement": round(ag, 1),
            }

        all_results.append(prompt_results)

    logger.info(f"\n{'='*55}")
    logger.info("SUMMARY")
    logger.info(f"{'='*55}")
    logger.info(f"{'Method':<30} P1      P2      P3")
    logger.info("-" * 55)
    for name, _, _ in configs:
        row = "  ".join(
            f"{r['outputs'][name]['agreement']:5.1f}%"
            for r in all_results
        )
        logger.info(f"{name:<30} {row}")

    out = Path("outputs/exception_comparison.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
