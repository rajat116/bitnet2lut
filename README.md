# bitnet2lut

**Hardware-agnostic weight extraction and LUT conversion pipeline for BitNet b1.58 models.**

Converts ternary {-1, 0, +1} weights from BitNet b1.58 into precomputed lookup tables (LUTs) suitable for zero-arithmetic inference on FPGA BRAM and memristive crossbar arrays.

## What this does

1. **Extract** ternary weights from `microsoft/bitnet-b1.58-2B-4T-bf16` using the same absmean quantization the model uses during training
2. **Tile** weight matrices into configurable blocks (e.g., 128×128 for FPGA BRAM fitting)
3. **Generate LUTs** using the T-MAC methodology: group g=4 ternary weights → 3⁴=81 precomputed partial sums per group
4. **Export** as FPGA BRAM init files (`.coe` for Xilinx, `.mem` for Vivado) + SystemVerilog headers
5. **Verify** numerical correctness against HuggingFace reference (token-for-token match)

## Architecture reference

BitNet b1.58 2B4T:
- 30 transformer layers
- hidden_size=2560, intermediate_size=6912
- 20 attention heads, 5 KV heads (GQA)
- RoPE (θ=500000), ReLU² activation, SubLN, no biases
- Vocab: 128,256 (LLaMA 3 tokenizer)

## Installation

```bash
git clone https://github.com/<your-username>/bitnet2lut.git
cd bitnet2lut

# One-command setup: creates conda env, installs everything, runs tests
./setup.sh

# Activate the environment
conda activate bitnet2lut
```

Or manually:

```bash
conda env create -f environment.yml
conda activate bitnet2lut
pip install -e ".[dev]" --no-deps
python scripts/run_tests.py   # verify everything works
```

> **Note**: Uses CPU-only PyTorch. We don't need GPU — weight extraction
> runs on CPU and the heavy lifting is numpy-based.

## Quick start

```bash
# Step 1: Extract and quantize weights
bitnet2lut extract --model microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir outputs/

# Step 2: Tile weight matrices
bitnet2lut tile --input outputs/ternary_weights/ --block-size 128 --output-dir outputs/tiles/

# Step 3: Generate T-MAC LUTs
bitnet2lut generate-luts --input outputs/tiles/ --group-size 4 --output-dir outputs/luts/

# Step 4: Export for FPGA
bitnet2lut export-fpga --input outputs/luts/ --format coe --output-dir outputs/fpga/

# Step 5: Verify correctness
bitnet2lut verify --model microsoft/bitnet-b1.58-2B-4T-bf16 --lut-dir outputs/luts/

# Or run the full pipeline at once:
bitnet2lut run-all --model microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir outputs/
```

## Project structure

```
bitnet2lut/
├── src/bitnet2lut/
│   ├── __init__.py
│   ├── cli.py              # CLI entry points
│   ├── extract.py           # Step 1: Weight extraction + absmean quantization
│   ├── tile.py              # Step 2: Block tiling with addressing map
│   ├── lut_gen.py           # Step 3: T-MAC style LUT generation
│   ├── export_fpga.py       # Step 4: BRAM init file export (.coe/.mem/.sv)
│   ├── verify.py            # Step 5: Numerical verification
│   ├── emulator.py          # Software LUT-based matmul emulator
│   ├── model_config.py      # BitNet 2B4T architecture constants
│   └── utils.py             # Shared utilities
├── tests/
│   ├── test_extract.py
│   ├── test_tile.py
│   ├── test_lut_gen.py
│   ├── test_export_fpga.py
│   ├── test_emulator.py
│   └── test_verify.py
├── configs/
│   └── default.yaml         # Default pipeline configuration
├── scripts/
│   └── run_pipeline.sh      # Convenience wrapper
├── pyproject.toml
├── LICENSE
└── README.md
```

## How the LUT approach works

Standard inference: `output = W @ x` where W is (M, K) and x is (K,)

T-MAC approach for ternary weights:
1. Group every 4 consecutive weights along K dimension
2. Each group of 4 ternary weights has 3⁴ = 81 possible configurations
3. For each config, precompute: `sum = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]`
   - But since w∈{-1,0,+1}, this is just additions/subtractions of x values
4. At inference time: pack 4 weights → index into LUT → accumulate across groups

On FPGA: LUTs live in BRAM, lookup is a single-cycle read, accumulation uses fabric adders. **Zero DSP slices needed for weight-activation products.**

## Backend support

- `--backend fpga` (default): Xilinx BRAM init files
- `--backend memristive` (planned): Conductance state mapping for HPE ACAM crossbar arrays

## References

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Ma et al., 2024
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)
- [T-MAC](https://arxiv.org/abs/2407.00088) — Wei et al., 2024
- [HPE ACAM](https://arxiv.org/abs/2602.15990) — Memristive analog CAM

## License

MIT
