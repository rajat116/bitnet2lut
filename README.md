# bitnet2lut

**Weight extraction, LUT conversion, and inference emulation pipeline for BitNet b1.58 on FPGA.**

Converts ternary {-1, 0, +1} weights from BitNet b1.58 into precomputed lookup tables (LUTs) for zero-arithmetic inference on FPGA. Includes a full standalone inference emulator that produces reference token sequences the hardware must match.

## What this does

1. **Extract** ternary weights from `microsoft/bitnet-b1.58-2B-4T-bf16` using absmean quantization
2. **Tile** weight matrices into configurable blocks (default 128×128 for FPGA BRAM)
3. **Generate LUT indices** using the T-MAC methodology: group g=4 ternary weights → 3⁴=81 precomputed partial sums
4. **Export** FPGA BRAM init files (`.coe` for Xilinx, `.mem` for Vivado, `.svh` SystemVerilog headers)
5. **Verify** correctness at three levels: weight roundtrip, LUT matvec, and real model activations
6. **Emulate** full 30-layer inference using LUT-based matmul, producing token-by-token output
7. **Report** FPGA resource estimates (BRAM utilization, DSP usage, storage requirements)

## Key result

```
✅ LUT emulator matches direct ternary matmul (token-for-token identical)
```

The LUT-based linear layers produce bit-exact results compared to direct ternary matrix-vector multiplication across all 30 layers and all 7 projections per layer. This validates that FPGA LUT lookups can replace arithmetic for the entire model.

## Architecture reference

BitNet b1.58 2B4T (2.08B ternary parameters):

- 30 transformer layers, hidden_size=2560, intermediate_size=6912
- 20 attention heads, 5 KV heads (GQA), head_dim=128
- RoPE (θ=500000), ReLU² activation, SubLN normalization, no biases
- Vocab: 128,256 (LLaMA 3 tokenizer)
- 7 BitLinear projections per layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Installation

```bash
git clone https://github.com/rajat116/bitnet2lut.git
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
pip install accelerate>=0.25.0
pip install -e ".[dev]" --no-deps
```

> **Note**: Uses CPU-only PyTorch. Weight extraction and LUT emulation run entirely on CPU.

## Quick start

### Full pipeline (extract → tile → LUT → export → verify)

```bash
bitnet2lut run-all -m microsoft/bitnet-b1.58-2B-4T-bf16 -o outputs/
```

This runs all steps sequentially for all 30 layers.

### Individual steps

```bash
# Step 1: Extract and quantize ternary weights
bitnet2lut extract -m microsoft/bitnet-b1.58-2B-4T-bf16 -o outputs/

# Step 2: Tile weight matrices into blocks
bitnet2lut tile -i outputs/ternary_weights/ -b 128 -o outputs/tiles/

# Step 3: Generate T-MAC LUT indices
bitnet2lut generate-luts -i outputs/tiles/ -g 4 -o outputs/luts/

# Step 4: Export for FPGA
bitnet2lut export-fpga -i outputs/luts/ -f coe -o outputs/fpga/

# Step 5: Verify (Level 1: roundtrip, Level 2: matvec)
bitnet2lut verify -o outputs/ --level 2
```

### Verification

```bash
# Level 3: verify against real HuggingFace model activations
bitnet2lut verify-model -m microsoft/bitnet-b1.58-2B-4T-bf16 -o outputs/ --layer 0

# Token comparison: LUT emulator vs HF reference vs direct ternary
bitnet2lut compare-tokens \
  -m microsoft/bitnet-b1.58-2B-4T-bf16 \
  -o outputs/ \
  --prompt "The capital of France is" \
  --max-tokens 3
```

### FPGA resource report

```bash
bitnet2lut report -o outputs/ --block-size 128
```

## Verification results

| Level | Test | Result |
|-------|------|--------|
| 1 | Weight roundtrip (ternary → LUT index → ternary) | 21/21 PASS |
| 2 | LUT matvec vs direct ternary matvec | 21/21 PASS |
| 3 | Real model activations (all 7 projections, layer 0) | ALL MATCH |
| Token | LUT emulator vs direct ternary (full 30-layer generation) | EXACT MATCH |

LUT vs HuggingFace outputs differ slightly due to floating-point precision (our emulator runs norms/attention in float32, HF uses BF16). The HF top token consistently ranks within top-3 in our emulator's logits — confirming correct model behavior. The ternary linear layer computation (the FPGA target) is bit-exact.

## FPGA resource estimates (block_size=128)

| Metric | Value |
|--------|-------|
| Total ternary parameters | 2,084,044,800 |
| Weight sparsity | 42.2% |
| Bits per LUT index | 7 |
| BRAM18 per tile | 2 |
| Total BRAM18 (30 layers) | 254,400 |
| Raw LUT index storage | 496.9 MB |
| DSP slices for linear layers | **0** |

The full model does not fit on-chip in any current FPGA. The intended architecture streams tiles from HBM one at a time.

## How the LUT approach works

Standard inference computes `output = W @ x` where W is ternary and x is INT8-quantized.

The T-MAC LUT approach replaces multiplication with table lookup:

1. Group every 4 consecutive ternary weights along the input dimension
2. Each group has 3⁴ = 81 possible weight configurations
3. Pack the group into a single index ∈ [0, 80] (base-3 encoding)
4. At runtime: build an 81-entry LUT from the current INT8 activation slice, then look up partial sums by index
5. Accumulate partial sums across all groups to get the output

On FPGA: LUT indices live in BRAM, lookup is a single-cycle read, accumulation uses fabric adders. **Zero DSP slices needed for weight-activation products.**

## Project structure

```
bitnet2lut/
├── src/bitnet2lut/
│   ├── __init__.py
│   ├── cli.py              # Click CLI with all commands
│   ├── extract.py          # Weight extraction + absmean quantization
│   ├── tile.py             # Block tiling with addressing map
│   ├── lut_gen.py          # T-MAC LUT index generation
│   ├── export_fpga.py      # BRAM init files (.coe/.mem/.svh)
│   ├── verify.py           # Level 1 + Level 2 verification
│   ├── verify_model.py     # Level 3 verification against HF model
│   ├── emulator.py         # LUT-based and direct ternary matvec
│   ├── inference.py         # Full 30-layer inference emulator
│   ├── report.py           # FPGA resource estimation
│   ├── model_config.py     # BitNet 2B4T architecture constants
│   └── utils.py            # Shared utilities
├── tests/
│   ├── test_extract.py
│   ├── test_tile.py
│   ├── test_lut_gen.py
│   ├── test_export_fpga.py
│   ├── test_emulator.py
│   └── test_verify.py
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_pipeline.sh
│   └── run_tests.py
├── environment.yml
├── pyproject.toml
├── setup.sh
└── README.md
```

## Pipeline output structure

After running `bitnet2lut run-all`, the output directory contains:

```
outputs/
├── ternary_weights/          # Extracted int8 ternary weights per layer
│   ├── layer_000.npz         # {self_attn_q_proj, self_attn_k_proj, ...}
│   ├── layer_000_alphas.json # Absmean scale factors per projection
│   ├── layer_001.npz
│   └── ...
├── tiles/                    # Tiled weight blocks
│   ├── layer_000/
│   │   ├── self_attn_q_proj_tile_*.npy
│   │   └── ...
│   └── tiling_map.json       # Block addressing map
├── luts/                     # LUT indices (packed base-3)
│   ├── layer_000/
│   │   └── *.npy
│   └── ...
└── fpga/                     # FPGA BRAM init files
    ├── layer_000/
    │   ├── *.coe              # Xilinx COE format
    │   ├── *.mem              # Vivado MEM format
    │   └── *.svh              # SystemVerilog headers
    └── ...
```

## Roadmap

- [x] **Phase 0**: Weight extraction, LUT conversion, verification, inference emulator
- [ ] **Phase 1**: Profile bitnet.cpp / T-MAC CPU baselines (tokens/sec, energy)
- [ ] **Phase 2**: FPGA RTL in SystemVerilog (streaming tile architecture)
- [ ] **Phase 3**: On-hardware inference and benchmarking

## References

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Ma et al., 2024
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)
- [T-MAC: LUT-based inference](https://arxiv.org/abs/2407.00088) — Wei et al., 2024
- [HPE ACAM memristive computing](https://arxiv.org/abs/2602.15990)

## License

MIT
