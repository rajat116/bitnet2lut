#!/bin/bash
# Run the complete bitnet2lut Phase 0 pipeline
# Usage: ./scripts/run_pipeline.sh [--layers 0,1,2] [--block-size 128] [--group-size 4]

set -euo pipefail

MODEL="${MODEL:-microsoft/bitnet-b1.58-2B-4T-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LAYERS="${LAYERS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-4}"

echo "============================================"
echo "bitnet2lut — Phase 0 Pipeline"
echo "============================================"
echo "Model:      ${MODEL}"
echo "Output:     ${OUTPUT_DIR}"
echo "Block size: ${BLOCK_SIZE}"
echo "Group size: ${GROUP_SIZE}"
echo "Layers:     ${LAYERS:-all}"
echo "============================================"

LAYER_FLAG=""
if [ -n "$LAYERS" ]; then
    LAYER_FLAG="--layers $LAYERS"
fi

echo ""
echo "[Step 1/5] Extracting and quantizing weights..."
bitnet2lut extract \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    $LAYER_FLAG

echo ""
echo "[Step 2/5] Tiling weight matrices..."
bitnet2lut tile \
    --input "${OUTPUT_DIR}/ternary_weights" \
    --output-dir "${OUTPUT_DIR}/tiles" \
    --block-size "$BLOCK_SIZE"

echo ""
echo "[Step 3/5] Generating LUT indices..."
bitnet2lut generate-luts \
    --input "${OUTPUT_DIR}/tiles" \
    --output-dir "${OUTPUT_DIR}/luts" \
    --group-size "$GROUP_SIZE"

echo ""
echo "[Step 4/5] Exporting FPGA BRAM init files..."
bitnet2lut export-fpga \
    --input "${OUTPUT_DIR}/luts" \
    --tiling-map "${OUTPUT_DIR}/tiles/tiling_map.json" \
    --lut-summary "${OUTPUT_DIR}/luts/lut_summary.json" \
    --output-dir "${OUTPUT_DIR}/fpga"

echo ""
echo "[Step 5/5] Running verification..."
bitnet2lut verify \
    --output-dir "$OUTPUT_DIR" \
    --group-size "$GROUP_SIZE" \
    $LAYER_FLAG

echo ""
echo "============================================"
echo "Pipeline complete! Outputs in: ${OUTPUT_DIR}/"
echo "============================================"
