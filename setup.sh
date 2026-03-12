#!/bin/bash
# ==============================================================
# bitnet2lut — One-command setup
#
# Usage:
#   ./setup.sh              # Create env + install + run tests
#   ./setup.sh --skip-tests # Create env + install only
#   ./setup.sh --clean      # Remove env and start fresh
# ==============================================================

set -euo pipefail

ENV_NAME="bitnet2lut"
SKIP_TESTS=false
CLEAN=false

for arg in "$@"; do
    case $arg in
        --skip-tests) SKIP_TESTS=true ;;
        --clean)      CLEAN=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [--skip-tests] [--clean]"
            echo "  --skip-tests  Skip running tests after install"
            echo "  --clean       Remove existing env and recreate"
            exit 0
            ;;
    esac
done

# ------------------------------------------------------------------
# Check for conda/mamba
# ------------------------------------------------------------------
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba (faster solver)"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda"
else
    echo "ERROR: Neither conda nor mamba found."
    echo ""
    echo "Install one of:"
    echo "  Miniforge (recommended): https://github.com/conda-forge/miniforge"
    echo "  Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ------------------------------------------------------------------
# Clean if requested
# ------------------------------------------------------------------
if [ "$CLEAN" = true ]; then
    echo "Removing existing environment '${ENV_NAME}'..."
    $CONDA_CMD env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

# ------------------------------------------------------------------
# Create or update conda environment
# ------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists."
    echo "Updating..."
    $CONDA_CMD env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo ""
    echo "Creating conda environment '${ENV_NAME}'..."
    $CONDA_CMD env create -f environment.yml
fi

# ------------------------------------------------------------------
# Activate and install package in dev mode
# ------------------------------------------------------------------
echo ""
echo "Installing bitnet2lut in development mode..."

# We need to use conda run since we can't source activate in a script
# that's already using set -e
conda run -n "$ENV_NAME" pip install -e ".[dev]" --no-deps

# ------------------------------------------------------------------
# Run tests
# ------------------------------------------------------------------
if [ "$SKIP_TESTS" = false ]; then
    echo ""
    echo "============================================"
    echo "Running tests..."
    echo "============================================"
    conda run -n "$ENV_NAME" python scripts/run_tests.py
    echo ""
    echo "Running pytest..."
    conda run -n "$ENV_NAME" pytest tests/ -v --tb=short
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Quick test (layers 0-2 only, ~5 min):"
echo "  bitnet2lut run-all --model microsoft/bitnet-b1.58-2B-4T-bf16 -o outputs/ --layers 0,1,2"
echo ""
echo "Full pipeline (all 30 layers, ~30 min):"
echo "  bitnet2lut run-all --model microsoft/bitnet-b1.58-2B-4T-bf16 -o outputs/"
echo ""
