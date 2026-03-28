#!/usr/bin/env python3
"""Measure the entropy rate of BitNet b1.58 2B4T ternary weight matrices.

PURPOSE:
    Determine whether ternary weights {-1, 0, +1} have exploitable correlations
    between consecutive values. If the conditional entropy rate H_rate is
    significantly below the per-symbol entropy H_1 ≈ 1.560 bits/weight,
    there is room for compression beyond naive encoding.

WHAT IT MEASURES:
    1. H_0: Per-symbol entropy (marginal distribution of {-1,0,+1})
    2. H_1: First-order conditional entropy H(W_i | W_{i-1})
    3. H_2: Second-order conditional entropy H(W_i | W_{i-1}, W_{i-2})
    4. H_3: Third-order conditional entropy H(W_i | W_{i-1}, W_{i-2}, W_{i-3})
    5. Row-direction vs column-direction entropy (memory layout matters)
    6. Per-layer breakdown to see if some layers are more compressible
    7. Zero-run-length distribution (clustered zeros = compressible)

USAGE:
    cd ~/bitnet2lut
    python measure_entropy.py --weights-dir outputs/ternary_weights

    Or if your weights are elsewhere:
    python measure_entropy.py --weights-dir /path/to/ternary_weights

OUTPUT:
    - Console summary with key numbers
    - entropy_report.json with full per-layer results
    - entropy_report.png with visualization

REQUIREMENTS:
    pip install numpy matplotlib  (you almost certainly have these already)
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The 6 main ternary projection names in BitNet 2B4T
PROJECTIONS = [
    "self_attn_q_proj",
    "self_attn_k_proj",
    "self_attn_v_proj",
    "self_attn_o_proj",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_down_proj",
]


def ternary_to_index(w: np.ndarray) -> np.ndarray:
    """Map {-1, 0, +1} → {0, 1, 2} for indexing."""
    return (w + 1).astype(np.int8)


def marginal_entropy(weights_flat: np.ndarray) -> tuple[float, dict]:
    """H_0: entropy of the marginal distribution P(w)."""
    idx = ternary_to_index(weights_flat)
    counts = np.bincount(idx.ravel(), minlength=3)
    total = counts.sum()
    probs = counts / total
    # Avoid log(0)
    H = -np.sum(p * np.log2(p) for p in probs if p > 0)
    dist = {"-1": float(probs[0]), "0": float(probs[1]), "+1": float(probs[2])}
    return float(H), dist


def conditional_entropy(weights_flat: np.ndarray, order: int) -> float:
    """H(W_i | W_{i-1}, ..., W_{i-order}): conditional entropy of order k.
    
    Uses the chain rule: H(W_i | context) = H(context, W_i) - H(context)
    where context = (W_{i-order}, ..., W_{i-1}).
    
    We compute joint entropy of (order+1)-grams minus joint entropy of order-grams.
    """
    idx = ternary_to_index(weights_flat)
    n = len(idx)
    
    if n <= order:
        return 0.0
    
    # Build (order+1)-grams as integer keys via base-3 encoding
    # E.g., for order=2: key = idx[i]*9 + idx[i+1]*3 + idx[i+2]
    num_joint = order + 1
    num_states_joint = 3 ** num_joint
    num_states_context = 3 ** order
    
    # Compute joint counts for (order+1)-grams
    joint_counts = np.zeros(num_states_joint, dtype=np.int64)
    
    # Build the key array efficiently
    # key = sum(idx[i+j] * 3^(num_joint-1-j) for j in 0..num_joint-1)
    powers = 3 ** np.arange(num_joint - 1, -1, -1)  # [3^(k), ..., 3^0]
    
    # Sliding window via stride tricks would be ideal but let's keep it simple
    # and fast enough for ~2B weights
    # Process in chunks to manage memory
    chunk_size = 10_000_000
    for start in range(0, n - order, chunk_size):
        end = min(start + chunk_size, n - order)
        # Build keys for this chunk
        keys = np.zeros(end - start, dtype=np.int64)
        for j in range(num_joint):
            keys += idx[start + j : start + j + (end - start)].astype(np.int64) * int(powers[j])
        # Accumulate counts
        np.add.at(joint_counts, keys, 1)
    
    # Joint entropy H(W_{i-order}, ..., W_i)
    total_joint = joint_counts.sum()
    p_joint = joint_counts / total_joint
    H_joint = -np.sum(p * np.log2(p) for p in p_joint if p > 0)
    
    # Context entropy H(W_{i-order}, ..., W_{i-1})
    # = joint entropy of order-grams
    context_counts = np.zeros(num_states_context, dtype=np.int64)
    powers_ctx = 3 ** np.arange(order - 1, -1, -1)
    
    for start in range(0, n - order + 1, chunk_size):
        end = min(start + chunk_size, n - order + 1)
        keys = np.zeros(end - start, dtype=np.int64)
        for j in range(order):
            keys += idx[start + j : start + j + (end - start)].astype(np.int64) * int(powers_ctx[j])
        np.add.at(context_counts, keys, 1)
    
    total_ctx = context_counts.sum()
    p_ctx = context_counts / total_ctx
    H_ctx = -np.sum(p * np.log2(p) for p in p_ctx if p > 0)
    
    # Conditional entropy = H(joint) - H(context)
    return float(H_joint - H_ctx)


def zero_run_lengths(weights_flat: np.ndarray) -> dict:
    """Analyze distribution of consecutive-zero runs.
    
    If zeros cluster, run-length encoding or similar can exploit this.
    Returns statistics about run lengths.
    """
    is_zero = (weights_flat == 0)
    runs = []
    current_run = 0
    
    for z in is_zero:
        if z:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    if not runs:
        return {"num_runs": 0, "mean_length": 0, "max_length": 0}
    
    runs_arr = np.array(runs)
    run_counter = Counter(runs)
    top_10 = dict(run_counter.most_common(10))
    
    return {
        "num_runs": len(runs),
        "mean_length": float(runs_arr.mean()),
        "median_length": float(np.median(runs_arr)),
        "max_length": int(runs_arr.max()),
        "std_length": float(runs_arr.std()),
        "top_10_lengths": {str(k): v for k, v in top_10.items()},
        "frac_length_1": float(np.sum(runs_arr == 1) / len(runs_arr)),
    }


def analyze_layer(npz_path: Path, layer_idx: int) -> dict:
    """Full entropy analysis for one layer."""
    data = np.load(npz_path)
    
    layer_result = {"layer": layer_idx, "projections": {}}
    
    # Collect all weights for this layer (row-major flattened)
    all_row_weights = []
    all_col_weights = []
    
    for proj in PROJECTIONS:
        key = proj.replace(".", "_")
        if key not in data:
            continue
        
        W = data[key]  # shape: (out_dim, in_dim), int8 in {-1, 0, +1}
        
        # Validate
        unique = set(np.unique(W).tolist())
        assert unique <= {-1, 0, 1}, f"Non-ternary values in {proj}: {unique}"
        
        # Row-major (C order) — how weights are stored in memory
        flat_row = W.flatten(order='C')
        # Column-major — transposed access pattern
        flat_col = W.flatten(order='F')
        
        all_row_weights.append(flat_row)
        all_col_weights.append(flat_col)
        
        # Per-projection analysis
        H0, dist = marginal_entropy(flat_row)
        H1_row = conditional_entropy(flat_row, order=1)
        H1_col = conditional_entropy(flat_col, order=1)
        
        proj_result = {
            "shape": list(W.shape),
            "num_weights": int(W.size),
            "sparsity": float(np.mean(W == 0)),
            "H0_marginal": round(H0, 6),
            "distribution": dist,
            "H1_row_major": round(H1_row, 6),
            "H1_col_major": round(H1_col, 6),
            "better_direction": "col" if H1_col < H1_row else "row",
            "H1_gap_bits": round(H0 - min(H1_row, H1_col), 6),
        }
        
        layer_result["projections"][proj] = proj_result
    
    # Aggregate layer-level stats
    if all_row_weights:
        all_row = np.concatenate(all_row_weights)
        all_col = np.concatenate(all_col_weights)
        
        H0_layer, dist_layer = marginal_entropy(all_row)
        H1_row_layer = conditional_entropy(all_row, order=1)
        H1_col_layer = conditional_entropy(all_col, order=1)
        H2_row_layer = conditional_entropy(all_row, order=2)
        H3_row_layer = conditional_entropy(all_row, order=3)
        
        zero_runs = zero_run_lengths(all_row)
        
        layer_result["aggregate"] = {
            "total_weights": int(all_row.size),
            "sparsity": float(np.mean(all_row == 0)),
            "distribution": dist_layer,
            "H0_marginal": round(H0_layer, 6),
            "H1_row": round(H1_row_layer, 6),
            "H1_col": round(H1_col_layer, 6),
            "H2_row": round(H2_row_layer, 6),
            "H3_row": round(H3_row_layer, 6),
            "compression_headroom_H1": round(H0_layer - min(H1_row_layer, H1_col_layer), 6),
            "compression_headroom_H3": round(H0_layer - H3_row_layer, 6),
            "zero_runs": zero_runs,
        }
    
    return layer_result


def make_plot(results: dict, output_path: Path):
    """Generate visualization of entropy across layers."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot. pip install matplotlib")
        return
    
    layers = []
    H0s = []
    H1_rows = []
    H1_cols = []
    H2s = []
    H3s = []
    
    for lr in results["layers"]:
        if "aggregate" not in lr:
            continue
        agg = lr["aggregate"]
        layers.append(lr["layer"])
        H0s.append(agg["H0_marginal"])
        H1_rows.append(agg["H1_row"])
        H1_cols.append(agg["H1_col"])
        H2s.append(agg["H2_row"])
        H3s.append(agg["H3_row"])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top panel: entropy rates per layer
    ax1.axhline(y=np.log2(3), color='gray', linestyle=':', alpha=0.5, label=f'log₂(3) = {np.log2(3):.3f} (max)')
    ax1.plot(layers, H0s, 'o-', label='H₀ (marginal)', color='#e74c3c', markersize=4)
    ax1.plot(layers, H1_rows, 's-', label='H₁ row-major', color='#3498db', markersize=4)
    ax1.plot(layers, H1_cols, 'd-', label='H₁ col-major', color='#2ecc71', markersize=4)
    ax1.plot(layers, H2s, '^-', label='H₂ row-major', color='#9b59b6', markersize=4)
    ax1.plot(layers, H3s, 'v-', label='H₃ row-major', color='#e67e22', markersize=4)
    ax1.set_ylabel('Entropy (bits/weight)')
    ax1.set_title('BitNet b1.58 2B4T — Ternary Weight Entropy Rate by Layer')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Bottom panel: compression headroom
    headroom_h1 = [h0 - min(h1r, h1c) for h0, h1r, h1c in zip(H0s, H1_rows, H1_cols)]
    headroom_h3 = [h0 - h3 for h0, h3 in zip(H0s, H3s)]
    
    ax2.bar(layers, headroom_h1, alpha=0.7, label='H₀ − H₁ (1st-order headroom)', color='#3498db')
    ax2.bar(layers, headroom_h3, alpha=0.5, label='H₀ − H₃ (3rd-order headroom)', color='#e67e22')
    ax2.set_xlabel('Layer index')
    ax2.set_ylabel('Compression headroom (bits/weight)')
    ax2.set_title('Potential Compression Savings from Exploiting Correlations')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure entropy rate of BitNet ternary weights"
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("outputs/ternary_weights"),
        help="Directory containing layer_XXX.npz files (default: outputs/ternary_weights)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("entropy_report.json"),
        help="Output JSON report path (default: entropy_report.json)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("entropy_report.png"),
        help="Output plot path (default: entropy_report.png)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (default: all). E.g., '0,1,15,29'",
    )
    args = parser.parse_args()
    
    weights_dir = args.weights_dir
    if not weights_dir.exists():
        logger.error(f"Weights directory not found: {weights_dir}")
        logger.error("Run from your bitnet2lut project root, or specify --weights-dir")
        sys.exit(1)
    
    # Find all layer files
    npz_files = sorted(weights_dir.glob("layer_*.npz"))
    if not npz_files:
        logger.error(f"No layer_*.npz files found in {weights_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(npz_files)} layer files in {weights_dir}")
    
    # Filter layers if specified
    if args.layers:
        selected = set(int(x) for x in args.layers.split(","))
        npz_files = [f for f in npz_files if int(f.stem.split("_")[1]) in selected]
        logger.info(f"Analyzing {len(npz_files)} selected layers: {sorted(selected)}")
    
    # Analyze each layer
    results = {"layers": [], "model": "BitNet b1.58 2B4T"}
    
    for npz_path in npz_files:
        layer_idx = int(npz_path.stem.split("_")[1])
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*60}")
        
        layer_result = analyze_layer(npz_path, layer_idx)
        results["layers"].append(layer_result)
        
        # Print summary for this layer
        if "aggregate" in layer_result:
            agg = layer_result["aggregate"]
            logger.info(f"  Weights:    {agg['total_weights']:,}")
            logger.info(f"  Sparsity:   {agg['sparsity']:.4f}")
            logger.info(f"  Dist:       -1={agg['distribution']['-1']:.4f}  "
                        f"0={agg['distribution']['0']:.4f}  "
                        f"+1={agg['distribution']['+1']:.4f}")
            logger.info(f"  H₀ (marg):  {agg['H0_marginal']:.6f} bits/weight")
            logger.info(f"  H₁ (row):   {agg['H1_row']:.6f} bits/weight")
            logger.info(f"  H₁ (col):   {agg['H1_col']:.6f} bits/weight")
            logger.info(f"  H₂ (row):   {agg['H2_row']:.6f} bits/weight")
            logger.info(f"  H₃ (row):   {agg['H3_row']:.6f} bits/weight")
            logger.info(f"  Headroom:   {agg['compression_headroom_H1']:.6f} bits (H1)  "
                        f"{agg['compression_headroom_H3']:.6f} bits (H3)")
            zr = agg["zero_runs"]
            logger.info(f"  Zero runs:  mean={zr['mean_length']:.2f}  "
                        f"max={zr['max_length']}  "
                        f"frac_len1={zr['frac_length_1']:.4f}")
    
    # Global summary across all layers
    all_H0 = [lr["aggregate"]["H0_marginal"] for lr in results["layers"] if "aggregate" in lr]
    all_H1_row = [lr["aggregate"]["H1_row"] for lr in results["layers"] if "aggregate" in lr]
    all_H1_col = [lr["aggregate"]["H1_col"] for lr in results["layers"] if "aggregate" in lr]
    all_H2 = [lr["aggregate"]["H2_row"] for lr in results["layers"] if "aggregate" in lr]
    all_H3 = [lr["aggregate"]["H3_row"] for lr in results["layers"] if "aggregate" in lr]
    
    if all_H0:
        results["global_summary"] = {
            "num_layers": len(all_H0),
            "log2_3": round(np.log2(3), 6),
            "H0_mean": round(np.mean(all_H0), 6),
            "H1_row_mean": round(np.mean(all_H1_row), 6),
            "H1_col_mean": round(np.mean(all_H1_col), 6),
            "H2_row_mean": round(np.mean(all_H2), 6),
            "H3_row_mean": round(np.mean(all_H3), 6),
            "compression_headroom_H1_mean": round(np.mean(all_H0) - min(np.mean(all_H1_row), np.mean(all_H1_col)), 6),
            "compression_headroom_H3_mean": round(np.mean(all_H0) - np.mean(all_H3), 6),
            "interpretation": "",  # Filled below
        }
        
        headroom = results["global_summary"]["compression_headroom_H3_mean"]
        H0_mean = results["global_summary"]["H0_mean"]
        pct = 100 * headroom / H0_mean if H0_mean > 0 else 0
        
        if headroom < 0.01:
            interp = (f"Weights are near-i.i.d. Headroom is only {headroom:.4f} bits "
                      f"({pct:.1f}%). Current encoding is near-optimal. "
                      f"Compression beyond 2 bits/weight is not viable via correlation exploitation.")
        elif headroom < 0.05:
            interp = (f"Small but measurable correlations exist. Headroom is {headroom:.4f} bits "
                      f"({pct:.1f}%). An ANS coder could save ~{pct:.1f}% bandwidth, "
                      f"but the complexity may not justify the gain for FPGA.")
        else:
            interp = (f"Significant correlations exist! Headroom is {headroom:.4f} bits "
                      f"({pct:.1f}%). There is real room for a context-dependent "
                      f"encoder (ANS/range coder) to reduce bandwidth below naive 2-bit packing.")
        
        results["global_summary"]["interpretation"] = interp
        
        logger.info(f"\n{'='*60}")
        logger.info("GLOBAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  log₂(3) = {np.log2(3):.6f} (theoretical max for uniform ternary)")
        logger.info(f"  H₀ mean = {results['global_summary']['H0_mean']:.6f}")
        logger.info(f"  H₁ mean = {min(np.mean(all_H1_row), np.mean(all_H1_col)):.6f} "
                    f"({'col' if np.mean(all_H1_col) < np.mean(all_H1_row) else 'row'}-major)")
        logger.info(f"  H₂ mean = {np.mean(all_H2):.6f}")
        logger.info(f"  H₃ mean = {np.mean(all_H3):.6f}")
        logger.info(f"  Headroom (H0-H3) = {headroom:.6f} bits ({pct:.1f}%)")
        logger.info(f"\n  → {interp}")
    
    # Save JSON report
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nFull report saved to {args.output}")
    
    # Generate plot
    make_plot(results, args.plot)


if __name__ == "__main__":
    main()
