"""Step 1: Extract weights from BitNet b1.58 BF16 model and quantize to ternary.

The BitNet b1.58 quantization scheme (arXiv:2402.17764):
    alpha = mean(|W|)                    # per-tensor absmean
    W_ternary = clip(round(W / alpha), -1, +1)   # round to {-1, 0, +1}

We apply this to the BF16 master weights to recover the exact ternary values
used during training. The BF16 repo stores the full-precision weights; during
the forward pass, BitLinear quantizes them on-the-fly using the formula above.

Important: We use the bf16 repo (microsoft/bitnet-b1.58-2B-4T-bf16), NOT the
packed repo (microsoft/bitnet-b1.58-2B-4T) which stores weights in a custom
packed format that's harder to unpack.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from .model_config import (
    BITLINEAR_PROJECTIONS,
    BitNetConfig,
    get_weight_name,
)
from .utils import ensure_dir, format_count, format_size, save_json

logger = logging.getLogger("bitnet2lut")


@dataclass
class WeightStats:
    """Statistics for a single ternary weight matrix."""

    name: str
    shape: tuple[int, int]
    num_params: int
    # Ternary distribution
    count_neg1: int
    count_zero: int
    count_pos1: int
    frac_neg1: float
    frac_zero: float
    frac_pos1: float
    # Quantization parameter
    absmean_alpha: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "num_params": self.num_params,
            "count_neg1": self.count_neg1,
            "count_zero": self.count_zero,
            "count_pos1": self.count_pos1,
            "frac_neg1": round(self.frac_neg1, 6),
            "frac_zero": round(self.frac_zero, 6),
            "frac_pos1": round(self.frac_pos1, 6),
            "absmean_alpha": round(self.absmean_alpha, 8),
        }


def absmean_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Apply BitNet b1.58 absmean quantization to get ternary weights.

    Args:
        weight: BF16/FP32 weight tensor of shape (out_features, in_features)

    Returns:
        ternary: INT8 tensor with values in {-1, 0, +1}
        alpha: The absmean scaling factor
    """
    alpha = weight.abs().mean().item()
    if alpha == 0:
        logger.warning("Zero absmean encountered — all weights are zero")
        return torch.zeros_like(weight, dtype=torch.int8), 0.0

    scaled = weight / alpha
    rounded = torch.round(scaled)
    ternary = torch.clamp(rounded, -1, 1).to(torch.int8)
    return ternary, alpha


def validate_ternary(tensor: torch.Tensor, name: str) -> WeightStats:
    """Validate that a tensor contains only {-1, 0, +1} and compute stats."""
    unique_vals = torch.unique(tensor)
    valid_vals = {-1, 0, 1}
    actual_vals = set(unique_vals.tolist())

    if not actual_vals.issubset(valid_vals):
        invalid = actual_vals - valid_vals
        raise ValueError(
            f"Weight '{name}' contains invalid values: {invalid}. "
            f"Expected only {{-1, 0, +1}}."
        )

    n = tensor.numel()
    count_neg1 = (tensor == -1).sum().item()
    count_zero = (tensor == 0).sum().item()
    count_pos1 = (tensor == 1).sum().item()

    assert count_neg1 + count_zero + count_pos1 == n, "Count mismatch"

    return WeightStats(
        name=name,
        shape=tuple(tensor.shape),
        num_params=n,
        count_neg1=count_neg1,
        count_zero=count_zero,
        count_pos1=count_pos1,
        frac_neg1=count_neg1 / n,
        frac_zero=count_zero / n,
        frac_pos1=count_pos1 / n,
        absmean_alpha=0.0,  # filled in by caller
    )


def extract_weights(
    model_path: str | Path,
    output_dir: str | Path,
    layers: list[int] | None = None,
    save_stats: bool = True,
) -> dict:
    """Extract and quantize all BitLinear weights from the BF16 model.

    Args:
        model_path: Path to local model directory or HuggingFace repo ID
        output_dir: Where to save ternary weights (.npz) and stats (.json)
        layers: Specific layer indices to extract (None = all 30)
        save_stats: Whether to save per-weight statistics

    Returns:
        Dictionary with extraction summary
    """
    output_dir = ensure_dir(output_dir)
    weights_dir = ensure_dir(output_dir / "ternary_weights")

    model_path = Path(model_path) if "/" not in str(model_path) else str(model_path)

    config = BitNetConfig()
    if layers is None:
        layers = list(range(config.num_hidden_layers))

    logger.info(f"Extracting ternary weights from: {model_path}")
    logger.info(f"Layers: {layers[0]}..{layers[-1]} ({len(layers)} layers)")
    logger.info(f"Projections per layer: {len(BITLINEAR_PROJECTIONS)}")

    # Load safetensors — handles both local path and HF repo
    safetensor_files = _find_safetensor_files(model_path)
    logger.info(f"Found {len(safetensor_files)} safetensor file(s)")

    all_stats = []
    total_params = 0
    total_nonzero = 0

    for layer_idx in tqdm(layers, desc="Extracting layers"):
        layer_weights = {}
        layer_alphas = {}

        for proj_name, expected_out, expected_in in BITLINEAR_PROJECTIONS:
            weight_name = get_weight_name(layer_idx, proj_name)

            # Load the BF16 weight tensor
            bf16_weight = _load_tensor(safetensor_files, weight_name)

            if bf16_weight is None:
                raise KeyError(
                    f"Weight '{weight_name}' not found in safetensor files. "
                    f"Available keys can be listed with `safetensors info`."
                )

            # Validate shape
            if bf16_weight.shape != (expected_out, expected_in):
                raise ValueError(
                    f"Shape mismatch for '{weight_name}': "
                    f"expected ({expected_out}, {expected_in}), "
                    f"got {tuple(bf16_weight.shape)}"
                )

            # Quantize to ternary
            ternary, alpha = absmean_quantize(bf16_weight.float())

            # Validate
            stats = validate_ternary(ternary, weight_name)
            stats.absmean_alpha = alpha
            all_stats.append(stats)

            # Store
            key = proj_name.replace(".", "_")
            layer_weights[key] = ternary.numpy()
            layer_alphas[key] = alpha

            total_params += stats.num_params
            total_nonzero += stats.count_neg1 + stats.count_pos1

        # Save this layer's ternary weights as compressed numpy
        npz_path = weights_dir / f"layer_{layer_idx:03d}.npz"
        np.savez_compressed(npz_path, **layer_weights)

        # Save alphas (needed for dequantization / verification)
        alpha_path = weights_dir / f"layer_{layer_idx:03d}_alphas.json"
        save_json(layer_alphas, alpha_path)

    # Summary
    sparsity = 1.0 - (total_nonzero / total_params) if total_params > 0 else 0.0
    summary = {
        "model": str(model_path),
        "num_layers_extracted": len(layers),
        "total_ternary_params": total_params,
        "total_nonzero_params": total_nonzero,
        "sparsity": round(sparsity, 6),
        "layers": [layers[0], layers[-1]],
    }

    logger.info(f"Total ternary parameters: {format_count(total_params)}")
    logger.info(f"Sparsity (fraction of zeros): {sparsity:.2%}")
    logger.info(
        f"Distribution: neg1={all_stats[0].frac_neg1:.2%}, "
        f"zero={all_stats[0].frac_zero:.2%}, "
        f"pos1={all_stats[0].frac_pos1:.2%} (layer 0, q_proj)"
    )

    # Save stats and summary
    if save_stats:
        stats_path = output_dir / "weight_stats.json"
        save_json(
            {
                "summary": summary,
                "per_weight": [s.to_dict() for s in all_stats],
            },
            stats_path,
        )
        logger.info(f"Statistics saved to {stats_path}")

    save_json(summary, output_dir / "extraction_summary.json")
    logger.info(f"Ternary weights saved to {weights_dir}")

    return summary


def _find_safetensor_files(model_path: str | Path) -> list[Path]:
    """Find all .safetensors files in a model directory.

    If model_path is a HF repo ID string (contains '/'), download first.
    """
    model_path_str = str(model_path)

    # Check if it's a HuggingFace repo ID
    if "/" in model_path_str and not Path(model_path_str).exists():
        logger.info(f"Downloading model from HuggingFace: {model_path_str}")
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                model_path_str,
                allow_patterns=["*.safetensors", "config.json"],
            )
            model_path = Path(local_dir)
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install with: pip install huggingface_hub"
            )
    else:
        model_path = Path(model_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path}. "
            f"Contents: {list(model_path.iterdir())}"
        )
    return files


def _load_tensor(
    safetensor_files: list[Path], tensor_name: str
) -> torch.Tensor | None:
    """Load a named tensor from a list of safetensor files.

    Searches through all shard files to find the tensor.
    """
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    return None
