"""BitNet b1.58 2B4T architecture constants and layer naming conventions.

Source: microsoft/bitnet-b1.58-2B-4T config.json + technical report (arXiv:2504.12285)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BitNetConfig:
    """Architecture parameters for BitNet b1.58 2B4T."""

    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_gqa_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


# The 7 linear projections per transformer layer that use ternary weights.
# These are the BitLinear layers — the ones we convert to LUTs.
# Format: (name_suffix, output_dim, input_dim)
BITLINEAR_PROJECTIONS = [
    ("self_attn.q_proj", 2560, 2560),   # (hidden, hidden)
    ("self_attn.k_proj", 640, 2560),    # (kv_heads * head_dim, hidden) = 5*128
    ("self_attn.v_proj", 640, 2560),    # (kv_heads * head_dim, hidden)
    ("self_attn.o_proj", 2560, 2560),   # (hidden, hidden)
    ("mlp.gate_proj", 6912, 2560),      # (intermediate, hidden)
    ("mlp.up_proj", 6912, 2560),        # (intermediate, hidden)
    ("mlp.down_proj", 2560, 6912),      # (hidden, intermediate)
]

# Non-ternary components (full-precision, not converted to LUTs):
# - model.embed_tokens.weight: (128256, 2560) — embedding lookup, not a matmul
# - model.layers.*.input_layernorm.weight: (2560,) — RMSNorm scale
# - model.layers.*.post_attention_layernorm.weight: (2560,) — RMSNorm scale
# - model.layers.*.self_attn.attn_sub_norm.weight: (2560,) — SubLN scale
# - model.layers.*.mlp.ffn_sub_norm.weight: (6912,) — SubLN scale
# - model.norm.weight: (2560,) — final RMSNorm
# - lm_head.weight: tied to embed_tokens (or separate if not tied)

# Weight tensor naming pattern in HuggingFace safetensors
WEIGHT_PATTERN = "model.layers.{layer_idx}.{proj_name}.weight"


def get_weight_name(layer_idx: int, proj_name: str) -> str:
    """Get the full weight tensor name for a given layer and projection."""
    return WEIGHT_PATTERN.format(layer_idx=layer_idx, proj_name=proj_name)


def get_all_weight_names(num_layers: int | None = None) -> list[str]:
    """Get all ternary weight tensor names across all layers."""
    if num_layers is None:
        num_layers = BitNetConfig.num_hidden_layers
    names = []
    for layer_idx in range(num_layers):
        for proj_name, _, _ in BITLINEAR_PROJECTIONS:
            names.append(get_weight_name(layer_idx, proj_name))
    return names


def total_ternary_params() -> int:
    """Total number of ternary parameters across all layers."""
    per_layer = sum(out_dim * in_dim for _, out_dim, in_dim in BITLINEAR_PROJECTIONS)
    return per_layer * BitNetConfig.num_hidden_layers


# Quick sanity check
# Per layer: 2560*2560 + 640*2560 + 640*2560 + 2560*2560 + 6912*2560 + 6912*2560 + 2560*6912
# = 6553600 + 1638400 + 1638400 + 6553600 + 17694720 + 17694720 + 17694720
# = 69,468,160 per layer
# × 30 layers = 2,084,044,800 ternary params (~2.08B, matches "2B" model name)
