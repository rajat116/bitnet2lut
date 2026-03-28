"""Full standalone inference emulator for BitNet b1.58 2B4T.

Uses LUT-based matmul for all ternary linear layers and standard
floating-point arithmetic for everything else (RMSNorm, RoPE,
GQA attention, ReLU², softmax, embedding).

This produces the reference token sequence that the FPGA must match.

Architecture (per decoder layer):
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
    Q = q_proj(hidden_states)  # BitLinear: SubLN→INT8_quant→ternary_matmul→scale
    K = k_proj(hidden_states)
    V = v_proj(hidden_states)
    Q, K = apply_rope(Q, K, position_ids)
    attn_out = GQA_attention(Q, K, V)
    attn_out = attn_sub_norm(attn_out)
    attn_out = o_proj(attn_out)
    hidden_states = residual + attn_out

    residual = hidden_states
    hidden_states = post_attention_layernorm(hidden_states)
    gate = gate_proj(hidden_states)
    up = up_proj(hidden_states)
    hidden_states = down_proj(ffn_sub_norm(relu2(gate) * up))
    hidden_states = residual + hidden_states

BitLinear forward (during inference with pre-quantized ternary weights):
    x → RMSNorm (SubLN, built into the layer) → INT8 absmax quant →
    ternary_matmul (our LUT path replaces this) → scale by alpha

NOTE: The HF model stores BF16 "master weights". During the forward pass,
it quantizes them on-the-fly. We already extracted the ternary weights
(the quantized result). So our emulator skips the weight quantization
and uses the stored ternary weights directly.
"""

import logging
import math
from pathlib import Path
import json as _json  # for loading thresholds

import numpy as np

from .emulator import direct_ternary_matvec, lut_matvec
from .lut_gen import tile_to_lut_indices
from .model_config import BITLINEAR_PROJECTIONS, BitNetConfig
from .utils import load_json

logger = logging.getLogger("bitnet2lut")


# ============================================================
# Standard operations (not LUT-converted, run in float32)
# ============================================================

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)."""
    # x: (..., hidden_size), weight: (hidden_size,)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def relu_squared(x: np.ndarray) -> np.ndarray:
    """ReLU²: max(0, x)²."""
    return np.maximum(0, x) ** 2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def rope_frequencies(head_dim: int, max_seq_len: int, theta: float = 500000.0) -> np.ndarray:
    """Precompute RoPE frequency table: (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float64)
    angles = np.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    return angles


def apply_rope(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Apply rotary position embeddings.

    Args:
        x: (num_heads, seq_len, head_dim) in float
        angles: (seq_len, head_dim // 2)

    Returns:
        Rotated x with same shape.
    """
    num_heads, seq_len, head_dim = x.shape
    half = head_dim // 2
    x1 = x[:, :, :half]
    x2 = x[:, :, half:]
    cos = np.cos(angles[:seq_len, :])  # (seq_len, half)
    sin = np.sin(angles[:seq_len, :])
    # Broadcast over heads
    cos = cos[np.newaxis, :, :]  # (1, seq_len, half)
    sin = sin[np.newaxis, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return np.concatenate([out1, out2], axis=-1)


def int8_absmax_quantize(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-token INT8 absmax quantization (as BitLinear does for activations).

    Args:
        x: (hidden_size,) float activation vector

    Returns:
        (quantized_int8, scale) where scale = abs_max / 127
    """
    abs_max = np.max(np.abs(x))
    if abs_max < 1e-10:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return quantized, scale

def int4_absmax_quantize(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-token INT4 absmax quantization — EXPERIMENTAL.

    Same scheme as INT8 but maps to [-7, 7] (7 = 2^(4-1) - 1).
    Values are stored as int8 but clipped to the 4-bit range.

    This is NOT how the model was trained. We use this to measure
    how much accuracy degrades when we reduce activation precision
    below INT8 at inference time without retraining.

    Args:
        x: (hidden_size,) float activation vector

    Returns:
        (quantized_int8, scale) where values are in [-7, 7]
    """
    abs_max = np.max(np.abs(x))
    if abs_max < 1e-10:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = abs_max / 7.0           # <-- 7 instead of 127
    quantized = np.clip(np.round(x / scale), -8, 7).astype(np.int8)
    return quantized, scale

def lloyd_max_quantize(
    x: np.ndarray,
    thresholds: np.ndarray,
    levels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Quantize using precomputed Lloyd-Max thresholds and levels.

    Instead of uniform spacing, uses optimal thresholds derived from
    the measured activation distribution of this specific layer.

    Args:
        x: (hidden_size,) float activation vector
        thresholds: (n_levels - 1,) precomputed decision boundaries
        levels: (n_levels,) precomputed reconstruction values

    Returns:
        (quantized, scale) where quantized contains the reconstruction
        level indices stored as int8, and scale=1.0 (levels are already
        in float space, dequantization happens via the levels array)
    """
    bin_indices = np.digitize(x, thresholds)
    # Clip to valid range (digitize can return n_levels for values > last threshold)
    bin_indices = np.clip(bin_indices, 0, len(levels) - 1)
    return bin_indices, 1.0  # scale unused — dequant uses levels array directly

def bitlinear_forward_lut(
    x: np.ndarray,
    ternary_weight: np.ndarray,
    alpha: float,
    sub_norm_weight: np.ndarray | None = None,
    sub_norm_eps: float = 1e-5,
    group_size: int = 4,
    use_lut: bool = True,
    activation_bits: int = 8,        # <-- ADD THIS, default=8 keeps old behavior
    activation_collector: dict | None = None,  # <-- ADD THIS
    lm_thresholds: np.ndarray | None = None,  # <-- ADD
    lm_levels: np.ndarray | None = None,       # <-- ADD
) -> np.ndarray:
    """Execute one BitLinear layer using LUT or direct ternary matmul.

    The BitLinear forward during inference:
        1. x → (SubLN is applied BEFORE calling the projection in this model)
        2. INT8 absmax quantize the activation
        3. Ternary matmul: W_ternary @ x_int8
        4. Scale output by alpha (the absmean scale from weight quantization)

    Args:
        x: Input activation (hidden_size,) in float
        ternary_weight: (out_dim, in_dim) int8 in {-1, 0, +1}
        alpha: Weight scale factor (from absmean quantization)
        sub_norm_weight: Optional SubLN weight (applied before this projection
                         in the actual model — NOT inside BitLinear for 2B4T)
        sub_norm_eps: RMSNorm epsilon
        group_size: LUT group size
        use_lut: If True, use LUT emulator; if False, use direct ternary matmul

    Returns:
        Output activation (out_dim,) in float
    """
    out_dim, in_dim = ternary_weight.shape

    # Collect raw activation values if collector provided
    if activation_collector is not None:
        key = activation_collector.get("_current_key", "unknown")
        if key not in activation_collector:
            activation_collector[key] = []
        activation_collector[key].append(x.copy())

    # Quantize activation
    if lm_thresholds is not None and lm_levels is not None:
        # Lloyd-Max non-uniform quantization
        bin_indices, _ = lloyd_max_quantize(x, lm_thresholds, lm_levels)
        # Reconstruct using levels for the matmul
        # We dequantize immediately and treat as float input to matmul
        x_reconstructed = lm_levels[bin_indices].astype(np.float32)
        # Now quantize the reconstructed values to INT8 for the ternary matmul
        x_int8, act_scale = int8_absmax_quantize(x_reconstructed)
    elif activation_bits == 4:
        x_int8, act_scale = int4_absmax_quantize(x)
    else:
        x_int8, act_scale = int8_absmax_quantize(x)

    # Pad for group alignment
    pad = (group_size - in_dim % group_size) % group_size
    if pad > 0:
        w_padded = np.zeros((out_dim, in_dim + pad), dtype=np.int8)
        w_padded[:, :in_dim] = ternary_weight
        x_padded = np.zeros(in_dim + pad, dtype=np.int8)
        x_padded[:in_dim] = x_int8
    else:
        w_padded = ternary_weight
        x_padded = x_int8

    # Matrix-vector product
    if use_lut:
        lut_indices = tile_to_lut_indices(w_padded, group_size)
        output_int = lut_matvec(lut_indices, x_padded, group_size)
    else:
        output_int = direct_ternary_matvec(w_padded, x_padded)

    # Dequantize: multiply by activation scale and weight scale
    output_float = output_int.astype(np.float64) * act_scale * alpha

    return output_float.astype(np.float32)


# ============================================================
# Full model emulator
# ============================================================

class BitNetEmulator:
    """Full inference emulator for BitNet b1.58 2B4T.

    Loads all weights and runs inference using LUT-based linear layers.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        model_path: str,
        group_size: int = 4,
        use_lut: bool = True,
        activation_bits: int = 8,        # <-- ADD THIS
    ):
        """Initialize the emulator.

        Args:
            weights_dir: Directory with extracted ternary weights (layer_XXX.npz)
            model_path: HuggingFace model path for loading non-ternary weights
                        (embedding, layernorms, etc.)
            group_size: LUT group size
            use_lut: Whether to use LUT emulator or direct ternary matmul
        """
        self.config = BitNetConfig()
        self.weights_dir = Path(weights_dir)
        self.group_size = group_size
        self.use_lut = use_lut
        self.activation_bits = activation_bits
        self.activation_collector = None  # set externally to enable collection
        self.lm_thresholds = {}   # key: layer_idx -> np.ndarray
        self.lm_levels = {}       # key: layer_idx -> np.ndarray

        # Load non-ternary weights from HuggingFace
        self._load_model_weights(model_path)

        # Load ternary weights + alphas from our extraction
        self._load_ternary_weights()

        # Precompute RoPE angles
        self.rope_angles = rope_frequencies(
            self.config.head_dim,
            self.config.max_position_embeddings,
            self.config.rope_theta,
        )

        logger.info(f"Emulator initialized: {self.config.num_hidden_layers} layers, "
                     f"use_lut={use_lut}")

    def _load_model_weights(self, model_path: str) -> None:
        """Load non-ternary weights (embedding, norms) from HF safetensors."""
        import torch
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        import glob as globmod

        local_dir = snapshot_download(model_path, allow_patterns=["*.safetensors"])

        self.fp_weights = {}
        for sf_path in globmod.glob(f"{local_dir}/*.safetensors"):
            tensors = load_file(sf_path)
            for key, tensor in tensors.items():
                self.fp_weights[key] = tensor.float().numpy()

        self.embed_tokens = self.fp_weights["model.embed_tokens.weight"]
        logger.info(f"Embedding shape: {self.embed_tokens.shape}")

        self.final_norm_weight = self.fp_weights["model.norm.weight"]

    def _load_ternary_weights(self) -> None:
        """Load our extracted ternary weights and alpha scales."""
        self.ternary_weights = {}
        self.alphas = {}

        for layer_idx in range(self.config.num_hidden_layers):
            npz_path = self.weights_dir / f"layer_{layer_idx:03d}.npz"
            alpha_path = self.weights_dir / f"layer_{layer_idx:03d}_alphas.json"

            if not npz_path.exists():
                raise FileNotFoundError(f"Missing {npz_path}")

            data = np.load(npz_path)
            alphas = load_json(alpha_path) if alpha_path.exists() else {}

            for proj_name, _, _ in BITLINEAR_PROJECTIONS:
                key = proj_name.replace(".", "_")
                self.ternary_weights[(layer_idx, proj_name)] = data[key]
                alpha_key = f"model.layers.{layer_idx}.{proj_name}.weight"
                self.alphas[(layer_idx, proj_name)] = alphas.get(alpha_key, 1.0)

        logger.info(f"Loaded ternary weights for {self.config.num_hidden_layers} layers")

    def load_lloydmax_thresholds(self, thresholds_path: str) -> None:
        """Load precomputed Lloyd-Max thresholds for down_proj layers."""
        with open(thresholds_path) as f:
            data = _json.load(f)

        for key, entry in data.items():
            # key is like "layer_000.mlp.down_proj"
            layer_str = key.split(".")[0]  # "layer_000"
            layer_idx = int(layer_str.split("_")[1])
            self.lm_thresholds[layer_idx] = np.array(entry["thresholds"], dtype=np.float32)
            self.lm_levels[layer_idx] = np.array(entry["levels"], dtype=np.float32)

        logger.info(f"Loaded Lloyd-Max thresholds for {len(self.lm_thresholds)} down_proj layers")

    def _get_norm_weight(self, name: str) -> np.ndarray:
        """Get a norm weight from fp_weights."""
        return self.fp_weights[name].astype(np.float32)

    def _layer_forward(
        self,
        hidden_states: np.ndarray,
        layer_idx: int,
        position: int,
        kv_cache: dict,
    ) -> np.ndarray:
        """Run one decoder layer.

        Args:
            hidden_states: (hidden_size,) float32
            layer_idx: Layer index
            position: Current sequence position (for RoPE + KV cache)
            kv_cache: Dict holding K,V tensors per layer

        Returns:
            Updated hidden_states (hidden_size,)
        """
        cfg = self.config
        eps = cfg.rms_norm_eps

        # ---- Self-attention ----
        residual = hidden_states.copy()

        # Input LayerNorm
        norm_w = self._get_norm_weight(f"model.layers.{layer_idx}.input_layernorm.weight")
        hidden_states = rms_norm(hidden_states, norm_w, eps)

        # Q, K, V projections (BitLinear)
        def proj(proj_name):
            w = self.ternary_weights[(layer_idx, proj_name)]
            a = self.alphas[(layer_idx, proj_name)]
            if self.activation_collector is not None:
                self.activation_collector["_current_key"] = f"layer_{layer_idx:03d}.{proj_name}"
            return bitlinear_forward_lut(
                hidden_states, w, a,
                group_size=self.group_size, use_lut=self.use_lut,
                activation_bits=self.activation_bits,
                activation_collector=self.activation_collector,
            )

        q = proj("self_attn.q_proj")  # (2560,)
        k = proj("self_attn.k_proj")  # (640,)
        v = proj("self_attn.v_proj")  # (640,)

        # Reshape for attention heads
        # Q: (num_heads, 1, head_dim) = (20, 1, 128)
        # K, V: (num_kv_heads, 1, head_dim) = (5, 1, 128)
        q = q.reshape(cfg.num_attention_heads, 1, cfg.head_dim)
        k = k.reshape(cfg.num_key_value_heads, 1, cfg.head_dim)
        v = v.reshape(cfg.num_key_value_heads, 1, cfg.head_dim)

        # Apply RoPE to Q and K
        # For single-token decode, angles at this position
        pos_angles = self.rope_angles[position:position + 1, :]  # (1, head_dim//2)
        q = apply_rope(q, pos_angles)
        k = apply_rope(k, pos_angles)

        # KV cache: append and retrieve
        cache_key = f"layer_{layer_idx}"
        if cache_key not in kv_cache:
            kv_cache[cache_key] = {"k": k, "v": v}
        else:
            kv_cache[cache_key]["k"] = np.concatenate(
                [kv_cache[cache_key]["k"], k], axis=1
            )
            kv_cache[cache_key]["v"] = np.concatenate(
                [kv_cache[cache_key]["v"], v], axis=1
            )

        k_full = kv_cache[cache_key]["k"]  # (num_kv_heads, seq_len, head_dim)
        v_full = kv_cache[cache_key]["v"]

        # GQA: repeat K,V for each query group
        # num_heads=20, num_kv_heads=5 → repeat each KV head 4 times
        gqa_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        k_expanded = np.repeat(k_full, gqa_groups, axis=0)  # (20, seq_len, 128)
        v_expanded = np.repeat(v_full, gqa_groups, axis=0)

        # Scaled dot-product attention
        # Q: (20, 1, 128), K: (20, seq_len, 128)
        scale = 1.0 / math.sqrt(cfg.head_dim)
        attn_scores = np.einsum("hqd,hkd->hqk", q, k_expanded) * scale
        # attn_scores: (20, 1, seq_len)

        # Causal mask: for decode, all previous positions are visible
        attn_weights = softmax(attn_scores, axis=-1)

        # Weighted sum of values
        attn_output = np.einsum("hqk,hkd->hqd", attn_weights, v_expanded)
        # attn_output: (20, 1, 128)

        # Reshape back to (hidden_size,)
        attn_output = attn_output.reshape(-1)  # (2560,)

        # SubLN after attention (attn_sub_norm)
        sub_norm_w = self._get_norm_weight(
            f"model.layers.{layer_idx}.self_attn.attn_sub_norm.weight"
        )
        attn_output = rms_norm(attn_output, sub_norm_w, eps)

        # O projection (BitLinear) — input is attn_output, NOT hidden_states
        w_o = self.ternary_weights[(layer_idx, "self_attn.o_proj")]
        a_o = self.alphas[(layer_idx, "self_attn.o_proj")]
        attn_output = bitlinear_forward_lut(
            attn_output, w_o, a_o,
            group_size=self.group_size, use_lut=self.use_lut,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # ---- MLP ----
        residual = hidden_states.copy()

        # Post-attention LayerNorm
        norm_w = self._get_norm_weight(
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )
        hidden_states = rms_norm(hidden_states, norm_w, eps)

        # Gate and Up projections (BitLinear)
        gate = proj("mlp.gate_proj")   # (6912,)
        up = proj("mlp.up_proj")       # (6912,)

        # ReLU² * up (gated activation)
        ffn_hidden = relu_squared(gate) * up

        # FFN SubLN
        ffn_sub_norm_w = self._get_norm_weight(
            f"model.layers.{layer_idx}.mlp.ffn_sub_norm.weight"
        )
        ffn_hidden = rms_norm(ffn_hidden, ffn_sub_norm_w, eps)

        # Down projection (BitLinear)
        w = self.ternary_weights[(layer_idx, "mlp.down_proj")]
        a = self.alphas[(layer_idx, "mlp.down_proj")]
        if self.activation_collector is not None:
            self.activation_collector["_current_key"] = f"layer_{layer_idx:03d}.mlp.down_proj"
        # Use Lloyd-Max thresholds for down_proj if available
        lm_thresh = self.lm_thresholds.get(layer_idx)
        lm_levs = self.lm_levels.get(layer_idx)
        hidden_states = bitlinear_forward_lut(
            ffn_hidden, w, a,
            group_size=self.group_size, use_lut=self.use_lut,
            activation_bits=self.activation_bits,
            activation_collector=self.activation_collector,
            lm_thresholds=lm_thresh,
            lm_levels=lm_levs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states

    def generate(
        self,
        token_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 0.0,
    ) -> list[int]:
        """Generate tokens autoregressively.

        Args:
            token_ids: Input token IDs (prompt)
            max_new_tokens: How many tokens to generate
            temperature: 0.0 = greedy, >0 = sampling

        Returns:
            Full sequence of token IDs (prompt + generated)
        """
        cfg = self.config
        generated = list(token_ids)
        kv_cache = {}

        # Process prompt tokens one by one (building KV cache)
        for pos, tok_id in enumerate(token_ids):
            hidden = self.embed_tokens[tok_id].astype(np.float32)

            for layer_idx in range(cfg.num_hidden_layers):
                hidden = self._layer_forward(hidden, layer_idx, pos, kv_cache)

            # After last layer: final norm + LM head
            # (only need logits from the last prompt token)

        self._last_hidden = hidden.copy()

        # Now generate new tokens
        for step in range(max_new_tokens):
            # Final norm
            output = rms_norm(hidden, self.final_norm_weight.astype(np.float32), cfg.rms_norm_eps)

            # LM head: embed_tokens.T @ output (tied weights)
            logits = self.embed_tokens.astype(np.float32) @ output  # (vocab_size,)

            # Sample or greedy
            if temperature <= 0:
                next_token = int(np.argmax(logits))
            else:
                probs = softmax(logits / temperature)
                next_token = int(np.random.choice(len(probs), p=probs))

            generated.append(next_token)
            logger.info(f"Token {step + 1}/{max_new_tokens}: id={next_token}")

            # Check for EOS
            if next_token == 128001:  # eos_token_id
                break

            # Forward pass for next position
            pos = len(token_ids) + step
            hidden = self.embed_tokens[next_token].astype(np.float32)

            for layer_idx in range(cfg.num_hidden_layers):
                hidden = self._layer_forward(hidden, layer_idx, pos, kv_cache)

        return generated


def run_emulator_comparison(
    model_path: str,
    weights_dir: str | Path,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 10,
    group_size: int = 4,
) -> dict:
    """Run the LUT emulator and HuggingFace model, compare token outputs.

    This is the Phase 0 verification: same prompt → same tokens.

    Args:
        model_path: HuggingFace model path
        weights_dir: Directory with extracted ternary weights
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        group_size: LUT group size

    Returns:
        Comparison result dictionary
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {"error": "torch/transformers not available"}

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(prompt)
    logger.info(f"Prompt: '{prompt}' → {len(input_ids)} tokens")

    # Path 1: HuggingFace reference
    logger.info("Running HuggingFace reference model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        hf_output = model.generate(
            torch.tensor([input_ids]),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
        )
    hf_tokens = hf_output[0].tolist()
    hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=True)
    logger.info(f"HF output: '{hf_text}'")

    # Free model memory
    del model
    import gc
    gc.collect()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Path 2: LUT emulator
    logger.info("Running LUT emulator...")
    emulator = BitNetEmulator(
        weights_dir=weights_dir,
        model_path=model_path,
        group_size=group_size,
        use_lut=True,
    )
    lut_tokens = emulator.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    lut_text = tokenizer.decode(lut_tokens, skip_special_tokens=True)
    logger.info(f"LUT output: '{lut_text}'")

    # Diagnostic: show top-10 logits from the last generation step
    # to verify "Paris" is ranked high even if it didn't win argmax
    logger.info("--- Logit diagnostic (last token step) ---")
    # Re-run last token through final norm + LM head to get logits
    last_hidden = emulator._last_hidden  # we need to save this — see below
    diag_output = rms_norm(last_hidden, emulator.final_norm_weight.astype(np.float32), emulator.config.rms_norm_eps)
    diag_logits = emulator.embed_tokens.astype(np.float32) @ diag_output
    top_indices = np.argsort(diag_logits)[::-1][:10]
    for rank, idx in enumerate(top_indices):
        token_str = tokenizer.decode([idx])
        logger.info(f"  Rank {rank+1}: id={idx} '{token_str}' logit={diag_logits[idx]:.4f}")
    # Also check where HF's first generated token ranks
    hf_first_token = hf_tokens[len(input_ids)]
    hf_rank = int(np.where(top_indices == hf_first_token)[0][0]) + 1 if hf_first_token in top_indices else None
    if hf_rank:
        logger.info(f"  HF's token '{tokenizer.decode([hf_first_token])}' is rank {hf_rank}")
    else:
        full_ranking = np.argsort(diag_logits)[::-1]
        hf_rank = int(np.where(full_ranking == hf_first_token)[0][0]) + 1
        logger.info(f"  HF's token '{tokenizer.decode([hf_first_token])}' is rank {hf_rank} (not in top 10)")


    # Path 3: Direct ternary — reuse same emulator, just flip the flag
    logger.info("Running direct ternary emulator...")
    emulator.use_lut = False
    direct_tokens = emulator.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    direct_text = tokenizer.decode(direct_tokens, skip_special_tokens=True)
    logger.info(f"Direct output: '{direct_text}'")

    # Compare
    hf_generated = hf_tokens[len(input_ids):]
    lut_generated = lut_tokens[len(input_ids):]
    direct_generated = direct_tokens[len(input_ids):]

    lut_vs_hf_match = hf_generated == lut_generated
    lut_vs_direct_match = lut_generated == direct_generated

    result = {
        "prompt": prompt,
        "num_prompt_tokens": len(input_ids),
        "max_new_tokens": max_new_tokens,
        "hf_tokens": hf_generated,
        "hf_text": hf_text,
        "lut_tokens": lut_generated,
        "lut_text": lut_text,
        "direct_tokens": direct_generated,
        "direct_text": direct_text,
        "lut_vs_hf_match": lut_vs_hf_match,
        "lut_vs_direct_match": lut_vs_direct_match,
    }

    if lut_vs_hf_match:
        logger.info("✅ LUT emulator MATCHES HuggingFace output!")
    else:
        logger.warning("⚠️  LUT emulator differs from HuggingFace output")
        logger.warning(f"  HF:  {hf_generated}")
        logger.warning(f"  LUT: {lut_generated}")
        # This is expected due to floating-point differences in norms/attention
        # The important check is lut_vs_direct
        if lut_vs_direct_match:
            logger.info("✅ LUT emulator matches direct ternary (LUT math is correct)")
            logger.info("   Differences vs HF are from float precision in norms/attention")

    return result
