# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Qwen3 text decoder with KV cache, GQA, QK-norm, and RoPE."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TextDecoderConfig


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Per-layer key/value cache for autoregressive decoding.

    Keys and values are stored per layer.  ``offset`` tracks the total
    number of tokens already processed (prefill + previously decoded).
    """

    def __init__(self) -> None:
        self.keys: list[mx.array] = []
        self.values: list[mx.array] = []
        self.offset: int = 0

    def update(
        self,
        layer_idx: int,
        key: mx.array,
        value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Append *key* / *value* to the cache and return the full sequence.

        Parameters
        ----------
        layer_idx:
            Index of the decoder layer.
        key:
            Shape ``(B, n_kv_heads, T_new, head_dim)``.
        value:
            Shape ``(B, n_kv_heads, T_new, head_dim)``.

        Returns
        -------
        tuple[mx.array, mx.array]
            Full accumulated ``(key, value)`` up to the current step.
        """
        if layer_idx >= len(self.keys):
            self.keys.append(key)
            self.values.append(value)
        else:
            self.keys[layer_idx] = mx.concatenate(
                [self.keys[layer_idx], key], axis=2
            )
            self.values[layer_idx] = mx.concatenate(
                [self.values[layer_idx], value], axis=2
            )
        return self.keys[layer_idx], self.values[layer_idx]


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _rotate_half(x: mx.array) -> mx.array:
    """Rotate the last dimension by 90° (NeoX / rotate-half style)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _build_rope(
    position_ids: mx.array,
    head_dim: int,
    theta: float = 1_000_000.0,
) -> tuple[mx.array, mx.array]:
    """Compute RoPE cos/sin tensors for the given position indices.

    Parameters
    ----------
    position_ids:
        Shape ``(B, T)`` — integer position indices.
    head_dim:
        Size of each attention head (128).
    theta:
        RoPE base frequency (1e6 for Qwen3).

    Returns
    -------
    cos, sin:
        Each shape ``(B, 1, T, head_dim)`` — broadcastable over heads.
    """
    half = head_dim // 2
    # inv_freq: (half,)
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, half, dtype=mx.float32) / head_dim)
    )
    # freqs: (B, T, half)
    freqs = position_ids[:, :, None].astype(mx.float32) * inv_freq[None, None, :]
    # emb: (B, T, head_dim)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb)[:, None, :, :]  # (B, 1, T, head_dim)
    sin = mx.sin(emb)[:, None, :, :]  # (B, 1, T, head_dim)
    return cos, sin


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply RoPE to query and key tensors.

    Parameters
    ----------
    q, k:
        Shape ``(B, n_heads, T, head_dim)``.
    cos, sin:
        Shape ``(B, 1, T, head_dim)`` or broadcastable equivalent.
    """
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Grouped-query attention with QK-norm and RoPE."""

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.n_heads    = config.num_attention_heads   # 16
        self.n_kv_heads = config.num_key_value_heads   # 8
        self.head_dim   = config.head_dim              # 128
        self.scale      = self.head_dim ** -0.5
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim,    bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size,    bias=False)

        # Per-head RMSNorm on the head dimension (eps=1e-6, no bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,
        cache: Optional[KVCache],
        layer_idx: int,
    ) -> mx.array:
        B, T, _ = x.shape

        # 1. Project
        q = self.q_proj(x)  # (B, T, n_heads*head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads*head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads*head_dim)

        # 2. Reshape to (B, T, n_heads, head_dim)
        q = q.reshape(B, T, self.n_heads,    self.head_dim)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim)

        # 3. QK-norm (RMSNorm on last dim — per head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Transpose to (B, n_heads, T, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # 5. RoPE
        cos, sin = _build_rope(position_ids, self.head_dim, self.rope_theta)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # 6. KV cache
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        # 7. GQA: repeat k/v to match n_heads
        #    k/v: (B, n_kv_heads, T_total, head_dim) → (B, n_heads, T_total, head_dim)
        repeat = self.n_heads // self.n_kv_heads  # 2
        T_total = k.shape[2]
        k = mx.repeat(k, repeat, axis=1)
        v = mx.repeat(v, repeat, axis=1)

        # 8. Build causal mask for non-cache (prefill) case
        #    For decode (T==1) no mask needed.
        mask: Optional[mx.array] = None
        if T > 1:
            # Standard additive causal mask: (1, 1, T, T_total)
            rows = mx.arange(T)[:, None]
            cols = mx.arange(T_total)[None, :]
            # During prefill T == T_total; the condition allows attending to all
            # past positions (including cache, which is empty during prefill).
            causal = mx.where(rows >= cols, mx.zeros((T, T_total)), mx.full((T, T_total), float("-inf")))
            mask = causal[None, None]  # (1, 1, T, T_total)

        # 9. Scaled dot-product attention
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        # (B, n_heads, T, head_dim) → (B, T, n_heads*head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)

        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single pre-norm Qwen3 decoder layer."""

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.input_layernorm        = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn              = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp                    = MLP(config)

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,
        cache: Optional[KVCache],
        layer_idx: int,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids, cache, layer_idx)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


# ---------------------------------------------------------------------------
# Text decoder
# ---------------------------------------------------------------------------

class TextDecoder(nn.Module):
    """Qwen3 text decoder with tied lm_head.

    Parameters
    ----------
    config:
        ``TextDecoderConfig`` instance.
    """

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm   = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # lm_head weight is tied to embed_tokens — no separate parameter.

    def __call__(
        self,
        inputs: mx.array,
        position_ids: mx.array,
        cache: Optional[KVCache] = None,
        is_embeds: bool = False,
    ) -> mx.array:
        """Forward pass through the decoder.

        Parameters
        ----------
        inputs:
            Either token IDs ``(B, T)`` or pre-computed embeddings
            ``(B, T, hidden_size)`` when *is_embeds* is ``True``.
        position_ids:
            Shape ``(B, T)`` — integer position indices for RoPE.
        cache:
            ``KVCache`` instance for incremental decoding.  Pass ``None``
            for a stateless full-context forward pass.
        is_embeds:
            When ``True``, *inputs* is treated as embeddings rather than IDs.

        Returns
        -------
        mx.array
            Logits of shape ``(B, T, vocab_size)``.
        """
        if is_embeds:
            h = inputs
        else:
            h = self.embed_tokens(inputs)

        for i, layer in enumerate(self.layers):
            h = layer(h, position_ids, cache, layer_idx=i)

        h = self.norm(h)
        # Tied lm_head: project back to vocab via embed_tokens weights
        logits = h @ self.embed_tokens.weight.T
        return logits


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_decoder_weights(decoder: TextDecoder, model_path: str | Path) -> None:
    """Load decoder weights from safetensors.

    The safetensors file uses the ``model.`` prefix for all decoder keys.
    This function strips that prefix and loads them into *decoder*.

    Parameters
    ----------
    decoder:
        A ``TextDecoder`` instance to populate.
    model_path:
        Either a local directory containing ``model.safetensors`` or a
        HuggingFace Hub repo ID (resolved via ``snapshot_download``).
    """
    path = Path(model_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=str(model_path)))

    weights_file = path / "model.safetensors"
    raw: dict[str, mx.array] = mx.load(str(weights_file))

    prefix = "model."
    mapped: list[tuple[str, mx.array]] = []
    for key, value in raw.items():
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix):]
        mapped.append((local_key, value))

    decoder.load_weights(mapped)
    mx.eval(decoder.parameters())
