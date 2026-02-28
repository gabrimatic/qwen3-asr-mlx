# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Qwen3 text decoder with KV cache, GQA, QK-norm, and RoPE."""

from __future__ import annotations

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
# Causal mask
# ---------------------------------------------------------------------------

def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    """Create an additive causal attention mask.

    Returns shape ``(N, offset + N)`` with ``-1e9`` at positions that
    should not be attended to (future positions).
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


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
    """Grouped-query attention with QK-norm and RoPE.

    Uses MLX's native ``nn.RoPE`` (delegates to ``mx.fast.rope``) for
    correct and efficient rotary position embeddings.  GQA broadcasting
    is handled natively by ``mx.fast.scaled_dot_product_attention``.
    """

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.n_heads    = config.num_attention_heads   # 16
        self.n_kv_heads = config.num_key_value_heads   # 8
        self.head_dim   = config.head_dim              # 128
        self.scale      = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim,    bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size,    bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache],
        layer_idx: int,
    ) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.n_heads,    self.head_dim)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        mask = create_additive_causal_mask(T, offset=offset).astype(q.dtype)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
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
        cache: Optional[KVCache],
        layer_idx: int,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cache, layer_idx)
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

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[KVCache] = None,
        is_embeds: bool = False,
    ) -> mx.array:
        """Forward pass through the decoder.

        Parameters
        ----------
        inputs:
            Either token IDs ``(B, T)`` or pre-computed embeddings
            ``(B, T, hidden_size)`` when *is_embeds* is ``True``.
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
            h = layer(h, cache, layer_idx=i)

        h = self.norm(h)
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
