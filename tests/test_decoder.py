# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Tests for the Qwen3-ASR text decoder."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from qwen3_asr_mlx.config import TextDecoderConfig
from qwen3_asr_mlx.decoder import (
    Attention,
    DecoderLayer,
    KVCache,
    MLP,
    TextDecoder,
    _apply_rotary_pos_emb,
    _build_rope,
    _rotate_half,
    load_decoder_weights,
)

MODEL_PATH = Path(
    "/Users/soroush/.cache/huggingface/hub"
    "/models--mlx-community--Qwen3-ASR-1.7B-bf16"
    "/snapshots/e1f6c266914abc5a46e8756e02580f834a6cf8a7"
)

# ---------------------------------------------------------------------------
# Tiny config for fast unit tests
# ---------------------------------------------------------------------------

def _tiny_config() -> TextDecoderConfig:
    return TextDecoderConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=512,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )


def _full_config() -> TextDecoderConfig:
    return TextDecoderConfig()


# ---------------------------------------------------------------------------
# RMSNorm behaviour
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape_unchanged(self):
        norm = nn.RMSNorm(64, eps=1e-6)
        x = mx.ones((2, 5, 64))
        out = norm(x)
        mx.eval(out)
        assert out.shape == (2, 5, 64)

    def test_unit_weight_normalises(self):
        """With weight=1 RMSNorm should produce unit RMS."""
        norm = nn.RMSNorm(64, eps=1e-6)
        # Reset weight to ones
        norm.load_weights([("weight", mx.ones((64,)))])
        x = mx.array(np.random.randn(1, 8, 64).astype(np.float32))
        out = norm(x)
        mx.eval(out)
        # RMS of output along last dim should be ≈ 1
        rms = mx.sqrt(mx.mean(out ** 2, axis=-1))
        assert np.allclose(np.array(rms), 1.0, atol=1e-4)

    def test_output_matches_manual(self):
        """RMSNorm output should match a manual numpy computation."""
        d = 32
        norm = nn.RMSNorm(d, eps=1e-6)
        norm.load_weights([("weight", mx.ones((d,)))])
        x_np = np.random.randn(1, 4, d).astype(np.float32)
        x = mx.array(x_np)
        out = norm(x)
        mx.eval(out)

        rms = np.sqrt((x_np ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        expected = x_np / rms
        assert np.allclose(np.array(out), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_cos_sin_shapes(self):
        B, T, head_dim = 1, 8, 16
        position_ids = mx.arange(T, dtype=mx.int32)[None]  # (1, T)
        cos, sin = _build_rope(position_ids, head_dim, theta=1e6)
        mx.eval(cos, sin)
        assert cos.shape == (B, 1, T, head_dim), f"cos shape: {cos.shape}"
        assert sin.shape == (B, 1, T, head_dim), f"sin shape: {sin.shape}"

    def test_rotate_half_shape(self):
        x = mx.ones((1, 4, 8, 16))
        out = _rotate_half(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_rotate_half_formula(self):
        """rotate_half([a, b]) should equal [-b, a]."""
        x_np = np.arange(8, dtype=np.float32).reshape(1, 8)
        x = mx.array(x_np)
        out = _rotate_half(x)
        mx.eval(out)
        half = 4
        expected = np.concatenate([-x_np[:, half:], x_np[:, :half]], axis=-1)
        assert np.allclose(np.array(out), expected, atol=1e-6)

    def test_rope_preserves_norm(self):
        """RoPE rotation should preserve the L2 norm of query/key vectors."""
        head_dim = 16
        B, H, T = 1, 4, 6
        position_ids = mx.arange(T, dtype=mx.int32)[None]
        cos, sin = _build_rope(position_ids, head_dim)
        q = mx.array(np.random.randn(B, H, T, head_dim).astype(np.float32))
        k = mx.array(np.random.randn(B, H, T, head_dim).astype(np.float32))
        q_rot, k_rot = _apply_rotary_pos_emb(q, k, cos, sin)
        mx.eval(q_rot, k_rot)

        q_norm     = np.linalg.norm(np.array(q),     axis=-1)
        q_rot_norm = np.linalg.norm(np.array(q_rot), axis=-1)
        assert np.allclose(q_norm, q_rot_norm, atol=1e-4), "RoPE changed query norms"

    def test_cos_sin_values(self):
        """Verify cos/sin at position 0 with a known frequency."""
        position_ids = mx.zeros((1, 1), dtype=mx.int32)
        cos, sin = _build_rope(position_ids, head_dim=4, theta=1e6)
        mx.eval(cos, sin)
        # At position 0: cos=1, sin=0 for all frequencies
        cos_np = np.array(cos).reshape(-1)
        sin_np = np.array(sin).reshape(-1)
        assert np.allclose(cos_np, 1.0, atol=1e-5), f"cos at pos 0: {cos_np}"
        assert np.allclose(sin_np, 0.0, atol=1e-5), f"sin at pos 0: {sin_np}"


# ---------------------------------------------------------------------------
# QK norm (Attention)
# ---------------------------------------------------------------------------

class TestQKNorm:
    def test_qk_norm_applied_per_head(self):
        """After q_norm, each head vector should have unit RMS (weight=1)."""
        config = _tiny_config()
        attn = Attention(config)
        # Reset norms to weight=1
        attn.q_norm.load_weights([("weight", mx.ones((config.head_dim,)))])
        attn.k_norm.load_weights([("weight", mx.ones((config.head_dim,)))])

        B, T = 1, 4
        x = mx.array(np.random.randn(B, T, config.hidden_size).astype(np.float32))
        q_raw = attn.q_proj(x)
        q_heads = q_raw.reshape(B, T, config.num_attention_heads, config.head_dim)
        q_normed = attn.q_norm(q_heads)
        mx.eval(q_normed)

        rms = np.sqrt((np.array(q_normed) ** 2).mean(axis=-1) + 1e-6)
        assert np.allclose(rms, 1.0, atol=1e-3), f"QK-norm RMS: {rms}"


# ---------------------------------------------------------------------------
# GQA output shape
# ---------------------------------------------------------------------------

class TestGQA:
    def test_attention_output_shape(self):
        """Attention output should be (B, T, hidden_size) with GQA."""
        config = _tiny_config()
        attn = Attention(config)
        B, T = 1, 5
        x = mx.array(np.random.randn(B, T, config.hidden_size).astype(np.float32))
        pos = mx.arange(T, dtype=mx.int32)[None]
        out = attn(x, pos, cache=None, layer_idx=0)
        mx.eval(out)
        assert out.shape == (B, T, config.hidden_size), f"Attn output shape: {out.shape}"

    def test_kv_head_expansion(self):
        """Internally k/v should be repeated to match n_heads."""
        config = _tiny_config()
        assert config.num_attention_heads // config.num_key_value_heads == 2
        # Just verify the forward pass succeeds and produces correct output shape
        attn = Attention(config)
        B, T = 1, 3
        x = mx.array(np.random.randn(B, T, config.hidden_size).astype(np.float32))
        pos = mx.arange(T, dtype=mx.int32)[None]
        out = attn(x, pos, cache=None, layer_idx=0)
        mx.eval(out)
        assert out.shape == (B, T, config.hidden_size)


# ---------------------------------------------------------------------------
# Full decoder forward pass
# ---------------------------------------------------------------------------

class TestDecoderForwardShape:
    def test_token_ids_input(self):
        """Token ID input should produce (1, T, vocab_size) logits."""
        config = _tiny_config()
        decoder = TextDecoder(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])  # (1, 5)
        pos = mx.arange(5, dtype=mx.int32)[None]
        logits = decoder(input_ids, pos)
        mx.eval(logits)
        assert logits.shape == (1, 5, config.vocab_size), f"Shape: {logits.shape}"

    def test_embedding_input(self):
        """Pre-computed embedding input should produce correct logit shape."""
        config = _tiny_config()
        decoder = TextDecoder(config)
        B, T, H = 1, 6, config.hidden_size
        embeds = mx.array(np.random.randn(B, T, H).astype(np.float32))
        pos = mx.arange(T, dtype=mx.int32)[None]
        logits = decoder(embeds, pos, is_embeds=True)
        mx.eval(logits)
        assert logits.shape == (B, T, config.vocab_size)

    def test_lm_head_tied(self):
        """lm_head should share weights with embed_tokens."""
        config = _tiny_config()
        decoder = TextDecoder(config)
        # Compute logits via forward
        input_ids = mx.array([[0]])
        pos = mx.zeros((1, 1), dtype=mx.int32)
        logits = decoder(input_ids, pos)
        mx.eval(logits)

        # Manually compute tied projection
        h = decoder.embed_tokens(input_ids)
        for i, layer in enumerate(decoder.layers):
            h = layer(h, pos, None, layer_idx=i)
        h = decoder.norm(h)
        manual_logits = h @ decoder.embed_tokens.weight.T
        mx.eval(manual_logits)

        assert np.allclose(
            np.array(logits), np.array(manual_logits), atol=1e-5
        ), "lm_head is not tied to embed_tokens"


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_cache_grows_across_layers(self):
        """Cache should accumulate keys/values layer by layer."""
        cache = KVCache()
        B, H, T, D = 1, 2, 3, 16
        k = mx.ones((B, H, T, D))
        v = mx.ones((B, H, T, D))

        k0, v0 = cache.update(0, k, v)
        mx.eval(k0, v0)
        assert k0.shape == (B, H, T, D)
        assert len(cache.keys) == 1

        k1, v1 = cache.update(1, k, v)
        mx.eval(k1, v1)
        assert len(cache.keys) == 2

    def test_cache_concatenates_along_time(self):
        """Subsequent updates to the same layer should concatenate along axis 2."""
        cache = KVCache()
        B, H, D = 1, 2, 16
        k1 = mx.ones((B, H, 3, D))
        v1 = mx.ones((B, H, 3, D))
        k2 = mx.ones((B, H, 2, D)) * 2.0
        v2 = mx.ones((B, H, 2, D)) * 2.0

        cache.update(0, k1, v1)
        k_full, v_full = cache.update(0, k2, v2)
        mx.eval(k_full, v_full)
        assert k_full.shape == (B, H, 5, D), f"Expected (1,2,5,16), got {k_full.shape}"

    def test_cached_decode_matches_full_context(self):
        """A cached single-token decode should match the last position of a full-context pass."""
        config = _tiny_config()
        decoder = TextDecoder(config)

        T = 4
        input_ids = mx.array([[1, 2, 3, 4]])  # (1, T)
        pos_full  = mx.arange(T, dtype=mx.int32)[None]

        # Full context pass (no cache)
        logits_full = decoder(input_ids, pos_full, cache=None)
        mx.eval(logits_full)
        last_logits = np.array(logits_full[:, -1, :])

        # Cached pass: prefill T-1 tokens, then decode the last one
        cache = KVCache()
        prefill_ids = input_ids[:, :-1]  # (1, T-1)
        pos_prefill = mx.arange(T - 1, dtype=mx.int32)[None]
        decoder(prefill_ids, pos_prefill, cache=cache)
        cache.offset = T - 1
        mx.eval(cache.keys[0], cache.values[0])

        last_id = input_ids[:, -1:]  # (1, 1)
        pos_decode = mx.array([[T - 1]], dtype=mx.int32)
        logits_cached = decoder(last_id, pos_decode, cache=cache)
        mx.eval(logits_cached)
        cached_logits = np.array(logits_cached[:, 0, :])

        assert np.allclose(last_logits, cached_logits, atol=1e-3), (
            "Cached decode does not match full-context decode"
        )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

class TestWeightLoading:
    @pytest.mark.skipif(
        not MODEL_PATH.is_dir(),
        reason="Cached model weights not available",
    )
    def test_load_decoder_weights_and_forward(self):
        """Loading real weights should allow a forward pass without error."""
        config = _full_config()
        decoder = TextDecoder(config)
        load_decoder_weights(decoder, MODEL_PATH)

        input_ids = mx.array([[1, 2, 3]])
        pos = mx.arange(3, dtype=mx.int32)[None]
        logits = decoder(input_ids, pos)
        mx.eval(logits)

        assert logits.shape == (1, 3, config.vocab_size)
        assert not mx.any(mx.isnan(logits)).item(),  "NaN in decoder output"
        assert not mx.any(mx.isinf(logits)).item(),  "Inf in decoder output"

    @pytest.mark.skipif(
        not MODEL_PATH.is_dir(),
        reason="Cached model weights not available",
    )
    def test_embed_tokens_weight_shape(self):
        """embed_tokens weight should be (vocab_size, hidden_size)."""
        config = _full_config()
        decoder = TextDecoder(config)
        load_decoder_weights(decoder, MODEL_PATH)
        w = decoder.embed_tokens.weight
        mx.eval(w)
        assert w.shape == (config.vocab_size, config.hidden_size), f"Shape: {w.shape}"
