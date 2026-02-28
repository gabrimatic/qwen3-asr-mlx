# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Tests for the Qwen3-ASR audio encoder."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from qwen3_asr_mlx.config import AudioEncoderConfig
from qwen3_asr_mlx.encoder import AudioEncoder, EncoderLayer, SinusoidalPositionEmbedding, load_encoder_weights

MODEL_PATH = Path(
    "/Users/soroush/.cache/huggingface/hub"
    "/models--mlx-community--Qwen3-ASR-1.7B-bf16"
    "/snapshots/e1f6c266914abc5a46e8756e02580f834a6cf8a7"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_mel(n_mels: int = 128, T: int = 100) -> mx.array:
    """Return a random mel spectrogram of shape (n_mels, T)."""
    return mx.array(np.random.randn(n_mels, T).astype(np.float32))


def _make_encoder() -> AudioEncoder:
    return AudioEncoder(AudioEncoderConfig())


# ---------------------------------------------------------------------------
# Conv stem shape
# ---------------------------------------------------------------------------

class TestConvStem:
    def test_single_chunk_stem_shape(self):
        """100 mel frames through conv stem should yield 13 time steps, 480 ch, 16 freq bins."""
        encoder = _make_encoder()
        chunk = mx.zeros((1, 128, 100, 1))
        # Apply just the three conv layers manually
        x = nn.gelu(encoder.conv2d1(chunk))
        assert x.shape == (1, 64, 50, 480), f"After conv1: {x.shape}"

        x = nn.gelu(encoder.conv2d2(x))
        assert x.shape == (1, 32, 25, 480), f"After conv2: {x.shape}"

        x = nn.gelu(encoder.conv2d3(x))
        assert x.shape[0] == 1
        assert x.shape[1] == 16, f"Expected 16 freq bins after conv3, got {x.shape[1]}"
        assert x.shape[2] == 13, f"Expected 13 time steps after conv3, got {x.shape[2]}"
        assert x.shape[3] == 480, f"Expected 480 channels after conv3, got {x.shape[3]}"


# ---------------------------------------------------------------------------
# Full encoder forward shape
# ---------------------------------------------------------------------------

class TestEncoderForwardShape:
    def test_single_chunk_output_shape(self):
        """(128, 100) mel input should produce (1, 13, 2048) output."""
        encoder = _make_encoder()
        mel = _random_mel(128, 100)
        out = encoder(mel)
        mx.eval(out)
        assert out.shape == (1, 13, 2048), f"Expected (1, 13, 2048), got {out.shape}"

    def test_three_chunks_output_shape(self):
        """(128, 300) mel input (3 chunks) should produce (1, 39, 2048) output."""
        encoder = _make_encoder()
        mel = _random_mel(128, 300)
        out = encoder(mel)
        mx.eval(out)
        assert out.shape == (1, 39, 2048), f"Expected (1, 39, 2048), got {out.shape}"

    def test_partial_last_chunk_shape(self):
        """(128, 250) mel input pads last chunk; only valid tokens are kept."""
        encoder = _make_encoder()
        mel = _random_mel(128, 250)
        out = encoder(mel)
        mx.eval(out)
        # Chunks: 100 (13 tokens) + 100 (13 tokens) + 50 padded to 100 (7 valid tokens)
        # _conv_output_length(50) = 7; total = 13 + 13 + 7 = 33
        assert out.shape == (1, 33, 2048), f"Expected (1, 33, 2048), got {out.shape}"

    def test_batched_input(self):
        """Encoder should accept (1, 128, 100) batched input."""
        encoder = _make_encoder()
        mel = _random_mel(128, 100)[None]  # (1, 128, 100)
        out = encoder(mel)
        mx.eval(out)
        assert out.shape == (1, 13, 2048)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

class TestWeightLoading:
    @pytest.mark.skipif(
        not MODEL_PATH.is_dir(),
        reason="Cached model weights not available",
    )
    def test_load_and_forward(self):
        """Loading real weights should allow a forward pass without error."""
        encoder = _make_encoder()
        load_encoder_weights(encoder, MODEL_PATH)
        mel = _random_mel(128, 100)
        out = encoder(mel)
        mx.eval(out)
        assert out.shape == (1, 13, 2048)
        # Output should be finite
        assert not mx.any(mx.isnan(out)).item(), "NaN in encoder output"
        assert not mx.any(mx.isinf(out)).item(), "Inf in encoder output"


# ---------------------------------------------------------------------------
# Block attention mask
# ---------------------------------------------------------------------------

class TestBlockAttentionMask:
    def test_single_block_returns_none(self):
        """A single block covering the full sequence should return None (dense attention)."""
        encoder = _make_encoder()
        mask = encoder._block_attention_mask(seq_len=13, cu_seqlens=[0, 13])
        assert mask is None, "Single block should use dense attention (mask=None)"

    def test_two_blocks_has_mask(self):
        """Two blocks should produce a non-None block-diagonal mask."""
        encoder = _make_encoder()
        mask = encoder._block_attention_mask(seq_len=26, cu_seqlens=[0, 13, 26])
        assert mask is not None, "Two blocks should produce a mask"
        assert mask.shape == (1, 1, 26, 26), f"Unexpected mask shape: {mask.shape}"

    def test_block_mask_zeros_within_block(self):
        """Positions within the same block should have mask value 0."""
        encoder = _make_encoder()
        mask = encoder._block_attention_mask(seq_len=6, cu_seqlens=[0, 3, 6])
        assert mask is not None
        mask_np = np.array(mask[0, 0])
        # Within block 0: positions 0-2 attending to 0-2
        assert mask_np[0, 0] == 0.0
        assert mask_np[1, 2] == 0.0
        assert mask_np[2, 0] == 0.0

    def test_block_mask_large_negative_across_blocks(self):
        """Positions across different blocks should have large negative mask values."""
        encoder = _make_encoder()
        mask = encoder._block_attention_mask(seq_len=6, cu_seqlens=[0, 3, 6])
        assert mask is not None
        mask_np = np.array(mask[0, 0])
        # Block 0 → block 1 cross-attention should be blocked
        assert mask_np[0, 3] < -1e8, f"Expected large negative, got {mask_np[0, 3]}"
        assert mask_np[3, 0] < -1e8, f"Expected large negative, got {mask_np[3, 0]}"

    def test_conv_output_length(self):
        """_conv_output_length should match empirical conv stem output."""
        assert AudioEncoder._conv_output_length(100) == 13
        assert AudioEncoder._conv_output_length(50) == 7


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class TestPositionalEmbeddings:
    def test_pe_shape(self):
        pe_module = SinusoidalPositionEmbedding(max_positions=13, d_model=1024)
        pe = pe_module(13)
        mx.eval(pe)
        assert pe.shape == (13, 1024)

    def test_pe_restarts_per_chunk(self):
        """PE should restart from 0 for each chunk (not be cumulative)."""
        pe_module = SinusoidalPositionEmbedding(max_positions=13, d_model=1024)
        pe1 = pe_module(13)
        pe2 = pe_module(13)
        mx.eval(pe1, pe2)
        assert np.allclose(np.array(pe1), np.array(pe2), atol=1e-6), (
            "PE should be deterministic and restart from position 0"
        )

    def test_pe_shorter_slice(self):
        """Requesting fewer positions should return a prefix of the full table."""
        pe_module = SinusoidalPositionEmbedding(max_positions=20, d_model=64)
        pe_full = pe_module(20)
        pe_short = pe_module(10)
        mx.eval(pe_full, pe_short)
        assert np.allclose(
            np.array(pe_full[:10]),
            np.array(pe_short),
            atol=1e-6,
        ), "Shorter PE should match the prefix of the full PE"
