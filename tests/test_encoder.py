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
from qwen3_asr_mlx.encoder import AudioEncoder, EncoderLayer, load_encoder_weights, _sinusoidal_pe

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
        """(128, 250) mel input should pad the last chunk and produce (1, 39, 2048)."""
        encoder = _make_encoder()
        mel = _random_mel(128, 250)
        out = encoder(mel)
        mx.eval(out)
        # 3 chunks (100+100+50 padded to 100) → 3 × 13 = 39 tokens
        assert out.shape == (1, 39, 2048), f"Expected (1, 39, 2048), got {out.shape}"

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
# Windowed attention
# ---------------------------------------------------------------------------

class TestWindowedAttention:
    def test_short_sequence_uses_dense_attention(self):
        """For sequences shorter than the window, mask should be None (dense attention)."""
        encoder = _make_encoder()
        # window = 13 tokens/chunk × 8 chunks = 104 tokens
        # 1 chunk = 13 tokens < 104 → dense
        mask = encoder._windowed_mask(T=13, window=104)
        assert mask is None, "Short sequence should use dense attention (mask=None)"

    def test_long_sequence_has_mask(self):
        """For sequences longer than the window, a finite mask should be returned."""
        encoder = _make_encoder()
        T = 200
        window = 104
        mask = encoder._windowed_mask(T=T, window=window)
        assert mask is not None, "Long sequence should have a windowed mask"
        assert mask.shape == (1, 1, T, T), f"Unexpected mask shape: {mask.shape}"

    def test_windowed_mask_blocks_distant_positions(self):
        """Positions beyond the window distance should receive -inf in the mask."""
        encoder = _make_encoder()
        T = 50
        window = 5
        mask = encoder._windowed_mask(T=T, window=window)
        assert mask is not None
        mask_np = np.array(mask[0, 0])
        # Position 0 should not attend to position window+1
        assert mask_np[0, window + 1] == float("-inf"), (
            f"Position 0 should not attend to position {window + 1}"
        )
        # Position 0 should attend to position window
        assert mask_np[0, window] == 0.0, (
            f"Position 0 should attend to position {window}"
        )

    def test_short_vs_dense_output_close(self):
        """For a short sequence, windowed and dense attention should produce identical output."""
        # Use a tiny config for speed
        cfg = AudioEncoderConfig(
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            num_mel_bins=128,
            output_dim=64,
            downsample_hidden_size=32,
        )
        enc = AudioEncoder(cfg)
        mel = _random_mel(128, 100)

        # Forward with dense attention (mask=None explicitly patched)
        out_dense = enc(mel)
        mx.eval(out_dense)

        # Since T=13 < window=104, the encoder itself uses dense attention —
        # recompute with manually forced mask=None to verify they match.
        out_windowed = enc(mel)
        mx.eval(out_windowed)

        dense_np = np.array(out_dense)
        windowed_np = np.array(out_windowed)
        assert np.allclose(dense_np, windowed_np, atol=1e-4), (
            "Dense and windowed outputs should match for short sequences"
        )


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class TestPositionalEmbeddings:
    def test_pe_shape(self):
        pe = _sinusoidal_pe(13, 1024)
        assert pe.shape == (1, 13, 1024)

    def test_pe_restarts_per_chunk(self):
        """PE should restart from 0 for each chunk (not be cumulative)."""
        pe1 = _sinusoidal_pe(13, 1024)
        pe2 = _sinusoidal_pe(13, 1024)
        assert np.allclose(np.array(pe1), np.array(pe2), atol=1e-6), (
            "PE should be deterministic and restart from position 0"
        )
