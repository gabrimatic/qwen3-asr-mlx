# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Qwen3-ASR audio encoder (Whisper-style) implemented in MLX."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import AudioEncoderConfig

# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings
# ---------------------------------------------------------------------------

def _sinusoidal_pe(n_positions: int, d_model: int) -> mx.array:
    """Compute sinusoidal position embeddings for *n_positions* steps.

    Returns shape (1, n_positions, d_model).  Embeddings restart from
    position 0, so callers must invoke this per-chunk if needed.
    """
    half = d_model // 2
    log_timescale = math.log(10000.0) / (half - 1)
    inv_timescales = mx.exp(-mx.arange(half, dtype=mx.float32) * log_timescale)
    positions = mx.arange(n_positions, dtype=mx.float32)
    # (n_positions, half)
    scaled = positions[:, None] * inv_timescales[None, :]
    pe = mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=-1)  # (T, d_model)
    return pe[None]  # (1, T, d_model)


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class _MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (no GQA) with optional windowed mask."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, H, T, D)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        # (B, H, T, D) -> (B, T, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder transformer layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Single Whisper-style encoder transformer layer (pre-norm residual)."""

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = _MultiHeadSelfAttention(
            config.d_model, config.encoder_attention_heads
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention block (pre-norm)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

        # Feed-forward block (pre-norm)
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Audio encoder
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """Qwen3-ASR audio encoder.

    Processes a log-mel spectrogram through a Conv2D stem, sinusoidal
    position embeddings, a Whisper-style transformer, and a final
    projection head to produce audio token embeddings.

    Parameters
    ----------
    config:
        ``AudioEncoderConfig`` instance.  Defaults match the 1.7B model.
    """

    # Number of mel frames per processing chunk
    CHUNK_FRAMES: int = 100

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Conv2D stem (3 × stride-2 with GELU, channels-last)
        self.conv2d1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Linear projection from flattened freq×channel to d_model (no bias)
        freq_after_conv = config.num_mel_bins // 8  # 128 // 8 = 16
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv,
            config.d_model,
            bias=False,
        )

        # Transformer layers
        self.layers = [EncoderLayer(config) for _ in range(config.encoder_layers)]

        # Post-encoder projection
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    # ------------------------------------------------------------------
    # Conv stem
    # ------------------------------------------------------------------

    def _conv_stem(self, chunk: mx.array) -> mx.array:
        """Run one 100-frame chunk through the conv stem.

        Parameters
        ----------
        chunk:
            Shape (1, n_mels, T_chunk, 1) — batch, freq, time, channels.

        Returns
        -------
        mx.array
            Shape (1, T_out, d_model) where T_out ≈ T_chunk // 8.
        """
        x = nn.gelu(self.conv2d1(chunk))   # (1, n_mels/2, T/2, 480)
        x = nn.gelu(self.conv2d2(x))       # (1, n_mels/4, T/4, 480)
        x = nn.gelu(self.conv2d3(x))       # (1, n_mels/8, T/8, 480)

        B, freq, time, ch = x.shape
        # (1, T/8, ch×freq) — channels-major order to match trained conv_out weights
        x = x.transpose(0, 2, 3, 1).reshape(B, time, ch * freq)
        # (1, T/8, d_model)
        x = self.conv_out(x)
        return x

    # ------------------------------------------------------------------
    # Windowed attention mask
    # ------------------------------------------------------------------

    def _windowed_mask(self, T: int, window: int) -> Optional[mx.array]:
        """Build an additive attention mask for windowed self-attention.

        Each position can attend only to positions within [i - window, i + window].
        Returns None when every position falls within the window (dense attention).

        Returns shape (1, 1, T, T) suitable for broadcasting over (B, H, T, T).
        """
        if T <= window:
            return None  # All positions within window — use dense attention

        # Absolute position indices
        rows = mx.arange(T)[:, None]   # (T, 1)
        cols = mx.arange(T)[None, :]   # (1, T)
        dist = mx.abs(rows - cols)
        # Positions outside the window get -inf
        mask = mx.where(dist <= window, mx.zeros((T, T)), mx.full((T, T), float("-inf")))
        return mask[None, None]  # (1, 1, T, T)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, mel: mx.array) -> mx.array:
        """Encode a log-mel spectrogram.

        Parameters
        ----------
        mel:
            Shape (n_mels, T) or (batch, n_mels, T).  Currently only
            batch-size 1 is supported when a batch dimension is present.

        Returns
        -------
        mx.array
            Shape (1, n_tokens, output_dim).
        """
        # Normalise to (n_mels, T)
        if mel.ndim == 3:
            mel = mel[0]  # take first item
        n_mels, T = mel.shape

        chunk_size = self.CHUNK_FRAMES

        # ------ chunk the mel into CHUNK_FRAMES windows ------
        chunks_out: list[mx.array] = []
        offset = 0
        while offset < T:
            segment = mel[:, offset : offset + chunk_size]  # (n_mels, ≤100)
            t = segment.shape[1]
            if t < chunk_size:
                # Pad last (or only) chunk with zeros along time axis
                pad_len = chunk_size - t
                pad = mx.zeros((n_mels, pad_len), dtype=segment.dtype)
                segment = mx.concatenate([segment, pad], axis=1)

            # Reshape to (1, n_mels, T_chunk, 1) for Conv2d
            x = segment.reshape(1, n_mels, chunk_size, 1)
            x = self._conv_stem(x)  # (1, T_out, d_model)

            # Sinusoidal PE — restart from 0 for every chunk
            _, t_out, d = x.shape
            pe = _sinusoidal_pe(t_out, d)  # (1, T_out, d_model)
            x = x + pe

            chunks_out.append(x)
            offset += chunk_size

        # Concatenate all chunks along time: (1, total_T_out, d_model)
        if len(chunks_out) == 1:
            x = chunks_out[0]
        else:
            x = mx.concatenate(chunks_out, axis=1)

        # ------ windowed attention through transformer ------
        # Whisper inference window: n_window_infer=800 frames / 100 frames per chunk × 13 tokens = 104 tokens
        tokens_per_chunk = x.shape[1] // len(chunks_out)
        n_chunks_in_window = self.config.n_window_infer // chunk_size
        window = tokens_per_chunk * n_chunks_in_window  # 13 × 8 = 104

        mask = self._windowed_mask(x.shape[1], window)

        for layer in self.layers:
            x = layer(x, mask=mask)

        # ------ post-encoder projection ------
        x = self.ln_post(x)
        x = nn.gelu(self.proj1(x))
        x = self.proj2(x)

        return x


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_encoder_weights(model: AudioEncoder, model_path: str | Path) -> None:
    """Load audio encoder weights from a safetensors model file.

    Parameters
    ----------
    model:
        An ``AudioEncoder`` instance to populate.
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

    prefix = "audio_tower."
    # Strip prefix and keep only audio_tower weights
    mapped: list[tuple[str, mx.array]] = []
    for key, value in raw.items():
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix):]
        mapped.append((local_key, value))

    model.load_weights(mapped)
    mx.eval(model.parameters())
