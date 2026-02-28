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

class SinusoidalPositionEmbedding(nn.Module):
    """Precomputed sinusoidal position embeddings.

    Stores a (max_positions, d_model) table at init time. The ``__call__``
    method slices the first *seqlen* rows, so every chunk gets the same
    positions starting from 0.
    """

    def __init__(self, max_positions: int, d_model: int) -> None:
        super().__init__()
        half = d_model // 2
        log_timescale = math.log(10000.0) / (half - 1)
        inv_timescales = mx.exp(
            -mx.arange(half, dtype=mx.float32) * log_timescale
        )
        positions = mx.arange(max_positions, dtype=mx.float32)[:, None]
        scaled = positions * inv_timescales[None, :]
        self._positional_embedding = mx.concatenate(
            [mx.sin(scaled), mx.cos(scaled)], axis=1
        )

    def __call__(self, seqlen: int) -> mx.array:
        """Return shape (seqlen, d_model)."""
        return self._positional_embedding[:seqlen, :]


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class _MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (no GQA) with optional block mask."""

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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
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
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

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

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = config.n_window * 2  # 50 * 2 = 100

        # Conv2D stem (3 x stride-2 with GELU, channels-last)
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

        # Linear projection from flattened freq x channel to d_model (no bias)
        freq_after_conv = (
            ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv,
            config.d_model,
            bias=False,
        )

        # Precomputed sinusoidal positional embeddings (per-chunk, not global)
        self.positional_embedding = SinusoidalPositionEmbedding(
            config.max_source_positions, config.d_model
        )

        # Transformer layers
        self.layers = [EncoderLayer(config) for _ in range(config.encoder_layers)]

        # Post-encoder projection
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_output_length(input_length: int) -> int:
        """Compute output length after the 3-layer stride-2 conv stem.

        Each ``Conv2d(stride=2, padding=1, kernel=3)`` maps length *L* to
        ``(L + 2*1 - 3) // 2 + 1 = (L - 1) // 2 + 1``.
        """
        L = input_length
        for _ in range(3):
            L = (L - 1) // 2 + 1
        return L

    @staticmethod
    def _block_attention_mask(
        seq_len: int, cu_seqlens: list[int],
    ) -> Optional[mx.array]:
        """Build an additive block-diagonal attention mask.

        Each block defined by consecutive entries in *cu_seqlens* can attend
        only within itself.  Returns ``None`` when a single block covers the
        entire sequence.

        Returns shape (1, 1, seq_len, seq_len).
        """
        if len(cu_seqlens) <= 2:
            return None

        mask = mx.full((seq_len, seq_len), -1e9)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            mask[start:end, start:end] = 0.0
        return mask[None, None]

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
        if mel.ndim == 3:
            mel = mel[0]
        n_mels, T = mel.shape
        chunk_size = self.chunk_size

        # ------ split mel into chunks, pad last chunk if needed ------
        chunk_mels: list[mx.array] = []
        chunk_real_lengths: list[int] = []
        offset = 0
        while offset < T:
            segment = mel[:, offset : offset + chunk_size]
            real_len = segment.shape[1]
            chunk_real_lengths.append(real_len)
            if real_len < chunk_size:
                pad = mx.zeros(
                    (n_mels, chunk_size - real_len), dtype=segment.dtype
                )
                segment = mx.concatenate([segment, pad], axis=1)
            chunk_mels.append(segment)
            offset += chunk_size

        # ------ batch all chunks through the conv stem ------
        # (num_chunks, n_mels, chunk_size, 1)
        batched = mx.stack(chunk_mels, axis=0)[:, :, :, None]
        x = nn.gelu(self.conv2d1(batched))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        B, freq, time, ch = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(B, time, ch * freq)
        x = self.conv_out(x)  # (num_chunks, tokens_per_chunk, d_model)

        # ------ per-chunk sinusoidal positional embeddings ------
        # Each chunk gets the same positions 0..tokens_per_chunk-1.
        # Applied BEFORE stripping padding, matching training behavior.
        tokens_per_chunk = x.shape[1]
        pe = self.positional_embedding(tokens_per_chunk)  # (tokens_per_chunk, d_model)
        x = x + pe[None, :, :]  # broadcast across all chunks

        # ------ strip padding tokens, keep only valid frames per chunk ------
        valid_lengths = [
            self._conv_output_length(rl) for rl in chunk_real_lengths
        ]
        hidden_list = [x[i, :valid_lengths[i]] for i in range(len(chunk_mels))]
        hidden = mx.concatenate(hidden_list, axis=0)  # (total_tokens, d_model)

        # ------ block attention mask ------
        # Window size in tokens: tokens_per_full_chunk * (n_window_infer / chunk_size)
        total_tokens = hidden.shape[0]
        window_tokens = tokens_per_chunk * (
            self.config.n_window_infer // chunk_size
        )

        # Build cumulative sequence lengths: fixed-size contiguous blocks
        cu_seqlens = [0]
        num_full_windows = total_tokens // window_tokens
        for _ in range(num_full_windows):
            cu_seqlens.append(cu_seqlens[-1] + window_tokens)
        remainder = total_tokens % window_tokens
        if remainder > 0:
            cu_seqlens.append(cu_seqlens[-1] + remainder)

        mask = self._block_attention_mask(total_tokens, cu_seqlens)

        # ------ transformer ------
        hidden = hidden[None]  # (1, total_tokens, d_model)
        for layer in self.layers:
            hidden = layer(hidden, mask=mask)

        # ------ post-encoder projection ------
        hidden = self.ln_post(hidden)
        hidden = nn.gelu(self.proj1(hidden))
        hidden = self.proj2(hidden)

        return hidden  # (1, total_tokens, output_dim)


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
