# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Model configuration dataclasses for Qwen3-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AudioEncoderConfig:
    """Configuration for the Qwen3-ASR audio encoder (Whisper-style)."""

    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    output_dim: int = 2048
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    activation_function: str = "gelu"
    downsample_hidden_size: int = 480

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AudioEncoderConfig":
        audio_cfg = d.get("audio_encoder_config", d)
        return cls(
            d_model=audio_cfg.get("d_model", cls.d_model),
            encoder_layers=audio_cfg.get(
                "encoder_layers",
                audio_cfg.get("num_hidden_layers", cls.encoder_layers),
            ),
            encoder_attention_heads=audio_cfg.get(
                "encoder_attention_heads", cls.encoder_attention_heads
            ),
            encoder_ffn_dim=audio_cfg.get("encoder_ffn_dim", cls.encoder_ffn_dim),
            num_mel_bins=audio_cfg.get("num_mel_bins", cls.num_mel_bins),
            max_source_positions=audio_cfg.get(
                "max_source_positions", cls.max_source_positions
            ),
            output_dim=audio_cfg.get("output_dim", cls.output_dim),
            n_window=audio_cfg.get("n_window", cls.n_window),
            n_window_infer=audio_cfg.get("n_window_infer", cls.n_window_infer),
            conv_chunksize=audio_cfg.get("conv_chunksize", cls.conv_chunksize),
            activation_function=audio_cfg.get(
                "activation_function", cls.activation_function
            ),
            downsample_hidden_size=audio_cfg.get(
                "downsample_hidden_size", cls.downsample_hidden_size
            ),
        )


@dataclass
class TextDecoderConfig:
    """Configuration for the Qwen3-ASR text decoder (Qwen2-style LLM)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    hidden_act: str = "silu"
    vocab_size: int = 151936
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    rope_interleaved: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextDecoderConfig":
        return cls(
            hidden_size=d.get("hidden_size", cls.hidden_size),
            num_hidden_layers=d.get("num_hidden_layers", cls.num_hidden_layers),
            num_attention_heads=d.get(
                "num_attention_heads", cls.num_attention_heads
            ),
            num_key_value_heads=d.get(
                "num_key_value_heads", cls.num_key_value_heads
            ),
            head_dim=d.get("head_dim", cls.head_dim),
            intermediate_size=d.get("intermediate_size", cls.intermediate_size),
            hidden_act=d.get("hidden_act", cls.hidden_act),
            vocab_size=d.get("vocab_size", cls.vocab_size),
            max_position_embeddings=d.get(
                "max_position_embeddings", cls.max_position_embeddings
            ),
            rms_norm_eps=d.get("rms_norm_eps", cls.rms_norm_eps),
            rope_theta=d.get("rope_theta", cls.rope_theta),
            mrope_section=d.get("mrope_section", [24, 20, 20]),
            rope_interleaved=d.get("rope_interleaved", cls.rope_interleaved),
        )


@dataclass
class ModelConfig:
    """Top-level Qwen3-ASR model configuration."""

    audio_encoder: AudioEncoderConfig = field(
        default_factory=AudioEncoderConfig
    )
    text_decoder: TextDecoderConfig = field(
        default_factory=TextDecoderConfig
    )
    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        audio_encoder = AudioEncoderConfig.from_dict(d)
        text_decoder = TextDecoderConfig.from_dict(d)
        return cls(
            audio_encoder=audio_encoder,
            text_decoder=text_decoder,
            audio_token_id=d.get("audio_token_id", 151676),
            audio_start_token_id=d.get("audio_start_token_id", 151669),
            audio_end_token_id=d.get("audio_end_token_id", 151670),
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "ModelConfig":
        """Load config from a local directory or HuggingFace Hub model ID.

        If *model_path* is an existing directory, ``config.json`` is read
        from it directly.  Otherwise the path is treated as a HuggingFace
        Hub ``repo_id`` and the config is downloaded.
        """
        path = Path(model_path)
        if path.is_dir():
            config_file = path / "config.json"
            d = json.loads(config_file.read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(
                repo_id=str(model_path), filename="config.json"
            )
            d = json.loads(Path(config_file).read_text(encoding="utf-8"))

        return cls.from_dict(d)
