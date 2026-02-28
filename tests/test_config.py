# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Tests for model configuration dataclasses."""

import json
import tempfile
from pathlib import Path

import pytest

from qwen3_asr_mlx.config import AudioEncoderConfig, ModelConfig, TextDecoderConfig


# Minimal config.json that mirrors the Qwen3-ASR-1.7B layout
_SAMPLE_CONFIG: dict = {
    "audio_token_id": 151676,
    "audio_start_token_id": 151669,
    "audio_end_token_id": 151670,
    "audio_encoder_config": {
        "d_model": 1024,
        "encoder_layers": 24,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "num_mel_bins": 128,
        "max_source_positions": 1500,
        "output_dim": 2048,
        "n_window": 50,
        "n_window_infer": 800,
        "conv_chunksize": 500,
        "activation_function": "gelu",
        "downsample_hidden_size": 480,
    },
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 6144,
    "hidden_act": "silu",
    "vocab_size": 151936,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "mrope_section": [24, 20, 20],
    "rope_interleaved": True,
}


class TestAudioEncoderConfig:
    def test_from_dict_defaults(self):
        cfg = AudioEncoderConfig.from_dict(_SAMPLE_CONFIG)
        assert cfg.d_model == 1024
        assert cfg.encoder_layers == 24
        assert cfg.encoder_attention_heads == 16
        assert cfg.encoder_ffn_dim == 4096
        assert cfg.num_mel_bins == 128
        assert cfg.max_source_positions == 1500
        assert cfg.output_dim == 2048
        assert cfg.n_window == 50
        assert cfg.n_window_infer == 800
        assert cfg.conv_chunksize == 500
        assert cfg.activation_function == "gelu"
        assert cfg.downsample_hidden_size == 480

    def test_default_values(self):
        cfg = AudioEncoderConfig()
        assert cfg.num_mel_bins == 128
        assert cfg.encoder_layers == 24


class TestTextDecoderConfig:
    def test_from_dict(self):
        cfg = TextDecoderConfig.from_dict(_SAMPLE_CONFIG)
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.intermediate_size == 6144
        assert cfg.hidden_act == "silu"
        assert cfg.vocab_size == 151936
        assert cfg.max_position_embeddings == 65536
        assert abs(cfg.rms_norm_eps - 1e-6) < 1e-12
        assert cfg.rope_theta == 1_000_000.0
        assert cfg.mrope_section == [24, 20, 20]
        assert cfg.rope_interleaved is True


class TestModelConfig:
    def test_from_dict(self):
        cfg = ModelConfig.from_dict(_SAMPLE_CONFIG)
        assert cfg.audio_token_id == 151676
        assert cfg.audio_start_token_id == 151669
        assert cfg.audio_end_token_id == 151670
        assert isinstance(cfg.audio_encoder, AudioEncoderConfig)
        assert isinstance(cfg.text_decoder, TextDecoderConfig)

    def test_nested_audio_encoder(self):
        cfg = ModelConfig.from_dict(_SAMPLE_CONFIG)
        assert cfg.audio_encoder.d_model == 1024
        assert cfg.audio_encoder.num_mel_bins == 128

    def test_nested_text_decoder(self):
        cfg = ModelConfig.from_dict(_SAMPLE_CONFIG)
        assert cfg.text_decoder.hidden_size == 2048
        assert cfg.text_decoder.num_key_value_heads == 8

    def test_from_pretrained_local(self):
        """ModelConfig.from_pretrained should load a local config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(_SAMPLE_CONFIG), encoding="utf-8")
            cfg = ModelConfig.from_pretrained(tmpdir)
            assert cfg.audio_token_id == 151676
            assert cfg.audio_encoder.d_model == 1024

    def test_default_config(self):
        cfg = ModelConfig()
        assert cfg.audio_token_id == 151676
        assert cfg.audio_encoder.d_model == 1024
        assert cfg.text_decoder.hidden_size == 2048
