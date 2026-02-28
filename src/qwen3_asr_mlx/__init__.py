# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""qwen3-asr-mlx: Qwen3-ASR inference on Apple Silicon via MLX."""

__version__ = "0.1.0"

from .audio import load_audio, log_mel_spectrogram
from .config import AudioEncoderConfig, ModelConfig, TextDecoderConfig
from .decoder import KVCache, TextDecoder, load_decoder_weights
from .encoder import AudioEncoder, load_encoder_weights
from .generate import generate, prepare_inputs, sample
from .model import LANGUAGE_MAP, Qwen3ASR, TranscriptionResult
from .tokenizer import Tokenizer, build_prompt, parse_output

__all__ = [
    "__version__",
    "load_audio",
    "log_mel_spectrogram",
    "AudioEncoderConfig",
    "TextDecoderConfig",
    "ModelConfig",
    "AudioEncoder",
    "load_encoder_weights",
    "KVCache",
    "TextDecoder",
    "load_decoder_weights",
    "generate",
    "prepare_inputs",
    "sample",
    "Tokenizer",
    "build_prompt",
    "parse_output",
    "Qwen3ASR",
    "TranscriptionResult",
    "LANGUAGE_MAP",
]
