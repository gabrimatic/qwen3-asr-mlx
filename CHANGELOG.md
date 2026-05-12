# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-05-12

### Changed

- Require MLX 0.31+ and align dependency floors with current MLX audio packages.
- Parse the current Hugging Face `thinker_config.audio_config` and `thinker_config.text_config` layout instead of relying on baked-in defaults.
- Let `language=None` use Qwen3-ASR auto language detection, while preserving explicit language hints.
- Decode the generated `<asr_text>` boundary from special token IDs so auto-detected language output is parsed correctly.
- Constrain Python support to 3.10-3.13, matching current MLX wheels.
- Update README language and runtime requirements.

## [0.1.0] - 2026-02-28

### Added

- Pure MLX implementation of Qwen3-ASR 1.7B for Apple Silicon. No PyTorch, no transformers, no third-party ML frameworks.
- Audio encoder: Conv2D stem (3x stride-2, 100-frame chunks), sinusoidal positional embeddings, 24-layer Whisper-style transformer with windowed attention, projection head.
- Text decoder: 28-layer Qwen3 with grouped-query attention (16Q/8KV), QK-normalization, SwiGLU FFN, rotary position embeddings, tied lm_head.
- Autoregressive generation with KV cache, greedy and sampling-based decoding, repetition penalty, top-k, and nucleus sampling.
- Mel spectrogram frontend: pure numpy STFT, Slaney-normalized 128-bin filterbank, Qwen3-ASR log normalization.
- BPE tokenizer wrapper with token-level output parsing (splits on `<asr_text>` token boundary).
- `Qwen3ASR` public API: `from_pretrained`, `transcribe`, `warm_up`, `close`, context manager.
- Automatic language detection from model output.
- Long audio chunking at low-energy boundaries for recordings exceeding 20 minutes.
- Thread-safe transcription via internal lock.
- WAV loader with PCM16/PCM32/float32 support, soundfile fallback for other formats.
- 115 tests covering all modules.
