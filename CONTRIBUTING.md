# Contributing

Bug fixes, performance improvements, better docs. Here's how to get involved.

## Dev Setup

```bash
git clone https://github.com/gabrimatic/qwen3-asr-mlx.git
cd qwen3-asr-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

Tests that load the full model weights are marked `slow`. To skip them:

```bash
python -m pytest tests/ -v -m "not slow"
```

## Architecture

```
src/qwen3_asr_mlx/
├── model.py        # Qwen3ASR public API, chunking, language map
├── audio.py        # Mel spectrogram frontend (pure numpy STFT + Slaney filterbank)
├── encoder.py      # Whisper-style audio encoder (Conv2D stem + transformer)
├── decoder.py      # Qwen3 text decoder (GQA, QK-norm, SwiGLU, RoPE)
├── generate.py     # Autoregressive generation with KV cache
├── tokenizer.py    # BPE tokenizer wrapper + prompt construction
└── config.py       # Model configuration dataclasses
```

Key constraint: **no PyTorch, no transformers.** The entire inference pipeline is implemented in pure MLX + numpy. Keep it that way.

## Testing

```bash
python -m pytest tests/ -v                   # all tests
python -m pytest tests/ -v -m "not slow"     # fast tests only (no model loading)
python -m pytest tests/test_model.py -v      # specific module
```

The `slow` marker covers tests that load the full 1.7B model. Fast tests cover unit-level logic (mel computation, tokenizer, config parsing, sampling, chunking).

## PR Checklist

- One feature or fix per PR. Keep scope tight.
- All tests pass before opening.
- Update `README.md` if user-facing behavior changes.
- Match existing code style. No reformatting unrelated files.
- No PyTorch or transformers dependencies.

## Reporting Issues

Use the [bug report template](https://github.com/gabrimatic/qwen3-asr-mlx/issues/new?template=bug_report.yml). Include:

- Python version and MLX version
- macOS version and chip (e.g., macOS 15.0, M4)
- Model ID or local path used
- Steps to reproduce, expected vs. actual behavior

## Vulnerability Reporting

See [SECURITY.md](SECURITY.md). Do **not** open public issues for security vulnerabilities. Use GitHub's private vulnerability reporting.
