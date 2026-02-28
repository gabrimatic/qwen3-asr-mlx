# qwen3-asr-mlx

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-required-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

**Qwen3-ASR speech-to-text inference on Apple Silicon via MLX.**

Pure MLX implementation of the full [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 1.7B pipeline. No PyTorch, no transformers, no third-party ML frameworks. Three lines to transcribe.

> This package provides inference code only. Model weights are developed by [Qwen Team, Alibaba Cloud](https://qwen.ai) under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) and downloaded separately from HuggingFace Hub on first use.

---

## Quick Start

**Apple Silicon required.** Python 3.10+, MLX 0.22+.

```bash
pip install qwen3-asr-mlx
```

```python
from qwen3_asr_mlx import Qwen3ASR

model = Qwen3ASR.from_pretrained("mlx-community/Qwen3-ASR-1.7B-bf16")
result = model.transcribe("audio.wav")
print(result.text)      # "Hello, world."
print(result.language)  # "English"
print(result.duration)  # 3.2
```

Model weights download automatically from HuggingFace Hub on first use.

---

## Features

- **Fully on-device** via MLX, no server, no cloud, no network during inference
- **Pure implementation** with no PyTorch or transformers dependency
- **All audio formats** via soundfile (WAV, FLAC, MP3, etc.)
- **Automatic language detection** from model output
- **Greedy and sampling-based decoding** with repetition penalty, top-k, and nucleus sampling
- **Thread-safe** with internal lock for concurrent callers
- **Long audio support** with automatic chunking at low-energy boundaries (up to 20 minutes)
- **Context manager** for automatic resource cleanup

---

## API

### `Qwen3ASR.from_pretrained(model_id_or_path, **kwargs)`

Load a model from a local directory or the HuggingFace Hub.

```python
model = Qwen3ASR.from_pretrained("mlx-community/Qwen3-ASR-1.7B-bf16")
# or from a local path
model = Qwen3ASR.from_pretrained("/path/to/model")
```

### `model.transcribe(audio, **kwargs) -> TranscriptionResult`

Transcribe audio to text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `str`, `Path`, or `np.ndarray` | required | File path or float32 numpy array at 16 kHz mono |
| `language` | `str` or `None` | `None` | Optional language hint (ISO 639-1 code or full name) |
| `temperature` | `float` | `0.0` | Sampling temperature; `0.0` = greedy |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `top_k` | `int` | `0` | Top-k cutoff (`0` = disabled) |
| `repetition_penalty` | `float` | `1.2` | Penalty for repeated tokens |
| `max_tokens` | `int` or `None` | `None` | Max output tokens; auto-computed from duration when `None` |
| `chunk_duration` | `float` | `1200.0` | Max seconds per chunk; longer audio is split automatically |

### `model.warm_up()`

Run a short dummy inference to pre-compile the MLX compute graph. Eliminates the latency spike on the first real transcription.

### `model.close()`

Release model weights and free memory. Called automatically when using the context manager.

```python
with Qwen3ASR.from_pretrained("mlx-community/Qwen3-ASR-1.7B-bf16") as model:
    result = model.transcribe("audio.wav")
```

### `TranscriptionResult`

```python
@dataclass
class TranscriptionResult:
    text: str        # clean transcription
    language: str    # detected language (e.g. "English")
    duration: float  # audio duration in seconds
```

---

## Architecture

```
Audio (WAV/FLAC/MP3/ndarray)
  │
  ▼
Mel Spectrogram (128-bin, numpy STFT, Slaney filterbank)
  │
  ▼
Audio Encoder (Conv2D stem → 24-layer transformer → projection)
  │
  ▼
Text Decoder (28-layer Qwen3, GQA 16Q/8KV, QK-norm, SwiGLU, RoPE)
  │
  ▼
TranscriptionResult { text, language, duration }
```

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Python 3.10+
- MLX 0.22+

---

## Development

```bash
git clone https://github.com/gabrimatic/qwen3-asr-mlx.git
cd qwen3-asr-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Skip model-loading tests with `-m "not slow"`.

---

## Credits

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by [Qwen Team](https://qwen.ai), [Alibaba Cloud](https://www.alibabacloud.com) · [MLX](https://github.com/ml-explore/mlx) by [Apple](https://ml-explore.github.io/mlx/) · Model weights from [mlx-community](https://huggingface.co/mlx-community)

<details>
<summary><strong>Legal notices</strong></summary>

### Model License

This package provides inference code only. It does not include model weights.

The [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) model weights are developed by [Qwen Team, Alibaba Cloud](https://qwen.ai) and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The [bf16 MLX conversion](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16) is hosted by [mlx-community](https://huggingface.co/mlx-community) under the same license. By downloading and using the model weights, you agree to the terms of the Apache 2.0 license.

### Trademarks

"Qwen" and "Tongyi Qianwen" are trademarks of Alibaba Cloud. "MLX" is a trademark of Apple Inc. "HuggingFace" is a trademark of Hugging Face, Inc.

This project is not affiliated with, endorsed by, or sponsored by Alibaba Cloud, Apple, Hugging Face, or any other trademark holder. All trademark names are used solely to describe compatibility with their respective technologies.

### Third-Party Licenses

This project depends on:

| Package | License |
|---------|---------|
| [mlx](https://github.com/ml-explore/mlx) | MIT |
| [numpy](https://numpy.org) | BSD-3-Clause |
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause |

</details>

## License

This inference code is released under the MIT License. See [LICENSE](LICENSE) for details.

The model weights have their own license (Apache 2.0). See [Model License](#legal-notices) above.

---

Created by [Soroush Yousefpour](https://gabrimatic.info)

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/gabrimatic)
