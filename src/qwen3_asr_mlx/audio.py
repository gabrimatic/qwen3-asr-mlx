# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Mel spectrogram frontend for Qwen3-ASR."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

# Mel spectrogram parameters (Qwen3-ASR spec)
SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
F_MIN = 0.0
F_MAX = 8_000.0

# Module-level mel filterbank cache: key -> (n_fft, n_mels, sr, f_min, f_max)
_mel_filterbank_cache: dict[tuple, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Mel filterbank (Slaney-style, same as librosa default)
# ---------------------------------------------------------------------------

def _hz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to mel using Slaney formula."""
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel to Hz using Slaney formula."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build a Slaney-normalised mel filterbank matrix.

    Returns shape (n_mels, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    # Evenly spaced mel points, then convert back to Hz
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    f_points = _mel_to_hz(mel_points)  # shape (n_mels + 2,)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        f_left = f_points[i]
        f_center = f_points[i + 1]
        f_right = f_points[i + 2]

        # Rising slope
        up_slope = (fft_freqs - f_left) / (f_center - f_left)
        # Falling slope
        down_slope = (f_right - fft_freqs) / (f_right - f_center)

        filters[i] = np.maximum(0.0, np.minimum(up_slope, down_slope))

        # Slaney area normalisation: divide by filter width in Hz
        width = f_right - f_left
        if width > 0.0:
            filters[i] /= width

    return filters


def _get_mel_filterbank(
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Return the mel filterbank, computing it once and caching."""
    key = (n_fft, n_mels, sample_rate, f_min, f_max)
    if key not in _mel_filterbank_cache:
        _mel_filterbank_cache[key] = _build_mel_filterbank(
            n_fft, n_mels, sample_rate, f_min, f_max
        )
    return _mel_filterbank_cache[key]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _read_wav_pcm(path: str | Path) -> tuple[np.ndarray, int]:
    """Fast path for plain PCM WAV files (PCM16, PCM32, float32).

    Returns (samples_float32, sample_rate).
    Falls back to soundfile on anything unusual.
    """
    path = Path(path)
    try:
        with open(path, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                raise ValueError("Not a RIFF file")
            f.read(4)  # file size
            wave = f.read(4)
            if wave != b"WAVE":
                raise ValueError("Not WAVE")

            audio_format = channels = sample_rate = bits_per_sample = None
            data_offset = data_size = None

            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                (chunk_size,) = struct.unpack("<I", f.read(4))
                if chunk_id == b"fmt ":
                    fmt = f.read(chunk_size)
                    audio_format = struct.unpack_from("<H", fmt, 0)[0]
                    channels = struct.unpack_from("<H", fmt, 2)[0]
                    sample_rate = struct.unpack_from("<I", fmt, 4)[0]
                    bits_per_sample = struct.unpack_from("<H", fmt, 14)[0]
                elif chunk_id == b"data":
                    data_offset = f.tell()
                    data_size = chunk_size
                    f.seek(chunk_size, 1)
                else:
                    f.seek(chunk_size, 1)

            if data_offset is None or audio_format is None:
                raise ValueError("Missing fmt or data chunk")

            # audio_format: 1 = PCM, 3 = IEEE float
            if audio_format not in (1, 3):
                raise ValueError(f"Unsupported audio format: {audio_format}")

            f.seek(data_offset)
            raw = f.read(data_size)

        n_samples = data_size // (bits_per_sample // 8) // channels

        if audio_format == 1 and bits_per_sample == 16:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif audio_format == 1 and bits_per_sample == 32:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif audio_format == 3 and bits_per_sample == 32:
            samples = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"Unsupported PCM depth: {bits_per_sample}-bit")

        # Interleaved channels -> mono by averaging
        if channels > 1:
            samples = samples[: n_samples * channels].reshape(n_samples, channels).mean(axis=1)

        return samples.astype(np.float32), sample_rate

    except Exception:
        # Fall through to soundfile
        raise


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and return a mono float32 numpy array at target_sr.

    Handles PCM16/PCM32/float32 WAV natively; falls back to soundfile for
    other formats.  Resamples with linear interpolation if needed.
    """
    path = Path(path)
    samples: Optional[np.ndarray] = None
    sr: Optional[int] = None

    if path.suffix.lower() == ".wav":
        try:
            samples, sr = _read_wav_pcm(path)
        except Exception:
            pass

    if samples is None:
        import soundfile as sf
        samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

    # Resample if necessary
    if sr != target_sr:
        target_len = int(len(samples) * target_sr / sr)
        samples = np.interp(
            np.linspace(0.0, len(samples) - 1, target_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

    return samples


# ---------------------------------------------------------------------------
# STFT + mel spectrogram
# ---------------------------------------------------------------------------

def _stft(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Compute the STFT using numpy.

    Returns complex array of shape (n_fft // 2 + 1, n_frames).
    Uses a Hann window.  Pads the signal symmetrically so that the first
    frame is centred on sample 0 (librosa-style centre=True).
    """
    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    waveform = np.pad(waveform, pad, mode="reflect")

    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    n_freqs = n_fft // 2 + 1

    stft = np.empty((n_freqs, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start : start + n_fft] * window
        stft[:, i] = np.fft.rfft(frame, n=n_fft)

    return stft


def log_mel_spectrogram(
    audio: np.ndarray | str | Path,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> mx.array:
    """Compute the log-mel spectrogram as specified by Qwen3-ASR.

    Processing steps:
      1. Load and resample to 16 kHz mono (if a path is given).
      2. STFT with n_fft=400, hop=160, Hann window.
      3. Power spectrum (drop last STFT frame).
      4. Slaney-normalised mel filterbank (128 bins, 0-8000 Hz).
      5. log10(clamp(mel, min=1e-10)).
      6. Dynamic range: max(log_spec, log_spec.max() - 8.0).
      7. Normalise: (log_spec + 4.0) / 4.0.

    Returns an mlx.core.array of shape (n_mels, n_frames).
    """
    if not isinstance(audio, np.ndarray):
        audio = load_audio(audio, target_sr=sample_rate)

    stft = _stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Power spectrum, drop last STFT frame
    magnitudes = np.abs(stft[:, :-1]) ** 2  # (n_freqs, n_frames)

    mel_filters = _get_mel_filterbank(
        n_fft=n_fft, n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max
    )  # (n_mels, n_freqs)

    mel_spec = mel_filters @ magnitudes  # (n_mels, n_frames)

    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return mx.array(log_spec)
