# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Tests for the audio mel spectrogram frontend."""

import numpy as np
import pytest

from qwen3_asr_mlx.audio import (
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
    _get_mel_filterbank,
    _mel_filterbank_cache,
    log_mel_spectrogram,
)


def _silence(duration_s: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a silent (zero) audio array."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _tone(freq_hz: float = 440.0, duration_s: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a pure sine tone."""
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestMelShape:
    def test_1s_silence_shape(self):
        audio = _silence(1.0)
        mel = log_mel_spectrogram(audio)
        # After centre-padding and STFT, n_frames = (n_samples + n_fft // 2 * 2 - n_fft) // hop + 1 - 1
        # For 1s at 16kHz: 16000 samples
        assert mel.shape[0] == N_MELS, f"Expected {N_MELS} mel bins, got {mel.shape[0]}"
        assert mel.shape[1] > 0, "Expected non-zero time frames"

    def test_1s_tone_shape(self):
        audio = _tone(440.0, 1.0)
        mel = log_mel_spectrogram(audio)
        assert mel.shape[0] == N_MELS

    def test_frame_count_approx(self):
        """1 second of audio at 16kHz should produce ~100 frames (10ms hop)."""
        audio = _silence(1.0)
        mel = log_mel_spectrogram(audio)
        # With centre padding: padded_len = 16000 + 2*(N_FFT//2) = 16400
        # n_stft_frames = (16400 - N_FFT) // HOP_LENGTH + 1 = (16400 - 400) // 160 + 1 = 101
        # mel frames = n_stft_frames - 1 = 100  (we drop last)
        assert mel.shape[1] == 100, f"Expected 100 frames for 1s audio, got {mel.shape[1]}"

    def test_varying_duration(self):
        for duration in (0.5, 2.0, 5.0):
            audio = _silence(duration)
            mel = log_mel_spectrogram(audio)
            assert mel.shape[0] == N_MELS, f"Wrong mel bins for {duration}s audio"
            assert mel.shape[1] > 0


# ---------------------------------------------------------------------------
# Value tests
# ---------------------------------------------------------------------------

class TestMelValues:
    def test_output_range(self):
        """After normalisation, values should cluster around [-1, 3] range."""
        audio = _tone(440.0, 1.0)
        mel = log_mel_spectrogram(audio)
        arr = np.array(mel)
        # Values after (x + 4.0) / 4.0: silence maps to ~0, peaks to higher
        assert arr.max() < 10.0, "Unexpectedly large values"
        assert arr.min() > -5.0, "Unexpectedly small values"

    def test_silence_low_values(self):
        """Silent audio should produce log10(1e-10) normalised values."""
        audio = _silence(1.0)
        mel = log_mel_spectrogram(audio)
        arr = np.array(mel)
        # log10(1e-10) = -10; after dynamic range clip and normalise: (-10 + 4) / 4 = -1.5
        # but dynamic range rule: max(x, x.max() - 8) so silence col is all -10, max=-10, all values >= -10-8=-18
        # actual: all -10, after dynamic range all -10, normalise: (-10+4)/4 = -1.5
        expected = (-10.0 + 4.0) / 4.0  # -1.5
        assert np.allclose(arr, expected, atol=1e-3), f"Silence should map to {expected}, got {arr.mean():.4f}"

    def test_mlx_array_type(self):
        import mlx.core as mx
        audio = _silence(0.5)
        mel = log_mel_spectrogram(audio)
        assert isinstance(mel, mx.array), f"Expected mx.array, got {type(mel)}"


# ---------------------------------------------------------------------------
# Filterbank cache tests
# ---------------------------------------------------------------------------

class TestMelFilterbankCache:
    def test_cache_same_object(self):
        """Calling _get_mel_filterbank twice should return the exact same array."""
        _mel_filterbank_cache.clear()
        fb1 = _get_mel_filterbank()
        fb2 = _get_mel_filterbank()
        assert fb1 is fb2, "Mel filterbank should be cached (same object)"

    def test_cache_populated_after_spectrogram(self):
        """log_mel_spectrogram should populate the cache."""
        _mel_filterbank_cache.clear()
        assert len(_mel_filterbank_cache) == 0
        audio = _silence(0.1)
        log_mel_spectrogram(audio)
        assert len(_mel_filterbank_cache) == 1

    def test_filterbank_shape(self):
        fb = _get_mel_filterbank()
        assert fb.shape == (N_MELS, N_FFT // 2 + 1), (
            f"Filterbank shape mismatch: {fb.shape}"
        )

    def test_filterbank_non_negative(self):
        fb = _get_mel_filterbank()
        assert (fb >= 0).all(), "Filterbank weights should be non-negative"
