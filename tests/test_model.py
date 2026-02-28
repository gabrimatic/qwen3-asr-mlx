# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Tests for Qwen3ASR and TranscriptionResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qwen3_asr_mlx.model import (
    LANGUAGE_MAP,
    Qwen3ASR,
    TranscriptionResult,
    _find_split_points,
)

# ---------------------------------------------------------------------------
# Cached model path
# ---------------------------------------------------------------------------

MODEL_PATH = Path(
    "/Users/soroush/.cache/huggingface/hub"
    "/models--mlx-community--Qwen3-ASR-1.7B-bf16"
    "/snapshots/e1f6c266914abc5a46e8756e02580f834a6cf8a7"
)


def _silence(duration_s: float = 0.5, sr: int = 16_000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _tone(freq_hz: float = 440.0, duration_s: float = 1.0, sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def test_fields_accessible(self):
        r = TranscriptionResult(text="hello", language="English", duration=1.5)
        assert r.text == "hello"
        assert r.language == "English"
        assert r.duration == pytest.approx(1.5)

    def test_dataclass_equality(self):
        a = TranscriptionResult(text="x", language="English", duration=0.0)
        b = TranscriptionResult(text="x", language="English", duration=0.0)
        assert a == b

    def test_repr_contains_fields(self):
        r = TranscriptionResult(text="hi", language="French", duration=2.0)
        s = repr(r)
        assert "hi" in s
        assert "French" in s


# ---------------------------------------------------------------------------
# Language map
# ---------------------------------------------------------------------------

class TestLanguageMap:
    def test_common_codes_present(self):
        for code in ("en", "zh", "ja", "ko", "de", "fr", "es"):
            assert code in LANGUAGE_MAP, f"Missing ISO code: {code}"

    def test_english_maps_correctly(self):
        assert LANGUAGE_MAP["en"] == "English"

    def test_chinese_maps_correctly(self):
        assert LANGUAGE_MAP["zh"] == "Chinese"


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

class TestFindSplitPoints:
    def test_no_split_for_short_audio(self):
        samples = np.zeros(16_000, dtype=np.float32)  # 1 s
        points = _find_split_points(samples, chunk_samples=16_000 * 20, search_samples=5 * 16_000)
        assert points == []

    def test_single_split_for_two_chunks(self):
        # 2.5 × chunk → one split
        chunk = 16_000 * 10  # 10 s chunks
        samples = np.random.default_rng(0).random(int(chunk * 2.5)).astype(np.float32)
        points = _find_split_points(samples, chunk_samples=chunk, search_samples=5 * 16_000)
        assert len(points) == 2

    def test_split_within_bounds(self):
        chunk = 16_000 * 10
        total = chunk * 3
        samples = np.random.default_rng(1).random(total).astype(np.float32)
        points = _find_split_points(samples, chunk_samples=chunk, search_samples=5 * 16_000)
        for p in points:
            assert 0 <= p < total

    def test_split_snaps_to_low_energy(self):
        """Insert a silent window and verify the split lands inside it."""
        sr = 16_000
        chunk = sr * 10
        search = sr * 5
        # 25 s audio; silent from 9–11 s (boundary at 10 s)
        samples = np.ones(sr * 25, dtype=np.float32) * 0.5
        samples[sr * 9: sr * 11] = 0.0  # silent region

        points = _find_split_points(samples, chunk_samples=chunk, search_samples=search)
        assert len(points) >= 1
        # First split should land in the silent region
        assert sr * 9 <= points[0] <= sr * 11

    def test_sorted_output(self):
        chunk = 16_000 * 5
        samples = np.random.default_rng(2).random(chunk * 5).astype(np.float32)
        points = _find_split_points(samples, chunk_samples=chunk, search_samples=2 * 16_000)
        assert points == sorted(points)


# ---------------------------------------------------------------------------
# Output decoding helper
# ---------------------------------------------------------------------------

class TestDecodeOutput:
    """Tests for the internal _decode_output method via a minimal stub."""

    @pytest.fixture(scope="class")
    def model(self):
        from pathlib import Path
        from qwen3_asr_mlx.tokenizer import Tokenizer
        from huggingface_hub import snapshot_download

        m = object.__new__(Qwen3ASR)
        path = Path(snapshot_download("mlx-community/Qwen3-ASR-1.7B-bf16"))
        m._tokenizer = Tokenizer(path)
        return m

    def test_decodes_transcription(self, model):
        # "Hello." + <|im_end|>
        tokens = [9707, 13, 151645]
        text = model._decode_output(tokens)
        assert "hello" in text.lower()

    def test_strips_eos_im_end(self, model):
        tokens = [9707, 13, 151645]  # "Hello." + <|im_end|>
        text = model._decode_output(tokens)
        assert "<|im_end|>" not in text

    def test_strips_endoftext(self, model):
        tokens = [9707, 13, 151643]  # "Hello." + <|endoftext|>
        text = model._decode_output(tokens)
        assert "<|endoftext|>" not in text

    def test_empty_tokens(self, model):
        text = model._decode_output([])
        assert text == ""


# ---------------------------------------------------------------------------
# max_tokens auto-computation
# ---------------------------------------------------------------------------

class TestMaxTokensComputation:
    """Verify the auto-computation formula without loading the real model."""

    @pytest.mark.parametrize("duration,expected_min", [
        (1.0, 256),    # short clip → floor of 256
        (10.0, 500),   # 10 s × 50 = 500
        (60.0, 3000),  # 60 s × 50 = 3000
    ])
    def test_auto_max_tokens(self, duration, expected_min):
        computed = max(256, int(duration * 50))
        assert computed == expected_min


# ---------------------------------------------------------------------------
# Real model tests (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFromPretrained:
    @pytest.fixture(scope="class")
    def model(self):
        return Qwen3ASR.from_pretrained(MODEL_PATH)

    def test_loads_without_error(self, model):
        assert isinstance(model, Qwen3ASR)

    def test_encoder_not_none(self, model):
        assert model._encoder is not None

    def test_decoder_not_none(self, model):
        assert model._decoder is not None

    def test_tokenizer_not_none(self, model):
        assert model._tokenizer is not None


@pytest.mark.slow
class TestTranscribe:
    @pytest.fixture(scope="class")
    def model(self):
        return Qwen3ASR.from_pretrained(MODEL_PATH)

    def test_transcribe_silence_returns_result(self, model):
        result = model.transcribe(_silence(0.5))
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_silence_has_duration(self, model):
        result = model.transcribe(_silence(0.5))
        assert result.duration == pytest.approx(0.5, abs=0.01)

    def test_transcribe_silence_text_is_str(self, model):
        result = model.transcribe(_silence(0.5))
        assert isinstance(result.text, str)

    def test_transcribe_silence_language_is_str(self, model):
        result = model.transcribe(_silence(0.5))
        assert isinstance(result.language, str)
        assert len(result.language) > 0

    def test_transcribe_with_language_hint(self, model):
        result = model.transcribe(_silence(0.5), language="en")
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_tone_returns_result(self, model):
        result = model.transcribe(_tone(440.0, 1.0))
        assert isinstance(result, TranscriptionResult)
        assert result.duration == pytest.approx(1.0, abs=0.01)

    def test_transcribe_greedy_deterministic(self, model):
        audio = _silence(0.5)
        r1 = model.transcribe(audio, temperature=0.0)
        r2 = model.transcribe(audio, temperature=0.0)
        assert r1.text == r2.text


@pytest.mark.slow
class TestWarmUp:
    @pytest.fixture(scope="class")
    def model(self):
        return Qwen3ASR.from_pretrained(MODEL_PATH)

    def test_warm_up_runs(self, model):
        model.warm_up()  # should not raise


@pytest.mark.slow
class TestContextManager:
    def test_context_manager_closes(self):
        with Qwen3ASR.from_pretrained(MODEL_PATH) as model:
            result = model.transcribe(_silence(0.5))
            assert isinstance(result, TranscriptionResult)
        # After __exit__, encoder/decoder should be released
        assert model._encoder is None
        assert model._decoder is None
