# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""High-level Qwen3ASR model: load, transcribe, and manage resources."""

from __future__ import annotations

import gc
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from .audio import load_audio, log_mel_spectrogram
from .config import ModelConfig
from .decoder import TextDecoder, load_decoder_weights
from .encoder import AudioEncoder, load_encoder_weights
from .generate import generate
from .tokenizer import Tokenizer, build_prompt, parse_output


# ---------------------------------------------------------------------------
# ISO 639-1 → full name map (subset covering common languages)
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """Result of a transcription call."""

    text: str
    """Clean transcription text."""

    language: str
    """Detected or specified language (e.g. ``"English"``)."""

    duration: float
    """Audio duration in seconds."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3ASR:
    """Qwen3-ASR speech-to-text model for Apple Silicon.

    Use ``from_pretrained`` to load from a local directory or the
    HuggingFace Hub, then call ``transcribe`` to convert audio to text.

    Examples
    --------
    >>> model = Qwen3ASR.from_pretrained("mlx-community/Qwen3-ASR-1.7B-bf16")
    >>> result = model.transcribe("audio.wav")
    >>> print(result.text)
    """

    def __init__(
        self,
        config: ModelConfig,
        encoder: AudioEncoder,
        decoder: TextDecoder,
        tokenizer: Tokenizer,
    ) -> None:
        self._config = config
        self._encoder = encoder
        self._decoder = decoder
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        **kwargs,
    ) -> "Qwen3ASR":
        """Load a Qwen3-ASR model from a local directory or HuggingFace Hub.

        Parameters
        ----------
        model_id_or_path:
            Path to a local model directory, or a HuggingFace Hub repo ID
            such as ``"mlx-community/Qwen3-ASR-1.7B-bf16"``.

        Returns
        -------
        Qwen3ASR
            A fully loaded and ready-to-use model instance.
        """
        path = Path(model_id_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(repo_id=str(model_id_or_path), **kwargs)
            )

        config = ModelConfig.from_pretrained(path)

        encoder = AudioEncoder(config.audio_encoder)
        load_encoder_weights(encoder, path)

        decoder = TextDecoder(config.text_decoder)
        load_decoder_weights(decoder, path)

        tokenizer = Tokenizer(path)

        return cls(config, encoder, decoder, tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.2,
        max_tokens: Optional[int] = None,
        repetition_context_size: int = 100,
        chunk_duration: float = 1200.0,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Parameters
        ----------
        audio:
            File path (``str`` / ``Path``) or numpy array
            (``float32``, 16 kHz, mono).
        language:
            Optional language hint.  Accepts ISO 639-1 codes (``"en"``)
            or full names (``"English"``).  Currently passed through for
            reference; the model auto-detects the language.
        temperature:
            Sampling temperature.  ``0.0`` selects the argmax (greedy).
        top_p:
            Nucleus sampling probability threshold.  ``1.0`` disables.
        top_k:
            Top-k sampling cutoff.  ``0`` disables.
        repetition_penalty:
            Multiplicative penalty applied to recently generated tokens.
        max_tokens:
            Maximum number of output tokens.  Auto-computed from audio
            duration when ``None``.
        repetition_context_size:
            Window size for repetition penalty tracking.
        chunk_duration:
            Maximum seconds per chunk.  Audio longer than this is split at
            low-energy boundaries and transcribed in pieces.

        Returns
        -------
        TranscriptionResult
            Dataclass with ``text``, ``language``, and ``duration`` fields.
        """
        with self._lock:
            return self._transcribe_impl(
                audio,
                language,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_tokens,
                repetition_context_size,
                chunk_duration,
            )

    def warm_up(self) -> None:
        """Pre-compile the MLX compute graph with a short dummy inference.

        Calling this once after loading avoids a noticeable latency spike
        on the first real transcription.
        """
        silence = np.zeros(8000, dtype=np.float32)  # 0.5 s at 16 kHz
        self.transcribe(silence)

    def close(self) -> None:
        """Release model weights and free memory."""
        self._encoder = None  # type: ignore[assignment]
        self._decoder = None  # type: ignore[assignment]
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def __enter__(self) -> "Qwen3ASR":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _transcribe_impl(
        self,
        audio: str | Path | np.ndarray,
        language: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_tokens: Optional[int],
        repetition_context_size: int,
        chunk_duration: float,
    ) -> TranscriptionResult:
        # 1. Load audio
        if isinstance(audio, (str, Path)):
            samples = load_audio(audio)
        else:
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim != 1:
                raise ValueError(
                    f"Audio array must be 1-D (mono), got shape {samples.shape}"
                )

        if len(samples) == 0:
            return TranscriptionResult(text="", language="Unknown", duration=0.0)

        duration = len(samples) / 16_000

        # 2. Auto-compute max_tokens
        if max_tokens is None:
            max_tokens = max(256, int(duration * 50))

        # 3. Chunked path for very long audio
        if duration > chunk_duration:
            return self._transcribe_chunked(
                samples,
                duration,
                language,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_tokens,
                repetition_context_size,
                chunk_duration,
            )

        # 4. Mel spectrogram
        mel = log_mel_spectrogram(samples)

        # 5. Encode
        encoder_output = self._encoder(mel)
        mx.eval(encoder_output)

        # 6. Build prompt
        n_audio_tokens = encoder_output.shape[1]
        input_ids = build_prompt(n_audio_tokens)

        # 7. Generate
        output_tokens = generate(
            self._decoder,
            encoder_output,
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        # 8. Decode and parse (token-level splitting on <asr_text>)
        text, detected_lang = self._parse_tokens(output_tokens)

        # If caller provided a language hint, prefer it when the model
        # returns "Unknown".
        if language is not None and detected_lang == "Unknown":
            detected_lang = LANGUAGE_MAP.get(language.lower(), language)

        return TranscriptionResult(text=text, language=detected_lang, duration=duration)

    def _parse_tokens(self, tokens: list[int]) -> tuple[str, str]:
        """Parse generated tokens into (text, language) at the token level.

        The model output format is:
            ``[language_token, lang_name_tokens..., <asr_text>, transcription_tokens..., <|im_end|>]``

        Splitting on the ``<asr_text>`` token ID avoids BPE decode artifacts
        that appear when decoding across the boundary.
        """
        from .tokenizer import ASR_TEXT_TOKEN_ID, EOS_TOKEN_IDS

        # Strip trailing EOS tokens
        while tokens and tokens[-1] in EOS_TOKEN_IDS:
            tokens = tokens[:-1]

        # Find <asr_text> boundary
        if ASR_TEXT_TOKEN_ID in tokens:
            idx = tokens.index(ASR_TEXT_TOKEN_ID)
            preamble_tokens = tokens[:idx]
            text_tokens = tokens[idx + 1:]
        else:
            preamble_tokens = []
            text_tokens = tokens

        # Decode transcription text
        text = self._tokenizer.decode(text_tokens, skip_special_tokens=True).strip()

        # Extract language from preamble
        preamble = self._tokenizer.decode(preamble_tokens, skip_special_tokens=True).strip()
        match = re.match(r"language\s+(.+)", preamble, re.IGNORECASE)
        language = match.group(1).strip() if match else "Unknown"

        return text, language

    def _transcribe_chunked(
        self,
        samples: np.ndarray,
        duration: float,
        language: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_tokens: int,
        repetition_context_size: int,
        chunk_duration: float,
    ) -> TranscriptionResult:
        """Transcribe audio longer than *chunk_duration* in segments.

        The audio is split at low-energy boundaries found near each
        chunk boundary.  Results are concatenated with spaces.
        """
        chunk_samples = int(chunk_duration * 16_000)
        search_samples = int(5.0 * 16_000)  # ±5 s search window

        split_points = _find_split_points(samples, chunk_samples, search_samples)

        texts: list[str] = []
        detected_lang = "Unknown"

        prev = 0
        for sp in split_points + [len(samples)]:
            chunk = samples[prev:sp]
            if len(chunk) == 0:
                prev = sp
                continue

            chunk_dur = len(chunk) / 16_000
            chunk_max_tokens = max(256, int(chunk_dur * 50))

            mel = log_mel_spectrogram(chunk)
            encoder_output = self._encoder(mel)
            mx.eval(encoder_output)

            n_audio_tokens = encoder_output.shape[1]
            input_ids = build_prompt(n_audio_tokens)

            output_tokens = generate(
                self._decoder,
                encoder_output,
                input_ids,
                max_tokens=chunk_max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )

            chunk_text, chunk_lang = self._parse_tokens(output_tokens)
            if chunk_text:
                texts.append(chunk_text)

            if detected_lang == "Unknown":
                detected_lang = chunk_lang

            prev = sp

        if language is not None and detected_lang == "Unknown":
            detected_lang = LANGUAGE_MAP.get(language.lower(), language)

        return TranscriptionResult(
            text=" ".join(texts),
            language=detected_lang,
            duration=duration,
        )


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _find_split_points(
    samples: np.ndarray,
    chunk_samples: int,
    search_samples: int,
    frame_samples: int = 480,  # ~30 ms at 16 kHz
) -> list[int]:
    """Find sample-level split points for long audio.

    For each chunk boundary (multiples of *chunk_samples*), search within
    a window of ±*search_samples* for the lowest-RMS frame and snap to it.

    Parameters
    ----------
    samples:
        1-D float32 audio at 16 kHz.
    chunk_samples:
        Target chunk length in samples.
    search_samples:
        Half-width of the search window around each boundary.
    frame_samples:
        Frame size for RMS energy computation.

    Returns
    -------
    list[int]
        Sorted list of split positions (sample indices).
    """
    total = len(samples)
    split_points: list[int] = []

    # Compute per-frame RMS energy
    n_frames = total // frame_samples
    if n_frames == 0:
        return []

    energy = np.array(
        [
            np.sqrt(np.mean(samples[i * frame_samples: (i + 1) * frame_samples] ** 2))
            for i in range(n_frames)
        ],
        dtype=np.float32,
    )

    boundary = chunk_samples
    while boundary < total:
        # Convert sample boundary to frame index
        center_frame = boundary // frame_samples

        lo_frame = max(0, center_frame - search_samples // frame_samples)
        hi_frame = min(n_frames - 1, center_frame + search_samples // frame_samples)

        if lo_frame >= hi_frame:
            split_points.append(boundary)
        else:
            best_frame = int(np.argmin(energy[lo_frame: hi_frame + 1])) + lo_frame
            split_points.append(best_frame * frame_samples)

        boundary += chunk_samples

    return split_points
