# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""BPE tokenizer wrapper for Qwen3-ASR."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Special token IDs (Qwen3-ASR)
# ---------------------------------------------------------------------------

AUDIO_START_TOKEN_ID: int = 151669
AUDIO_END_TOKEN_ID: int = 151670
AUDIO_PAD_TOKEN_ID: int = 151676
IM_START_TOKEN_ID: int = 151644
IM_END_TOKEN_ID: int = 151645
ENDOFTEXT_TOKEN_ID: int = 151643
ASR_TEXT_TOKEN_ID: int = 151704

EOS_TOKEN_IDS: frozenset[int] = frozenset({ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID})

# Prompt prefix and suffix token IDs (Qwen3-ASR chat template)
_PROMPT_PREFIX: list[int] = [
    IM_START_TOKEN_ID,  # <|im_start|>
    8948,               # "system"
    198,                # "\n"
    IM_END_TOKEN_ID,    # <|im_end|>
    198,                # "\n"
    IM_START_TOKEN_ID,  # <|im_start|>
    872,                # "user"
    198,                # "\n"
    AUDIO_START_TOKEN_ID,  # <|audio_start|>
]

_PROMPT_SUFFIX: list[int] = [
    AUDIO_END_TOKEN_ID,  # <|audio_end|>
    IM_END_TOKEN_ID,     # <|im_end|>
    198,                 # "\n"
    IM_START_TOKEN_ID,   # <|im_start|>
    77091,               # "assistant"
    198,                 # "\n"
]

# Token ID for "language" (single BPE token in Qwen3-ASR vocabulary)
_LANGUAGE_TOKEN_ID: int = 11528


# ---------------------------------------------------------------------------
# Public helpers (usable without a Tokenizer instance)
# ---------------------------------------------------------------------------

def build_prompt(
    n_audio_tokens: int,
    language_name_tokens: list[int] | None = None,
) -> list[int]:
    """Return the full input_ids for a Qwen3-ASR inference prompt.

    The Qwen3-ASR chat template requires the assistant turn to begin with
    ``language {name}<asr_text>`` before the model generates transcription
    tokens.  If *language_name_tokens* is ``None``, the caller must provide
    pre-encoded tokens for the language name via ``Tokenizer.encode()``.

    Structure::

        <|im_start|>system\\n<|im_end|>\\n
        <|im_start|>user\\n
        <|audio_start|><|audio_pad|>×N<|audio_end|>
        <|im_end|>\\n
        <|im_start|>assistant\\n
        language {name}<asr_text>
    """
    if language_name_tokens is None:
        language_name_tokens = []

    return (
        _PROMPT_PREFIX
        + [AUDIO_PAD_TOKEN_ID] * n_audio_tokens
        + _PROMPT_SUFFIX
        + [_LANGUAGE_TOKEN_ID]
        + language_name_tokens
        + [ASR_TEXT_TOKEN_ID]
    )


def parse_output(text: str) -> str:
    """Extract the transcription from a Qwen3-ASR model output string.

    Expected format:
        "language English<asr_text>transcription text.<|im_end|>"

    Returns the transcription text, stripped of leading/trailing whitespace.
    """
    # Strip EOS markers
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")

    # Find <asr_text> boundary (may be literal tag or bare text after it)
    asr_tag = "<asr_text>"
    idx = text.find(asr_tag)
    if idx != -1:
        return text[idx + len(asr_tag) :].strip()

    # Fallback: strip "language <lang>" preamble if present
    stripped = re.sub(r"^language\s+\S+\s*", "", text, flags=re.IGNORECASE)
    return stripped.strip()


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------

class Tokenizer:
    """Thin wrapper around HuggingFace tokenizers for Qwen3-ASR.

    Parameters
    ----------
    model_path:
        Path to a directory containing ``vocab.json`` and ``merges.txt``.
    """

    def __init__(self, model_path: str | Path) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        model_path = Path(model_path)
        tokenizer_file = model_path / "tokenizer.json"
        vocab_file = model_path / "vocab.json"
        merges_file = model_path / "merges.txt"

        if tokenizer_file.exists():
            self._tok = HFTokenizer.from_file(str(tokenizer_file))
        elif vocab_file.exists() and merges_file.exists():
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder

            vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
            merges_lines = merges_file.read_text(encoding="utf-8").splitlines()
            # Drop header line if present
            if merges_lines and merges_lines[0].startswith("#"):
                merges_lines = merges_lines[1:]
            merges = [tuple(line.split()) for line in merges_lines if line.strip()]

            self._tok = HFTokenizer(BPE(vocab=vocab, merges=merges))  # type: ignore[arg-type]
            self._tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self._tok.decoder = ByteLevelDecoder()
        else:
            raise FileNotFoundError(
                f"No tokenizer.json or vocab.json+merges.txt found in {model_path}"
            )

    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token IDs."""
        return self._tok.encode(text).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # Convenience wrappers that delegate to the module-level functions so that
    # callers do not need to import them separately.

    def build_prompt(self, n_audio_tokens: int, language: str = "English") -> list[int]:
        """Return the full prompt input_ids with language baked in."""
        lang_tokens = self.encode(f" {language}")
        return build_prompt(n_audio_tokens, lang_tokens)

    def parse_output(self, text: str) -> str:
        """Extract transcription from model output (delegates to module-level helper)."""
        return parse_output(text)
