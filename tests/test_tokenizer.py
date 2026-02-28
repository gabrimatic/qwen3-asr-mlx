# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Tests for the Qwen3-ASR tokenizer helpers."""

import pytest

from qwen3_asr_mlx.tokenizer import (
    ASR_TEXT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    EOS_TOKEN_IDS,
    IM_END_TOKEN_ID,
    IM_START_TOKEN_ID,
    _PROMPT_PREFIX,
    _PROMPT_SUFFIX,
    build_prompt,
    parse_output,
)


class TestBuildPrompt:
    def test_zero_audio_tokens(self):
        # With no language tokens, prompt ends with suffix + language_token + asr_text_token
        ids = build_prompt(0)
        assert ids == _PROMPT_PREFIX + _PROMPT_SUFFIX + [11528, ASR_TEXT_TOKEN_ID]

    def test_audio_tokens_present(self):
        n = 10
        ids = build_prompt(n)
        audio_section = ids[len(_PROMPT_PREFIX) : len(_PROMPT_PREFIX) + n]
        assert audio_section == [AUDIO_PAD_TOKEN_ID] * n

    def test_prefix_tokens(self):
        ids = build_prompt(5)
        assert ids[: len(_PROMPT_PREFIX)] == _PROMPT_PREFIX

    def test_suffix_tokens(self):
        # _PROMPT_SUFFIX appears immediately after the audio tokens; the prompt
        # continues with language prefix tokens before <asr_text>.
        n = 5
        ids = build_prompt(n)
        suffix_start = len(_PROMPT_PREFIX) + n
        suffix_section = ids[suffix_start : suffix_start + len(_PROMPT_SUFFIX)]
        assert suffix_section == _PROMPT_SUFFIX

    def test_exact_prefix_values(self):
        """Verify the exact hardcoded prefix matches the Qwen3-ASR spec."""
        expected_prefix = [
            151644,  # <|im_start|>
            8948,    # system
            198,     # \n
            151645,  # <|im_end|>
            198,     # \n
            151644,  # <|im_start|>
            872,     # user
            198,     # \n
            151669,  # <|audio_start|>
        ]
        assert _PROMPT_PREFIX == expected_prefix

    def test_exact_suffix_values(self):
        """Verify the exact hardcoded suffix matches the Qwen3-ASR spec."""
        expected_suffix = [
            151670,  # <|audio_end|>
            151645,  # <|im_end|>
            198,     # \n
            151644,  # <|im_start|>
            77091,   # assistant
            198,     # \n
        ]
        assert _PROMPT_SUFFIX == expected_suffix

    def test_total_length(self):
        # build_prompt with no language_name_tokens appends: language_token + asr_text_token (2 tokens)
        n = 7
        ids = build_prompt(n)
        assert len(ids) == len(_PROMPT_PREFIX) + n + len(_PROMPT_SUFFIX) + 2

    def test_ends_with_asr_text_token(self):
        ids = build_prompt(5)
        assert ids[-1] == ASR_TEXT_TOKEN_ID

    def test_large_audio_token_count(self):
        n = 1500
        ids = build_prompt(n)
        assert ids.count(AUDIO_PAD_TOKEN_ID) == n


class TestParseOutput:
    def test_basic_extraction(self):
        text = "language English<asr_text>Hello world.<|im_end|>"
        assert parse_output(text) == "Hello world."

    def test_strips_im_end(self):
        text = "language English<asr_text>Test.<|im_end|>"
        result = parse_output(text)
        assert "<|im_end|>" not in result

    def test_strips_endoftext(self):
        text = "language English<asr_text>Test.<|endoftext|>"
        result = parse_output(text)
        assert "<|endoftext|>" not in result

    def test_no_asr_tag_fallback(self):
        text = "language English some transcription"
        result = parse_output(text)
        assert "language" not in result.lower()

    def test_empty_transcription(self):
        text = "language English<asr_text><|im_end|>"
        result = parse_output(text)
        assert result == ""

    def test_multiword_transcription(self):
        text = "language English<asr_text>The quick brown fox.<|im_end|>"
        assert parse_output(text) == "The quick brown fox."


class TestSpecialTokenIds:
    def test_audio_start(self):
        assert AUDIO_START_TOKEN_ID == 151669

    def test_audio_end(self):
        assert AUDIO_END_TOKEN_ID == 151670

    def test_audio_pad(self):
        assert AUDIO_PAD_TOKEN_ID == 151676

    def test_im_start(self):
        assert IM_START_TOKEN_ID == 151644

    def test_im_end(self):
        assert IM_END_TOKEN_ID == 151645

    def test_eos_tokens(self):
        assert 151643 in EOS_TOKEN_IDS  # endoftext
        assert 151645 in EOS_TOKEN_IDS  # im_end
