# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Autoregressive generation for Qwen3-ASR."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .decoder import KVCache, TextDecoder
from .tokenizer import AUDIO_PAD_TOKEN_ID, EOS_TOKEN_IDS

# ---------------------------------------------------------------------------
# Embedding preparation
# ---------------------------------------------------------------------------

def prepare_inputs(
    encoder_output: mx.array,
    input_ids: list[int],
    embed_tokens: nn.Embedding,
    audio_pad_id: int = AUDIO_PAD_TOKEN_ID,
) -> mx.array:
    """Replace audio-pad token embeddings with encoder output features.

    Parameters
    ----------
    encoder_output:
        Shape ``(1, n_audio_tokens, hidden_size)`` — output from the audio
        encoder.  The number of audio tokens must equal the number of
        ``audio_pad_id`` occurrences in *input_ids*.
    input_ids:
        Full prompt token ID sequence including audio-pad placeholders.
    embed_tokens:
        The decoder's ``nn.Embedding`` layer used to look up text tokens.
    audio_pad_id:
        Token ID used as the audio-pad placeholder (default 151676).

    Returns
    -------
    mx.array
        Combined embeddings of shape ``(1, len(input_ids), hidden_size)``.
    """
    ids_arr = mx.array(input_ids)
    # Embed all tokens first — audio-pad positions will be overwritten.
    embeddings = embed_tokens(ids_arr[None])  # (1, T, hidden_size)

    # Locate audio-pad positions
    audio_positions = [i for i, tok in enumerate(input_ids) if tok == audio_pad_id]
    n_audio = len(audio_positions)
    n_encoder = encoder_output.shape[1]

    if n_audio == 0:
        return embeddings

    if n_audio != n_encoder:
        raise ValueError(
            f"Number of audio-pad tokens ({n_audio}) does not match "
            f"encoder output length ({n_encoder})."
        )

    T = embeddings.shape[1]
    H = embeddings.shape[2]

    # Build a full-length audio tensor by placing encoder features at the
    # correct sequence positions, then select with a boolean mask.
    audio_full = mx.zeros((1, T, H), dtype=embeddings.dtype)
    for i, pos in enumerate(audio_positions):
        audio_full = audio_full.at[0:1, pos : pos + 1, :].add(
            encoder_output[0:1, i : i + 1, :]
        )

    # Boolean mask: True at audio-pad positions
    audio_mask_1d = mx.array(
        [pos in set(audio_positions) for pos in range(T)], dtype=mx.bool_
    )
    audio_mask = audio_mask_1d[None, :, None]  # (1, T, 1) → broadcasts over H

    return mx.where(audio_mask, audio_full, embeddings)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: mx.array,
    recent_tokens: list[int],
    penalty: float,
) -> mx.array:
    """Apply a multiplicative repetition penalty to *logits* in-place style.

    Tokens that appear in *recent_tokens* have their logit divided (if positive)
    or multiplied (if negative) by *penalty*, making repeated tokens less likely.

    Parameters
    ----------
    logits:
        Shape ``(vocab_size,)`` — raw logits for the next token.
    recent_tokens:
        List of recently generated token IDs to penalise.
    penalty:
        Penalty factor > 1.0.  Values close to 1.0 have little effect.

    Returns
    -------
    mx.array
        Modified logits of the same shape.
    """
    if penalty == 1.0 or not recent_tokens:
        return logits

    unique_tokens = sorted(set(recent_tokens))
    V = logits.shape[0]

    # Build a mask of shape (V,): True at positions to penalise
    mask_np = [False] * V
    for t in unique_tokens:
        if 0 <= t < V:
            mask_np[t] = True
    mask = mx.array(mask_np, dtype=mx.bool_)

    penalised = mx.where(
        logits > 0,
        logits / penalty,
        logits * penalty,
    )

    return mx.where(mask, penalised, logits)


def _top_k_filter(logits: mx.array, top_k: int) -> mx.array:
    """Zero out logits outside the top-k values (additive -inf mask)."""
    if top_k <= 0:
        return logits
    # sort descending, find the k-th value threshold
    sorted_logits = mx.sort(logits, axis=-1)[::-1]
    threshold = sorted_logits[top_k - 1]
    return mx.where(logits >= threshold, logits, mx.full(logits.shape, float("-inf")))


def _top_p_filter(logits: mx.array, top_p: float) -> mx.array:
    """Zero out logits outside the nucleus (cumulative probability > top_p)."""
    if top_p >= 1.0:
        return logits

    probs = mx.softmax(logits, axis=-1)
    # Sort probabilities descending
    sorted_indices = mx.argsort(-probs)
    sorted_probs   = probs[sorted_indices]
    cumsum         = mx.cumsum(sorted_probs, axis=-1)

    # Keep tokens where cumulative prob (shifted by one) is < top_p
    # i.e. remove the smallest tokens that push cumsum over top_p
    remove_mask = (cumsum - sorted_probs) >= top_p
    sorted_logits = logits[sorted_indices]
    sorted_logits = mx.where(remove_mask, mx.full(sorted_logits.shape, float("-inf")), sorted_logits)

    # Unsort
    unsort_indices = mx.argsort(sorted_indices)
    return sorted_logits[unsort_indices]


def sample(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    recent_tokens: Optional[list[int]] = None,
) -> int:
    """Sample a token ID from a logit distribution.

    Parameters
    ----------
    logits:
        Shape ``(1, vocab_size)`` or ``(vocab_size,)`` — unnormalised logits.
    temperature:
        Sampling temperature.  0.0 selects the argmax (greedy).
    top_p:
        Nucleus sampling probability threshold.  1.0 disables.
    top_k:
        Top-k sampling.  0 disables.
    repetition_penalty:
        Multiplicative penalty for recently seen tokens.  1.0 disables.
    recent_tokens:
        List of recent token IDs for repetition penalty.

    Returns
    -------
    int
        The sampled token ID.
    """
    # Flatten to (vocab_size,)
    logits = logits.reshape(-1)

    # Repetition penalty
    if repetition_penalty != 1.0 and recent_tokens:
        logits = _apply_repetition_penalty(logits, recent_tokens, repetition_penalty)

    if temperature == 0.0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        logits = _top_k_filter(logits, top_k)

    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)

    # mx.random.categorical takes unnormalized logits directly
    return int(mx.random.categorical(logits[None]).item())


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate(
    decoder: TextDecoder,
    encoder_output: mx.array,
    input_ids: list[int],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 100,
) -> list[int]:
    """Autoregressive transcription generation.

    Runs a prefill pass over the full prompt (with audio embeddings injected),
    then decodes token by token until an EOS token is produced or *max_tokens*
    is reached.

    Parameters
    ----------
    decoder:
        A loaded ``TextDecoder`` instance.
    encoder_output:
        Audio encoder output of shape ``(1, n_audio_tokens, hidden_size)``.
    input_ids:
        Full prompt token ID list (from ``build_prompt``).
    max_tokens:
        Maximum number of new tokens to generate.
    temperature:
        Sampling temperature (0.0 = greedy).
    top_p:
        Nucleus sampling threshold.
    top_k:
        Top-k sampling cutoff.
    repetition_penalty:
        Multiplicative penalty for recently generated tokens.
    repetition_context_size:
        Window of recent tokens considered for repetition penalty.

    Returns
    -------
    list[int]
        List of generated token IDs (not including the prompt).
    """
    # 1. Prepare embeddings: inject encoder features at audio-pad positions
    embeddings = prepare_inputs(encoder_output, input_ids, decoder.embed_tokens)

    # 2. Prefill: process the full prompt in one forward pass
    cache = KVCache()
    seq_len = embeddings.shape[1]
    logits = decoder(embeddings, cache=cache, is_embeds=True)
    mx.eval(logits)

    # Update cache offset after prefill
    cache.offset = seq_len

    # 3. Sample first token from the last prefill position
    next_token = sample(
        logits[:, -1, :],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        recent_tokens=None,
    )
    output_tokens: list[int] = [next_token]

    # 4. Autoregressive decode loop
    for _ in range(max_tokens - 1):
        if next_token in EOS_TOKEN_IDS:
            break

        # Embed the new token: (1, 1, hidden_size)
        token_embed = decoder.embed_tokens(mx.array([[next_token]]))
        logits = decoder(token_embed, cache=cache, is_embeds=True)
        mx.eval(logits)

        cache.offset += 1

        recent = (
            output_tokens[-repetition_context_size:]
            if repetition_context_size > 0
            else None
        )
        next_token = sample(
            logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            recent_tokens=recent,
        )
        output_tokens.append(next_token)

    return output_tokens
