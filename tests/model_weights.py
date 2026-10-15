# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Soroush Yousefpour

"""Locate locally cached model weights for the tests that need them."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_REPO = "mlx-community/Qwen3-ASR-1.7B-bf16"


def _find_model_path() -> Path | None:
    """Return a local model directory with weights, or ``None`` when there is none.

    ``QWEN3_ASR_MODEL_PATH`` wins when set; otherwise the HuggingFace cache is
    searched without touching the network.
    """
    override = os.environ.get("QWEN3_ASR_MODEL_PATH")
    if override:
        path = Path(override).expanduser()
    else:
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo_id=MODEL_REPO, local_files_only=True))
        except Exception:
            return None
    # A partial snapshot (for example tokenizer files only) does not count
    return path if (path / "model.safetensors").is_file() else None


MODEL_PATH = _find_model_path()

requires_weights = pytest.mark.skipif(
    MODEL_PATH is None,
    reason=f"{MODEL_REPO} weights not cached; set QWEN3_ASR_MODEL_PATH or download them first",
)
