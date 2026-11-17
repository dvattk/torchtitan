# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Looped MoE recipes (torchtitan/experiments/looped_moe)."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.looped_moe import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.qwen3 import (
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
    qwen3_moe_debug,
)


def looped_moe_debug(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    attn_backend: str = "varlen",
) -> Trainer.Config:
    """Qwen3 MoE debug recipe (Qwen3 test tokenizer, c4_test, 10 steps) with
    the looped model swapped in."""
    config = qwen3_moe_debug(
        seq_len=seq_len, parallelism=parallelism, attn_backend=attn_backend
    )
    config.model = build_model_config(
        "debugmodel",
        parallelism=config.parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
    )
    return config
