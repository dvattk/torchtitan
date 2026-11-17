# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Looped MoE flavors: a Qwen3-MoE flavor plus a loop count."""

from dataclasses import fields

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models import qwen3

from .model import LoopedMoEModel

__all__ = ["MODEL_FLAVORS", "build_model_config"]


# flavor -> (Qwen3 flavor that defines the layer stack, num_loops)
MODEL_FLAVORS: dict[str, tuple[str, int]] = {
    # 8 unique MoE layers (64 experts, top-8) x 4 loops = 32 effective layers.
    "debugmodel": ("debugmodel_moe", 4),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
) -> LoopedMoEModel.Config:
    qwen3_flavor, num_loops = MODEL_FLAVORS[flavor]
    qwen3_config = qwen3.build_model_config(
        qwen3_flavor,
        parallelism=parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
    )
    return LoopedMoEModel.Config(
        **{f.name: getattr(qwen3_config, f.name) for f in fields(qwen3_config)},
        num_loops=num_loops,
    )
