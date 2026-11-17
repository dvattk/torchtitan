# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import Attention, AttentionMetadataMap
from torchtitan.models.common.flops import active_parameter_flops_per_token
from torchtitan.models.qwen3 import Qwen3Model


class LoopedMoEModel(Qwen3Model):
    """Qwen3-MoE decoder whose whole layer stack runs ``num_loops`` times.

    tokens -> tok_embeddings -> (layers[0] -> ... -> layers[N-1]) x num_loops
           -> norm -> lm_head

    Every loop reuses the same layer weights, so the parameter count (and the
    checkpoint layout) matches the plain Qwen3 model with the same flavor.
    """

    # Pipeline stages split ``layers`` across ranks, which breaks the loop
    # over the full stack.
    supports_pipeline_parallel = False

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        num_loops: int
        """Number of times the full layer stack is applied per forward."""

        def __post_init__(self) -> None:
            # slots=True dataclasses break zero-argument super().
            Qwen3Model.Config.__post_init__(self)
            if self.num_loops < 1:
                raise ValueError(f"num_loops must be >= 1, got {self.num_loops}.")

        def build_flops_estimator(
            self,
            model: Module,
            *,
            seq_len: int,
            excluded_submodules: tuple[Module | None, ...] = (),
        ) -> FlopsEstimator:
            """Count the layer stack (parameters and attention) once per loop."""
            all_parameter_flops = active_parameter_flops_per_token(
                model, excluded_submodules=excluded_submodules
            )
            non_layer_parameter_flops = active_parameter_flops_per_token(
                model, excluded_submodules=(*excluded_submodules, model.layers)
            )
            layer_parameter_flops = all_parameter_flops - non_layer_parameter_flops
            attention_flops = sum(
                attention.flops_per_token(seq_len)
                for _, attention, _, _ in self.traverse(Attention.Config)
            )
            flops_per_token = non_layer_parameter_flops + self.num_loops * (
                layer_parameter_flops + attention_flops
            )
            return lambda input_dict: flops_per_token * input_dict["input"].numel()

    def forward_embeddings(
        self,
        h: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        """``Decoder.forward_embeddings`` with the layer stack run ``num_loops`` times."""
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for _ in range(self.config.num_loops):
            for layer in self.layers.values():
                layer_attention_metadata = (
                    None
                    if attention_metadata is None
                    else attention_metadata.get(type(layer.attention.inner_attention))
                )
                h = layer(
                    h,
                    layer_attention_metadata,
                    positions,
                    padding_mask=padding_mask,
                    aux_loss_denominator=aux_loss_denominator,
                )

        h = self._pre_lm_head_norm_and_gather(h)

        # See Decoder.forward_embeddings for why this is an attribute.
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
