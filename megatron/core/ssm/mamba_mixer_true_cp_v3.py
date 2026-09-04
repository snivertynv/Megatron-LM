# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mamba mixer selecting the additive two-collective True CP v3 prototype."""

import torch

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.ssm.mamba_true_context_parallel_v3 import (
    mamba_true_context_parallel_v3_training,
)


class MambaMixerTrueCPV3(MambaMixer):
    """Route true-context execution through CP v3 without changing production v2."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.true_cp_chunk_size = getattr(config, "mamba_true_cp_chunk_size", 256)
        self.mamba_true_cp_bwd_subsequence_size = getattr(
            config, "mamba_true_cp_bwd_subsequence_size", 64
        )
        if self.pg_collection.cp.size() <= 1:
            raise ValueError("Mamba True CP v3 requires context_parallel_size > 1")
        if self.d_conv != 4:
            raise ValueError("Mamba True CP v3 currently requires d_conv == 4")

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        """Execute the training-only True CP path without CP-to-head all-to-all."""
        if not self.training or inference_context is not None or inference_params is not None:
            raise ValueError("Mamba True CP v3 currently supports training only")
        if packed_seq_params is not None:
            raise ValueError("Mamba True CP v3 does not yet support packed sequences")
        if not self.use_mem_eff_path:
            raise ValueError("Mamba True CP v3 requires use_mamba_mem_eff_path=True")

        zxBCdt, _ = self.in_proj(hidden_states)
        y = self._ssm_training_true_context(zxBCdt)
        return self.out_proj(y)

    def _ssm_training_true_context(self, zxBCdt: torch.Tensor) -> torch.Tensor:
        y = mamba_true_context_parallel_v3_training(
            zxBCdt,
            self.conv1d_weight,
            self.conv1d_bias,
            self.dt_bias,
            self.A_log,
            self.D,
            cp_group=self.pg_collection.cp,
            d_inner_local_tp=self.d_inner_local_tp,
            ngroups_local_tp=self.ngroups_local_tp,
            d_state=self.d_state,
            headdim=self.headdim,
            chunk_size=self.true_cp_chunk_size,
            activation=self.activation,
            D_has_hdim=self.D_has_hdim,
            bwd_subsequence_size=self.mamba_true_cp_bwd_subsequence_size,
        )

        if self.rmsnorm:
            y = self.norm(y)
        return y
