# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model spec selecting experimental True CP v3 without changing production."""

from copy import deepcopy

from megatron.core.models.mamba.mamba_layer_specs import (
    mamba_stack_spec as _production_spec,
)
from megatron.core.ssm.mamba_mixer_true_cp_v3 import MambaMixerTrueCPV3


mamba_stack_spec = deepcopy(_production_spec)
mamba_stack_spec.submodules.mamba_layer.submodules.mixer.module = MambaMixerTrueCPV3
