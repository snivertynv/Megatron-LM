# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Simplified true context-parallel training path for Mamba-2.

This module wraps the distributed forward and state-free backward in a custom
autograd function and targets the width-4 Triton fast path used by the
large_16k parity case. Backward recomputes convolution and scan intermediates
rather than retaining pre-gate output or full recurrence-state tensors.
"""

import os
from typing import Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
from torch.autograd.profiler import record_function

from megatron.core.ssm.mamba2_utils.mamba2_bwd import (
    compose_subsequence_summaries_triton,
    mamba2_bwd_decay_map_cp_triton,
    mamba2_bwd_dk_dv_cp_triton,
    mamba2_bwd_dq_cp_triton,
    mamba2_get_initial_dstate_triton,
    mamba2_get_subsequence_dstates_triton,
    mamba2_get_subsequence_states_triton,
)
from megatron.core.ssm.mamba2_utils.mamba2_fwd import (
    compute_dacs_compact_triton,
    mamba2_fwd_triton,
    mamba2_get_final_state_triton,
)
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

_SEGMENT_INDEX_CACHE = {}
_ENABLE_RECORD_FUNCTIONS = os.environ.get("MAMBA_TRUE_CP_RECORD_FUNCTIONS", "0") == "1"


class _NoOpRecordFunction:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_NOOP_RECORD_FUNCTION = _NoOpRecordFunction()


def _record_function(name: str):
    return record_function(name) if _ENABLE_RECORD_FUNCTIONS else _NOOP_RECORD_FUNCTION


# =============================================================================
# Width-4 Triton causal-convolution kernels
# =============================================================================


@triton.jit
def _causal_conv_format_streams_kernel(
    xbc_ptr,
    halos_ptr,
    weight_ptr,
    bias_ptr,
    x_ptr,
    b_ptr,
    c_ptr,
    local_seqlen: tl.constexpr,
    batch: tl.constexpr,
    conv_dim: tl.constexpr,
    d_inner: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    stride_xbc_s: tl.constexpr,
    stride_xbc_l: tl.constexpr,
    stride_xbc_b: tl.constexpr,
    stride_xbc_d: tl.constexpr,
    stride_halo_s: tl.constexpr,
    stride_halo_l: tl.constexpr,
    stride_halo_b: tl.constexpr,
    stride_halo_d: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_x_sb: tl.constexpr,
    stride_x_l: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_p: tl.constexpr,
    stride_b_sb: tl.constexpr,
    stride_b_l: tl.constexpr,
    stride_b_g: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_c_sb: tl.constexpr,
    stride_c_l: tl.constexpr,
    stride_c_g: tl.constexpr,
    stride_c_n: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    sb = tl.program_id(2)
    stream = sb // batch
    batch_idx = sb - stream * batch

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_offsets < local_seqlen
    d_mask = d_offsets < conv_dim

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for k in tl.static_range(0, 4):
        source_pos = t_offsets + k
        weight = tl.load(
            weight_ptr + d_offsets * stride_weight_d + k * stride_weight_w, mask=d_mask, other=0.0
        ).to(tl.float32)

        x_time = source_pos - 3
        x_time = tl.minimum(tl.maximum(x_time, 0), local_seqlen - 1)
        x_values = tl.load(
            xbc_ptr
            + stream * stride_xbc_s
            + x_time[:, None] * stride_xbc_l
            + batch_idx * stride_xbc_b
            + d_offsets[None, :] * stride_xbc_d,
            mask=(t_mask[:, None] & d_mask[None, :] & (source_pos[:, None] >= 3)),
            other=0.0,
        ).to(tl.float32)

        halo_time = tl.minimum(source_pos, 2)
        halo_values = tl.load(
            halos_ptr
            + stream * stride_halo_s
            + halo_time[:, None] * stride_halo_l
            + batch_idx * stride_halo_b
            + d_offsets[None, :] * stride_halo_d,
            mask=(t_mask[:, None] & d_mask[None, :] & (source_pos[:, None] < 3)),
            other=0.0,
        ).to(tl.float32)
        values = x_values + halo_values

        acc += values * weight[None, :]

    bias = tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    acc = acc * tl.sigmoid(acc)

    x_mask = d_offsets < d_inner
    x_head = d_offsets // headdim
    x_headdim = d_offsets - x_head * headdim
    tl.store(
        x_ptr
        + sb * stride_x_sb
        + t_offsets[:, None] * stride_x_l
        + x_head[None, :] * stride_x_h
        + x_headdim[None, :] * stride_x_p,
        acc,
        mask=t_mask[:, None] & x_mask[None, :],
    )

    b_dim = ngroups * d_state
    b_offsets = d_offsets - d_inner
    b_mask = (b_offsets >= 0) & (b_offsets < b_dim)
    b_group = b_offsets // d_state
    b_state = b_offsets - b_group * d_state
    tl.store(
        b_ptr
        + sb * stride_b_sb
        + t_offsets[:, None] * stride_b_l
        + b_group[None, :] * stride_b_g
        + b_state[None, :] * stride_b_n,
        acc,
        mask=t_mask[:, None] & b_mask[None, :],
    )

    c_offsets = d_offsets - d_inner - b_dim
    c_mask = (c_offsets >= 0) & (c_offsets < b_dim)
    c_group = c_offsets // d_state
    c_state = c_offsets - c_group * d_state
    tl.store(
        c_ptr
        + sb * stride_c_sb
        + t_offsets[:, None] * stride_c_l
        + c_group[None, :] * stride_c_g
        + c_state[None, :] * stride_c_n,
        acc,
        mask=t_mask[:, None] & c_mask[None, :],
    )


@triton.jit
def _causal_conv_backward_fused_width4_tiled_kernel(
    xbc_ptr,
    halos_ptr,
    weight_ptr,
    bias_ptr,
    dx_ptr,
    db_ptr,
    dc_ptr,
    dxbc_ptr,
    partials_ptr,
    edges_ptr,
    local_seqlen: tl.constexpr,
    batch: tl.constexpr,
    conv_dim: tl.constexpr,
    d_inner: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    num_tiles: tl.constexpr,
    stride_xbc_s: tl.constexpr,
    stride_xbc_l: tl.constexpr,
    stride_xbc_b: tl.constexpr,
    stride_xbc_d: tl.constexpr,
    stride_halo_s: tl.constexpr,
    stride_halo_l: tl.constexpr,
    stride_halo_b: tl.constexpr,
    stride_halo_d: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_dx_sb: tl.constexpr,
    stride_dx_l: tl.constexpr,
    stride_dx_h: tl.constexpr,
    stride_dx_p: tl.constexpr,
    stride_db_sb: tl.constexpr,
    stride_db_l: tl.constexpr,
    stride_db_g: tl.constexpr,
    stride_db_n: tl.constexpr,
    stride_dc_sb: tl.constexpr,
    stride_dc_l: tl.constexpr,
    stride_dc_g: tl.constexpr,
    stride_dc_n: tl.constexpr,
    stride_dxbc_s: tl.constexpr,
    stride_dxbc_l: tl.constexpr,
    stride_dxbc_b: tl.constexpr,
    stride_dxbc_d: tl.constexpr,
    stride_partials_p: tl.constexpr,
    stride_partials_d: tl.constexpr,
    stride_partials_k: tl.constexpr,
    stride_edges_p: tl.constexpr,
    stride_edges_e: tl.constexpr,
    stride_edges_d: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    sb = tl.program_id(2)
    stream = sb // batch
    batch_idx = sb - stream * batch

    tile_start = pid_t * BLOCK_T
    t_offsets = tile_start + tl.arange(0, BLOCK_T)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, 4)
    t_mask = t_offsets < local_seqlen
    d_mask = d_offsets < conv_dim
    element_mask = t_mask[:, None] & d_mask[None, :]

    x_mask = d_offsets < d_inner
    x_head = d_offsets // headdim
    x_headdim = d_offsets - x_head * headdim
    grad_output = tl.load(
        dx_ptr
        + sb * stride_dx_sb
        + t_offsets[:, None] * stride_dx_l
        + x_head[None, :] * stride_dx_h
        + x_headdim[None, :] * stride_dx_p,
        mask=t_mask[:, None] & x_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    b_dim = ngroups * d_state
    b_offsets = d_offsets - d_inner
    b_mask = (b_offsets >= 0) & (b_offsets < b_dim)
    b_group = b_offsets // d_state
    b_state = b_offsets - b_group * d_state
    grad_output += tl.load(
        db_ptr
        + sb * stride_db_sb
        + t_offsets[:, None] * stride_db_l
        + b_group[None, :] * stride_db_g
        + b_state[None, :] * stride_db_n,
        mask=t_mask[:, None] & b_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    c_offsets = d_offsets - d_inner - b_dim
    c_mask = (c_offsets >= 0) & (c_offsets < b_dim)
    c_group = c_offsets // d_state
    c_state = c_offsets - c_group * d_state
    grad_output += tl.load(
        dc_ptr
        + sb * stride_dc_sb
        + t_offsets[:, None] * stride_dc_l
        + c_group[None, :] * stride_dc_g
        + c_state[None, :] * stride_dc_n,
        mask=t_mask[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    weights = tl.load(
        weight_ptr + d_offsets[:, None] * stride_weight_d + k_offsets[None, :] * stride_weight_w,
        mask=d_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    source_pos = t_offsets[:, None, None] + k_offsets[None, None, :]
    x_time = tl.minimum(tl.maximum(source_pos - 3, 0), local_seqlen - 1)
    values = tl.load(
        xbc_ptr
        + stream * stride_xbc_s
        + x_time * stride_xbc_l
        + batch_idx * stride_xbc_b
        + d_offsets[None, :, None] * stride_xbc_d,
        mask=t_mask[:, None, None] & d_mask[None, :, None] & (source_pos >= 3),
        other=0.0,
    ).to(tl.float32)
    halo_time = tl.minimum(source_pos, 2)
    values += tl.load(
        halos_ptr
        + stream * stride_halo_s
        + halo_time * stride_halo_l
        + batch_idx * stride_halo_b
        + d_offsets[None, :, None] * stride_halo_d,
        mask=t_mask[:, None, None] & d_mask[None, :, None] & (source_pos < 3),
        other=0.0,
    ).to(tl.float32)
    preact = tl.sum(values * weights[None, :, :], axis=2)
    bias = tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    preact += bias[None, :]
    dpreact = grad_output
    sigmoid = tl.sigmoid(preact)
    dpreact *= sigmoid * (1.0 + preact * (1.0 - sigmoid))
    dpreact = tl.where(element_mask, dpreact, 0.0)

    partial = sb * num_tiles + pid_t
    partial_dweight = tl.sum(values * dpreact[:, :, None], axis=0)
    tl.store(
        partials_ptr
        + partial * stride_partials_p
        + d_offsets[:, None] * stride_partials_d
        + k_offsets[None, :] * stride_partials_k,
        partial_dweight,
        mask=d_mask[:, None],
    )
    partial_dbias = tl.sum(dpreact, axis=0)
    tl.store(
        partials_ptr
        + partial * stride_partials_p
        + d_offsets * stride_partials_d
        + 4 * stride_partials_k,
        partial_dbias,
        mask=d_mask,
    )

    edge_lanes = tl.arange(0, 4)[:, None] + 0 * d_offsets[None, :]
    first_edges = tl.gather(dpreact, edge_lanes, axis=0)
    tl.store(
        edges_ptr
        + partial * stride_edges_p
        + edge_lanes * stride_edges_e
        + d_offsets[None, :] * stride_edges_d,
        first_edges,
        mask=(edge_lanes < 3) & d_mask[None, :],
    )
    last_edge_rows = BLOCK_T - 3 + edge_lanes
    last_edges = tl.gather(dpreact, tl.minimum(last_edge_rows, BLOCK_T - 1), axis=0)
    tl.store(
        edges_ptr
        + partial * stride_edges_p
        + (3 + edge_lanes) * stride_edges_e
        + d_offsets[None, :] * stride_edges_d,
        last_edges,
        mask=(edge_lanes < 3) & d_mask[None, :],
    )

    gather_rows = tl.arange(0, BLOCK_T)[:, None] + 0 * d_offsets[None, :]
    dp_1 = tl.gather(dpreact, tl.maximum(gather_rows - 1, 0), axis=0)
    dp_2 = tl.gather(dpreact, tl.maximum(gather_rows - 2, 0), axis=0)
    dp_3 = tl.gather(dpreact, tl.maximum(gather_rows - 3, 0), axis=0)
    weight_0 = tl.load(weight_ptr + d_offsets * stride_weight_d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    weight_1 = tl.load(
        weight_ptr + d_offsets * stride_weight_d + stride_weight_w, mask=d_mask, other=0.0
    ).to(tl.float32)
    weight_2 = tl.load(
        weight_ptr + d_offsets * stride_weight_d + 2 * stride_weight_w, mask=d_mask, other=0.0
    ).to(tl.float32)
    weight_3 = tl.load(
        weight_ptr + d_offsets * stride_weight_d + 3 * stride_weight_w, mask=d_mask, other=0.0
    ).to(tl.float32)
    dx_combined = dpreact * weight_0[None, :]
    dx_combined += dp_1 * weight_1[None, :]
    dx_combined += dp_2 * weight_2[None, :]
    dx_combined += dp_3 * weight_3[None, :]

    input_mask = t_offsets < local_seqlen + 3
    local_mask = input_mask & (tl.arange(0, BLOCK_T) >= 3)
    tl.store(
        dxbc_ptr
        + stream * stride_dxbc_s
        + tl.maximum(t_offsets - 3, 0)[:, None] * stride_dxbc_l
        + batch_idx * stride_dxbc_b
        + d_offsets[None, :] * stride_dxbc_d,
        dx_combined,
        mask=local_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _causal_conv_backward_boundary_width4_kernel(
    edges_ptr,
    weight_ptr,
    dxbc_ptr,
    dhalos_ptr,
    local_seqlen: tl.constexpr,
    batch: tl.constexpr,
    conv_dim: tl.constexpr,
    num_tiles: tl.constexpr,
    stride_edges_p: tl.constexpr,
    stride_edges_e: tl.constexpr,
    stride_edges_d: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_dxbc_s: tl.constexpr,
    stride_dxbc_l: tl.constexpr,
    stride_dxbc_b: tl.constexpr,
    stride_dxbc_d: tl.constexpr,
    stride_dhalo_s: tl.constexpr,
    stride_dhalo_l: tl.constexpr,
    stride_dhalo_b: tl.constexpr,
    stride_dhalo_d: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    sb = tl.program_id(2)
    stream = sb // batch
    batch_idx = sb - stream * batch
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < conv_dim
    row_lanes = tl.arange(0, 4)[:, None] + 0 * d_offsets[None, :]
    partial = sb * num_tiles + pid_t

    current = tl.load(
        edges_ptr
        + partial * stride_edges_p
        + row_lanes * stride_edges_e
        + d_offsets[None, :] * stride_edges_d,
        mask=(row_lanes < 3) & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    previous_partial = tl.maximum(partial - 1, 0)
    previous = tl.load(
        edges_ptr
        + previous_partial * stride_edges_p
        + (3 + row_lanes) * stride_edges_e
        + d_offsets[None, :] * stride_edges_d,
        mask=(pid_t > 0) & (row_lanes < 3) & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    dp_1 = tl.gather(current, tl.maximum(row_lanes - 1, 0), axis=0)
    previous_2 = tl.gather(previous, 2 + 0 * row_lanes, axis=0)
    dp_1 = tl.where(row_lanes == 0, previous_2, dp_1)
    dp_2 = tl.gather(current, tl.maximum(row_lanes - 2, 0), axis=0)
    previous_1 = tl.gather(previous, 1 + 0 * row_lanes, axis=0)
    dp_2 = tl.where(row_lanes == 0, previous_1, dp_2)
    dp_2 = tl.where(row_lanes == 1, previous_2, dp_2)
    dp_3 = tl.gather(current, tl.maximum(row_lanes - 3, 0), axis=0)
    previous_0 = tl.gather(previous, 0 * row_lanes, axis=0)
    dp_3 = tl.where(row_lanes == 0, previous_0, dp_3)
    dp_3 = tl.where(row_lanes == 1, previous_1, dp_3)
    dp_3 = tl.where(row_lanes == 2, previous_2, dp_3)

    weight_0 = tl.load(weight_ptr + d_offsets * stride_weight_d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    weight_1 = tl.load(
        weight_ptr + d_offsets * stride_weight_d + stride_weight_w, mask=d_mask, other=0.0
    ).to(tl.float32)
    weight_2 = tl.load(
        weight_ptr + d_offsets * stride_weight_d + 2 * stride_weight_w, mask=d_mask, other=0.0
    ).to(tl.float32)
    weight_3 = tl.load(
        weight_ptr + d_offsets * stride_weight_d + 3 * stride_weight_w, mask=d_mask, other=0.0
    ).to(tl.float32)
    dx_combined = current * weight_0[None, :]
    dx_combined += dp_1 * weight_1[None, :]
    dx_combined += dp_2 * weight_2[None, :]
    dx_combined += dp_3 * weight_3[None, :]

    input_pos = pid_t * BLOCK_T + tl.arange(0, 4)
    input_mask = (tl.arange(0, 4) < 3) & (input_pos < local_seqlen + 3)
    halo_mask = input_mask & (input_pos < 3)
    tl.store(
        dhalos_ptr
        + stream * stride_dhalo_s
        + tl.minimum(input_pos, 2)[:, None] * stride_dhalo_l
        + batch_idx * stride_dhalo_b
        + d_offsets[None, :] * stride_dhalo_d,
        dx_combined,
        mask=halo_mask[:, None] & d_mask[None, :],
    )
    local_mask = input_mask & (input_pos >= 3)
    tl.store(
        dxbc_ptr
        + stream * stride_dxbc_s
        + tl.maximum(input_pos - 3, 0)[:, None] * stride_dxbc_l
        + batch_idx * stride_dxbc_b
        + d_offsets[None, :] * stride_dxbc_d,
        dx_combined,
        mask=local_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _causal_conv_backward_reduce_width4_kernel(
    partials_ptr,
    dweight_ptr,
    dbias_ptr,
    conv_dim: tl.constexpr,
    num_partials: tl.constexpr,
    stride_partials_p: tl.constexpr,
    stride_partials_d: tl.constexpr,
    stride_partials_k: tl.constexpr,
    stride_dweight_d: tl.constexpr,
    stride_dweight_w: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d_offsets = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < conv_dim
    dweight_0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dweight_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dweight_2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dweight_3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dbias = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for partial in tl.range(0, num_partials):
        partial_ptr = partials_ptr + partial * stride_partials_p + d_offsets * stride_partials_d
        dweight_0 += tl.load(partial_ptr, mask=d_mask, other=0.0).to(tl.float32)
        dweight_1 += tl.load(partial_ptr + stride_partials_k, mask=d_mask, other=0.0).to(tl.float32)
        dweight_2 += tl.load(partial_ptr + 2 * stride_partials_k, mask=d_mask, other=0.0).to(
            tl.float32
        )
        dweight_3 += tl.load(partial_ptr + 3 * stride_partials_k, mask=d_mask, other=0.0).to(
            tl.float32
        )
        dbias += tl.load(partial_ptr + 4 * stride_partials_k, mask=d_mask, other=0.0).to(tl.float32)

    dweight_ptrs = dweight_ptr + d_offsets * stride_dweight_d
    tl.store(dweight_ptrs, dweight_0, mask=d_mask)
    tl.store(dweight_ptrs + stride_dweight_w, dweight_1, mask=d_mask)
    tl.store(dweight_ptrs + 2 * stride_dweight_w, dweight_2, mask=d_mask)
    tl.store(dweight_ptrs + 3 * stride_dweight_w, dweight_3, mask=d_mask)
    tl.store(dbias_ptr + d_offsets, dbias, mask=d_mask)


# =============================================================================
# True context-parallel training entrypoint and autograd wrapper
# =============================================================================


def mamba_true_context_parallel_training(
    zxBCdt: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
    chunk_size: int,
    activation: str,
    D_has_hdim: bool,
    cache_conv_halos: Optional[bool] = None,
    cache_initial_states: bool = True,
    conv_halo_exchange: Literal["all_gather", "all_to_all"] = "all_gather",
    summary_backward_mode: Literal["specialized", "state_passing"] = "specialized",
    bwd_subsequence_size: int = 64,
) -> torch.Tensor:
    """Width-4 Triton true context-parallel Mamba-2 training forward.

    ``cache_conv_halos=None`` retains the compact predecessor halos when grad
    mode is enabled and skips them under no-grad/inference execution. Explicit
    booleans override that policy.

    By default, training retains the forward prefix states and the causally
    ordered global log decays so backward skips the whole-segment forward
    summary and global affine scan, and gathers only the additive adjoint
    summary instead of the packed ``(log_alpha, beta, gamma)`` exchange. Pass
    ``cache_initial_states=False`` to opt into backward recomputation.

    ``bwd_subsequence_size=64`` schedules the chunk-64 backward over rank-local
    64-token subsections. Set it to zero to restore the unsplit v2 backward;
    other positive values must be multiples of 64. This does not alter the CP
    collective count or payload.
    """

    if cp_group.size() == 1:
        raise ValueError("mamba_true_context_parallel_training expects context_parallel_size > 1")
    if conv_bias is None:
        raise ValueError("the simplified true-CP path requires conv_bias")
    if activation != "silu":
        raise ValueError("the simplified true-CP path requires activation='silu'")
    if D_has_hdim:
        raise ValueError("the simplified true-CP path requires D_has_hdim=False")
    if conv_halo_exchange != "all_gather":
        raise ValueError("the simplified true-CP path requires conv_halo_exchange='all_gather'")
    if summary_backward_mode != "specialized":
        raise ValueError("the simplified true-CP path requires summary_backward_mode='specialized'")
    if bwd_subsequence_size < 0 or (
        bwd_subsequence_size and bwd_subsequence_size % 64
    ):
        raise ValueError("bwd_subsequence_size must be zero or a positive multiple of 64")
    # Resolve the automatic policy outside autograd.Function.forward, which
    # PyTorch executes with grad mode disabled even during training.
    save_conv_halos = torch.is_grad_enabled() if cache_conv_halos is None else cache_conv_halos
    save_initial_states = cache_initial_states and torch.is_grad_enabled()
    return _MambaTrueContextParallelFn.apply(
        zxBCdt,
        conv_weight,
        conv_bias,
        dt_bias,
        A_log,
        D,
        cp_group,
        d_inner_local_tp,
        ngroups_local_tp,
        d_state,
        headdim,
        chunk_size,
        bwd_subsequence_size,
        save_conv_halos,
        save_initial_states,
    )


class _MambaTrueContextParallelFn(torch.autograd.Function):
    """Checkpointed true-CP wrapper around local Mamba conv and SSD kernels."""

    @staticmethod
    def forward(
        ctx,
        zxBCdt,
        conv_weight,
        conv_bias,
        dt_bias,
        A_log,
        D,
        cp_group,
        d_inner_local_tp,
        ngroups_local_tp,
        d_state,
        headdim,
        chunk_size,
        bwd_subsequence_size,
        save_conv_halos,
        save_initial_states,
    ):
        ctx.cp_group = cp_group
        ctx.d_inner_local_tp = d_inner_local_tp
        ctx.ngroups_local_tp = ngroups_local_tp
        ctx.d_state = d_state
        ctx.headdim = headdim
        ctx.chunk_size = chunk_size
        ctx.bwd_subsequence_size = bwd_subsequence_size
        ctx.save_conv_halos = save_conv_halos
        ctx.save_initial_states = save_initial_states

        _check_inputs(
            zxBCdt,
            conv_weight,
            conv_bias,
            dt_bias,
            A_log,
            D,
            cp_group=cp_group,
            d_inner_local_tp=d_inner_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
        )
        prepared = _prepare_folded_forward(
            zxBCdt,
            conv_weight,
            conv_bias,
            dt_bias,
            A_log,
            D,
            cp_group=cp_group,
            d_inner_local_tp=d_inner_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
            chunk_size=chunk_size,
            return_conv_halos=save_conv_halos,
            return_initial_states=save_initial_states,
        )

        out = prepared.pop("out")
        conv_halos = prepared.pop("xBC_conv_halos") if save_conv_halos else None
        initial_states = prepared.pop("local_initial_states") if save_initial_states else None
        global_log_alpha = prepared.pop("global_log_alpha") if save_initial_states else None
        prepared.clear()

        tensors_to_save = [zxBCdt, conv_weight, conv_bias, dt_bias, A_log, D]
        if save_conv_halos:
            tensors_to_save.append(conv_halos)
        if save_initial_states:
            tensors_to_save.extend([initial_states, global_log_alpha])
        ctx.save_for_backward(*tensors_to_save)

        return out.view(zxBCdt.shape[0], zxBCdt.shape[1], d_inner_local_tp)

    @staticmethod
    def backward(ctx, grad_output):
        zxBCdt, conv_weight, conv_bias, dt_bias, A_log, D, *rest = ctx.saved_tensors
        rest_idx = 0
        cached_conv_halos = None
        if ctx.save_conv_halos:
            cached_conv_halos = rest[rest_idx]
            rest_idx += 1
        cached_initial_states = None
        cached_global_log_alpha = None
        if ctx.save_initial_states:
            cached_initial_states = rest[rest_idx]
            cached_global_log_alpha = rest[rest_idx + 1]
            rest_idx += 2

        (grad_zxBCdt, grad_conv_weight, grad_conv_bias, grad_dt_bias, grad_A_log, grad_D) = (
            _true_context_backward_impl(
                grad_output,
                zxBCdt,
                conv_weight,
                conv_bias,
                dt_bias,
                A_log,
                D,
                cp_group=ctx.cp_group,
                d_inner_local_tp=ctx.d_inner_local_tp,
                ngroups_local_tp=ctx.ngroups_local_tp,
                d_state=ctx.d_state,
                headdim=ctx.headdim,
                chunk_size=ctx.chunk_size,
                bwd_subsequence_size=ctx.bwd_subsequence_size,
                cached_conv_halos=cached_conv_halos,
                cached_initial_states=cached_initial_states,
                cached_global_log_alpha=cached_global_log_alpha,
            )
        )

        return (
            grad_zxBCdt,
            grad_conv_weight,
            grad_conv_bias,
            grad_dt_bias,
            grad_A_log,
            grad_D,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# =============================================================================
# Backward recomputation and gradient assembly
# =============================================================================


def _subsequence_state_passing(
    summaries: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_states: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 incoming states for each rank-local subsection.

    Mamba's state-passing kernel consumes a flattened state dimension. The
    subsection summaries and boundary states keep P and N contiguous, so the
    input flattening and output restoration below are views rather than
    materializations. The returned final state is not needed because the CP
    boundary helpers already supplied the adjacent segment's boundary.
    """

    folded_batch, nsubsequences, nheads, headdim, dstate = summaries.shape
    summaries_flat = summaries.view(
        folded_batch, nsubsequences, nheads, headdim * dstate
    )
    initial_states_flat = initial_states.view(
        folded_batch, nheads, headdim * dstate
    )
    incoming_flat, final_states = _state_passing_fwd(
        summaries_flat,
        log_alpha,
        initial_states=initial_states_flat,
        seq_idx=None,
        chunk_size=1,
        out_dtype=torch.float32,
    )
    del final_states
    return incoming_flat.view(
        folded_batch, nsubsequences, nheads, headdim, dstate
    )


def _true_context_backward_impl(
    grad_output: torch.Tensor,
    zxBCdt: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
    chunk_size: int,
    bwd_subsequence_size: int = 64,
    cached_conv_halos: Optional[torch.Tensor] = None,
    cached_initial_states: Optional[torch.Tensor] = None,
    cached_global_log_alpha: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the checkpointed state-free true-CP backward.

    The default saved-halo/cache path performs two collectives: one gamma-only
    recurrence-boundary exchange followed by one halo-gradient exchange after
    convolution backward. Initial-state recomputation uses one packed forward/
    reverse boundary exchange instead. Passing no cached halos is still
    correct, but requires an additional predecessor-halo exchange while
    recomputing the convolution. The default chunk-64 subsection path computes
    rank-local summaries for its CTA seeds without changing communication.
    """

    if (cached_initial_states is None) != (cached_global_log_alpha is None):
        raise ValueError(
            "cached_initial_states and cached_global_log_alpha must be provided together"
        )
    use_cached_initial_states = cached_initial_states is not None
    segment_len = zxBCdt.shape[0] // 2
    subsequence_chunks = (
        bwd_subsequence_size // chunk_size
        if chunk_size == 64 and bwd_subsequence_size > 0
        else 0
    )
    nchunks = (segment_len + chunk_size - 1) // chunk_size
    nsubsequences = (
        (nchunks + subsequence_chunks - 1) // subsequence_chunks
        if subsequence_chunks
        else 1
    )
    use_subsequence = subsequence_chunks > 0 and nsubsequences > 1

    # =============================================================================
    # No-copy gradient view and checkpointed local forward recomputation
    # =============================================================================
    batch = zxBCdt.shape[1]
    nheads_local_tp = d_inner_local_tp // headdim
    dout = _format_grad_output_streams(grad_output, headdim=headdim)

    with _record_function("mamba_true_cp_v2:bwd_recompute"):
        prepared = _prepare_folded_scan_state(
            zxBCdt,
            conv_weight,
            conv_bias,
            dt_bias,
            A_log,
            cp_group=cp_group,
            d_inner_local_tp=d_inner_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
            chunk_size=chunk_size,
            corrected_states_out_dtype=zxBCdt.dtype,
            return_backward_state=True,
            cached_conv_halos=cached_conv_halos,
            defer_boundary_exchange=True,
            skip_segment_summary=use_cached_initial_states or use_subsequence,
        )

    x = prepared.pop("x")
    B = prepared.pop("B")
    C = prepared.pop("C")
    z = prepared.pop("z")
    dA_cumsum = prepared.pop("dA_cumsum")
    dA_cs_rev = prepared.pop("dA_cs_rev")
    dt_out = prepared.pop("dt_out")
    local_beta = prepared.pop("local_beta")
    local_log_alpha = prepared.pop("local_log_alpha")
    deferred_initial_states = prepared.pop("local_initial_states")
    dt_streams = prepared.pop("dt_streams")
    xBC_streams = prepared.pop("xBC_streams")
    conv_halos = prepared.pop("xBC_conv_halos")
    prepared.clear()
    if deferred_initial_states is not None:
        raise RuntimeError("backward recomputation unexpectedly materialized deferred tensors")

    # =============================================================================
    # Compact adjoint summary and combined forward/reverse CP boundaries
    # =============================================================================
    A = -torch.exp(A_log.float())
    with _record_function("mamba_true_cp_v2:bwd_boundary_prepare"):
        subsequence_beta = None
        subsequence_gamma = None
        subsequence_log_alpha = None
        reverse_log_alpha = None
        folded_beta = None
        if use_subsequence:
            subsequence_beta, subsequence_log_alpha = (
                mamba2_get_subsequence_states_triton(
                    B,
                    x,
                    dt_out,
                    dA_cumsum,
                    dA_cs_rev,
                    chunk_size=chunk_size,
                    subsequence_size=bwd_subsequence_size,
                )
            )
            if not use_cached_initial_states:
                folded_beta = compose_subsequence_summaries_triton(
                    subsequence_beta, subsequence_log_alpha
                )
                local_beta = folded_beta.view(
                    2, batch, nheads_local_tp, headdim, d_state
                )
                local_log_alpha = subsequence_log_alpha.sum(dim=-1).view(
                    2, batch, nheads_local_tp
                )
            subsequence_gamma = mamba2_get_subsequence_dstates_triton(
                C,
                dout,
                z,
                dA_cumsum,
                chunk_size=chunk_size,
                subsequence_size=bwd_subsequence_size,
            )
            reverse_log_alpha = subsequence_log_alpha.flip(-1).contiguous()
            folded_gamma = compose_subsequence_summaries_triton(
                subsequence_gamma, reverse_log_alpha
            )
        else:
            folded_gamma = mamba2_get_initial_dstate_triton(
                C,
                dout,
                z,
                dA_cumsum,
                chunk_size=chunk_size,
            )
        local_gamma = folded_gamma.view(2, batch, nheads_local_tp, headdim, d_state)
        if use_cached_initial_states:
            local_initial_states = cached_initial_states
            local_final_dstates = _local_segment_final_dstates(
                cached_global_log_alpha,
                local_gamma,
                cp_group,
            )
        else:
            local_initial_states, local_final_dstates = _local_segment_boundary_states(
                local_log_alpha,
                local_beta,
                local_gamma,
                cp_group,
            )
        del folded_gamma, local_gamma, local_beta, local_log_alpha
        if use_subsequence:
            del folded_beta
            folded_initial_states = local_initial_states.view(
                2 * batch, nheads_local_tp, headdim, d_state
            )
            folded_final_dstates = local_final_dstates.view(
                2 * batch, nheads_local_tp, headdim, d_state
            )
            subsequence_beta = _subsequence_state_passing(
                subsequence_beta,
                subsequence_log_alpha,
                folded_initial_states,
            )
            del folded_initial_states, local_initial_states
            subsequence_gamma = _subsequence_state_passing(
                subsequence_gamma,
                reverse_log_alpha,
                folded_final_dstates,
            )
            del folded_final_dstates, local_final_dstates
    if use_subsequence:
        del (
            reverse_log_alpha,
            subsequence_log_alpha,
        )

    # =============================================================================
    # State-free SSD gradients and compact decay accumulation
    # =============================================================================
    with _record_function("mamba_true_cp_v2:bwd_dq"):
        dC, dz, chunk_start_states, processed_ddt, dA_chunk = mamba2_bwd_dq_cp_triton(
            C,
            B,
            x,
            dout,
            z,
            dt_out,
            dA_cumsum,
            dA_cs_rev,
            subsequence_beta if use_subsequence else local_initial_states,
            A,
            D,
            chunk_size=chunk_size,
            subsequence_size=bwd_subsequence_size if use_subsequence else 0,
        )
    if use_subsequence:
        del subsequence_beta
    else:
        del local_initial_states

    with _record_function("mamba_true_cp_v2:bwd_dk_dv"):
        (
            dB,
            dx,
            grad_D,
            processed_ddt,
            dA_chunk,
        ) = mamba2_bwd_dk_dv_cp_triton(
            C,
            B,
            x,
            dout,
            z,
            dt_out,
            dA_cumsum,
            dA_cs_rev,
            chunk_start_states,
            subsequence_gamma if use_subsequence else local_final_dstates,
            A,
            D,
            processed_ddt,
            dA_chunk,
            chunk_size=chunk_size,
            subsequence_size=bwd_subsequence_size if use_subsequence else 0,
        )
    if use_subsequence:
        del subsequence_gamma
    else:
        del local_final_dstates
    del (
        C,
        B,
        x,
        z,
        dout,
        dA_cumsum,
        dA_cs_rev,
        chunk_start_states,
    )

    # =============================================================================
    # Raw-dt softplus mapping and final parameter reductions
    # =============================================================================
    with _record_function("mamba_true_cp_v2:bwd_decay"):
        grad_dt_streams, grad_A_log, grad_dt_bias = mamba2_bwd_decay_map_cp_triton(
            dt_streams,
            processed_ddt,
            dA_chunk,
            A,
            dt_bias,
            chunk_size=chunk_size,
        )
    del (
        A,
        dt_streams,
        dt_out,
        processed_ddt,
        dA_chunk,
    )

    # =============================================================================
    # Fused convolution backward and cross-rank halo-gradient routing
    # =============================================================================
    with _record_function("mamba_true_cp_v2:bwd_conv"):
        grad_xBC_streams, grad_halos, grad_conv_weight, grad_conv_bias = (
            _causal_conv_format_streams_backward_triton(
                xBC_streams,
                conv_halos,
                conv_weight,
                conv_bias,
                dx,
                dB,
                dC,
                d_inner_local_tp=d_inner_local_tp,
                ngroups_local_tp=ngroups_local_tp,
                d_state=d_state,
                headdim=headdim,
            )
        )
    del xBC_streams, conv_halos, dx, dB, dC
    grad_xBC_streams = _route_conv_halo_grads(
        grad_xBC_streams, grad_halos, cp_group
    )
    del grad_halos

    # =============================================================================
    # Projected-input gradient assembly
    # =============================================================================
    grad_streams = torch.cat(
        (dz.flatten(-2), grad_xBC_streams, grad_dt_streams), dim=-1
    )
    grad_zxBCdt = grad_streams.view_as(zxBCdt)
    return (
        grad_zxBCdt,
        grad_conv_weight,
        grad_conv_bias,
        grad_dt_bias,
        grad_A_log,
        grad_D,
    )


# =============================================================================
# Fixed fast-path input validation
# =============================================================================


def _check_inputs(
    zxBCdt: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
) -> None:
    cp_size = cp_group.size()
    assert cp_size > 1, "true context-parallel Mamba requires context_parallel_size > 1"
    assert zxBCdt.dim() == 3, "zxBCdt must have shape [local_seq, batch, hidden]"
    assert zxBCdt.shape[0] % 2 == 0, "non-packed CP input must contain two local segments"
    nheads_local_tp = d_inner_local_tp // headdim
    projected_dim = 2 * d_inner_local_tp + 2 * ngroups_local_tp * d_state + nheads_local_tp
    conv_dim = d_inner_local_tp + 2 * ngroups_local_tp * d_state

    assert d_inner_local_tp % headdim == 0
    assert zxBCdt.shape[-1] == projected_dim
    assert conv_weight.shape == (conv_dim, 1, 4)
    assert conv_bias.shape == (conv_dim,)
    assert dt_bias.shape == (nheads_local_tp,)
    assert A_log.shape == (nheads_local_tp,)
    assert D.shape == (nheads_local_tp,)

    segment_len = zxBCdt.shape[0] // 2
    assert segment_len >= 3, (
        "true CP Mamba requires local segment length >= 3 for width-4 convolution; "
        f"got segment_len={segment_len}"
    )


# =============================================================================
# Forward convolution, state passing, and scan
# =============================================================================


def _prepare_folded_forward(
    zxBCdt: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
    chunk_size: int,
    return_conv_halos: bool = False,
    return_initial_states: bool = False,
) -> dict[str, torch.Tensor]:
    """Run the compact, state-free true-CP forward scan."""

    prepared = _prepare_folded_scan_state(
        zxBCdt,
        conv_weight,
        conv_bias,
        dt_bias,
        A_log,
        cp_group=cp_group,
        d_inner_local_tp=d_inner_local_tp,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
        headdim=headdim,
        chunk_size=chunk_size,
        corrected_states_out_dtype=zxBCdt.dtype,
        return_backward_state=False,
        return_conv_halos=return_conv_halos,
        return_global_log_alpha=return_initial_states,
    )
    C = prepared.pop("C")
    B = prepared.pop("B")
    x = prepared.pop("x")
    dt_out = prepared.pop("dt_out")
    dA_cumsum = prepared.pop("dA_cumsum")
    dA_cs_rev = prepared.pop("dA_cs_rev")
    local_initial_states = prepared.pop("local_initial_states")
    z = prepared.pop("z")
    conv_halos = prepared.pop("xBC_conv_halos") if return_conv_halos else None
    global_log_alpha = prepared.pop("global_log_alpha") if return_initial_states else None
    prepared.clear()
    out = mamba2_fwd_triton(
        C,
        B,
        x,
        dt_out,
        dA_cumsum,
        dA_cs_rev,
        local_initial_states,
        D,
        z,
        chunk_size=chunk_size,
    )
    result = {"out": out}
    if return_conv_halos:
        result["xBC_conv_halos"] = conv_halos
    if return_initial_states:
        result["local_initial_states"] = local_initial_states
        result["global_log_alpha"] = global_log_alpha
    return result


def _prepare_folded_scan_state(
    zxBCdt: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
    chunk_size: int,
    corrected_states_out_dtype: Optional[torch.dtype],
    return_backward_state: bool,
    cached_conv_halos: Optional[torch.Tensor] = None,
    return_conv_halos: bool = False,
    defer_boundary_exchange: bool = False,
    return_global_log_alpha: bool = False,
    skip_segment_summary: bool = False,
) -> dict[str, Optional[torch.Tensor]]:
    """Recompute folded conv and SSD state-passing inputs without running chunk scan."""

    if defer_boundary_exchange and not return_backward_state:
        raise ValueError("defer_boundary_exchange is only valid for backward recomputation")
    if skip_segment_summary and not defer_boundary_exchange:
        raise ValueError("skip_segment_summary requires defer_boundary_exchange")
    if return_global_log_alpha and defer_boundary_exchange:
        raise ValueError("return_global_log_alpha requires the non-deferred boundary exchange")

    batch = zxBCdt.shape[1]
    nheads_local_tp = d_inner_local_tp // headdim
    z_streams, xBC_streams, dt_streams = _split_projected_streams(
        zxBCdt,
        d_inner_local_tp=d_inner_local_tp,
        nheads_local_tp=nheads_local_tp,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
    )
    halos = _resolve_conv_halos_streams(xBC_streams, cp_group, cached_conv_halos)

    # =============================================================================
    # Width-4 causal convolution and scan-layout formatting
    # =============================================================================
    x, B, C = _causal_conv_format_streams_triton(
        xBC_streams,
        halos,
        conv_weight,
        conv_bias,
        d_inner_local_tp=d_inner_local_tp,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
        headdim=headdim,
    )
    z = _format_z_streams(z_streams, headdim=headdim)
    if not return_backward_state:
        del xBC_streams
    if not return_backward_state and not return_conv_halos:
        del halos

    # =============================================================================
    # Local SSD summaries and global context-parallel prefix
    # =============================================================================
    A = -torch.exp(A_log.float())
    dt_out, dA_cumsum, dA_cs_rev = compute_dacs_compact_triton(
        dt_streams,
        A,
        dt_bias.float(),
        chunk_size,
    )
    del A
    global_log_alpha = None
    if skip_segment_summary:
        local_beta = None
        local_log_alpha = None
        local_initial_states = None
    else:
        beta, folded_log_alpha = mamba2_get_final_state_triton(
            B, x, dt_out, dA_cumsum, dA_cs_rev, chunk_size
        )
        local_beta = beta.view(2, batch, nheads_local_tp, headdim, d_state)
        local_log_alpha = folded_log_alpha.view(2, batch, nheads_local_tp)
        if defer_boundary_exchange:
            local_initial_states = None
        elif return_global_log_alpha:
            local_initial_states, global_log_alpha = _local_segment_initial_states(
                local_log_alpha, local_beta, cp_group, return_global_log_alpha=True
            )
        else:
            local_initial_states = _local_segment_initial_states(
                local_log_alpha, local_beta, cp_group
            )

    result = {
        "x": x,
        "B": B,
        "C": C,
        "z": z,
        "dA_cumsum": dA_cumsum,
        "dA_cs_rev": dA_cs_rev,
        "dt_out": dt_out,
        "local_beta": local_beta,
        "local_log_alpha": local_log_alpha,
        "local_initial_states": local_initial_states,
    }
    if return_global_log_alpha:
        result["global_log_alpha"] = global_log_alpha
    if return_backward_state:
        result["dt_streams"] = dt_streams
        result["xBC_streams"] = xBC_streams
    else:
        del dt_streams
    if return_backward_state or return_conv_halos:
        result["xBC_conv_halos"] = halos
    _ = corrected_states_out_dtype
    return result


def _split_projected_streams(
    zxBCdt: torch.Tensor,
    *,
    d_inner_local_tp: int,
    nheads_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_len = zxBCdt.shape[0] // 2
    streams = rearrange(zxBCdt, "(s l) b d -> s l b d", s=2, l=segment_len)
    z, xBC, dt = torch.split(
        streams,
        [d_inner_local_tp, d_inner_local_tp + 2 * ngroups_local_tp * d_state, nheads_local_tp],
        dim=-1,
    )
    return z, xBC, dt


def _format_z_streams(z: torch.Tensor, *, headdim: int) -> torch.Tensor:
    """Expose the head dimension without folding stream and batch or copying."""
    if z.shape[-1] % headdim != 0:
        raise ValueError("z hidden dimension must be divisible by headdim")
    return z.unflatten(-1, (z.shape[-1] // headdim, headdim))


def _format_grad_output_streams(grad_output: torch.Tensor, *, headdim: int) -> torch.Tensor:
    """Expose two streams and the head dimension without copying grad_output."""
    if grad_output.ndim != 3 or grad_output.shape[0] % 2 != 0:
        raise ValueError("grad_output must have shape [2 * segment, batch, hidden]")
    if grad_output.shape[-1] % headdim != 0:
        raise ValueError("grad_output hidden dimension must be divisible by headdim")
    segment_len = grad_output.shape[0] // 2
    nheads = grad_output.shape[-1] // headdim
    return grad_output.unflatten(0, (2, segment_len)).unflatten(-1, (nheads, headdim))


# =============================================================================
# Context-parallel halo exchange
# =============================================================================


def _resolve_conv_halos_streams(
    xBC_streams: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    cached_conv_halos: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reuse selected forward halos or perform their rank-slot exchange."""

    if cached_conv_halos is None:
        return _exchange_conv_halos_streams(xBC_streams, cp_group)

    expected_shape = (2, 3, xBC_streams.shape[2], xBC_streams.shape[3])
    if cached_conv_halos.shape != expected_shape:
        raise ValueError(
            f"cached_conv_halos must have shape {expected_shape}, "
            f"got {tuple(cached_conv_halos.shape)}"
        )
    if cached_conv_halos.device != xBC_streams.device:
        raise ValueError("cached_conv_halos must be on the same device as xBC_streams")
    if cached_conv_halos.dtype != xBC_streams.dtype:
        raise ValueError("cached_conv_halos must have the same dtype as xBC_streams")
    return cached_conv_halos


def _exchange_conv_halos_streams(
    xBC_streams: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    local_halos = xBC_streams[:, -3:].contiguous()
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    gathered = _all_gather_rank_tensor(local_halos, cp_group)
    flat_halos = gathered.reshape(2 * cp_size, *gathered.shape[2:])
    source_indices = _segment_indices(cp_rank, cp_size, local_halos.device)["conv_sources"]
    halos = flat_halos.index_select(0, source_indices).contiguous()
    del gathered

    if cp_rank == 0:
        halos[0].zero_()
    return halos


# =============================================================================
# Width-4 causal-convolution launchers
# =============================================================================


def _causal_conv_format_streams_triton(
    xBC: torch.Tensor,
    halos: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    *,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    streams, local_seqlen, batch, conv_dim = xBC.shape
    nheads_local_tp = d_inner_local_tp // headdim
    folded_batch = streams * batch
    x = torch.empty(
        folded_batch, local_seqlen, nheads_local_tp, headdim, device=xBC.device, dtype=xBC.dtype
    )
    B = torch.empty(
        folded_batch, local_seqlen, ngroups_local_tp, d_state, device=xBC.device, dtype=xBC.dtype
    )
    C = torch.empty_like(B)

    block_t = 16
    block_d = 64
    grid = (triton.cdiv(local_seqlen, block_t), triton.cdiv(conv_dim, block_d), folded_batch)
    _causal_conv_format_streams_kernel[grid](
        xBC,
        halos,
        conv_weight,
        conv_bias,
        x,
        B,
        C,
        local_seqlen,
        batch,
        conv_dim,
        d_inner_local_tp,
        headdim,
        ngroups_local_tp,
        d_state,
        xBC.stride(0),
        xBC.stride(1),
        xBC.stride(2),
        xBC.stride(3),
        halos.stride(0),
        halos.stride(1),
        halos.stride(2),
        halos.stride(3),
        conv_weight.stride(0),
        conv_weight.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(3),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return x, B, C


def _causal_conv_format_streams_backward_triton(
    xBC: torch.Tensor,
    halos: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dx: torch.Tensor,
    dB: torch.Tensor,
    dC: torch.Tensor,
    *,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    streams, local_seqlen, batch, conv_dim = xBC.shape
    folded_batch = streams * batch
    dxBC = torch.empty_like(xBC)
    dhalos = torch.empty_like(halos)
    dweight = torch.empty_like(conv_weight, dtype=torch.float32)
    dbias = torch.empty(conv_dim, device=xBC.device, dtype=torch.float32)

    # =============================================================================
    # Fused convolution gradients and compact partial reductions
    # =============================================================================
    # Time tiling keeps the long sequence parallel. Compact FP32 partial and
    # edge workspaces avoid both atomics and a full native-layout dpreact.
    block_t = 64
    block_d = 16
    num_tiles = triton.cdiv(local_seqlen + 3, block_t)
    num_partials = folded_batch * num_tiles
    partials = torch.empty(num_partials, conv_dim, 5, device=xBC.device, dtype=torch.float32)
    edges = torch.empty(num_partials, 6, conv_dim, device=xBC.device, dtype=torch.float32)
    grid = (num_tiles, triton.cdiv(conv_dim, block_d), folded_batch)
    _causal_conv_backward_fused_width4_tiled_kernel[grid](
        xBC,
        halos,
        conv_weight,
        conv_bias,
        dx,
        dB,
        dC,
        dxBC,
        partials,
        edges,
        local_seqlen,
        batch,
        conv_dim,
        d_inner_local_tp,
        headdim,
        ngroups_local_tp,
        d_state,
        num_tiles,
        xBC.stride(0),
        xBC.stride(1),
        xBC.stride(2),
        xBC.stride(3),
        halos.stride(0),
        halos.stride(1),
        halos.stride(2),
        halos.stride(3),
        conv_weight.stride(0),
        conv_weight.stride(2),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        dx.stride(3),
        dB.stride(0),
        dB.stride(1),
        dB.stride(2),
        dB.stride(3),
        dC.stride(0),
        dC.stride(1),
        dC.stride(2),
        dC.stride(3),
        dxBC.stride(0),
        dxBC.stride(1),
        dxBC.stride(2),
        dxBC.stride(3),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        edges.stride(0),
        edges.stride(1),
        edges.stride(2),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _causal_conv_backward_boundary_width4_kernel[grid](
        edges,
        conv_weight,
        dxBC,
        dhalos,
        local_seqlen,
        batch,
        conv_dim,
        num_tiles,
        edges.stride(0),
        edges.stride(1),
        edges.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(2),
        dxBC.stride(0),
        dxBC.stride(1),
        dxBC.stride(2),
        dxBC.stride(3),
        dhalos.stride(0),
        dhalos.stride(1),
        dhalos.stride(2),
        dhalos.stride(3),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    reduce_block_d = 64
    reduce_grid = (triton.cdiv(conv_dim, reduce_block_d),)
    _causal_conv_backward_reduce_width4_kernel[reduce_grid](
        partials,
        dweight,
        dbias,
        conv_dim,
        num_partials,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        dweight.stride(0),
        dweight.stride(2),
        BLOCK_D=reduce_block_d,
        num_warps=4,
    )
    return (dxBC, dhalos, dweight.to(dtype=conv_weight.dtype), dbias.to(dtype=conv_bias.dtype))


# =============================================================================
# Context-parallel summary, halo, and segment routing
# =============================================================================


def _local_segment_initial_states(
    local_log_alpha: torch.Tensor,
    local_beta: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    *,
    return_global_log_alpha: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Build the incoming recurrent state for each segment owned by this CP rank.

    Attention-style load balancing gives rank ``r`` two noncontiguous global
    segments: ``r`` and ``2 * cp_size - r - 1``.  Each local segment summarizes
    its recurrence as ``state_out = exp(log_alpha) * state_in + beta``.  We
    gather those summaries from every CP rank, restore causal segment order,
    run an exclusive affine prefix, and select the two incoming states needed
    to evaluate this rank's local segments.

    ``return_global_log_alpha=True`` additionally returns the causally ordered
    global log decays as a compact copy, so a caching forward can hand them to
    the reverse-scan-only backward boundary exchange.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    indices = _segment_indices(cp_rank, cp_size, local_beta.device)

    beta_flat = local_beta.float().reshape(*local_beta.shape[:-2], -1)
    packed = torch.cat([local_log_alpha.float().unsqueeze(-1), beta_flat], dim=-1)
    rank_slot = _all_gather_rank_tensor(packed.contiguous(), cp_group)
    global_summary = rank_slot.reshape(2 * cp_size, *rank_slot.shape[2:]).index_select(
        0, indices["global_order"]
    )
    global_initial_states = _exclusive_affine_prefix_states(global_summary, local_beta.shape[-2:])
    local_initial_states = global_initial_states.index_select(
        0, indices["local_segments"]
    ).contiguous()
    if return_global_log_alpha:
        # Copy the slice so the cache does not pin the full packed gather buffer.
        return local_initial_states, global_summary[..., 0].contiguous()
    return local_initial_states


def _local_segment_boundary_states(
    local_log_alpha: torch.Tensor,
    local_beta: torch.Tensor,
    local_gamma: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return this rank's forward and adjoint boundary states with one gather.

    Each segment has the paired affine summaries

    ``state_out = exp(log_alpha) * state_in + beta`` and
    ``dstate_in = exp(log_alpha) * dstate_out + gamma``.

    The forward exclusive scan starts from a zero global initial state. The
    adjoint scan traverses segments in reverse and starts from a zero terminal
    adjoint. Packing both additive summaries into one FP32 collective keeps the
    backward boundary exchange to a single all-gather.
    """

    if local_log_alpha.ndim != 3 or local_log_alpha.shape[0] != 2:
        raise ValueError("local_log_alpha must have shape [2, batch, heads]")
    if local_beta.ndim != 5 or local_beta.shape[0] != 2:
        raise ValueError("local_beta must have shape [2, batch, heads, P, N]")
    if local_gamma.shape != local_beta.shape:
        raise ValueError("local_gamma must have the same shape as local_beta")
    if local_log_alpha.shape != local_beta.shape[:-2]:
        raise ValueError("local_log_alpha and local_beta leading dimensions must match")
    if local_beta.device != local_log_alpha.device or local_gamma.device != local_beta.device:
        raise ValueError("local_log_alpha, local_beta, and local_gamma must share a device")

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    indices = _segment_indices(cp_rank, cp_size, local_beta.device)
    state_shape = local_beta.shape[-2:]
    state_numel = state_shape[0] * state_shape[1]

    beta_flat = local_beta.float().reshape(*local_beta.shape[:-2], state_numel)
    gamma_flat = local_gamma.float().reshape(*local_gamma.shape[:-2], state_numel)
    packed = torch.cat(
        [local_log_alpha.float().unsqueeze(-1), beta_flat, gamma_flat], dim=-1
    ).contiguous()
    with _record_function("mamba_true_cp_v2:bwd_boundary_summary_all_gather"):
        rank_slot = _all_gather_rank_tensor(packed, cp_group)
    global_packed = rank_slot.reshape(2 * cp_size, *rank_slot.shape[2:]).index_select(
        0, indices["global_order"]
    )

    global_log_alpha = global_packed[..., :1]
    global_beta = global_packed[..., 1 : 1 + state_numel]
    global_gamma = global_packed[..., 1 + state_numel :]
    forward_summary = torch.cat([global_log_alpha, global_beta], dim=-1)
    global_initial_states = _exclusive_affine_prefix_states(forward_summary, state_shape)

    reverse_summary = torch.cat([global_log_alpha, global_gamma], dim=-1).flip(0).contiguous()
    global_final_dstates = _exclusive_affine_prefix_states(reverse_summary, state_shape).flip(0)

    local_segments = indices["local_segments"]
    local_initial_states = global_initial_states.index_select(0, local_segments).contiguous()
    local_final_dstates = global_final_dstates.index_select(0, local_segments).contiguous()
    return local_initial_states, local_final_dstates


def _local_segment_final_dstates(
    global_log_alpha: torch.Tensor,
    local_gamma: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Return this rank's adjoint boundary states from cached forward decays.

    Each segment's adjoint recurrence is
    ``dstate_in = exp(log_alpha) * dstate_out + gamma``. The caching forward
    already retained the causally ordered global log decays, so backward only
    gathers each segment's additive adjoint summary and runs the reversed
    exclusive affine scan, roughly halving the boundary payload versus the
    packed ``(log_alpha, beta, gamma)`` exchange.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    if local_gamma.ndim != 5 or local_gamma.shape[0] != 2:
        raise ValueError("local_gamma must have shape [2, batch, heads, P, N]")
    if global_log_alpha.shape != (2 * cp_size, *local_gamma.shape[1:3]):
        raise ValueError("global_log_alpha must have shape [2 * cp_size, batch, heads]")
    if global_log_alpha.device != local_gamma.device:
        raise ValueError("global_log_alpha and local_gamma must share a device")

    indices = _segment_indices(cp_rank, cp_size, local_gamma.device)
    state_shape = local_gamma.shape[-2:]
    state_numel = state_shape[0] * state_shape[1]

    gamma_flat = local_gamma.float().reshape(*local_gamma.shape[:-2], state_numel)
    with _record_function("mamba_true_cp_v2:bwd_gamma_all_gather"):
        rank_slot = _all_gather_rank_tensor(gamma_flat, cp_group)
    global_gamma = rank_slot.reshape(2 * cp_size, *rank_slot.shape[2:]).index_select(
        0, indices["global_order"]
    )

    reverse_summary = (
        torch.cat([global_log_alpha.float().unsqueeze(-1), global_gamma], dim=-1)
        .flip(0)
        .contiguous()
    )
    global_final_dstates = _exclusive_affine_prefix_states(reverse_summary, state_shape).flip(0)
    return global_final_dstates.index_select(0, indices["local_segments"]).contiguous()


def _route_conv_halo_grads(
    dxBC_streams: torch.Tensor, halo_grads: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Return width-4 convolution halo gradients to the owning CP segments.

    Forward gives each segment three values from its predecessor's tail.  The
    gradient of that halo therefore belongs to the predecessor, not to the
    segment that computed it.  After gathering halo gradients in rank-slot
    order, each local segment selects the contribution produced by its global
    successor and adds it to the last three positions of ``dxBC_streams``.
    The final global segment has no successor, so its synthetic contribution is
    cleared explicitly.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    successor_indices = _segment_indices(cp_rank, cp_size, halo_grads.device)["successors"]
    with _record_function("mamba_true_cp_v2:bwd_conv_halo_grad_all_gather"):
        gathered = _all_gather_rank_tensor(halo_grads.contiguous(), cp_group)
    successor_halo_grads = gathered.reshape(2 * cp_size, *gathered.shape[2:]).index_select(
        0, successor_indices
    )
    if cp_rank == 0:
        successor_halo_grads[1].zero_()

    dxBC_streams[:, -3:].add_(successor_halo_grads)
    return dxBC_streams


def _exclusive_affine_prefix_states(
    packed_summary: torch.Tensor, state_shape: Tuple[int, int]
) -> torch.Tensor:
    """Compose packed affine summaries into per-segment incoming states.

    ``packed_summary[..., 0]`` stores each segment's log decay and the remaining
    fields store its flattened additive state.  Treating each global segment as
    a size-one chunk lets Mamba's state-passing kernel perform the associative
    affine scan.  With a zero global initial state, its output is exactly the
    state entering each segment, which is consumed by both forward prefix setup
    and the reversed prefix used during backward.
    """

    states_in = rearrange(packed_summary[..., 1:], "s b h pn -> b s h pn").contiguous()
    dA = rearrange(packed_summary[..., 0], "s b h -> b h s").contiguous()
    states, _ = _state_passing_fwd(
        states_in, dA, initial_states=None, seq_idx=None, chunk_size=1, out_dtype=torch.float32
    )
    return rearrange(states, "b s h (p n) -> s b h p n", n=state_shape[-1]).contiguous()


def _all_gather_rank_tensor(
    tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Gather one tensor from every CP rank with differentiable ownership.

    The result has an explicit leading CP-rank dimension followed by the local
    rank-slot dimensions.  Megatron's sequence-parallel gather is used because
    its autograd rule reduce-scatters gradients back to the rank that produced
    each value.  The summary-prefix and convolution-halo exchanges both rely on
    that ownership behavior.
    """

    gathered = gather_from_sequence_parallel_region(
        tensor.contiguous(), tensor_parallel_output_grad=True, group=cp_group
    )
    return gathered.reshape(cp_group.size(), *tensor.shape).contiguous()


def _segment_indices(cp_rank: int, cp_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Cache all mappings for the two-segment load-balanced CP layout.

    Rank ``r`` owns global segments ``r`` and ``2 * cp_size - r - 1``, while an
    all-gather stores values in rank-major, two-slot order.  ``global_order``
    converts that storage to causal segment order, ``local_segments`` selects
    this rank's entries from a globally ordered tensor, ``conv_sources`` finds
    the predecessor tails used as forward convolution halos, and ``successors``
    finds the backward halo gradients destined for those tails.  The tensors
    are immutable for a given rank, CP size, and device, so they are constructed
    once and reused by all CP routing phases.
    """

    key = (cp_rank, cp_size, device.type, device.index)
    indices = _SEGMENT_INDEX_CACHE.get(key)
    if indices is not None:
        return indices

    local_segments = (cp_rank, 2 * cp_size - cp_rank - 1)

    def owner_flat_index(segment_id: int) -> int:
        if segment_id < cp_size:
            return 2 * segment_id
        return 2 * (2 * cp_size - segment_id - 1) + 1

    conv_sources = tuple(owner_flat_index(max(segment - 1, 0)) for segment in local_segments)
    successors = tuple(
        owner_flat_index(min(segment + 1, 2 * cp_size - 1)) for segment in local_segments
    )
    values = {
        "global_order": tuple(range(0, 2 * cp_size, 2)) + tuple(range(2 * cp_size - 1, 0, -2)),
        "local_segments": local_segments,
        "conv_sources": conv_sources,
        "successors": successors,
    }
    indices = {
        name: torch.tensor(index_values, device=device, dtype=torch.long)
        for name, index_values in values.items()
    }
    _SEGMENT_INDEX_CACHE[key] = indices
    return indices
