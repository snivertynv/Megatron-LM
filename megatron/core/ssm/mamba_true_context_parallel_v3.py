# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-collective training prototype for Mamba-2 true context parallelism.

True CP v3 splits every width-4 convolution segment into a three-token prefix
and a halo-independent body.  Each rank summarizes its two bodies, packs those
summaries with the raw convolution boundaries and prefix dt values, and issues
one all-gather.  A small persistent prefix kernel replays the globally earlier
prefixes and composes the gathered body summaries before the existing v2
state-free scan evaluates the local bodies.

Backward uses one additional all-gather.  It communicates body adjoint
summaries plus raw prefix replay operands, resolves the global adjoint chain,
and replays only the immediate successor prefix needed to return each local
convolution-tail gradient.  The separate v3 model spec leaves production v2
unchanged.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
from mamba_ssm.ops.triton.softplus import softplus
from torch.autograd.profiler import record_function

from megatron.core.ssm.mamba2_utils.mamba2_fwd import (
    compute_dacs_compact_triton,
    mamba2_fwd_triton,
    mamba2_get_final_state_triton,
)
from megatron.core.ssm.mamba2_utils.mamba2_bwd import (
    mamba2_bwd_decay_map_cp_triton,
    mamba2_bwd_dk_dv_cp_triton,
    mamba2_bwd_dq_cp_triton,
    mamba2_get_initial_dstate_triton,
    mamba2_get_subsequence_dstates_triton,
    mamba2_get_subsequence_states_triton,
)
from megatron.core.ssm.mamba_true_context_parallel_v2 import (
    _all_gather_rank_tensor,
    _causal_conv_format_streams_backward_triton,
    _causal_conv_format_streams_triton,
    _check_inputs,
    _format_grad_output_streams,
    _format_z_streams,
    _segment_indices,
    _split_projected_streams,
)


_PREFIX_TOKENS = 3
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


@dataclass(frozen=True)
class _PayloadLayout:
    """Flat FP32 payload offsets for one ``(rank-slot, batch)`` item."""

    body_log_alpha: int
    body_beta: int
    prefix_xbc: int
    tail_xbc: int
    prefix_dt: int
    width: int


@dataclass(frozen=True)
class _BackwardPayloadLayout:
    """Flat FP32 backward payload offsets for one ``(rank-slot, batch)`` item."""

    body_log_alpha: int
    body_gamma: int
    prefix_xbc: int
    tail_xbc: int
    prefix_dt: int
    prefix_dpre: int
    width: int


def _payload_layout(
    nheads: int, headdim: int, d_state: int, conv_dim: int
) -> _PayloadLayout:
    body_log_alpha = 0
    body_beta = body_log_alpha + nheads
    prefix_xbc = body_beta + nheads * headdim * d_state
    tail_xbc = prefix_xbc + _PREFIX_TOKENS * conv_dim
    prefix_dt = tail_xbc + _PREFIX_TOKENS * conv_dim
    width = prefix_dt + _PREFIX_TOKENS * nheads
    return _PayloadLayout(
        body_log_alpha=body_log_alpha,
        body_beta=body_beta,
        prefix_xbc=prefix_xbc,
        tail_xbc=tail_xbc,
        prefix_dt=prefix_dt,
        width=width,
    )


def _backward_payload_layout(
    nheads: int, headdim: int, d_state: int, conv_dim: int
) -> _BackwardPayloadLayout:
    body_log_alpha = 0
    body_gamma = body_log_alpha + nheads
    prefix_xbc = body_gamma + nheads * headdim * d_state
    tail_xbc = prefix_xbc + _PREFIX_TOKENS * conv_dim
    prefix_dt = tail_xbc + _PREFIX_TOKENS * conv_dim
    prefix_dpre = prefix_dt + _PREFIX_TOKENS * nheads
    width = prefix_dpre + _PREFIX_TOKENS * nheads * headdim
    return _BackwardPayloadLayout(
        body_log_alpha=body_log_alpha,
        body_gamma=body_gamma,
        prefix_xbc=prefix_xbc,
        tail_xbc=tail_xbc,
        prefix_dt=prefix_dt,
        prefix_dpre=prefix_dpre,
        width=width,
    )


@triton.jit
def _causal_conv_body_width4_kernel(
    xbc_ptr,
    weight_ptr,
    bias_ptr,
    x_ptr,
    b_ptr,
    c_ptr,
    body_seqlen: tl.constexpr,
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
    """Convolve global-local positions 3..L-1 from local raw positions only."""

    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    folded_batch = tl.program_id(2)
    stream = folded_batch // batch
    batch_idx = folded_batch - stream * batch

    body_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    body_mask = body_offsets < body_seqlen
    d_mask = d_offsets < conv_dim

    # Body output ``body_t`` is original token ``body_t + 3``.  Width four
    # therefore reads original local raw positions ``body_t .. body_t + 3``.
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for tap in tl.static_range(0, 4):
        raw_pos = body_offsets + tap
        values = tl.load(
            xbc_ptr
            + stream * stride_xbc_s
            + raw_pos[:, None] * stride_xbc_l
            + batch_idx * stride_xbc_b
            + d_offsets[None, :] * stride_xbc_d,
            mask=body_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + d_offsets * stride_weight_d + tap * stride_weight_w,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        acc += values * weight[None, :]

    bias = tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = acc * tl.sigmoid(acc)

    x_mask = d_offsets < d_inner
    x_head = d_offsets // headdim
    x_dim = d_offsets - x_head * headdim
    tl.store(
        x_ptr
        + folded_batch * stride_x_sb
        + body_offsets[:, None] * stride_x_l
        + x_head[None, :] * stride_x_h
        + x_dim[None, :] * stride_x_p,
        acc,
        mask=body_mask[:, None] & x_mask[None, :],
    )

    qk_dim = ngroups * d_state
    b_offsets = d_offsets - d_inner
    b_mask = (b_offsets >= 0) & (b_offsets < qk_dim)
    b_group = b_offsets // d_state
    b_state = b_offsets - b_group * d_state
    tl.store(
        b_ptr
        + folded_batch * stride_b_sb
        + body_offsets[:, None] * stride_b_l
        + b_group[None, :] * stride_b_g
        + b_state[None, :] * stride_b_n,
        acc,
        mask=body_mask[:, None] & b_mask[None, :],
    )

    c_offsets = d_offsets - d_inner - qk_dim
    c_mask = (c_offsets >= 0) & (c_offsets < qk_dim)
    c_group = c_offsets // d_state
    c_state = c_offsets - c_group * d_state
    tl.store(
        c_ptr
        + folded_batch * stride_c_sb
        + body_offsets[:, None] * stride_c_l
        + c_group[None, :] * stride_c_g
        + c_state[None, :] * stride_c_n,
        acc,
        mask=body_mask[:, None] & c_mask[None, :],
    )


def _causal_conv_body_width4_triton(
    xbc: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    *,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    streams, segment_len, batch, conv_dim = xbc.shape
    body_seqlen = segment_len - _PREFIX_TOKENS
    if body_seqlen <= 0:
        raise ValueError("the body convolution requires at least one body token")
    nheads = d_inner_local_tp // headdim
    folded_batch = streams * batch
    x = torch.empty(
        folded_batch,
        body_seqlen,
        nheads,
        headdim,
        device=xbc.device,
        dtype=xbc.dtype,
    )
    B = torch.empty(
        folded_batch,
        body_seqlen,
        ngroups_local_tp,
        d_state,
        device=xbc.device,
        dtype=xbc.dtype,
    )
    C = torch.empty_like(B)
    block_t = 64
    block_d = 16
    grid = (
        triton.cdiv(body_seqlen, block_t),
        triton.cdiv(conv_dim, block_d),
        folded_batch,
    )
    _causal_conv_body_width4_kernel[grid](
        xbc,
        conv_weight,
        conv_bias,
        x,
        B,
        C,
        body_seqlen,
        batch,
        conv_dim,
        d_inner_local_tp,
        headdim,
        ngroups_local_tp,
        d_state,
        xbc.stride(0),
        xbc.stride(1),
        xbc.stride(2),
        xbc.stride(3),
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


@triton.jit
def _pack_forward_payload_kernel(
    body_log_ptr,
    body_beta_ptr,
    xbc_ptr,
    dt_ptr,
    payload_ptr,
    batch: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    d_state: tl.constexpr,
    conv_dim: tl.constexpr,
    segment_len: tl.constexpr,
    body_log_offset: tl.constexpr,
    body_beta_offset: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    prefix_dt_offset: tl.constexpr,
    payload_width: tl.constexpr,
    stride_log_s: tl.constexpr,
    stride_log_b: tl.constexpr,
    stride_log_h: tl.constexpr,
    stride_beta_s: tl.constexpr,
    stride_beta_b: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_beta_p: tl.constexpr,
    stride_beta_n: tl.constexpr,
    stride_xbc_s: tl.constexpr,
    stride_xbc_l: tl.constexpr,
    stride_xbc_b: tl.constexpr,
    stride_xbc_d: tl.constexpr,
    stride_dt_s: tl.constexpr,
    stride_dt_l: tl.constexpr,
    stride_dt_b: tl.constexpr,
    stride_dt_h: tl.constexpr,
    stride_payload_s: tl.constexpr,
    stride_payload_b: tl.constexpr,
    stride_payload_w: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    stream_batch = tl.program_id(0)
    block = tl.program_id(1)
    stream = stream_batch // batch
    batch_idx = stream_batch - stream * batch
    offsets = block * BLOCK_W + tl.arange(0, BLOCK_W)
    valid = offsets < payload_width
    values = tl.zeros((BLOCK_W,), dtype=tl.float32)

    log_rel = offsets - body_log_offset
    log_mask = valid & (log_rel >= 0) & (log_rel < nheads)
    log_h = tl.minimum(tl.maximum(log_rel, 0), nheads - 1)
    log_values = tl.load(
        body_log_ptr
        + stream * stride_log_s
        + batch_idx * stride_log_b
        + log_h * stride_log_h,
        mask=log_mask,
        other=0.0,
    ).to(tl.float32)
    values += tl.where(log_mask, log_values, 0.0)

    beta_width = nheads * headdim * d_state
    beta_rel = offsets - body_beta_offset
    beta_mask = valid & (beta_rel >= 0) & (beta_rel < beta_width)
    beta_safe = tl.minimum(tl.maximum(beta_rel, 0), beta_width - 1)
    beta_h = beta_safe // (headdim * d_state)
    beta_rem = beta_safe - beta_h * headdim * d_state
    beta_p = beta_rem // d_state
    beta_n = beta_rem - beta_p * d_state
    beta_values = tl.load(
        body_beta_ptr
        + stream * stride_beta_s
        + batch_idx * stride_beta_b
        + beta_h * stride_beta_h
        + beta_p * stride_beta_p
        + beta_n * stride_beta_n,
        mask=beta_mask,
        other=0.0,
    ).to(tl.float32)
    values += tl.where(beta_mask, beta_values, 0.0)

    boundary_width = 3 * conv_dim
    prefix_rel = offsets - prefix_xbc_offset
    prefix_mask = valid & (prefix_rel >= 0) & (prefix_rel < boundary_width)
    prefix_safe = tl.minimum(tl.maximum(prefix_rel, 0), boundary_width - 1)
    prefix_t = prefix_safe // conv_dim
    prefix_d = prefix_safe - prefix_t * conv_dim
    prefix_values = tl.load(
        xbc_ptr
        + stream * stride_xbc_s
        + prefix_t * stride_xbc_l
        + batch_idx * stride_xbc_b
        + prefix_d * stride_xbc_d,
        mask=prefix_mask,
        other=0.0,
    ).to(tl.float32)
    values += tl.where(prefix_mask, prefix_values, 0.0)

    tail_rel = offsets - tail_xbc_offset
    tail_mask = valid & (tail_rel >= 0) & (tail_rel < boundary_width)
    tail_safe = tl.minimum(tl.maximum(tail_rel, 0), boundary_width - 1)
    tail_t = tail_safe // conv_dim
    tail_d = tail_safe - tail_t * conv_dim
    tail_values = tl.load(
        xbc_ptr
        + stream * stride_xbc_s
        + (segment_len - 3 + tail_t) * stride_xbc_l
        + batch_idx * stride_xbc_b
        + tail_d * stride_xbc_d,
        mask=tail_mask,
        other=0.0,
    ).to(tl.float32)
    values += tl.where(tail_mask, tail_values, 0.0)

    dt_width = 3 * nheads
    dt_rel = offsets - prefix_dt_offset
    dt_mask = valid & (dt_rel >= 0) & (dt_rel < dt_width)
    dt_safe = tl.minimum(tl.maximum(dt_rel, 0), dt_width - 1)
    dt_t = dt_safe // nheads
    dt_h = dt_safe - dt_t * nheads
    dt_values = tl.load(
        dt_ptr
        + stream * stride_dt_s
        + dt_t * stride_dt_l
        + batch_idx * stride_dt_b
        + dt_h * stride_dt_h,
        mask=dt_mask,
        other=0.0,
    ).to(tl.float32)
    values += tl.where(dt_mask, dt_values, 0.0)

    tl.store(
        payload_ptr
        + stream * stride_payload_s
        + batch_idx * stride_payload_b
        + offsets * stride_payload_w,
        values,
        mask=valid,
    )


def _pack_forward_payload_triton(
    body_log_alpha: torch.Tensor,
    body_beta: torch.Tensor,
    xbc_streams: torch.Tensor,
    dt_streams: torch.Tensor,
    layout: _PayloadLayout,
) -> torch.Tensor:
    streams, segment_len, batch, conv_dim = xbc_streams.shape
    _, _, nheads, headdim, d_state = body_beta.shape
    payload = torch.empty(
        streams, batch, layout.width, device=xbc_streams.device, dtype=torch.float32
    )
    block_w = 256
    grid = (streams * batch, triton.cdiv(layout.width, block_w))
    _pack_forward_payload_kernel[grid](
        body_log_alpha,
        body_beta,
        xbc_streams,
        dt_streams,
        payload,
        batch,
        nheads,
        headdim,
        d_state,
        conv_dim,
        segment_len,
        layout.body_log_alpha,
        layout.body_beta,
        layout.prefix_xbc,
        layout.tail_xbc,
        layout.prefix_dt,
        layout.width,
        body_log_alpha.stride(0),
        body_log_alpha.stride(1),
        body_log_alpha.stride(2),
        body_beta.stride(0),
        body_beta.stride(1),
        body_beta.stride(2),
        body_beta.stride(3),
        body_beta.stride(4),
        xbc_streams.stride(0),
        xbc_streams.stride(1),
        xbc_streams.stride(2),
        xbc_streams.stride(3),
        dt_streams.stride(0),
        dt_streams.stride(1),
        dt_streams.stride(2),
        dt_streams.stride(3),
        payload.stride(0),
        payload.stride(1),
        payload.stride(2),
        BLOCK_W=block_w,
        num_warps=4,
    )
    return payload


@triton.jit
def _rank_slot_payload_base(
    segment,
    batch_idx,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    CP_SIZE: tl.constexpr,
):
    owner_rank = tl.where(segment < CP_SIZE, segment, 2 * CP_SIZE - segment - 1)
    owner_slot = tl.where(segment < CP_SIZE, 0, 1)
    return (
        owner_rank * stride_payload_rank
        + owner_slot * stride_payload_slot
        + batch_idx * stride_payload_batch
    )


@triton.jit
def _convolve_payload_prefix_token(
    payload_ptr,
    weight_ptr,
    bias_ptr,
    out_type_ptr,
    segment,
    batch_idx,
    head,
    group,
    offs_p,
    offs_n,
    TOKEN: tl.constexpr,
    COMPUTE_C: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    d_inner: tl.constexpr,
    conv_dim: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    stride_payload_width: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    CP_SIZE: tl.constexpr,
):
    p_mask = offs_p < headdim
    n_mask = offs_n < d_state
    x_channel = head * headdim + offs_p
    b_channel = d_inner + group * d_state + offs_n
    c_channel = d_inner + ngroups * d_state + group * d_state + offs_n
    segment_base = _rank_slot_payload_base(
        segment,
        batch_idx,
        stride_payload_rank,
        stride_payload_slot,
        stride_payload_batch,
        CP_SIZE,
    )
    predecessor = tl.maximum(segment - 1, 0)
    predecessor_base = _rank_slot_payload_base(
        predecessor,
        batch_idx,
        stride_payload_rank,
        stride_payload_slot,
        stride_payload_batch,
        CP_SIZE,
    )

    acc_x = tl.zeros((offs_p.shape[0],), dtype=tl.float32)
    acc_b = tl.zeros((offs_n.shape[0],), dtype=tl.float32)
    acc_c = tl.zeros((offs_n.shape[0],), dtype=tl.float32)
    for tap in tl.static_range(0, 4):
        source = TOKEN + tap - 3
        if source >= 0:
            source_base = (
                segment_base
                + (prefix_xbc_offset + source * conv_dim) * stride_payload_width
            )
            source_mask = True
        else:
            tail_pos = 3 + source
            source_base = (
                predecessor_base
                + (tail_xbc_offset + tail_pos * conv_dim) * stride_payload_width
            )
            source_mask = segment > 0

        raw_x = tl.load(
            payload_ptr + source_base + x_channel * stride_payload_width,
            mask=p_mask & source_mask,
            other=0.0,
        ).to(tl.float32)
        raw_b = tl.load(
            payload_ptr + source_base + b_channel * stride_payload_width,
            mask=n_mask & source_mask,
            other=0.0,
        ).to(tl.float32)
        weight_x = tl.load(
            weight_ptr + x_channel * stride_weight_d + tap * stride_weight_w,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)
        weight_b = tl.load(
            weight_ptr + b_channel * stride_weight_d + tap * stride_weight_w,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        acc_x += raw_x * weight_x
        acc_b += raw_b * weight_b
        if COMPUTE_C:
            raw_c = tl.load(
                payload_ptr + source_base + c_channel * stride_payload_width,
                mask=n_mask & source_mask,
                other=0.0,
            ).to(tl.float32)
            weight_c = tl.load(
                weight_ptr + c_channel * stride_weight_d + tap * stride_weight_w,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            acc_c += raw_c * weight_c

    acc_x += tl.load(bias_ptr + x_channel, mask=p_mask, other=0.0).to(tl.float32)
    acc_b += tl.load(bias_ptr + b_channel, mask=n_mask, other=0.0).to(tl.float32)
    acc_x = acc_x * tl.sigmoid(acc_x)
    acc_b = acc_b * tl.sigmoid(acc_b)
    # Match v2's convolution store followed by the scan's reload.
    x = acc_x.to(out_type_ptr.dtype.element_ty).to(tl.float32)
    b = acc_b.to(out_type_ptr.dtype.element_ty).to(tl.float32)
    if COMPUTE_C:
        acc_c += tl.load(bias_ptr + c_channel, mask=n_mask, other=0.0).to(tl.float32)
        acc_c = acc_c * tl.sigmoid(acc_c)
        c = acc_c.to(out_type_ptr.dtype.element_ty).to(tl.float32)
    else:
        c = tl.zeros((offs_n.shape[0],), dtype=tl.float32)
    return x, b, c


@triton.jit
def _prefix_state_replay_kernel(
    payload_ptr,
    weight_ptr,
    bias_ptr,
    dt_bias_ptr,
    a_ptr,
    d_ptr,
    z_ptr,
    out_ptr,
    body_initial_ptr,
    segment_initial_ptr,
    batch: tl.constexpr,
    segment_len: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    d_inner: tl.constexpr,
    conv_dim: tl.constexpr,
    body_log_offset: tl.constexpr,
    body_beta_offset: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    prefix_dt_offset: tl.constexpr,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    stride_payload_width: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_z_s: tl.constexpr,
    stride_z_l: tl.constexpr,
    stride_z_b: tl.constexpr,
    stride_z_h: tl.constexpr,
    stride_z_p: tl.constexpr,
    stride_out_s: tl.constexpr,
    stride_out_l: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_p: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_b: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_p: tl.constexpr,
    stride_state_n: tl.constexpr,
    stride_segment_state_s: tl.constexpr,
    stride_segment_state_b: tl.constexpr,
    stride_segment_state_h: tl.constexpr,
    stride_segment_state_p: tl.constexpr,
    stride_segment_state_n: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
    STORE_SEGMENT_INITIAL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    folded_batch = tl.program_id(1)
    stream = folded_batch // batch
    batch_idx = folded_batch - stream * batch
    target_segment = tl.where(stream == 0, CP_RANK, 2 * CP_SIZE - CP_RANK - 1)
    group = head // (nheads // ngroups)

    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    p_mask = offs_p < headdim
    n_mask = offs_n < d_state
    state_mask = p_mask[:, None] & n_mask[None, :]
    state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    a = tl.load(a_ptr + head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)

    # Replay complete earlier segments.  The runtime loop avoids unrolling a
    # separate P x N state update for every CP segment into the program.
    for segment in tl.range(0, target_segment, num_stages=1):
        segment_base = _rank_slot_payload_base(
            segment,
            batch_idx,
            stride_payload_rank,
            stride_payload_slot,
            stride_payload_batch,
            CP_SIZE,
        )
        for token in tl.static_range(0, 3):
            x, b, _ = _convolve_payload_prefix_token(
                payload_ptr,
                weight_ptr,
                bias_ptr,
                out_ptr,
                segment,
                batch_idx,
                head,
                group,
                offs_p,
                offs_n,
                token,
                False,
                nheads,
                headdim,
                ngroups,
                d_state,
                d_inner,
                conv_dim,
                prefix_xbc_offset,
                tail_xbc_offset,
                stride_payload_rank,
                stride_payload_slot,
                stride_payload_batch,
                stride_payload_width,
                stride_weight_d,
                stride_weight_w,
                CP_SIZE,
            )
            raw_dt = tl.load(
                payload_ptr
                + segment_base
                + (prefix_dt_offset + token * nheads + head) * stride_payload_width
            ).to(tl.float32)
            biased_dt = raw_dt + dt_bias
            dt = tl.where(biased_dt <= 20.0, softplus(biased_dt), biased_dt)
            state = tl.exp(dt * a) * state + dt * x[:, None] * b[None, :]

        body_log = tl.load(
            payload_ptr + segment_base + (body_log_offset + head) * stride_payload_width
        ).to(tl.float32)
        beta_offset = (
            body_beta_offset
            + head * headdim * d_state
            + offs_p[:, None] * d_state
            + offs_n[None, :]
        )
        beta = tl.load(
            payload_ptr + segment_base + beta_offset * stride_payload_width,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        state = tl.exp(body_log) * state + beta

    # The carried state now enters the current segment.  Training retains this
    # compact boundary for local dQ; no-grad/prefill skips the store entirely.
    if STORE_SEGMENT_INITIAL:
        tl.store(
            segment_initial_ptr
            + stream * stride_segment_state_s
            + batch_idx * stride_segment_state_b
            + head * stride_segment_state_h
            + offs_p[:, None] * stride_segment_state_p
            + offs_n[None, :] * stride_segment_state_n,
            state,
            mask=state_mask,
        )

    # Replay the current prefix and write its local outputs.
    target_base = _rank_slot_payload_base(
        target_segment,
        batch_idx,
        stride_payload_rank,
        stride_payload_slot,
        stride_payload_batch,
        CP_SIZE,
    )
    d_value = tl.load(d_ptr + head).to(tl.float32)
    for token in tl.static_range(0, 3):
        x, b, c = _convolve_payload_prefix_token(
            payload_ptr,
            weight_ptr,
            bias_ptr,
            out_ptr,
            target_segment,
            batch_idx,
            head,
            group,
            offs_p,
            offs_n,
            token,
            True,
            nheads,
            headdim,
            ngroups,
            d_state,
            d_inner,
            conv_dim,
            prefix_xbc_offset,
            tail_xbc_offset,
            stride_payload_rank,
            stride_payload_slot,
            stride_payload_batch,
            stride_payload_width,
            stride_weight_d,
            stride_weight_w,
            CP_SIZE,
        )
        raw_dt = tl.load(
            payload_ptr
            + target_base
            + (prefix_dt_offset + token * nheads + head) * stride_payload_width
        ).to(tl.float32)
        biased_dt = raw_dt + dt_bias
        dt = tl.where(biased_dt <= 20.0, softplus(biased_dt), biased_dt)
        state = tl.exp(dt * a) * state + dt * x[:, None] * b[None, :]
        pre_gate = tl.sum(state * c[None, :], axis=1) + d_value * x
        z = tl.load(
            z_ptr
            + stream * stride_z_s
            + token * stride_z_l
            + batch_idx * stride_z_b
            + head * stride_z_h
            + offs_p * stride_z_p,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)
        output = pre_gate * z * tl.sigmoid(z)
        tl.store(
            out_ptr
            + stream * stride_out_s
            + token * stride_out_l
            + batch_idx * stride_out_b
            + head * stride_out_h
            + offs_p * stride_out_p,
            output,
            mask=p_mask,
        )

    tl.store(
        body_initial_ptr
        + stream * stride_state_s
        + batch_idx * stride_state_b
        + head * stride_state_h
        + offs_p[:, None] * stride_state_p
        + offs_n[None, :] * stride_state_n,
        state,
        mask=state_mask,
    )


def _replay_prefixes_triton(
    gathered_payload: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    layout: _PayloadLayout,
    *,
    cp_rank: int,
    cp_size: int,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
    return_segment_initial_states: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _, streams, batch, payload_width = gathered_payload.shape
    nheads = d_inner_local_tp // headdim
    segment_len = z.shape[1]
    if streams != 2 or payload_width != layout.width:
        raise ValueError("gathered v3 payload has an invalid shape")
    initial_states = torch.empty(
        2,
        batch,
        nheads,
        headdim,
        d_state,
        device=out.device,
        dtype=torch.float32,
    )
    segment_initial_states = (
        torch.empty_like(initial_states) if return_segment_initial_states else None
    )
    segment_initial_arg = (
        segment_initial_states if segment_initial_states is not None else initial_states
    )
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(d_state))
    grid = (nheads, streams * batch)
    _prefix_state_replay_kernel[grid](
        gathered_payload,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        D,
        z,
        out,
        initial_states,
        segment_initial_arg,
        batch,
        segment_len,
        nheads,
        headdim,
        ngroups_local_tp,
        d_state,
        d_inner_local_tp,
        d_inner_local_tp + 2 * ngroups_local_tp * d_state,
        layout.body_log_alpha,
        layout.body_beta,
        layout.prefix_xbc,
        layout.tail_xbc,
        layout.prefix_dt,
        gathered_payload.stride(0),
        gathered_payload.stride(1),
        gathered_payload.stride(2),
        gathered_payload.stride(3),
        conv_weight.stride(0),
        conv_weight.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        z.stride(3),
        z.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        initial_states.stride(0),
        initial_states.stride(1),
        initial_states.stride(2),
        initial_states.stride(3),
        initial_states.stride(4),
        segment_initial_arg.stride(0),
        segment_initial_arg.stride(1),
        segment_initial_arg.stride(2),
        segment_initial_arg.stride(3),
        segment_initial_arg.stride(4),
        CP_RANK=cp_rank,
        CP_SIZE=cp_size,
        STORE_SEGMENT_INITIAL=return_segment_initial_states,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    return initial_states, segment_initial_states


def _mamba_true_context_parallel_v3_forward_impl(
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
    return_backward_cache: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_len = zxBCdt.shape[0] // 2
    batch = zxBCdt.shape[1]
    nheads = d_inner_local_tp // headdim
    conv_dim = d_inner_local_tp + 2 * ngroups_local_tp * d_state
    z_streams, xbc_streams, dt_streams = _split_projected_streams(
        zxBCdt,
        d_inner_local_tp=d_inner_local_tp,
        nheads_local_tp=nheads,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
    )
    z = _format_z_streams(z_streams, headdim=headdim)
    A = -torch.exp(A_log.float())
    body_len = segment_len - _PREFIX_TOKENS

    body_x: Optional[torch.Tensor]
    body_B: Optional[torch.Tensor]
    body_C: Optional[torch.Tensor]
    body_dt: Optional[torch.Tensor]
    body_dA: Optional[torch.Tensor]
    body_dA_rev: Optional[torch.Tensor]
    if body_len > 0:
        body_x, body_B, body_C = _causal_conv_body_width4_triton(
            xbc_streams,
            conv_weight,
            conv_bias,
            d_inner_local_tp=d_inner_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
        )
        body_dt, body_dA, body_dA_rev = compute_dacs_compact_triton(
            dt_streams[:, _PREFIX_TOKENS:], A, dt_bias.float(), chunk_size
        )
        folded_beta, folded_log = mamba2_get_final_state_triton(
            body_B, body_x, body_dt, body_dA, body_dA_rev, chunk_size
        )
        body_beta = folded_beta.view(2, batch, nheads, headdim, d_state)
        body_log = folded_log.view(2, batch, nheads)
    else:
        body_x = body_B = body_C = None
        body_dt = body_dA = body_dA_rev = None
        body_beta = torch.zeros(
            2,
            batch,
            nheads,
            headdim,
            d_state,
            device=zxBCdt.device,
            dtype=torch.float32,
        )
        body_log = torch.zeros(
            2, batch, nheads, device=zxBCdt.device, dtype=torch.float32
        )

    layout = _payload_layout(nheads, headdim, d_state, conv_dim)
    local_payload = _pack_forward_payload_triton(
        body_log, body_beta, xbc_streams, dt_streams, layout
    )
    with _record_function("mamba_true_cp_v3:fwd_combined_all_gather"):
        gathered_payload = _all_gather_rank_tensor(local_payload, cp_group)

    out = torch.empty(
        2,
        segment_len,
        batch,
        nheads,
        headdim,
        device=zxBCdt.device,
        dtype=zxBCdt.dtype,
    )
    body_initial_states, segment_initial_states = _replay_prefixes_triton(
        gathered_payload,
        conv_weight,
        conv_bias,
        dt_bias.float(),
        A,
        D,
        z,
        out,
        layout,
        cp_rank=cp_group.rank(),
        cp_size=cp_group.size(),
        d_inner_local_tp=d_inner_local_tp,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
        headdim=headdim,
        return_segment_initial_states=return_backward_cache,
    )

    if body_len > 0:
        mamba2_fwd_triton(
            body_C,
            body_B,
            body_x,
            body_dt,
            body_dA,
            body_dA_rev,
            body_initial_states,
            D,
            z[:, _PREFIX_TOKENS:],
            out=out[:, _PREFIX_TOKENS:],
            chunk_size=chunk_size,
        )
    public_out = out.view(2 * segment_len, batch, d_inner_local_tp)
    if not return_backward_cache:
        return public_out

    if segment_initial_states is None:
        raise RuntimeError("v3 training forward did not produce segment initial states")
    flat_payload = gathered_payload.reshape(
        2 * cp_group.size(), batch, gathered_payload.shape[-1]
    )
    conv_sources = _segment_indices(
        cp_group.rank(), cp_group.size(), gathered_payload.device
    )["conv_sources"]
    predecessor_payload = flat_payload.index_select(0, conv_sources)
    predecessor_halos = predecessor_payload[
        ..., layout.tail_xbc : layout.prefix_dt
    ].reshape(2, batch, _PREFIX_TOKENS, conv_dim)
    predecessor_halos = predecessor_halos.permute(0, 2, 1, 3).to(
        dtype=zxBCdt.dtype
    ).contiguous()
    if cp_group.rank() == 0:
        predecessor_halos[0].zero_()
    return public_out, segment_initial_states, predecessor_halos


def _pack_backward_payload(
    body_log_alpha: torch.Tensor,
    body_gamma: torch.Tensor,
    xbc_streams: torch.Tensor,
    dt_streams: torch.Tensor,
    dout: torch.Tensor,
    z: torch.Tensor,
    layout: _BackwardPayloadLayout,
) -> torch.Tensor:
    """Pack the full-replay backward operands in rank-slot/batch order."""

    streams, segment_len, batch, conv_dim = xbc_streams.shape
    _, _, nheads, headdim, d_state = body_gamma.shape
    if segment_len < _PREFIX_TOKENS:
        raise ValueError("v3 backward requires at least three tokens per segment")
    dpre = dout[:, :_PREFIX_TOKENS].float() * z[:, :_PREFIX_TOKENS].float()
    dpre *= torch.sigmoid(z[:, :_PREFIX_TOKENS].float())
    fields = (
        body_log_alpha.float(),
        body_gamma.float().flatten(2),
        xbc_streams[:, :_PREFIX_TOKENS]
        .permute(0, 2, 1, 3)
        .reshape(streams, batch, _PREFIX_TOKENS * conv_dim)
        .float(),
        xbc_streams[:, -_PREFIX_TOKENS:]
        .permute(0, 2, 1, 3)
        .reshape(streams, batch, _PREFIX_TOKENS * conv_dim)
        .float(),
        dt_streams[:, :_PREFIX_TOKENS]
        .permute(0, 2, 1, 3)
        .reshape(streams, batch, _PREFIX_TOKENS * nheads)
        .float(),
        dpre.permute(0, 2, 1, 3, 4)
        .reshape(streams, batch, _PREFIX_TOKENS * nheads * headdim),
    )
    payload = torch.cat(fields, dim=-1).contiguous()
    if payload.shape[-1] != layout.width:
        raise RuntimeError("packed v3 backward payload has an invalid width")
    return payload


@triton.jit
def _backward_boundary_replay_kernel(
    payload_ptr,
    weight_ptr,
    bias_ptr,
    dt_bias_ptr,
    a_ptr,
    out_type_ptr,
    local_terminal_ptr,
    successor_terminal_ptr,
    batch: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    d_inner: tl.constexpr,
    conv_dim: tl.constexpr,
    body_log_offset: tl.constexpr,
    body_gamma_offset: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    prefix_dt_offset: tl.constexpr,
    prefix_dpre_offset: tl.constexpr,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    stride_payload_width: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_local_s: tl.constexpr,
    stride_local_b: tl.constexpr,
    stride_local_h: tl.constexpr,
    stride_local_p: tl.constexpr,
    stride_local_n: tl.constexpr,
    stride_successor_s: tl.constexpr,
    stride_successor_b: tl.constexpr,
    stride_successor_h: tl.constexpr,
    stride_successor_p: tl.constexpr,
    stride_successor_n: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reverse-replay all global prefixes and emit only locally needed boundaries."""

    head = tl.program_id(0)
    batch_idx = tl.program_id(1)
    group = head // (nheads // ngroups)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    p_mask = offs_p < headdim
    n_mask = offs_n < d_state
    state_mask = p_mask[:, None] & n_mask[None, :]
    state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    a = tl.load(a_ptr + head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)
    local_segment_0: tl.constexpr = CP_RANK
    local_segment_1: tl.constexpr = 2 * CP_SIZE - CP_RANK - 1
    successor_0: tl.constexpr = CP_RANK + 1
    successor_1: tl.constexpr = 2 * CP_SIZE - CP_RANK

    for reverse_segment in tl.range(0, 2 * CP_SIZE, num_stages=1):
        segment = 2 * CP_SIZE - reverse_segment - 1

        local_base_0 = (
            batch_idx * stride_local_b
            + head * stride_local_h
            + offs_p[:, None] * stride_local_p
            + offs_n[None, :] * stride_local_n
        )
        tl.store(
            local_terminal_ptr + local_base_0,
            state,
            mask=state_mask & (segment == local_segment_0),
        )
        tl.store(
            local_terminal_ptr + stride_local_s + local_base_0,
            state,
            mask=state_mask & (segment == local_segment_1),
        )

        segment_base = _rank_slot_payload_base(
            segment,
            batch_idx,
            stride_payload_rank,
            stride_payload_slot,
            stride_payload_batch,
            CP_SIZE,
        )
        body_log = tl.load(
            payload_ptr
            + segment_base
            + (body_log_offset + head) * stride_payload_width
        ).to(tl.float32)
        gamma_offset = (
            body_gamma_offset
            + head * headdim * d_state
            + offs_p[:, None] * d_state
            + offs_n[None, :]
        )
        gamma = tl.load(
            payload_ptr + segment_base + gamma_offset * stride_payload_width,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        state = tl.exp(body_log) * state + gamma

        successor_base = (
            batch_idx * stride_successor_b
            + head * stride_successor_h
            + offs_p[:, None] * stride_successor_p
            + offs_n[None, :] * stride_successor_n
        )
        tl.store(
            successor_terminal_ptr + successor_base,
            state,
            mask=state_mask & (segment == successor_0),
        )
        tl.store(
            successor_terminal_ptr + stride_successor_s + successor_base,
            state,
            mask=state_mask & (segment == successor_1),
        )

        for reverse_token in tl.static_range(0, 3):
            token = 2 - reverse_token
            _, _, c = _convolve_payload_prefix_token(
                payload_ptr,
                weight_ptr,
                bias_ptr,
                out_type_ptr,
                segment,
                batch_idx,
                head,
                group,
                offs_p,
                offs_n,
                token,
                True,
                nheads,
                headdim,
                ngroups,
                d_state,
                d_inner,
                conv_dim,
                prefix_xbc_offset,
                tail_xbc_offset,
                stride_payload_rank,
                stride_payload_slot,
                stride_payload_batch,
                stride_payload_width,
                stride_weight_d,
                stride_weight_w,
                CP_SIZE,
            )
            raw_dt = tl.load(
                payload_ptr
                + segment_base
                + (prefix_dt_offset + token * nheads + head)
                * stride_payload_width
            ).to(tl.float32)
            biased_dt = raw_dt + dt_bias
            dt = tl.where(biased_dt <= 20.0, softplus(biased_dt), biased_dt)
            dpre = tl.load(
                payload_ptr
                + segment_base
                + (
                    prefix_dpre_offset
                    + token * d_inner
                    + head * headdim
                    + offs_p
                )
                * stride_payload_width,
                mask=p_mask,
                other=0.0,
            ).to(tl.float32)
            state = tl.exp(dt * a) * (state + dpre[:, None] * c[None, :])


def _backward_boundaries_triton(
    gathered_payload: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    out_type: torch.Tensor,
    layout: _BackwardPayloadLayout,
    *,
    cp_rank: int,
    cp_size: int,
    d_inner: int,
    ngroups: int,
    d_state: int,
    headdim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve local ``lambda_L`` and immediate-successor ``lambda_3`` in Triton."""

    _, streams, batch, width = gathered_payload.shape
    nheads = d_inner // headdim
    if streams != 2 or width != layout.width:
        raise ValueError("gathered v3 backward payload has an invalid shape")
    local_terminal = torch.empty(
        2,
        batch,
        nheads,
        headdim,
        d_state,
        device=gathered_payload.device,
        dtype=torch.float32,
    )
    successor_terminal = torch.empty_like(local_terminal)
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(d_state))
    _backward_boundary_replay_kernel[(nheads, batch)](
        gathered_payload,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        out_type,
        local_terminal,
        successor_terminal,
        batch,
        nheads,
        headdim,
        ngroups,
        d_state,
        d_inner,
        d_inner + 2 * ngroups * d_state,
        layout.body_log_alpha,
        layout.body_gamma,
        layout.prefix_xbc,
        layout.tail_xbc,
        layout.prefix_dt,
        layout.prefix_dpre,
        gathered_payload.stride(0),
        gathered_payload.stride(1),
        gathered_payload.stride(2),
        gathered_payload.stride(3),
        conv_weight.stride(0),
        conv_weight.stride(2),
        local_terminal.stride(0),
        local_terminal.stride(1),
        local_terminal.stride(2),
        local_terminal.stride(3),
        local_terminal.stride(4),
        successor_terminal.stride(0),
        successor_terminal.stride(1),
        successor_terminal.stride(2),
        successor_terminal.stride(3),
        successor_terminal.stride(4),
        CP_RANK=cp_rank,
        CP_SIZE=cp_size,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    return local_terminal, successor_terminal


@triton.jit
def _successor_prefix_replay_kernel(
    payload_ptr,
    weight_ptr,
    bias_ptr,
    dt_bias_ptr,
    a_ptr,
    d_ptr,
    out_type_ptr,
    predecessor_final_ptr,
    successor_terminal_ptr,
    dv_ptr,
    db_ptr,
    dc_ptr,
    batch: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    d_inner: tl.constexpr,
    conv_dim: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    prefix_dt_offset: tl.constexpr,
    prefix_dpre_offset: tl.constexpr,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    stride_payload_width: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_b: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_p: tl.constexpr,
    stride_state_n: tl.constexpr,
    stride_terminal_s: tl.constexpr,
    stride_terminal_b: tl.constexpr,
    stride_terminal_h: tl.constexpr,
    stride_terminal_p: tl.constexpr,
    stride_terminal_n: tl.constexpr,
    stride_dv_s: tl.constexpr,
    stride_dv_b: tl.constexpr,
    stride_dv_t: tl.constexpr,
    stride_dv_h: tl.constexpr,
    stride_dv_p: tl.constexpr,
    stride_db_s: tl.constexpr,
    stride_db_b: tl.constexpr,
    stride_db_t: tl.constexpr,
    stride_db_h: tl.constexpr,
    stride_db_n: tl.constexpr,
    stride_dc_s: tl.constexpr,
    stride_dc_b: tl.constexpr,
    stride_dc_t: tl.constexpr,
    stride_dc_h: tl.constexpr,
    stride_dc_n: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Replay two immediate successors and emit per-value-head post-conv grads."""

    head = tl.program_id(0)
    folded_batch = tl.program_id(1)
    stream = folded_batch // batch
    batch_idx = folded_batch - stream * batch
    local_segment = tl.where(stream == 0, CP_RANK, 2 * CP_SIZE - CP_RANK - 1)
    successor = tl.minimum(local_segment + 1, 2 * CP_SIZE - 1)
    valid_successor = local_segment < 2 * CP_SIZE - 1
    group = head // (nheads // ngroups)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    p_mask = offs_p < headdim
    n_mask = offs_n < d_state
    state_mask = p_mask[:, None] & n_mask[None, :]
    state_base = (
        stream * stride_state_s
        + batch_idx * stride_state_b
        + head * stride_state_h
        + offs_p[:, None] * stride_state_p
        + offs_n[None, :] * stride_state_n
    )
    state = tl.load(
        predecessor_final_ptr + state_base,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    a = tl.load(a_ptr + head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)
    d_value = tl.load(d_ptr + head).to(tl.float32)
    successor_base = _rank_slot_payload_base(
        successor,
        batch_idx,
        stride_payload_rank,
        stride_payload_slot,
        stride_payload_batch,
        CP_SIZE,
    )

    for token in tl.static_range(0, 3):
        x, b, c = _convolve_payload_prefix_token(
            payload_ptr,
            weight_ptr,
            bias_ptr,
            out_type_ptr,
            successor,
            batch_idx,
            head,
            group,
            offs_p,
            offs_n,
            token,
            True,
            nheads,
            headdim,
            ngroups,
            d_state,
            d_inner,
            conv_dim,
            prefix_xbc_offset,
            tail_xbc_offset,
            stride_payload_rank,
            stride_payload_slot,
            stride_payload_batch,
            stride_payload_width,
            stride_weight_d,
            stride_weight_w,
            CP_SIZE,
        )
        raw_dt = tl.load(
            payload_ptr
            + successor_base
            + (prefix_dt_offset + token * nheads + head) * stride_payload_width
        ).to(tl.float32)
        biased_dt = raw_dt + dt_bias
        dt = tl.where(biased_dt <= 20.0, softplus(biased_dt), biased_dt)
        decay = tl.exp(dt * a)
        dpre = tl.load(
            payload_ptr
            + successor_base
            + (
                prefix_dpre_offset
                + token * d_inner
                + head * headdim
                + offs_p
            )
            * stride_payload_width,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)
        state = decay * state + dt * x[:, None] * b[None, :]
        dc = tl.sum(state * dpre[:, None], axis=0)
        tl.store(
            dc_ptr
            + stream * stride_dc_s
            + batch_idx * stride_dc_b
            + token * stride_dc_t
            + head * stride_dc_h
            + offs_n * stride_dc_n,
            dc,
            mask=n_mask & valid_successor,
        )

    terminal_base = (
        stream * stride_terminal_s
        + batch_idx * stride_terminal_b
        + head * stride_terminal_h
        + offs_p[:, None] * stride_terminal_p
        + offs_n[None, :] * stride_terminal_n
    )
    adjoint = tl.load(
        successor_terminal_ptr + terminal_base,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    for reverse_token in tl.static_range(0, 3):
        token = 2 - reverse_token
        x, b, c = _convolve_payload_prefix_token(
            payload_ptr,
            weight_ptr,
            bias_ptr,
            out_type_ptr,
            successor,
            batch_idx,
            head,
            group,
            offs_p,
            offs_n,
            token,
            True,
            nheads,
            headdim,
            ngroups,
            d_state,
            d_inner,
            conv_dim,
            prefix_xbc_offset,
            tail_xbc_offset,
            stride_payload_rank,
            stride_payload_slot,
            stride_payload_batch,
            stride_payload_width,
            stride_weight_d,
            stride_weight_w,
            CP_SIZE,
        )
        raw_dt = tl.load(
            payload_ptr
            + successor_base
            + (prefix_dt_offset + token * nheads + head) * stride_payload_width
        ).to(tl.float32)
        biased_dt = raw_dt + dt_bias
        dt = tl.where(biased_dt <= 20.0, softplus(biased_dt), biased_dt)
        decay = tl.exp(dt * a)
        dpre = tl.load(
            payload_ptr
            + successor_base
            + (
                prefix_dpre_offset
                + token * d_inner
                + head * headdim
                + offs_p
            )
            * stride_payload_width,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)
        total = adjoint + dpre[:, None] * c[None, :]
        dv = dt * tl.sum(total * b[None, :], axis=1) + d_value * dpre
        db = dt * tl.sum(total * x[:, None], axis=0)
        tl.store(
            dv_ptr
            + stream * stride_dv_s
            + batch_idx * stride_dv_b
            + token * stride_dv_t
            + head * stride_dv_h
            + offs_p * stride_dv_p,
            dv,
            mask=p_mask & valid_successor,
        )
        tl.store(
            db_ptr
            + stream * stride_db_s
            + batch_idx * stride_db_b
            + token * stride_db_t
            + head * stride_db_h
            + offs_n * stride_db_n,
            db,
            mask=n_mask & valid_successor,
        )
        adjoint = decay * total


@triton.jit
def _payload_prefix_preact_channels(
    payload_ptr,
    weight_ptr,
    bias_ptr,
    segment,
    batch_idx,
    channels,
    channel_mask,
    TOKEN: tl.constexpr,
    conv_dim: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    stride_payload_width: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    CP_SIZE: tl.constexpr,
):
    segment_base = _rank_slot_payload_base(
        segment,
        batch_idx,
        stride_payload_rank,
        stride_payload_slot,
        stride_payload_batch,
        CP_SIZE,
    )
    predecessor_base = _rank_slot_payload_base(
        tl.maximum(segment - 1, 0),
        batch_idx,
        stride_payload_rank,
        stride_payload_slot,
        stride_payload_batch,
        CP_SIZE,
    )
    acc = tl.zeros((channels.shape[0],), dtype=tl.float32)
    for tap in tl.static_range(0, 4):
        source = TOKEN + tap - 3
        if source >= 0:
            source_base = (
                segment_base
                + (prefix_xbc_offset + source * conv_dim) * stride_payload_width
            )
            source_mask = channel_mask
        else:
            tail_pos = 3 + source
            source_base = (
                predecessor_base
                + (tail_xbc_offset + tail_pos * conv_dim) * stride_payload_width
            )
            source_mask = channel_mask & (segment > 0)
        raw = tl.load(
            payload_ptr + source_base + channels * stride_payload_width,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + channels * stride_weight_d + tap * stride_weight_w,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        acc += raw * weight
    return acc + tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )


@triton.jit
def _successor_halo_reduce_kernel(
    payload_ptr,
    weight_ptr,
    bias_ptr,
    dv_ptr,
    db_ptr,
    dc_ptr,
    halo_ptr,
    batch: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    ngroups: tl.constexpr,
    d_state: tl.constexpr,
    d_inner: tl.constexpr,
    conv_dim: tl.constexpr,
    prefix_xbc_offset: tl.constexpr,
    tail_xbc_offset: tl.constexpr,
    stride_payload_rank: tl.constexpr,
    stride_payload_slot: tl.constexpr,
    stride_payload_batch: tl.constexpr,
    stride_payload_width: tl.constexpr,
    stride_weight_d: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_dv_s: tl.constexpr,
    stride_dv_b: tl.constexpr,
    stride_dv_t: tl.constexpr,
    stride_dv_h: tl.constexpr,
    stride_dv_p: tl.constexpr,
    stride_db_s: tl.constexpr,
    stride_db_b: tl.constexpr,
    stride_db_t: tl.constexpr,
    stride_db_h: tl.constexpr,
    stride_db_n: tl.constexpr,
    stride_dc_s: tl.constexpr,
    stride_dc_b: tl.constexpr,
    stride_dc_t: tl.constexpr,
    stride_dc_h: tl.constexpr,
    stride_dc_n: tl.constexpr,
    stride_halo_s: tl.constexpr,
    stride_halo_t: tl.constexpr,
    stride_halo_b: tl.constexpr,
    stride_halo_d: tl.constexpr,
    CP_RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reduce grouped prefix grads, apply SiLU backward, and transpose conv."""

    folded_batch = tl.program_id(0)
    block = tl.program_id(1)
    stream = folded_batch // batch
    batch_idx = folded_batch - stream * batch
    channels = block * BLOCK_D + tl.arange(0, BLOCK_D)
    channel_mask = channels < conv_dim
    local_segment = tl.where(stream == 0, CP_RANK, 2 * CP_SIZE - CP_RANK - 1)
    successor = tl.minimum(local_segment + 1, 2 * CP_SIZE - 1)
    valid_successor = local_segment < 2 * CP_SIZE - 1
    qk_dim: tl.constexpr = ngroups * d_state

    is_x = channels < d_inner
    x_safe = tl.minimum(channels, d_inner - 1)
    x_head = x_safe // headdim
    x_p = x_safe - x_head * headdim
    b_rel = channels - d_inner
    is_b = (b_rel >= 0) & (b_rel < qk_dim)
    b_safe = tl.minimum(tl.maximum(b_rel, 0), qk_dim - 1)
    b_group = b_safe // d_state
    b_n = b_safe - b_group * d_state
    c_rel = channels - d_inner - qk_dim
    is_c = (c_rel >= 0) & (c_rel < qk_dim)
    c_safe = tl.minimum(tl.maximum(c_rel, 0), qk_dim - 1)
    c_group = c_safe // d_state
    c_n = c_safe - c_group * d_state

    grad_preact_0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    grad_preact_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    grad_preact_2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for token in tl.static_range(0, 3):
        dv = tl.load(
            dv_ptr
            + stream * stride_dv_s
            + batch_idx * stride_dv_b
            + token * stride_dv_t
            + x_head * stride_dv_h
            + x_p * stride_dv_p,
            mask=channel_mask & is_x & valid_successor,
            other=0.0,
        ).to(tl.float32)
        db = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for head_offset in tl.static_range(0, HEADS_PER_GROUP):
            b_head = b_group * HEADS_PER_GROUP + head_offset
            c_head = c_group * HEADS_PER_GROUP + head_offset
            db += tl.load(
                db_ptr
                + stream * stride_db_s
                + batch_idx * stride_db_b
                + token * stride_db_t
                + b_head * stride_db_h
                + b_n * stride_db_n,
                mask=channel_mask & is_b & valid_successor,
                other=0.0,
            ).to(tl.float32)
            dc += tl.load(
                dc_ptr
                + stream * stride_dc_s
                + batch_idx * stride_dc_b
                + token * stride_dc_t
                + c_head * stride_dc_h
                + c_n * stride_dc_n,
                mask=channel_mask & is_c & valid_successor,
                other=0.0,
            ).to(tl.float32)
        grad_post = tl.where(is_x, dv, tl.where(is_b, db, dc))
        preact = _payload_prefix_preact_channels(
            payload_ptr,
            weight_ptr,
            bias_ptr,
            successor,
            batch_idx,
            channels,
            channel_mask,
            token,
            conv_dim,
            prefix_xbc_offset,
            tail_xbc_offset,
            stride_payload_rank,
            stride_payload_slot,
            stride_payload_batch,
            stride_payload_width,
            stride_weight_d,
            stride_weight_w,
            CP_SIZE,
        )
        sigmoid = tl.sigmoid(preact)
        grad_value = grad_post * sigmoid * (1.0 + preact * (1.0 - sigmoid))
        if token == 0:
            grad_preact_0 = grad_value
        elif token == 1:
            grad_preact_1 = grad_value
        else:
            grad_preact_2 = grad_value

    w0 = tl.load(
        weight_ptr + channels * stride_weight_d,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(
        weight_ptr + channels * stride_weight_d + stride_weight_w,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    w2 = tl.load(
        weight_ptr + channels * stride_weight_d + 2 * stride_weight_w,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    halo0 = grad_preact_0 * w0
    halo1 = grad_preact_0 * w1 + grad_preact_1 * w0
    halo2 = grad_preact_0 * w2 + grad_preact_1 * w1 + grad_preact_2 * w0
    halo_base = (
        stream * stride_halo_s
        + batch_idx * stride_halo_b
        + channels * stride_halo_d
    )
    tl.store(
        halo_ptr + halo_base,
        halo0,
        mask=channel_mask & valid_successor,
    )
    tl.store(
        halo_ptr + halo_base + stride_halo_t,
        halo1,
        mask=channel_mask & valid_successor,
    )
    tl.store(
        halo_ptr + halo_base + 2 * stride_halo_t,
        halo2,
        mask=channel_mask & valid_successor,
    )


def _successor_halos_triton(
    gathered_payload: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    out_type: torch.Tensor,
    predecessor_final_states: torch.Tensor,
    successor_terminal_states: torch.Tensor,
    layout: _BackwardPayloadLayout,
    *,
    cp_rank: int,
    cp_size: int,
    d_inner: int,
    ngroups: int,
    d_state: int,
    headdim: int,
) -> torch.Tensor:
    """Fully replay immediate successors and return raw predecessor-tail grads."""

    batch = gathered_payload.shape[2]
    nheads = d_inner // headdim
    dv = torch.empty(
        2,
        batch,
        _PREFIX_TOKENS,
        nheads,
        headdim,
        device=gathered_payload.device,
        dtype=torch.float32,
    )
    db = torch.empty(
        2,
        batch,
        _PREFIX_TOKENS,
        nheads,
        d_state,
        device=gathered_payload.device,
        dtype=torch.float32,
    )
    dc = torch.empty_like(db)
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(d_state))
    _successor_prefix_replay_kernel[(nheads, 2 * batch)](
        gathered_payload,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        D,
        out_type,
        predecessor_final_states,
        successor_terminal_states,
        dv,
        db,
        dc,
        batch,
        nheads,
        headdim,
        ngroups,
        d_state,
        d_inner,
        d_inner + 2 * ngroups * d_state,
        layout.prefix_xbc,
        layout.tail_xbc,
        layout.prefix_dt,
        layout.prefix_dpre,
        gathered_payload.stride(0),
        gathered_payload.stride(1),
        gathered_payload.stride(2),
        gathered_payload.stride(3),
        conv_weight.stride(0),
        conv_weight.stride(2),
        predecessor_final_states.stride(0),
        predecessor_final_states.stride(1),
        predecessor_final_states.stride(2),
        predecessor_final_states.stride(3),
        predecessor_final_states.stride(4),
        successor_terminal_states.stride(0),
        successor_terminal_states.stride(1),
        successor_terminal_states.stride(2),
        successor_terminal_states.stride(3),
        successor_terminal_states.stride(4),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        dv.stride(4),
        db.stride(0),
        db.stride(1),
        db.stride(2),
        db.stride(3),
        db.stride(4),
        dc.stride(0),
        dc.stride(1),
        dc.stride(2),
        dc.stride(3),
        dc.stride(4),
        CP_RANK=cp_rank,
        CP_SIZE=cp_size,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )

    conv_dim = d_inner + 2 * ngroups * d_state
    halo = torch.zeros(
        2,
        _PREFIX_TOKENS,
        batch,
        conv_dim,
        device=gathered_payload.device,
        dtype=torch.float32,
    )
    block_d = 64
    _successor_halo_reduce_kernel[(2 * batch, triton.cdiv(conv_dim, block_d))](
        gathered_payload,
        conv_weight,
        conv_bias,
        dv,
        db,
        dc,
        halo,
        batch,
        nheads,
        headdim,
        ngroups,
        d_state,
        d_inner,
        conv_dim,
        layout.prefix_xbc,
        layout.tail_xbc,
        gathered_payload.stride(0),
        gathered_payload.stride(1),
        gathered_payload.stride(2),
        gathered_payload.stride(3),
        conv_weight.stride(0),
        conv_weight.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        dv.stride(4),
        db.stride(0),
        db.stride(1),
        db.stride(2),
        db.stride(3),
        db.stride(4),
        dc.stride(0),
        dc.stride(1),
        dc.stride(2),
        dc.stride(3),
        dc.stride(4),
        halo.stride(0),
        halo.stride(1),
        halo.stride(2),
        halo.stride(3),
        CP_RANK=cp_rank,
        CP_SIZE=cp_size,
        HEADS_PER_GROUP=nheads // ngroups,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    del dv, db, dc
    return halo


def _subsequence_state_passing_v3(
    summaries: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    folded_batch, nsubsections, nheads, headdim, dstate = summaries.shape
    incoming, final = _state_passing_fwd(
        summaries.view(folded_batch, nsubsections, nheads, headdim * dstate),
        log_alpha,
        initial_states=initial_states.view(folded_batch, nheads, headdim * dstate),
        seq_idx=None,
        chunk_size=1,
        out_dtype=torch.float32,
    )
    return (
        incoming.view(folded_batch, nsubsections, nheads, headdim, dstate),
        final.view(folded_batch, nheads, headdim, dstate),
    )


def _mamba_true_context_parallel_v3_backward_impl(
    grad_output: torch.Tensor,
    zxBCdt: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    segment_initial_states: torch.Tensor,
    predecessor_halos: torch.Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup,
    d_inner_local_tp: int,
    ngroups_local_tp: int,
    d_state: int,
    headdim: int,
    chunk_size: int,
    bwd_subsequence_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one-all-gather full-replay CP-v3 backward."""

    segment_len = zxBCdt.shape[0] // 2
    batch = zxBCdt.shape[1]
    nheads = d_inner_local_tp // headdim
    conv_dim = d_inner_local_tp + 2 * ngroups_local_tp * d_state
    z_streams, xbc_streams, dt_streams = _split_projected_streams(
        zxBCdt,
        d_inner_local_tp=d_inner_local_tp,
        nheads_local_tp=nheads,
        ngroups_local_tp=ngroups_local_tp,
        d_state=d_state,
    )
    z = _format_z_streams(z_streams, headdim=headdim)
    dout = _format_grad_output_streams(grad_output, headdim=headdim)
    A = -torch.exp(A_log.float())

    # Cached predecessor halos make this complete local convolution independent
    # of communication.  Body views feed the compact pre-gather gamma summary.
    with _record_function("mamba_true_cp_v3:bwd_pre_gather_recompute"):
        x, B, C = _causal_conv_format_streams_triton(
            xbc_streams,
            predecessor_halos,
            conv_weight,
            conv_bias,
            d_inner_local_tp=d_inner_local_tp,
            ngroups_local_tp=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
        )
        body_len = segment_len - _PREFIX_TOKENS
        if body_len > 0:
            body_dt, body_dA, body_dA_rev = compute_dacs_compact_triton(
                dt_streams[:, _PREFIX_TOKENS:], A, dt_bias.float(), chunk_size
            )
            folded_gamma = mamba2_get_initial_dstate_triton(
                C[:, _PREFIX_TOKENS:],
                dout[:, _PREFIX_TOKENS:],
                z[:, _PREFIX_TOKENS:],
                body_dA,
                chunk_size=chunk_size,
            )
            body_gamma = folded_gamma.view(
                2, batch, nheads, headdim, d_state
            )
            body_log_alpha = body_dA[..., -1].sum(dim=-1).view(
                2, batch, nheads
            )
            del body_dt, body_dA, body_dA_rev, folded_gamma
        else:
            body_gamma = torch.zeros(
                2,
                batch,
                nheads,
                headdim,
                d_state,
                device=zxBCdt.device,
                dtype=torch.float32,
            )
            body_log_alpha = torch.zeros(
                2, batch, nheads, device=zxBCdt.device, dtype=torch.float32
            )
        layout = _backward_payload_layout(
            nheads, headdim, d_state, conv_dim
        )
        local_payload = _pack_backward_payload(
            body_log_alpha,
            body_gamma,
            xbc_streams,
            dt_streams,
            dout,
            z,
            layout,
        )

    with _record_function("mamba_true_cp_v3:bwd_combined_all_gather"):
        gathered_payload = _all_gather_rank_tensor(local_payload, cp_group)
    del local_payload, body_log_alpha, body_gamma

    with _record_function("mamba_true_cp_v3:bwd_boundary_replay"):
        local_dSL, successor_dS3 = _backward_boundaries_triton(
            gathered_payload,
            conv_weight,
            conv_bias,
            dt_bias,
            A,
            zxBCdt,
            layout,
            cp_rank=cp_group.rank(),
            cp_size=cp_group.size(),
            d_inner=d_inner_local_tp,
            ngroups=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
        )

    # Full compact preparation is intentionally after the collective.  It
    # feeds both ordinary local backward and the S0 -> SL recomputation.
    dt_out, dA_cumsum, dA_cs_rev = compute_dacs_compact_triton(
        dt_streams, A, dt_bias.float(), chunk_size
    )
    nchunks = (segment_len + chunk_size - 1) // chunk_size
    subsection_chunks = (
        bwd_subsequence_size // chunk_size
        if chunk_size == 64 and bwd_subsequence_size > 0
        else 0
    )
    nsubsections = (
        (nchunks + subsection_chunks - 1) // subsection_chunks
        if subsection_chunks
        else 1
    )
    use_subsequence = subsection_chunks > 0 and nsubsections > 1
    folded_initial = segment_initial_states.view(
        2 * batch, nheads, headdim, d_state
    )
    folded_terminal = local_dSL.view(2 * batch, nheads, headdim, d_state)

    subsection_beta = None
    subsection_gamma = None
    if use_subsequence:
        subsection_beta_raw, subsection_log_alpha = (
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
        subsection_beta, folded_final = _subsequence_state_passing_v3(
            subsection_beta_raw, subsection_log_alpha, folded_initial
        )
        predecessor_final_states = folded_final.view(
            2, batch, nheads, headdim, d_state
        )
        subsection_gamma_raw = mamba2_get_subsequence_dstates_triton(
            C,
            dout,
            z,
            dA_cumsum,
            chunk_size=chunk_size,
            subsequence_size=bwd_subsequence_size,
        )
        subsection_gamma, _ = _subsequence_state_passing_v3(
            subsection_gamma_raw,
            subsection_log_alpha.flip(-1).contiguous(),
            folded_terminal,
        )
        del subsection_beta_raw, subsection_gamma_raw, subsection_log_alpha, folded_final
    else:
        folded_beta, folded_log = mamba2_get_final_state_triton(
            B, x, dt_out, dA_cumsum, dA_cs_rev, chunk_size
        )
        predecessor_final_states = (
            torch.exp(folded_log)[..., None, None] * folded_initial + folded_beta
        ).view(2, batch, nheads, headdim, d_state)
        del folded_beta, folded_log

    with _record_function("mamba_true_cp_v3:bwd_successor_replay"):
        successor_halos = _successor_halos_triton(
            gathered_payload,
            conv_weight,
            conv_bias,
            dt_bias,
            A,
            D,
            zxBCdt,
            predecessor_final_states,
            successor_dS3,
            layout,
            cp_rank=cp_group.rank(),
            cp_size=cp_group.size(),
            d_inner=d_inner_local_tp,
            ngroups=ngroups_local_tp,
            d_state=d_state,
            headdim=headdim,
        )
    del gathered_payload, predecessor_final_states, successor_dS3

    with _record_function("mamba_true_cp_v3:bwd_dq"):
        dC, dz, chunk_start_states, processed_ddt, dA_chunk = (
            mamba2_bwd_dq_cp_triton(
                C,
                B,
                x,
                dout,
                z,
                dt_out,
                dA_cumsum,
                dA_cs_rev,
                subsection_beta if use_subsequence else segment_initial_states,
                A,
                D,
                chunk_size=chunk_size,
                subsequence_size=bwd_subsequence_size if use_subsequence else 0,
            )
        )
    if use_subsequence:
        del subsection_beta

    with _record_function("mamba_true_cp_v3:bwd_dk_dv"):
        dB, dx, grad_D, processed_ddt, dA_chunk = mamba2_bwd_dk_dv_cp_triton(
            C,
            B,
            x,
            dout,
            z,
            dt_out,
            dA_cumsum,
            dA_cs_rev,
            chunk_start_states,
            subsection_gamma if use_subsequence else local_dSL,
            A,
            D,
            processed_ddt,
            dA_chunk,
            chunk_size=chunk_size,
            subsequence_size=bwd_subsequence_size if use_subsequence else 0,
        )
    if use_subsequence:
        del subsection_gamma
    del C, B, x, z, dout, dA_cumsum, dA_cs_rev, chunk_start_states

    with _record_function("mamba_true_cp_v3:bwd_decay"):
        grad_dt_streams, grad_A_log, grad_dt_bias = mamba2_bwd_decay_map_cp_triton(
            dt_streams,
            processed_ddt,
            dA_chunk,
            A,
            dt_bias,
            chunk_size=chunk_size,
        )
    del A, dt_out, processed_ddt, dA_chunk

    with _record_function("mamba_true_cp_v3:bwd_conv"):
        grad_xBC_streams, discarded_halos, grad_conv_weight, grad_conv_bias = (
            _causal_conv_format_streams_backward_triton(
                xbc_streams,
                predecessor_halos,
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
    del discarded_halos, dx, dB, dC
    grad_xBC_streams[:, -_PREFIX_TOKENS:].add_(
        successor_halos.to(dtype=grad_xBC_streams.dtype)
    )
    grad_streams = torch.cat(
        (dz.flatten(-2), grad_xBC_streams, grad_dt_streams), dim=-1
    )
    return (
        grad_streams.view_as(zxBCdt),
        grad_conv_weight,
        grad_conv_bias,
        grad_dt_bias,
        grad_A_log,
        grad_D,
    )


class _MambaTrueContextParallelV3Fn(torch.autograd.Function):
    """Checkpointed autograd boundary for the additive CP-v3 experiment."""

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
        save_backward_cache,
    ):
        ctx.cp_group = cp_group
        ctx.d_inner_local_tp = d_inner_local_tp
        ctx.ngroups_local_tp = ngroups_local_tp
        ctx.d_state = d_state
        ctx.headdim = headdim
        ctx.chunk_size = chunk_size
        ctx.bwd_subsequence_size = bwd_subsequence_size
        ctx.save_backward_cache = save_backward_cache
        result = _mamba_true_context_parallel_v3_forward_impl(
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
            return_backward_cache=save_backward_cache,
        )
        if save_backward_cache:
            out, segment_initial_states, predecessor_halos = result
            ctx.save_for_backward(
                zxBCdt,
                conv_weight,
                conv_bias,
                dt_bias,
                A_log,
                D,
                segment_initial_states,
                predecessor_halos,
            )
            return out
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.save_backward_cache:
            raise RuntimeError(
                "CP-v3 backward requires the automatic training boundary cache"
            )
        (
            zxBCdt,
            conv_weight,
            conv_bias,
            dt_bias,
            A_log,
            D,
            segment_initial_states,
            predecessor_halos,
        ) = ctx.saved_tensors
        grads = _mamba_true_context_parallel_v3_backward_impl(
            grad_output,
            zxBCdt,
            conv_weight,
            conv_bias,
            dt_bias,
            A_log,
            D,
            segment_initial_states,
            predecessor_halos,
            cp_group=ctx.cp_group,
            d_inner_local_tp=ctx.d_inner_local_tp,
            ngroups_local_tp=ctx.ngroups_local_tp,
            d_state=ctx.d_state,
            headdim=ctx.headdim,
            chunk_size=ctx.chunk_size,
            bwd_subsequence_size=ctx.bwd_subsequence_size,
        )
        return (*grads, None, None, None, None, None, None, None, None)


def mamba_true_context_parallel_v3_training(
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
    bwd_subsequence_size: int = 64,
) -> torch.Tensor:
    """Evaluate the one-all-gather-per-pass CP-v3 training prototype."""

    if conv_bias is None:
        raise ValueError("Mamba true CP v3 requires convolution bias")
    if activation != "silu":
        raise ValueError("Mamba true CP v3 requires activation='silu'")
    if D_has_hdim:
        raise ValueError("Mamba true CP v3 requires D_has_hdim=False")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if bwd_subsequence_size < 0 or (
        bwd_subsequence_size and bwd_subsequence_size % 64
    ):
        raise ValueError("bwd_subsequence_size must be zero or a positive multiple of 64")
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
    # Resolve outside autograd.Function.forward, where PyTorch disables grad
    # mode even for a training invocation.
    save_backward_cache = torch.is_grad_enabled()
    return _MambaTrueContextParallelV3Fn.apply(
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
        save_backward_cache,
    )
