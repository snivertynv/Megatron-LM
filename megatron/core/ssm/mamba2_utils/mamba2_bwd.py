"""
Copyright (c) 2026, Dao AI Lab, Goombalab


Mamba2 bwd implementation in Triton.

In is consisted of two main kernels:
- dQ kernel: save States
- dKdV kernel: load States, and also computes ddt/dA involved
    in chunked updates: dM_S, dM_Prev,
                        dM_next, and dM_Curr
"""

import math
from typing import Type, Callable, Optional, List
from functools import partial

import torch

# from mamba2_fwd import mamba2_fwd_triton

import triton
import triton.language as tl


def _prune_dk_dv_compact_configs(configs, nargs, **kwargs):
    del nargs
    if kwargs.get("SUBSEQUENCE_CHUNKS", 0) > 0:
        return [
            config
            for config in configs
            if config.kwargs["BLOCK_M"] == 64
            and config.kwargs["BLOCK_K"] == 64
            and config.num_warps == 8
            and config.num_stages == 1
        ]
    return configs


@triton.autotune(
    configs=[
        # Small adjoint matrices can benefit from two warps, while wider
        # matrices need more parallelism. The one-stage variants cap resource
        # pressure and the two-stage variants can overlap the streamed loads
        # with the dot product.
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    # Sequence length only changes the number of reverse recurrence
    # iterations, so it should not create another autotune cache entry.
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "BLOCK_HEADDIM_V"],
)
@triton.jit
def _mamba2_get_initial_dstate_kernel(
    C,
    DOUT,
    Z,
    DA_CS,
    GAMMA,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_gamma_batch,
    stride_gamma_head,
    stride_gamma_v_dim,
    stride_gamma_qk_dim,
    seqlen,
    headdim_qk,
    headdim_v,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
):
    """Compute the zero-terminal additive adjoint summary of one segment."""
    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)

    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    c_offset = pid_folded_batch * stride_c_batch + qk_head * stride_c_head
    dout_offset = (
        stream * stride_dout_stream
        + batch_idx * stride_dout_batch
        + pid_head * stride_dout_head
    )
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    dacs_offset = pid_folded_batch * stride_dacs_batch + pid_head * stride_dacs_head

    offs_chunk = tl.arange(0, BLOCK_CHUNK)
    offs_qk_dim = tl.arange(0, BLOCK_HEADDIM_QK)
    offs_v_dim = tl.arange(0, BLOCK_HEADDIM_V)
    acc_dstates = tl.zeros([BLOCK_HEADDIM_QK, BLOCK_HEADDIM_V], dtype=tl.float32)
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx_rev in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_rev
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seq = chunk_start + offs_chunk
        logical_chunk_mask = offs_chunk < CHUNK_SIZE
        seq_mask = logical_chunk_mask & (offs_seq < seqlen)

        chunk_dacs_offset = dacs_offset + chunk_idx * stride_dacs_chunk
        dA_cs = tl.load(
            DA_CS + chunk_dacs_offset + offs_chunk * stride_dacs_csize,
            mask=logical_chunk_mask,
            other=0.0,
        ).to(tl.float32)
        dA_cs_last = tl.load(
            DA_CS + chunk_dacs_offset + (CHUNK_SIZE - 1) * stride_dacs_csize
        ).to(tl.float32)

        c_ptrs = (
            C
            + c_offset
            + offs_seq[:, None] * stride_c_seqlen
            + offs_qk_dim[None, :] * stride_c_dim
        )
        c_mask = seq_mask[:, None] & (offs_qk_dim[None, :] < headdim_qk)
        c_block = tl.load(c_ptrs, mask=c_mask, other=0.0)

        dout_ptrs = (
            DOUT
            + dout_offset
            + offs_seq[:, None] * stride_dout_seqlen
            + offs_v_dim[None, :] * stride_dout_dim
        )
        dout_mask = seq_mask[:, None] & (offs_v_dim[None, :] < headdim_v)
        dout_block = tl.load(dout_ptrs, mask=dout_mask, other=0.0).to(tl.float32)

        z_ptrs = (
            Z
            + z_offset
            + offs_seq[:, None] * stride_z_seqlen
            + offs_v_dim[None, :] * stride_z_dim
        )
        z_block = tl.load(z_ptrs, mask=dout_mask, other=0.0).to(tl.float32)
        dpre = dout_block * z_block * tl.sigmoid(z_block)
        dpre *= tl.exp(dA_cs)[:, None]

        gamma_chunk = tl.dot(tl.trans(c_block), dpre.to(c_block.dtype))
        acc_dstates = tl.exp(dA_cs_last) * acc_dstates + gamma_chunk

    gamma_offset = (
        pid_folded_batch * stride_gamma_batch + pid_head * stride_gamma_head
    )
    gamma_ptrs = (
        GAMMA
        + gamma_offset
        + offs_v_dim[:, None] * stride_gamma_v_dim
        + offs_qk_dim[None, :] * stride_gamma_qk_dim
    )
    gamma_mask = (offs_v_dim[:, None] < headdim_v) & (
        offs_qk_dim[None, :] < headdim_qk
    )
    tl.store(gamma_ptrs, tl.trans(acc_dstates), mask=gamma_mask)


def mamba2_get_initial_dstate_triton(
    C: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, headdim_qk)
    dout: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim_v)
    z: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim_v)
    dA_cs: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return each segment's zero-terminal additive adjoint summary ``gamma``.

    The returned FP32 tensor has shape
    ``[folded_batch, nheads, headdim_v, headdim_qk]`` and represents
    ``dstate_in = exp(log_alpha) * dstate_out + gamma``. This function computes
    only ``gamma``; the context-parallel reverse affine scan supplies the actual
    per-segment ``dstate_out`` boundary in a later stage.
    """
    if C.ndim != 4:
        raise ValueError("C must be 4D")
    if dout.ndim != 5 or z.ndim != 5:
        raise ValueError("dout and z must be 5D stream-major tensors")
    if dA_cs.ndim != 4:
        raise ValueError("dA_cs must be 4D")
    if dout.shape != z.shape:
        raise ValueError("dout and z must have the same shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    folded_batch, seqlen, nheads_qk, headdim_qk = C.shape
    streams, dout_seqlen, batch, nheads, headdim_v = dout.shape
    if streams != 2:
        raise ValueError("true-CP adjoint summaries require exactly two streams")
    if dout_seqlen != seqlen:
        raise ValueError("C, dout, and z must have the same sequence length")
    if folded_batch != streams * batch:
        raise ValueError("C folded batch must equal streams * batch")
    if nheads_qk <= 0 or nheads <= 0 or nheads % nheads_qk != 0:
        raise ValueError("the number of value heads must be divisible by C heads")
    if headdim_qk <= 0 or headdim_v <= 0:
        raise ValueError("head dimensions must be positive")

    nchunks = triton.cdiv(seqlen, chunk_size)
    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    if dA_cs.shape != decay_shape:
        raise ValueError(f"dA_cs must have shape {decay_shape}, got {tuple(dA_cs.shape)}")

    tensors = (C, dout, z, dA_cs)
    if not C.is_cuda:
        raise ValueError("mamba2_get_initial_dstate_triton requires CUDA tensors")
    if any(tensor.device != C.device for tensor in tensors[1:]):
        raise ValueError("C, dout, z, and dA_cs must be on the same device")
    if C.dtype != dout.dtype or C.dtype != z.dtype:
        raise ValueError("C, dout, and z must have the same dtype")
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("C, dout, z, and dA_cs must use floating-point dtypes")

    gamma = torch.empty(
        folded_batch,
        nheads,
        headdim_v,
        headdim_qk,
        device=C.device,
        dtype=torch.float32,
    )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_headdim_qk = max(16, triton.next_power_of_2(headdim_qk))
    block_headdim_v = max(16, triton.next_power_of_2(headdim_v))
    grid = (nheads, folded_batch)
    _mamba2_get_initial_dstate_kernel[grid](
        C,
        dout,
        z,
        dA_cs,
        gamma,
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        dout.stride(4),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        z.stride(3),
        z.stride(4),
        dA_cs.stride(0),
        dA_cs.stride(1),
        dA_cs.stride(2),
        dA_cs.stride(3),
        gamma.stride(0),
        gamma.stride(1),
        gamma.stride(2),
        gamma.stride(3),
        seqlen,
        headdim_qk,
        headdim_v,
        nheads_qk,
        BATCH=batch,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
        BLOCK_HEADDIM_QK=block_headdim_qk,
        BLOCK_HEADDIM_V=block_headdim_v,
    )
    return gamma


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=[
        "CHUNK_SIZE", "SUBSEQUENCE_CHUNKS", "BLOCK_P", "BLOCK_N",
        "FULL_SUBSEQUENCE", "EXACT_STATE",
    ],
)
@triton.jit
def _mamba2_get_subsequence_state_kernel(
    B,
    X,
    DT,
    DA_CS,
    DA_CS_REV,
    BETA,
    LOG_ALPHA,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_chunk,
    stride_dacsrev_csize,
    stride_beta_batch,
    stride_beta_subsequence,
    stride_beta_head,
    stride_beta_p,
    stride_beta_n,
    stride_log_batch,
    stride_log_head,
    stride_log_subsequence,
    seqlen,
    nchunks,
    headdim_p,
    dstate_n,
    nheads_qk,
    CHUNK_SIZE: tl.constexpr,
    SUBSEQUENCE_CHUNKS: tl.constexpr,
    SUBSEQUENCE_OFFSET: tl.constexpr,
    FULL_SUBSEQUENCE: tl.constexpr,
    EXACT_STATE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute zero-input affine summaries for local subsequences."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_subsequence = tl.program_id(2) + SUBSEQUENCE_OFFSET
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    first_chunk = pid_subsequence * SUBSEQUENCE_CHUNKS

    offs_chunk = tl.arange(0, BLOCK_CHUNK)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    acc_states = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    log_alpha = 0.0

    for local_chunk in range(SUBSEQUENCE_CHUNKS):
        chunk_idx = first_chunk + local_chunk
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seq = chunk_start + offs_chunk
        logical = offs_chunk < CHUNK_SIZE
        if FULL_SUBSEQUENCE:
            valid_chunk = True
            valid = logical
        else:
            valid_chunk = chunk_idx < nchunks
            valid = valid_chunk & logical & (offs_seq < seqlen)
        dt_base = (
            pid_batch * stride_dt_batch
            + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
        )
        dacs_base = (
            pid_batch * stride_dacs_batch
            + pid_head * stride_dacs_head
            + chunk_idx * stride_dacs_chunk
        )
        dacsrev_base = (
            pid_batch * stride_dacsrev_batch
            + pid_head * stride_dacsrev_head
            + chunk_idx * stride_dacsrev_chunk
        )
        if FULL_SUBSEQUENCE and BLOCK_CHUNK == CHUNK_SIZE:
            dt_block = tl.load(
                DT + dt_base + offs_chunk * stride_dt_csize
            ).to(tl.float32)
            dA_cs_rev = tl.load(
                DA_CS_REV + dacsrev_base + offs_chunk * stride_dacsrev_csize
            ).to(tl.float32)
            dA_cs_last = tl.load(
                DA_CS + dacs_base + (CHUNK_SIZE - 1) * stride_dacs_csize
            ).to(tl.float32)
        else:
            dt_block = tl.load(
                DT + dt_base + offs_chunk * stride_dt_csize,
                mask=valid_chunk & logical,
                other=0.0,
            ).to(tl.float32)
            dA_cs_rev = tl.load(
                DA_CS_REV + dacsrev_base + offs_chunk * stride_dacsrev_csize,
                mask=valid_chunk & logical,
                other=0.0,
            ).to(tl.float32)
            dA_cs_last = tl.load(
                DA_CS + dacs_base + (CHUNK_SIZE - 1) * stride_dacs_csize,
                mask=valid_chunk,
                other=0.0,
            ).to(tl.float32)
        log_alpha += dA_cs_last
        b_ptrs = (
            B + pid_batch * stride_b_batch + qk_head * stride_b_head
            + offs_seq[:, None] * stride_b_seqlen
            + offs_n[None, :] * stride_b_dim
        )
        x_ptrs = (
            X + pid_batch * stride_x_batch + pid_head * stride_x_head
            + offs_seq[:, None] * stride_x_seqlen
            + offs_p[None, :] * stride_x_dim
        )
        if (FULL_SUBSEQUENCE and EXACT_STATE) and BLOCK_CHUNK == CHUNK_SIZE:
            b_block = tl.load(b_ptrs)
            x_block = tl.load(x_ptrs)
        elif FULL_SUBSEQUENCE and BLOCK_CHUNK == CHUNK_SIZE:
            b_block = tl.load(
                b_ptrs,
                mask=offs_n[None, :] < dstate_n,
                other=0.0,
            )
            x_block = tl.load(
                x_ptrs,
                mask=offs_p[None, :] < headdim_p,
                other=0.0,
            )
        else:
            b_block = tl.load(
                b_ptrs,
                mask=valid[:, None] & (offs_n[None, :] < dstate_n),
                other=0.0,
            )
            x_block = tl.load(
                x_ptrs,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p),
                other=0.0,
            )
        state_scale = tl.exp(dA_cs_rev) * dt_block
        b_scaled = b_block * state_scale[:, None].to(b_block.dtype)
        acc_states = tl.exp(dA_cs_last) * acc_states + tl.dot(
            tl.trans(x_block).to(b_block.dtype), b_scaled
        )

    beta_ptrs = (
        BETA + pid_batch * stride_beta_batch
        + pid_subsequence * stride_beta_subsequence
        + pid_head * stride_beta_head + offs_p[:, None] * stride_beta_p
        + offs_n[None, :] * stride_beta_n
    )
    if FULL_SUBSEQUENCE and EXACT_STATE:
        tl.store(beta_ptrs, acc_states)
    else:
        tl.store(
            beta_ptrs,
            acc_states,
            mask=(offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n),
        )
    tl.store(
        LOG_ALPHA
        + pid_batch * stride_log_batch
        + pid_head * stride_log_head
        + pid_subsequence * stride_log_subsequence,
        log_alpha,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=[
        "CHUNK_SIZE", "SUBSEQUENCE_CHUNKS", "BLOCK_P", "BLOCK_N",
        "FULL_SUBSEQUENCE", "EXACT_STATE",
    ],
)
@triton.jit
def _mamba2_get_subsequence_dstate_kernel(
    C,
    DOUT,
    Z,
    DA_CS,
    GAMMA_REV,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_gamma_batch,
    stride_gamma_subsequence,
    stride_gamma_head,
    stride_gamma_p,
    stride_gamma_n,
    seqlen,
    nchunks,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SUBSEQUENCE_CHUNKS: tl.constexpr,
    REVERSE_SLOT_OFFSET: tl.constexpr,
    ORIGINAL_SUBSEQUENCE_BASE: tl.constexpr,
    ORIGINAL_SUBSEQUENCE_COUNT: tl.constexpr,
    FULL_SUBSEQUENCE: tl.constexpr,
    EXACT_STATE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute zero-terminal adjoint summaries in reverse subsequence order."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_subsequence_rev_local = tl.program_id(2)
    pid_subsequence_rev = pid_subsequence_rev_local + REVERSE_SLOT_OFFSET
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_batch // BATCH
    batch_idx = pid_batch - stream * BATCH
    original_subsequence = (
        ORIGINAL_SUBSEQUENCE_BASE
        + ORIGINAL_SUBSEQUENCE_COUNT - 1 - pid_subsequence_rev_local
    )
    first_chunk = original_subsequence * SUBSEQUENCE_CHUNKS

    offs_chunk = tl.arange(0, BLOCK_CHUNK)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    acc_dstates = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)

    for local_chunk_rev in range(SUBSEQUENCE_CHUNKS):
        chunk_idx = first_chunk + SUBSEQUENCE_CHUNKS - 1 - local_chunk_rev
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seq = chunk_start + offs_chunk
        logical = offs_chunk < CHUNK_SIZE
        if FULL_SUBSEQUENCE:
            valid_chunk = True
            valid = logical
        else:
            valid_chunk = chunk_idx < nchunks
            valid = valid_chunk & logical & (offs_seq < seqlen)
        dacs_base = (
            pid_batch * stride_dacs_batch
            + pid_head * stride_dacs_head
            + chunk_idx * stride_dacs_chunk
        )
        if FULL_SUBSEQUENCE and BLOCK_CHUNK == CHUNK_SIZE:
            dacs = tl.load(
                DA_CS + dacs_base + offs_chunk * stride_dacs_csize
            ).to(tl.float32)
            chunk_total = tl.load(
                DA_CS + dacs_base + (CHUNK_SIZE - 1) * stride_dacs_csize
            ).to(tl.float32)
        else:
            dacs = tl.load(
                DA_CS + dacs_base + offs_chunk * stride_dacs_csize,
                mask=valid_chunk & logical,
                other=0.0,
            ).to(tl.float32)
            chunk_total = tl.load(
                DA_CS + dacs_base + (CHUNK_SIZE - 1) * stride_dacs_csize,
                mask=valid_chunk,
                other=0.0,
            ).to(tl.float32)
        c_ptrs = (
            C + pid_batch * stride_c_batch + qk_head * stride_c_head
            + offs_seq[:, None] * stride_c_seqlen
            + offs_n[None, :] * stride_c_dim
        )
        dout_ptrs = (
            DOUT + stream * stride_dout_stream + batch_idx * stride_dout_batch
            + pid_head * stride_dout_head + offs_seq[:, None] * stride_dout_seqlen
            + offs_p[None, :] * stride_dout_dim
        )
        z_ptrs = (
            Z + stream * stride_z_stream + batch_idx * stride_z_batch
            + pid_head * stride_z_head + offs_seq[:, None] * stride_z_seqlen
            + offs_p[None, :] * stride_z_dim
        )
        if (FULL_SUBSEQUENCE and EXACT_STATE) and BLOCK_CHUNK == CHUNK_SIZE:
            c_block = tl.load(c_ptrs)
            dout_block = tl.load(dout_ptrs).to(tl.float32)
            z_block = tl.load(z_ptrs).to(tl.float32)
        elif FULL_SUBSEQUENCE and BLOCK_CHUNK == CHUNK_SIZE:
            c_block = tl.load(
                c_ptrs, mask=offs_n[None, :] < dstate_n, other=0.0
            )
            dout_block = tl.load(
                dout_ptrs, mask=offs_p[None, :] < headdim_p, other=0.0
            ).to(tl.float32)
            z_block = tl.load(
                z_ptrs, mask=offs_p[None, :] < headdim_p, other=0.0
            ).to(tl.float32)
        else:
            c_block = tl.load(
                c_ptrs,
                mask=valid[:, None] & (offs_n[None, :] < dstate_n),
                other=0.0,
            )
            dout_block = tl.load(
                dout_ptrs,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p),
                other=0.0,
            ).to(tl.float32)
            z_block = tl.load(
                z_ptrs,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p),
                other=0.0,
            ).to(tl.float32)
        dpre = dout_block * z_block * tl.sigmoid(z_block)
        c_scaled = c_block * tl.exp(dacs)[:, None].to(c_block.dtype)
        gamma_chunk = tl.dot(tl.trans(c_scaled), dpre.to(c_block.dtype))
        acc_dstates = tl.exp(chunk_total) * acc_dstates + gamma_chunk

    gamma_ptrs = (
        GAMMA_REV + pid_batch * stride_gamma_batch
        + pid_subsequence_rev * stride_gamma_subsequence
        + pid_head * stride_gamma_head + offs_p[:, None] * stride_gamma_p
        + offs_n[None, :] * stride_gamma_n
    )
    if FULL_SUBSEQUENCE and EXACT_STATE:
        tl.store(gamma_ptrs, tl.trans(acc_dstates))
    else:
        tl.store(
            gamma_ptrs,
            tl.trans(acc_dstates),
            mask=(offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n),
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_STATE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_STATE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_STATE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_STATE": 1024}, num_warps=8, num_stages=1),
    ],
    key=["STATE_SIZE"],
)
@triton.jit
def _compose_subsequence_summaries_kernel(
    SUMMARIES,
    LOG_ALPHA,
    FINAL_SUMMARY,
    stride_summary_batch,
    stride_summary_subsequence,
    stride_summary_head,
    stride_summary_state,
    stride_log_batch,
    stride_log_head,
    stride_log_subsequence,
    stride_final_batch,
    stride_final_head,
    stride_final_state,
    nsubsequences,
    STATE_SIZE: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """Compose local affine summaries without materializing scan outputs."""

    pid_state = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)
    offs_state = pid_state * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
    state_mask = offs_state < STATE_SIZE
    acc = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
    for subsequence in range(nsubsequences):
        value = tl.load(
            SUMMARIES
            + pid_batch * stride_summary_batch
            + subsequence * stride_summary_subsequence
            + pid_head * stride_summary_head
            + offs_state * stride_summary_state,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        log_alpha = tl.load(
            LOG_ALPHA
            + pid_batch * stride_log_batch
            + pid_head * stride_log_head
            + subsequence * stride_log_subsequence
        ).to(tl.float32)
        acc = tl.exp(log_alpha) * acc + value
    tl.store(
        FINAL_SUMMARY
        + pid_batch * stride_final_batch
        + pid_head * stride_final_head
        + offs_state * stride_final_state,
        acc,
        mask=state_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_STATE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_STATE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_STATE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_STATE": 1024}, num_warps=8, num_stages=1),
    ],
    key=["STATE_SIZE"],
    restore_value=["SUMMARIES"],
)
@triton.jit
def _subsequence_state_passing_inplace_serial_kernel(
    SUMMARIES,
    LOG_ALPHA,
    INITIAL_STATES,
    stride_summary_batch,
    stride_summary_subsequence,
    stride_summary_head,
    stride_summary_state,
    stride_log_batch,
    stride_log_head,
    stride_log_subsequence,
    stride_initial_batch,
    stride_initial_head,
    stride_initial_state,
    nsubsequences,
    STATE_SIZE: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """Convert additive summaries to actual incoming states in place."""

    pid_state = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)
    offs_state = pid_state * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
    state_mask = offs_state < STATE_SIZE
    state = tl.load(
        INITIAL_STATES
        + pid_batch * stride_initial_batch
        + pid_head * stride_initial_head
        + offs_state * stride_initial_state,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    for subsequence in range(nsubsequences):
        summary_ptrs = (
            SUMMARIES
            + pid_batch * stride_summary_batch
            + subsequence * stride_summary_subsequence
            + pid_head * stride_summary_head
            + offs_state * stride_summary_state
        )
        additive = tl.load(summary_ptrs, mask=state_mask, other=0.0).to(tl.float32)
        tl.store(summary_ptrs, state, mask=state_mask)
        log_alpha = tl.load(
            LOG_ALPHA
            + pid_batch * stride_log_batch
            + pid_head * stride_log_head
            + subsequence * stride_log_subsequence
        ).to(tl.float32)
        state = tl.exp(log_alpha) * state + additive


@triton.jit
def _affine_scan_combine(alpha_left, beta_left, alpha_right, beta_right):
    """Compose the left affine prefix with the affine transform on its right."""

    return (
        alpha_right * alpha_left,
        alpha_right * beta_left + beta_right,
    )


@triton.jit
def _subsequence_state_passing_inplace_scan_kernel(
    SUMMARIES,
    LOG_ALPHA,
    INITIAL_STATES,
    stride_summary_batch,
    stride_summary_subsequence,
    stride_summary_head,
    stride_summary_state,
    stride_log_batch,
    stride_log_head,
    stride_log_subsequence,
    stride_initial_batch,
    stride_initial_head,
    stride_initial_state,
    nsubsequences,
    STATE_SIZE: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    BLOCK_SUBSEQUENCES: tl.constexpr,
):
    """Parallel exclusive affine scan, overwriting summaries with inputs."""

    pid_state = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)
    offs_state = pid_state * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
    offs_scan = tl.arange(0, BLOCK_SUBSEQUENCES)
    summary_idx = offs_scan - 1
    state_mask = offs_state < STATE_SIZE
    # Prepending the identity transform turns the inclusive associative scan
    # into the exclusive prefix needed by each subsection. The final additive
    # summary is intentionally not read: no later subsection consumes it.
    valid_summary = (offs_scan > 0) & (offs_scan < nsubsequences)
    additive = tl.load(
        SUMMARIES
        + pid_batch * stride_summary_batch
        + summary_idx[:, None] * stride_summary_subsequence
        + pid_head * stride_summary_head
        + offs_state[None, :] * stride_summary_state,
        mask=valid_summary[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    log_alpha = tl.load(
        LOG_ALPHA
        + pid_batch * stride_log_batch
        + pid_head * stride_log_head
        + summary_idx * stride_log_subsequence,
        mask=valid_summary,
        other=0.0,
    ).to(tl.float32)
    alpha = tl.where(valid_summary[:, None], tl.exp(log_alpha[:, None]), 1.0)
    alpha += tl.zeros((BLOCK_SUBSEQUENCES, BLOCK_STATE), dtype=tl.float32)
    prefix_alpha, prefix_beta = tl.associative_scan(
        (alpha, additive), axis=0, combine_fn=_affine_scan_combine
    )
    initial = tl.load(
        INITIAL_STATES
        + pid_batch * stride_initial_batch
        + pid_head * stride_initial_head
        + offs_state * stride_initial_state,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    incoming = prefix_alpha * initial[None, :] + prefix_beta
    tl.store(
        SUMMARIES
        + pid_batch * stride_summary_batch
        + offs_scan[:, None] * stride_summary_subsequence
        + pid_head * stride_summary_head
        + offs_state[None, :] * stride_summary_state,
        incoming,
        mask=(offs_scan[:, None] < nsubsequences) & state_mask[None, :],
    )


def mamba2_get_subsequence_states_triton(
    B: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cs: torch.Tensor,
    dA_cs_rev: torch.Tensor,
    *,
    chunk_size: int,
    subsequence_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 ``(beta, log_alpha)`` for each local subsequence."""

    if B.ndim != 4 or x.ndim != 4:
        raise ValueError("B and x must be 4D")
    if any(tensor.ndim != 4 for tensor in (dt, dA_cs, dA_cs_rev)):
        raise ValueError("dt, dA_cs, and dA_cs_rev must be 4D")
    if chunk_size <= 0 or subsequence_size < chunk_size:
        raise ValueError("subsequence_size must be at least chunk_size")
    if subsequence_size % chunk_size:
        raise ValueError("subsequence_size must be a multiple of chunk_size")
    folded_batch, seqlen, nheads_qk, dstate = B.shape
    x_batch, x_seqlen, nheads, headdim = x.shape
    if (x_batch, x_seqlen) != (folded_batch, seqlen):
        raise ValueError("B and x must share folded batch and sequence dimensions")
    if nheads % nheads_qk:
        raise ValueError("x heads must be divisible by B heads")
    nchunks = triton.cdiv(seqlen, chunk_size)
    subsequence_chunks = subsequence_size // chunk_size
    nsubsequences = triton.cdiv(nchunks, subsequence_chunks)
    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    for name, tensor in (("dt", dt), ("dA_cs", dA_cs), ("dA_cs_rev", dA_cs_rev)):
        if tensor.shape != decay_shape:
            raise ValueError(f"{name} must have shape {decay_shape}")
    tensors = (B, x, dt, dA_cs, dA_cs_rev)
    if not B.is_cuda or any(tensor.device != B.device for tensor in tensors[1:]):
        raise ValueError("all subsequence-summary inputs must be CUDA tensors on one device")
    beta = torch.empty(
        folded_batch,
        nsubsequences,
        nheads,
        headdim,
        dstate,
        device=B.device,
        dtype=torch.float32,
    )
    log_alpha = torch.empty(
        folded_batch, nheads, nsubsequences, device=B.device, dtype=torch.float32
    )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(dstate))
    kernel_args = (
        B, x, dt, dA_cs, dA_cs_rev, beta, log_alpha,
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1),
        dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2), beta.stride(3), beta.stride(4),
        log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
        seqlen, nchunks, headdim, dstate, nheads_qk,
    )
    exact_state = headdim == block_p and dstate == block_n
    nfull = seqlen // subsequence_size
    has_tail = seqlen % subsequence_size != 0
    if nfull:
        _mamba2_get_subsequence_state_kernel[(nheads, folded_batch, nfull)](
            *kernel_args,
            CHUNK_SIZE=chunk_size,
            SUBSEQUENCE_CHUNKS=subsequence_chunks,
            SUBSEQUENCE_OFFSET=0,
            FULL_SUBSEQUENCE=True,
            EXACT_STATE=exact_state,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )
    if has_tail:
        _mamba2_get_subsequence_state_kernel[(nheads, folded_batch, 1)](
            *kernel_args,
            CHUNK_SIZE=chunk_size,
            SUBSEQUENCE_CHUNKS=subsequence_chunks,
            SUBSEQUENCE_OFFSET=nfull,
            FULL_SUBSEQUENCE=False,
            EXACT_STATE=False,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )
    return beta, log_alpha


def mamba2_get_subsequence_dstates_triton(
    C: torch.Tensor,
    dout: torch.Tensor,
    z: torch.Tensor,
    dA_cs: torch.Tensor,
    *,
    chunk_size: int,
    subsequence_size: int,
) -> torch.Tensor:
    """Return FP32 gamma summaries in reverse subsequence order."""

    if C.ndim != 4 or dout.ndim != 5 or z.ndim != 5 or dA_cs.ndim != 4:
        raise ValueError("invalid subsequence-adjoint input rank")
    if chunk_size <= 0 or subsequence_size < chunk_size or subsequence_size % chunk_size:
        raise ValueError("subsequence_size must be a positive multiple of chunk_size")
    folded_batch, seqlen, nheads_qk, dstate = C.shape
    streams, dout_seqlen, batch, nheads, headdim = dout.shape
    if z.shape != dout.shape or folded_batch != streams * batch or dout_seqlen != seqlen:
        raise ValueError("C, dout, and z shapes are inconsistent")
    if nheads % nheads_qk:
        raise ValueError("dout heads must be divisible by C heads")
    nchunks = triton.cdiv(seqlen, chunk_size)
    subsequence_chunks = subsequence_size // chunk_size
    nsubsequences = triton.cdiv(nchunks, subsequence_chunks)
    if dA_cs.shape != (folded_batch, nheads, nchunks, chunk_size):
        raise ValueError("dA_cs has the wrong shape")
    tensors = (C, dout, z, dA_cs)
    if not C.is_cuda or any(tensor.device != C.device for tensor in tensors[1:]):
        raise ValueError("all subsequence-adjoint inputs must be CUDA tensors on one device")
    gamma = torch.empty(
        folded_batch,
        nsubsequences,
        nheads,
        headdim,
        dstate,
        device=C.device,
        dtype=torch.float32,
    )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(dstate))
    kernel_args = (
        C, dout, z, dA_cs, gamma,
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        gamma.stride(0), gamma.stride(1), gamma.stride(2), gamma.stride(3), gamma.stride(4),
        seqlen,
        nchunks,
        headdim,
        dstate,
        nheads_qk,
    )
    exact_state = headdim == block_p and dstate == block_n
    nfull = seqlen // subsequence_size
    has_tail = seqlen % subsequence_size != 0
    # Reverse slots are causal-reverse ordered: a partial tail, when present,
    # owns slot zero and complete subsequences follow it.
    if has_tail:
        _mamba2_get_subsequence_dstate_kernel[(nheads, folded_batch, 1)](
            *kernel_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            SUBSEQUENCE_CHUNKS=subsequence_chunks,
            REVERSE_SLOT_OFFSET=0,
            ORIGINAL_SUBSEQUENCE_BASE=nfull,
            ORIGINAL_SUBSEQUENCE_COUNT=1,
            FULL_SUBSEQUENCE=False,
            EXACT_STATE=False,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )
    if nfull:
        _mamba2_get_subsequence_dstate_kernel[(nheads, folded_batch, nfull)](
            *kernel_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            SUBSEQUENCE_CHUNKS=subsequence_chunks,
            REVERSE_SLOT_OFFSET=int(has_tail),
            ORIGINAL_SUBSEQUENCE_BASE=0,
            ORIGINAL_SUBSEQUENCE_COUNT=nfull,
            FULL_SUBSEQUENCE=True,
            EXACT_STATE=exact_state,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )
    return gamma


def compose_subsequence_summaries_triton(
    summaries: torch.Tensor, log_alpha: torch.Tensor
) -> torch.Tensor:
    """Compose FP32 subsequence summaries into one summary per segment."""

    if summaries.ndim != 5 or log_alpha.ndim != 3:
        raise ValueError("summaries must be 5D and log_alpha must be 3D")
    batch, nsubsequences, nheads, headdim, dstate = summaries.shape
    if log_alpha.shape != (batch, nheads, nsubsequences):
        raise ValueError("log_alpha shape does not match summaries")
    if summaries.dtype is not torch.float32 or log_alpha.dtype is not torch.float32:
        raise ValueError("subsequence affine composition requires FP32 inputs")
    if summaries.device != log_alpha.device or not summaries.is_cuda:
        raise ValueError("subsequence affine composition requires one CUDA device")
    final = torch.empty(
        batch, nheads, headdim, dstate, device=summaries.device, dtype=torch.float32
    )
    state_size = headdim * dstate
    summary_flat = summaries.view(batch, nsubsequences, nheads, state_size)
    final_flat = final.view(batch, nheads, state_size)
    grid = lambda META: (triton.cdiv(state_size, META["BLOCK_STATE"]), nheads, batch)
    _compose_subsequence_summaries_kernel[grid](
        summary_flat,
        log_alpha,
        final_flat,
        summary_flat.stride(0), summary_flat.stride(1),
        summary_flat.stride(2), summary_flat.stride(3),
        log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
        final_flat.stride(0), final_flat.stride(1), final_flat.stride(2),
        nsubsequences,
        STATE_SIZE=state_size,
    )
    return final


def subsequence_state_passing_inplace_triton(
    summaries: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_states: torch.Tensor,
) -> torch.Tensor:
    """Overwrite FP32 additive summaries with actual incoming states."""

    if summaries.ndim != 5 or log_alpha.ndim != 3 or initial_states.ndim != 4:
        raise ValueError("invalid in-place subsequence state-passing input rank")
    batch, nsubsequences, nheads, headdim, dstate = summaries.shape
    if log_alpha.shape != (batch, nheads, nsubsequences):
        raise ValueError("log_alpha shape does not match summaries")
    if initial_states.shape != (batch, nheads, headdim, dstate):
        raise ValueError("initial_states shape does not match summaries")
    tensors = (summaries, log_alpha, initial_states)
    if any(tensor.dtype is not torch.float32 for tensor in tensors):
        raise ValueError("in-place subsequence state passing requires FP32")
    if not summaries.is_cuda or any(tensor.device != summaries.device for tensor in tensors[1:]):
        raise ValueError("in-place subsequence state passing requires one CUDA device")
    state_size = headdim * dstate
    summary_flat = summaries.view(batch, nsubsequences, nheads, state_size)
    initial_flat = initial_states.view(batch, nheads, state_size)
    common_args = (
        summary_flat,
        log_alpha,
        initial_flat,
        summary_flat.stride(0), summary_flat.stride(1),
        summary_flat.stride(2), summary_flat.stride(3),
        log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
        initial_flat.stride(0), initial_flat.stride(1), initial_flat.stride(2),
        nsubsequences,
    )
    if nsubsequences <= 128:
        block_subsequences = triton.next_power_of_2(nsubsequences)
        # Bound the scan tile to 8192 elements. This exposes subsection
        # parallelism without recreating the latency-bound serial recurrence.
        block_state = max(16, min(128, 8192 // block_subsequences))
        grid = (triton.cdiv(state_size, block_state), nheads, batch)
        _subsequence_state_passing_inplace_scan_kernel[grid](
            *common_args,
            STATE_SIZE=state_size,
            BLOCK_STATE=block_state,
            BLOCK_SUBSEQUENCES=block_subsequences,
            num_warps=8,
            num_stages=1,
        )
    else:
        grid = lambda META: (
            triton.cdiv(state_size, META["BLOCK_STATE"]), nheads, batch
        )
        _subsequence_state_passing_inplace_serial_kernel[grid](
            *common_args,
            STATE_SIZE=state_size,
        )
    return summaries


def _prune_dq_cp_configs(configs, nargs, **kwargs):
    del nargs
    subsequence_chunks = kwargs.get("SUBSEQUENCE_CHUNKS", 0)
    if subsequence_chunks == 0:
        return [
            config
            for config in configs
            if config.maxnreg is None
            and not (config.num_warps == 8 and config.num_stages == 2)
        ]
    # The split schedule is consistently fastest at eight warps / one stage.
    # Two-stage multi-chunk variants also hit a B300 tensor-memory compiler
    # failure, so do not autotune through configurations we cannot dispatch.
    return [
        config
        for config in configs
        if config.num_warps == 8 and config.num_stages == 1
    ]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1, maxnreg=256),
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=256),
        triton.Config({}, num_warps=8, num_stages=1, maxnreg=128),
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=128),
    ],
    key=[
        "CHUNK_SIZE",
        "BLOCK_CHUNK",
        "BLOCK_P",
        "BLOCK_N",
        "SUBSEQUENCE_CHUNKS",
        "SKIP_CHUNK_STATE_STORE",
        "FULL_SUBSEQUENCE",
        "EXACT_SHAPE",
    ],
    prune_configs_by={"early_config_prune": _prune_dq_cp_configs},
)
@triton.jit
def _mamba2_bwd_dq_cp_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    DT,
    DA_CS,
    DA_CS_REV,
    INITIAL_STATES,
    D,
    DC_PARTIAL,
    DZ,
    CHUNK_STATES,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_chunk,
    stride_dacsrev_csize,
    stride_initial_stream,
    stride_initial_batch,
    stride_initial_head,
    stride_initial_p,
    stride_initial_n,
    stride_d_head,
    stride_dc_batch,
    stride_dc_seqlen,
    stride_dc_head,
    stride_dc_dim,
    stride_dz_stream,
    stride_dz_seqlen,
    stride_dz_batch,
    stride_dz_head,
    stride_dz_dim,
    stride_states_batch,
    stride_states_head,
    stride_states_chunk,
    stride_states_p,
    stride_states_n,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    A,
    PROCESSED_DDT,
    DA_CHUNK,
    GRAD_SCALE_INTERMEDIATE,
    stride_a_head,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dachunk_batch,
    stride_dachunk_head,
    stride_dachunk_chunk,
    stride_gsi_batch,
    stride_gsi_head,
    stride_gsi_chunk,
    stride_gsi_csize,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBSEQUENCE_CHUNKS: tl.constexpr = 0,
    SUBSEQUENCE_OFFSET: tl.constexpr = 0,
    SKIP_CHUNK_STATE_STORE: tl.constexpr = False,
    FULL_SUBSEQUENCE: tl.constexpr = False,
    EXACT_SHAPE: tl.constexpr = False,
):
    """Compute dC/dz, chunk starts, current decay, and cached next-decay GSI."""

    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)
    pid_subsequence = tl.program_id(2) + SUBSEQUENCE_OFFSET
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    c_offset = pid_folded_batch * stride_c_batch + qk_head * stride_c_head
    b_offset = pid_folded_batch * stride_b_batch + qk_head * stride_b_head
    x_offset = pid_folded_batch * stride_x_batch + pid_head * stride_x_head
    dout_offset = (
        stream * stride_dout_stream
        + batch_idx * stride_dout_batch
        + pid_head * stride_dout_head
    )
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    decay_offset = pid_folded_batch * stride_dt_batch + pid_head * stride_dt_head
    dacs_offset = pid_folded_batch * stride_dacs_batch + pid_head * stride_dacs_head
    dacsrev_offset = (
        pid_folded_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    )
    if SUBSEQUENCE_CHUNKS > 0:
        initial_offset = (
            pid_folded_batch * stride_initial_stream
            + pid_subsequence * stride_initial_batch
            + pid_head * stride_initial_head
        )
    else:
        initial_offset = (
            stream * stride_initial_stream
            + batch_idx * stride_initial_batch
            + pid_head * stride_initial_head
        )
    dc_offset = pid_folded_batch * stride_dc_batch + pid_head * stride_dc_head
    dz_offset = (
        stream * stride_dz_stream
        + batch_idx * stride_dz_batch
        + pid_head * stride_dz_head
    )
    states_offset = (
        pid_folded_batch * stride_states_batch + pid_head * stride_states_head
    )
    ddt_offset = pid_folded_batch * stride_ddt_batch + pid_head * stride_ddt_head
    dachunk_offset = (
        pid_folded_batch * stride_dachunk_batch + pid_head * stride_dachunk_head
    )
    gsi_offset = pid_folded_batch * stride_gsi_batch + pid_head * stride_gsi_head

    offs_chunk = tl.arange(0, BLOCK_CHUNK)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    initial_ptrs = (
        INITIAL_STATES
        + initial_offset
        + offs_p[:, None] * stride_initial_p
        + offs_n[None, :] * stride_initial_n
    )
    if EXACT_SHAPE:
        state_mask = (offs_p[:, None] < BLOCK_P) & (offs_n[None, :] < BLOCK_N)
    else:
        state_mask = (offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n)
    if FULL_SUBSEQUENCE and EXACT_SHAPE:
        acc_states = tl.load(initial_ptrs).to(tl.float32)
    else:
        acc_states = tl.load(initial_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)

    total_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    if SUBSEQUENCE_CHUNKS > 0:
        num_chunks = SUBSEQUENCE_CHUNKS
        first_chunk = pid_subsequence * SUBSEQUENCE_CHUNKS
    else:
        num_chunks = total_chunks
        first_chunk = 0
    for local_chunk_idx in range(num_chunks):
        chunk_idx = first_chunk + local_chunk_idx
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seq = chunk_start + offs_chunk
        logical_chunk_mask = offs_chunk < CHUNK_SIZE
        if FULL_SUBSEQUENCE:
            valid_chunk = True
            seq_mask = logical_chunk_mask
        else:
            valid_chunk = chunk_idx < total_chunks
            seq_mask = logical_chunk_mask & valid_chunk & (offs_seq < seqlen)

        if not SKIP_CHUNK_STATE_STORE:
            chunk_state_ptrs = (
                CHUNK_STATES
                + states_offset
                + chunk_idx * stride_states_chunk
                + offs_p[:, None] * stride_states_p
                + offs_n[None, :] * stride_states_n
            )
            if FULL_SUBSEQUENCE and EXACT_SHAPE:
                tl.store(chunk_state_ptrs, acc_states)
            else:
                tl.store(chunk_state_ptrs, acc_states, mask=state_mask & valid_chunk)

        dt_ptrs = (
            DT + decay_offset + chunk_idx * stride_dt_chunk
            + offs_chunk * stride_dt_csize
        )
        dacs_ptrs = (
            DA_CS + dacs_offset + chunk_idx * stride_dacs_chunk
            + offs_chunk * stride_dacs_csize
        )
        dacsrev_ptrs = (
            DA_CS_REV + dacsrev_offset + chunk_idx * stride_dacsrev_chunk
            + offs_chunk * stride_dacsrev_csize
        )
        if FULL_SUBSEQUENCE:
            dt_block = tl.load(dt_ptrs).to(tl.float32)
            dA_cs = tl.load(dacs_ptrs).to(tl.float32)
            dA_cs_rev = tl.load(dacsrev_ptrs).to(tl.float32)
            dA_cs_last = tl.load(
                DA_CS + dacs_offset + chunk_idx * stride_dacs_chunk
                + (CHUNK_SIZE - 1) * stride_dacs_csize
            ).to(tl.float32)
        else:
            dt_block = tl.load(
                dt_ptrs, mask=valid_chunk & logical_chunk_mask, other=0.0
            ).to(tl.float32)
            dA_cs = tl.load(
                dacs_ptrs, mask=valid_chunk & logical_chunk_mask, other=0.0
            ).to(tl.float32)
            dA_cs_rev = tl.load(
                dacsrev_ptrs, mask=valid_chunk & logical_chunk_mask, other=0.0
            ).to(tl.float32)
            dA_cs_last = tl.load(
                DA_CS + dacs_offset + chunk_idx * stride_dacs_chunk
                + (CHUNK_SIZE - 1) * stride_dacs_csize,
                mask=valid_chunk,
                other=0.0,
            ).to(tl.float32)

        c_ptrs = (
            C
            + c_offset
            + offs_seq[:, None] * stride_c_seqlen
            + offs_n[None, :] * stride_c_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            c_block = tl.load(c_ptrs)
        else:
            if FULL_SUBSEQUENCE:
                qk_mask = offs_n[None, :] < dstate_n
            else:
                qk_mask = seq_mask[:, None] & (offs_n[None, :] < dstate_n)
            c_block = tl.load(c_ptrs, mask=qk_mask, other=0.0)

        b_ptrs = (
            B
            + b_offset
            + offs_seq[:, None] * stride_b_seqlen
            + offs_n[None, :] * stride_b_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            b_block = tl.load(b_ptrs)
        else:
            b_block = tl.load(b_ptrs, mask=qk_mask, other=0.0)

        x_ptrs = (
            X
            + x_offset
            + offs_seq[:, None] * stride_x_seqlen
            + offs_p[None, :] * stride_x_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            x_block = tl.load(x_ptrs)
        else:
            if FULL_SUBSEQUENCE:
                value_mask = offs_p[None, :] < headdim_p
            else:
                value_mask = seq_mask[:, None] & (offs_p[None, :] < headdim_p)
            x_block = tl.load(x_ptrs, mask=value_mask, other=0.0)

        dout_ptrs = (
            DOUT
            + dout_offset
            + offs_seq[:, None] * stride_dout_seqlen
            + offs_p[None, :] * stride_dout_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            dout_block = tl.load(dout_ptrs).to(tl.float32)
        else:
            dout_block = tl.load(dout_ptrs, mask=value_mask, other=0.0).to(tl.float32)
        z_ptrs = (
            Z
            + z_offset
            + offs_seq[:, None] * stride_z_seqlen
            + offs_p[None, :] * stride_z_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            z_block = tl.load(z_ptrs).to(tl.float32)
        else:
            z_block = tl.load(z_ptrs, mask=value_mask, other=0.0).to(tl.float32)
        sigmoid_z = tl.sigmoid(z_block)
        dpre = dout_block * z_block * sigmoid_z

        causal = offs_chunk[:, None] >= offs_chunk[None, :]
        if FULL_SUBSEQUENCE:
            valid_matrix = causal
            segsum = tl.where(
                causal, dA_cs[:, None] - dA_cs[None, :], -float("inf")
            )
        else:
            valid_matrix = seq_mask[:, None] & seq_mask[None, :]
            segsum = tl.where(
                valid_matrix & causal,
                dA_cs[:, None] - dA_cs[None, :],
                -float("inf"),
            )
        segsum = tl.minimum(segsum, 0.0)
        exp_segsum = tl.exp(segsum)

        dpre_for_dot = dpre.to(x_block.dtype)
        acc_dc = tl.dot(dpre_for_dot, acc_states.to(dpre_for_dot.dtype))
        acc_dc *= tl.exp(dA_cs)[:, None]
        grad_scale_intermediate = tl.sum(
            acc_dc * c_block.to(tl.float32), axis=1
        )
        gsi_ptrs = (
            GRAD_SCALE_INTERMEDIATE + gsi_offset
            + chunk_idx * stride_gsi_chunk + offs_chunk * stride_gsi_csize
        )
        if FULL_SUBSEQUENCE:
            tl.store(gsi_ptrs, grad_scale_intermediate)
        else:
            tl.store(
                gsi_ptrs,
                tl.where(seq_mask, grad_scale_intermediate, 0.0),
                mask=valid_chunk & logical_chunk_mask,
            )
        s_block_intermediate = tl.dot(dpre_for_dot, tl.trans(x_block))
        s_block_intermediate *= exp_segsum
        s_block_intermediate = tl.where(
            valid_matrix & causal, s_block_intermediate, 0.0
        )
        s_block = s_block_intermediate * dt_block[None, :]
        acc_dc += tl.dot(s_block.to(b_block.dtype), b_block)

        pre_gate = tl.dot(c_block, tl.trans(acc_states.to(c_block.dtype)))
        pre_gate *= tl.exp(dA_cs)[:, None]
        cb = tl.dot(c_block, tl.trans(b_block))
        cb *= exp_segsum * dt_block[None, :]
        pre_gate += tl.dot(cb.to(x_block.dtype), x_block)
        d_value = tl.load(D + pid_head * stride_d_head).to(tl.float32)
        pre_gate += x_block.to(tl.float32) * d_value

        dz_block = dout_block * pre_gate * sigmoid_z * (
            1.0 + z_block * (1.0 - sigmoid_z)
        )
        dc_ptrs = (
            DC_PARTIAL
            + dc_offset
            + offs_seq[:, None] * stride_dc_seqlen
            + offs_n[None, :] * stride_dc_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            tl.store(dc_ptrs, acc_dc.to(c_block.dtype))
        else:
            tl.store(dc_ptrs, acc_dc.to(c_block.dtype), mask=qk_mask)
        dz_ptrs = (
            DZ
            + dz_offset
            + offs_seq[:, None] * stride_dz_seqlen
            + offs_p[None, :] * stride_dz_dim
        )
        if FULL_SUBSEQUENCE and EXACT_SHAPE:
            tl.store(dz_ptrs, dz_block.to(x_block.dtype))
        else:
            tl.store(dz_ptrs, dz_block.to(x_block.dtype), mask=value_mask)

        # Match the legacy dQ schedule: retain the decayed dpre @ x^T score
        # used above and replay only C @ B^T for the current decay reduction.
        dM_curr = tl.dot(c_block, tl.trans(b_block)) * s_block_intermediate
        dM_curr = tl.where(valid_matrix & causal, dM_curr, 0.0)
        direct_ddt = tl.sum(dM_curr, axis=0)
        strict_lower = offs_chunk[:, None] > offs_chunk[None, :]
        g = tl.where(
            valid_matrix & strict_lower,
            dM_curr * dt_block[None, :],
            0.0,
        )
        reverse_g = tl.cumsum(g, axis=0, reverse=True)
        ddA_current = tl.sum(
            tl.where(valid_matrix & strict_lower, reverse_g, 0.0), axis=1
        )
        ddt_ptrs = (
            PROCESSED_DDT + ddt_offset + chunk_idx * stride_ddt_chunk
            + offs_chunk * stride_ddt_csize
        )
        dachunk_ptr = DA_CHUNK + dachunk_offset + chunk_idx * stride_dachunk_chunk
        if FULL_SUBSEQUENCE:
            tl.store(ddt_ptrs, direct_ddt + a_value * ddA_current)
            tl.store(dachunk_ptr, tl.sum(ddA_current * dt_block))
        else:
            processed_ddt = tl.where(
                seq_mask, direct_ddt + a_value * ddA_current, 0.0
            )
            tl.store(
                ddt_ptrs,
                processed_ddt,
                mask=valid_chunk & logical_chunk_mask,
            )
            tl.store(
                dachunk_ptr,
                tl.sum(ddA_current * dt_block),
                mask=valid_chunk,
            )

        if SUBSEQUENCE_CHUNKS == 0:
            state_scale = tl.exp(dA_cs_rev) * dt_block
            b_scaled = b_block * state_scale[:, None].to(b_block.dtype)
            acc_states = tl.exp(dA_cs_last) * acc_states + tl.dot(
                tl.trans(x_block).to(b_block.dtype), b_scaled
            )
        elif local_chunk_idx < SUBSEQUENCE_CHUNKS - 1:
            state_scale = tl.exp(dA_cs_rev) * dt_block
            b_scaled = b_block * state_scale[:, None].to(b_block.dtype)
            acc_states = tl.exp(dA_cs_last) * acc_states + tl.dot(
                tl.trans(x_block).to(b_block.dtype), b_scaled
            )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_P", "BLOCK_N"],
)
@triton.jit
def _mamba2_bwd_dq_cp_large_chunk_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    DT,
    DA_CS,
    DA_CS_REV,
    INITIAL_STATES,
    D,
    DC_PARTIAL,
    DZ,
    CHUNK_STATES,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_chunk,
    stride_dacsrev_csize,
    stride_initial_stream,
    stride_initial_batch,
    stride_initial_head,
    stride_initial_p,
    stride_initial_n,
    stride_d_head,
    stride_dc_batch,
    stride_dc_seqlen,
    stride_dc_head,
    stride_dc_dim,
    stride_dz_stream,
    stride_dz_seqlen,
    stride_dz_batch,
    stride_dz_head,
    stride_dz_dim,
    stride_states_batch,
    stride_states_head,
    stride_states_chunk,
    stride_states_p,
    stride_states_n,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Large-chunk dC/dz with tiled sequence interactions."""

    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    c_offset = pid_folded_batch * stride_c_batch + qk_head * stride_c_head
    b_offset = pid_folded_batch * stride_b_batch + qk_head * stride_b_head
    x_offset = pid_folded_batch * stride_x_batch + pid_head * stride_x_head
    dout_offset = (
        stream * stride_dout_stream
        + batch_idx * stride_dout_batch
        + pid_head * stride_dout_head
    )
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    decay_offset = pid_folded_batch * stride_dt_batch + pid_head * stride_dt_head
    dacs_offset = pid_folded_batch * stride_dacs_batch + pid_head * stride_dacs_head
    dacsrev_offset = (
        pid_folded_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    )
    initial_offset = (
        stream * stride_initial_stream
        + batch_idx * stride_initial_batch
        + pid_head * stride_initial_head
    )
    dc_offset = pid_folded_batch * stride_dc_batch + pid_head * stride_dc_head
    dz_offset = (
        stream * stride_dz_stream
        + batch_idx * stride_dz_batch
        + pid_head * stride_dz_head
    )
    states_offset = pid_folded_batch * stride_states_batch + pid_head * stride_states_head

    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    state_mask = (offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n)
    acc_states = tl.load(
        INITIAL_STATES
        + initial_offset
        + offs_p[:, None] * stride_initial_p
        + offs_n[None, :] * stride_initial_n,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    d_value = tl.load(D + pid_head * stride_d_head).to(tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_dt_offset = decay_offset + chunk_idx * stride_dt_chunk
        chunk_dacs_offset = dacs_offset + chunk_idx * stride_dacs_chunk
        chunk_dacsrev_offset = dacsrev_offset + chunk_idx * stride_dacsrev_chunk

        tl.store(
            CHUNK_STATES
            + states_offset
            + chunk_idx * stride_states_chunk
            + offs_p[:, None] * stride_states_p
            + offs_n[None, :] * stride_states_n,
            acc_states,
            mask=state_mask,
        )
        dA_cs_last = tl.load(
            DA_CS + chunk_dacs_offset + (CHUNK_SIZE - 1) * stride_dacs_csize
        ).to(tl.float32)

        for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            offs_m = row_start + tl.arange(0, BLOCK_M)
            offs_seq_m = chunk_start + offs_m
            row_logical = offs_m < CHUNK_SIZE
            row_valid = row_logical & (offs_seq_m < seqlen)
            qk_row_mask = row_valid[:, None] & (offs_n[None, :] < dstate_n)
            value_row_mask = row_valid[:, None] & (offs_p[None, :] < headdim_p)

            c_row = tl.load(
                C
                + c_offset
                + offs_seq_m[:, None] * stride_c_seqlen
                + offs_n[None, :] * stride_c_dim,
                mask=qk_row_mask,
                other=0.0,
            )
            x_row = tl.load(
                X
                + x_offset
                + offs_seq_m[:, None] * stride_x_seqlen
                + offs_p[None, :] * stride_x_dim,
                mask=value_row_mask,
                other=0.0,
            )
            dout_row = tl.load(
                DOUT
                + dout_offset
                + offs_seq_m[:, None] * stride_dout_seqlen
                + offs_p[None, :] * stride_dout_dim,
                mask=value_row_mask,
                other=0.0,
            ).to(tl.float32)
            z_row = tl.load(
                Z
                + z_offset
                + offs_seq_m[:, None] * stride_z_seqlen
                + offs_p[None, :] * stride_z_dim,
                mask=value_row_mask,
                other=0.0,
            ).to(tl.float32)
            sigmoid_z = tl.sigmoid(z_row)
            dpre = dout_row * z_row * sigmoid_z
            dpre_dot = dpre.to(x_row.dtype)
            dA_row = tl.load(
                DA_CS + chunk_dacs_offset + offs_m * stride_dacs_csize,
                mask=row_logical,
                other=0.0,
            ).to(tl.float32)

            acc_dc = tl.dot(dpre_dot, acc_states.to(dpre_dot.dtype))
            acc_dc *= tl.exp(dA_row)[:, None]
            pre_gate = tl.dot(c_row, tl.trans(acc_states.to(c_row.dtype)))
            pre_gate *= tl.exp(dA_row)[:, None]

            for key_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
                offs_k = key_start + tl.arange(0, BLOCK_K)
                offs_seq_k = chunk_start + offs_k
                key_logical = offs_k < CHUNK_SIZE
                key_valid = key_logical & (offs_seq_k < seqlen)
                b_block = tl.load(
                    B
                    + b_offset
                    + offs_seq_k[:, None] * stride_b_seqlen
                    + offs_n[None, :] * stride_b_dim,
                    mask=key_valid[:, None] & (offs_n[None, :] < dstate_n),
                    other=0.0,
                )
                x_block = tl.load(
                    X
                    + x_offset
                    + offs_seq_k[:, None] * stride_x_seqlen
                    + offs_p[None, :] * stride_x_dim,
                    mask=key_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                )
                dt_block = tl.load(
                    DT + chunk_dt_offset + offs_k * stride_dt_csize,
                    mask=key_logical,
                    other=0.0,
                ).to(tl.float32)
                dA_col = tl.load(
                    DA_CS + chunk_dacs_offset + offs_k * stride_dacs_csize,
                    mask=key_logical,
                    other=0.0,
                ).to(tl.float32)
                causal = offs_m[:, None] >= offs_k[None, :]
                valid_matrix = row_valid[:, None] & key_valid[None, :]
                segsum = tl.where(
                    valid_matrix & causal,
                    dA_row[:, None] - dA_col[None, :],
                    -float("inf"),
                )
                segsum = tl.minimum(segsum, 0.0)
                weights = tl.exp(segsum) * dt_block[None, :]
                dpre_x = tl.dot(dpre_dot, tl.trans(x_block)) * weights
                acc_dc += tl.dot(dpre_x.to(b_block.dtype), b_block)
                cb = tl.dot(c_row, tl.trans(b_block)) * weights
                pre_gate += tl.dot(cb.to(x_block.dtype), x_block)

            pre_gate += x_row.to(tl.float32) * d_value
            dz_row = dout_row * pre_gate * sigmoid_z * (
                1.0 + z_row * (1.0 - sigmoid_z)
            )
            tl.store(
                DC_PARTIAL
                + dc_offset
                + offs_seq_m[:, None] * stride_dc_seqlen
                + offs_n[None, :] * stride_dc_dim,
                acc_dc.to(c_row.dtype),
                mask=qk_row_mask,
            )
            tl.store(
                DZ
                + dz_offset
                + offs_seq_m[:, None] * stride_dz_seqlen
                + offs_p[None, :] * stride_dz_dim,
                dz_row.to(x_row.dtype),
                mask=value_row_mask,
            )

        acc_states *= tl.exp(dA_cs_last)
        for key_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
            offs_k = key_start + tl.arange(0, BLOCK_K)
            offs_seq_k = chunk_start + offs_k
            key_logical = offs_k < CHUNK_SIZE
            key_valid = key_logical & (offs_seq_k < seqlen)
            b_block = tl.load(
                B
                + b_offset
                + offs_seq_k[:, None] * stride_b_seqlen
                + offs_n[None, :] * stride_b_dim,
                mask=key_valid[:, None] & (offs_n[None, :] < dstate_n),
                other=0.0,
            )
            x_block = tl.load(
                X
                + x_offset
                + offs_seq_k[:, None] * stride_x_seqlen
                + offs_p[None, :] * stride_x_dim,
                mask=key_valid[:, None] & (offs_p[None, :] < headdim_p),
                other=0.0,
            )
            dt_block = tl.load(
                DT + chunk_dt_offset + offs_k * stride_dt_csize,
                mask=key_logical,
                other=0.0,
            ).to(tl.float32)
            dA_rev = tl.load(
                DA_CS_REV + chunk_dacsrev_offset + offs_k * stride_dacsrev_csize,
                mask=key_logical,
                other=0.0,
            ).to(tl.float32)
            b_scaled = b_block * (tl.exp(dA_rev) * dt_block)[:, None].to(b_block.dtype)
            acc_states += tl.dot(tl.trans(x_block).to(b_block.dtype), b_scaled)


@triton.jit
def _mamba2_bwd_dq_current_decay_cp_tiled_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    DT,
    DA_CS,
    A,
    PROCESSED_DDT,
    DA_CHUNK,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dout_stream, stride_dout_seqlen, stride_dout_batch,
    stride_dout_head, stride_dout_dim,
    stride_z_stream, stride_z_seqlen, stride_z_batch, stride_z_head, stride_z_dim,
    stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
    stride_dacs_batch, stride_dacs_head, stride_dacs_chunk, stride_dacs_csize,
    stride_a_head,
    stride_ddt_batch, stride_ddt_head, stride_ddt_chunk, stride_ddt_csize,
    stride_dachunk_batch, stride_dachunk_head, stride_dachunk_chunk,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tile and reduce current dM with one deterministic chunk owner."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_batch // BATCH
    batch_idx = pid_batch - stream * BATCH
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_all = tl.arange(0, BLOCK_CHUNK)
    all_logical = offs_all < CHUNK_SIZE
    all_valid = all_logical & (chunk_start + offs_all < seqlen)
    ddt_base = (
        pid_batch * stride_ddt_batch + pid_head * stride_ddt_head
        + pid_chunk * stride_ddt_chunk
    )
    tl.store(
        PROCESSED_DDT + ddt_base + offs_all * stride_ddt_csize,
        tl.zeros((BLOCK_CHUNK,), tl.float32),
        mask=all_logical,
    )
    a = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    dA_value = 0.0
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)

    for key_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
        cols = key_start + tl.arange(0, BLOCK_K)
        seq_cols = chunk_start + cols
        col_valid = (cols < CHUNK_SIZE) & (seq_cols < seqlen)
        b_col = tl.load(
            B + pid_batch * stride_b_batch + seq_cols[:, None] * stride_b_seqlen
            + qk_head * stride_b_head + offs_n[None, :] * stride_b_dim,
            mask=col_valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
        )
        x_col = tl.load(
            X + pid_batch * stride_x_batch + seq_cols[:, None] * stride_x_seqlen
            + pid_head * stride_x_head + offs_p[None, :] * stride_x_dim,
            mask=col_valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
        )
        dt_col = tl.load(
            DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head
            + pid_chunk * stride_dt_chunk + cols * stride_dt_csize,
            mask=cols < CHUNK_SIZE, other=0.0,
        ).to(tl.float32)
        dacs_col = tl.load(
            DA_CS + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
            + pid_chunk * stride_dacs_chunk + cols * stride_dacs_csize,
            mask=cols < CHUNK_SIZE, other=0.0,
        ).to(tl.float32)
        direct = tl.zeros((BLOCK_K,), tl.float32)
        carry = tl.zeros((BLOCK_K,), tl.float32)

        for row_rev in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            row_start = CHUNK_SIZE - BLOCK_M - row_rev
            rows = row_start + tl.arange(0, BLOCK_M)
            seq_rows = chunk_start + rows
            row_valid = (rows >= 0) & (rows < CHUNK_SIZE) & (seq_rows < seqlen)
            c_row = tl.load(
                C + pid_batch * stride_c_batch + seq_rows[:, None] * stride_c_seqlen
                + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim,
                mask=row_valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
            )
            x_row = tl.load(
                X + pid_batch * stride_x_batch + seq_rows[:, None] * stride_x_seqlen
                + pid_head * stride_x_head + offs_p[None, :] * stride_x_dim,
                mask=row_valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            )
            dout_row = tl.load(
                DOUT + stream * stride_dout_stream
                + seq_rows[:, None] * stride_dout_seqlen
                + batch_idx * stride_dout_batch + pid_head * stride_dout_head
                + offs_p[None, :] * stride_dout_dim,
                mask=row_valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            ).to(tl.float32)
            z_row = tl.load(
                Z + stream * stride_z_stream + seq_rows[:, None] * stride_z_seqlen
                + batch_idx * stride_z_batch + pid_head * stride_z_head
                + offs_p[None, :] * stride_z_dim,
                mask=row_valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            ).to(tl.float32)
            dacs_row = tl.load(
                DA_CS + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
                + pid_chunk * stride_dacs_chunk + rows * stride_dacs_csize,
                mask=(rows >= 0) & (rows < CHUNK_SIZE), other=0.0,
            ).to(tl.float32)
            dpre_row = dout_row * z_row * tl.sigmoid(z_row)
            dM = tl.dot(c_row, tl.trans(b_col)) * tl.dot(
                dpre_row.to(x_row.dtype), tl.trans(x_col)
            )
            matrix_valid = row_valid[:, None] & col_valid[None, :]
            causal = rows[:, None] >= cols[None, :]
            segsum = tl.where(
                matrix_valid & causal,
                dacs_row[:, None] - dacs_col[None, :],
                -float("inf"),
            )
            dM *= tl.exp(tl.minimum(segsum, 0.0))
            dM = tl.where(matrix_valid & causal, dM, 0.0)
            direct += tl.sum(dM, axis=0)
            strict = rows[:, None] > cols[None, :]
            g = tl.where(matrix_valid & strict, dM * dt_col[None, :], 0.0)
            reverse_g = tl.cumsum(g, axis=0, reverse=True) + carry[None, :]
            ddA_rows = tl.sum(
                tl.where(matrix_valid & strict, reverse_g, 0.0), axis=1
            )
            out_ptrs = PROCESSED_DDT + ddt_base + rows * stride_ddt_csize
            old = tl.load(
                out_ptrs,
                mask=(rows >= 0) & (rows < CHUNK_SIZE),
                other=0.0,
            ).to(tl.float32)
            tl.store(out_ptrs, old + a * ddA_rows, mask=row_valid)
            dt_row = tl.load(
                DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head
                + pid_chunk * stride_dt_chunk + rows * stride_dt_csize,
                mask=(rows >= 0) & (rows < CHUNK_SIZE), other=0.0,
            ).to(tl.float32)
            dA_value += tl.sum(tl.where(row_valid, ddA_rows * dt_row, 0.0))
            carry += tl.sum(g, axis=0)

        col_ptrs = PROCESSED_DDT + ddt_base + cols * stride_ddt_csize
        old_col = tl.load(col_ptrs, mask=cols < CHUNK_SIZE, other=0.0).to(tl.float32)
        tl.store(col_ptrs, old_col + direct, mask=col_valid)

    tl.store(
        DA_CHUNK + pid_batch * stride_dachunk_batch + pid_head * stride_dachunk_head
        + pid_chunk * stride_dachunk_chunk,
        dA_value,
    )


@triton.jit
def _mamba2_bwd_dq_cached_next_decay_cp_kernel(
    PROCESSED_DDT,
    DT,
    GRAD_SCALE_INTERMEDIATE,
    DA_CHUNK,
    A,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_gsi_batch,
    stride_gsi_head,
    stride_gsi_chunk,
    stride_gsi_csize,
    stride_dachunk_batch,
    stride_dachunk_head,
    stride_dachunk_chunk,
    stride_a_head,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Add cached dM-next contributions with one deterministic chunk owner."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    offs = tl.arange(0, BLOCK_CHUNK)
    logical = offs < CHUNK_SIZE
    valid = logical & (pid_chunk * CHUNK_SIZE + offs < seqlen)

    ddt_ptrs = (
        PROCESSED_DDT
        + pid_batch * stride_ddt_batch
        + pid_head * stride_ddt_head
        + pid_chunk * stride_ddt_chunk
        + offs * stride_ddt_csize
    )
    dt_ptrs = (
        DT
        + pid_batch * stride_dt_batch
        + pid_head * stride_dt_head
        + pid_chunk * stride_dt_chunk
        + offs * stride_dt_csize
    )
    gsi_ptrs = (
        GRAD_SCALE_INTERMEDIATE
        + pid_batch * stride_gsi_batch
        + pid_head * stride_gsi_head
        + pid_chunk * stride_gsi_chunk
        + offs * stride_gsi_csize
    )
    ddt = tl.load(ddt_ptrs, mask=logical, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptrs, mask=logical, other=0.0).to(tl.float32)
    gsi = tl.load(gsi_ptrs, mask=logical, other=0.0).to(tl.float32)
    dt = tl.where(valid, dt, 0.0)
    gsi = tl.where(valid, gsi, 0.0)

    dt_prefix = tl.cumsum(dt, axis=0)
    gsi_suffix = tl.cumsum(gsi, axis=0, reverse=True)
    a = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    tl.store(ddt_ptrs, tl.where(valid, ddt + a * gsi_suffix, 0.0), mask=logical)

    dA_ptr = (
        DA_CHUNK
        + pid_batch * stride_dachunk_batch
        + pid_head * stride_dachunk_head
        + pid_chunk * stride_dachunk_chunk
    )
    dA = tl.load(dA_ptr).to(tl.float32)
    tl.store(dA_ptr, dA + tl.sum(dt_prefix * gsi))


def _mamba2_bwd_dq_cached_next_decay_cp_triton(
    processed_ddt: torch.Tensor,
    dt: torch.Tensor,
    grad_scale_intermediate: torch.Tensor,
    dA_chunk: torch.Tensor,
    A: torch.Tensor,
    seqlen: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map cached dM-next GSI into the compact true-CP decay accumulators."""

    if any(tensor.ndim != 4 for tensor in (processed_ddt, dt, grad_scale_intermediate)):
        raise ValueError("processed_ddt, dt, and grad_scale_intermediate must be 4D")
    if dA_chunk.ndim != 3 or A.ndim != 1:
        raise ValueError("dA_chunk must be 3D and A must be 1D")
    if chunk_size <= 0 or seqlen <= 0:
        raise ValueError("chunk_size and seqlen must be positive")
    if processed_ddt.shape != dt.shape or grad_scale_intermediate.shape != dt.shape:
        raise ValueError("compact ddt, dt, and GSI tensors must have the same shape")

    folded_batch, nheads, nchunks, stored_chunk_size = dt.shape
    if stored_chunk_size != chunk_size:
        raise ValueError("the compact tensor width must equal chunk_size")
    if nchunks != triton.cdiv(seqlen, chunk_size):
        raise ValueError("the compact chunk count does not match seqlen")
    if dA_chunk.shape != (folded_batch, nheads, nchunks):
        raise ValueError("dA_chunk has the wrong compact shape")
    if A.shape != (nheads,):
        raise ValueError(f"A must have shape {(nheads,)}")

    tensors = (processed_ddt, dt, grad_scale_intermediate, dA_chunk, A)
    if not processed_ddt.is_cuda:
        raise ValueError("the true-CP dM-next mapper requires CUDA tensors")
    if any(tensor.device != processed_ddt.device for tensor in tensors[1:]):
        raise ValueError("all dM-next mapper inputs must share a device")
    if any(tensor.dtype is not torch.float32 for tensor in tensors):
        raise ValueError("all dM-next mapper inputs must be FP32")

    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    _mamba2_bwd_dq_cached_next_decay_cp_kernel[(nheads, folded_batch, nchunks)](
        processed_ddt,
        dt,
        grad_scale_intermediate,
        dA_chunk,
        A,
        processed_ddt.stride(0),
        processed_ddt.stride(1),
        processed_ddt.stride(2),
        processed_ddt.stride(3),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        dt.stride(3),
        grad_scale_intermediate.stride(0),
        grad_scale_intermediate.stride(1),
        grad_scale_intermediate.stride(2),
        grad_scale_intermediate.stride(3),
        dA_chunk.stride(0),
        dA_chunk.stride(1),
        dA_chunk.stride(2),
        A.stride(0),
        seqlen,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
        num_warps=4,
        num_stages=1,
    )
    return processed_ddt, dA_chunk


@triton.jit
def _mamba2_bwd_dq_next_decay_cp_kernel(
    C,
    DOUT,
    Z,
    DT,
    DA_CS,
    CHUNK_STATES,
    A,
    PROCESSED_DDT,
    DA_CHUNK,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dim,
    stride_dout_stream, stride_dout_seqlen, stride_dout_batch,
    stride_dout_head, stride_dout_dim,
    stride_z_stream, stride_z_seqlen, stride_z_batch, stride_z_head, stride_z_dim,
    stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
    stride_dacs_batch, stride_dacs_head, stride_dacs_chunk, stride_dacs_csize,
    stride_states_batch, stride_states_head, stride_states_chunk,
    stride_states_p, stride_states_n,
    stride_a_head,
    stride_ddt_batch, stride_ddt_head, stride_ddt_chunk, stride_ddt_csize,
    stride_dachunk_batch, stride_dachunk_head, stride_dachunk_chunk,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Add the next-state decay term without materializing grad_scale."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_batch // BATCH
    batch_idx = pid_batch - stream * BATCH
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    state = tl.load(
        CHUNK_STATES + pid_batch * stride_states_batch + pid_head * stride_states_head
        + pid_chunk * stride_states_chunk + offs_p[:, None] * stride_states_p
        + offs_n[None, :] * stride_states_n,
        mask=(offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n),
        other=0.0,
    ).to(tl.float32)
    a = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    dA_value = 0.0
    dt_carry = 0.0

    for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
        rows = row_start + tl.arange(0, BLOCK_M)
        seq = chunk_start + rows
        valid = (rows < CHUNK_SIZE) & (seq < seqlen)
        c = tl.load(
            C + pid_batch * stride_c_batch + seq[:, None] * stride_c_seqlen
            + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim,
            mask=valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
        )
        dout = tl.load(
            DOUT + stream * stride_dout_stream + seq[:, None] * stride_dout_seqlen
            + batch_idx * stride_dout_batch + pid_head * stride_dout_head
            + offs_p[None, :] * stride_dout_dim,
            mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
        ).to(tl.float32)
        z = tl.load(
            Z + stream * stride_z_stream + seq[:, None] * stride_z_seqlen
            + batch_idx * stride_z_batch + pid_head * stride_z_head
            + offs_p[None, :] * stride_z_dim,
            mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
        ).to(tl.float32)
        dacs = tl.load(
            DA_CS + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
            + pid_chunk * stride_dacs_chunk + rows * stride_dacs_csize,
            mask=rows < CHUNK_SIZE, other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head
            + pid_chunk * stride_dt_chunk + rows * stride_dt_csize,
            mask=rows < CHUNK_SIZE, other=0.0,
        ).to(tl.float32)
        dpre = dout * z * tl.sigmoid(z)
        next_unscaled = tl.dot(dpre.to(c.dtype), state.to(c.dtype))
        grad_scale = tl.sum(next_unscaled * c, axis=1) * tl.exp(dacs)
        grad_scale = tl.where(valid, grad_scale, 0.0)
        dt = tl.where(valid, dt, 0.0)
        dt_prefix = tl.cumsum(dt, axis=0) + dt_carry
        dA_value += tl.sum(dt_prefix * grad_scale)
        dt_carry += tl.sum(dt)

    grad_carry = 0.0
    for row_rev in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
        row_start = CHUNK_SIZE - BLOCK_M - row_rev
        rows = row_start + tl.arange(0, BLOCK_M)
        seq = chunk_start + rows
        valid = (rows >= 0) & (rows < CHUNK_SIZE) & (seq < seqlen)
        c = tl.load(
            C + pid_batch * stride_c_batch + seq[:, None] * stride_c_seqlen
            + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim,
            mask=valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
        )
        dout = tl.load(
            DOUT + stream * stride_dout_stream + seq[:, None] * stride_dout_seqlen
            + batch_idx * stride_dout_batch + pid_head * stride_dout_head
            + offs_p[None, :] * stride_dout_dim,
            mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
        ).to(tl.float32)
        z = tl.load(
            Z + stream * stride_z_stream + seq[:, None] * stride_z_seqlen
            + batch_idx * stride_z_batch + pid_head * stride_z_head
            + offs_p[None, :] * stride_z_dim,
            mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
        ).to(tl.float32)
        dacs = tl.load(
            DA_CS + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
            + pid_chunk * stride_dacs_chunk + rows * stride_dacs_csize,
            mask=(rows >= 0) & (rows < CHUNK_SIZE), other=0.0,
        ).to(tl.float32)
        dpre = dout * z * tl.sigmoid(z)
        next_unscaled = tl.dot(dpre.to(c.dtype), state.to(c.dtype))
        grad_scale = tl.sum(next_unscaled * c, axis=1) * tl.exp(dacs)
        grad_scale = tl.where(valid, grad_scale, 0.0)
        suffix = tl.cumsum(grad_scale, axis=0, reverse=True) + grad_carry
        out_ptrs = (
            PROCESSED_DDT + pid_batch * stride_ddt_batch + pid_head * stride_ddt_head
            + pid_chunk * stride_ddt_chunk + rows * stride_ddt_csize
        )
        old = tl.load(
            out_ptrs, mask=(rows >= 0) & (rows < CHUNK_SIZE), other=0.0
        ).to(tl.float32)
        tl.store(out_ptrs, old + a * suffix, mask=valid)
        grad_carry += tl.sum(grad_scale)

    dA_ptr = (
        DA_CHUNK + pid_batch * stride_dachunk_batch + pid_head * stride_dachunk_head
        + pid_chunk * stride_dachunk_chunk
    )
    tl.store(dA_ptr, tl.load(dA_ptr).to(tl.float32) + dA_value)


def mamba2_bwd_dq_cp_triton(
    C: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, dstate)
    B: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, dstate)
    x: torch.Tensor,  # (folded_batch, seqlen, nheads, headdim)
    dout: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim)
    z: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim)
    dt: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs_rev: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    # Unsplit: (streams, batch, nheads, P, N).
    # Split: (folded_batch, nsubsequences, nheads, P, N).
    initial_states: torch.Tensor,
    A: torch.Tensor,  # (nheads,)
    D: torch.Tensor,  # (nheads,)
    chunk_size: int = 64,
    subsequence_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dC/dz, chunk starts, and compact current/next decay gradients.

    Through chunk 64, dQ emits temporary FP32 grad_scale_intermediate for a
    lightweight next-decay mapper. Larger chunks retain next-term
    recomputation. Both paths keep the same chunk-state handoff and fixed
    return structure.
    """

    if C.ndim != 4 or B.ndim != 4 or x.ndim != 4:
        raise ValueError("C, B, and x must be 4D")
    if dout.ndim != 5 or z.ndim != 5 or initial_states.ndim != 5:
        raise ValueError("dout, z, and initial_states must be 5D")
    if any(tensor.ndim != 4 for tensor in (dt, dA_cs, dA_cs_rev)):
        raise ValueError("dt, dA_cs, and dA_cs_rev must be 4D")
    if C.shape != B.shape:
        raise ValueError("C and B must have the same shape")
    if dout.shape != z.shape:
        raise ValueError("dout and z must have the same shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if subsequence_size < 0:
        raise ValueError("subsequence_size must be nonnegative")
    if subsequence_size and (
        chunk_size != 64
        or subsequence_size < chunk_size
        or subsequence_size % chunk_size
    ):
        raise ValueError(
            "subsequence dQ requires chunk_size=64 and a subsequence_size multiple of 64"
        )

    folded_batch, seqlen, nheads_qk, dstate = C.shape
    x_batch, x_seqlen, nheads, headdim = x.shape
    streams, dout_seqlen, batch, dout_nheads, dout_headdim = dout.shape
    if streams != 2:
        raise ValueError("true-CP dQ requires exactly two local streams")
    if (x_batch, x_seqlen) != (folded_batch, seqlen):
        raise ValueError("C, B, and x must share folded batch and sequence dimensions")
    if (dout_seqlen, dout_nheads, dout_headdim) != (seqlen, nheads, headdim):
        raise ValueError("dout and z dimensions must match x")
    if folded_batch != streams * batch:
        raise ValueError("folded C/B/x batch must equal streams * batch")
    if nheads_qk <= 0 or nheads <= 0 or nheads % nheads_qk != 0:
        raise ValueError("the number of value heads must be divisible by C/B heads")
    if dstate <= 0 or headdim <= 0:
        raise ValueError("state and head dimensions must be positive")
    nchunks = triton.cdiv(seqlen, chunk_size)
    subsequence_chunks = subsequence_size // chunk_size if subsequence_size else 0
    nsubsequences = (
        triton.cdiv(nchunks, subsequence_chunks) if subsequence_chunks else 1
    )
    expected_initial_shape = (
        (folded_batch, nsubsequences, nheads, headdim, dstate)
        if subsequence_chunks
        else (streams, batch, nheads, headdim, dstate)
    )
    if initial_states.shape != expected_initial_shape:
        raise ValueError(
            f"initial_states must have shape {expected_initial_shape}, "
            f"got {tuple(initial_states.shape)}"
        )

    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    for name, tensor in (("dt", dt), ("dA_cs", dA_cs), ("dA_cs_rev", dA_cs_rev)):
        if tensor.shape != decay_shape:
            raise ValueError(f"{name} must have shape {decay_shape}, got {tuple(tensor.shape)}")
    if A.shape != (nheads,) or D.shape != (nheads,):
        raise ValueError(f"A and D must each have shape {(nheads,)}")

    tensors = (C, B, x, dout, z, dt, dA_cs, dA_cs_rev, initial_states, A, D)
    if not C.is_cuda:
        raise ValueError("mamba2_bwd_dq_cp_triton requires CUDA tensors")
    if any(tensor.device != C.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if any(tensor.dtype != C.dtype for tensor in (B, x, dout, z)):
        raise ValueError("C, B, x, dout, and z must have the same dtype")
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must use floating-point dtypes")

    dC_partial = torch.empty(
        folded_batch, seqlen, nheads, dstate, device=C.device, dtype=C.dtype
    )
    dz = torch.empty_like(dout)
    if subsequence_chunks == 1:
        chunk_start_states = initial_states.permute(0, 2, 1, 3, 4)
    else:
        chunk_start_states = torch.empty(
            folded_batch,
            nheads,
            nchunks,
            headdim,
            dstate,
            device=C.device,
            dtype=torch.float32,
        )
    processed_ddt = torch.empty(decay_shape, device=C.device, dtype=torch.float32)
    dA_chunk = torch.empty(
        folded_batch, nheads, nchunks, device=C.device, dtype=torch.float32
    )
    grad_scale_intermediate = (
        torch.empty(decay_shape, device=C.device, dtype=torch.float32)
        if chunk_size <= 64
        else None
    )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(dstate))
    grid = (
        (nheads, folded_batch, nsubsequences)
        if subsequence_chunks
        else (nheads, folded_batch)
    )
    kernel_args = (
        C,
        B,
        x,
        dout,
        z,
        dt,
        dA_cs,
        dA_cs_rev,
        initial_states,
        D,
        dC_partial,
        dz,
        chunk_start_states,
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1), dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        initial_states.stride(0), initial_states.stride(1), initial_states.stride(2),
        initial_states.stride(3), initial_states.stride(4),
        D.stride(0),
        dC_partial.stride(0), dC_partial.stride(1), dC_partial.stride(2), dC_partial.stride(3),
        dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3), dz.stride(4),
        chunk_start_states.stride(0), chunk_start_states.stride(1),
        chunk_start_states.stride(2), chunk_start_states.stride(3),
        chunk_start_states.stride(4),
        seqlen,
        headdim,
        dstate,
        nheads_qk,
    )
    if chunk_size <= 64:
        def launch_dq(launch_grid, subsequence_offset, full_subsequence, exact_shape):
            _mamba2_bwd_dq_cp_kernel[launch_grid](
                *kernel_args,
                A,
                processed_ddt,
                dA_chunk,
                grad_scale_intermediate,
                A.stride(0),
                processed_ddt.stride(0),
                processed_ddt.stride(1),
                processed_ddt.stride(2),
                processed_ddt.stride(3),
                dA_chunk.stride(0),
                dA_chunk.stride(1),
                dA_chunk.stride(2),
                grad_scale_intermediate.stride(0),
                grad_scale_intermediate.stride(1),
                grad_scale_intermediate.stride(2),
                grad_scale_intermediate.stride(3),
                BATCH=batch,
                CHUNK_SIZE=chunk_size,
                BLOCK_CHUNK=block_chunk,
                BLOCK_P=block_p,
                BLOCK_N=block_n,
                SUBSEQUENCE_CHUNKS=subsequence_chunks,
                SUBSEQUENCE_OFFSET=subsequence_offset,
                SKIP_CHUNK_STATE_STORE=(subsequence_chunks == 1),
                FULL_SUBSEQUENCE=full_subsequence,
                EXACT_SHAPE=exact_shape,
            )

        if subsequence_chunks:
            nfull = seqlen // subsequence_size
            has_tail = seqlen % subsequence_size != 0
            if nfull:
                launch_dq(
                    (nheads, folded_batch, nfull),
                    0,
                    True,
                    headdim == block_p and dstate == block_n,
                )
            if has_tail:
                launch_dq((nheads, folded_batch, 1), nfull, False, False)
        else:
            launch_dq(grid, 0, False, False)
    else:
        _mamba2_bwd_dq_cp_large_chunk_kernel[grid](
            *kernel_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )

    decay_grid = (nheads, folded_batch, nchunks)
    if chunk_size > 64:
        current_decay_args = (
            C, B, x, dout, z, dt, dA_cs, A, processed_ddt, dA_chunk,
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2),
            dout.stride(3), dout.stride(4),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
            A.stride(0),
            processed_ddt.stride(0), processed_ddt.stride(1),
            processed_ddt.stride(2), processed_ddt.stride(3),
            dA_chunk.stride(0), dA_chunk.stride(1), dA_chunk.stride(2),
            seqlen, headdim, dstate, nheads_qk,
        )
        _mamba2_bwd_dq_current_decay_cp_tiled_kernel[decay_grid](
            *current_decay_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            BLOCK_M=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=1,
        )

    if chunk_size <= 64:
        processed_ddt, dA_chunk = _mamba2_bwd_dq_cached_next_decay_cp_triton(
            processed_ddt,
            dt,
            grad_scale_intermediate,
            dA_chunk,
            A,
            seqlen,
            chunk_size,
        )
        del grad_scale_intermediate
    else:
        _mamba2_bwd_dq_next_decay_cp_kernel[decay_grid](
            C, dout, z, dt, dA_cs, chunk_start_states, A,
            processed_ddt, dA_chunk,
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2),
            dout.stride(3), dout.stride(4),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
            chunk_start_states.stride(0), chunk_start_states.stride(1),
            chunk_start_states.stride(2), chunk_start_states.stride(3),
            chunk_start_states.stride(4),
            A.stride(0),
            processed_ddt.stride(0), processed_ddt.stride(1),
            processed_ddt.stride(2), processed_ddt.stride(3),
            dA_chunk.stride(0), dA_chunk.stride(1), dA_chunk.stride(2),
            seqlen, headdim, dstate, nheads_qk,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            BLOCK_M=32,
            num_warps=4,
            num_stages=1,
        )

    if nheads == nheads_qk:
        dC = dC_partial
    else:
        dC = torch.empty_like(C)
        block_size = 128
        reduce_grid = (
            folded_batch
            * seqlen
            * nheads_qk
            * triton.cdiv(dstate, block_size),
        )
        reduce_head_groups_kernel[reduce_grid](
            dC_partial,
            dC,
            folded_batch,
            seqlen,
            nheads,
            nheads_qk,
            dstate,
            dC_partial.stride(0),
            dC_partial.stride(1),
            dC_partial.stride(2),
            dC_partial.stride(3),
            dC.stride(0),
            dC.stride(1),
            dC.stride(2),
            dC.stride(3),
            BLOCK_SIZE=block_size,
        )
    return dC, dz, chunk_start_states, processed_ddt, dA_chunk


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK", "BLOCK_P", "BLOCK_N"],
)
@triton.jit
def _mamba2_bwd_dk_dv_cp_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    DT,
    DA_CS,
    DA_CS_REV,
    SEGSUM,
    CHUNK_STATES,
    FINAL_DSTATES,
    A,
    D,
    DB_PARTIAL,
    DX,
    DD_PARTIAL,
    DA_PARTIAL,
    INITIAL_DDT,
    DDACSREV,
    GRAD_SCALE,
    DM_CURR,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_chunk,
    stride_dacsrev_csize,
    stride_segsum_batch,
    stride_segsum_head,
    stride_segsum_chunk,
    stride_segsum_row,
    stride_segsum_col,
    stride_states_batch,
    stride_states_head,
    stride_states_chunk,
    stride_states_p,
    stride_states_n,
    stride_final_stream,
    stride_final_batch,
    stride_final_head,
    stride_final_p,
    stride_final_n,
    stride_a_head,
    stride_d_head,
    stride_db_batch,
    stride_db_seqlen,
    stride_db_head,
    stride_db_dim,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_dim,
    stride_dd_head,
    stride_dd_batch,
    stride_da_head,
    stride_da_batch,
    stride_initial_ddt_batch,
    stride_initial_ddt_head,
    stride_initial_ddt_chunk,
    stride_initial_ddt_csize,
    stride_ddacs_batch,
    stride_ddacs_head,
    stride_ddacs_chunk,
    stride_ddacs_csize,
    stride_grad_scale_batch,
    stride_grad_scale_head,
    stride_grad_scale_chunk,
    stride_grad_scale_csize,
    stride_dm_batch,
    stride_dm_head,
    stride_dm_chunk,
    stride_dm_row,
    stride_dm_col,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute true-CP dB/dx and decay-gradient workspaces in reverse."""

    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    c_offset = pid_folded_batch * stride_c_batch + qk_head * stride_c_head
    b_offset = pid_folded_batch * stride_b_batch + qk_head * stride_b_head
    x_offset = pid_folded_batch * stride_x_batch + pid_head * stride_x_head
    dout_offset = (
        stream * stride_dout_stream
        + batch_idx * stride_dout_batch
        + pid_head * stride_dout_head
    )
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    dt_offset = pid_folded_batch * stride_dt_batch + pid_head * stride_dt_head
    dacs_offset = pid_folded_batch * stride_dacs_batch + pid_head * stride_dacs_head
    dacsrev_offset = (
        pid_folded_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    )
    segsum_offset = (
        pid_folded_batch * stride_segsum_batch + pid_head * stride_segsum_head
    )
    states_offset = (
        pid_folded_batch * stride_states_batch + pid_head * stride_states_head
    )
    final_offset = (
        stream * stride_final_stream
        + batch_idx * stride_final_batch
        + pid_head * stride_final_head
    )
    db_offset = pid_folded_batch * stride_db_batch + pid_head * stride_db_head
    dx_offset = pid_folded_batch * stride_dx_batch + pid_head * stride_dx_head

    offs_chunk = tl.arange(0, BLOCK_CHUNK)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    state_mask = (offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n)
    final_ptrs = (
        FINAL_DSTATES
        + final_offset
        + offs_p[:, None] * stride_final_p
        + offs_n[None, :] * stride_final_n
    )
    acc_dstates = tl.load(final_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    dD_partial = 0.0
    dA_partial = 0.0
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    d_value = tl.load(D + pid_head * stride_d_head).to(tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    for chunk_idx_rev in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_rev
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seq = chunk_start + offs_chunk
        logical_chunk_mask = offs_chunk < CHUNK_SIZE
        seq_mask = logical_chunk_mask & (offs_seq < seqlen)

        dt_block = tl.load(
            DT
            + dt_offset
            + chunk_idx * stride_dt_chunk
            + offs_chunk * stride_dt_csize,
            mask=logical_chunk_mask,
            other=0.0,
        ).to(tl.float32)
        dA_cs = tl.load(
            DA_CS
            + dacs_offset
            + chunk_idx * stride_dacs_chunk
            + offs_chunk * stride_dacs_csize,
            mask=logical_chunk_mask,
            other=0.0,
        ).to(tl.float32)
        dA_cs_rev = tl.load(
            DA_CS_REV
            + dacsrev_offset
            + chunk_idx * stride_dacsrev_chunk
            + offs_chunk * stride_dacsrev_csize,
            mask=logical_chunk_mask,
            other=0.0,
        ).to(tl.float32)
        dA_cs_last = tl.load(
            DA_CS
            + dacs_offset
            + chunk_idx * stride_dacs_chunk
            + (CHUNK_SIZE - 1) * stride_dacs_csize
        ).to(tl.float32)

        c_ptrs = (
            C
            + c_offset
            + offs_seq[:, None] * stride_c_seqlen
            + offs_n[None, :] * stride_c_dim
        )
        qk_mask = seq_mask[:, None] & (offs_n[None, :] < dstate_n)
        c_block = tl.load(c_ptrs, mask=qk_mask, other=0.0)
        b_ptrs = (
            B
            + b_offset
            + offs_seq[:, None] * stride_b_seqlen
            + offs_n[None, :] * stride_b_dim
        )
        b_block = tl.load(b_ptrs, mask=qk_mask, other=0.0)
        x_ptrs = (
            X
            + x_offset
            + offs_seq[:, None] * stride_x_seqlen
            + offs_p[None, :] * stride_x_dim
        )
        value_mask = seq_mask[:, None] & (offs_p[None, :] < headdim_p)
        x_block = tl.load(x_ptrs, mask=value_mask, other=0.0)
        dout_ptrs = (
            DOUT
            + dout_offset
            + offs_seq[:, None] * stride_dout_seqlen
            + offs_p[None, :] * stride_dout_dim
        )
        dout_block = tl.load(dout_ptrs, mask=value_mask, other=0.0).to(tl.float32)
        z_ptrs = (
            Z
            + z_offset
            + offs_seq[:, None] * stride_z_seqlen
            + offs_p[None, :] * stride_z_dim
        )
        z_block = tl.load(z_ptrs, mask=value_mask, other=0.0).to(tl.float32)
        dpre = dout_block * z_block * tl.sigmoid(z_block)

        segsum_ptrs = (
            SEGSUM
            + segsum_offset
            + chunk_idx * stride_segsum_chunk
            + offs_chunk[:, None] * stride_segsum_row
            + offs_chunk[None, :] * stride_segsum_col
        )
        segsum_mask = logical_chunk_mask[:, None] & logical_chunk_mask[None, :]
        segsum = tl.load(segsum_ptrs, mask=segsum_mask, other=-float("inf")).to(
            tl.float32
        )
        exp_segsum = tl.exp(segsum)

        chunk_state_ptrs = (
            CHUNK_STATES
            + states_offset
            + chunk_idx * stride_states_chunk
            + offs_p[:, None] * stride_states_p
            + offs_n[None, :] * stride_states_n
        )
        chunk_state = tl.load(chunk_state_ptrs, mask=state_mask, other=0.0).to(
            tl.float32
        )

        dpre_dot = dpre.to(x_block.dtype)
        exp_dA_cs_rev = tl.exp(dA_cs_rev)
        state_scale = exp_dA_cs_rev * dt_block

        db_state = tl.dot(x_block, acc_dstates.to(x_block.dtype))
        acc_db = db_state * state_scale[:, None]
        x_dpre = tl.dot(x_block, tl.trans(dpre_dot))
        x_dpre *= tl.trans(exp_segsum) * dt_block[:, None]
        acc_db += tl.dot(x_dpre.to(c_block.dtype), c_block)

        dx_state_unscaled = tl.dot(
            b_block, tl.trans(acc_dstates.to(b_block.dtype))
        )
        acc_dx = dx_state_unscaled * state_scale[:, None]
        bc = tl.dot(b_block, tl.trans(c_block))
        bc *= tl.trans(exp_segsum) * dt_block[:, None]
        acc_dx += tl.dot(bc.to(dpre_dot.dtype), dpre_dot)
        acc_dx += dpre * d_value

        db_ptrs = (
            DB_PARTIAL
            + db_offset
            + offs_seq[:, None] * stride_db_seqlen
            + offs_n[None, :] * stride_db_dim
        )
        tl.store(db_ptrs, acc_db.to(b_block.dtype), mask=qk_mask)
        dx_ptrs = (
            DX
            + dx_offset
            + offs_seq[:, None] * stride_dx_seqlen
            + offs_p[None, :] * stride_dx_dim
        )
        tl.store(dx_ptrs, acc_dx.to(x_block.dtype), mask=value_mask)
        dD_partial += tl.sum(dpre * x_block.to(tl.float32))

        next_unscaled = tl.dot(
            dpre_dot, chunk_state.to(dpre_dot.dtype)
        )
        grad_scale = tl.sum(next_unscaled * c_block, axis=1) * tl.exp(dA_cs)
        grad_scale_dt = tl.sum(
            dx_state_unscaled.to(tl.float32) * x_block.to(tl.float32), axis=1
        )
        ddA_cs_rev = grad_scale_dt * state_scale
        initial_ddt = grad_scale_dt * exp_dA_cs_rev

        cb = tl.dot(c_block, tl.trans(b_block))
        dpre_x = tl.dot(dpre_dot, tl.trans(x_block))
        dM_curr = cb * dpre_x * exp_segsum

        chunk_transition_grad = (
            tl.sum(chunk_state * acc_dstates) * tl.exp(dA_cs_last)
        )
        initial_ddt += tl.where(seq_mask, a_value * chunk_transition_grad, 0.0)
        dA_partial += tl.sum(dt_block * chunk_transition_grad)

        initial_ddt_ptrs = (
            INITIAL_DDT
            + pid_folded_batch * stride_initial_ddt_batch
            + pid_head * stride_initial_ddt_head
            + chunk_idx * stride_initial_ddt_chunk
            + offs_chunk * stride_initial_ddt_csize
        )
        tl.store(initial_ddt_ptrs, initial_ddt, mask=logical_chunk_mask)
        ddacs_ptrs = (
            DDACSREV
            + pid_folded_batch * stride_ddacs_batch
            + pid_head * stride_ddacs_head
            + chunk_idx * stride_ddacs_chunk
            + offs_chunk * stride_ddacs_csize
        )
        tl.store(ddacs_ptrs, ddA_cs_rev, mask=logical_chunk_mask)
        grad_scale_ptrs = (
            GRAD_SCALE
            + pid_folded_batch * stride_grad_scale_batch
            + pid_head * stride_grad_scale_head
            + chunk_idx * stride_grad_scale_chunk
            + offs_chunk * stride_grad_scale_csize
        )
        tl.store(grad_scale_ptrs, grad_scale, mask=logical_chunk_mask)
        dm_ptrs = (
            DM_CURR
            + pid_folded_batch * stride_dm_batch
            + pid_head * stride_dm_head
            + chunk_idx * stride_dm_chunk
            + offs_chunk[:, None] * stride_dm_row
            + offs_chunk[None, :] * stride_dm_col
        )
        tl.store(dm_ptrs, dM_curr, mask=segsum_mask)

        c_scaled = c_block * tl.exp(dA_cs)[:, None]
        acc_dstates = tl.exp(dA_cs_last) * acc_dstates + tl.dot(
            tl.trans(dpre_dot), c_scaled.to(dpre_dot.dtype)
        )

    tl.store(
        DD_PARTIAL
        + pid_head * stride_dd_head
        + pid_folded_batch * stride_dd_batch,
        dD_partial,
    )
    tl.store(
        DA_PARTIAL
        + pid_head * stride_da_head
        + pid_folded_batch * stride_da_batch,
        dA_partial,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_P", "BLOCK_N"],
)
@triton.jit
def _mamba2_bwd_dk_dv_cp_large_chunk_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    DT,
    DA_CS,
    DA_CS_REV,
    SEGSUM,
    CHUNK_STATES,
    FINAL_DSTATES,
    A,
    D,
    DB_PARTIAL,
    DX,
    DD_PARTIAL,
    DA_PARTIAL,
    INITIAL_DDT,
    DDACSREV,
    GRAD_SCALE,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_chunk,
    stride_dacs_csize,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_chunk,
    stride_dacsrev_csize,
    stride_segsum_batch,
    stride_segsum_head,
    stride_segsum_chunk,
    stride_segsum_row,
    stride_segsum_col,
    stride_states_batch,
    stride_states_head,
    stride_states_chunk,
    stride_states_p,
    stride_states_n,
    stride_final_stream,
    stride_final_batch,
    stride_final_head,
    stride_final_p,
    stride_final_n,
    stride_a_head,
    stride_d_head,
    stride_db_batch,
    stride_db_seqlen,
    stride_db_head,
    stride_db_dim,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_dim,
    stride_dd_head,
    stride_dd_batch,
    stride_da_head,
    stride_da_batch,
    stride_initial_ddt_batch,
    stride_initial_ddt_head,
    stride_initial_ddt_chunk,
    stride_initial_ddt_csize,
    stride_ddacs_batch,
    stride_ddacs_head,
    stride_ddacs_chunk,
    stride_ddacs_csize,
    stride_grad_scale_batch,
    stride_grad_scale_head,
    stride_grad_scale_chunk,
    stride_grad_scale_csize,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Large-chunk dB/dx with tiled transposed sequence interactions."""

    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    c_offset = pid_folded_batch * stride_c_batch + qk_head * stride_c_head
    b_offset = pid_folded_batch * stride_b_batch + qk_head * stride_b_head
    x_offset = pid_folded_batch * stride_x_batch + pid_head * stride_x_head
    dout_offset = (
        stream * stride_dout_stream
        + batch_idx * stride_dout_batch
        + pid_head * stride_dout_head
    )
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    dt_offset = pid_folded_batch * stride_dt_batch + pid_head * stride_dt_head
    dacs_offset = pid_folded_batch * stride_dacs_batch + pid_head * stride_dacs_head
    dacsrev_offset = (
        pid_folded_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    )
    segsum_offset = (
        pid_folded_batch * stride_segsum_batch + pid_head * stride_segsum_head
    )
    states_offset = pid_folded_batch * stride_states_batch + pid_head * stride_states_head
    final_offset = (
        stream * stride_final_stream
        + batch_idx * stride_final_batch
        + pid_head * stride_final_head
    )
    db_offset = pid_folded_batch * stride_db_batch + pid_head * stride_db_head
    dx_offset = pid_folded_batch * stride_dx_batch + pid_head * stride_dx_head

    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    state_mask = (offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n)
    acc_dstates = tl.load(
        FINAL_DSTATES
        + final_offset
        + offs_p[:, None] * stride_final_p
        + offs_n[None, :] * stride_final_n,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    dD_partial = 0.0
    dA_partial = 0.0
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    d_value = tl.load(D + pid_head * stride_d_head).to(tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    for chunk_idx_rev in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_rev
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_dt_offset = dt_offset + chunk_idx * stride_dt_chunk
        chunk_dacs_offset = dacs_offset + chunk_idx * stride_dacs_chunk
        chunk_dacsrev_offset = dacsrev_offset + chunk_idx * stride_dacsrev_chunk
        chunk_segsum_offset = segsum_offset + chunk_idx * stride_segsum_chunk
        dA_cs_last = tl.load(
            DA_CS + chunk_dacs_offset + (CHUNK_SIZE - 1) * stride_dacs_csize
        ).to(tl.float32)
        chunk_state = tl.load(
            CHUNK_STATES
            + states_offset
            + chunk_idx * stride_states_chunk
            + offs_p[:, None] * stride_states_p
            + offs_n[None, :] * stride_states_n,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        chunk_transition_grad = tl.sum(chunk_state * acc_dstates) * tl.exp(dA_cs_last)

        for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            offs_m = row_start + tl.arange(0, BLOCK_M)
            offs_seq_m = chunk_start + offs_m
            row_logical = offs_m < CHUNK_SIZE
            row_valid = row_logical & (offs_seq_m < seqlen)
            qk_row_mask = row_valid[:, None] & (offs_n[None, :] < dstate_n)
            value_row_mask = row_valid[:, None] & (offs_p[None, :] < headdim_p)

            b_row = tl.load(
                B
                + b_offset
                + offs_seq_m[:, None] * stride_b_seqlen
                + offs_n[None, :] * stride_b_dim,
                mask=qk_row_mask,
                other=0.0,
            )
            x_row = tl.load(
                X
                + x_offset
                + offs_seq_m[:, None] * stride_x_seqlen
                + offs_p[None, :] * stride_x_dim,
                mask=value_row_mask,
                other=0.0,
            )
            dout_row = tl.load(
                DOUT
                + dout_offset
                + offs_seq_m[:, None] * stride_dout_seqlen
                + offs_p[None, :] * stride_dout_dim,
                mask=value_row_mask,
                other=0.0,
            ).to(tl.float32)
            z_row = tl.load(
                Z
                + z_offset
                + offs_seq_m[:, None] * stride_z_seqlen
                + offs_p[None, :] * stride_z_dim,
                mask=value_row_mask,
                other=0.0,
            ).to(tl.float32)
            dpre_row = dout_row * z_row * tl.sigmoid(z_row)
            dpre_row_dot = dpre_row.to(x_row.dtype)
            dt_row = tl.load(
                DT + chunk_dt_offset + offs_m * stride_dt_csize,
                mask=row_logical,
                other=0.0,
            ).to(tl.float32)
            dA_row = tl.load(
                DA_CS + chunk_dacs_offset + offs_m * stride_dacs_csize,
                mask=row_logical,
                other=0.0,
            ).to(tl.float32)
            dA_rev_row = tl.load(
                DA_CS_REV + chunk_dacsrev_offset + offs_m * stride_dacsrev_csize,
                mask=row_logical,
                other=0.0,
            ).to(tl.float32)
            state_scale = tl.exp(dA_rev_row) * dt_row

            db_state = tl.dot(x_row, acc_dstates.to(x_row.dtype))
            acc_db = db_state * state_scale[:, None]
            dx_state_unscaled = tl.dot(
                b_row, tl.trans(acc_dstates.to(b_row.dtype))
            )
            acc_dx = dx_state_unscaled * state_scale[:, None]

            for query_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
                offs_k = query_start + tl.arange(0, BLOCK_K)
                offs_seq_k = chunk_start + offs_k
                query_logical = offs_k < CHUNK_SIZE
                query_valid = query_logical & (offs_seq_k < seqlen)
                c_block = tl.load(
                    C
                    + c_offset
                    + offs_seq_k[:, None] * stride_c_seqlen
                    + offs_n[None, :] * stride_c_dim,
                    mask=query_valid[:, None] & (offs_n[None, :] < dstate_n),
                    other=0.0,
                )
                dout_block = tl.load(
                    DOUT
                    + dout_offset
                    + offs_seq_k[:, None] * stride_dout_seqlen
                    + offs_p[None, :] * stride_dout_dim,
                    mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                ).to(tl.float32)
                z_block = tl.load(
                    Z
                    + z_offset
                    + offs_seq_k[:, None] * stride_z_seqlen
                    + offs_p[None, :] * stride_z_dim,
                    mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                ).to(tl.float32)
                dpre_block = dout_block * z_block * tl.sigmoid(z_block)
                dpre_dot = dpre_block.to(x_row.dtype)
                segsum = tl.load(
                    SEGSUM
                    + chunk_segsum_offset
                    + offs_k[:, None] * stride_segsum_row
                    + offs_m[None, :] * stride_segsum_col,
                    mask=query_logical[:, None] & row_logical[None, :],
                    other=-float("inf"),
                ).to(tl.float32)
                weights_t = tl.trans(tl.exp(segsum)) * dt_row[:, None]
                x_dpre = tl.dot(x_row, tl.trans(dpre_dot)) * weights_t
                acc_db += tl.dot(x_dpre.to(c_block.dtype), c_block)
                bc = tl.dot(b_row, tl.trans(c_block)) * weights_t
                acc_dx += tl.dot(bc.to(dpre_dot.dtype), dpre_dot)

            acc_dx += dpre_row * d_value
            tl.store(
                DB_PARTIAL
                + db_offset
                + offs_seq_m[:, None] * stride_db_seqlen
                + offs_n[None, :] * stride_db_dim,
                acc_db.to(b_row.dtype),
                mask=qk_row_mask,
            )
            tl.store(
                DX
                + dx_offset
                + offs_seq_m[:, None] * stride_dx_seqlen
                + offs_p[None, :] * stride_dx_dim,
                acc_dx.to(x_row.dtype),
                mask=value_row_mask,
            )
            dD_partial += tl.sum(dpre_row * x_row.to(tl.float32))

            c_row = tl.load(
                C
                + c_offset
                + offs_seq_m[:, None] * stride_c_seqlen
                + offs_n[None, :] * stride_c_dim,
                mask=qk_row_mask,
                other=0.0,
            )
            next_unscaled = tl.dot(dpre_row_dot, chunk_state.to(dpre_row_dot.dtype))
            grad_scale = tl.sum(next_unscaled * c_row, axis=1) * tl.exp(dA_row)
            grad_scale_dt = tl.sum(
                dx_state_unscaled.to(tl.float32) * x_row.to(tl.float32), axis=1
            )
            ddA_cs_rev = grad_scale_dt * state_scale
            initial_ddt = grad_scale_dt * tl.exp(dA_rev_row)
            initial_ddt += tl.where(row_valid, a_value * chunk_transition_grad, 0.0)
            dA_partial += tl.sum(dt_row * chunk_transition_grad)

            tl.store(
                INITIAL_DDT
                + pid_folded_batch * stride_initial_ddt_batch
                + pid_head * stride_initial_ddt_head
                + chunk_idx * stride_initial_ddt_chunk
                + offs_m * stride_initial_ddt_csize,
                initial_ddt,
                mask=row_logical,
            )
            tl.store(
                DDACSREV
                + pid_folded_batch * stride_ddacs_batch
                + pid_head * stride_ddacs_head
                + chunk_idx * stride_ddacs_chunk
                + offs_m * stride_ddacs_csize,
                ddA_cs_rev,
                mask=row_logical,
            )
            tl.store(
                GRAD_SCALE
                + pid_folded_batch * stride_grad_scale_batch
                + pid_head * stride_grad_scale_head
                + chunk_idx * stride_grad_scale_chunk
                + offs_m * stride_grad_scale_csize,
                grad_scale,
                mask=row_logical,
            )

        acc_dstates *= tl.exp(dA_cs_last)
        for query_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
            offs_k = query_start + tl.arange(0, BLOCK_K)
            offs_seq_k = chunk_start + offs_k
            query_logical = offs_k < CHUNK_SIZE
            query_valid = query_logical & (offs_seq_k < seqlen)
            c_block = tl.load(
                C
                + c_offset
                + offs_seq_k[:, None] * stride_c_seqlen
                + offs_n[None, :] * stride_c_dim,
                mask=query_valid[:, None] & (offs_n[None, :] < dstate_n),
                other=0.0,
            )
            dout_block = tl.load(
                DOUT
                + dout_offset
                + offs_seq_k[:, None] * stride_dout_seqlen
                + offs_p[None, :] * stride_dout_dim,
                mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                other=0.0,
            ).to(tl.float32)
            z_block = tl.load(
                Z
                + z_offset
                + offs_seq_k[:, None] * stride_z_seqlen
                + offs_p[None, :] * stride_z_dim,
                mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                other=0.0,
            ).to(tl.float32)
            dpre_dot = (dout_block * z_block * tl.sigmoid(z_block)).to(c_block.dtype)
            dA_block = tl.load(
                DA_CS + chunk_dacs_offset + offs_k * stride_dacs_csize,
                mask=query_logical,
                other=0.0,
            ).to(tl.float32)
            c_scaled = c_block * tl.exp(dA_block)[:, None].to(c_block.dtype)
            acc_dstates += tl.dot(tl.trans(dpre_dot), c_scaled)

    tl.store(
        DD_PARTIAL + pid_head * stride_dd_head + pid_folded_batch * stride_dd_batch,
        dD_partial,
    )
    tl.store(
        DA_PARTIAL + pid_head * stride_da_head + pid_folded_batch * stride_da_batch,
        dA_partial,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=1),
    ],
    key=[
        "CHUNK_SIZE", "BLOCK_P", "BLOCK_N", "SUBSEQUENCE_CHUNKS",
        "FULL_SUBSEQUENCE", "EXACT_STATE",
    ],
    restore_value=["PROCESSED_DDT", "DA_CHUNK"],
    prune_configs_by={"early_config_prune": _prune_dk_dv_compact_configs},
)
@triton.jit
def _mamba2_bwd_dk_dv_compact_cp_kernel(
    C, B, X, DOUT, Z, DT, DA_CS, DA_CS_REV,
    CHUNK_STATES, FINAL_DSTATES, A, D,
    DB_PARTIAL, DX, DD_PARTIAL, PROCESSED_DDT, DA_CHUNK, DDACSREV,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dout_stream, stride_dout_seqlen, stride_dout_batch,
    stride_dout_head, stride_dout_dim,
    stride_z_stream, stride_z_seqlen, stride_z_batch, stride_z_head, stride_z_dim,
    stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
    stride_dacs_batch, stride_dacs_head, stride_dacs_chunk, stride_dacs_csize,
    stride_dacsrev_batch, stride_dacsrev_head,
    stride_dacsrev_chunk, stride_dacsrev_csize,
    stride_states_batch, stride_states_head, stride_states_chunk,
    stride_states_p, stride_states_n,
    stride_final_stream, stride_final_batch, stride_final_head,
    stride_final_p, stride_final_n,
    stride_a_head,
    stride_d_head,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_dim,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_dim,
    stride_dd_head, stride_dd_batch, stride_dd_subsequence,
    stride_ddt_batch, stride_ddt_head, stride_ddt_chunk, stride_ddt_csize,
    stride_dachunk_batch, stride_dachunk_head, stride_dachunk_chunk,
    stride_ddacs_batch, stride_ddacs_head, stride_ddacs_chunk, stride_ddacs_csize,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SUBSEQUENCE_CHUNKS: tl.constexpr = 0,
    REVERSE_SLOT_OFFSET: tl.constexpr = 0,
    ORIGINAL_SUBSEQUENCE_BASE: tl.constexpr = 0,
    ORIGINAL_SUBSEQUENCE_COUNT: tl.constexpr = 0,
    FULL_SUBSEQUENCE: tl.constexpr = False,
    EXACT_STATE: tl.constexpr = False,
):
    """Compute dB/dx/dD and offload compact previous-decay intermediates."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_subsequence_rev_local = tl.program_id(2)
    pid_subsequence_rev = pid_subsequence_rev_local + REVERSE_SLOT_OFFSET
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_batch // BATCH
    batch_idx = pid_batch - stream * BATCH
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    if EXACT_STATE:
        state_mask = (offs_p[:, None] < BLOCK_P) & (offs_n[None, :] < BLOCK_N)
    else:
        state_mask = (offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n)
    if SUBSEQUENCE_CHUNKS > 0:
        final_offset = (
            pid_batch * stride_final_stream
            + pid_subsequence_rev * stride_final_batch
            + pid_head * stride_final_head
        )
    else:
        final_offset = (
            stream * stride_final_stream
            + batch_idx * stride_final_batch
            + pid_head * stride_final_head
        )
    final_ptrs = (
        FINAL_DSTATES + final_offset + offs_p[:, None] * stride_final_p
        + offs_n[None, :] * stride_final_n
    )
    if FULL_SUBSEQUENCE and EXACT_STATE:
        acc_dstates = tl.load(final_ptrs).to(tl.float32)
    else:
        acc_dstates = tl.load(final_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    d_value = tl.load(D + pid_head * stride_d_head).to(tl.float32)
    dD_value = 0.0
    total_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    if SUBSEQUENCE_CHUNKS > 0:
        num_chunks = SUBSEQUENCE_CHUNKS
        first_chunk = (
            ORIGINAL_SUBSEQUENCE_BASE + ORIGINAL_SUBSEQUENCE_COUNT
            - 1 - pid_subsequence_rev_local
        ) * SUBSEQUENCE_CHUNKS
    else:
        num_chunks = total_chunks
        first_chunk = 0

    for chunk_rev in range(num_chunks):
        chunk_idx = first_chunk + num_chunks - 1 - chunk_rev
        if FULL_SUBSEQUENCE:
            valid_chunk = True
        else:
            valid_chunk = chunk_idx < total_chunks
        chunk_start = chunk_idx * CHUNK_SIZE
        dacs_base = (
            pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
            + chunk_idx * stride_dacs_chunk
        )
        dacsrev_base = (
            pid_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
            + chunk_idx * stride_dacsrev_chunk
        )
        dt_base = (
            pid_batch * stride_dt_batch + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
        )
        chunk_total_ptr = (
            DA_CS + dacs_base + (CHUNK_SIZE - 1) * stride_dacs_csize
        )
        chunk_state_ptrs = (
            CHUNK_STATES + pid_batch * stride_states_batch
            + pid_head * stride_states_head + chunk_idx * stride_states_chunk
            + offs_p[:, None] * stride_states_p + offs_n[None, :] * stride_states_n
        )
        if FULL_SUBSEQUENCE and EXACT_STATE:
            chunk_total = tl.load(chunk_total_ptr).to(tl.float32)
            chunk_state = tl.load(chunk_state_ptrs).to(tl.float32)
        else:
            chunk_total = tl.load(
                chunk_total_ptr, mask=valid_chunk, other=0.0
            ).to(tl.float32)
            chunk_state = tl.load(
                chunk_state_ptrs, mask=state_mask & valid_chunk, other=0.0
            ).to(tl.float32)
        transition = tl.sum(chunk_state * acc_dstates) * tl.exp(chunk_total)
        dt_sum = 0.0

        # The chunk-64 schedule has a single row tile. Keep its dpre live so
        # the direct D path and terminal-adjoint update share one sigmoid/load
        # pass. Smaller row tiles retain the resource-safe replay schedule.
        if BLOCK_M == CHUNK_SIZE:
            chunk_rows = tl.arange(0, BLOCK_M)
            chunk_seq = chunk_start + chunk_rows
            if FULL_SUBSEQUENCE:
                chunk_valid = True
            else:
                chunk_valid = valid_chunk & (chunk_rows < CHUNK_SIZE) & (chunk_seq < seqlen)
            dout_chunk_ptrs = (
                DOUT + stream * stride_dout_stream
                + chunk_seq[:, None] * stride_dout_seqlen
                + batch_idx * stride_dout_batch + pid_head * stride_dout_head
                + offs_p[None, :] * stride_dout_dim
            )
            z_chunk_ptrs = (
                Z + stream * stride_z_stream
                + chunk_seq[:, None] * stride_z_seqlen
                + batch_idx * stride_z_batch + pid_head * stride_z_head
                + offs_p[None, :] * stride_z_dim
            )
            if FULL_SUBSEQUENCE and EXACT_STATE:
                dout_chunk = tl.load(dout_chunk_ptrs).to(tl.float32)
                z_chunk = tl.load(z_chunk_ptrs).to(tl.float32)
            elif FULL_SUBSEQUENCE:
                dout_chunk = tl.load(
                    dout_chunk_ptrs,
                    mask=offs_p[None, :] < headdim_p,
                    other=0.0,
                ).to(tl.float32)
                z_chunk = tl.load(
                    z_chunk_ptrs,
                    mask=offs_p[None, :] < headdim_p,
                    other=0.0,
                ).to(tl.float32)
            else:
                dout_chunk = tl.load(
                    dout_chunk_ptrs,
                    mask=chunk_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                ).to(tl.float32)
                z_chunk = tl.load(
                    z_chunk_ptrs,
                    mask=chunk_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                ).to(tl.float32)
            dpre_chunk = dout_chunk * z_chunk * tl.sigmoid(z_chunk)

        for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            rows = row_start + tl.arange(0, BLOCK_M)
            seq_rows = chunk_start + rows
            if FULL_SUBSEQUENCE:
                row_valid = True
            else:
                row_valid = valid_chunk & (rows < CHUNK_SIZE) & (seq_rows < seqlen)
            b_row_ptrs = (
                B + pid_batch * stride_b_batch + seq_rows[:, None] * stride_b_seqlen
                + qk_head * stride_b_head + offs_n[None, :] * stride_b_dim
            )
            x_row_ptrs = (
                X + pid_batch * stride_x_batch + seq_rows[:, None] * stride_x_seqlen
                + pid_head * stride_x_head + offs_p[None, :] * stride_x_dim
            )
            if FULL_SUBSEQUENCE and EXACT_STATE:
                b_row = tl.load(b_row_ptrs)
                x_row = tl.load(x_row_ptrs)
            elif FULL_SUBSEQUENCE:
                b_row = tl.load(
                    b_row_ptrs, mask=offs_n[None, :] < dstate_n, other=0.0
                )
                x_row = tl.load(
                    x_row_ptrs, mask=offs_p[None, :] < headdim_p, other=0.0
                )
            else:
                b_row = tl.load(
                    b_row_ptrs,
                    mask=row_valid[:, None] & (offs_n[None, :] < dstate_n),
                    other=0.0,
                )
                x_row = tl.load(
                    x_row_ptrs,
                    mask=row_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                )
            if BLOCK_M == CHUNK_SIZE:
                dpre_row = dpre_chunk
            else:
                dout_row = tl.load(
                    DOUT + stream * stride_dout_stream
                    + seq_rows[:, None] * stride_dout_seqlen
                    + batch_idx * stride_dout_batch + pid_head * stride_dout_head
                    + offs_p[None, :] * stride_dout_dim,
                    mask=row_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                ).to(tl.float32)
                z_row = tl.load(
                    Z + stream * stride_z_stream
                    + seq_rows[:, None] * stride_z_seqlen
                    + batch_idx * stride_z_batch + pid_head * stride_z_head
                    + offs_p[None, :] * stride_z_dim,
                    mask=row_valid[:, None] & (offs_p[None, :] < headdim_p),
                    other=0.0,
                ).to(tl.float32)
                dpre_row = dout_row * z_row * tl.sigmoid(z_row)
            if FULL_SUBSEQUENCE:
                dt_row = tl.load(
                    DT + dt_base + rows * stride_dt_csize
                ).to(tl.float32)
                dacsrev_row = tl.load(
                    DA_CS_REV + dacsrev_base + rows * stride_dacsrev_csize
                ).to(tl.float32)
            else:
                dt_row = tl.load(
                    DT + dt_base + rows * stride_dt_csize,
                    mask=valid_chunk & (rows < CHUNK_SIZE), other=0.0,
                ).to(tl.float32)
                dacsrev_row = tl.load(
                    DA_CS_REV + dacsrev_base + rows * stride_dacsrev_csize,
                    mask=valid_chunk & (rows < CHUNK_SIZE), other=0.0,
                ).to(tl.float32)
            state_scale = tl.exp(dacsrev_row) * dt_row
            acc_db = tl.dot(x_row, acc_dstates.to(x_row.dtype)) * state_scale[:, None]
            acc_dv_intermediate = tl.dot(
                b_row, tl.trans(acc_dstates.to(b_row.dtype))
            )
            acc_dx = acc_dv_intermediate * state_scale[:, None]
            grad_scale_dt = tl.sum(
                acc_dv_intermediate.to(tl.float32) * x_row.to(tl.float32), axis=1
            )
            initial_ddt = grad_scale_dt * tl.exp(dacsrev_row)
            ddA_cs_rev = grad_scale_dt * state_scale

            for query_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
                queries = query_start + tl.arange(0, BLOCK_K)
                seq_queries = chunk_start + queries
                if FULL_SUBSEQUENCE:
                    query_valid = True
                else:
                    query_valid = valid_chunk & (queries < CHUNK_SIZE) & (seq_queries < seqlen)
                c_query_ptrs = (
                    C + pid_batch * stride_c_batch
                    + seq_queries[:, None] * stride_c_seqlen
                    + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim
                )
                if FULL_SUBSEQUENCE and EXACT_STATE:
                    c_query = tl.load(c_query_ptrs)
                elif FULL_SUBSEQUENCE:
                    c_query = tl.load(
                        c_query_ptrs,
                        mask=offs_n[None, :] < dstate_n,
                        other=0.0,
                    )
                else:
                    c_query = tl.load(
                        c_query_ptrs,
                        mask=query_valid[:, None] & (offs_n[None, :] < dstate_n),
                        other=0.0,
                    )
                if BLOCK_M == CHUNK_SIZE and BLOCK_K == CHUNK_SIZE:
                    dpre_query_dot = dpre_chunk.to(x_row.dtype)
                else:
                    dout_query = tl.load(
                        DOUT + stream * stride_dout_stream
                        + seq_queries[:, None] * stride_dout_seqlen
                        + batch_idx * stride_dout_batch
                        + pid_head * stride_dout_head
                        + offs_p[None, :] * stride_dout_dim,
                        mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                        other=0.0,
                    ).to(tl.float32)
                    z_query = tl.load(
                        Z + stream * stride_z_stream
                        + seq_queries[:, None] * stride_z_seqlen
                        + batch_idx * stride_z_batch + pid_head * stride_z_head
                        + offs_p[None, :] * stride_z_dim,
                        mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                        other=0.0,
                    ).to(tl.float32)
                    dpre_query = dout_query * z_query * tl.sigmoid(z_query)
                    dpre_query_dot = dpre_query.to(x_row.dtype)
                if FULL_SUBSEQUENCE:
                    dacs_query = tl.load(
                        DA_CS + dacs_base + queries * stride_dacs_csize
                    ).to(tl.float32)
                    dacs_row = tl.load(
                        DA_CS + dacs_base + rows * stride_dacs_csize
                    ).to(tl.float32)
                else:
                    dacs_query = tl.load(
                        DA_CS + dacs_base + queries * stride_dacs_csize,
                        mask=valid_chunk & (queries < CHUNK_SIZE), other=0.0,
                    ).to(tl.float32)
                    dacs_row = tl.load(
                        DA_CS + dacs_base + rows * stride_dacs_csize,
                        mask=valid_chunk & (rows < CHUNK_SIZE), other=0.0,
                    ).to(tl.float32)
                causal = queries[:, None] >= rows[None, :]
                if FULL_SUBSEQUENCE:
                    segsum = tl.where(
                        causal,
                        dacs_query[:, None] - dacs_row[None, :],
                        -float("inf"),
                    )
                else:
                    valid_matrix = query_valid[:, None] & row_valid[None, :]
                    segsum = tl.where(
                        valid_matrix & causal,
                        dacs_query[:, None] - dacs_row[None, :],
                        -float("inf"),
                    )
                weights_t = tl.trans(tl.exp(tl.minimum(segsum, 0.0)))
                weights_t *= dt_row[:, None]
                x_dpre = tl.dot(
                    x_row, tl.trans(dpre_query_dot)
                ) * weights_t
                acc_db += tl.dot(x_dpre.to(c_query.dtype), c_query)
                bc = tl.dot(b_row, tl.trans(c_query)) * weights_t
                acc_dx += tl.dot(bc.to(dpre_query_dot.dtype), dpre_query_dot)

            acc_dx += dpre_row * d_value
            db_ptrs = (
                DB_PARTIAL + pid_batch * stride_db_batch
                + seq_rows[:, None] * stride_db_seqlen + pid_head * stride_db_head
                + offs_n[None, :] * stride_db_dim
            )
            dx_ptrs = (
                DX + pid_batch * stride_dx_batch + seq_rows[:, None] * stride_dx_seqlen
                + pid_head * stride_dx_head + offs_p[None, :] * stride_dx_dim
            )
            if FULL_SUBSEQUENCE and EXACT_STATE:
                tl.store(db_ptrs, acc_db.to(b_row.dtype))
                tl.store(dx_ptrs, acc_dx.to(x_row.dtype))
            elif FULL_SUBSEQUENCE:
                tl.store(
                    db_ptrs,
                    acc_db.to(b_row.dtype),
                    mask=offs_n[None, :] < dstate_n,
                )
                tl.store(
                    dx_ptrs,
                    acc_dx.to(x_row.dtype),
                    mask=offs_p[None, :] < headdim_p,
                )
            else:
                tl.store(
                    db_ptrs,
                    acc_db.to(b_row.dtype),
                    mask=row_valid[:, None] & (offs_n[None, :] < dstate_n),
                )
                tl.store(
                    dx_ptrs,
                    acc_dx.to(x_row.dtype),
                    mask=row_valid[:, None] & (offs_p[None, :] < headdim_p),
                )
            dD_value += tl.sum(dpre_row * x_row.to(tl.float32))
            ddt_ptrs = (
                PROCESSED_DDT + pid_batch * stride_ddt_batch
                + pid_head * stride_ddt_head + chunk_idx * stride_ddt_chunk
                + rows * stride_ddt_csize
            )
            ddacsrev_ptrs = (
                DDACSREV + pid_batch * stride_ddacs_batch
                + pid_head * stride_ddacs_head + chunk_idx * stride_ddacs_chunk
                + rows * stride_ddacs_csize
            )
            if FULL_SUBSEQUENCE:
                previous_ddt = tl.load(ddt_ptrs).to(tl.float32)
                tl.store(
                    ddt_ptrs, previous_ddt + initial_ddt + a_value * transition
                )
                tl.store(ddacsrev_ptrs, ddA_cs_rev)
                dt_sum += tl.sum(dt_row)
            else:
                previous_ddt = tl.load(
                    ddt_ptrs, mask=valid_chunk & (rows < CHUNK_SIZE), other=0.0
                ).to(tl.float32)
                tl.store(
                    ddt_ptrs,
                    tl.where(
                        row_valid,
                        previous_ddt + initial_ddt + a_value * transition,
                        previous_ddt,
                    ),
                    mask=valid_chunk & (rows < CHUNK_SIZE),
                )
                tl.store(
                    ddacsrev_ptrs,
                    tl.where(row_valid, ddA_cs_rev, 0.0),
                    mask=valid_chunk & (rows < CHUNK_SIZE),
                )
                dt_sum += tl.sum(tl.where(row_valid, dt_row, 0.0))

        dA_ptr = (
            DA_CHUNK + pid_batch * stride_dachunk_batch
            + pid_head * stride_dachunk_head + chunk_idx * stride_dachunk_chunk
        )
        if FULL_SUBSEQUENCE:
            tl.store(
                dA_ptr, tl.load(dA_ptr).to(tl.float32) + transition * dt_sum
            )
        else:
            tl.store(
                dA_ptr,
                tl.load(dA_ptr).to(tl.float32) + transition * dt_sum,
                mask=valid_chunk,
            )

        if SUBSEQUENCE_CHUNKS == 0 or chunk_rev < SUBSEQUENCE_CHUNKS - 1:
            acc_dstates *= tl.exp(chunk_total)
            if BLOCK_M == CHUNK_SIZE:
                queries = tl.arange(0, BLOCK_M)
                seq_queries = chunk_start + queries
                if FULL_SUBSEQUENCE:
                    query_valid = True
                else:
                    query_valid = valid_chunk & (queries < CHUNK_SIZE) & (seq_queries < seqlen)
                c_query_ptrs = (
                    C + pid_batch * stride_c_batch
                    + seq_queries[:, None] * stride_c_seqlen
                    + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim
                )
                if FULL_SUBSEQUENCE and EXACT_STATE:
                    c_query = tl.load(c_query_ptrs)
                elif FULL_SUBSEQUENCE:
                    c_query = tl.load(
                        c_query_ptrs,
                        mask=offs_n[None, :] < dstate_n,
                        other=0.0,
                    )
                else:
                    c_query = tl.load(
                        c_query_ptrs,
                        mask=query_valid[:, None] & (offs_n[None, :] < dstate_n),
                        other=0.0,
                    )
                if FULL_SUBSEQUENCE:
                    dacs_query = tl.load(
                        DA_CS + dacs_base + queries * stride_dacs_csize
                    ).to(tl.float32)
                else:
                    dacs_query = tl.load(
                        DA_CS + dacs_base + queries * stride_dacs_csize,
                        mask=valid_chunk & (queries < CHUNK_SIZE), other=0.0,
                    ).to(tl.float32)
                c_scaled = c_query * tl.exp(dacs_query)[:, None].to(c_query.dtype)
                acc_dstates += tl.dot(tl.trans(dpre_chunk.to(c_query.dtype)), c_scaled)
            else:
                for query_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
                    queries = query_start + tl.arange(0, BLOCK_K)
                    seq_queries = chunk_start + queries
                    query_valid = valid_chunk & (queries < CHUNK_SIZE) & (seq_queries < seqlen)
                    c_query = tl.load(
                        C + pid_batch * stride_c_batch
                        + seq_queries[:, None] * stride_c_seqlen
                        + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim,
                        mask=query_valid[:, None] & (offs_n[None, :] < dstate_n),
                        other=0.0,
                    )
                    dout_query = tl.load(
                        DOUT + stream * stride_dout_stream
                        + seq_queries[:, None] * stride_dout_seqlen
                        + batch_idx * stride_dout_batch + pid_head * stride_dout_head
                        + offs_p[None, :] * stride_dout_dim,
                        mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                        other=0.0,
                    ).to(tl.float32)
                    z_query = tl.load(
                        Z + stream * stride_z_stream
                        + seq_queries[:, None] * stride_z_seqlen
                        + batch_idx * stride_z_batch + pid_head * stride_z_head
                        + offs_p[None, :] * stride_z_dim,
                        mask=query_valid[:, None] & (offs_p[None, :] < headdim_p),
                        other=0.0,
                    ).to(tl.float32)
                    dacs_query = tl.load(
                        DA_CS + dacs_base + queries * stride_dacs_csize,
                        mask=valid_chunk & (queries < CHUNK_SIZE), other=0.0,
                    ).to(tl.float32)
                    dpre_query = (
                        dout_query * z_query * tl.sigmoid(z_query)
                    ).to(c_query.dtype)
                    c_scaled = c_query * tl.exp(dacs_query)[:, None].to(c_query.dtype)
                    acc_dstates += tl.dot(tl.trans(dpre_query), c_scaled)

    tl.store(
        DD_PARTIAL + pid_head * stride_dd_head + pid_batch * stride_dd_batch
        + pid_subsequence_rev * stride_dd_subsequence,
        dD_value,
    )


@triton.jit
def _mamba2_bwd_dk_dv_cached_prev_decay_cp_kernel(
    PROCESSED_DDT,
    DT,
    DDACSREV,
    DA_CHUNK,
    A,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_ddacs_batch,
    stride_ddacs_head,
    stride_ddacs_chunk,
    stride_ddacs_csize,
    stride_dachunk_batch,
    stride_dachunk_head,
    stride_dachunk_chunk,
    stride_a_head,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Map cached previous-state decay into compact ddt/dA accumulators."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    offs = tl.arange(0, BLOCK_CHUNK)
    logical = offs < CHUNK_SIZE
    seq = pid_chunk * CHUNK_SIZE + offs
    valid = logical & (seq < seqlen)
    ddt_base = (
        pid_batch * stride_ddt_batch + pid_head * stride_ddt_head
        + pid_chunk * stride_ddt_chunk
    )
    dt_base = (
        pid_batch * stride_dt_batch + pid_head * stride_dt_head
        + pid_chunk * stride_dt_chunk
    )
    ddacs_base = (
        pid_batch * stride_ddacs_batch + pid_head * stride_ddacs_head
        + pid_chunk * stride_ddacs_chunk
    )
    ddt = tl.load(
        PROCESSED_DDT + ddt_base + offs * stride_ddt_csize,
        mask=logical, other=0.0,
    ).to(tl.float32)
    ddA_cs_rev = tl.load(
        DDACSREV + ddacs_base + offs * stride_ddacs_csize,
        mask=logical, other=0.0,
    ).to(tl.float32)
    ddA_cs_rev = tl.where(valid, ddA_cs_rev, 0.0)

    # Legacy dM_Prev mapping, expressed as directional shifted scans. Shifting
    # before scanning makes both operations exclusive without subtraction.
    shifted_dt = tl.load(
        DT + dt_base + (offs + 1) * stride_dt_csize,
        mask=(offs + 1 < CHUNK_SIZE) & (seq + 1 < seqlen),
        other=0.0,
    ).to(tl.float32)
    exclusive_suffix_dt = tl.cumsum(shifted_dt, axis=0, reverse=True)
    shifted_ddA = tl.load(
        DDACSREV + ddacs_base + (offs - 1) * stride_ddacs_csize,
        mask=(offs > 0) & valid,
        other=0.0,
    ).to(tl.float32)
    exclusive_prefix_ddA = tl.cumsum(shifted_ddA, axis=0)

    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    tl.store(
        PROCESSED_DDT + ddt_base + offs * stride_ddt_csize,
        ddt + a_value * exclusive_prefix_ddA,
        mask=valid,
    )
    dA_ptr = (
        DA_CHUNK + pid_batch * stride_dachunk_batch
        + pid_head * stride_dachunk_head + pid_chunk * stride_dachunk_chunk
    )
    tl.store(
        dA_ptr,
        tl.load(dA_ptr).to(tl.float32)
        + tl.sum(ddA_cs_rev * exclusive_suffix_dt),
    )


@triton.jit
def _mamba2_bwd_dm_curr_inplace_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    SEGSUM_DM,
    stride_c_batch,
    stride_c_seqlen,
    stride_c_head,
    stride_c_dim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_dim,
    stride_dout_stream,
    stride_dout_seqlen,
    stride_dout_batch,
    stride_dout_head,
    stride_dout_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_dm_batch,
    stride_dm_head,
    stride_dm_chunk,
    stride_dm_row,
    stride_dm_col,
    seqlen,
    headdim_p,
    dstate_n,
    nheads,
    nheads_qk,
    nchunks,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Overwrite one segsum tile with its dM-current tile."""

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_bhc = tl.program_id(2)
    chunk_idx = pid_bhc % nchunks
    pid_bh = pid_bhc // nchunks
    pid_head = pid_bh % nheads
    pid_folded_batch = pid_bh // nheads
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    chunk_start = chunk_idx * CHUNK_SIZE
    seq_m = chunk_start + offs_m
    seq_k = chunk_start + offs_k
    valid_m = (offs_m < CHUNK_SIZE) & (seq_m < seqlen)
    valid_k = (offs_k < CHUNK_SIZE) & (seq_k < seqlen)

    c_block = tl.load(
        C
        + pid_folded_batch * stride_c_batch
        + qk_head * stride_c_head
        + seq_m[:, None] * stride_c_seqlen
        + offs_n[None, :] * stride_c_dim,
        mask=valid_m[:, None] & (offs_n[None, :] < dstate_n),
        other=0.0,
    )
    b_block = tl.load(
        B
        + pid_folded_batch * stride_b_batch
        + qk_head * stride_b_head
        + seq_k[:, None] * stride_b_seqlen
        + offs_n[None, :] * stride_b_dim,
        mask=valid_k[:, None] & (offs_n[None, :] < dstate_n),
        other=0.0,
    )
    x_block = tl.load(
        X
        + pid_folded_batch * stride_x_batch
        + pid_head * stride_x_head
        + seq_k[:, None] * stride_x_seqlen
        + offs_p[None, :] * stride_x_dim,
        mask=valid_k[:, None] & (offs_p[None, :] < headdim_p),
        other=0.0,
    )
    dout_block = tl.load(
        DOUT
        + stream * stride_dout_stream
        + batch_idx * stride_dout_batch
        + pid_head * stride_dout_head
        + seq_m[:, None] * stride_dout_seqlen
        + offs_p[None, :] * stride_dout_dim,
        mask=valid_m[:, None] & (offs_p[None, :] < headdim_p),
        other=0.0,
    ).to(tl.float32)
    z_block = tl.load(
        Z
        + stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
        + seq_m[:, None] * stride_z_seqlen
        + offs_p[None, :] * stride_z_dim,
        mask=valid_m[:, None] & (offs_p[None, :] < headdim_p),
        other=0.0,
    ).to(tl.float32)
    dpre = (dout_block * z_block * tl.sigmoid(z_block)).to(x_block.dtype)

    dm_ptrs = (
        SEGSUM_DM
        + pid_folded_batch * stride_dm_batch
        + pid_head * stride_dm_head
        + chunk_idx * stride_dm_chunk
        + offs_m[:, None] * stride_dm_row
        + offs_k[None, :] * stride_dm_col
    )
    tile_mask = (offs_m[:, None] < CHUNK_SIZE) & (offs_k[None, :] < CHUNK_SIZE)
    segsum = tl.load(dm_ptrs, mask=tile_mask, other=-float("inf")).to(tl.float32)
    dM_curr = tl.dot(c_block, tl.trans(b_block))
    dM_curr *= tl.dot(dpre, tl.trans(x_block))
    dM_curr *= tl.exp(segsum)
    tl.store(dm_ptrs, dM_curr, mask=tile_mask)


@triton.jit
def _mamba2_bwd_dk_dv_decay_cp_kernel(
    C,
    B,
    X,
    DOUT,
    Z,
    DT,
    DA_CS,
    DA_CS_REV,
    CHUNK_STATES,
    FINAL_DSTATES,
    A,
    PROCESSED_DDT,
    DA_CHUNK,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dout_stream, stride_dout_seqlen, stride_dout_batch,
    stride_dout_head, stride_dout_dim,
    stride_z_stream, stride_z_seqlen, stride_z_batch, stride_z_head, stride_z_dim,
    stride_dt_batch, stride_dt_head, stride_dt_chunk, stride_dt_csize,
    stride_dacs_batch, stride_dacs_head, stride_dacs_chunk, stride_dacs_csize,
    stride_dacsrev_batch, stride_dacsrev_head,
    stride_dacsrev_chunk, stride_dacsrev_csize,
    stride_states_batch, stride_states_head, stride_states_chunk,
    stride_states_p, stride_states_n,
    stride_final_stream, stride_final_batch, stride_final_head,
    stride_final_p, stride_final_n,
    stride_a_head,
    stride_ddt_batch, stride_ddt_head, stride_ddt_chunk, stride_ddt_csize,
    stride_dachunk_batch, stride_dachunk_head, stride_dachunk_chunk,
    seqlen,
    headdim_p,
    dstate_n,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Legacy replay oracle for previous/state decay contributions."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    qk_head = pid_head // (nheads // nheads_qk)
    stream = pid_batch // BATCH
    batch_idx = pid_batch - stream * BATCH
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    state_mask = (offs_p[:, None] < headdim_p) & (offs_n[None, :] < dstate_n)
    acc_dstates = tl.load(
        FINAL_DSTATES + stream * stride_final_stream + batch_idx * stride_final_batch
        + pid_head * stride_final_head + offs_p[:, None] * stride_final_p
        + offs_n[None, :] * stride_final_n,
        mask=state_mask, other=0.0,
    ).to(tl.float32)
    a = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_rev in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_rev
        chunk_start = chunk_idx * CHUNK_SIZE
        dt_base = (
            pid_batch * stride_dt_batch + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
        )
        dacs_base = (
            pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
            + chunk_idx * stride_dacs_chunk
        )
        dacsrev_base = (
            pid_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
            + chunk_idx * stride_dacsrev_chunk
        )
        ddt_base = (
            pid_batch * stride_ddt_batch + pid_head * stride_ddt_head
            + chunk_idx * stride_ddt_chunk
        )
        chunk_total = tl.load(
            DA_CS + dacs_base + (CHUNK_SIZE - 1) * stride_dacs_csize
        ).to(tl.float32)
        chunk_state = tl.load(
            CHUNK_STATES + pid_batch * stride_states_batch
            + pid_head * stride_states_head + chunk_idx * stride_states_chunk
            + offs_p[:, None] * stride_states_p + offs_n[None, :] * stride_states_n,
            mask=state_mask, other=0.0,
        ).to(tl.float32)
        transition = tl.sum(chunk_state * acc_dstates) * tl.exp(chunk_total)
        prefix_carry = 0.0
        dt_sum = 0.0

        # Shift-right before the forward scan. The carry contains only complete
        # earlier row blocks, so no subtraction from a chunk total is needed.
        for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            local = tl.arange(0, BLOCK_M)
            rows = row_start + local
            seq = chunk_start + rows
            valid = (rows < CHUNK_SIZE) & (seq < seqlen)
            b = tl.load(
                B + pid_batch * stride_b_batch + seq[:, None] * stride_b_seqlen
                + qk_head * stride_b_head + offs_n[None, :] * stride_b_dim,
                mask=valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
            )
            x = tl.load(
                X + pid_batch * stride_x_batch + seq[:, None] * stride_x_seqlen
                + pid_head * stride_x_head + offs_p[None, :] * stride_x_dim,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            )
            dA_rev = tl.load(
                DA_CS_REV + dacsrev_base + rows * stride_dacsrev_csize,
                mask=rows < CHUNK_SIZE, other=0.0,
            ).to(tl.float32)
            dt = tl.load(
                DT + dt_base + rows * stride_dt_csize,
                mask=rows < CHUNK_SIZE, other=0.0,
            ).to(tl.float32)
            dx_state = tl.dot(b, tl.trans(acc_dstates.to(b.dtype)))
            grad_dt_state = tl.sum(dx_state.to(tl.float32) * x.to(tl.float32), axis=1)
            direct = tl.where(valid, grad_dt_state * tl.exp(dA_rev), 0.0)
            prev_signal = direct * tl.where(valid, dt, 0.0)

            prev_rows = rows - 1
            prev_seq = seq - 1
            prev_valid = (local > 0) & (prev_rows < CHUNK_SIZE) & (prev_seq < seqlen)
            b_prev = tl.load(
                B + pid_batch * stride_b_batch
                + prev_seq[:, None] * stride_b_seqlen + qk_head * stride_b_head
                + offs_n[None, :] * stride_b_dim,
                mask=prev_valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
            )
            x_prev = tl.load(
                X + pid_batch * stride_x_batch
                + prev_seq[:, None] * stride_x_seqlen + pid_head * stride_x_head
                + offs_p[None, :] * stride_x_dim,
                mask=prev_valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            )
            rev_prev = tl.load(
                DA_CS_REV + dacsrev_base + prev_rows * stride_dacsrev_csize,
                mask=prev_valid, other=0.0,
            ).to(tl.float32)
            dt_prev = tl.load(
                DT + dt_base + prev_rows * stride_dt_csize,
                mask=prev_valid, other=0.0,
            ).to(tl.float32)
            dx_prev = tl.dot(b_prev, tl.trans(acc_dstates.to(b_prev.dtype)))
            grad_prev = tl.sum(dx_prev.to(tl.float32) * x_prev.to(tl.float32), axis=1)
            shifted_prev = tl.where(
                prev_valid, grad_prev * tl.exp(rev_prev) * dt_prev, 0.0
            )
            exclusive_prefix = tl.cumsum(shifted_prev, axis=0) + prefix_carry
            out_ptrs = PROCESSED_DDT + ddt_base + rows * stride_ddt_csize
            old = tl.load(out_ptrs, mask=rows < CHUNK_SIZE, other=0.0).to(tl.float32)
            tl.store(
                out_ptrs,
                old + direct + a * (exclusive_prefix + transition),
                mask=valid,
            )
            prefix_carry += tl.sum(prev_signal)
            dt_sum += tl.sum(tl.where(valid, dt, 0.0))

        # Shift-left before the reverse scan for the dA contribution.
        dA_value = transition * dt_sum
        suffix_carry = 0.0
        for row_rev in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            row_start = CHUNK_SIZE - BLOCK_M - row_rev
            local = tl.arange(0, BLOCK_M)
            rows = row_start + local
            seq = chunk_start + rows
            valid = (rows >= 0) & (rows < CHUNK_SIZE) & (seq < seqlen)
            b = tl.load(
                B + pid_batch * stride_b_batch + seq[:, None] * stride_b_seqlen
                + qk_head * stride_b_head + offs_n[None, :] * stride_b_dim,
                mask=valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
            )
            x = tl.load(
                X + pid_batch * stride_x_batch + seq[:, None] * stride_x_seqlen
                + pid_head * stride_x_head + offs_p[None, :] * stride_x_dim,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            )
            rev = tl.load(
                DA_CS_REV + dacsrev_base + rows * stride_dacsrev_csize,
                mask=(rows >= 0) & (rows < CHUNK_SIZE), other=0.0,
            ).to(tl.float32)
            dt = tl.load(
                DT + dt_base + rows * stride_dt_csize,
                mask=(rows >= 0) & (rows < CHUNK_SIZE), other=0.0,
            ).to(tl.float32)
            dx_state = tl.dot(b, tl.trans(acc_dstates.to(b.dtype)))
            grad_dt_state = tl.sum(dx_state.to(tl.float32) * x.to(tl.float32), axis=1)
            prev_signal = tl.where(valid, grad_dt_state * tl.exp(rev) * dt, 0.0)
            next_rows = rows + 1
            shifted_valid = (
                (local + 1 < BLOCK_M) & (next_rows < CHUNK_SIZE)
                & (chunk_start + next_rows < seqlen)
            )
            shifted_dt = tl.load(
                DT + dt_base + next_rows * stride_dt_csize,
                mask=shifted_valid, other=0.0,
            ).to(tl.float32)
            exclusive_suffix = tl.cumsum(
                tl.where(shifted_valid, shifted_dt, 0.0), axis=0, reverse=True
            ) + suffix_carry
            dA_value += tl.sum(prev_signal * exclusive_suffix)
            suffix_carry += tl.sum(tl.where(valid, dt, 0.0))

        dA_ptr = (
            DA_CHUNK + pid_batch * stride_dachunk_batch + pid_head * stride_dachunk_head
            + chunk_idx * stride_dachunk_chunk
        )
        tl.store(dA_ptr, tl.load(dA_ptr).to(tl.float32) + dA_value)

        # Advance the reverse recurrence only after all contributions above
        # have consumed the chunk-output adjoint.
        acc_dstates *= tl.exp(chunk_total)
        for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            rows = row_start + tl.arange(0, BLOCK_M)
            seq = chunk_start + rows
            valid = (rows < CHUNK_SIZE) & (seq < seqlen)
            c = tl.load(
                C + pid_batch * stride_c_batch + seq[:, None] * stride_c_seqlen
                + qk_head * stride_c_head + offs_n[None, :] * stride_c_dim,
                mask=valid[:, None] & (offs_n[None, :] < dstate_n), other=0.0,
            )
            dout = tl.load(
                DOUT + stream * stride_dout_stream + seq[:, None] * stride_dout_seqlen
                + batch_idx * stride_dout_batch + pid_head * stride_dout_head
                + offs_p[None, :] * stride_dout_dim,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            ).to(tl.float32)
            z = tl.load(
                Z + stream * stride_z_stream + seq[:, None] * stride_z_seqlen
                + batch_idx * stride_z_batch + pid_head * stride_z_head
                + offs_p[None, :] * stride_z_dim,
                mask=valid[:, None] & (offs_p[None, :] < headdim_p), other=0.0,
            ).to(tl.float32)
            dacs = tl.load(
                DA_CS + dacs_base + rows * stride_dacs_csize,
                mask=rows < CHUNK_SIZE, other=0.0,
            ).to(tl.float32)
            dpre = (dout * z * tl.sigmoid(z)).to(c.dtype)
            c_scaled = c * tl.exp(dacs)[:, None].to(c.dtype)
            acc_dstates += tl.dot(tl.trans(dpre), c_scaled)


def _mamba2_bwd_dk_dv_cp_workspace_oracle(
    C: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, dstate)
    B: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, dstate)
    x: torch.Tensor,  # (folded_batch, seqlen, nheads, headdim)
    dout: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim)
    z: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim)
    dt: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs_rev: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    segsum: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size, chunk_size)
    chunk_start_states: torch.Tensor,  # (folded_batch, nheads, nchunks, headdim, dstate)
    final_dstates: torch.Tensor,  # (streams, batch, nheads, headdim, dstate)
    A: torch.Tensor,  # (nheads,)
    D: torch.Tensor,  # (nheads,)
    chunk_size: int = 64,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return true-CP dB/dx and the fixed decay-workspace tuple.

    The returned tuple is ``(dB, dx, dD, initial_dA, initial_ddt,
    ddA_cs_rev, grad_scale, dM_curr)``. The reverse recurrence is seeded from
    ``final_dstates`` but its segment-input adjoint is intentionally not stored.
    For chunks larger than 64, dQ must have finished consuming ``segsum``;
    this wrapper overwrites that buffer tile-by-tile and returns it as
    ``dM_curr`` instead of allocating a second chunk-square FP32 tensor.
    """

    if C.ndim != 4 or B.ndim != 4 or x.ndim != 4:
        raise ValueError("C, B, and x must be 4D")
    if dout.ndim != 5 or z.ndim != 5:
        raise ValueError("dout and z must be 5D stream-major tensors")
    if any(tensor.ndim != 4 for tensor in (dt, dA_cs, dA_cs_rev)):
        raise ValueError("dt, dA_cs, and dA_cs_rev must be 4D")
    if segsum.ndim != 5 or chunk_start_states.ndim != 5 or final_dstates.ndim != 5:
        raise ValueError("segsum, chunk_start_states, and final_dstates must be 5D")
    if C.shape != B.shape:
        raise ValueError("C and B must have the same shape")
    if dout.shape != z.shape:
        raise ValueError("dout and z must have the same shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    folded_batch, seqlen, nheads_qk, dstate = C.shape
    x_batch, x_seqlen, nheads, headdim = x.shape
    streams, dout_seqlen, batch, dout_nheads, dout_headdim = dout.shape
    if streams != 2:
        raise ValueError("true-CP dK/dV requires exactly two local streams")
    if (x_batch, x_seqlen) != (folded_batch, seqlen):
        raise ValueError("C, B, and x must share folded batch and sequence dimensions")
    if (dout_seqlen, dout_nheads, dout_headdim) != (seqlen, nheads, headdim):
        raise ValueError("dout and z dimensions must match x")
    if folded_batch != streams * batch:
        raise ValueError("folded C/B/x batch must equal streams * batch")
    if nheads_qk <= 0 or nheads <= 0 or nheads % nheads_qk != 0:
        raise ValueError("the number of value heads must be divisible by C/B heads")
    if dstate <= 0 or headdim <= 0:
        raise ValueError("state and head dimensions must be positive")

    nchunks = triton.cdiv(seqlen, chunk_size)
    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    for name, tensor in (("dt", dt), ("dA_cs", dA_cs), ("dA_cs_rev", dA_cs_rev)):
        if tensor.shape != decay_shape:
            raise ValueError(f"{name} must have shape {decay_shape}, got {tuple(tensor.shape)}")
    segsum_shape = (*decay_shape, chunk_size)
    if segsum.shape != segsum_shape:
        raise ValueError(f"segsum must have shape {segsum_shape}, got {tuple(segsum.shape)}")
    state_shape = (folded_batch, nheads, nchunks, headdim, dstate)
    if chunk_start_states.shape != state_shape:
        raise ValueError(
            f"chunk_start_states must have shape {state_shape}, "
            f"got {tuple(chunk_start_states.shape)}"
        )
    final_shape = (streams, batch, nheads, headdim, dstate)
    if final_dstates.shape != final_shape:
        raise ValueError(
            f"final_dstates must have shape {final_shape}, got {tuple(final_dstates.shape)}"
        )
    if A.shape != (nheads,) or D.shape != (nheads,):
        raise ValueError(f"A and D must each have shape {(nheads,)}")

    tensors = (
        C,
        B,
        x,
        dout,
        z,
        dt,
        dA_cs,
        dA_cs_rev,
        segsum,
        chunk_start_states,
        final_dstates,
        A,
        D,
    )
    if not C.is_cuda:
        raise ValueError("mamba2_bwd_dk_dv_cp_triton requires CUDA tensors")
    if any(tensor.device != C.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if any(tensor.dtype != C.dtype for tensor in (B, x, dout, z)):
        raise ValueError("C, B, x, dout, and z must have the same dtype")
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must use floating-point dtypes")

    dB_partial = torch.empty(
        folded_batch, seqlen, nheads, dstate, device=C.device, dtype=C.dtype
    )
    dx = torch.empty(
        folded_batch, seqlen, nheads, headdim, device=C.device, dtype=C.dtype
    )
    dD_partial = torch.empty(nheads, folded_batch, device=C.device, dtype=torch.float32)
    dA_partial = torch.empty_like(dD_partial)
    initial_ddt = torch.empty(decay_shape, device=C.device, dtype=torch.float32)
    ddA_cs_rev = torch.empty_like(initial_ddt)
    grad_scale = torch.empty_like(initial_ddt)
    if chunk_size <= 64:
        dM_curr = torch.empty(segsum_shape, device=C.device, dtype=torch.float32)
    else:
        if segsum.dtype is not torch.float32:
            raise ValueError("large-chunk segsum must be FP32 for in-place dM reuse")
        dM_curr = segsum

    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(dstate))
    grid = (nheads, folded_batch)
    if chunk_size <= 64:
        _mamba2_bwd_dk_dv_cp_kernel[grid](
            C, B, x, dout, z, dt, dA_cs, dA_cs_rev, segsum,
            chunk_start_states, final_dstates, A, D,
            dB_partial, dx, dD_partial, dA_partial, initial_ddt,
            ddA_cs_rev, grad_scale, dM_curr,
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
            dA_cs_rev.stride(0), dA_cs_rev.stride(1), dA_cs_rev.stride(2), dA_cs_rev.stride(3),
            segsum.stride(0), segsum.stride(1), segsum.stride(2), segsum.stride(3), segsum.stride(4),
            chunk_start_states.stride(0), chunk_start_states.stride(1),
            chunk_start_states.stride(2), chunk_start_states.stride(3),
            chunk_start_states.stride(4),
            final_dstates.stride(0), final_dstates.stride(1), final_dstates.stride(2),
            final_dstates.stride(3), final_dstates.stride(4),
            A.stride(0), D.stride(0),
            dB_partial.stride(0), dB_partial.stride(1), dB_partial.stride(2),
            dB_partial.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            dD_partial.stride(0), dD_partial.stride(1),
            dA_partial.stride(0), dA_partial.stride(1),
            initial_ddt.stride(0), initial_ddt.stride(1), initial_ddt.stride(2),
            initial_ddt.stride(3),
            ddA_cs_rev.stride(0), ddA_cs_rev.stride(1), ddA_cs_rev.stride(2),
            ddA_cs_rev.stride(3),
            grad_scale.stride(0), grad_scale.stride(1), grad_scale.stride(2),
            grad_scale.stride(3),
            dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2),
            dM_curr.stride(3), dM_curr.stride(4),
            seqlen, headdim, dstate, nheads_qk,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )
    else:
        _mamba2_bwd_dk_dv_cp_large_chunk_kernel[grid](
            C, B, x, dout, z, dt, dA_cs, dA_cs_rev, segsum,
            chunk_start_states, final_dstates, A, D,
            dB_partial, dx, dD_partial, dA_partial, initial_ddt,
            ddA_cs_rev, grad_scale,
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
            dA_cs_rev.stride(0), dA_cs_rev.stride(1), dA_cs_rev.stride(2), dA_cs_rev.stride(3),
            segsum.stride(0), segsum.stride(1), segsum.stride(2), segsum.stride(3), segsum.stride(4),
            chunk_start_states.stride(0), chunk_start_states.stride(1),
            chunk_start_states.stride(2), chunk_start_states.stride(3),
            chunk_start_states.stride(4),
            final_dstates.stride(0), final_dstates.stride(1), final_dstates.stride(2),
            final_dstates.stride(3), final_dstates.stride(4),
            A.stride(0), D.stride(0),
            dB_partial.stride(0), dB_partial.stride(1), dB_partial.stride(2),
            dB_partial.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            dD_partial.stride(0), dD_partial.stride(1),
            dA_partial.stride(0), dA_partial.stride(1),
            initial_ddt.stride(0), initial_ddt.stride(1), initial_ddt.stride(2),
            initial_ddt.stride(3),
            ddA_cs_rev.stride(0), ddA_cs_rev.stride(1), ddA_cs_rev.stride(2),
            ddA_cs_rev.stride(3),
            grad_scale.stride(0), grad_scale.stride(1), grad_scale.stride(2),
            grad_scale.stride(3),
            seqlen, headdim, dstate, nheads_qk,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )
        dm_block_m = 32
        dm_block_k = 32
        dm_grid = (
            triton.cdiv(chunk_size, dm_block_m),
            triton.cdiv(chunk_size, dm_block_k),
            folded_batch * nheads * nchunks,
        )
        _mamba2_bwd_dm_curr_inplace_kernel[dm_grid](
            C, B, x, dout, z, dM_curr,
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
            dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2),
            dM_curr.stride(3), dM_curr.stride(4),
            seqlen, headdim, dstate, nheads, nheads_qk, nchunks,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            BLOCK_M=dm_block_m,
            BLOCK_K=dm_block_k,
            num_warps=4,
            num_stages=1,
        )

    if nheads == nheads_qk:
        dB = dB_partial
    else:
        dB = torch.empty_like(B)
        block_size = 128
        reduce_grid = (
            folded_batch
            * seqlen
            * nheads_qk
            * triton.cdiv(dstate, block_size),
        )
        reduce_head_groups_kernel[reduce_grid](
            dB_partial,
            dB,
            folded_batch,
            seqlen,
            nheads,
            nheads_qk,
            dstate,
            dB_partial.stride(0), dB_partial.stride(1),
            dB_partial.stride(2), dB_partial.stride(3),
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3),
            BLOCK_SIZE=block_size,
        )

    dD_float = torch.empty(nheads, device=C.device, dtype=torch.float32)
    initial_dA = torch.empty_like(dD_float)
    reduce_batch_dimension_kernel[(nheads,)](
        dD_partial,
        dD_float,
        nheads,
        folded_batch,
        dD_partial.stride(0),
        dD_partial.stride(1),
    )
    reduce_batch_dimension_kernel[(nheads,)](
        dA_partial,
        initial_dA,
        nheads,
        folded_batch,
        dA_partial.stride(0),
        dA_partial.stride(1),
    )
    dD = dD_float.to(dtype=D.dtype)
    return dB, dx, dD, initial_dA, initial_ddt, ddA_cs_rev, grad_scale, dM_curr


def mamba2_bwd_dk_dv_cp_triton(
    C: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, dstate)
    B: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, dstate)
    x: torch.Tensor,  # (folded_batch, seqlen, nheads, headdim)
    dout: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim)
    z: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim)
    dt: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs: torch.Tensor,  # chunked like dt
    dA_cs_rev: torch.Tensor,  # chunked like dt
    chunk_start_states: torch.Tensor,  # (folded_batch, nheads, nchunks, P, N)
    # Unsplit: (streams, batch, nheads, P, N).
    # Split, reverse order: (folded_batch, nsubsequences, nheads, P, N).
    final_dstates: torch.Tensor,
    A: torch.Tensor,  # (nheads,)
    D: torch.Tensor,  # (nheads,)
    processed_ddt: torch.Tensor,  # chunked FP32 accumulator from dQ
    dA_chunk: torch.Tensor,  # (folded_batch, nheads, nchunks), FP32
    chunk_size: int = 64,
    subsequence_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dB/dx/dD and compact, fully accumulated decay gradients."""

    if C.ndim != 4 or B.ndim != 4 or x.ndim != 4:
        raise ValueError("C, B, and x must be 4D")
    if dout.ndim != 5 or z.ndim != 5:
        raise ValueError("dout and z must be 5D stream-major tensors")
    if any(tensor.ndim != 4 for tensor in (dt, dA_cs, dA_cs_rev, processed_ddt)):
        raise ValueError("prepared decay and processed_ddt must be 4D")
    if chunk_start_states.ndim != 5 or final_dstates.ndim != 5 or dA_chunk.ndim != 3:
        raise ValueError("state tensors must be 5D and dA_chunk must be 3D")
    if C.shape != B.shape or dout.shape != z.shape:
        raise ValueError("C/B and dout/z shapes must match")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if subsequence_size < 0:
        raise ValueError("subsequence_size must be nonnegative")
    if subsequence_size and (
        chunk_size != 64
        or subsequence_size < chunk_size
        or subsequence_size % chunk_size
    ):
        raise ValueError(
            "subsequence dK/dV requires chunk_size=64 and a subsequence_size multiple of 64"
        )

    folded_batch, seqlen, nheads_qk, dstate = C.shape
    x_batch, x_seqlen, nheads, headdim = x.shape
    streams, dout_seqlen, batch, dout_nheads, dout_headdim = dout.shape
    if streams != 2 or folded_batch != streams * batch:
        raise ValueError("true-CP dK/dV requires two folded streams")
    if (x_batch, x_seqlen) != (folded_batch, seqlen):
        raise ValueError("C, B, and x must share folded batch and sequence")
    if (dout_seqlen, dout_nheads, dout_headdim) != (seqlen, nheads, headdim):
        raise ValueError("dout/z dimensions must match x")
    if nheads_qk <= 0 or nheads % nheads_qk != 0:
        raise ValueError("value heads must be divisible by C/B heads")
    nchunks = triton.cdiv(seqlen, chunk_size)
    subsequence_chunks = subsequence_size // chunk_size if subsequence_size else 0
    nsubsequences = (
        triton.cdiv(nchunks, subsequence_chunks) if subsequence_chunks else 1
    )
    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    for name, tensor in (
        ("dt", dt), ("dA_cs", dA_cs), ("dA_cs_rev", dA_cs_rev),
        ("processed_ddt", processed_ddt),
    ):
        if tensor.shape != decay_shape:
            raise ValueError(f"{name} must have shape {decay_shape}, got {tuple(tensor.shape)}")
    state_shape = (folded_batch, nheads, nchunks, headdim, dstate)
    final_shape = (
        (folded_batch, nsubsequences, nheads, headdim, dstate)
        if subsequence_chunks
        else (streams, batch, nheads, headdim, dstate)
    )
    if chunk_start_states.shape != state_shape:
        raise ValueError(f"chunk_start_states must have shape {state_shape}")
    if final_dstates.shape != final_shape:
        raise ValueError(f"final_dstates must have shape {final_shape}")
    if dA_chunk.shape != (folded_batch, nheads, nchunks):
        raise ValueError("dA_chunk has the wrong shape")
    if A.shape != (nheads,) or D.shape != (nheads,):
        raise ValueError(f"A and D must each have shape {(nheads,)}")
    tensors = (
        C, B, x, dout, z, dt, dA_cs, dA_cs_rev, chunk_start_states,
        final_dstates, A, D, processed_ddt, dA_chunk,
    )
    if not C.is_cuda:
        raise ValueError("mamba2_bwd_dk_dv_cp_triton requires CUDA tensors")
    if any(t.device != C.device for t in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if any(t.dtype != C.dtype for t in (B, x, dout, z)):
        raise ValueError("C, B, x, dout, and z must have the same dtype")
    if processed_ddt.dtype is not torch.float32 or dA_chunk.dtype is not torch.float32:
        raise ValueError("compact decay accumulators must be FP32")

    dB_partial = torch.empty(
        folded_batch, seqlen, nheads, dstate, device=C.device, dtype=C.dtype
    )
    dx = torch.empty(
        folded_batch, seqlen, nheads, headdim, device=C.device, dtype=C.dtype
    )
    dD_partial = (
        torch.empty(
            nheads,
            folded_batch,
            nsubsequences,
            device=C.device,
            dtype=torch.float32,
        )
        if subsequence_chunks
        else torch.empty(nheads, folded_batch, device=C.device, dtype=torch.float32)
    )
    ddA_cs_rev_intermediate = torch.empty(
        decay_shape, device=C.device, dtype=torch.float32
    )
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(dstate))
    compact_args = (
        C, B, x, dout, z, dt, dA_cs, dA_cs_rev,
        chunk_start_states, final_dstates, A, D,
        dB_partial, dx, dD_partial, processed_ddt, dA_chunk,
        ddA_cs_rev_intermediate,
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2),
        dout.stride(3), dout.stride(4),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1),
        dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        chunk_start_states.stride(0), chunk_start_states.stride(1),
        chunk_start_states.stride(2), chunk_start_states.stride(3),
        chunk_start_states.stride(4),
        final_dstates.stride(0), final_dstates.stride(1),
        final_dstates.stride(2), final_dstates.stride(3), final_dstates.stride(4),
        A.stride(0),
        D.stride(0),
        dB_partial.stride(0), dB_partial.stride(1),
        dB_partial.stride(2), dB_partial.stride(3),
        dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
        dD_partial.stride(0), dD_partial.stride(1),
        dD_partial.stride(2) if subsequence_chunks else 0,
        processed_ddt.stride(0), processed_ddt.stride(1),
        processed_ddt.stride(2), processed_ddt.stride(3),
        dA_chunk.stride(0), dA_chunk.stride(1), dA_chunk.stride(2),
        ddA_cs_rev_intermediate.stride(0), ddA_cs_rev_intermediate.stride(1),
        ddA_cs_rev_intermediate.stride(2), ddA_cs_rev_intermediate.stride(3),
        seqlen, headdim, dstate, nheads_qk,
    )

    def launch_dk_dv(
        launch_grid,
        reverse_slot_offset,
        original_subsequence_base,
        original_subsequence_count,
        full_subsequence,
        exact_state,
    ):
        _mamba2_bwd_dk_dv_compact_cp_kernel[launch_grid](
            *compact_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            SUBSEQUENCE_CHUNKS=subsequence_chunks,
            REVERSE_SLOT_OFFSET=reverse_slot_offset,
            ORIGINAL_SUBSEQUENCE_BASE=original_subsequence_base,
            ORIGINAL_SUBSEQUENCE_COUNT=original_subsequence_count,
            FULL_SUBSEQUENCE=full_subsequence,
            EXACT_STATE=exact_state,
        )

    if subsequence_chunks:
        nfull = seqlen // subsequence_size
        has_tail = seqlen % subsequence_size != 0
        if has_tail:
            launch_dk_dv(
                (nheads, folded_batch, 1),
                0,
                nfull,
                1,
                False,
                False,
            )
        if nfull:
            launch_dk_dv(
                (nheads, folded_batch, nfull),
                int(has_tail),
                0,
                nfull,
                True,
                headdim == block_p and dstate == block_n,
            )
    else:
        launch_dk_dv(
            (nheads, folded_batch),
            0,
            0,
            0,
            False,
            False,
        )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    prev_grid = (nheads, folded_batch, nchunks)
    _mamba2_bwd_dk_dv_cached_prev_decay_cp_kernel[prev_grid](
        processed_ddt, dt, ddA_cs_rev_intermediate, dA_chunk, A,
        processed_ddt.stride(0), processed_ddt.stride(1),
        processed_ddt.stride(2), processed_ddt.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        ddA_cs_rev_intermediate.stride(0), ddA_cs_rev_intermediate.stride(1),
        ddA_cs_rev_intermediate.stride(2), ddA_cs_rev_intermediate.stride(3),
        dA_chunk.stride(0), dA_chunk.stride(1), dA_chunk.stride(2),
        A.stride(0),
        seqlen,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
        num_warps=4,
        num_stages=1,
    )
    del ddA_cs_rev_intermediate

    if nheads == nheads_qk:
        dB = dB_partial
    else:
        dB = torch.empty_like(B)
        block_size = 128
        reduce_grid = (
            folded_batch * seqlen * nheads_qk * triton.cdiv(dstate, block_size),
        )
        reduce_head_groups_kernel[reduce_grid](
            dB_partial, dB, folded_batch, seqlen, nheads, nheads_qk, dstate,
            dB_partial.stride(0), dB_partial.stride(1),
            dB_partial.stride(2), dB_partial.stride(3),
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3),
            BLOCK_SIZE=block_size,
        )
    dD_float = torch.empty(nheads, device=C.device, dtype=torch.float32)
    dD_reduce_input = (
        dD_partial.view(nheads, folded_batch * nsubsequences)
        if subsequence_chunks
        else dD_partial
    )
    reduce_batch_dimension_kernel[(nheads,)](
        dD_reduce_input,
        dD_float,
        nheads,
        folded_batch * nsubsequences if subsequence_chunks else folded_batch,
        dD_reduce_input.stride(0),
        dD_reduce_input.stride(1),
    )
    return dB, dx, dD_float.to(dtype=D.dtype), processed_ddt, dA_chunk


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK"],
)
@triton.jit
def _dA_ddt_from_dM_next_cp_kernel(
    DDT,
    DT,
    GRAD_SCALE,
    A,
    DDT_OUT,
    DA_PARTIAL,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_grad_batch,
    stride_grad_head,
    stride_grad_chunk,
    stride_grad_csize,
    stride_a_head,
    stride_out_batch,
    stride_out_head,
    stride_out_chunk,
    stride_out_csize,
    stride_partial_head,
    stride_partial_batch,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Add the forward-cumulative (dM-next) decay contribution."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offs = tl.arange(0, BLOCK_CHUNK)
    logical_mask = offs < CHUNK_SIZE
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    dA_partial = 0.0
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        seq = chunk_idx * CHUNK_SIZE + offs
        valid = logical_mask & (seq < seqlen)
        ddt = tl.load(
            DDT
            + pid_batch * stride_ddt_batch
            + pid_head * stride_ddt_head
            + chunk_idx * stride_ddt_chunk
            + offs * stride_ddt_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            DT
            + pid_batch * stride_dt_batch
            + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
            + offs * stride_dt_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        grad_scale = tl.load(
            GRAD_SCALE
            + pid_batch * stride_grad_batch
            + pid_head * stride_grad_head
            + chunk_idx * stride_grad_chunk
            + offs * stride_grad_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        dt = tl.where(valid, dt, 0.0)
        grad_scale = tl.where(valid, grad_scale, 0.0)

        dt_prefix = tl.cumsum(dt, axis=0)
        grad_scale_reverse = tl.cumsum(grad_scale, axis=0, reverse=True)
        dA_partial += tl.sum(dt_prefix * grad_scale)
        ddt_out = tl.where(valid, ddt + a_value * grad_scale_reverse, 0.0)
        tl.store(
            DDT_OUT
            + pid_batch * stride_out_batch
            + pid_head * stride_out_head
            + chunk_idx * stride_out_chunk
            + offs * stride_out_csize,
            ddt_out,
            mask=logical_mask,
        )

    tl.store(
        DA_PARTIAL
        + pid_head * stride_partial_head
        + pid_batch * stride_partial_batch,
        dA_partial,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK"],
)
@triton.jit
def _dA_ddt_from_dM_curr_cp_kernel(
    DDT,
    DT,
    DM_CURR,
    A,
    DDT_OUT,
    DA_PARTIAL,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dm_batch,
    stride_dm_head,
    stride_dm_chunk,
    stride_dm_row,
    stride_dm_col,
    stride_a_head,
    stride_out_batch,
    stride_out_head,
    stride_out_chunk,
    stride_out_csize,
    stride_partial_head,
    stride_partial_batch,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Add explicit-dt and stable segsum-backward contributions."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offs = tl.arange(0, BLOCK_CHUNK)
    logical_mask = offs < CHUNK_SIZE
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    dA_partial = 0.0
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        seq = chunk_idx * CHUNK_SIZE + offs
        valid = logical_mask & (seq < seqlen)
        ddt = tl.load(
            DDT
            + pid_batch * stride_ddt_batch
            + pid_head * stride_ddt_head
            + chunk_idx * stride_ddt_chunk
            + offs * stride_ddt_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            DT
            + pid_batch * stride_dt_batch
            + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
            + offs * stride_dt_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        dt = tl.where(valid, dt, 0.0)

        matrix_mask = logical_mask[:, None] & logical_mask[None, :]
        valid_matrix = valid[:, None] & valid[None, :]
        dM = tl.load(
            DM_CURR
            + pid_batch * stride_dm_batch
            + pid_head * stride_dm_head
            + chunk_idx * stride_dm_chunk
            + offs[:, None] * stride_dm_row
            + offs[None, :] * stride_dm_col,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        dM = tl.where(valid_matrix, dM, 0.0)

        ddt_direct = tl.sum(dM, axis=0)
        d_segsum = dM * dt[None, :]
        strict_lower = offs[:, None] >= offs[None, :] + 1
        d_segsum = tl.where(valid_matrix & strict_lower, d_segsum, 0.0)
        # Stable transpose of the forward cumulative-sum construction:
        # reverse cumsum down rows, re-mask, then sum over columns.
        ddA_matrix = tl.cumsum(d_segsum, axis=0, reverse=True)
        ddA_matrix = tl.where(valid_matrix & strict_lower, ddA_matrix, 0.0)
        ddA = tl.sum(ddA_matrix, axis=1)

        dA_partial += tl.sum(ddA * dt)
        ddt_out = tl.where(valid, ddt + ddt_direct + a_value * ddA, 0.0)
        tl.store(
            DDT_OUT
            + pid_batch * stride_out_batch
            + pid_head * stride_out_head
            + chunk_idx * stride_out_chunk
            + offs * stride_out_csize,
            ddt_out,
            mask=logical_mask,
        )

    tl.store(
        DA_PARTIAL
        + pid_head * stride_partial_head
        + pid_batch * stride_partial_batch,
        dA_partial,
    )


@triton.jit
def _dA_ddt_from_dM_curr_columns_cp_kernel(
    DDT,
    DT,
    DM_CURR,
    DDT_OUT,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dm_batch,
    stride_dm_head,
    stride_dm_chunk,
    stride_dm_row,
    stride_dm_col,
    stride_out_batch,
    stride_out_head,
    stride_out_chunk,
    stride_out_csize,
    seqlen,
    nchunks,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Reverse-scan one dM column and initialize its direct dt gradient."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk_col = tl.program_id(2)
    chunk_idx = pid_chunk_col // CHUNK_SIZE
    col = pid_chunk_col - chunk_idx * CHUNK_SIZE
    offs = tl.arange(0, BLOCK_CHUNK)
    logical = offs < CHUNK_SIZE
    row_valid = logical & (chunk_idx * CHUNK_SIZE + offs < seqlen)
    col_valid = (chunk_idx < nchunks) & (chunk_idx * CHUNK_SIZE + col < seqlen)
    strict_lower = offs > col

    dm_ptrs = (
        DM_CURR
        + pid_batch * stride_dm_batch
        + pid_head * stride_dm_head
        + chunk_idx * stride_dm_chunk
        + offs * stride_dm_row
        + col * stride_dm_col
    )
    dM = tl.load(dm_ptrs, mask=logical & (chunk_idx < nchunks), other=0.0).to(
        tl.float32
    )
    dM = tl.where(row_valid & col_valid, dM, 0.0)
    direct = tl.sum(dM)
    dt_col = tl.load(
        DT
        + pid_batch * stride_dt_batch
        + pid_head * stride_dt_head
        + chunk_idx * stride_dt_chunk
        + col * stride_dt_csize,
        mask=col_valid,
        other=0.0,
    ).to(tl.float32)
    reverse_column = tl.cumsum(
        tl.where(row_valid & col_valid & strict_lower, dM * dt_col, 0.0),
        axis=0,
        reverse=True,
    )
    reverse_column = tl.where(
        row_valid & col_valid & strict_lower, reverse_column, 0.0
    )
    tl.store(dm_ptrs, reverse_column, mask=logical & (chunk_idx < nchunks))

    ddt = tl.load(
        DDT
        + pid_batch * stride_ddt_batch
        + pid_head * stride_ddt_head
        + chunk_idx * stride_ddt_chunk
        + col * stride_ddt_csize,
        mask=col_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        DDT_OUT
        + pid_batch * stride_out_batch
        + pid_head * stride_out_head
        + chunk_idx * stride_out_chunk
        + col * stride_out_csize,
        tl.where(col_valid, ddt + direct, 0.0),
        mask=chunk_idx < nchunks,
    )


@triton.jit
def _dA_ddt_from_dM_curr_rows_cp_kernel(
    DDT_OUT,
    DT,
    DM_SCAN,
    A,
    DA_PARTIAL,
    stride_out_batch,
    stride_out_head,
    stride_out_chunk,
    stride_out_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_dm_batch,
    stride_dm_head,
    stride_dm_chunk,
    stride_dm_row,
    stride_dm_col,
    stride_a_head,
    stride_partial_head,
    stride_partial_batch,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Reduce reverse-scanned dM rows without materializing a square tile."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offs_col = tl.arange(0, BLOCK_CHUNK)
    logical_col = offs_col < CHUNK_SIZE
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    dA_partial = 0.0
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        for row in tl.range(0, CHUNK_SIZE, 1, num_stages=1):
            row_valid = chunk_idx * CHUNK_SIZE + row < seqlen
            col_valid = logical_col & (chunk_idx * CHUNK_SIZE + offs_col < seqlen)
            reverse_row = tl.load(
                DM_SCAN
                + pid_batch * stride_dm_batch
                + pid_head * stride_dm_head
                + chunk_idx * stride_dm_chunk
                + row * stride_dm_row
                + offs_col * stride_dm_col,
                mask=logical_col,
                other=0.0,
            ).to(tl.float32)
            ddA = tl.sum(
                tl.where(row_valid & col_valid & (offs_col < row), reverse_row, 0.0)
            )
            out_ptr = (
                DDT_OUT
                + pid_batch * stride_out_batch
                + pid_head * stride_out_head
                + chunk_idx * stride_out_chunk
                + row * stride_out_csize
            )
            ddt = tl.load(out_ptr, mask=row_valid, other=0.0).to(tl.float32)
            dt = tl.load(
                DT
                + pid_batch * stride_dt_batch
                + pid_head * stride_dt_head
                + chunk_idx * stride_dt_chunk
                + row * stride_dt_csize,
                mask=row_valid,
                other=0.0,
            ).to(tl.float32)
            tl.store(out_ptr, tl.where(row_valid, ddt + a_value * ddA, 0.0))
            dA_partial += ddA * dt

    tl.store(
        DA_PARTIAL
        + pid_head * stride_partial_head
        + pid_batch * stride_partial_batch,
        dA_partial,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK"],
)
@triton.jit
def _dA_ddt_from_dM_prev_cp_kernel(
    DDT,
    DT,
    DDACSREV,
    A,
    DDT_OUT,
    DA_PARTIAL,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize,
    stride_ddacs_batch,
    stride_ddacs_head,
    stride_ddacs_chunk,
    stride_ddacs_csize,
    stride_a_head,
    stride_out_batch,
    stride_out_head,
    stride_out_chunk,
    stride_out_csize,
    stride_partial_head,
    stride_partial_batch,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Add reverse-cumulative (dM-prev) decay using shifted scans."""

    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offs = tl.arange(0, BLOCK_CHUNK)
    logical_mask = offs < CHUNK_SIZE
    a_value = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    dA_partial = 0.0
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        chunk_base_ddt = (
            DDT
            + pid_batch * stride_ddt_batch
            + pid_head * stride_ddt_head
            + chunk_idx * stride_ddt_chunk
        )
        chunk_base_dt = (
            DT
            + pid_batch * stride_dt_batch
            + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
        )
        chunk_base_ddacs = (
            DDACSREV
            + pid_batch * stride_ddacs_batch
            + pid_head * stride_ddacs_head
            + chunk_idx * stride_ddacs_chunk
        )
        seq = chunk_idx * CHUNK_SIZE + offs
        valid = logical_mask & (seq < seqlen)
        ddt = tl.load(
            chunk_base_ddt + offs * stride_ddt_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        ddA_cs_rev = tl.load(
            chunk_base_ddacs + offs * stride_ddacs_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        ddA_cs_rev = tl.where(valid, ddA_cs_rev, 0.0)

        # Shift left before a reverse cumsum: sum(dt[j] for j > i).
        shifted_dt = tl.load(
            chunk_base_dt + (offs + 1) * stride_dt_csize,
            mask=(offs + 1 < CHUNK_SIZE)
            & (chunk_idx * CHUNK_SIZE + offs + 1 < seqlen),
            other=0.0,
        ).to(tl.float32)
        exclusive_suffix_dt = tl.cumsum(shifted_dt, axis=0, reverse=True)

        # Shift right before a forward cumsum: sum(ddA[t] for t < i).
        shifted_ddA = tl.load(
            chunk_base_ddacs + (offs - 1) * stride_ddacs_csize,
            mask=(offs > 0) & valid,
            other=0.0,
        ).to(tl.float32)
        exclusive_prefix_ddA = tl.cumsum(shifted_ddA, axis=0)

        dA_partial += tl.sum(ddA_cs_rev * exclusive_suffix_dt)
        ddt_out = tl.where(valid, ddt + a_value * exclusive_prefix_ddA, 0.0)
        tl.store(
            DDT_OUT
            + pid_batch * stride_out_batch
            + pid_head * stride_out_head
            + chunk_idx * stride_out_chunk
            + offs * stride_out_csize,
            ddt_out,
            mask=logical_mask,
        )

    tl.store(
        DA_PARTIAL
        + pid_head * stride_partial_head
        + pid_batch * stride_partial_batch,
        dA_partial,
    )


def _validate_decay_stage_inputs(
    ddt: torch.Tensor,
    dt: torch.Tensor,
    workspace: torch.Tensor,
    dA: torch.Tensor,
    A: torch.Tensor,
    chunk_size: int,
    *,
    matrix_workspace: bool = False,
) -> tuple[int, int, int, int]:
    if ddt.ndim != 4 or dt.ndim != 4:
        raise ValueError("ddt and dt must use padded [batch, head, chunk, offset] layout")
    if ddt.shape != dt.shape:
        raise ValueError("ddt and dt must have the same shape")
    folded_batch, nheads, nchunks, stored_chunk_size = dt.shape
    if chunk_size <= 0 or stored_chunk_size != chunk_size:
        raise ValueError("chunk_size must match the padded dt layout")
    expected_workspace_shape = (
        (*dt.shape, chunk_size) if matrix_workspace else dt.shape
    )
    if workspace.shape != expected_workspace_shape:
        raise ValueError(
            f"workspace must have shape {expected_workspace_shape}, "
            f"got {tuple(workspace.shape)}"
        )
    if dA.shape != (nheads,) or A.shape != (nheads,):
        raise ValueError(f"dA and A must each have shape {(nheads,)}")
    tensors = (ddt, dt, workspace, dA, A)
    if not ddt.is_cuda:
        raise ValueError("CP decay-gradient helpers require CUDA tensors")
    if any(tensor.device != ddt.device for tensor in tensors[1:]):
        raise ValueError("all decay-gradient inputs must share a device")
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all decay-gradient inputs must use floating-point dtypes")
    return folded_batch, nheads, nchunks, stored_chunk_size


def _finish_decay_stage(
    dA: torch.Tensor, dA_partial: torch.Tensor
) -> torch.Tensor:
    nheads, folded_batch = dA_partial.shape
    contribution = torch.empty(nheads, device=dA.device, dtype=torch.float32)
    reduce_batch_dimension_kernel[(nheads,)](
        dA_partial,
        contribution,
        nheads,
        folded_batch,
        dA_partial.stride(0),
        dA_partial.stride(1),
    )
    return dA.float() + contribution


def dA_ddt_from_dM_next_cp_triton(
    ddt: torch.Tensor,
    dt: torch.Tensor,
    grad_scale: torch.Tensor,
    dA: torch.Tensor,
    A: torch.Tensor,
    *,
    seqlen: int,
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the dM-next contribution using inclusive directional scans."""

    folded_batch, nheads, nchunks, _ = _validate_decay_stage_inputs(
        ddt, dt, grad_scale, dA, A, chunk_size
    )
    if nchunks != triton.cdiv(seqlen, chunk_size):
        raise ValueError("seqlen does not match the number of padded chunks")
    if out is None:
        out = torch.empty_like(ddt)
    elif out.shape != ddt.shape or out.device != ddt.device or out.dtype != ddt.dtype:
        raise ValueError("out must match ddt shape, device, and dtype")
    dA_partial = torch.empty(nheads, folded_batch, device=ddt.device, dtype=torch.float32)
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    _dA_ddt_from_dM_next_cp_kernel[(nheads, folded_batch)](
        ddt,
        dt,
        grad_scale,
        A,
        out,
        dA_partial,
        ddt.stride(0), ddt.stride(1), ddt.stride(2), ddt.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        grad_scale.stride(0), grad_scale.stride(1),
        grad_scale.stride(2), grad_scale.stride(3),
        A.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dA_partial.stride(0), dA_partial.stride(1),
        seqlen,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
    )
    return out, _finish_decay_stage(dA, dA_partial)


def dA_ddt_from_dM_curr_cp_triton(
    ddt: torch.Tensor,
    dt: torch.Tensor,
    dM_curr: torch.Tensor,
    dA: torch.Tensor,
    A: torch.Tensor,
    *,
    seqlen: int,
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add explicit-dt and stable segsum-backward contributions.

    The large-chunk path consumes and transforms ``dM_curr`` in place using
    separate column reverse scans and row reductions; callers must not use the
    matrix afterward.
    """

    folded_batch, nheads, nchunks, _ = _validate_decay_stage_inputs(
        ddt, dt, dM_curr, dA, A, chunk_size, matrix_workspace=True
    )
    if nchunks != triton.cdiv(seqlen, chunk_size):
        raise ValueError("seqlen does not match the number of padded chunks")
    if out is None:
        out = torch.empty_like(ddt)
    elif out.shape != ddt.shape or out.device != ddt.device or out.dtype != ddt.dtype:
        raise ValueError("out must match ddt shape, device, and dtype")
    dA_partial = torch.empty(nheads, folded_batch, device=ddt.device, dtype=torch.float32)
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    if chunk_size <= 64:
        _dA_ddt_from_dM_curr_cp_kernel[(nheads, folded_batch)](
            ddt,
            dt,
            dM_curr,
            A,
            out,
            dA_partial,
            ddt.stride(0), ddt.stride(1), ddt.stride(2), ddt.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2),
            dM_curr.stride(3), dM_curr.stride(4),
            A.stride(0),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dA_partial.stride(0), dA_partial.stride(1),
            seqlen,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
        )
    else:
        column_grid = (nheads, folded_batch, nchunks * chunk_size)
        _dA_ddt_from_dM_curr_columns_cp_kernel[column_grid](
            ddt,
            dt,
            dM_curr,
            out,
            ddt.stride(0), ddt.stride(1), ddt.stride(2), ddt.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2),
            dM_curr.stride(3), dM_curr.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seqlen,
            nchunks,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            num_warps=4,
            num_stages=1,
        )
        _dA_ddt_from_dM_curr_rows_cp_kernel[(nheads, folded_batch)](
            out,
            dt,
            dM_curr,
            A,
            dA_partial,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2),
            dM_curr.stride(3), dM_curr.stride(4),
            A.stride(0),
            dA_partial.stride(0), dA_partial.stride(1),
            seqlen,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            num_warps=4,
            num_stages=1,
        )
    return out, _finish_decay_stage(dA, dA_partial)


def dA_ddt_from_dM_prev_cp_triton(
    ddt: torch.Tensor,
    dt: torch.Tensor,
    ddA_cs_rev: torch.Tensor,
    dA: torch.Tensor,
    A: torch.Tensor,
    *,
    seqlen: int,
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add dM-prev with shift-then-scan exclusive prefix/suffix sums."""

    folded_batch, nheads, nchunks, _ = _validate_decay_stage_inputs(
        ddt, dt, ddA_cs_rev, dA, A, chunk_size
    )
    if nchunks != triton.cdiv(seqlen, chunk_size):
        raise ValueError("seqlen does not match the number of padded chunks")
    if out is None:
        out = torch.empty_like(ddt)
    elif out.shape != ddt.shape or out.device != ddt.device or out.dtype != ddt.dtype:
        raise ValueError("out must match ddt shape, device, and dtype")
    dA_partial = torch.empty(nheads, folded_batch, device=ddt.device, dtype=torch.float32)
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    _dA_ddt_from_dM_prev_cp_kernel[(nheads, folded_batch)](
        ddt,
        dt,
        ddA_cs_rev,
        A,
        out,
        dA_partial,
        ddt.stride(0), ddt.stride(1), ddt.stride(2), ddt.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        ddA_cs_rev.stride(0), ddA_cs_rev.stride(1),
        ddA_cs_rev.stride(2), ddA_cs_rev.stride(3),
        A.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dA_partial.stride(0), dA_partial.stride(1),
        seqlen,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
    )
    return out, _finish_decay_stage(dA, dA_partial)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK"],
)
@triton.jit
def _mamba2_bwd_dt_softplus_cp_kernel(
    RAW_DT,
    PROCESSED_DDT,
    DT_BIAS,
    RAW_DDT,
    BIAS_PARTIAL,
    stride_raw_stream,
    stride_raw_seq,
    stride_raw_batch,
    stride_raw_head,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_chunk,
    stride_ddt_csize,
    stride_bias_head,
    stride_out_stream,
    stride_out_seq,
    stride_out_batch,
    stride_out_head,
    stride_partial_head,
    stride_partial_batch,
    seqlen,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Map chunked processed-dt gradients back through stable softplus."""

    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH
    offs = tl.arange(0, BLOCK_CHUNK)
    logical_mask = offs < CHUNK_SIZE
    bias = tl.load(DT_BIAS + pid_head * stride_bias_head).to(tl.float32)
    bias_partial = 0.0
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        seq = chunk_idx * CHUNK_SIZE + offs
        valid = logical_mask & (seq < seqlen)
        raw = tl.load(
            RAW_DT
            + stream * stride_raw_stream
            + seq * stride_raw_seq
            + batch_idx * stride_raw_batch
            + pid_head * stride_raw_head,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        ddt = tl.load(
            PROCESSED_DDT
            + pid_folded_batch * stride_ddt_batch
            + pid_head * stride_ddt_head
            + chunk_idx * stride_ddt_chunk
            + offs * stride_ddt_csize,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        u = raw + bias
        softplus_grad = tl.where(u <= 20.0, tl.sigmoid(u), 1.0)
        raw_grad = tl.where(valid, ddt * softplus_grad, 0.0)
        tl.store(
            RAW_DDT
            + stream * stride_out_stream
            + seq * stride_out_seq
            + batch_idx * stride_out_batch
            + pid_head * stride_out_head,
            raw_grad,
            mask=valid,
        )
        bias_partial += tl.sum(raw_grad)

    tl.store(
        BIAS_PARTIAL
        + pid_head * stride_partial_head
        + pid_folded_batch * stride_partial_batch,
        bias_partial,
    )


def _mamba2_bwd_decay_cp_workspace_oracle(
    raw_dt_streams: torch.Tensor,  # (streams, seqlen, batch, nheads)
    dt: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    A: torch.Tensor,  # (nheads,)
    dt_bias: torch.Tensor,  # (nheads,)
    initial_dA: torch.Tensor,  # (nheads,)
    initial_ddt: torch.Tensor,  # chunked like dt
    ddA_cs_rev: torch.Tensor,  # chunked like dt
    grad_scale: torch.Tensor,  # chunked like dt
    dM_curr: torch.Tensor,  # chunk-square
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(raw_dt_grad_streams, grad_A_log, grad_dt_bias)``."""

    if raw_dt_streams.ndim != 4:
        raise ValueError("raw_dt_streams must have shape [streams, seqlen, batch, heads]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    streams, seqlen, batch, nheads = raw_dt_streams.shape
    if streams != 2:
        raise ValueError("true-CP decay backward requires exactly two streams")
    folded_batch = streams * batch
    expected_dt_shape = (
        folded_batch,
        nheads,
        triton.cdiv(seqlen, chunk_size),
        chunk_size,
    )
    if dt.shape != expected_dt_shape:
        raise ValueError(f"dt must have shape {expected_dt_shape}, got {tuple(dt.shape)}")
    for name, tensor in (
        ("initial_ddt", initial_ddt),
        ("ddA_cs_rev", ddA_cs_rev),
        ("grad_scale", grad_scale),
    ):
        if tensor.shape != expected_dt_shape:
            raise ValueError(
                f"{name} must have shape {expected_dt_shape}, got {tuple(tensor.shape)}"
            )
    expected_dm_shape = (*expected_dt_shape, chunk_size)
    if dM_curr.shape != expected_dm_shape:
        raise ValueError(
            f"dM_curr must have shape {expected_dm_shape}, got {tuple(dM_curr.shape)}"
        )
    if A.shape != (nheads,) or dt_bias.shape != (nheads,) or initial_dA.shape != (nheads,):
        raise ValueError(f"A, dt_bias, and initial_dA must each have shape {(nheads,)}")
    tensors = (
        raw_dt_streams,
        dt,
        A,
        dt_bias,
        initial_dA,
        initial_ddt,
        ddA_cs_rev,
        grad_scale,
        dM_curr,
    )
    if not raw_dt_streams.is_cuda:
        raise ValueError("mamba2_bwd_decay_cp_triton requires CUDA tensors")
    if any(tensor.device != raw_dt_streams.device for tensor in tensors[1:]):
        raise ValueError("all decay-backward inputs must share a device")
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all decay-backward inputs must use floating-point dtypes")

    buffer_a = torch.empty_like(initial_ddt)
    buffer_b = torch.empty_like(initial_ddt)
    ddt, dA = dA_ddt_from_dM_next_cp_triton(
        initial_ddt,
        dt,
        grad_scale,
        initial_dA,
        A,
        seqlen=seqlen,
        chunk_size=chunk_size,
        out=buffer_a,
    )
    ddt, dA = dA_ddt_from_dM_curr_cp_triton(
        ddt,
        dt,
        dM_curr,
        dA,
        A,
        seqlen=seqlen,
        chunk_size=chunk_size,
        out=buffer_b,
    )
    ddt, dA = dA_ddt_from_dM_prev_cp_triton(
        ddt,
        dt,
        ddA_cs_rev,
        dA,
        A,
        seqlen=seqlen,
        chunk_size=chunk_size,
        out=buffer_a,
    )

    raw_dt_grad = torch.empty(
        raw_dt_streams.shape,
        device=raw_dt_streams.device,
        dtype=raw_dt_streams.dtype,
    )
    bias_partial = torch.empty(
        nheads, folded_batch, device=raw_dt_streams.device, dtype=torch.float32
    )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    _mamba2_bwd_dt_softplus_cp_kernel[(nheads, folded_batch)](
        raw_dt_streams,
        ddt,
        dt_bias,
        raw_dt_grad,
        bias_partial,
        raw_dt_streams.stride(0), raw_dt_streams.stride(1),
        raw_dt_streams.stride(2), raw_dt_streams.stride(3),
        ddt.stride(0), ddt.stride(1), ddt.stride(2), ddt.stride(3),
        dt_bias.stride(0),
        raw_dt_grad.stride(0), raw_dt_grad.stride(1),
        raw_dt_grad.stride(2), raw_dt_grad.stride(3),
        bias_partial.stride(0), bias_partial.stride(1),
        seqlen,
        BATCH=batch,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
    )
    grad_dt_bias_float = torch.empty(
        nheads, device=raw_dt_streams.device, dtype=torch.float32
    )
    reduce_batch_dimension_kernel[(nheads,)](
        bias_partial,
        grad_dt_bias_float,
        nheads,
        folded_batch,
        bias_partial.stride(0),
        bias_partial.stride(1),
    )
    grad_A_log = dA.float() * A.float()
    grad_dt_bias = grad_dt_bias_float.to(dtype=dt_bias.dtype)
    return raw_dt_grad, grad_A_log, grad_dt_bias


@triton.jit
def _mamba2_bwd_reduce_dA_chunk_cp_kernel(
    DA_CHUNK,
    A,
    GRAD_A_LOG,
    stride_da_batch,
    stride_da_head,
    stride_da_chunk,
    stride_a_head,
    stride_out_head,
    folded_batch,
    nchunks,
):
    """Deterministically reduce compact chunk-local dA contributions."""

    pid_head = tl.program_id(0)
    total = 0.0
    for batch_idx in range(folded_batch):
        for chunk_idx in range(nchunks):
            total += tl.load(
                DA_CHUNK + batch_idx * stride_da_batch + pid_head * stride_da_head
                + chunk_idx * stride_da_chunk
            ).to(tl.float32)
    a = tl.load(A + pid_head * stride_a_head).to(tl.float32)
    tl.store(GRAD_A_LOG + pid_head * stride_out_head, total * a)


def mamba2_bwd_decay_map_cp_triton(
    raw_dt_streams: torch.Tensor,  # (streams, seqlen, batch, nheads)
    processed_ddt: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_chunk: torch.Tensor,  # (folded_batch, nheads, nchunks)
    A: torch.Tensor,  # (nheads,)
    dt_bias: torch.Tensor,  # (nheads,)
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map compact decay gradients to raw dt, A_log, and dt_bias."""

    if raw_dt_streams.ndim != 4:
        raise ValueError("raw_dt_streams must have shape [streams, seqlen, batch, heads]")
    if processed_ddt.ndim != 4 or dA_chunk.ndim != 3:
        raise ValueError("processed_ddt must be 4D and dA_chunk must be 3D")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    streams, seqlen, batch, nheads = raw_dt_streams.shape
    if streams != 2:
        raise ValueError("true-CP decay mapping requires exactly two streams")
    folded_batch = streams * batch
    nchunks = triton.cdiv(seqlen, chunk_size)
    expected_ddt = (folded_batch, nheads, nchunks, chunk_size)
    if processed_ddt.shape != expected_ddt:
        raise ValueError(
            f"processed_ddt must have shape {expected_ddt}, got {tuple(processed_ddt.shape)}"
        )
    if dA_chunk.shape != (folded_batch, nheads, nchunks):
        raise ValueError("dA_chunk has the wrong compact shape")
    if A.shape != (nheads,) or dt_bias.shape != (nheads,):
        raise ValueError(f"A and dt_bias must each have shape {(nheads,)}")
    tensors = (raw_dt_streams, processed_ddt, dA_chunk, A, dt_bias)
    if not raw_dt_streams.is_cuda:
        raise ValueError("mamba2_bwd_decay_map_cp_triton requires CUDA tensors")
    if any(t.device != raw_dt_streams.device for t in tensors[1:]):
        raise ValueError("all decay-mapping inputs must share a device")
    if processed_ddt.dtype is not torch.float32 or dA_chunk.dtype is not torch.float32:
        raise ValueError("compact decay accumulators must be FP32")

    raw_dt_grad = torch.empty_like(raw_dt_streams)
    bias_partial = torch.empty(
        nheads, folded_batch, device=raw_dt_streams.device, dtype=torch.float32
    )
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    _mamba2_bwd_dt_softplus_cp_kernel[(nheads, folded_batch)](
        raw_dt_streams,
        processed_ddt,
        dt_bias,
        raw_dt_grad,
        bias_partial,
        raw_dt_streams.stride(0), raw_dt_streams.stride(1),
        raw_dt_streams.stride(2), raw_dt_streams.stride(3),
        processed_ddt.stride(0), processed_ddt.stride(1),
        processed_ddt.stride(2), processed_ddt.stride(3),
        dt_bias.stride(0),
        raw_dt_grad.stride(0), raw_dt_grad.stride(1),
        raw_dt_grad.stride(2), raw_dt_grad.stride(3),
        bias_partial.stride(0), bias_partial.stride(1),
        seqlen,
        BATCH=batch,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
    )
    grad_dt_bias_float = torch.empty(
        nheads, device=raw_dt_streams.device, dtype=torch.float32
    )
    reduce_batch_dimension_kernel[(nheads,)](
        bias_partial,
        grad_dt_bias_float,
        nheads,
        folded_batch,
        bias_partial.stride(0),
        bias_partial.stride(1),
    )
    grad_A_log = torch.empty(nheads, device=A.device, dtype=torch.float32)
    _mamba2_bwd_reduce_dA_chunk_cp_kernel[(nheads,)](
        dA_chunk,
        A,
        grad_A_log,
        dA_chunk.stride(0), dA_chunk.stride(1), dA_chunk.stride(2),
        A.stride(0),
        grad_A_log.stride(0),
        folded_batch,
        nchunks,
        num_warps=1,
        num_stages=1,
    )
    return raw_dt_grad, grad_A_log, grad_dt_bias_float.to(dtype=dt_bias.dtype)


@triton.jit
def chunk_cumsum_kernel(
    dA,  # Input: (batch, nheads, seqlen)
    dA_cs,  # Output: (batch, nheads, seqlen) - forward cumsum
    dA_cs_rev,  # Output: (batch, nheads, seqlen) - reverse cumsum
    scalar, # multiply by scalar
    stride_da_batch,
    stride_da_head,
    stride_da_seqlen,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_seqlen,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_seqlen,
    stride_scalar_head,
    seqlen,
    nheads,
    CHUNK_SIZE: tl.constexpr,
    EXP2: tl.constexpr = False,
):
    """
    Compute both forward and reverse cumsum within each chunk of size CHUNK_SIZE.
    Each program handles one (batch, head, chunk) combination.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)
    # Load chunk from global memory
    da_ptr = dA + pid_batch * stride_da_batch + pid_head * stride_da_head
    mask = offs_seqlen < seqlen
    da_chunk = tl.load(da_ptr + offs_seqlen * stride_da_seqlen, mask=mask, other=0.0)

    # load scalar
    scale_val = tl.load(scalar + pid_head * stride_scalar_head)
    da_chunk = da_chunk * scale_val

    # Compute forward cumsum within the chunk
    da_cs_chunk = tl.cumsum(da_chunk, axis=0)
    if EXP2:
        da_cs_chunk = tl.exp2(da_cs_chunk)
    # Store forward cumsum to global memory
    # Exclusive cumsum, so shifted by 1 and 0th element is 0
    dacs_ptr = dA_cs + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
    tl.store(dacs_ptr + offs_seqlen * stride_dacs_seqlen, da_cs_chunk, mask=mask)
    # tl.store(dacs_ptr + 1 + offs_seqlen * stride_dacs_seqlen, da_cs_chunk, mask=tl.arange(0, CHUNK_SIZE) < min(seqlen - 1 - chunk_start, CHUNK_SIZE - 1))
    # tl.store(dacs_ptr + chunk_start, 0.0)
    # Compute and store exclusive reverse cumsum
    da_cs_rev_chunk = tl.cumsum(da_chunk, axis=0, reverse=True)
    if EXP2:
        da_cs_rev_chunk = tl.exp2(da_cs_rev_chunk)
    dacsrev_ptr = dA_cs_rev + pid_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    # tl.store(dacsrev_ptr + offs_seqlen * stride_dacsrev_seqlen, da_cs_rev_chunk, mask=mask)
    tl.store(dacsrev_ptr + (offs_seqlen - 1) * stride_dacsrev_seqlen, da_cs_rev_chunk, mask=mask & (tl.arange(0, CHUNK_SIZE) >= 1))
    tl.store(dacsrev_ptr + min(chunk_start + CHUNK_SIZE - 1, seqlen - 1) * stride_dacsrev_seqlen, 0.0 if not EXP2 else 1.0)


def chunk_cumsum_triton(
    dA: torch.Tensor,  # (batch, nheads, seqlen)
    chunk_size: int = 64,
    dA_cs: Optional[torch.Tensor] = None,
    dA_cs_rev: Optional[torch.Tensor] = None,
    scalar: Optional[torch.Tensor] = None,
    exp2: bool = False,
):
    """
    Compute both forward and reverse cumsum within each chunk of size chunk_size.

    Args:
        dA: Input tensor of shape (batch, nheads, seqlen)
        chunk_size: Size of chunks for cumsum (default: 64)
        dA_cs: Optional output tensor for forward cumsum (batch, nheads, seqlen)
        dA_cs_rev: Optional output tensor for reverse cumsum (batch, nheads, seqlen)

    Returns:
        Tuple of (dA_cs, dA_cs_rev)

    Examples:
        Forward cumsum: [a, b, c, d] -> [a, a+b, a+b+c, a+b+c+d]
        Reverse cumsum: [a, b, c, d] -> [a+b+c+d, b+c+d, c+d, d]
    """
    batch, nheads, seqlen = dA.shape
    if dA_cs is None:
        dA_cs = torch.empty_like(dA)
    if dA_cs_rev is None:
        dA_cs_rev = torch.empty_like(dA)
    assert dA.is_cuda, "Input tensor must be on CUDA"
    assert dA_cs.shape == dA.shape, "Output tensor must have same shape as input"
    assert dA_cs_rev.shape == dA.shape, "Reverse cumsum output must have same shape as input"
    num_chunks = triton.cdiv(seqlen, chunk_size)
    # Grid: (batch, nheads, num_chunks)
    grid = (batch, nheads, num_chunks)
    chunk_cumsum_kernel[grid](
        dA,
        dA_cs,
        dA_cs_rev,
        scalar,
        dA.stride(0),
        dA.stride(1),
        dA.stride(2),
        dA_cs.stride(0),
        dA_cs.stride(1),
        dA_cs.stride(2),
        dA_cs_rev.stride(0),
        dA_cs_rev.stride(1),
        dA_cs_rev.stride(2),
        scalar.stride(0),
        seqlen,
        nheads,
        CHUNK_SIZE=chunk_size,
        EXP2=exp2,
    )
    return dA_cs, dA_cs_rev


@triton.jit
def dA_ddt_from_dM_Next_kernel(
    DDT,         # (batch, nheads, seqlen): ddt_block
    DT,          # (batch, nheads, seqlen): dt
    GSI,         # (batch, nheads, seqlen): grad_scale_intermediate
    DDT_out,     # (batch, nheads, seqlen): output ddt
    DA,          # (nheads,)
    DA_out,
    scalar,      # (nheads,)
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_seqlen,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_seqlen,
    stride_gsi_batch,
    stride_gsi_head,
    stride_gsi_seqlen,
    stride_out_batch,
    stride_out_head,
    stride_out_seqlen,
    stride_da,
    stride_da_out,
    stride_scalar,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs = chunk_start + tl.arange(0, CHUNK_SIZE)
    mask = offs < seqlen

    base_ddt = DDT + pid_batch * stride_ddt_batch + pid_head * stride_ddt_head
    ddt_chunk = tl.load(base_ddt + offs * stride_ddt_seqlen, mask=mask, other=0.0)

    base_dt = DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head
    dt_chunk = tl.load(base_dt + offs * stride_dt_seqlen, mask=mask, other=0.0)

    base_gsi = GSI + pid_batch * stride_gsi_batch + pid_head * stride_gsi_head
    gsi_chunk = tl.load(base_gsi + offs * stride_gsi_seqlen, mask=mask, other=0.0)

    #  a_val
    a_val = tl.load(scalar + pid_head * stride_scalar)

    dt_cumsum = tl.cumsum(dt_chunk)
    da_contrib = tl.sum(dt_cumsum * gsi_chunk)

    da_ptr = DA + pid_head * stride_da
    tl.atomic_add(da_ptr, da_contrib) # TODO: don't use atomic add??

    rev_cumsum = tl.cumsum(gsi_chunk * a_val, axis=0, reverse=True)
    out_chunk = ddt_chunk + rev_cumsum

    base_out = DDT_out + pid_batch * stride_out_batch + pid_head * stride_out_head
    tl.store(base_out + offs * stride_out_seqlen, out_chunk, mask=mask)

def dA_ddt_from_dM_Next(
    ddt_block,                # (batch, nheads, seqlen)
    dt,                       # (batch, nheads, seqlen)
    grad_scale_intermediate,  # (batch, nheads, seqlen)
    dA,                       # (nheads,)
    scalar,                   # (nheads,)
    chunk_size=64,
    out=None,
):
    """
    offloading grad_scale_intermediate to do following computation outside kernel:

    dt_cumsum = tl.cumsum(dt_block)
    da_contribution = tl.sum(dt_cumsum * grad_scale_intermediate)
    dA_val += da_contribution

    ddt = ddt_block+ tl.cumsum(grad_scale_intermediate * a_val, reverse=True)
    """
    batch, nheads, seqlen = ddt_block.shape
    if out is None:
        out = torch.empty_like(ddt_block)

    DA_out = torch.zeros(1, device=dA.device, dtype=dA.dtype)

    num_chunks = triton.cdiv(seqlen, chunk_size)
    grid = (batch, nheads, num_chunks)

    dA_ddt_from_dM_Next_kernel[grid](
        ddt_block,
        dt,
        grad_scale_intermediate,
        out,
        dA,
        DA_out,
        scalar,
        ddt_block.stride(0),
        ddt_block.stride(1),
        ddt_block.stride(2),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        grad_scale_intermediate.stride(0),
        grad_scale_intermediate.stride(1),
        grad_scale_intermediate.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dA.stride(0),
        0,
        scalar.stride(0),
        seqlen,
        CHUNK_SIZE=chunk_size,
    )

    return out, dA

@triton.jit
def dA_ddt_from_dM_Prev_kernel(
    DDT, DT, DDACSREV,
    DDT_out, DA, scalar,
    stride_ddt_batch, stride_ddt_head, stride_ddt_seqlen,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen,
    stride_ddacs_batch, stride_ddacs_head, stride_ddacs_seqlen,
    stride_out_batch, stride_out_head, stride_out_seqlen,
    stride_da, stride_scalar,
    seqlen,
    CHUNK_SIZE: tl.constexpr,
):

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    chunk_start = pid_c * CHUNK_SIZE
    offs = chunk_start + tl.arange(0, CHUNK_SIZE)
    mask = offs < seqlen

    dt_block = tl.load(
        DT + pid_b*stride_dt_batch + pid_h*stride_dt_head + offs*stride_dt_seqlen,
        mask=mask, other=0.0
    )

    ddA_cs_rev = tl.load(
        DDACSREV + pid_b*stride_ddacs_batch + pid_h*stride_ddacs_head + offs*stride_ddacs_seqlen,
        mask=mask, other=0.0
    )

    a_val = tl.load(scalar + pid_h * stride_scalar)

    dt_last = tl.sum(dt_block, axis=0)
    dt_cumsum = tl.cumsum(dt_block)
    dt_cs_rev = dt_last - dt_cumsum

    dA_contrib = tl.sum(ddA_cs_rev * dt_cs_rev, axis=0)
    tl.atomic_add(DA + pid_h * stride_da, dA_contrib) #TODO: don't use atomic add?

    # weird hack where to get shifted cumsum we just load again with a shift
    # old code:

    # ddt_shift_left = tl.cumsum(a_val*ddA_cs_rev)
    # # Hacky roll one to the right:
    # i = tl.arange(0, CHUNK_SIZE)[:, None]          # [N,1]
    # j = tl.arange(0, CHUNK_SIZE)[None, :]          # [1,N]
    # S = (i == j + 1)                      # strictly lower diagonal (one below main)
    # ddt += tl.sum(tl.where(S, ddt_shift_left, 0), axis=1)

    # BUT this was quite slow (materializing a NxN matrix)
    # -> so i came up with this weird hack since now we are offloading -> so we can load it again
    # and weirdly it does not take up too much time
    # 86 us -> 22 us

    base_ptr = (
        DDACSREV
        + pid_b * stride_ddacs_batch
        + pid_h * stride_ddacs_head
    )

    local = offs - chunk_start

    shifted_offs   = offs - 1
    shift_mask     = (local > 0) & (offs < seqlen)

    ddA_cs_rev_shifted = tl.load(
        base_ptr + shifted_offs * stride_ddacs_seqlen,
        mask=shift_mask,
        other=0.0
    )
    ddA_cs_rev_shifted = tl.where(mask, ddA_cs_rev_shifted, 0.0)
    ddt_shift_left = tl.cumsum(a_val * ddA_cs_rev_shifted)

    ddt_block = tl.load(
        DDT + pid_b*stride_ddt_batch + pid_h*stride_ddt_head + offs*stride_ddt_seqlen,
        mask=mask, other=0.0
    )

    out_block = ddt_block + ddt_shift_left

    tl.store(
        DDT_out + pid_b*stride_out_batch + pid_h*stride_out_head + offs*stride_out_seqlen,
        out_block, mask=mask
    )


def dA_ddt_from_dM_Prev(
    ddt, dt, ddA_cs_rev,
    dA, a, chunk_size=64,
    ddt_out=None,
):
    batch, nheads, seqlen = ddt.shape
    if ddt_out is None:
        ddt_out = torch.empty_like(ddt)

    num_chunks = triton.cdiv(seqlen, chunk_size)
    grid = (batch, nheads, num_chunks)

    dA_ddt_from_dM_Prev_kernel[grid](
        ddt,
        dt,
        ddA_cs_rev,
        ddt_out,
        dA,
        a,
        ddt.stride(0), ddt.stride(1), ddt.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        ddA_cs_rev.stride(0), ddA_cs_rev.stride(1), ddA_cs_rev.stride(2),
        ddt_out.stride(0), ddt_out.stride(1), ddt_out.stride(2),
        dA.stride(0),
        a.stride(0),
        seqlen,
        CHUNK_SIZE=chunk_size,
    )

    return ddt_out, dA

@triton.jit
def dA_ddt_from_dM_Curr_kernel(
    DM,         # (batch, nheads, nchunks, CS, CS)
    DT,         # (batch, nheads, seqlen)
    DDT_OUT,    # (batch, nheads, seqlen)
    D_A,        # (nheads,)
    AVAL,       # (nheads,)
    stride_dm_b, stride_dm_h, stride_dm_c, stride_dm_i, stride_dm_j,
    stride_dt_b, stride_dt_h, stride_dt_s,
    stride_ddt_b, stride_ddt_h, stride_ddt_s,
    stride_a,
    stride_da,
    seqlen, nchunks,
    CHUNK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    chunk_start = pid_c * CHUNK_SIZE
    offs_s = chunk_start + tl.arange(0, CHUNK_SIZE)
    seq_mask = offs_s < seqlen

    # load dM_Curr
    offs_i = tl.arange(0, CHUNK_SIZE)[:, None]
    offs_j = tl.arange(0, CHUNK_SIZE)[None, :]

    dm_ptr = (
        DM
        + pid_b * stride_dm_b
        + pid_h * stride_dm_h
        + pid_c * stride_dm_c
    )

    dM = tl.load(
        dm_ptr + offs_i * stride_dm_i + offs_j * stride_dm_j,
        mask=(offs_s[:, None] < seqlen) & (offs_s[None, :] < seqlen),
        other=0.0,
    )

    dt_ptr = DT + pid_b * stride_dt_b + pid_h * stride_dt_h
    dt_block = tl.load(dt_ptr + offs_s * stride_dt_s, mask=seq_mask, other=0.0)


    ddt_update = tl.sum(dM, axis=0)
    ddt_ptr = DDT_OUT + pid_b * stride_ddt_b + pid_h * stride_ddt_h
    ddt_block = tl.load(ddt_ptr + offs_s * stride_ddt_s, mask=seq_mask, other=0.0)

    ddt_update += ddt_block

    dSSdA = dM * dt_block[None, :]

    SS_mask = offs_i >= (offs_j + 1)
    dSSdA = tl.where(SS_mask, dSSdA, 0.0)
    dSSdA = tl.cumsum(dSSdA, axis=0, reverse=True)
    dSSdA = tl.where(SS_mask, dSSdA, 0.0)
    ddA = tl.sum(dSSdA, axis=1)

    a_val = tl.load(AVAL + pid_h * stride_a)
    ddt_update += ddA * a_val

    tl.store(ddt_ptr + offs_s * stride_ddt_s,
             ddt_update,
             mask=seq_mask)


    dA_chunk = tl.sum(ddA * dt_block)
    da_ptr = D_A + pid_h * stride_da
    tl.atomic_add(da_ptr, dA_chunk)


def dA_ddt_from_dM_Curr(
    dM,        # (batch, nheads, nchunks, CS, CS)
    dt,        # (batch, nheads, seqlen)
    ddt,       # (batch, nheads, seqlen)
    a_val,     # (nheads,)
    dA,
    chunk_size,
):
    """
    Computes:
        ddt += sum(dM, axis=0)
        dSSdA = dM * dt_block[j]
        backward-through-segsum(dSSdA) -> ddA
        ddt += a_val * ddA
        dA[h] += sum(ddA * dt_block)

    Returns:
        ddt_out  (same shape as ddt)
        dA       (nheads,)
    """
    device = dM.device
    dtype = dM.dtype

    batch, nheads, nchunks, _, _ = dM.shape
    seqlen = dt.shape[-1]
    CHUNK_SIZE = chunk_size


    grid = (batch, nheads, nchunks)

    dA_ddt_from_dM_Curr_kernel[grid](
        dM,
        dt,
        ddt,
        dA,
        a_val,
        dM.stride(0), dM.stride(1), dM.stride(2),
        dM.stride(3), dM.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2),
        ddt.stride(0), ddt.stride(1), ddt.stride(2),
        a_val.stride(0),
        dA.stride(0),
        seqlen,
        nchunks,
        CHUNK_SIZE=CHUNK_SIZE,
    )

    return ddt, dA


@triton.jit
def reduce_batch_dimension_kernel(
    input_ptr, output_ptr,
    nheads, batch,
    stride_input_head, stride_input_batch,
):
    """
    Reduce over batch dimension for small tensors like dA, dD.
    Input: (nheads, batch)
    Output: (nheads,)

    Each program handles one head.
    """
    head_id = tl.program_id(0)

    # Accumulate over batch dimension
    acc = 0.0
    for batch_id in range(batch):
        offset = head_id * stride_input_head + batch_id * stride_input_batch
        val = tl.load(input_ptr + offset)
        acc += val

    # Store result
    tl.store(output_ptr + head_id, acc)


@triton.jit
def reduce_head_groups_kernel(
    input_ptr, output_ptr,
    batch, seqlen, nheads, nheads_qk, headdim,
    stride_input_batch, stride_input_seq, stride_input_head, stride_input_dim,
    stride_output_batch, stride_output_seq, stride_output_head, stride_output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduce over head groups for GQA.
    Input: (batch, seqlen, nheads, headdim) where nheads = nheads_qk * ngroups
    Output: (batch, seqlen, nheads_qk, headdim)

    Each program handles one (batch, seqlen, qk_head, dim_block) position.
    """
    pid = tl.program_id(0)
    ngroups = nheads // nheads_qk

    # Decode program ID
    num_dim_blocks = tl.cdiv(headdim, BLOCK_SIZE)

    pos_id = pid // num_dim_blocks
    dim_block_id = pid % num_dim_blocks

    batch_id = pos_id // (seqlen * nheads_qk)
    remaining = pos_id % (seqlen * nheads_qk)
    seq_id = remaining // nheads_qk
    qk_head_id = remaining % nheads_qk

    # Dimension offsets
    dim_offset = dim_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_offset < headdim

    # Accumulate over head groups
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for group_idx in range(ngroups):
        v_head_id = qk_head_id * ngroups + group_idx
        input_offset = (batch_id * stride_input_batch +
                       seq_id * stride_input_seq +
                       v_head_id * stride_input_head)
        input_ptrs = input_ptr + input_offset + dim_offset * stride_input_dim
        data = tl.load(input_ptrs, mask=dim_mask, other=0.0)
        acc += data.to(tl.float32)

    # Store result
    output_offset = (batch_id * stride_output_batch +
                    seq_id * stride_output_seq +
                    qk_head_id * stride_output_head)
    output_ptrs = output_ptr + output_offset + dim_offset * stride_output_dim
    tl.store(output_ptrs, acc, mask=dim_mask)


# best config selected: num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: 256;
# best config selected: num_warps: 8, num_ctas: 1, num_stages: 1, maxnreg: None;
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)#, maxnreg=r)
        for s in [1] #[1, 2, 3, 4] #[1, 2, 3, 4] #[1]
        for w in [8] #[1, 2, 4, 8] #[2, 4, 8] #[4]
        # for r in [256, 512] #[256] #[128, 256]
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "BLOCK_HEADDIM_V"]
)
@triton.jit
def mamba2_bwd_kernel_dk_dv(
    Q, K, V, dO, dK, dV, A, DT, DA, DDT, D, DD, Z, states, grad_scale_dt_out, DA_CS, GRAD_SCALE_INTERM, dSSdA_OUT,
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_dim,
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_dim,
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_dim,
    stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_dim,
    stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_dim,
    stride_a_nhead,
    stride_dt_batch, stride_dt_seqlen, stride_dt_nhead,
    stride_da_nhead, stride_da_batch,
    stride_ddt_batch, stride_ddt_seqlen, stride_ddt_nhead,
    stride_d_nhead,
    stride_dd_nhead, stride_dd_batch,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_dim,
    stride_states_batch, stride_states_nheads, stride_states_nchunks, stride_states_qk_dim, stride_states_v_dim,
    stride_grad_scale_batch, stride_grad_scale_head, stride_grad_scale_seqlen,
    stride_dacs_batch, stride_dacs_head, stride_dacs_seqlen,
    stride_gsi_batch, stride_gsi_nhead, stride_gsi_seqlen,
    stride_dSS_batch, stride_dSS_head, stride_dSS_chunk, stride_dSS_i, stride_dSS_j,
    seqlen, headdim_qk, headdim_v, nheads_qk,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    COMPUTE_SCALE_GRAD: tl.constexpr,
    TIED_HEADS: tl.constexpr,
    OFFLOAD_DDA_CSR: tl.constexpr,
    OFFLOAD_GSI: tl.constexpr,
    OFFLOAD_dM_Curr: tl.constexpr,
    LOAD_DDT: tl.constexpr
):
    """
    Mamba2 backward kernel for dK and dV in Triton.
    Each program instance handles one entire sequence for one (batch, head) pair.

    Algorithm (iterate backwards through chunks):
    For each chunk from last to first:
        1. Compute dK[chunk]:
           - dP^T = V[chunk] @ dO[chunk]^T (with causal transpose mask and exponential decay)
           - dK = V[chunk] @ dStates_future^T * scale + dP^T @ Q[chunk]
           - scale = exp(cumsum_rev(dt * a)) * dt
        2. Compute dV[chunk]:
           - P^T = K[chunk] @ Q[chunk]^T (with causal transpose mask and exponential decay)
           - dV = K[chunk] @ dStates_future * scale + P^T @ dO[chunk] + dO[chunk] * D
        3. Optionally compute dA and ddt gradients (when COMPUTE_SCALE_GRAD=True):
           - Compute dM_S from state contribution
           - Compute dM_Prev from inter-chunk state passing
           - Accumulate gradients for A and dt parameters
        4. Update dStates for next (previous) chunk:
           - dStates = exp(da_cs_last) * dStates + Q[chunk]^T @ (dO[chunk] * scale_m)
           - scale_m = exp(cumsum(dt * a))
        5. Optionally compute dD gradient from dO * V
    """
    # Program ID: which head and batch
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    # Compute head index for Q/K (for GQA support)
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    # Offset pointers to current batch and head
    q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head
    # Use pid_head for dK to avoid atomic operations in TIED_HEADS case
    dk_offset = pid_batch * stride_dk_batch + pid_head * stride_dk_head
    dv_offset = pid_batch * stride_dv_batch + pid_head * stride_dv_head

    dt_offset = pid_batch * stride_dt_batch + pid_head * stride_dt_nhead

    if OFFLOAD_DDA_CSR:
        base_grad_scale = grad_scale_dt_out + pid_batch * stride_grad_scale_batch + pid_head  * stride_grad_scale_head

    if COMPUTE_SCALE_GRAD:
        ddt_offset = pid_batch * stride_ddt_batch + pid_head * stride_ddt_nhead
        states_offset = pid_batch * stride_states_batch + pid_head * stride_states_nheads
        # TODO: load dA depending whether it's been passed in
        dA_val = 0.0 #tl.load(DA + pid_head * stride_da_nhead)
    if Z is not None:
        z_offset = pid_batch * stride_z_batch + pid_head * stride_z_head

    a_val = tl.load(A + pid_head * stride_a_nhead)
    if DD is not None:
        dD_val = 0.0  # initialize gradient for D

    # Initialize cumulative gradient states: dStates = sum(Q[i]^T @ dO[i]) for future blocks
    acc_dstates = tl.zeros([BLOCK_HEADDIM_QK, BLOCK_HEADDIM_V], dtype=tl.float32)

    # Compute number of chunks
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    offs_dstates = tl.arange(0, BLOCK_HEADDIM_QK)
    offs_headdim = tl.arange(0, BLOCK_HEADDIM_V)

    # Process each chunk sequentially in REVERSE order
    for chunk_idx_rev in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_rev
        chunk_start = chunk_idx * CHUNK_SIZE

        # ============================================================
        # Load Q, K, V, dO for current chunk
        # ============================================================
        # Load Q block: (CHUNK_SIZE, headdim_qk)
        offs_q_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_q_dim = tl.arange(0, BLOCK_HEADDIM_QK)
        q_ptrs = Q + q_offset + offs_q_seq[:, None] * stride_q_seqlen + offs_q_dim[None, :] * stride_q_dim
        q_mask = (offs_q_seq[:, None] < seqlen) & (offs_q_dim[None, :] < headdim_qk)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
        # q_block = tl.load(q_ptrs)

        # Load K block: (CHUNK_SIZE, headdim_qk)
        offs_k_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_k_dim = tl.arange(0, BLOCK_HEADDIM_QK)
        k_ptrs = K + k_offset + offs_k_seq[:, None] * stride_k_seqlen + offs_k_dim[None, :] * stride_k_dim
        k_mask = (offs_k_seq[:, None] < seqlen) & (offs_k_dim[None, :] < headdim_qk)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # k_block = tl.load(k_ptrs)

        # Load V block: (CHUNK_SIZE, headdim_v)
        offs_v_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_v_dim = tl.arange(0, BLOCK_HEADDIM_V)
        v_ptrs = V + v_offset + offs_v_seq[:, None] * stride_v_seqlen + offs_v_dim[None, :] * stride_v_dim
        v_mask = (offs_v_seq[:, None] < seqlen) & (offs_v_dim[None, :] < headdim_v)
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)
        # v_block = tl.load(v_ptrs)

        # Load dO block: (CHUNK_SIZE, headdim_v)
        offs_do_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_do_dim = tl.arange(0, BLOCK_HEADDIM_V)
        do_ptrs = dO + do_offset + offs_do_seq[:, None] * stride_do_seqlen + offs_do_dim[None, :] * stride_do_dim
        do_mask = (offs_do_seq[:, None] < seqlen) & (offs_do_dim[None, :] < headdim_v)
        do_block = tl.load(do_ptrs, mask=do_mask, other=0.0)
        # do_block = tl.load(do_ptrs)

        if Z is not None:
            # load Z
            offs_z_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
            offs_z_dim = tl.arange(0, BLOCK_HEADDIM_V)
            z_ptrs = Z + z_offset + offs_z_seq[:, None] * stride_z_seqlen + offs_z_dim[None, :] * stride_z_dim
            z_mask = (offs_z_seq[:, None] < seqlen) & (offs_z_dim[None, :] < headdim_v)
            z_block = tl.load(z_ptrs, mask=z_mask, other=0.0)
            # z_block = tl.load(z_ptrs)

            sigmoid_z = tl.sigmoid(z_block.to(tl.float32))
            # Scale do_block by sigmoid(z)
            # Convert sigmoid_z back to original dtype to maintain consistency
            do_block = do_block * sigmoid_z.to(do_block.dtype)


        # Compute dD gradient
        if D is not None:
            dD_val += tl.sum(do_block * v_block)

        # ============================================================
        # Load dt values and compute cumulative decay for this chunk
        # ============================================================
        offs_dt_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        dt_ptrs = DT + dt_offset + offs_dt_seq * stride_dt_seqlen
        dt_mask = offs_dt_seq < seqlen
        dt_block = tl.load(dt_ptrs, mask=dt_mask, other=0.0)  # (CHUNK_SIZE,)
        # dt_block = tl.load(dt_ptrs)

        if COMPUTE_SCALE_GRAD:
            ddt_ptrs = DDT + ddt_offset + offs_dt_seq * stride_ddt_seqlen
            if LOAD_DDT:
                ddt_block = tl.load(ddt_ptrs)
            else:
                ddt_block = tl.zeros_like(dt_block)

        # Compute log_decay = dt * a for each position in chunk
        # Only compute for valid positions within sequence length
        da = dt_block * a_val  # (CHUNK_SIZE,)
        da_cs = tl.cumsum(da)

        dacs_ptrs = (
            DA_CS
            + pid_batch * stride_dacs_batch
            + pid_head  * stride_dacs_head
            + offs_dt_seq * stride_dacs_seqlen
        )

        da_cs_last = tl.sum(da)
        dA_cs_rev = da_cs_last - da_cs # Reverse cumsum: sum(dA[i+1:])
        e_dA_cs_rev = tl.exp(dA_cs_rev)
        scale = e_dA_cs_rev * dt_block
        tl.store(dacs_ptrs, e_dA_cs_rev, mask=dt_mask) #TODO: why does it get slower when you remove this?

        # ============================================================
        # Compute dK for current chunk
        # ============================================================
        # dP^T = V @ dO^T: (CHUNK_SIZE, headdim_v) @ (headdim_v, CHUNK_SIZE)
        dp_t_block = tl.dot(v_block, tl.trans(do_block))

        # Compute the segsum and dt
        # dp_t_block[j, i] needs scaling by exp(min(dA_cs[i] - dA_cs[j], 0)) * dt_block[j]
        fwd_segsum = tl.exp(tl.minimum((da_cs[None, :] - da_cs[:, None]), 0.0))
        dp_t_block *= fwd_segsum
        dp_t_block *= dt_block[:, None]

        # Apply causal mask (same as forward pass)
        offs_m_local = tl.arange(0, CHUNK_SIZE)
        offs_n_local = tl.arange(0, CHUNK_SIZE)
        causal_mask = offs_m_local[None, :] >= offs_n_local[:, None]
        dp_t_block = tl.where(causal_mask, dp_t_block, 0.0)

        acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)


        # dK = V @ dStates^T: (CHUNK_SIZE, headdim_v) @ (headdim_v, headdim_qk)
        # NOTE (BC): the hideous transpose expression below is due to speed,
        # where the obvious way of tl.dot is actually 200us slower.
        acc_dk += tl.trans(
            tl.dot(
                acc_dstates.to(v_block.dtype),
                tl.trans(v_block)
                )* scale[None, :]
            )

        # dK += dP^T @ Q: (CHUNK_SIZE, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_qk)
        # acc_dk += tl.dot(dp_t_block.to(q_block.dtype), q_block)

        # Store dK (now using pid_head instead of head_idx_qk in dk_offset)
        offs_dk_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_dk_dim = tl.arange(0, BLOCK_HEADDIM_QK)
        dk_ptrs = dK + dk_offset + offs_dk_seq[:, None] * stride_dk_seqlen + offs_dk_dim[None, :] * stride_dk_dim
        dk_mask = (offs_dk_seq[:, None] < seqlen) & (offs_dk_dim[None, :] < headdim_qk)
        tl.store(dk_ptrs, acc_dk.to(q_block.dtype), mask=dk_mask)

        # ============================================================
        # Compute dV for current chunk
        # ============================================================
        # P^T = K @ Q^T: (CHUNK_SIZE, headdim_qk) @ (headdim_qk, CHUNK_SIZE)
        p_t_block = tl.dot(k_block, tl.trans(q_block))

        # Compute the segsum and dt scaling (with transposed indices)
        p_t_block *= tl.exp(tl.minimum((da_cs[None, :] - da_cs[:, None]), 0.0))
        p_t_block *= dt_block[:, None]

        # Apply causal mask (same as forward pass)
        p_t_block = tl.where(causal_mask, p_t_block, 0.0)

        acc_dv = tl.dot(p_t_block.to(do_block.dtype), do_block)

        # TODO (BC): acc_dv_intermediate was designed to be reused in dM_Prev; however,
        # if we do
        # acc_dv_intermediate = tl.dot(k_block, acc_dstates.to(k_block.dtype)) * scale[:, None]
        # acc_dv += acc_dv_intermediate * scale[:, None]
        # which is incorrect for dM_Prev but correct for dV, the runtime is ~100us faster.
        # The cause for the speedup is unknown atm.

        # dV = K @ dStates: (CHUNK_SIZE, headdim_qk) @ (headdim_qk, headdim_v)
        acc_dv_intermediate = tl.dot(k_block, acc_dstates.to(k_block.dtype))

        # dV += P^T @ dO: (CHUNK_SIZE, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_v)
        acc_dv += acc_dv_intermediate * scale[:, None]

        # Add gradient contribution from D: dV += dO * D
        if D is not None:
            d_val = tl.load(D + pid_head * stride_d_nhead)
            acc_dv += do_block * d_val

        # Store dV
        offs_dv_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_dv_dim = tl.arange(0, BLOCK_HEADDIM_V)
        dv_ptrs = dV + dv_offset + offs_dv_seq[:, None] * stride_dv_seqlen + offs_dv_dim[None, :] * stride_dv_dim
        dv_mask = (offs_dv_seq[:, None] < seqlen) & (offs_dv_dim[None, :] < headdim_v)
        tl.store(dv_ptrs, acc_dv.to(do_block.dtype), mask=dv_mask)
        # tl.store(dv_ptrs, acc_dv.to(do_block.dtype))

        # ============================================================
        # Compute da and ddt for current chunk (if applicable)
        # ============================================================

        if COMPUTE_SCALE_GRAD:
            ddt = ddt_block

            ### Compute dM_Curr:
            if OFFLOAD_dM_Curr:
                QKt = tl.dot(q_block, tl.trans(k_block)) # TODO Consider using p_t_block
                # TODO: may want to modify the other kernel to multiply s_block_intermediate,
                # in order to make things faster
                eSSdA = tl.exp(tl.minimum((da_cs[:, None] - da_cs[None, :]), 0.0))
                s_block_intermediate = tl.dot(do_block, tl.trans(v_block)) # TODO Consider using dp_t_block
                s_block_intermediate *= eSSdA
                offs_m_local = tl.arange(0, CHUNK_SIZE)
                offs_n_local = tl.arange(0, CHUNK_SIZE)
                causal_mask = offs_m_local[:, None] >= offs_n_local[None, :]
                s_block_intermediate = tl.where(causal_mask, s_block_intermediate, 0.0)
                dM_Curr = QKt * s_block_intermediate # This is actually dM_Curr * L * exp(SegSum(dA))
                # dM_Curr = p_t_block * dp_t_block # This is actually dM_Curr * L * exp(SegSum(dA))

                i = tl.arange(0, CHUNK_SIZE)
                dSS_ptrs = (
                    dSSdA_OUT
                    + pid_batch * stride_dSS_batch
                    + pid_head  * stride_dSS_head
                    + chunk_idx * stride_dSS_chunk
                    + i[:, None] * stride_dSS_i
                    + i[None, :] * stride_dSS_j
                )

                tl.store(dSS_ptrs, dM_Curr)


            ### Compute dM_Prev:
            dM_Prev = acc_dv_intermediate * v_block # dim: (chunksize, BLOCK_HEADDIM_V)
            # dM_Prev = tl.dot(k_block, acc_dstates.to(k_block.dtype)) * v_block
            # Use dM_Prev to compute dA_val and ddt:
            grad_scale_dt = tl.sum(dM_Prev, axis=1)
            ddt += grad_scale_dt * e_dA_cs_rev
            grad_scale = grad_scale_dt * dt_block
            ddA_cs_rev = e_dA_cs_rev*grad_scale

            if OFFLOAD_DDA_CSR:
                # store ddA_cs_rev for offloading
                tl.store(
                    base_grad_scale + offs_k_seq * stride_grad_scale_seqlen,
                    ddA_cs_rev,
                    mask=offs_k_seq < seqlen
                )
            else:
                dt_last = tl.sum(dt_block)
                dt_cs_rev = dt_last - tl.cumsum(dt_block)
                dA_val += tl.sum(ddA_cs_rev * dt_cs_rev)

                ddt_shift_left = tl.cumsum(a_val*ddA_cs_rev)

                # Hacky roll one to the right:
                i = tl.arange(0, CHUNK_SIZE)[:, None]          # [N,1]
                j = tl.arange(0, CHUNK_SIZE)[None, :]          # [1,N]
                S = (i == j + 1)                      # strictly lower diagonal (one below main)
                ddt += tl.sum(tl.where(S, ddt_shift_left, 0), axis=1)


            ## Load curr States from gmem:
            if chunk_idx == 0:
                states_curr = tl.zeros([BLOCK_HEADDIM_QK, BLOCK_HEADDIM_V], dtype=tl.float32)
            else:
                # Load state at the end of chunk (chunk_idx - 1), which is the state at the start of chunk chunk_idx
                states_ptr = states + states_offset + (chunk_idx - 1) * stride_states_nchunks + offs_dstates[:, None]*stride_states_qk_dim + offs_headdim[None, :] * stride_states_v_dim
                states_mask = (offs_dstates[:, None] < headdim_qk) & (offs_headdim[None, :] < headdim_v)
                states_curr = tl.load(states_ptr, mask = states_mask, other = 0.0)
                # states_curr = tl.load(states_ptr)

            ### Compute dM_Next, or rather, store grad_scale_intermediate,
            #  which is needed to compute dM_Next's contribution to ddt and dA:
            if OFFLOAD_GSI:
                # DQI = dO @ S^T: (CHUNK_SIZE, headdim_v) @ (headdim_v @ headdim_qk)
                acc_dq_intermediate = tl.dot(do_block, tl.trans(states_curr.to(do_block.dtype)))
                grad_scale = tl.sum(acc_dq_intermediate * q_block, axis=1)
                scale_m = tl.exp(da_cs)
                grad_scale_intermediate = grad_scale * scale_m

                # gsi stands for "grad_scale_intermediate", which is the intermediate quantity
                # that allows us to offload computation of dM_Next's contribution to dA and ddt.
                gsi_offset = pid_batch * stride_gsi_batch + pid_head * stride_gsi_nhead
                offs_gsi_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
                gsi_ptrs = GRAD_SCALE_INTERM + gsi_offset + offs_gsi_seq * stride_gsi_seqlen
                gsi_mask = offs_gsi_seq < seqlen
                tl.store(gsi_ptrs, grad_scale_intermediate, mask=gsi_mask)

            ### Compute dM_S:
            #da_cs_last = tl.sum(da)
            # Compute dM_S, and use it to compute dA_val and ddt
            dM_S = tl.sum(states_curr * acc_dstates) # Just a scalar
            M_S = tl.exp(da_cs_last)
            dA_val += tl.sum(dt_block * dM_S) * M_S
            ddt += a_val * dM_S * M_S #[None,] # Just a scalar, but should broadcast across chunksize

            # Store ddt:
            tl.store(ddt_ptrs, ddt)

        # ============================================================
        # Update gradient states for next (previous) chunk
        # dStates += Q^T @ dO: (headdim_qk, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_v)
        # ============================================================
        scale_m = tl.exp(da_cs)
        do_block_scaled = (do_block * scale_m[:, None]).to(q_block.dtype)

        acc_dstates = tl.exp(da_cs_last) * acc_dstates + tl.dot(tl.trans(q_block), do_block_scaled)

    if COMPUTE_SCALE_GRAD:
        # Store dA:
        tl.store(DA + pid_head * stride_da_nhead + pid_batch * stride_da_batch, dA_val)

    # Store dD:
    if DD is not None:
        tl.store(DD + pid_head * stride_dd_nhead + pid_batch * stride_dd_batch, dD_val)


## Readd the masking behavior

def mamba2_bwd_dk_dv_triton(
    q: torch.Tensor,  # (b, s, h_qk, d)
    k: torch.Tensor,  # (b, s, h_qk, d)
    v: torch.Tensor,  # (b, s, h, dv)
    do: torch.Tensor,  # (b, s, h, dv)
    dk: Optional[torch.Tensor] = None,  # (b, s, h_qk, d)
    dv: Optional[torch.Tensor] = None,  # (b, s, h, dv)
    a: Optional[torch.Tensor] = None, # (h,)
    dt: Optional[torch.Tensor] = None, # (b, s, h)
    dA: Optional[torch.Tensor] = None, # (h,)
    ddt: Optional[torch.Tensor] = None, # (b, s, h)
    D: Optional[torch.Tensor] = None, # (h,)
    dD: Optional[torch.Tensor] = None, # (h,)
    z: Optional[torch.Tensor] = None, # (b, s, h, dv)
    states: Optional[torch.Tensor] = None, # (b, h, c, hh_qk, dv)
    chunk_size: int = 64,
    offload_dda_csr: bool = True,
    offload_gsi: bool = True,
    offload_dm_curr: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of Mamba2 backward pass for computing gradients dK and dV.

    This function computes the gradients with respect to K and V by iterating backwards
    through sequence chunks and accumulating gradient states. The algorithm includes
    exponential decay scaling factors and optional gradient computations for A, dt, and D.

    Algorithm (iterate backwards through chunks):
        For each chunk from last to first:
            1. Compute dK[chunk] with exponential decay scaling
            2. Compute dV[chunk] with exponential decay scaling and D contribution
            3. Optionally compute dA and ddt gradients (when states provided)
            4. Update dStates for next (previous) chunk
            5. Optionally accumulate dD gradient
            6. Handle Z gating by scaling dO[chunk] with sigmoid(Z) before computations

    Args:
        q: Query tensor (batch, seqlen, nheads_qk, headdim_qk)
        k: Key tensor (batch, seqlen, nheads_qk, headdim_qk)
        v: Value tensor (batch, seqlen, nheads, headdim_v)
        do: Output gradient tensor (batch, seqlen, nheads, headdim_v)
        dk: Optional pre-allocated gradient for K (will be zero-initialized if None)
        dv: Optional pre-allocated gradient for V (will be allocated if None)
        a: Optional decay parameter (nheads,)
        dt: Optional time delta (batch, seqlen, nheads)
        dA: Optional pre-allocated gradient for A (allocated if None and states provided)
        ddt: Optional pre-allocated gradient for dt (allocated if None and states provided)
        D: Optional per-head skip connection weight (nheads,)
        dD: Optional pre-allocated gradient for D (allocated if None and D provided)
        z: Optional gating tensor (batch, seqlen, nheads, headdim_v)
        states: Optional saved states from forward pass (batch, nheads, nchunks, headdim_qk, headdim_v)
        chunk_size: Size of chunks to process (default: 64)

    Returns:
        - (dk, dv) when states and D are None
        - (dk, dv, dD) when D is provided but states are None
        - (dk, dv, dA, ddt) when states are provided but D is None
        - (dk, dv, dA, ddt, dD) when both states and D are provided
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    _, _, nheads, headdim_v = v.shape

    assert nheads % nheads_qk == 0

    return_scale_grad = (dA is None or ddt is None) and states is not None
    return_d_grad = (dD is None) and D is not None

    # Track if we need to reduce dK over head groups
    dk_needs_reduction = False
    if dk is None:
        if nheads_qk != nheads:
            # For GQA, allocate dK with nheads dimension to avoid atomics
            # Use empty instead of zeros since kernel writes to all locations
            dk = torch.empty(batch, seqlen, nheads, headdim_qk, dtype=k.dtype, device=k.device)
            dk_needs_reduction = True
        else:
            dk = torch.zeros_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    # Handle dA allocation
    # Note: dA cannot be accumulated in-place when passed from another kernel
    # because we removed atomic operations. Caller must add results manually.
    if dA is None and states is not None:
        dA = torch.zeros(nheads, batch, dtype=a.dtype, device=a.device)

    load_ddt = ddt is not None
    if ddt is None and states is not None:
        ddt = torch.zeros_like(dt)

    # Handle dD allocation
    # Note: dD cannot be accumulated in-place when passed from another kernel
    # because we removed atomic operations. Caller must add results manually.
    if dD is None and D is not None:
        dD = torch.zeros(nheads, batch, dtype=D.dtype, device=D.device)

    # assert q.is_cuda and k.is_cuda and v.is_cuda and do.is_cuda, "Tensors must be on CUDA"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only fp16/bf16 supported"

    # Round up head dims to multiples of 16 for efficient loading
    BLOCK_HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    BLOCK_HEADDIM_V = triton.next_power_of_2(headdim_v)

    if offload_dda_csr:
        ddA_cs_rev = torch.empty(batch, nheads, seqlen, device=a.device, dtype=torch.float32)
    else:
        ddA_cs_rev = None

    if offload_gsi:
        grad_scale = torch.empty(batch, nheads, seqlen, dtype=torch.float32, device=q.device)
    else:
        grad_scale = None

    if offload_dm_curr:
        nchunks = seqlen // chunk_size
        dM_curr = torch.empty(batch, nheads, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=q.device)
    else:
        dM_curr = None

    da_cs = torch.empty(batch, nheads, seqlen, device=a.device, dtype=torch.float32)

    # Grid: each program handles one (head, batch) pair and processes all chunks sequentially
    grid = (nheads, batch)

    mamba2_bwd_kernel_dk_dv[grid](
        q, k, v, do, dk, dv, a, dt, dA, ddt, D, dD, z, states, ddA_cs_rev, da_cs, grad_scale, dM_curr,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        a.stride(0) if a is not None else 0,
        *((dt.stride(0), dt.stride(2), dt.stride(1)) if dt is not None else (0, 0, 0)),
        *((dA.stride(0), dA.stride(1)) if dA is not None else (0, 0)),
        *((ddt.stride(0), ddt.stride(2), ddt.stride(1)) if ddt is not None else (0, 0, 0)),
        D.stride(0) if D is not None else 0,
        *((dD.stride(0), dD.stride(1)) if dD is not None else (0, 0)),
        *((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0)),
        *((states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4)) if states is not None else (0, 0, 0, 0, 0)),
        *((ddA_cs_rev.stride(0), ddA_cs_rev.stride(1), ddA_cs_rev.stride(2)) if ddA_cs_rev is not None else (0, 0, 0)),
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        *((grad_scale.stride(0), grad_scale.stride(1), grad_scale.stride(2)) if grad_scale is not None else (0, 0, 0)),
        *((dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2), dM_curr.stride(3), dM_curr.stride(4)) if dM_curr is not None else (0, 0, 0, 0, 0)),
        seqlen, headdim_qk, headdim_v, nheads_qk,
        CHUNK_SIZE=chunk_size,
        BLOCK_HEADDIM_QK=BLOCK_HEADDIM_QK,
        BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
        COMPUTE_SCALE_GRAD=states is not None,
        TIED_HEADS=nheads_qk != nheads,
        OFFLOAD_DDA_CSR = offload_dda_csr,
        OFFLOAD_GSI=offload_gsi,
        OFFLOAD_dM_Curr=offload_dm_curr,
        LOAD_DDT=load_ddt,
    )
    # Reduce dK over head groups if needed (for GQA)
    if dk_needs_reduction:
        # dk shape: (batch, seqlen, nheads, headdim_qk)
        # Reduce to: (batch, seqlen, nheads_qk, headdim_qk)
        dk_reduced = torch.empty(batch, seqlen, nheads_qk, headdim_qk, dtype=dk.dtype, device=dk.device)
        BLOCK_SIZE = 128  # Optimal for common headdim sizes (64, 128)
        grid = (batch * seqlen * nheads_qk * triton.cdiv(headdim_qk, BLOCK_SIZE),)
        reduce_head_groups_kernel[grid](
            dk, dk_reduced,
            batch, seqlen, nheads, nheads_qk, headdim_qk,
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dk_reduced.stride(0), dk_reduced.stride(1), dk_reduced.stride(2), dk_reduced.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        dk = dk_reduced

    # Sum dA and dD over batch dimension before returning
    if states is not None:
        dA_reduced = torch.empty(nheads, dtype=dA.dtype, device=dA.device)
        reduce_batch_dimension_kernel[(nheads,)](
            dA, dA_reduced,
            nheads, batch,
            dA.stride(0), dA.stride(1),
        )
        dA = dA_reduced
    if D is not None:
        dD_reduced = torch.empty(nheads, dtype=dD.dtype, device=dD.device)
        reduce_batch_dimension_kernel[(nheads,)](
            dD, dD_reduced,
            nheads, batch,
            dD.stride(0), dD.stride(1),
        )
        dD = dD_reduced

    # TODO (BC): instead of many nested branching, just have
    # a fix number of returned vars which are set to None
    # whenever appropriate.
    if return_scale_grad:
        if return_d_grad:
            return dk, dv, dA, ddt, dD, ddA_cs_rev, grad_scale, dM_curr
        else:
            return dk, dv, dA, ddt, ddA_cs_rev, grad_scale, dM_curr
    else:
        if return_d_grad:
            return dk, dv, dD, ddA_cs_rev, grad_scale, dM_curr
        else:
            return dk, dv, ddA_cs_rev, grad_scale, dM_curr


def segsum(x):
    """More stable segment sum calculation."""
    from einops import repeat
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba2_fwd_ref_iter(q, k, v, a, dt, z=None, D=None, chunksize=64, device='cuda', output_states=False):
    """Reference implementation of mamba2 forward using PyTorch,
    Using the iterative approach as in the Triton kernel.
    """
    from einops import rearrange, repeat
    batch, seqlen, nheads_qk, dstate = q.shape
    _, _, nheads, headdim = v.shape
    nchunks = seqlen // chunksize

    q = repeat(q, "b t i n -> b t (i g) n", g=nheads // nheads_qk)
    k = repeat(k, "b t i n -> b t (i g) n", g=nheads // nheads_qk)

    M_S = torch.zeros(dstate, headdim, device = device)
    M_prev = torch.zeros(chunksize, chunksize, device = device)
    M_next = torch.zeros(chunksize, headdim, device = device)
    M_curr = torch.zeros(chunksize, chunksize, device = device)

    output_state = torch.zeros(batch, nheads, nchunks, dstate, headdim, device = device)

    # acc_states = torch.zeros(dstate, headdim, device=device)
    acc_out = torch.zeros(batch, nheads, nchunks, chunksize, headdim, device=device)

    da = dt * a[None, :, None]  # (b, t, h)
    da = rearrange(da, "b h (c s) -> b h c s", c=nchunks)
    da_cs_all = torch.cumsum(da, dim=-1)

    dt_chunked = rearrange(dt, "b h (c s) -> b h c s", c=nchunks)
    k_chunked = rearrange(k, "b (c s) h n -> b c s h n", c=nchunks)
    v_chunked = rearrange(v, "b (c s) h n -> b c s h n", c=nchunks)
    q_chunked = rearrange(q, "b (c s) h n -> b c s h n", c=nchunks)

    for b in range(batch):
        for h in range(nheads):
            acc_states = torch.zeros(dstate, headdim, device=device)
            for chunk_idx in range(nchunks):
                da_cs = da_cs_all[b, h, chunk_idx, :] #
                da_cs_last = da_cs[-1]
                dt_block = dt_chunked[b, h, chunk_idx, :]

                M_S = torch.ones_like(M_S) * torch.exp(da_cs_last) #.detach().clone()
                M_prev = torch.diag(torch.exp(da_cs_last - da_cs) * dt_block) #.detach().clone()
                M_next = repeat(torch.exp(da_cs), "c -> c p", p=headdim) #.detach().clone()
                M_curr = torch.exp(segsum(da[b, h, chunk_idx, :])) * dt_block[None, :] #).detach().clone()

                C = q_chunked[b, chunk_idx, :, h, :] # (chunksize, dstate)
                B = k_chunked[b, chunk_idx, :, h, :] # (chunksize, dstate)
                x = v_chunked[b, chunk_idx, :, h, :] # (chunksize, headdim)

                O =  (C @ acc_states) * M_next + torch.tril(M_curr * (C @ B.T)) @ x #chunk_size, headdim
                acc_out[b, h, chunk_idx, :, :] = O

                acc_states = acc_states * M_S + B.T @ M_prev @ x

                output_state[b, h, chunk_idx, :, :] = acc_states
            # scale = dt * a[None, None, :] # (b, t, h)
            # scale = rearrange(scale, "b t h -> b h t")
            # scale = torch.exp(segsum(scale)) # (b, h, t, s)

            # qk = torch.einsum("bthd,bshd,bhts->bhts", q, k, scale)  # (b, h, t, s)
            # qk_causal = torch.tril(qk)
            # out = torch.einsum("bhts,bshd,bsh->bthd", qk_causal, v, dt)  # (b, t, h, d)

    if output_states:
        out_x = rearrange(acc_out, "b h c s d -> b (c s) h d")
        if D is not None:
            out_x = out_x + v * D[None, None, :, None]

        if z is not None:
            out = out_x * torch.sigmoid(z)
        else:
            out = out_x
        return out, out_x, output_state

    else:
        out_x = rearrange(acc_out, "b h c s d -> b (c s) h d")
        if D is not None:
            out_x = out_x + v * D[None, None, :, None]
        if z is not None:
            out = out_x * torch.sigmoid(z)
        else:
            out = out_x
        return out, out_x

# best config selected: num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w) #, maxnreg=r)
        for s in [2] #[1, 2, 3, 4] #[2]
        for w in [4] #[2, 4, 8] #[4] #[2, 4, 8]
        # for r in [256] #[128, 256]
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "BLOCK_HEADDIM_V"],
    # restore_value=["DA", "DDT"],
)
# forward scan
@triton.jit
def mamba2_bwd_dq_triton_kernel(
    dOut, V, K, dQ, DA, DDT, A, DT, DT_CUMSUM, Q, states, Z, dZ, Out_x, GRAD_SCALE_INTERM, dSSdA_OUT,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_dim,
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_dim,
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_dim,
    stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_dim,
    stride_da_nhead, stride_da_batch,
    stride_ddt_batch, stride_ddt_seqlen, stride_ddt_nhead,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_dt_cumsum_batch, stride_dt_cumsum_nhead, stride_dt_cumsum_seqlen,
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_dim,
    stride_states_batch, stride_states_nheads, stride_states_nchunks, stride_states_v_dim, stride_states_qk_dim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_dim,
    stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_dim,
    stride_out_x_batch, stride_out_x_seqlen, stride_out_x_head, stride_out_x_dim,
    stride_gsi_batch, stride_gsi_nhead, stride_gsi_seqlen,
    stride_dSS_batch, stride_dSS_head, stride_dSS_chunk, stride_dSS_i, stride_dSS_j,
    seqlen, headdim_qk, headdim_v, ngroups,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    COMPUTE_SCALE_GRAD: tl.constexpr,
    SAVE_STATES: tl.constexpr,
    TIED_HEADS: tl.constexpr,
    OFFLOAD_GSI: tl.constexpr,
    OFFLOAD_dM_Curr: tl.constexpr,
):
    """
    Mamba2 dQ backward kernel for Triton.

    This kernel computes dQ by performing a forward-style pass with remapped inputs:
    dQ computation is equivalent to the forward pass with q->dout, k->v, v->k.

    Note: "headdim_qk" is actually headdim ("P"),
          "headdim_v" is actually dstate ("N"), and
          "acc_states" has dim (P, N).

    Each program instance handles one entire sequence for one (batch, head) pair.

    Algorithm (iterate forward through chunks):
    For each chunk from first to last:
        1. Compute S = dOut[chunk] @ V[chunk]^T (with causal mask and exponential decay)
        2. Compute dQ[chunk]:
           - dQ = dOut[chunk] @ States_prev * scale_m + causal(S) @ K[chunk]
           - scale_m = exp(cumsum(dt * a))
        3. Optionally compute dA and ddt gradients (when COMPUTE_SCALE_GRAD=True):
           - Compute dM_Next from dOut @ States contribution
           - Compute dM_Curr from intra-chunk attention
           - Accumulate gradients for A and dt parameters
        4. Optionally compute dZ gradient (when Z is not None):
           - dZ = dOut * out_x * sigmoid(Z) * (1 - sigmoid(Z))
           - Scale dOut by sigmoid(Z) before other computations
        5. Update States for next chunk:
           - States = exp(da_cs_last) * States + V[chunk]^T @ (K[chunk] * scale)
           - scale = exp(cumsum_rev(dt * a)) * dt
        6. Optionally save states for use in dK/dV kernel
    """
    # Program ID: which head and batch
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    # Compute head index for Q/K (for GQA support)
    # nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (ngroups)

    # Load decay parameter for current head
    a_val = tl.load(A + pid_head)

    # Offset pointers to current batch and head
    dout_offset = pid_batch * stride_dout_batch + pid_head * stride_dout_head
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head
    # Use pid_head for dQ to avoid atomic operations in TIED_HEADS case
    dq_offset = pid_batch * stride_dq_batch + pid_head * stride_dq_head
    dt_offset = pid_batch * stride_dt_batch + pid_head * stride_dt_head
    states_offset = pid_batch * stride_states_batch + pid_head * stride_states_nheads
    z_offset = pid_batch * stride_z_batch + pid_head * stride_z_head

    if COMPUTE_SCALE_GRAD:
        ddt_offset = pid_batch * stride_ddt_batch + pid_head * stride_ddt_nhead
        q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head
        dA_val = 0.0

    dz_offset = pid_batch * stride_dz_batch + pid_head * stride_dz_head
    out_x_offset = pid_batch * stride_out_x_batch + pid_head * stride_out_x_head

    # Initialize cumulative states: States = sum(K[i]^T @ V[i])
    # Note: Variable names V, K, Q refer to the actual forward pass tensors (not remapped)
    # - V pointer -> forward pass v tensor
    # - K pointer -> forward pass k tensor
    # - Q pointer -> forward pass q tensor (only loaded for scale gradients)
    # - dOut pointer -> gradient of output (plays the role of Q in the computation)
    acc_states = tl.zeros([BLOCK_HEADDIM_QK, BLOCK_HEADDIM_V], dtype=tl.float32)

    # Compute number of chunks
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    # offs_dstates = tl.arange(0, BLOCK_HEADDIM_QK)
    # offs_headdim = tl.arange(0, BLOCK_HEADDIM_V)

    # Process each chunk sequentially
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE

        # ============================================================
        # Load dt values and compute cumulative decay for this chunk
        # ============================================================
        offs_dt_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        dt_ptrs = DT + dt_offset + offs_dt_seq * stride_dt_seqlen
        dt_mask = offs_dt_seq < seqlen
        dt_block = tl.load(dt_ptrs, mask=dt_mask, other=0.0)  # (CHUNK_SIZE,)
        # dt_block = tl.load(dt_ptrs)

        # ============================================================
        # Load V, K, Q for current chunk
        # (In backward: V=forward_v, K=forward_k, Q=forward_q for scale gradients)
        # ============================================================
        # Load V block (v from forward): (CHUNK_SIZE, headdim_qk)
        offs_v_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_v_dim = tl.arange(0, BLOCK_HEADDIM_QK)
        v_ptrs = V + v_offset + offs_v_seq[:, None] * stride_v_seqlen + offs_v_dim[None, :] * stride_v_dim
        v_mask = (offs_v_seq[:, None] < seqlen) & (offs_v_dim[None, :] < headdim_qk)
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)
        # v_block = tl.load(v_ptrs)

        # Load K block (k from forward): (CHUNK_SIZE, headdim_v)
        offs_k_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_k_dim = tl.arange(0, BLOCK_HEADDIM_V)
        k_ptrs = K + k_offset + offs_k_seq[:, None] * stride_k_seqlen + offs_k_dim[None, :] * stride_k_dim
        k_mask = (offs_k_seq[:, None] < seqlen) & (offs_k_dim[None, :] < headdim_v)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # k_block = tl.load(k_ptrs)


        if COMPUTE_SCALE_GRAD:
            # Load Q block (q from forward, only needed for scale gradients): (CHUNK_SIZE, headdim_v)
            q_ptrs = Q + q_offset + offs_k_seq[:, None] * stride_q_seqlen + offs_k_dim[None, :] * stride_q_dim
            q_block = tl.load(q_ptrs, mask=k_mask, other=0.0)
            # q_block = tl.load(q_ptrs)

        # ============================================================
        # Compute dQ for current chunk
        # ============================================================
        # Load dOut for current chunk: (CHUNK_SIZE, headdim_qk)
        # (dOut plays the role of Q in the forward pass)
        offs_dout_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_dout_dim = tl.arange(0, BLOCK_HEADDIM_QK)
        dout_ptrs = dOut + dout_offset + offs_dout_seq[:, None] * stride_dout_seqlen + offs_dout_dim[None, :] * stride_dout_dim
        dout_mask = (offs_dout_seq[:, None] < seqlen) & (offs_dout_dim[None, :] < headdim_qk)
        dout_block = tl.load(dout_ptrs, mask=dout_mask, other=0.0)

        # Save original dout for dz computation if needed
        #dout_block_original = dout_block if Z is not None else None

        if Z is not None:
            # Load Z block (Z has shape (batch, seqlen, nheads, headdim_qk))
            offs_z_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
            offs_z_dim = tl.arange(0, BLOCK_HEADDIM_QK)
            z_ptrs = Z + z_offset + offs_z_seq[:, None] * stride_z_seqlen + offs_z_dim[None, :] * stride_z_dim
            z_mask = (offs_z_seq[:, None] < seqlen) & (offs_z_dim[None, :] < headdim_qk)
            z_block = tl.load(z_ptrs, mask=z_mask, other=0.0)
            # z_block = tl.load(z_ptrs)

            # Load out_x block (out_x has shape (batch, seqlen, nheads, headdim_qk))
            offs_out_x_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
            offs_out_x_dim = tl.arange(0, BLOCK_HEADDIM_QK)
            out_x_ptrs = Out_x + out_x_offset + offs_out_x_seq[:, None] * stride_out_x_seqlen + offs_out_x_dim[None, :] * stride_out_x_dim
            out_x_mask = (offs_out_x_seq[:, None] < seqlen) & (offs_out_x_dim[None, :] < headdim_qk)
            out_x_block = tl.load(out_x_ptrs, mask=out_x_mask, other=0.0)
            # out_x_block = tl.load(out_x_ptrs)

            # Compute dZ gradient: dZ = dOut * out_x * sigmoid(Z) * (1 - sigmoid(Z))
            sigmoid_z = tl.sigmoid(z_block.to(tl.float32))
            dz_block = dout_block * out_x_block * sigmoid_z * (1.0 - sigmoid_z)

            # Store dZ
            dz_ptrs = dZ + dz_offset + offs_z_seq[:, None] * stride_dz_seqlen + offs_z_dim[None, :] * stride_dz_dim
            tl.store(dz_ptrs, dz_block.to(z_block.dtype), mask=z_mask)

            # Scale dOut by sigmoid(Z) for subsequent computations
            # Convert sigmoid_z back to original dtype to maintain consistency
            dout_block = dout_block * sigmoid_z.to(dout_block.dtype)



        if COMPUTE_SCALE_GRAD:
             # Compute inter-chunk contribution: dOut @ States
            # (CHUNK_SIZE, headdim_qk) @ (headdim_qk, headdim_v)
            # Uses accumulated states from all previous chunks
            acc_dq_intermediate = tl.dot(dout_block, acc_states.to(dout_block.dtype))

            # Compute log_decay = dt * a for each position in chunk
            # Only compute for valid positions within sequence length
            dA = tl.where(dt_mask, dt_block * a_val, 0.0)  # (CHUNK_SIZE,)
            dA_cs = tl.cumsum(dA)
            scale_m = tl.exp(dA_cs)

            grad_scale = tl.sum(acc_dq_intermediate * q_block, axis=1)
            grad_scale_intermediate = grad_scale * scale_m

            # gsi stands for "grad_scale_intermediate", which is the intermediate quantity
            # that allows us to offload computation of dM_Next's contribution to dA and ddt.
            # NOTE: We have the offloading code at this location on purpose; moving it
            # further down the code causes dramatic slowdown (621.70 GB/s -> 455.69 GB/s)
            gsi_offset = pid_batch * stride_gsi_batch + pid_head * stride_gsi_nhead
            offs_gsi_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
            gsi_ptrs = GRAD_SCALE_INTERM + gsi_offset + offs_gsi_seq * stride_gsi_seqlen
            gsi_mask = offs_gsi_seq < seqlen
            tl.store(gsi_ptrs, grad_scale_intermediate, mask=gsi_mask)

        # Compute intra-chunk attention score matrix: S = dOut @ V^T
        # (CHUNK_SIZE, headdim_qk) @ (headdim_qk, CHUNK_SIZE) -> (CHUNK_SIZE, CHUNK_SIZE)
        s_block_intermediate = tl.dot(dout_block, tl.trans(v_block))

        if COMPUTE_SCALE_GRAD:
            ddt_ptrs = DDT + ddt_offset + offs_dt_seq * stride_ddt_seqlen
            ddt_block = tl.load(ddt_ptrs)

        # Apply exponential decay scaling: exp(SegSum(dA))
        dA = dt_block * a_val  # (CHUNK_SIZE,)
        dA_cs = tl.cumsum(dA)
        scale_m = tl.exp(dA_cs)
        eSSdA = tl.exp(tl.minimum((dA_cs[:, None] - dA_cs[None, :]), 0.0))
        s_block_intermediate *= eSSdA

        # Apply causal mask (only attend to current and previous positions)
        offs_m_local = tl.arange(0, CHUNK_SIZE)
        offs_n_local = tl.arange(0, CHUNK_SIZE)
        causal_mask = offs_m_local[:, None] >= offs_n_local[None, :]
        s_block_intermediate = tl.where(causal_mask, s_block_intermediate, 0.0)

        # Apply dt scaling
        s_block = s_block_intermediate * dt_block[None, :]

        acc_dq = tl.dot(s_block.to(k_block.dtype), k_block)

        # Compute inter-chunk contribution: dOut @ States
        # (CHUNK_SIZE, headdim_qk) @ (headdim_qk, headdim_v)
        # Uses accumulated states from all previous chunks
        acc_dq_intermediate = tl.dot(dout_block, acc_states.to(dout_block.dtype))

        # Compute dQ = inter-chunk contribution + intra-chunk contribution
        # dQ = (dOut @ States) * scale_m + causal(S) @ K
        # (CHUNK_SIZE, headdim_v) = scaled inter-chunk + (CHUNK_SIZE, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_v)
        acc_dq += acc_dq_intermediate * scale_m[:, None]

        if COMPUTE_SCALE_GRAD:
            # ============================================================
            # Compute da and ddt for current chunk
            # ============================================================
            ### Compute dM_Next:
            if OFFLOAD_GSI:
                ddt = ddt_block
            else:
                dt_cumsum = tl.cumsum(dt_block)
                da_contribution = tl.sum(dt_cumsum * grad_scale_intermediate)
                dA_val += da_contribution
                ddt = ddt_block + tl.cumsum(grad_scale_intermediate * a_val, reverse=True)

            ### Compute dM_Curr:
            # Compute QK^T = Q @ K^T: (CHUNK_SIZE, headdim_v) @ (headdim_v, CHUNK_SIZE)
            QKt = tl.dot(q_block, tl.trans(k_block))
            dM_Curr = QKt * s_block_intermediate # This is actually dM_Curr * L * exp(SegSum(dA))

            if OFFLOAD_dM_Curr:
                i = tl.arange(0, CHUNK_SIZE)
                j = tl.arange(0, CHUNK_SIZE)

                dSS_ptrs = (
                    dSSdA_OUT
                    + pid_batch * stride_dSS_batch
                    + pid_head  * stride_dSS_head
                    + chunk_idx * stride_dSS_chunk
                    + i[:, None] * stride_dSS_i
                    + j[None, :] * stride_dSS_j
                )

                tl.store(dSS_ptrs, dM_Curr)
            else:
                ddt += tl.sum(dM_Curr, axis=0)
                dSSdA = dM_Curr * dt_block[None, :]

                #The following is a more numerically stable version of: d_dA_cs = tl.sum(dSSdA, axis=1) - tl.sum(dSSdA, axis=0); ddA = tl.cumsum(d_dA_cs, reverse=True)
                #Backward through segsum: reverse column cumsum -> mask -> row sum
                offs_k = tl.arange(0, CHUNK_SIZE)
                SS_mask = (offs_k[:, None] >= offs_k[None, :] + 1)
                ddA = tl.where(SS_mask, dSSdA, 0.0)              # Apply mask (to mask out the diagonal, whose gradient does not flow back)
                ddA = tl.cumsum(ddA, axis=0, reverse=True)  # Reverse column cumsum (transpose of fwd cumsum)
                ddA = tl.where(SS_mask, ddA, 0.0)              # Apply mask
                ddA = tl.sum(ddA, axis=1)                      # Row sum (transpose of fwd expand)

                ddt += a_val * ddA
                dA_val += tl.sum(ddA * dt_block)

                #Store ddt:
                tl.store(ddt_ptrs, ddt)

        # ============================================================
        # Store dQ output (now using pid_head instead of head_idx_qk in dq_offset)
        # ============================================================
        offs_dq_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_dq_dim = tl.arange(0, BLOCK_HEADDIM_V)
        dq_ptrs = dQ + dq_offset + offs_dq_seq[:, None] * stride_dq_seqlen + offs_dq_dim[None, :] * stride_dq_dim
        dq_mask = (offs_dq_seq[:, None] < seqlen) & (offs_dq_dim[None, :] < headdim_v)
        tl.store(dq_ptrs, acc_dq.to(k_block.dtype), mask=dq_mask)

        # ============================================================
        # Update states for next chunk
        # ============================================================
        # In the backward dQ pass, we accumulate states in forward order:
        # States = exp(da_cs_last) * States + V^T @ (K * scale)
        # where V=v_fwd, K=k_fwd (not remapped), and scale handles exponential decay
        dA_cs_last = tl.sum(dA)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs), 0.0)) * dt_block
        v_block *= scale[:,None]

        # Update accumulated states: (headdim_qk, headdim_v) = (headdim_qk, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_v)
        acc_states = tl.exp(dA_cs_last) * acc_states + tl.dot(tl.trans(v_block).to(k_block.dtype), k_block)

        if SAVE_STATES:
            states_ptrs = states + states_offset + offs_v_dim[:, None]*stride_states_qk_dim + offs_k_dim[None, :] * stride_states_v_dim
            tl.store(states_ptrs, acc_states)
            states_offset += stride_states_nchunks

    if not OFFLOAD_dM_Curr:
        if COMPUTE_SCALE_GRAD:
            # Store da:
            tl.store(DA + pid_head * stride_da_nhead + pid_batch * stride_da_batch, dA_val)

def mamba2_bwd_dq_triton(
    dout: torch.Tensor,  # (b, s, h, dv)
    v: torch.Tensor,  # (b, s, h, dv)
    k: torch.Tensor,  # (b, s, h_qk, d)
    a: torch.Tensor,  # (h,)
    dt: torch.Tensor,  # (b, s, h)
    dt_cumsum: Optional[torch.Tensor] = None,  # (b, h, s)
    dq: Optional[torch.Tensor] = None,  # (b, s, h, d)
    dA: Optional[torch.Tensor] = None, # (h,)
    ddt: Optional[torch.Tensor] = None, # (b, s, h)
    q: Optional[torch.Tensor] = None, # (b, s, h, d)
    z: Optional[torch.Tensor] = None, # (b, s, h, dv)
    dz: Optional[torch.Tensor] = None, # (b, s, h, dv)
    out_x: Optional[torch.Tensor] = None, # (b, s, h, dv)
    chunk_size: int = 64,
    save_states: bool = False,
    offload_gsi: bool = True,
    offload_dm_curr: bool = True,
) -> torch.Tensor:
    """
    Triton implementation of Mamba2 backward pass for computing gradient dQ.

    This function computes the gradient with respect to Q by performing a forward-style
    pass with remapped inputs. Conceptually, computing dQ is equivalent to running
    the forward pass with the remapping: q -> dout, k -> v, v -> k.

    The algorithm includes exponential decay scaling factors and optional gradient
    computations for A, dt, and z (gating parameter).

    Algorithm (iterate forward through chunks):
        For each chunk from first to last:
            1. Optionally compute dZ gradient and scale dOut by sigmoid(Z)
            2. Compute S = dOut[chunk] @ V[chunk]^T with causal mask and exponential decay
            3. Compute dQ[chunk] = dOut[chunk] @ States_prev * scale_m + causal(S) @ K[chunk]
            4. Optionally compute dA and ddt gradients from dM_Next and dM_Curr
            5. Update States for next chunk with exponential decay scaling
            6. Optionally save states for use in dK/dV backward kernel

    Note: "headdim_qk" is actually headdim ("P"), and
          "headdim_v" is actually dstate ("N").

    Args:
        dout: Output gradient tensor (batch, seqlen, nheads, headdim_qk)
        v: Value tensor from forward (batch, seqlen, nheads, headdim_qk)
        k: Key tensor from forward (batch, seqlen, nheads_qk, headdim_v)
        a: Decay parameter (nheads,)
        dt: Time delta (batch, seqlen, nheads)
        dq: Optional pre-allocated gradient for Q (allocated if None)
        dA: Optional pre-allocated gradient for A (allocated if None and q provided)
        ddt: Optional pre-allocated gradient for dt (allocated if None and q provided)
        q: Optional query tensor for computing scale gradients (batch, seqlen, nheads_qk, headdim_v)
        z: Optional gating tensor (batch, seqlen, nheads, headdim_qk)
        dz: Optional pre-allocated gradient for z (allocated if None and z/out_x provided)
        out_x: Optional output before gating from forward pass (batch, seqlen, nheads, headdim_qk)
        chunk_size: Size of chunks to process (default: 64)
        save_states: Whether to save intermediate states for dK/dV kernel (default: False)

    Returns:
        Depending on which optional parameters are provided:
        - dq (minimal case)
        - (dq, states) if save_states=True
        - (dq, dA, ddt) if q is provided
        - (dq, dz) if z and out_x are provided
        - (dq, dA, ddt, dz) if both q and z/out_x are provided
        - (dq, dA, ddt, states) if q is provided and save_states=True
        - (dq, dz, states) if z/out_x are provided and save_states=True
        - (dq, dA, ddt, dz, states) if all optional gradients requested and save_states=True
    """
    batch, seqlen, nheads, headdim_qk = dout.shape
    _, _, nheads_qk, headdim_v = k.shape

    return_scale_grad = (dA is None or ddt is None) and q is not None
    return_z_grad = (dz is None) and z is not None and out_x is not None

    ngroups = nheads // nheads_qk
    # breakpoint()

    # Track if we need to reduce dQ over head groups
    dq_needs_reduction = False
    if dq is None:
        if ngroups > 1:
            # For GQA, allocate dQ with nheads dimension to avoid atomics
            # Use empty instead of zeros since kernel writes to all locations
            dq = torch.empty(batch, seqlen, nheads, headdim_v, dtype=k.dtype, device=k.device)
            dq_needs_reduction = True
        else:
            dq = torch.zeros_like(k)
    if dA is None and q is not None:
        dA = torch.zeros(nheads, batch, dtype=a.dtype, device=a.device)
    if ddt is None and q is not None:
        ddt = torch.zeros_like(dt)
    if dz is None and z is not None and out_x is not None:
        dz = torch.empty_like(z)

    if save_states:
        nchunks = seqlen // chunk_size
        states = torch.empty(batch, nheads, nchunks, headdim_v, headdim_qk,
                             dtype=torch.float32, device='cuda')
    else:
        states = None

    # assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only fp16/bf16 supported"

    if offload_gsi:
        grad_scale = torch.empty(batch, nheads, seqlen, dtype=torch.float32, device=q.device)
    else:
        grad_scale = None

    if offload_dm_curr:
        dM_curr = torch.empty(batch, nheads, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=q.device)
    else:
        dM_curr = None

    # Round up head dims to multiples of 16 for efficient loading
    BLOCK_HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    BLOCK_HEADDIM_V = triton.next_power_of_2(headdim_v)

    # Grid: each program handles one (head, batch) pair and processes all chunks sequentially
    grid = (nheads, batch)

    mamba2_bwd_dq_triton_kernel[grid](
        dout, v, k, dq, dA, ddt, a, dt, dt_cumsum, q, states, z, dz, out_x, grad_scale, dM_curr,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        *((dA.stride(0), dA.stride(1)) if dA is not None else (0, 0)),
        *((ddt.stride(0), ddt.stride(2), ddt.stride(1)) if ddt is not None else (0, 0, 0)),
        dt.stride(0), dt.stride(2), dt.stride(1),
        *((dt_cumsum.stride(0), dt_cumsum.stride(1), dt_cumsum.stride(2)) if dt_cumsum is not None else (0, 0, 0)),
        *((q.stride(0), q.stride(1), q.stride(2), q.stride(3)) if q is not None else (0, 0, 0, 0)),
        *((states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4)) if save_states else (0, 0, 0, 0, 0)),
        *((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0)),
        *((dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3)) if dz is not None else (0, 0, 0, 0)),
        *((out_x.stride(0), out_x.stride(1), out_x.stride(2), out_x.stride(3)) if out_x is not None else (0, 0, 0, 0)),
        *((grad_scale.stride(0), grad_scale.stride(1), grad_scale.stride(2)) if grad_scale is not None else (0, 0, 0)),
        *((dM_curr.stride(0), dM_curr.stride(1), dM_curr.stride(2), dM_curr.stride(3), dM_curr.stride(4)) if dM_curr is not None else (0, 0, 0, 0, 0)) ,
        seqlen, headdim_qk, headdim_v, ngroups,
        CHUNK_SIZE=chunk_size,
        BLOCK_HEADDIM_QK=BLOCK_HEADDIM_QK,
        BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
        COMPUTE_SCALE_GRAD=q is not None,
        SAVE_STATES=save_states,
        TIED_HEADS=ngroups != 1,
        OFFLOAD_GSI=offload_gsi,
        OFFLOAD_dM_Curr=offload_dm_curr,
    )

    # breakpoint()

    # Reduce dQ over head groups if needed (for GQA)
    if dq_needs_reduction:
        # dq shape: (batch, seqlen, nheads, headdim_v)
        # Reduce to: (batch, seqlen, nheads_qk, headdim_v)
        dq_reduced = torch.empty(batch, seqlen, nheads_qk, headdim_v, dtype=dq.dtype, device=dq.device)
        BLOCK_SIZE = 128  # Optimal for common headdim sizes (64, 128)
        grid = (batch * seqlen * nheads_qk * triton.cdiv(headdim_v, BLOCK_SIZE),)
        reduce_head_groups_kernel[grid](
            dq, dq_reduced,
            batch, seqlen, nheads, nheads_qk, headdim_v,
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dq_reduced.stride(0), dq_reduced.stride(1), dq_reduced.stride(2), dq_reduced.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        dq = dq_reduced

    # Reduce dA over batch dimension if needed
    if return_scale_grad:
        dA_reduced = torch.empty(nheads, dtype=dA.dtype, device=dA.device)
        reduce_batch_dimension_kernel[(nheads,)](
            dA, dA_reduced,
            nheads, batch,
            dA.stride(0), dA.stride(1),
        )
        dA = dA_reduced

    # TODO (BC): instead of many nested branching, just have
    # a fix number of returned vars which are set to None
    # whenever appropriate.
    if save_states:
        if return_scale_grad:
            if return_z_grad:
                return dq, dA, ddt, dz, states, grad_scale, dM_curr
            else:
                return dq, dA, ddt, states, grad_scale, dM_curr
        elif return_z_grad:
            return dq, dz, states, grad_scale, dM_curr
        else:
            return dq, states, grad_scale, dM_curr
    else:
        if return_scale_grad:
            if return_z_grad:
                return dq, dA, ddt, dz, grad_scale, dM_curr
            else:
                return dq, dA, ddt, grad_scale, dM_curr
        elif return_z_grad:
            return dq, dz, grad_scale, dM_curr
        else:
            return dq, grad_scale, dM_curr


def test_mamba2_bwd(
    batch=16,
    seqlen=2048,
    nheads=32,
    nheads_qk=32,
    headdim_qk=64,
    headdim_v=64,
    dtype=torch.bfloat16,
    scale_dtype=torch.float32,
    device="cuda",
    test_triton=True,
    chunk_size=64,
):
    assert nheads % nheads_qk == 0
    # assert nheads == nheads_qk, "Only support nheads == nheads_qk for now"
    import torch.nn.functional as F

    print("="*80)
    print(f"Config: batch={batch}, seqlen={seqlen}, nheads={nheads}, nheads_qk={nheads_qk}")
    print(f"        headdim_qk={headdim_qk}, headdim_v={headdim_v}, chunk_size={chunk_size}")
    print(f"        dtype={dtype}, scale_dtype={scale_dtype}")
    print("="*80)

    # Create input tensors
    a = -torch.empty(nheads, device=device, dtype=scale_dtype).uniform_(1.0, 16.0)
    q = torch.randn(batch, seqlen, nheads_qk, headdim_qk, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads_qk, headdim_qk, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads, headdim_v, dtype=dtype, device=device, requires_grad=True)

    dd_dt = torch.randn((batch, nheads, seqlen), dtype=scale_dtype, device=device) * 0.2 + 0.5
    dt_bias = torch.randn((nheads,), dtype=scale_dtype, device=device) * 0.5 - 5

    dt = F.softplus(dd_dt + dt_bias[:, None])

    # For debugging:
    # dt = torch.ones((batch, seqlen, nheads), dtype=scale_dtype, device=device)
    # a = torch.zeros(nheads, device=device, dtype=scale_dtype)

    # Create D and Z parameters for testing (optional features)
    # D: per-head skip connection weight (nheads,)
    # Z: gating values (batch, seqlen, nheads, headdim_v)
    d = torch.randn((nheads,), dtype=scale_dtype, device=device) * 0.1
    z = torch.randn((batch, seqlen, nheads, headdim_v), dtype=dtype, device=device)

    # Setup inputs with gradient tracking for reference
    q_ref_iter = q.detach().float().requires_grad_()
    k_ref_iter = k.detach().float().requires_grad_()
    v_ref_iter = v.detach().float().requires_grad_()
    a_ref_iter = a.float().requires_grad_()
    dt_ref_iter = dt.float().requires_grad_()
    D_ref_iter = d.float().requires_grad_()
    z_ref_iter = z.float().requires_grad_()

    # Forward pass with reference implementation (base attention without D and Z)
    out_ref_iter, out_ref_x_iter, states_ref = mamba2_fwd_ref_iter(q_ref_iter, k_ref_iter, v_ref_iter, a_ref_iter, dt_ref_iter, D=D_ref_iter, z=z_ref_iter, output_states=True)

    # Random gradient for output
    dout = torch.randn(batch, seqlen, nheads, headdim_v, dtype=dtype, device=device)
    dout_ref_iter = dout.float()

    # Compute all gradients using autograd
    grads = torch.autograd.grad(
        outputs=out_ref_iter,
        inputs=(q_ref_iter, k_ref_iter, v_ref_iter, a_ref_iter, dt_ref_iter, D_ref_iter, z_ref_iter),
        grad_outputs=dout_ref_iter,
        retain_graph=False,
        allow_unused=True,
    )
    grad_q_ref, grad_k_ref, grad_v_ref, grad_a_ref, grad_dt_ref, grad_D_ref, grad_z_ref = grads

    # Handle None gradients by creating zero tensors
    if grad_a_ref is None:
        grad_a_ref = torch.zeros_like(a_ref_iter)
    if grad_dt_ref is None:
        grad_dt_ref = torch.zeros_like(dt_ref_iter)
    if grad_D_ref is None:
        grad_D_ref = torch.zeros_like(D_ref_iter)
    if grad_z_ref is None:
        grad_z_ref = torch.zeros_like(z_ref_iter)

    # Allocate gradient tensors
    grad_D_triton = torch.zeros_like(d)
    grad_z_triton = torch.zeros_like(z)

   #dt_cumsum, _ = chunk_cumsum_triton(dt, chunk_size=chunk_size)

    dout1 = dout.clone()
    out_x1 = out_ref_x_iter.clone()

    grad_q_triton, grad_z_triton, states, _, _ = mamba2_bwd_dq_triton(
        dout=dout, v=v, k=k, a=a, dt=dt, dt_cumsum=None, chunk_size=chunk_size, z=z, q=None,
        dA=None, ddt=None, out_x = out_ref_x_iter, save_states=True, offload_gsi=False, offload_dm_curr=False
    )
    grad_k_triton, grad_v_triton, grad_a_dkdv_triton, grad_dt_dkdv_triton, grad_D_triton, ddA_cs_rev, grad_scale, dM_Curr  = mamba2_bwd_dk_dv_triton(
        q, k, v, dout, a=a, dt=dt, z=z, D=d,
        states=states, chunk_size=chunk_size, ddt=None, dA=None,
        offload_dda_csr=True, offload_gsi=True, offload_dm_curr=True,
    )
    grad_dt_dkdv_triton, grad_a_dkdv_triton = dA_ddt_from_dM_Next(grad_dt_dkdv_triton, dt, grad_scale, grad_a_dkdv_triton, a)
    grad_dt_dkdv_triton, grad_a_dkdv_triton = dA_ddt_from_dM_Curr(dM_Curr, dt, grad_dt_dkdv_triton, a, grad_a_dkdv_triton, chunk_size=chunk_size)
    grad_dt_dkdv_triton, grad_a_dkdv_triton = dA_ddt_from_dM_Prev(grad_dt_dkdv_triton, dt, ddA_cs_rev, grad_a_dkdv_triton, a)
    #Accumulate dA gradients from both kernels
    grad_a_triton = grad_a_dkdv_triton

    print("\n=== Testing Triton saved states from dQ Kernel ===")
    print(f"Triton vs Ref for states, max diff = {(states - states_ref).abs().max().item():.6f}, mean diff = {(states - states_ref).abs().mean().item():.6f}")

    print("\n=== Testing Triton Backward Implementation ===")
    print(f"Triton vs Ref for dQ, max diff = {(grad_q_triton - grad_q_ref).abs().max().item():.6f}, mean diff = {(grad_q_triton - grad_q_ref).abs().mean().item():.6f}")
    print(f"Triton vs Ref for dK, max diff = {(grad_k_triton - grad_k_ref).abs().max().item():.6f}, mean diff = {(grad_k_triton - grad_k_ref).abs().mean().item():.6f}")
    print(f"Triton vs Ref for dV, max diff = {(grad_v_triton - grad_v_ref).abs().max().item():.6f}, mean diff = {(grad_v_triton - grad_v_ref).abs().mean().item():.6f}")
    print(f"Triton vs Ref for dA, max diff = {(grad_a_triton - grad_a_ref).abs().max().item():.6f}, mean diff = {(grad_a_triton - grad_a_ref).abs().mean().item():.6f}")
    print(f"Triton vs Ref for ddt, max diff = {(grad_dt_dkdv_triton - grad_dt_ref).abs().max().item():.6f}, mean diff = {(grad_dt_dkdv_triton - grad_dt_ref).abs().mean().item():.6f}")
    print(f"Triton vs Ref for dD, max diff = {(grad_D_triton - grad_D_ref).abs().max().item():.6f}, mean diff = {(grad_D_triton - grad_D_ref).abs().mean().item():.6f}")
    print(f"Triton vs Ref for dZ, max diff = {(grad_z_triton - grad_z_ref).abs().max().item():.6f}, mean diff = {(grad_z_triton - grad_z_ref).abs().mean().item():.6f}")

   #Print values around max error locations for qualitative comparison
    print("\n=== Values around max errors ===")

    # dQ
    idx_dq = torch.unravel_index(torch.argmax((grad_q_ref - grad_q_triton).abs()), shape=grad_q_triton.shape)
    b, s, h, d = idx_dq
    s_start, s_end = max(0, s-3), min(seqlen, s+4)
    print(f"dQ max error at [{b},{s},{h},{d}] (positions {s_start}:{s_end})")
    print(f"  Triton: {grad_q_triton[b, s_start:s_end, h, d].tolist()}")
    print(f"  Ref:    {grad_q_ref[b, s_start:s_end, h, d].tolist()}")

    # dK
    idx_dk = torch.unravel_index(torch.argmax((grad_k_ref - grad_k_triton).abs()), shape=grad_k_triton.shape)
    b, s, h, d = idx_dk
    s_start, s_end = max(0, s-3), min(seqlen, s+4)
    print(f"dK max error at [{b},{s},{h},{d}] (positions {s_start}:{s_end})")
    print(f"  Triton: {grad_k_triton[b, s_start:s_end, h, d].tolist()}")
    print(f"  Ref:    {grad_k_ref[b, s_start:s_end, h, d].tolist()}")

    # dV
    idx_dv = torch.unravel_index(torch.argmax((grad_v_ref - grad_v_triton).abs()), shape=grad_v_triton.shape)
    b, s, h, d = idx_dv
    s_start, s_end = max(0, s-3), min(seqlen, s+4)
    print(f"dV max error at [{b},{s},{h},{d}] (positions {s_start}:{s_end})")
    print(f"  Triton: {grad_v_triton[b, s_start:s_end, h, d].tolist()}")
    print(f"  Ref:    {grad_v_ref[b, s_start:s_end, h, d].tolist()}")

    # dA
    idx_da = torch.argmax((grad_a_ref - grad_a_triton).abs())
    h_start, h_end = max(0, idx_da-3), min(nheads, idx_da+4)
    print(f"dA max error at head {idx_da} (heads {h_start}:{h_end})")
    print(f"  Triton: {grad_a_triton[h_start:h_end].tolist()}")
    print(f"  Ref:    {grad_a_ref[h_start:h_end].tolist()}")

    # ddt
    idx_ddt = torch.unravel_index(torch.argmax((grad_dt_ref - grad_dt_dkdv_triton).abs()), shape=grad_dt_dkdv_triton.shape)
    b, s, h = idx_ddt
    s_start, s_end = max(0, s-3), min(seqlen, s+4)
    print(f"ddt max error at [{b},{s},{h}] (positions {s_start}:{s_end})")
    print(f"  Triton: {grad_dt_dkdv_triton[b, s_start:s_end, h].tolist()}")
    print(f"  Ref:    {grad_dt_ref[b, s_start:s_end, h].tolist()}")

    # dD
    idx_dd = torch.argmax((grad_D_ref - grad_D_triton).abs())
    h_start, h_end = max(0, idx_dd-3), min(nheads, idx_dd+4)
    print(f"dD max error at head {idx_dd} (heads {h_start}:{h_end})")
    print(f"  Triton: {grad_D_triton[h_start:h_end].tolist()}")
    print(f"  Ref:    {grad_D_ref[h_start:h_end].tolist()}")

    # dZ
    idx_dz = torch.unravel_index(torch.argmax((grad_z_ref - grad_z_triton).abs()), shape=grad_z_triton.shape)
    b, s, h, d = idx_dz
    s_start, s_end = max(0, s-3), min(seqlen, s+4)
    print(f"dZ max error at [{b},{s},{h},{d}] (positions {s_start}:{s_end})")
    print(f"  Triton: {grad_z_triton[b, s_start:s_end, h, d].tolist()}")
    print(f"  Ref:    {grad_z_ref[b, s_start:s_end, h, d].tolist()}")

    # breakpoint()


def benchmark_speed(
    batch=1,
    local_seqlen=8192,
    nheads=128,
    nheads_qk=8,
    headdim=64,
    dstate=128,
    dtype=torch.bfloat16,
    device="cuda",
    chunk_size=64,
    warmup_ms=25,
    rep_ms=100,
    seed=0,
):
    """Benchmark the four production true-CP backward kernels in isolation.

    ``local_seqlen`` is the public per-rank sequence length. True CP splits it
    into two independent recurrence segments, so every raw kernel receives
    ``seqlen=local_seqlen // 2`` with ``folded_batch=2 * batch``.

    The logical traffic model counts each semantic tensor region once. The
    source traffic model counts explicit masked ``tl.load``/``tl.store``
    elements, including GQA replay, tiled replay, and read-modify-write
    accumulators. Neither model is a hardware DRAM-byte counter.
    """

    from triton.testing import do_bench

    if batch <= 0 or local_seqlen <= 0 or nheads <= 0 or nheads_qk <= 0:
        raise ValueError("batch, sequence length, and head counts must be positive")
    if local_seqlen % 2 != 0:
        raise ValueError("true-CP local sequence length must be even")
    if nheads % nheads_qk != 0:
        raise ValueError("nheads must be divisible by nheads_qk")
    if headdim <= 0 or dstate <= 0:
        raise ValueError("headdim and dstate must be positive")
    if chunk_size <= 0 or chunk_size > 64:
        raise ValueError(
            "this benchmark targets the named whole-chunk kernels and requires chunk_size <= 64"
        )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if warmup_ms < 0 or rep_ms <= 0:
        raise ValueError("warmup_ms must be nonnegative and rep_ms must be positive")

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the true-CP kernel benchmark requires a CUDA device")
    device_properties = torch.cuda.get_device_properties(device)
    if device_properties.memory_clock_rate <= 0 or device_properties.memory_bus_width <= 0:
        raise RuntimeError("CUDA did not report a usable memory clock and bus width")
    # CUDA reports the maximum memory clock in kHz and the bus width in bits.
    # HBM/GDDR transfers on both clock edges, hence the factor of two.
    property_peak_gmem_gbps = (
        2.0
        * device_properties.memory_clock_rate
        * 1_000.0
        * (device_properties.memory_bus_width / 8.0)
        / 1e9
    )
    if "B300" in device_properties.name:
        peak_gmem_gbps = 8_000.0
        peak_gmem_source = "B300 specification"
    else:
        peak_gmem_gbps = property_peak_gmem_gbps
        peak_gmem_source = "CUDA max-clock/bus-width estimate"

    torch.manual_seed(seed)
    streams = 2
    seqlen = local_seqlen // streams
    folded_batch = streams * batch
    nchunks = triton.cdiv(seqlen, chunk_size)
    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    state_shape = (folded_batch, nheads, nchunks, headdim, dstate)
    boundary_shape = (streams, batch, nheads, headdim, dstate)
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_p = max(16, triton.next_power_of_2(headdim))
    block_n = max(16, triton.next_power_of_2(dstate))
    main_grid = (nheads, folded_batch)
    decay_grid = (nheads, folded_batch, nchunks)

    def _randn(shape, tensor_dtype=dtype, scale=0.02):
        tensor = torch.empty(shape, device=device, dtype=tensor_dtype)
        tensor.normal_(mean=0.0, std=scale)
        return tensor

    # Production layouts: C/B/x are folded, while dout/z and boundaries retain
    # their no-copy stream-major layout.
    C = _randn((folded_batch, seqlen, nheads_qk, dstate))
    B = _randn((folded_batch, seqlen, nheads_qk, dstate))
    x = _randn((folded_batch, seqlen, nheads, headdim))
    dout = _randn((streams, seqlen, batch, nheads, headdim))
    z = _randn((streams, seqlen, batch, nheads, headdim))
    initial_states = _randn(boundary_shape, torch.float32)
    final_dstates = _randn(boundary_shape, torch.float32)
    A = -torch.empty(nheads, device=device, dtype=torch.float32).uniform_(1.0, 16.0)
    D = _randn((nheads,), torch.float32)

    dt = torch.empty(decay_shape, device=device, dtype=torch.float32).uniform_(0.001, 0.01)
    valid_decay = torch.arange(
        nchunks * chunk_size, device=device
    ).reshape(1, 1, nchunks, chunk_size) < seqlen
    dt.masked_fill_(~valid_decay, 0.0)
    dA_cs = torch.cumsum(dt * A[None, :, None, None], dim=-1)
    dA_cs_rev = dA_cs[..., -1, None] - dA_cs

    dC_partial = torch.empty(
        folded_batch, seqlen, nheads, dstate, device=device, dtype=dtype
    )
    dz = torch.empty_like(dout)
    chunk_states = torch.empty(state_shape, device=device, dtype=torch.float32)

    processed_current = torch.empty(decay_shape, device=device, dtype=torch.float32)
    dA_current = torch.empty(
        folded_batch, nheads, nchunks, device=device, dtype=torch.float32
    )
    grad_scale_intermediate = torch.empty(
        decay_shape, device=device, dtype=torch.float32
    )
    processed_next = torch.zeros(decay_shape, device=device, dtype=torch.float32)
    dA_next = torch.zeros(
        folded_batch, nheads, nchunks, device=device, dtype=torch.float32
    )

    dB_partial = torch.empty(
        folded_batch, seqlen, nheads, dstate, device=device, dtype=dtype
    )
    dx = torch.empty_like(x)
    dD_partial = torch.empty(nheads, folded_batch, device=device, dtype=torch.float32)
    processed_dkdv = torch.zeros(decay_shape, device=device, dtype=torch.float32)
    dA_dkdv = torch.zeros(
        folded_batch, nheads, nchunks, device=device, dtype=torch.float32
    )
    ddA_cs_rev_intermediate = torch.empty(
        decay_shape, device=device, dtype=torch.float32
    )

    dq_args = (
        C, B, x, dout, z, dt, dA_cs, dA_cs_rev, initial_states, D,
        dC_partial, dz, chunk_states,
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1),
        dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        initial_states.stride(0), initial_states.stride(1),
        initial_states.stride(2), initial_states.stride(3), initial_states.stride(4),
        D.stride(0),
        dC_partial.stride(0), dC_partial.stride(1),
        dC_partial.stride(2), dC_partial.stride(3),
        dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3), dz.stride(4),
        chunk_states.stride(0), chunk_states.stride(1),
        chunk_states.stride(2), chunk_states.stride(3), chunk_states.stride(4),
        seqlen, headdim, dstate, nheads_qk,
    )

    def launch_dq():
        _mamba2_bwd_dq_cp_kernel[main_grid](
            *dq_args,
            A,
            processed_current,
            dA_current,
            grad_scale_intermediate,
            A.stride(0),
            processed_current.stride(0),
            processed_current.stride(1),
            processed_current.stride(2),
            processed_current.stride(3),
            dA_current.stride(0),
            dA_current.stride(1),
            dA_current.stride(2),
            grad_scale_intermediate.stride(0),
            grad_scale_intermediate.stride(1),
            grad_scale_intermediate.stride(2),
            grad_scale_intermediate.stride(3),
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )

    def launch_next_decay():
        _mamba2_bwd_dq_cached_next_decay_cp_kernel[decay_grid](
            processed_next,
            dt,
            grad_scale_intermediate,
            dA_next,
            A,
            processed_next.stride(0),
            processed_next.stride(1),
            processed_next.stride(2),
            processed_next.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            grad_scale_intermediate.stride(0),
            grad_scale_intermediate.stride(1),
            grad_scale_intermediate.stride(2),
            grad_scale_intermediate.stride(3),
            dA_next.stride(0),
            dA_next.stride(1),
            dA_next.stride(2),
            A.stride(0),
            seqlen,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            num_warps=4,
            num_stages=1,
        )

    compact_args = (
        C, B, x, dout, z, dt, dA_cs, dA_cs_rev,
        chunk_states, final_dstates, A, D,
        dB_partial, dx, dD_partial, processed_dkdv, dA_dkdv,
        ddA_cs_rev_intermediate,
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2),
        dout.stride(3), dout.stride(4),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1),
        dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        chunk_states.stride(0), chunk_states.stride(1),
        chunk_states.stride(2), chunk_states.stride(3), chunk_states.stride(4),
        final_dstates.stride(0), final_dstates.stride(1),
        final_dstates.stride(2), final_dstates.stride(3), final_dstates.stride(4),
        A.stride(0),
        D.stride(0),
        dB_partial.stride(0), dB_partial.stride(1),
        dB_partial.stride(2), dB_partial.stride(3),
        dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
        dD_partial.stride(0), dD_partial.stride(1),
        processed_dkdv.stride(0), processed_dkdv.stride(1),
        processed_dkdv.stride(2), processed_dkdv.stride(3),
        dA_dkdv.stride(0), dA_dkdv.stride(1), dA_dkdv.stride(2),
        ddA_cs_rev_intermediate.stride(0), ddA_cs_rev_intermediate.stride(1),
        ddA_cs_rev_intermediate.stride(2), ddA_cs_rev_intermediate.stride(3),
        seqlen, headdim, dstate, nheads_qk,
    )

    def launch_dkdv_compact():
        _mamba2_bwd_dk_dv_compact_cp_kernel[main_grid](
            *compact_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
        )

    def launch_prev_decay():
        _mamba2_bwd_dk_dv_cached_prev_decay_cp_kernel[decay_grid](
            processed_dkdv, dt, ddA_cs_rev_intermediate, dA_dkdv, A,
            processed_dkdv.stride(0), processed_dkdv.stride(1),
            processed_dkdv.stride(2), processed_dkdv.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            ddA_cs_rev_intermediate.stride(0),
            ddA_cs_rev_intermediate.stride(1),
            ddA_cs_rev_intermediate.stride(2),
            ddA_cs_rev_intermediate.stride(3),
            dA_dkdv.stride(0), dA_dkdv.stride(1), dA_dkdv.stride(2),
            A.stride(0),
            seqlen,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            num_warps=4,
            num_stages=1,
        )

    # Compile/autotune before timing. The first launch also creates the actual
    # cached GSI, chunk-start states, and previous-decay intermediate consumed
    # by the lightweight mapping kernels.
    for launch in (
        launch_dq,
        launch_next_decay,
        launch_dkdv_compact,
        launch_prev_decay,
    ):
        launch()
    torch.cuda.synchronize(device)

    compact_config = _mamba2_bwd_dk_dv_compact_cp_kernel.best_config
    compact_block_m = compact_config.kwargs["BLOCK_M"]
    compact_block_k = compact_config.kwargs["BLOCK_K"]

    def tensor_bytes(*tensors):
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    logical_traffic = {
        "dq_fused_current": (
            tensor_bytes(C, B, x, dout, z, dt, dA_cs, dA_cs_rev,
                         initial_states, A, D),
            tensor_bytes(dC_partial, dz, chunk_states, processed_current, dA_current,
                         grad_scale_intermediate),
        ),
        "dq_next": (
            tensor_bytes(dt, grad_scale_intermediate, A, processed_next, dA_next),
            tensor_bytes(processed_next, dA_next),
        ),
        "dkdv_compact": (
            tensor_bytes(C, B, x, dout, z, dt, dA_cs, dA_cs_rev,
                         chunk_states, final_dstates, A, D,
                         processed_dkdv, dA_dkdv),
            tensor_bytes(dB_partial, dx, dD_partial, processed_dkdv, dA_dkdv,
                         ddA_cs_rev_intermediate),
        ),
        "dkdv_prev": (
            tensor_bytes(dt, ddA_cs_rev_intermediate, A,
                         processed_dkdv, dA_dkdv),
            tensor_bytes(processed_dkdv, dA_dkdv),
        ),
    }

    activation_bytes = torch.empty((), dtype=dtype).element_size()
    fp32_bytes = torch.empty((), dtype=torch.float32).element_size()
    padded_seqlen = nchunks * chunk_size
    seq_n = folded_batch * nheads * seqlen * dstate
    seq_p = folded_batch * nheads * seqlen * headdim
    seq_scalar = folded_batch * nheads * seqlen
    decay = folded_batch * nheads * padded_seqlen
    chunk_scalar = folded_batch * nheads * nchunks
    boundary_state = folded_batch * nheads * headdim * dstate
    chunk_state = chunk_scalar * headdim * dstate
    head_programs = folded_batch * nheads

    compact_row_tiles = triton.cdiv(chunk_size, compact_block_m)
    compact_query_tiles = triton.cdiv(chunk_size, compact_block_k)

    mapper_shifted_scalar = seq_scalar - chunk_scalar

    source_traffic = {
        "dq_fused_current": (
            (
                boundary_state + head_programs + 3 * decay + 2 * chunk_scalar
            ) * fp32_bytes
            + (2 * seq_n + 3 * seq_p) * activation_bytes,
            (chunk_state + 2 * decay + chunk_scalar) * fp32_bytes
            + (seq_n + seq_p) * activation_bytes,
        ),
        "dq_next": (
            (3 * decay + 2 * chunk_scalar) * fp32_bytes,
            (decay + chunk_scalar) * fp32_bytes,
        ),
        "dkdv_compact": (
            (
                (compact_row_tiles + 2) * seq_n
                + (2 * compact_row_tiles + 5) * seq_p
            ) * activation_bytes
            + (
                boundary_state
                + chunk_state
                + 2 * head_programs
                + 2 * chunk_scalar
                + (compact_row_tiles + compact_query_tiles + 4) * decay
            ) * fp32_bytes,
            (seq_n + seq_p) * activation_bytes
            + (head_programs + 2 * decay + chunk_scalar) * fp32_bytes,
        ),
        "dkdv_prev": (
            (2 * decay + 2 * mapper_shifted_scalar + 2 * chunk_scalar) * fp32_bytes,
            (decay + chunk_scalar) * fp32_bytes,
        ),
    }

    benchmarks = (
        ("_mamba2_bwd_dq_cp_kernel[fused-current]", "dq_fused_current", launch_dq),
        ("_mamba2_bwd_dq_cached_next_decay_cp_kernel", "dq_next", launch_next_decay),
        ("_mamba2_bwd_dk_dv_compact_cp_kernel[offload-prev]", "dkdv_compact",
         launch_dkdv_compact),
        ("_mamba2_bwd_dk_dv_cached_prev_decay_cp_kernel", "dkdv_prev",
         launch_prev_decay),
    )

    results = []
    for kernel_name, traffic_key, launch in benchmarks:
        time_ms = do_bench(
            launch,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
        results.append(
            (kernel_name, time_ms, logical_traffic[traffic_key], source_traffic[traffic_key])
        )

    table_width = 154
    print("=" * table_width)
    print("TRUE-CP V2 BACKWARD RAW-KERNEL BENCHMARK")
    print("=" * table_width)
    print(
        f"public_local_seqlen={local_seqlen} segment_seqlen={seqlen} "
        f"batch={batch} folded_batch={folded_batch} heads={nheads} "
        f"qk_groups={nheads_qk} P={headdim} N={dstate} chunk={chunk_size} "
        f"nchunks={nchunks} dtype={dtype} device={device}"
    )
    print(
        f"GPU={device_properties.name} max_memory_clock="
        f"{device_properties.memory_clock_rate / 1_000:.0f} MHz "
        f"memory_bus_width={device_properties.memory_bus_width} bits "
        f"theoretical_peak_gmem={peak_gmem_gbps:.1f} GB/s "
        f"peak_source={peak_gmem_source}"
    )
    print(f"dQ autotune: {_mamba2_bwd_dq_cp_kernel.best_config}")
    print(f"dK/dV compact autotune: {compact_config}")
    print(
        "Logical = one copy of each semantic tensor region. Source = explicit "
        "masked tl.load/tl.store element traffic, including replay and RMW."
    )
    print(
        "Neither byte model is a measured hardware DRAM counter; %peak is effective GB/s "
        "divided by theoretical peak. MB and GB/s are decimal."
    )
    print("-" * table_width)
    print(
        f"{'kernel':50s} {'ms':>9s}  "
        f"{'logical R/W/T MB':>27s} {'logical GB/s':>12s} {'%peak':>7s}  "
        f"{'source R/W/T MB':>27s} {'source GB/s':>11s} {'%peak':>7s}"
    )
    print("-" * table_width)

    total_time_ms = 0.0
    total_logical_read = 0
    total_logical_write = 0
    total_source_read = 0
    total_source_write = 0
    for kernel_name, time_ms, logical, source in results:
        logical_read, logical_write = logical
        source_read, source_write = source
        total_time_ms += time_ms
        total_logical_read += logical_read
        total_logical_write += logical_write
        total_source_read += source_read
        total_source_write += source_write
        logical_total = logical_read + logical_write
        source_total = source_read + source_write
        logical_gbps = logical_total / (time_ms * 1e6)
        source_gbps = source_total / (time_ms * 1e6)
        print(
            f"{kernel_name:50s} {time_ms:9.4f}  "
            f"{logical_read / 1e6:8.1f}/{logical_write / 1e6:8.1f}/"
            f"{logical_total / 1e6:8.1f} {logical_gbps:12.1f} "
            f"{100.0 * logical_gbps / peak_gmem_gbps:6.1f}%  "
            f"{source_read / 1e6:8.1f}/{source_write / 1e6:8.1f}/"
            f"{source_total / 1e6:8.1f} {source_gbps:11.1f} "
            f"{100.0 * source_gbps / peak_gmem_gbps:6.1f}%"
        )

    total_logical = total_logical_read + total_logical_write
    total_source = total_source_read + total_source_write
    total_logical_gbps = total_logical / (total_time_ms * 1e6)
    total_source_gbps = total_source / (total_time_ms * 1e6)
    print("-" * table_width)
    print(
        f"{'sum of four isolated kernels':50s} {total_time_ms:9.4f}  "
        f"{total_logical_read / 1e6:8.1f}/{total_logical_write / 1e6:8.1f}/"
        f"{total_logical / 1e6:8.1f} {total_logical_gbps:12.1f} "
        f"{100.0 * total_logical_gbps / peak_gmem_gbps:6.1f}%  "
        f"{total_source_read / 1e6:8.1f}/{total_source_write / 1e6:8.1f}/"
        f"{total_source / 1e6:8.1f} {total_source_gbps:11.1f} "
        f"{100.0 * total_source_gbps / peak_gmem_gbps:6.1f}%"
    )
    print("=" * table_width)


def main():
    """Command-line entry point for the isolated true-CP backward benchmark."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the four production Mamba true-CP v2 backward kernels."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--local-seqlen",
        type=int,
        default=8192,
        help="Public per-rank sequence length before the two-stream split.",
    )
    parser.add_argument("--nheads", type=int, default=128)
    parser.add_argument("--nheads-qk", type=int, default=8, help="Number of local B/C groups.")
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--warmup-ms", type=int, default=25, help="Target warmup duration passed to do_bench."
    )
    parser.add_argument(
        "--rep-ms", type=int, default=100, help="Target measurement duration passed to do_bench."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    benchmark_speed(
        batch=args.batch,
        local_seqlen=args.local_seqlen,
        nheads=args.nheads,
        nheads_qk=args.nheads_qk,
        headdim=args.headdim,
        dstate=args.dstate,
        dtype=dtype,
        device=args.device,
        chunk_size=args.chunk_size,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
