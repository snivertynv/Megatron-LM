"""
Copyright (c) 2026, Dao AI Lab, Goombalab

Mamba2 fwd implementation in Triton.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from mamba_ssm.ops.triton.softplus import softplus


@triton.jit
def mamba2_fwd_kernel(
    Q,
    K,
    V,
    DT,
    DA_CS,
    DA_CS_REV,
    INITIAL_STATES,
    Z,
    D,
    OUT,
    stride_q_batch,
    stride_q_seqlen,
    stride_q_head,
    stride_q_dim,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_head,
    stride_k_dim,
    stride_v_batch,
    stride_v_seqlen,
    stride_v_head,
    stride_v_dim,
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
    stride_initial_v_dim,
    stride_initial_qk_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_stream,
    stride_out_seqlen,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
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
    """Evaluate one complete true-CP segment for one folded-batch/head pair."""
    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)

    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    q_offset = pid_folded_batch * stride_q_batch + head_idx_qk * stride_q_head
    k_offset = pid_folded_batch * stride_k_batch + head_idx_qk * stride_k_head
    v_offset = pid_folded_batch * stride_v_batch + pid_head * stride_v_head
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
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    out_offset = (
        stream * stride_out_stream
        + batch_idx * stride_out_batch
        + pid_head * stride_out_head
    )

    offs_chunk = tl.arange(0, BLOCK_CHUNK)
    offs_qk_dim = tl.arange(0, BLOCK_HEADDIM_QK)
    offs_v_dim = tl.arange(0, BLOCK_HEADDIM_V)
    initial_ptrs = (
        INITIAL_STATES
        + initial_offset
        + offs_qk_dim[:, None] * stride_initial_qk_dim
        + offs_v_dim[None, :] * stride_initial_v_dim
    )
    initial_mask = (offs_qk_dim[:, None] < headdim_qk) & (
        offs_v_dim[None, :] < headdim_v
    )
    acc_states = tl.load(initial_ptrs, mask=initial_mask, other=0.0).to(tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seq = chunk_start + offs_chunk
        logical_chunk_mask = offs_chunk < CHUNK_SIZE
        seq_mask = logical_chunk_mask & (offs_seq < seqlen)

        dt_block = tl.load(
            DT
            + decay_offset
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

        k_ptrs = (
            K
            + k_offset
            + offs_seq[:, None] * stride_k_seqlen
            + offs_qk_dim[None, :] * stride_k_dim
        )
        k_mask = seq_mask[:, None] & (offs_qk_dim[None, :] < headdim_qk)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = (
            V
            + v_offset
            + offs_seq[:, None] * stride_v_seqlen
            + offs_v_dim[None, :] * stride_v_dim
        )
        v_mask = seq_mask[:, None] & (offs_v_dim[None, :] < headdim_v)
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        q_ptrs = (
            Q
            + q_offset
            + offs_seq[:, None] * stride_q_seqlen
            + offs_qk_dim[None, :] * stride_q_dim
        )
        q_mask = seq_mask[:, None] & (offs_qk_dim[None, :] < headdim_qk)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        acc_o = tl.dot(q_block, acc_states.to(q_block.dtype)) * tl.exp(dA_cs)[:, None]
        s_block = tl.dot(q_block, tl.trans(k_block))
        valid_matrix = seq_mask[:, None] & seq_mask[None, :]
        causal = offs_chunk[:, None] >= offs_chunk[None, :]
        segsum = tl.where(
            valid_matrix & causal,
            dA_cs[:, None] - dA_cs[None, :],
            -float("inf"),
        )
        segsum = tl.minimum(segsum, 0.0)
        s_block *= tl.exp(segsum)
        s_block *= dt_block[None, :]
        acc_o += tl.dot(s_block.to(v_block.dtype), v_block)

        d_val = tl.load(D + pid_head).to(tl.float32)
        acc_o += v_block * d_val
        z_ptrs = (
            Z
            + z_offset
            + offs_seq[:, None] * stride_z_seqlen
            + offs_v_dim[None, :] * stride_z_dim
        )
        z_block = tl.load(z_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        acc_o *= z_block * tl.sigmoid(z_block)

        out_ptrs = (
            OUT
            + out_offset
            + offs_seq[:, None] * stride_out_seqlen
            + offs_v_dim[None, :] * stride_out_dim
        )
        out_mask = seq_mask[:, None] & (offs_v_dim[None, :] < headdim_v)
        tl.store(out_ptrs, acc_o.to(v_block.dtype), mask=out_mask)

        state_scale = tl.exp(dA_cs_rev) * dt_block
        k_block *= state_scale[:, None]
        acc_states = tl.exp(dA_cs_last) * acc_states + tl.dot(
            tl.trans(k_block).to(v_block.dtype), v_block
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "BLOCK_HEADDIM_V"],
)
@triton.jit
def _mamba2_fwd_large_chunk_kernel(
    Q,
    K,
    V,
    DT,
    DA_CS,
    DA_CS_REV,
    INITIAL_STATES,
    Z,
    D,
    OUT,
    stride_q_batch,
    stride_q_seqlen,
    stride_q_head,
    stride_q_dim,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_head,
    stride_k_dim,
    stride_v_batch,
    stride_v_seqlen,
    stride_v_head,
    stride_v_dim,
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
    stride_initial_v_dim,
    stride_initial_qk_dim,
    stride_z_stream,
    stride_z_seqlen,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_stream,
    stride_out_seqlen,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    seqlen,
    headdim_qk,
    headdim_v,
    nheads_qk,
    BATCH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Evaluate a segment without keeping a full chunk-square tile live."""

    pid_head = tl.program_id(0)
    pid_folded_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH

    q_offset = pid_folded_batch * stride_q_batch + head_idx_qk * stride_q_head
    k_offset = pid_folded_batch * stride_k_batch + head_idx_qk * stride_k_head
    v_offset = pid_folded_batch * stride_v_batch + pid_head * stride_v_head
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
    z_offset = (
        stream * stride_z_stream
        + batch_idx * stride_z_batch
        + pid_head * stride_z_head
    )
    out_offset = (
        stream * stride_out_stream
        + batch_idx * stride_out_batch
        + pid_head * stride_out_head
    )

    offs_qk = tl.arange(0, BLOCK_HEADDIM_QK)
    offs_v = tl.arange(0, BLOCK_HEADDIM_V)
    state_mask = (offs_qk[:, None] < headdim_qk) & (offs_v[None, :] < headdim_v)
    acc_states = tl.load(
        INITIAL_STATES
        + initial_offset
        + offs_qk[:, None] * stride_initial_qk_dim
        + offs_v[None, :] * stride_initial_v_dim,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    d_value = tl.load(D + pid_head).to(tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_dacs_offset = dacs_offset + chunk_idx * stride_dacs_chunk
        chunk_dacsrev_offset = dacsrev_offset + chunk_idx * stride_dacsrev_chunk
        chunk_dt_offset = decay_offset + chunk_idx * stride_dt_chunk
        dA_cs_last = tl.load(
            DA_CS + chunk_dacs_offset + (CHUNK_SIZE - 1) * stride_dacs_csize
        ).to(tl.float32)

        for row_start in tl.range(0, CHUNK_SIZE, BLOCK_M, num_stages=1):
            offs_m = row_start + tl.arange(0, BLOCK_M)
            offs_seq_m = chunk_start + offs_m
            row_logical = offs_m < CHUNK_SIZE
            row_valid = row_logical & (offs_seq_m < seqlen)

            q_block = tl.load(
                Q
                + q_offset
                + offs_seq_m[:, None] * stride_q_seqlen
                + offs_qk[None, :] * stride_q_dim,
                mask=row_valid[:, None] & (offs_qk[None, :] < headdim_qk),
                other=0.0,
            )
            v_row = tl.load(
                V
                + v_offset
                + offs_seq_m[:, None] * stride_v_seqlen
                + offs_v[None, :] * stride_v_dim,
                mask=row_valid[:, None] & (offs_v[None, :] < headdim_v),
                other=0.0,
            )
            dA_row = tl.load(
                DA_CS + chunk_dacs_offset + offs_m * stride_dacs_csize,
                mask=row_logical,
                other=0.0,
            ).to(tl.float32)
            acc_o = tl.dot(q_block, acc_states.to(q_block.dtype))
            acc_o *= tl.exp(dA_row)[:, None]

            for key_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
                offs_k = key_start + tl.arange(0, BLOCK_K)
                offs_seq_k = chunk_start + offs_k
                key_logical = offs_k < CHUNK_SIZE
                key_valid = key_logical & (offs_seq_k < seqlen)
                k_block = tl.load(
                    K
                    + k_offset
                    + offs_seq_k[:, None] * stride_k_seqlen
                    + offs_qk[None, :] * stride_k_dim,
                    mask=key_valid[:, None] & (offs_qk[None, :] < headdim_qk),
                    other=0.0,
                )
                v_block = tl.load(
                    V
                    + v_offset
                    + offs_seq_k[:, None] * stride_v_seqlen
                    + offs_v[None, :] * stride_v_dim,
                    mask=key_valid[:, None] & (offs_v[None, :] < headdim_v),
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
                valid_matrix = row_valid[:, None] & key_valid[None, :]
                causal = offs_m[:, None] >= offs_k[None, :]
                segsum = tl.where(
                    valid_matrix & causal,
                    dA_row[:, None] - dA_col[None, :],
                    -float("inf"),
                )
                segsum = tl.minimum(segsum, 0.0)
                scores = tl.dot(q_block, tl.trans(k_block))
                scores *= tl.exp(segsum) * dt_block[None, :]
                acc_o += tl.dot(scores.to(v_block.dtype), v_block)

            acc_o += v_row.to(tl.float32) * d_value
            z_block = tl.load(
                Z
                + z_offset
                + offs_seq_m[:, None] * stride_z_seqlen
                + offs_v[None, :] * stride_z_dim,
                mask=row_valid[:, None] & (offs_v[None, :] < headdim_v),
                other=0.0,
            ).to(tl.float32)
            acc_o *= z_block * tl.sigmoid(z_block)
            tl.store(
                OUT
                + out_offset
                + offs_seq_m[:, None] * stride_out_seqlen
                + offs_v[None, :] * stride_out_dim,
                acc_o.to(v_row.dtype),
                mask=row_valid[:, None] & (offs_v[None, :] < headdim_v),
            )

        acc_states *= tl.exp(dA_cs_last)
        for key_start in tl.range(0, CHUNK_SIZE, BLOCK_K, num_stages=1):
            offs_k = key_start + tl.arange(0, BLOCK_K)
            offs_seq_k = chunk_start + offs_k
            key_logical = offs_k < CHUNK_SIZE
            key_valid = key_logical & (offs_seq_k < seqlen)
            k_block = tl.load(
                K
                + k_offset
                + offs_seq_k[:, None] * stride_k_seqlen
                + offs_qk[None, :] * stride_k_dim,
                mask=key_valid[:, None] & (offs_qk[None, :] < headdim_qk),
                other=0.0,
            )
            v_block = tl.load(
                V
                + v_offset
                + offs_seq_k[:, None] * stride_v_seqlen
                + offs_v[None, :] * stride_v_dim,
                mask=key_valid[:, None] & (offs_v[None, :] < headdim_v),
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
            k_block *= (tl.exp(dA_rev) * dt_block)[:, None].to(k_block.dtype)
            acc_states += tl.dot(tl.trans(k_block).to(v_block.dtype), v_block)


def mamba2_fwd_triton(
    q: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, headdim_qk)
    k: torch.Tensor,  # (folded_batch, seqlen, nheads_qk, headdim_qk)
    v: torch.Tensor,  # (folded_batch, seqlen, nheads, headdim_v)
    dt: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    dA_cs_rev: torch.Tensor,  # (folded_batch, nheads, nchunks, chunk_size)
    initial_states: torch.Tensor,  # (streams, batch, nheads, headdim_v, headdim_qk)
    D: torch.Tensor,  # (nheads,)
    z: torch.Tensor,  # (streams, seqlen, batch, nheads, headdim_v)
    out: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Evaluate true-CP Mamba segments from prepared decay and prefix inputs.

    Causal log-decay is reconstructed from ``dA_cs`` inside the kernels rather
    than read from a materialized segsum tensor. Chunks up to 64 use one
    whole-chunk program-local tile. Larger chunks use a persistent-state kernel
    that reconstructs only output/key subtiles.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be 4D")
    if any(tensor.ndim != 4 for tensor in (dt, dA_cs, dA_cs_rev)):
        raise ValueError("dt, dA_cs, and dA_cs_rev must be 4D")
    if initial_states.ndim != 5 or z.ndim != 5:
        raise ValueError("initial_states and z must be 5D")

    folded_batch, seqlen, nheads_qk, headdim_qk = q.shape
    if k.shape != q.shape:
        raise ValueError("q and k must have the same shape")
    v_batch, v_seqlen, nheads, headdim_v = v.shape
    if (v_batch, v_seqlen) != (folded_batch, seqlen):
        raise ValueError("q, k, and v must share folded batch and sequence dimensions")
    if nheads_qk <= 0 or nheads <= 0 or nheads % nheads_qk != 0:
        raise ValueError("the number of value heads must be divisible by q/k heads")
    if headdim_qk <= 0 or headdim_v <= 0:
        raise ValueError("head dimensions must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    streams, batch, initial_nheads, initial_headdim_v, initial_headdim_qk = (
        initial_states.shape
    )
    if streams != 2:
        raise ValueError("true-CP forward requires exactly two local streams")
    if folded_batch != streams * batch:
        raise ValueError("folded q/k/v batch must equal streams * initial-state batch")
    if (initial_nheads, initial_headdim_v, initial_headdim_qk) != (
        nheads,
        headdim_v,
        headdim_qk,
    ):
        raise ValueError("initial-state head dimensions must match q/k/v")
    stream_shape = (streams, seqlen, batch, nheads, headdim_v)
    if z.shape != stream_shape:
        raise ValueError(f"z must have shape {stream_shape}, got {tuple(z.shape)}")

    nchunks = triton.cdiv(seqlen, chunk_size)
    decay_shape = (folded_batch, nheads, nchunks, chunk_size)
    for name, tensor in (("dt", dt), ("dA_cs", dA_cs), ("dA_cs_rev", dA_cs_rev)):
        if tensor.shape != decay_shape:
            raise ValueError(f"{name} must have shape {decay_shape}, got {tuple(tensor.shape)}")
    if D.shape != (nheads,):
        raise ValueError(f"D must have shape {(nheads,)}, got {tuple(D.shape)}")

    tensors = (q, k, v, dt, dA_cs, dA_cs_rev, initial_states, D, z)
    if not q.is_cuda:
        raise ValueError("mamba2_fwd_triton requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != z.dtype:
        raise ValueError("q, k, v, and z must have the same dtype")
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must use floating-point dtypes")

    if out is None:
        out = torch.empty(stream_shape, device=v.device, dtype=v.dtype)
    elif out.shape != stream_shape or out.device != v.device or out.dtype != v.dtype:
        raise ValueError("out must match the stream shape, device, and activation dtype")

    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_headdim_qk = max(16, triton.next_power_of_2(headdim_qk))
    block_headdim_v = max(16, triton.next_power_of_2(headdim_v))
    grid = (nheads, folded_batch)
    kernel_args = (
        q,
        k,
        v,
        dt,
        dA_cs,
        dA_cs_rev,
        initial_states,
        z,
        D,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1), dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        initial_states.stride(0), initial_states.stride(1), initial_states.stride(2),
        initial_states.stride(3), initial_states.stride(4),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3), z.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        seqlen,
        headdim_qk,
        headdim_v,
        nheads_qk,
    )
    if chunk_size <= 64:
        mamba2_fwd_kernel[grid](
            *kernel_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_CHUNK=block_chunk,
            BLOCK_HEADDIM_QK=block_headdim_qk,
            BLOCK_HEADDIM_V=block_headdim_v,
            num_warps=4,
            num_stages=2,
        )
    else:
        _mamba2_fwd_large_chunk_kernel[grid](
            *kernel_args,
            BATCH=batch,
            CHUNK_SIZE=chunk_size,
            BLOCK_HEADDIM_QK=block_headdim_qk,
            BLOCK_HEADDIM_V=block_headdim_v,
        )
    return out


@triton.autotune(
    configs=[
        # Two warps can win for small state matrices; larger matrices generally
        # need four or eight. One-stage variants limit resource pressure while
        # two-stage variants can overlap the K/V loads with the dot product.
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    # Sequence length changes only the number of recurrent iterations, so it
    # should not create another autotune cache entry.
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "BLOCK_HEADDIM_V"],
)
@triton.jit
def _mamba2_get_final_state_kernel(
    K,
    V,
    DT,
    DA_CS,
    DA_CS_REV,
    FinalState,
    LogAlpha,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_head,
    stride_k_dim,
    stride_v_batch,
    stride_v_seqlen,
    stride_v_head,
    stride_v_dim,
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
    stride_final_state_batch,
    stride_final_state_nheads,
    stride_final_state_v_dim,
    stride_final_state_qk_dim,
    stride_log_alpha_batch,
    stride_log_alpha_head,
    seqlen,
    headdim_qk,
    headdim_v,
    nheads_qk,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
):
    """Compute one final recurrent state for each (batch, head) pair."""
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head
    final_state_offset = pid_batch * stride_final_state_batch + pid_head * stride_final_state_nheads

    acc_states = tl.zeros([BLOCK_HEADDIM_QK, BLOCK_HEADDIM_V], dtype=tl.float32)
    log_alpha = 0.0
    offs_qk_dim = tl.arange(0, BLOCK_HEADDIM_QK)
    offs_v_dim = tl.arange(0, BLOCK_HEADDIM_V)
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_chunk = tl.arange(0, BLOCK_CHUNK)
        offs_seq = chunk_start + offs_chunk
        seq_mask = (offs_chunk < CHUNK_SIZE) & (offs_seq < seqlen)
        dt_offset = (
            pid_batch * stride_dt_batch
            + pid_head * stride_dt_head
            + chunk_idx * stride_dt_chunk
        )
        dt_block = tl.load(
            DT + dt_offset + offs_chunk * stride_dt_csize,
            mask=offs_chunk < CHUNK_SIZE,
            other=0.0,
        )
        dacsrev_offset = (
            pid_batch * stride_dacsrev_batch
            + pid_head * stride_dacsrev_head
            + chunk_idx * stride_dacsrev_chunk
        )
        dA_cs_rev = tl.load(
            DA_CS_REV + dacsrev_offset + offs_chunk * stride_dacsrev_csize,
            mask=offs_chunk < CHUNK_SIZE,
            other=0.0,
        )
        dacs_offset = (
            pid_batch * stride_dacs_batch
            + pid_head * stride_dacs_head
            + chunk_idx * stride_dacs_chunk
        )
        dA_cs_last = tl.load(
            DA_CS + dacs_offset + (CHUNK_SIZE - 1) * stride_dacs_csize
        )
        log_alpha += dA_cs_last

        k_ptrs = (
            K
            + k_offset
            + offs_seq[:, None] * stride_k_seqlen
            + offs_qk_dim[None, :] * stride_k_dim
        )
        k_mask = seq_mask[:, None] & (offs_qk_dim[None, :] < headdim_qk)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = (
            V
            + v_offset
            + offs_seq[:, None] * stride_v_seqlen
            + offs_v_dim[None, :] * stride_v_dim
        )
        v_mask = seq_mask[:, None] & (offs_v_dim[None, :] < headdim_v)
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        scale = tl.exp(tl.minimum(dA_cs_rev, 0.0)) * dt_block
        k_block *= scale[:, None]
        acc_states = tl.exp(dA_cs_last) * acc_states + tl.dot(
            tl.trans(k_block).to(v_block.dtype), v_block
        )

    final_state_ptrs = (
        FinalState
        + final_state_offset
        + offs_v_dim[:, None] * stride_final_state_v_dim
        + offs_qk_dim[None, :] * stride_final_state_qk_dim
    )
    final_state_mask = (offs_v_dim[:, None] < headdim_v) & (
        offs_qk_dim[None, :] < headdim_qk
    )
    tl.store(final_state_ptrs, tl.trans(acc_states), mask=final_state_mask)
    tl.store(
        LogAlpha + pid_batch * stride_log_alpha_batch + pid_head * stride_log_alpha_head,
        log_alpha,
    )


def mamba2_get_final_state_triton(
    k: torch.Tensor,  # (b, s, h_qk, d)
    v: torch.Tensor,  # (b, s, h, dv)
    dt: torch.Tensor,  # (b, h, c, l)
    dA_cs: torch.Tensor,  # (b, h, c, l)
    dA_cs_rev: torch.Tensor,  # (b, h, c, l)
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each segment's compact affine summary ``(beta, log_alpha)``.

    ``beta`` is FP32 with shape ``(batch, nheads, headdim_v, headdim_qk)`` and
    ``log_alpha`` is FP32 with shape ``(batch, nheads)``. Query/key heads may
    be shared across multiple value heads. The decay inputs use the padded
    chunk layout produced by :func:`compute_dacs_compact_triton`.
    """
    if any(tensor.ndim != 4 for tensor in (dt, dA_cs, dA_cs_rev)):
        raise ValueError("expected dt, dA_cs, and dA_cs_rev to be 4D")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected k and v to be 4D")

    batch, seqlen, nheads_qk, headdim_qk = k.shape
    v_batch, v_seqlen, nheads, headdim_v = v.shape
    if (v_batch, v_seqlen) != (batch, seqlen):
        raise ValueError("k and v must have the same batch and sequence dimensions")
    if nheads_qk <= 0 or nheads <= 0 or nheads % nheads_qk != 0:
        raise ValueError("the number of value heads must be divisible by the number of k heads")
    if headdim_qk <= 0 or headdim_v <= 0:
        raise ValueError("head dimensions must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    nchunks = triton.cdiv(seqlen, chunk_size)
    decay_shape = (batch, nheads, nchunks, chunk_size)
    for name, tensor in (("dt", dt), ("dA_cs", dA_cs), ("dA_cs_rev", dA_cs_rev)):
        if tensor.shape != decay_shape:
            raise ValueError(
                f"{name} must have shape {decay_shape}, got {tuple(tensor.shape)}"
            )
    if not k.is_cuda:
        raise ValueError("mamba2_get_final_state_triton requires CUDA tensors")
    if any(tensor.device != k.device for tensor in (v, dt, dA_cs, dA_cs_rev)):
        raise ValueError("k, v, dt, dA_cs, and dA_cs_rev must be on the same device")
    if k.dtype != v.dtype:
        raise ValueError("k and v must have the same dtype")
    if not all(tensor.is_floating_point() for tensor in (k, v, dt, dA_cs, dA_cs_rev)):
        raise ValueError("k, v, dt, dA_cs, and dA_cs_rev must use floating-point dtypes")

    final_state = torch.empty(
        batch,
        nheads,
        headdim_v,
        headdim_qk,
        dtype=torch.float32,
        device=k.device,
    )
    log_alpha = torch.empty(batch, nheads, dtype=torch.float32, device=k.device)
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    block_headdim_qk = max(16, triton.next_power_of_2(headdim_qk))
    block_headdim_v = max(16, triton.next_power_of_2(headdim_v))
    grid = (nheads, batch)

    _mamba2_get_final_state_kernel[grid](
        k,
        v,
        dt,
        dA_cs,
        dA_cs_rev,
        final_state,
        log_alpha,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2), dA_cs.stride(3),
        dA_cs_rev.stride(0),
        dA_cs_rev.stride(1),
        dA_cs_rev.stride(2),
        dA_cs_rev.stride(3),
        final_state.stride(0),
        final_state.stride(1),
        final_state.stride(2),
        final_state.stride(3),
        log_alpha.stride(0),
        log_alpha.stride(1),
        seqlen,
        headdim_qk,
        headdim_v,
        nheads_qk,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
        BLOCK_HEADDIM_QK=block_headdim_qk,
        BLOCK_HEADDIM_V=block_headdim_v,
    )

    return final_state, log_alpha




@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK", "WRITE_DT_INPUT"],
)
@triton.jit
def dacs_segsum_kernel(
    dt_ptr,
    a_ptr,
    dt_bias_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    segsum_ptr,
    dt_out_ptr,
    dt_input_ptr,
    stride_dt_stream,
    stride_dt_seq,
    stride_dt_batch,
    stride_dt_head,
    stride_a_head,
    stride_dt_bias_head,
    stride_da_cs_batch,
    stride_da_cs_head,
    stride_da_cs_chunk,
    stride_da_cs_csize,
    stride_da_cs_rev_batch,
    stride_da_cs_rev_head,
    stride_da_cs_rev_chunk,
    stride_da_cs_rev_csize,
    stride_segsum_batch,
    stride_segsum_head,
    stride_segsum_chunk,
    stride_segsum_row,
    stride_segsum_col,
    stride_dt_out_batch,
    stride_dt_out_head,
    stride_dt_out_chunk,
    stride_dt_out_csize,
    stride_dt_input_batch,
    stride_dt_input_seq,
    stride_dt_input_head,
    BATCH: tl.constexpr,
    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    WRITE_DT_INPUT: tl.constexpr,
):
    """Prepare chunked decay tensors for independent strided CP streams."""
    pid_folded_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH
    chunk_start = pid_chunk * CHUNK_SIZE
    offs = tl.arange(0, BLOCK_CHUNK)
    offs_seq = chunk_start + offs
    valid_seq = (offs < CHUNK_SIZE) & (offs_seq < SEQLEN)

    dt_base = (
        stream * stride_dt_stream
        + batch_idx * stride_dt_batch
        + pid_head * stride_dt_head
    )
    dt_input = tl.load(
        dt_ptr + dt_base + offs_seq * stride_dt_seq,
        mask=valid_seq,
        other=0.0,
    )
    if WRITE_DT_INPUT:
        folded_dt_ptrs = (
            dt_input_ptr
            + pid_folded_batch * stride_dt_input_batch
            + offs_seq * stride_dt_input_seq
            + pid_head * stride_dt_input_head
        )
        tl.store(folded_dt_ptrs, dt_input, mask=valid_seq)

    dt = dt_input.to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + pid_head * stride_dt_bias_head).to(tl.float32)
    dt += dt_bias
    dt = tl.where(dt <= 20.0, softplus(dt), dt)
    dt = tl.where(valid_seq, dt, 0.0)

    a = tl.load(a_ptr + pid_head * stride_a_head).to(tl.float32)
    dA = dt * a
    dA_cs = tl.minimum(tl.cumsum(dA, axis=0), 0.0)
    dA_last = tl.sum(dA)
    dA_cs_rev = tl.where(valid_seq, tl.minimum(dA_last - dA_cs, 0.0), 0.0)

    output_base = pid_folded_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    output_ptrs = (
        output_base
        + pid_chunk * stride_da_cs_chunk
        + offs * stride_da_cs_csize
    )
    logical_chunk_mask = offs < CHUNK_SIZE
    tl.store(da_cs_ptr + output_ptrs, dA_cs, mask=logical_chunk_mask)

    output_rev_base = (
        pid_folded_batch * stride_da_cs_rev_batch
        + pid_head * stride_da_cs_rev_head
        + pid_chunk * stride_da_cs_rev_chunk
    )
    tl.store(
        da_cs_rev_ptr + output_rev_base + offs * stride_da_cs_rev_csize,
        dA_cs_rev,
        mask=logical_chunk_mask,
    )

    dt_out_base = (
        pid_folded_batch * stride_dt_out_batch
        + pid_head * stride_dt_out_head
        + pid_chunk * stride_dt_out_chunk
    )
    tl.store(
        dt_out_ptr + dt_out_base + offs * stride_dt_out_csize,
        dt,
        mask=logical_chunk_mask,
    )

    offs_i = offs[:, None]
    offs_j = offs[None, :]
    valid_matrix = (
        (offs_i < CHUNK_SIZE)
        & (offs_j < CHUNK_SIZE)
        & (chunk_start + offs_i < SEQLEN)
        & (chunk_start + offs_j < SEQLEN)
    )
    segsum = tl.where(
        valid_matrix & (offs_i >= offs_j),
        dA_cs[:, None] - dA_cs[None, :],
        -float("inf"),
    )
    segsum = tl.minimum(segsum, 0.0)
    segsum_base = (
        pid_folded_batch * stride_segsum_batch
        + pid_head * stride_segsum_head
        + pid_chunk * stride_segsum_chunk
    )
    tl.store(
        segsum_ptr
        + segsum_base
        + offs_i * stride_segsum_row
        + offs_j * stride_segsum_col,
        segsum,
        mask=(offs_i < CHUNK_SIZE) & (offs_j < CHUNK_SIZE),
    )


def compute_dacs_segsum_triton(
    dt_streams: torch.Tensor,  # [streams, seqlen, batch, nheads]
    A: torch.Tensor,  # [nheads]
    dt_bias: torch.Tensor,  # [nheads]
    chunk_size: int,
    *,
    return_dt_input: bool = False,
):
    """Prepare the full decay oracle without materializing a folded stream batch.

    Returns ``(dA_cumsum, dA_cs_rev, segsum, dt_out, dt_input)``. The first,
    second, and fourth tensors use shape ``[streams * batch, heads, chunks,
    chunk_size]``. Valid causal segsum entries contain log decay; noncausal and
    padded entries are negative infinity. ``dt_input`` is ``None`` unless
    requested. This compatibility helper is for focused tests/reference only;
    production true-CP uses :func:`compute_dacs_compact_triton`.
    """
    if dt_streams.ndim != 4:
        raise ValueError("dt_streams must have shape [streams, seqlen, batch, heads]")
    streams, seqlen, batch, nheads = dt_streams.shape
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if A.shape != (nheads,) or dt_bias.shape != (nheads,):
        raise ValueError("A and dt_bias must each have shape [nheads]")
    if not dt_streams.is_cuda:
        raise ValueError("compute_dacs_segsum_triton requires CUDA tensors")
    if A.device != dt_streams.device or dt_bias.device != dt_streams.device:
        raise ValueError("dt_streams, A, and dt_bias must be on the same device")
    if not all(tensor.is_floating_point() for tensor in (dt_streams, A, dt_bias)):
        raise ValueError("dt_streams, A, and dt_bias must use floating-point dtypes")

    folded_batch = streams * batch
    nchunks = triton.cdiv(seqlen, chunk_size)
    chunk_shape = (folded_batch, nheads, nchunks, chunk_size)
    dA_cumsum = torch.empty(chunk_shape, device=dt_streams.device, dtype=torch.float32)
    dA_cs_rev = torch.empty_like(dA_cumsum)
    dt_out = torch.empty_like(dA_cumsum)
    segsum = torch.empty(
        folded_batch,
        nheads,
        nchunks,
        chunk_size,
        chunk_size,
        device=dt_streams.device,
        dtype=torch.float32,
    )
    dt_input = (
        torch.empty(
            folded_batch,
            seqlen,
            nheads,
            device=dt_streams.device,
            dtype=dt_streams.dtype,
        )
        if return_dt_input
        else None
    )

    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    grid = (folded_batch, nheads, nchunks)
    dacs_segsum_kernel[grid](
        dt_streams,
        A,
        dt_bias,
        dA_cumsum,
        dA_cs_rev,
        segsum,
        dt_out,
        dt_input,
        dt_streams.stride(0),
        dt_streams.stride(1),
        dt_streams.stride(2),
        dt_streams.stride(3),
        A.stride(0),
        dt_bias.stride(0),
        dA_cumsum.stride(0),
        dA_cumsum.stride(1),
        dA_cumsum.stride(2),
        dA_cumsum.stride(3),
        dA_cs_rev.stride(0),
        dA_cs_rev.stride(1),
        dA_cs_rev.stride(2),
        dA_cs_rev.stride(3),
        segsum.stride(0),
        segsum.stride(1),
        segsum.stride(2),
        segsum.stride(3),
        segsum.stride(4),
        dt_out.stride(0),
        dt_out.stride(1),
        dt_out.stride(2),
        dt_out.stride(3),
        *(
            (dt_input.stride(0), dt_input.stride(1), dt_input.stride(2))
            if dt_input is not None
            else (0, 0, 0)
        ),
        batch,
        seqlen,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
        WRITE_DT_INPUT=return_dt_input,
    )
    return dA_cumsum, dA_cs_rev, segsum, dt_out, dt_input


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE", "BLOCK_CHUNK"],
)
@triton.jit
def _dacs_compact_kernel(
    dt_ptr,
    a_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    stride_dt_stream,
    stride_dt_seq,
    stride_dt_batch,
    stride_dt_head,
    stride_a_head,
    stride_dt_bias_head,
    stride_dt_out_batch,
    stride_dt_out_head,
    stride_dt_out_chunk,
    stride_dt_out_csize,
    stride_da_cs_batch,
    stride_da_cs_head,
    stride_da_cs_chunk,
    stride_da_cs_csize,
    stride_da_cs_rev_batch,
    stride_da_cs_rev_head,
    stride_da_cs_rev_chunk,
    stride_da_cs_rev_csize,
    BATCH: tl.constexpr,
    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """Prepare the compact decay tensors shared by forward and backward."""

    pid_folded_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    stream = pid_folded_batch // BATCH
    batch_idx = pid_folded_batch - stream * BATCH
    chunk_start = pid_chunk * CHUNK_SIZE
    offs = tl.arange(0, BLOCK_CHUNK)
    seq = chunk_start + offs
    logical = offs < CHUNK_SIZE
    valid = logical & (seq < SEQLEN)

    raw_dt = tl.load(
        dt_ptr
        + stream * stride_dt_stream
        + seq * stride_dt_seq
        + batch_idx * stride_dt_batch
        + pid_head * stride_dt_head,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(dt_bias_ptr + pid_head * stride_dt_bias_head).to(tl.float32)
    biased_dt = raw_dt + bias
    dt = tl.where(biased_dt <= 20.0, softplus(biased_dt), biased_dt)
    dt = tl.where(valid, dt, 0.0)

    a = tl.load(a_ptr + pid_head * stride_a_head).to(tl.float32)
    dA = dt * a
    dA_cs = tl.minimum(tl.cumsum(dA, axis=0), 0.0)
    chunk_total = tl.sum(dA)
    dA_cs_rev = tl.where(valid, tl.minimum(chunk_total - dA_cs, 0.0), 0.0)

    dt_base = (
        pid_folded_batch * stride_dt_out_batch
        + pid_head * stride_dt_out_head
        + pid_chunk * stride_dt_out_chunk
    )
    dacs_base = (
        pid_folded_batch * stride_da_cs_batch
        + pid_head * stride_da_cs_head
        + pid_chunk * stride_da_cs_chunk
    )
    dacsrev_base = (
        pid_folded_batch * stride_da_cs_rev_batch
        + pid_head * stride_da_cs_rev_head
        + pid_chunk * stride_da_cs_rev_chunk
    )
    tl.store(dt_out_ptr + dt_base + offs * stride_dt_out_csize, dt, mask=logical)
    tl.store(da_cs_ptr + dacs_base + offs * stride_da_cs_csize, dA_cs, mask=logical)
    tl.store(
        da_cs_rev_ptr + dacsrev_base + offs * stride_da_cs_rev_csize,
        dA_cs_rev,
        mask=logical,
    )


def compute_dacs_compact_triton(
    dt_streams: torch.Tensor,  # [streams, seqlen, batch, nheads]
    A: torch.Tensor,  # [nheads]
    dt_bias: torch.Tensor,  # [nheads]
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare ``(dt_out, dA_cumsum, dA_cs_rev)`` without materializing segsum.

    All outputs use padded ``[streams * batch, heads, chunks, chunk_size]``
    layout. The input is addressed with its actual stream-major strides, so no
    folded sequence copy is required.
    """

    if dt_streams.ndim != 4:
        raise ValueError("dt_streams must have shape [streams, seqlen, batch, heads]")
    streams, seqlen, batch, nheads = dt_streams.shape
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if A.shape != (nheads,) or dt_bias.shape != (nheads,):
        raise ValueError("A and dt_bias must each have shape [nheads]")
    if not dt_streams.is_cuda:
        raise ValueError("compute_dacs_compact_triton requires CUDA tensors")
    if A.device != dt_streams.device or dt_bias.device != dt_streams.device:
        raise ValueError("dt_streams, A, and dt_bias must be on the same device")
    if not all(tensor.is_floating_point() for tensor in (dt_streams, A, dt_bias)):
        raise ValueError("dt_streams, A, and dt_bias must use floating-point dtypes")

    shape = (
        streams * batch,
        nheads,
        triton.cdiv(seqlen, chunk_size),
        chunk_size,
    )
    dt_out = torch.empty(shape, device=dt_streams.device, dtype=torch.float32)
    dA_cumsum = torch.empty_like(dt_out)
    dA_cs_rev = torch.empty_like(dt_out)
    block_chunk = max(16, triton.next_power_of_2(chunk_size))
    _dacs_compact_kernel[(shape[0], nheads, shape[2])](
        dt_streams,
        A,
        dt_bias,
        dt_out,
        dA_cumsum,
        dA_cs_rev,
        dt_streams.stride(0), dt_streams.stride(1),
        dt_streams.stride(2), dt_streams.stride(3),
        A.stride(0),
        dt_bias.stride(0),
        dt_out.stride(0), dt_out.stride(1),
        dt_out.stride(2), dt_out.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(1),
        dA_cumsum.stride(2), dA_cumsum.stride(3),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1),
        dA_cs_rev.stride(2), dA_cs_rev.stride(3),
        BATCH=batch,
        SEQLEN=seqlen,
        CHUNK_SIZE=chunk_size,
        BLOCK_CHUNK=block_chunk,
    )
    return dt_out, dA_cumsum, dA_cs_rev
