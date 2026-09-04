# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm import mamba_true_context_parallel_v2 as true_cp_v2
from megatron.core.ssm.mamba2_utils.mamba2_bwd import mamba2_bwd_dq_cp_triton
from megatron.core.ssm.mamba2_utils.mamba2_fwd import compute_dacs_segsum_triton


def _dq_reference(C, B, x, dout, z, dt, dA_cs, dA_cs_rev, initial_states, A, D):
    streams, seqlen, batch, nheads, headdim = dout.shape
    folded_batch, _, nheads_qk, dstate = C.shape
    heads_per_qk = nheads // nheads_qk
    chunk_size = dt.shape[-1]
    nchunks = dt.shape[2]

    C_ref = C.detach().float().requires_grad_()
    z_ref = z.detach().float().requires_grad_()
    B_float = B.float()
    x_float = x.float()
    dout_float = dout.float()
    out = torch.zeros_like(z_ref)
    chunk_states = torch.empty(
        folded_batch,
        nheads,
        nchunks,
        headdim,
        dstate,
        device=C.device,
        dtype=torch.float32,
    )

    for stream in range(streams):
        for batch_idx in range(batch):
            folded_idx = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                state = initial_states[stream, batch_idx, head].float()
                for chunk_idx in range(nchunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, seqlen)
                    chunk_len = chunk_end - chunk_start
                    chunk_states[folded_idx, head, chunk_idx] = state

                    c_block = C_ref[folded_idx, chunk_start:chunk_end, qk_head]
                    b_block = B_float[folded_idx, chunk_start:chunk_end, qk_head]
                    x_block = x_float[folded_idx, chunk_start:chunk_end, head]
                    dt_block = dt[folded_idx, head, chunk_idx, :chunk_len].float()
                    dA_block = dA_cs[folded_idx, head, chunk_idx, :chunk_len].float()
                    dA_rev_block = dA_cs_rev[
                        folded_idx, head, chunk_idx, :chunk_len
                    ].float()
                    segsum_block = dA_block[:, None] - dA_block[None, :]
                    segsum_block = segsum_block.masked_fill(
                        torch.ones_like(segsum_block, dtype=torch.bool).triu(1),
                        -torch.inf,
                    )

                    pre_gate = (c_block @ state.transpose(0, 1)) * torch.exp(
                        dA_block
                    )[:, None]
                    pre_gate = pre_gate + (
                        (c_block @ b_block.transpose(0, 1))
                        * torch.exp(segsum_block)
                        * dt_block[None, :]
                    ) @ x_block
                    pre_gate = pre_gate + D[head].float() * x_block
                    z_block = z_ref[stream, chunk_start:chunk_end, batch_idx, head]
                    out[stream, chunk_start:chunk_end, batch_idx, head] = (
                        pre_gate * F.silu(z_block)
                    )

                    chunk_total = dA_cs[folded_idx, head, chunk_idx, -1].float()
                    state = torch.exp(chunk_total) * state + x_block.transpose(0, 1) @ (
                        b_block * (torch.exp(dA_rev_block) * dt_block)[:, None]
                    )

    dC, dz = torch.autograd.grad(out, (C_ref, z_ref), grad_outputs=dout_float)
    processed_ddt = torch.zeros_like(dt, dtype=torch.float32)
    dA_chunk = torch.zeros(
        folded_batch, nheads, nchunks, device=C.device, dtype=torch.float32
    )
    dpre = dout.float() * z.float() * torch.sigmoid(z.float())
    for stream in range(streams):
        for batch_idx in range(batch):
            folded_idx = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                for chunk_idx in range(nchunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, seqlen)
                    length = end - start
                    c = C.float()[folded_idx, start:end, qk_head]
                    b = B_float[folded_idx, start:end, qk_head]
                    xv = x_float[folded_idx, start:end, head]
                    dp = dpre[stream, start:end, batch_idx, head]
                    dtv = dt[folded_idx, head, chunk_idx, :length].float()
                    dacs = dA_cs[folded_idx, head, chunk_idx, :length].float()
                    seg = (dacs[:, None] - dacs[None, :]).masked_fill(
                        torch.ones(length, length, device=C.device, dtype=torch.bool).triu(1),
                        -torch.inf,
                    )
                    dM = (c @ b.T) * (dp @ xv.T) * torch.exp(seg)
                    strict = torch.ones_like(dM, dtype=torch.bool).tril(-1)
                    g = torch.where(strict, dM * dtv[None, :], 0.0)
                    reverse_g = torch.flip(
                        torch.cumsum(torch.flip(g, dims=(0,)), dim=0), dims=(0,)
                    )
                    ddA = torch.where(strict, reverse_g, 0.0).sum(dim=1)
                    state = chunk_states[folded_idx, head, chunk_idx]
                    grad_scale = ((dp @ state) * c).sum(dim=1) * torch.exp(dacs)
                    reverse_scale = torch.flip(
                        torch.cumsum(torch.flip(grad_scale, dims=(0,)), dim=0), dims=(0,)
                    )
                    processed_ddt[folded_idx, head, chunk_idx, :length] = (
                        dM.sum(dim=0) + A[head] * (ddA + reverse_scale)
                    )
                    dA_chunk[folded_idx, head, chunk_idx] = (
                        (ddA * dtv).sum() + (torch.cumsum(dtv, dim=0) * grad_scale).sum()
                    )
    return dC, dz, chunk_states, processed_ddt, dA_chunk


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads_qk", "nheads", "dstate", "headdim"),
    (
        (35, 17, 1, 2, 13, 11),
        (70, 64, 2, 4, 16, 16),
        (269, 256, 1, 2, 128, 64),
    ),
)
def test_mamba2_bwd_dq_cp_matches_reference(
    dtype, seqlen, chunk_size, nheads_qk, nheads, dstate, headdim
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton CP dQ kernel")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(97531)
    device = torch.device("cuda")
    streams, batch = 2, 2
    folded_batch = streams * batch

    c_storage = 0.12 * torch.randn(
        folded_batch,
        seqlen,
        nheads_qk,
        dstate + 3,
        device=device,
        dtype=dtype,
    )
    b_storage = 0.12 * torch.randn_like(c_storage)
    x_storage = 0.12 * torch.randn(
        folded_batch,
        seqlen,
        nheads,
        headdim + 5,
        device=device,
        dtype=dtype,
    )
    C = c_storage[..., :dstate]
    B = b_storage[..., :dstate]
    x = x_storage[..., :headdim]

    dout_storage = 0.1 * torch.randn(
        streams,
        seqlen,
        batch,
        nheads,
        headdim + 7,
        device=device,
        dtype=dtype,
    )
    z_storage = torch.randn_like(dout_storage)
    dout = dout_storage[..., :headdim]
    z = z_storage[..., :headdim]
    dout[1].add_(0.15)
    z[1].sub_(0.35)

    initial_storage = 0.1 * torch.randn(
        streams,
        batch,
        nheads,
        headdim,
        dstate + 3,
        device=device,
        dtype=torch.float32,
    )
    initial_states = initial_storage[..., :dstate]
    initial_states[1].add_(0.2)
    D = torch.randn(nheads, device=device, dtype=torch.float32)

    dt_storage = torch.randn(
        streams,
        seqlen,
        batch,
        nheads + 2,
        device=device,
        dtype=dtype,
    )
    dt_streams = dt_storage[..., :nheads]
    A = -torch.rand(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)
    dA_cs, dA_cs_rev, segsum, dt_out, _ = compute_dacs_segsum_triton(
        dt_streams, A, dt_bias, chunk_size
    )

    # Make the prepared tensors intentionally inconsistent with one another.
    # Both the reference and kernel must consume these exact values.
    dt_out = dt_out.clone()
    dA_cs = dA_cs.clone()
    dA_cs_rev = dA_cs_rev.clone()
    dt_out[..., 0] *= 0.9
    dA_cs[..., 0] -= 0.05
    dA_cs_rev[..., 0] -= 0.07

    assert not C.is_contiguous()
    assert not B.is_contiguous()
    assert not x.is_contiguous()
    assert not dout.is_contiguous()
    assert not z.is_contiguous()
    assert not initial_states.is_contiguous()

    expected_dC, expected_dz, expected_states, expected_ddt, expected_dA = _dq_reference(
        C,
        B,
        x,
        dout,
        z,
        dt_out,
        dA_cs,
        dA_cs_rev,
        initial_states,
        A,
        D,
    )
    actual_dC, actual_dz, actual_states, actual_ddt, actual_dA = mamba2_bwd_dq_cp_triton(
        C,
        B,
        x,
        dout,
        z,
        dt_out,
        dA_cs,
        dA_cs_rev,
        initial_states,
        A,
        D,
        chunk_size=chunk_size,
    )

    assert actual_dC.shape == C.shape
    assert actual_dz.shape == dout.shape
    assert actual_states.shape == expected_states.shape
    assert actual_dC.dtype is dtype
    assert actual_dz.dtype is dtype
    assert actual_states.dtype is torch.float32
    assert actual_ddt.dtype is torch.float32
    assert actual_dA.dtype is torch.float32
    assert actual_dC.is_contiguous()
    assert actual_dz.is_contiguous()
    assert actual_states.is_contiguous()
    torch.testing.assert_close(actual_dC.float(), expected_dC, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(actual_dz.float(), expected_dz, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(actual_states, expected_states, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(actual_ddt, expected_ddt, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(actual_dA, expected_dA, atol=1e-1, rtol=1e-1)

    if chunk_size == 64:
        # X=64 receives every chunk's already-scanned incoming state. The
        # returned dK/dV handoff must be a strided view of that allocation,
        # not another 512 MiB chunk-state tensor at the primary shape.
        split_initial_states = expected_states.permute(0, 2, 1, 3, 4).contiguous()
        split_dC, split_dz, split_states, split_ddt, split_dA = (
            mamba2_bwd_dq_cp_triton(
                C,
                B,
                x,
                dout,
                z,
                dt_out,
                dA_cs,
                dA_cs_rev,
                split_initial_states,
                A,
                D,
                chunk_size=chunk_size,
                subsequence_size=64,
            )
        )
        assert split_states.untyped_storage().data_ptr() == (
            split_initial_states.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(split_dC.float(), expected_dC, atol=7e-2, rtol=7e-2)
        torch.testing.assert_close(split_dz.float(), expected_dz, atol=7e-2, rtol=7e-2)
        torch.testing.assert_close(split_states, expected_states, atol=6e-2, rtol=6e-2)
        torch.testing.assert_close(split_ddt, expected_ddt, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(split_dA, expected_dA, atol=1e-1, rtol=1e-1)


def test_format_grad_output_streams_is_a_no_copy_strided_view():
    streams, seqlen, batch, nheads, headdim = 2, 5, 3, 4, 7
    storage = torch.randn(streams * seqlen, batch, nheads * headdim + 3)
    grad_output = storage[..., : nheads * headdim]
    formatted = true_cp_v2._format_grad_output_streams(grad_output, headdim=headdim)

    assert formatted.shape == (streams, seqlen, batch, nheads, headdim)
    assert formatted.untyped_storage().data_ptr() == grad_output.untyped_storage().data_ptr()
    assert formatted.storage_offset() == grad_output.storage_offset()
    assert formatted.stride() == (
        seqlen * grad_output.stride(0),
        grad_output.stride(0),
        grad_output.stride(1),
        headdim * grad_output.stride(2),
        grad_output.stride(2),
    )
    torch.testing.assert_close(
        formatted[1, 2, 1, 3, 4],
        grad_output[seqlen + 2, 1, 3 * headdim + 4],
    )
