# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.mamba2_utils.mamba2_bwd import mamba2_bwd_dk_dv_cp_triton
from megatron.core.ssm.mamba2_utils.mamba2_fwd import compute_dacs_segsum_triton


def _chunk_start_states(B, x, dt, dA_cs, dA_cs_rev, initial_states):
    streams, batch, nheads, headdim, dstate = initial_states.shape
    folded_batch, seqlen, nheads_qk, _ = B.shape
    heads_per_qk = nheads // nheads_qk
    chunk_size = dt.shape[-1]
    nchunks = dt.shape[2]
    states = torch.empty(
        folded_batch,
        nheads,
        nchunks,
        headdim,
        dstate,
        device=B.device,
        dtype=torch.float32,
    )
    B_float = B.float()
    x_float = x.float()
    for stream in range(streams):
        for batch_idx in range(batch):
            folded_idx = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                state = initial_states[stream, batch_idx, head].float()
                for chunk_idx in range(nchunks):
                    states[folded_idx, head, chunk_idx] = state
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, seqlen)
                    chunk_len = chunk_end - chunk_start
                    b_block = B_float[folded_idx, chunk_start:chunk_end, qk_head]
                    x_block = x_float[folded_idx, chunk_start:chunk_end, head]
                    dt_block = dt[folded_idx, head, chunk_idx, :chunk_len].float()
                    dA_rev_block = dA_cs_rev[
                        folded_idx, head, chunk_idx, :chunk_len
                    ].float()
                    chunk_total = dA_cs[folded_idx, head, chunk_idx, -1].float()
                    state = torch.exp(chunk_total) * state + x_block.transpose(0, 1) @ (
                        b_block * (torch.exp(dA_rev_block) * dt_block)[:, None]
                    )
    return states


def _dk_dv_reference(
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
):
    streams, seqlen, batch, nheads, headdim = dout.shape
    folded_batch, _, nheads_qk, _ = C.shape
    heads_per_qk = nheads // nheads_qk
    nchunks, chunk_size = dt.shape[2:]

    B_ref = B.detach().float().requires_grad_()
    x_ref = x.detach().float().requires_grad_()
    D_ref = D.detach().float().requires_grad_()
    dt_prev = dt.detach().float().requires_grad_()
    next_probe = torch.ones_like(dA_cs, dtype=torch.float32, requires_grad=True)
    prev_probe = torch.ones_like(dA_cs_rev, dtype=torch.float32, requires_grad=True)
    curr_probe = torch.ones_like(segsum, dtype=torch.float32, requires_grad=True)
    transition_probe = torch.ones(
        folded_batch,
        nheads,
        nchunks,
        device=C.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    C_float = C.float()
    dpre = dout.float() * z.float() * torch.sigmoid(z.float())
    loss = torch.zeros((), device=C.device, dtype=torch.float32)
    for stream in range(streams):
        for batch_idx in range(batch):
            folded_idx = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                state = chunk_start_states[folded_idx, head, 0].float()
                for chunk_idx in range(nchunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, seqlen)
                    chunk_len = chunk_end - chunk_start
                    c_block = C_float[folded_idx, chunk_start:chunk_end, qk_head]
                    b_block = B_ref[folded_idx, chunk_start:chunk_end, qk_head]
                    x_block = x_ref[folded_idx, chunk_start:chunk_end, head]
                    dpre_block = dpre[stream, chunk_start:chunk_end, batch_idx, head]
                    dt_block = dt[folded_idx, head, chunk_idx, :chunk_len].float()
                    dt_prev_block = dt_prev[
                        folded_idx, head, chunk_idx, :chunk_len
                    ]
                    dA_block = dA_cs[
                        folded_idx, head, chunk_idx, :chunk_len
                    ].float()
                    dA_rev_block = dA_cs_rev[
                        folded_idx, head, chunk_idx, :chunk_len
                    ].float()
                    segsum_block = segsum[
                        folded_idx, head, chunk_idx, :chunk_len, :chunk_len
                    ].float()

                    pre_gate = (
                        (c_block @ state.transpose(0, 1))
                        * torch.exp(dA_block)[:, None]
                        * next_probe[
                            folded_idx, head, chunk_idx, :chunk_len
                        ][:, None]
                    )
                    pre_gate = pre_gate + (
                        (c_block @ b_block.transpose(0, 1))
                        * torch.exp(segsum_block)
                        * curr_probe[
                            folded_idx,
                            head,
                            chunk_idx,
                            :chunk_len,
                            :chunk_len,
                        ]
                        * dt_block[None, :]
                    ) @ x_block
                    pre_gate = pre_gate + D_ref[head] * x_block
                    loss = loss + (pre_gate * dpre_block).sum()

                    chunk_total = dA_cs[folded_idx, head, chunk_idx, -1].float()
                    state = (
                        torch.exp(chunk_total)
                        * transition_probe[folded_idx, head, chunk_idx]
                        * state
                        + x_block.transpose(0, 1)
                        @ (
                            b_block
                            * (
                                torch.exp(dA_rev_block)
                                * prev_probe[
                                    folded_idx, head, chunk_idx, :chunk_len
                                ]
                                * dt_prev_block
                            )[:, None]
                        )
                    )
                loss = loss + (
                    state * final_dstates[stream, batch_idx, head].float()
                ).sum()

    (
        dB,
        dx,
        dD,
        grad_scale,
        ddA_cs_rev,
        curr_probe_grad,
        transition_grad,
        dt_prev_grad,
    ) = torch.autograd.grad(
        loss,
        (
            B_ref,
            x_ref,
            D_ref,
            next_probe,
            prev_probe,
            curr_probe,
            transition_probe,
            dt_prev,
        ),
    )

    dM_curr = torch.zeros_like(curr_probe_grad)
    initial_ddt = dt_prev_grad.clone()
    initial_dA = torch.zeros(nheads, device=C.device, dtype=torch.float32)
    for chunk_idx, chunk_start in enumerate(range(0, seqlen, chunk_size)):
        chunk_len = min(chunk_size, seqlen - chunk_start)
        dt_valid = dt[:, :, chunk_idx, :chunk_len].float()
        dM_curr[:, :, chunk_idx, :chunk_len, :chunk_len] = (
            curr_probe_grad[:, :, chunk_idx, :chunk_len, :chunk_len]
            / dt_valid[:, :, None, :]
        )
        initial_ddt[:, :, chunk_idx, :chunk_len] += (
            A[None, :, None].float() * transition_grad[:, :, chunk_idx, None]
        )
        initial_dA += (
            transition_grad[:, :, chunk_idx] * dt_valid.sum(dim=-1)
        ).sum(dim=0)

    return (
        dB,
        dx,
        dD,
        initial_dA,
        initial_ddt,
        ddA_cs_rev,
        grad_scale,
        dM_curr,
    )


def _compact_decay_reference(initial_ddt, ddA_cs_rev, dt, A, seqlen):
    processed = torch.zeros_like(initial_ddt)
    dA_chunk = torch.zeros(
        *dt.shape[:3], device=dt.device, dtype=torch.float32
    )
    folded_batch, nheads, nchunks, chunk_size = dt.shape
    for batch_idx in range(folded_batch):
        for head in range(nheads):
            for chunk_idx in range(nchunks):
                start = chunk_idx * chunk_size
                length = min(chunk_size, max(0, seqlen - start))
                if length == 0:
                    continue
                dtv = dt[batch_idx, head, chunk_idx, :length].float()
                prev = ddA_cs_rev[batch_idx, head, chunk_idx, :length].float()
                shifted_prev = torch.cat((prev.new_zeros(1), prev[:-1]))
                prefix = torch.cumsum(shifted_prev, dim=0)
                processed[batch_idx, head, chunk_idx, :length] = (
                    initial_ddt[batch_idx, head, chunk_idx, :length] + A[head] * prefix
                )
                shifted_dt = torch.cat((dtv[1:], dtv.new_zeros(1)))
                suffix = torch.flip(
                    torch.cumsum(torch.flip(shifted_dt, dims=(0,)), dim=0), dims=(0,)
                )
                direct = prev / dtv
                transition = (
                    initial_ddt[batch_idx, head, chunk_idx, 0] - direct[0]
                ) / A[head]
                dA_chunk[batch_idx, head, chunk_idx] = (
                    (prev * suffix).sum() + transition * dtv.sum()
                )
    return processed, dA_chunk


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads_qk", "nheads", "dstate", "headdim"),
    (
        (35, 17, 1, 2, 13, 11),
        (70, 64, 2, 4, 16, 16),
        (269, 256, 1, 2, 128, 64),
    ),
)
def test_mamba2_bwd_dk_dv_cp_matches_reference(
    dtype, seqlen, chunk_size, nheads_qk, nheads, dstate, headdim
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton CP dK/dV kernel")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(86420)
    device = torch.device("cuda")
    streams, batch = 2, 2
    folded_batch = streams * batch

    c_storage = 0.1 * torch.randn(
        folded_batch, seqlen, nheads_qk, dstate + 3, device=device, dtype=dtype
    )
    b_storage = 0.1 * torch.randn_like(c_storage)
    x_storage = 0.1 * torch.randn(
        folded_batch, seqlen, nheads, headdim + 5, device=device, dtype=dtype
    )
    C = c_storage[..., :dstate]
    B = b_storage[..., :dstate]
    x = x_storage[..., :headdim]

    dout_storage = 0.08 * torch.randn(
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
    dout[1].add_(0.12)
    z[1].sub_(0.3)

    initial_storage = 0.08 * torch.randn(
        streams,
        batch,
        nheads,
        headdim,
        dstate + 3,
        device=device,
        dtype=torch.float32,
    )
    initial_states = initial_storage[..., :dstate]
    initial_states[1].add_(0.15)
    final_storage = 0.08 * torch.randn_like(initial_storage)
    final_dstates = final_storage[..., :dstate]
    final_dstates[1].sub_(0.1)

    dt_storage = torch.randn(
        streams, seqlen, batch, nheads + 2, device=device, dtype=dtype
    )
    dt_streams = dt_storage[..., :nheads]
    A = -torch.rand(nheads, device=device, dtype=torch.float32)
    D = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)
    dA_cs, dA_cs_rev, segsum, dt_out, _ = compute_dacs_segsum_triton(
        dt_streams, A, dt_bias, chunk_size
    )

    dt_out = dt_out.clone()
    dA_cs = dA_cs.clone()
    dA_cs_rev = dA_cs_rev.clone()
    dt_out[..., 0] *= 0.91
    dA_cs[..., 0] -= 0.04
    dA_cs_rev[..., 0] -= 0.06
    for chunk_idx, chunk_start in enumerate(range(0, seqlen, chunk_size)):
        chunk_len = min(chunk_size, seqlen - chunk_start)
        values = dA_cs[:, :, chunk_idx, :chunk_len]
        rebuilt = values[:, :, :, None] - values[:, :, None, :]
        causal = torch.ones(
            chunk_len, chunk_len, device=device, dtype=torch.bool
        ).tril()
        segsum[:, :, chunk_idx].fill_(-torch.inf)
        segsum[:, :, chunk_idx, :chunk_len, :chunk_len] = rebuilt.masked_fill(
            ~causal, -torch.inf
        )

    states = _chunk_start_states(
        B, x, dt_out, dA_cs, dA_cs_rev, initial_states
    )
    state_storage = torch.empty(
        *states.shape[:-1], dstate + 3, device=device, dtype=torch.float32
    )
    state_storage[..., :dstate] = states
    chunk_start_states = state_storage[..., :dstate]

    assert not C.is_contiguous()
    assert not B.is_contiguous()
    assert not x.is_contiguous()
    assert not dout.is_contiguous()
    assert not z.is_contiguous()
    assert not chunk_start_states.is_contiguous()
    assert not final_dstates.is_contiguous()

    expected = _dk_dv_reference(
        C,
        B,
        x,
        dout,
        z,
        dt_out,
        dA_cs,
        dA_cs_rev,
        segsum,
        chunk_start_states,
        final_dstates,
        A,
        D,
    )
    expected_dB, expected_dx, expected_dD, _, initial_ddt, prev_grad, _, _ = expected
    expected_ddt, expected_dA_chunk = _compact_decay_reference(
        initial_ddt, prev_grad, dt_out, A, seqlen
    )
    # Production enters dK/dV with current/next contributions already owned by
    # dQ. Nonzero seeds also catch repeated RMW updates during autotuning.
    processed_ddt = 0.03 * torch.randn_like(dt_out)
    dA_chunk = 0.03 * torch.randn(
        folded_batch, nheads, dt_out.shape[2], device=device, dtype=torch.float32
    )
    expected_ddt += processed_ddt.clone()
    expected_dA_chunk += dA_chunk.clone()
    actual = mamba2_bwd_dk_dv_cp_triton(
        C,
        B,
        x,
        dout,
        z,
        dt_out,
        dA_cs,
        dA_cs_rev,
        chunk_start_states,
        final_dstates,
        A,
        D,
        processed_ddt,
        dA_chunk,
        chunk_size=chunk_size,
    )

    actual_dB, actual_dx, actual_dD, actual_ddt, actual_dA_chunk = actual
    assert actual_dB.shape == B.shape
    assert actual_dx.shape == x.shape
    assert actual_dD.shape == D.shape
    assert actual_dB.dtype is dtype
    assert actual_dx.dtype is dtype
    assert actual_dD.dtype is D.dtype
    assert actual_ddt.dtype is torch.float32
    assert actual_dA_chunk.dtype is torch.float32
    torch.testing.assert_close(actual_dB.float(), expected_dB, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(actual_dx.float(), expected_dx, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(actual_dD.float(), expected_dD, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(actual_ddt, expected_ddt, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(actual_dA_chunk, expected_dA_chunk, atol=1e-1, rtol=1e-1)
