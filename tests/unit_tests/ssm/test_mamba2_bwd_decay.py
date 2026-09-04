# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.mamba2_utils.mamba2_bwd import (
    _mamba2_bwd_dq_cached_next_decay_cp_triton,
    dA_ddt_from_dM_curr_cp_triton,
    dA_ddt_from_dM_next_cp_triton,
    dA_ddt_from_dM_prev_cp_triton,
    mamba2_bwd_decay_map_cp_triton,
    mamba2_bwd_dk_dv_cp_triton,
    mamba2_bwd_dq_cp_triton,
)
from megatron.core.ssm.mamba2_utils.mamba2_fwd import (
    compute_dacs_compact_triton,
    compute_dacs_segsum_triton,
)


def _reverse_cumsum(x):
    return torch.flip(torch.cumsum(torch.flip(x, dims=(0,)), dim=0), dims=(0,))


@pytest.mark.parametrize(("seqlen", "chunk_size"), ((35, 17), (70, 64)))
def test_cached_next_decay_mapper_matches_reference(seqlen, chunk_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the cached dM-next mapper")

    torch.manual_seed(161803)
    device = torch.device("cuda")
    folded_batch, nheads = 4, 3
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    shape = (folded_batch, nheads, nchunks, chunk_size)
    processed = torch.randn(shape, device=device, dtype=torch.float32)
    dt = torch.rand(shape, device=device, dtype=torch.float32) + 0.1
    gsi = 0.1 * torch.randn(shape, device=device, dtype=torch.float32)
    dA_chunk = torch.randn(
        folded_batch, nheads, nchunks, device=device, dtype=torch.float32
    )
    A = -torch.rand(nheads, device=device, dtype=torch.float32)

    expected_ddt = processed.clone()
    expected_dA = dA_chunk.clone()
    for chunk_idx in range(nchunks):
        chunk_len = min(chunk_size, seqlen - chunk_idx * chunk_size)
        dt_chunk = dt[:, :, chunk_idx, :chunk_len]
        gsi_chunk = gsi[:, :, chunk_idx, :chunk_len]
        expected_ddt[:, :, chunk_idx, :chunk_len] += (
            A[None, :, None] * torch.flip(
                torch.cumsum(torch.flip(gsi_chunk, dims=(-1,)), dim=-1),
                dims=(-1,),
            )
        )
        expected_ddt[:, :, chunk_idx, chunk_len:] = 0.0
        expected_dA[:, :, chunk_idx] += (
            torch.cumsum(dt_chunk, dim=-1) * gsi_chunk
        ).sum(dim=-1)

    actual_ddt, actual_dA = _mamba2_bwd_dq_cached_next_decay_cp_triton(
        processed,
        dt,
        gsi,
        dA_chunk,
        A,
        seqlen,
        chunk_size,
    )
    assert actual_ddt.data_ptr() == processed.data_ptr()
    assert actual_dA.data_ptr() == dA_chunk.data_ptr()
    torch.testing.assert_close(actual_ddt, expected_ddt, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_dA, expected_dA, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(("seqlen", "chunk_size"), ((35, 17), (70, 64), (269, 256)))
def test_compact_decay_preparation_matches_full_oracle(seqlen, chunk_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton decay preparation")
    torch.manual_seed(314159)
    streams, batch, nheads = 2, 3, 4
    storage = torch.randn(
        streams, seqlen, batch, nheads + 3, device="cuda", dtype=torch.float16
    )
    raw_dt = storage[..., :nheads]
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32)
    bias = torch.randn(nheads, device="cuda", dtype=torch.float32)
    expected_dacs, expected_rev, _, expected_dt, _ = compute_dacs_segsum_triton(
        raw_dt, A, bias, chunk_size
    )
    actual_dt, actual_dacs, actual_rev = compute_dacs_compact_triton(
        raw_dt, A, bias, chunk_size
    )
    assert not raw_dt.is_contiguous()
    torch.testing.assert_close(actual_dt, expected_dt)
    torch.testing.assert_close(actual_dacs, expected_dacs)
    torch.testing.assert_close(actual_rev, expected_rev)


def _decay_stage_reference(ddt, dt, grad_scale, dM_curr, ddA_cs_rev, dA, A, seqlen):
    expected_next = torch.zeros_like(ddt)
    expected_curr = torch.zeros_like(ddt)
    expected_prev = torch.zeros_like(ddt)
    dA_next = dA.float().clone()
    dA_curr = None
    dA_prev = None
    folded_batch, nheads, nchunks, chunk_size = dt.shape

    for batch_idx in range(folded_batch):
        for head in range(nheads):
            for chunk_idx in range(nchunks):
                chunk_start = chunk_idx * chunk_size
                chunk_len = min(chunk_size, max(0, seqlen - chunk_start))
                if chunk_len == 0:
                    continue
                dt_block = dt[batch_idx, head, chunk_idx, :chunk_len].float()
                ddt_block = ddt[batch_idx, head, chunk_idx, :chunk_len].float()
                next_block = grad_scale[
                    batch_idx, head, chunk_idx, :chunk_len
                ].float()
                next_update = A[head].float() * _reverse_cumsum(next_block)
                expected_next[batch_idx, head, chunk_idx, :chunk_len] = (
                    ddt_block + next_update
                )
                dA_next[head] += (torch.cumsum(dt_block, dim=0) * next_block).sum()

    dA_curr = dA_next.clone()
    for batch_idx in range(folded_batch):
        for head in range(nheads):
            for chunk_idx in range(nchunks):
                chunk_start = chunk_idx * chunk_size
                chunk_len = min(chunk_size, max(0, seqlen - chunk_start))
                if chunk_len == 0:
                    continue
                dt_block = dt[batch_idx, head, chunk_idx, :chunk_len].float()
                ddt_block = expected_next[
                    batch_idx, head, chunk_idx, :chunk_len
                ].float()
                dM = dM_curr[
                    batch_idx, head, chunk_idx, :chunk_len, :chunk_len
                ].float()
                strict_lower = torch.tril(
                    torch.ones_like(dM, dtype=torch.bool), diagonal=-1
                )
                d_segsum = torch.where(strict_lower, dM * dt_block[None, :], 0.0)
                ddA_matrix = torch.flip(
                    torch.cumsum(torch.flip(d_segsum, dims=(0,)), dim=0), dims=(0,)
                )
                ddA = torch.where(strict_lower, ddA_matrix, 0.0).sum(dim=1)
                expected_curr[batch_idx, head, chunk_idx, :chunk_len] = (
                    ddt_block + dM.sum(dim=0) + A[head].float() * ddA
                )
                dA_curr[head] += (ddA * dt_block).sum()

    dA_prev = dA_curr.clone()
    for batch_idx in range(folded_batch):
        for head in range(nheads):
            for chunk_idx in range(nchunks):
                chunk_start = chunk_idx * chunk_size
                chunk_len = min(chunk_size, max(0, seqlen - chunk_start))
                if chunk_len == 0:
                    continue
                dt_block = dt[batch_idx, head, chunk_idx, :chunk_len].float()
                ddt_block = expected_curr[
                    batch_idx, head, chunk_idx, :chunk_len
                ].float()
                prev_block = ddA_cs_rev[
                    batch_idx, head, chunk_idx, :chunk_len
                ].float()
                shifted_dt = torch.cat((dt_block[1:], dt_block.new_zeros(1)))
                exclusive_suffix_dt = _reverse_cumsum(shifted_dt)
                shifted_prev = torch.cat((prev_block.new_zeros(1), prev_block[:-1]))
                exclusive_prefix_prev = torch.cumsum(shifted_prev, dim=0)
                expected_prev[batch_idx, head, chunk_idx, :chunk_len] = (
                    ddt_block + A[head].float() * exclusive_prefix_prev
                )
                dA_prev[head] += (prev_block * exclusive_suffix_dt).sum()

    return (expected_next, dA_next), (expected_curr, dA_curr), (expected_prev, dA_prev)


@pytest.mark.parametrize(("seqlen", "chunk_size"), ((35, 17), (70, 64), (269, 256)))
def test_decay_stages_match_stable_shifted_references(seqlen, chunk_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton CP decay kernels")

    torch.manual_seed(112358)
    device = torch.device("cuda")
    folded_batch, nheads = 4, 3
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    shape = (folded_batch, nheads, nchunks, chunk_size)
    ddt = torch.randn(shape, device=device, dtype=torch.float32)
    dt = torch.rand(shape, device=device, dtype=torch.float32) + 0.1
    grad_scale = 0.1 * torch.randn(shape, device=device, dtype=torch.float32)
    ddA_cs_rev = 0.1 * torch.randn(shape, device=device, dtype=torch.float32)
    dM_curr = 0.1 * torch.randn(
        *shape, chunk_size, device=device, dtype=torch.float32
    )
    dA = torch.randn(nheads, device=device, dtype=torch.float32)
    A = -torch.rand(nheads, device=device, dtype=torch.float32)

    # Adversarial ordering makes total-minus-prefix scans lose low-order terms.
    grad_scale[0, 0, 0, :4] = torch.tensor(
        [1.0e4, 1.0, -1.0e4, 3.0], device=device
    )
    ddA_cs_rev[0, 0, 0, :4] = torch.tensor(
        [1.0e4, 2.0, -1.0e4, 4.0], device=device
    )

    expected_next, expected_curr, expected_prev = _decay_stage_reference(
        ddt, dt, grad_scale, dM_curr, ddA_cs_rev, dA, A, seqlen
    )
    actual_next = dA_ddt_from_dM_next_cp_triton(
        ddt,
        dt,
        grad_scale,
        dA,
        A,
        seqlen=seqlen,
        chunk_size=chunk_size,
    )
    actual_curr = dA_ddt_from_dM_curr_cp_triton(
        actual_next[0],
        dt,
        dM_curr,
        actual_next[1],
        A,
        seqlen=seqlen,
        chunk_size=chunk_size,
    )
    actual_prev = dA_ddt_from_dM_prev_cp_triton(
        actual_curr[0],
        dt,
        ddA_cs_rev,
        actual_curr[1],
        A,
        seqlen=seqlen,
        chunk_size=chunk_size,
    )

    for actual, expected in (
        (actual_next, expected_next),
        (actual_curr, expected_curr),
        (actual_prev, expected_prev),
    ):
        torch.testing.assert_close(actual[0], expected[0], atol=2e-3, rtol=2e-5)
        # The per-head reduction may associate folded-batch contributions in a
        # different order after the cancellation-sensitive shifted scans.
        torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-4)


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
                    b_block = B[
                        folded_idx, chunk_start:chunk_end, qk_head
                    ].float()
                    x_block = x[folded_idx, chunk_start:chunk_end, head].float()
                    dt_block = dt[folded_idx, head, chunk_idx, :chunk_len].float()
                    dA_rev_block = dA_cs_rev[
                        folded_idx, head, chunk_idx, :chunk_len
                    ].float()
                    chunk_total = dA_cs[folded_idx, head, chunk_idx, -1].float()
                    state = torch.exp(chunk_total) * state + x_block.transpose(0, 1) @ (
                        b_block * (torch.exp(dA_rev_block) * dt_block)[:, None]
                    )
    return states


def _raw_dt_autograd_reference(
    C,
    B,
    x,
    dout,
    z,
    raw_dt,
    A_log,
    dt_bias,
    initial_states,
    final_dstates,
    D,
):
    streams, seqlen, batch, nheads, _ = dout.shape
    nheads_qk = C.shape[2]
    heads_per_qk = nheads // nheads_qk
    raw_ref = raw_dt.detach().float().requires_grad_()
    A_log_ref = A_log.detach().float().requires_grad_()
    bias_ref = dt_bias.detach().float().requires_grad_()
    A_ref = -torch.exp(A_log_ref)
    C_float = C.float()
    B_float = B.float()
    x_float = x.float()
    dout_float = dout.float()
    z_float = z.float()
    loss = torch.zeros((), device=C.device, dtype=torch.float32)

    for stream in range(streams):
        for batch_idx in range(batch):
            folded_idx = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                state = initial_states[stream, batch_idx, head].float()
                for seq_idx in range(seqlen):
                    u = raw_ref[stream, seq_idx, batch_idx, head] + bias_ref[head]
                    dt_value = F.softplus(u, beta=1.0, threshold=20.0)
                    decay = torch.exp(A_ref[head] * dt_value)
                    x_value = x_float[folded_idx, seq_idx, head]
                    b_value = B_float[folded_idx, seq_idx, qk_head]
                    state = decay * state + dt_value * x_value[:, None] * b_value[None, :]
                    pre_gate = C_float[folded_idx, seq_idx, qk_head] @ state.transpose(0, 1)
                    pre_gate = pre_gate + D[head].float() * x_value
                    gated_grad = (
                        dout_float[stream, seq_idx, batch_idx, head]
                        * F.silu(z_float[stream, seq_idx, batch_idx, head])
                    )
                    loss = loss + (pre_gate * gated_grad).sum()
                loss = loss + (
                    state * final_dstates[stream, batch_idx, head].float()
                ).sum()

    return torch.autograd.grad(loss, (raw_ref, A_log_ref, bias_ref))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads_qk", "nheads", "dstate", "headdim"),
    (
        (35, 17, 1, 2, 13, 11),
        (70, 64, 2, 4, 16, 16),
        (269, 256, 1, 2, 128, 64),
    ),
)
def test_mamba2_bwd_decay_cp_matches_raw_autograd(
    dtype, seqlen, chunk_size, nheads_qk, nheads, dstate, headdim
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton CP decay kernels")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(271828)
    device = torch.device("cuda")
    streams, batch = 2, 2
    folded_batch = streams * batch
    C = (0.06 * torch.randn(
        folded_batch, seqlen, nheads_qk, dstate + 3, device=device, dtype=dtype
    ))[..., :dstate]
    b_storage = 0.06 * torch.randn(
        folded_batch, seqlen, nheads_qk, dstate + 3, device=device, dtype=dtype
    )
    B = b_storage[..., :dstate]
    x = (0.06 * torch.randn(
        folded_batch, seqlen, nheads, headdim + 5, device=device, dtype=dtype
    ))[..., :headdim]
    dout = (0.05 * torch.randn(
        streams, seqlen, batch, nheads, headdim + 7, device=device, dtype=dtype
    ))[..., :headdim]
    z = torch.randn(
        streams, seqlen, batch, nheads, headdim + 7, device=device, dtype=dtype
    )[..., :headdim]
    raw_dt = torch.randn(
        streams, seqlen, batch, nheads + 2, device=device, dtype=dtype
    )[..., :nheads]
    raw_dt[0, 0, 0, 0] = 25.0
    raw_dt[1, 0, 0, 0] = -25.0
    A_log = -torch.rand(nheads, device=device, dtype=torch.float32)
    A = -torch.exp(A_log)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)
    D = 0.1 * torch.randn(nheads, device=device, dtype=torch.float32)
    initial_states = 0.05 * torch.randn(
        streams, batch, nheads, headdim, dstate, device=device, dtype=torch.float32
    )
    final_dstates = 0.05 * torch.randn_like(initial_states)

    dA_cs, dA_cs_rev, segsum, dt_out, _ = compute_dacs_segsum_triton(
        raw_dt, A, dt_bias, chunk_size
    )
    _, _, actual_chunk_states, processed_ddt, dA_chunk = mamba2_bwd_dq_cp_triton(
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
    _, _, _, processed_ddt, dA_chunk = mamba2_bwd_dk_dv_cp_triton(
        C,
        B,
        x,
        dout,
        z,
        dt_out,
        dA_cs,
        dA_cs_rev,
        actual_chunk_states,
        final_dstates,
        A,
        D,
        processed_ddt,
        dA_chunk,
        chunk_size=chunk_size,
    )
    actual_raw, actual_A_log, actual_bias = mamba2_bwd_decay_map_cp_triton(
        raw_dt,
        processed_ddt,
        dA_chunk,
        A,
        dt_bias,
        chunk_size=chunk_size,
    )
    expected_raw, expected_A_log, expected_bias = _raw_dt_autograd_reference(
        C,
        B,
        x,
        dout,
        z,
        raw_dt,
        A_log,
        dt_bias,
        initial_states,
        final_dstates,
        D,
    )

    assert not raw_dt.is_contiguous()
    assert actual_raw.shape == raw_dt.shape
    assert actual_raw.dtype is dtype
    assert actual_A_log.dtype is torch.float32
    assert actual_bias.dtype is dt_bias.dtype
    torch.testing.assert_close(actual_raw.float(), expected_raw, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(actual_A_log, expected_A_log, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(actual_bias, expected_bias, atol=1e-1, rtol=1e-1)
