# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.mamba2_utils.mamba2_fwd import (
    _mamba2_get_final_state_kernel,
    compute_dacs_compact_triton,
    compute_dacs_segsum_triton,
    mamba2_fwd_kernel,
    mamba2_fwd_triton,
    mamba2_get_final_state_triton,
)


def _fold_streams(tensor):
    return tensor.permute(0, 2, 1, 3).reshape(
        tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]
    )


def _decay_reference(dt_streams, a, dt_bias, chunk_size):
    dt_input = _fold_streams(dt_streams)
    dt = F.softplus(dt_input.float() + dt_bias.float()[None, None, :])
    folded_batch, seqlen, nheads = dt.shape
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    chunk_shape = (folded_batch, nheads, nchunks, chunk_size)
    dt_out = torch.zeros(chunk_shape, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.zeros_like(dt_out)
    dA_cs_rev = torch.zeros_like(dt_out)
    segsum = torch.full(
        (folded_batch, nheads, nchunks, chunk_size, chunk_size),
        -torch.inf,
        device=dt.device,
        dtype=torch.float32,
    )
    for chunk_idx, chunk_start in enumerate(range(0, seqlen, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, seqlen)
        chunk_len = chunk_end - chunk_start
        chunk_dt = dt[:, chunk_start:chunk_end].transpose(1, 2)
        chunk_dA = chunk_dt * a.float()[None, :, None]
        chunk_cs = torch.cumsum(chunk_dA, dim=-1)
        dt_out[:, :, chunk_idx, :chunk_len] = chunk_dt
        dA_cumsum[:, :, chunk_idx] = chunk_cs[..., -1:]
        dA_cumsum[:, :, chunk_idx, :chunk_len] = chunk_cs
        dA_cs_rev[:, :, chunk_idx, :chunk_len] = chunk_cs[..., -1:] - chunk_cs
        row = chunk_cs[..., :, None]
        col = chunk_cs[..., None, :]
        causal = torch.tril(
            torch.ones(chunk_len, chunk_len, device=dt.device, dtype=torch.bool),
        )
        segsum[:, :, chunk_idx, :chunk_len, :chunk_len] = torch.where(
            causal, row - col, -torch.inf
        )
    return dA_cumsum, dA_cs_rev, segsum, dt_out, dt_input


def _final_state_reference(k, v, a, dt):
    """Sequential FP32 reference for the zero-initialized state recurrence."""
    batch, seqlen, nheads_qk, headdim_qk = k.shape
    nheads, headdim_v = v.shape[2:]
    heads_per_qk = nheads // nheads_qk
    qk_head = torch.arange(nheads, device=k.device) // heads_per_qk
    k = k.index_select(2, qk_head).float()
    v = v.float()
    a = a.float()
    dt = dt.float()

    state = torch.zeros(
        batch, nheads, headdim_qk, headdim_v, device=k.device, dtype=torch.float32
    )
    for position in range(seqlen):
        dt_t = dt[:, position]
        decay = torch.exp(dt_t * a[None, :])[:, :, None, None]
        update = torch.einsum(
            "bhn,bhp->bhnp", k[:, position], v[:, position]
        ) * dt_t[:, :, None, None]
        state = decay * state + update
    return state


def _forward_chunk_reference(
    q,
    k,
    v,
    dt,
    dA_cs,
    dA_cs_rev,
    initial_states,
    D,
    z,
    chunk_size,
):
    streams, seqlen, batch, nheads, headdim_v = z.shape
    _, _, nheads_qk, _ = q.shape
    heads_per_qk = nheads // nheads_qk
    out = torch.empty_like(z, dtype=torch.float32)
    for stream in range(streams):
        for batch_idx in range(batch):
            folded_batch = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                state = initial_states[stream, batch_idx, head].transpose(0, 1).float()
                for chunk_idx, chunk_start in enumerate(range(0, seqlen, chunk_size)):
                    chunk_end = min(chunk_start + chunk_size, seqlen)
                    chunk_len = chunk_end - chunk_start
                    q_block = q[folded_batch, chunk_start:chunk_end, qk_head].float()
                    k_block = k[folded_batch, chunk_start:chunk_end, qk_head].float()
                    v_block = v[folded_batch, chunk_start:chunk_end, head].float()
                    dt_block = dt[folded_batch, head, chunk_idx, :chunk_len].float()
                    dA_block = dA_cs[folded_batch, head, chunk_idx, :chunk_len].float()
                    dA_rev_block = dA_cs_rev[
                        folded_batch, head, chunk_idx, :chunk_len
                    ].float()
                    row = dA_block[:, None]
                    col = dA_block[None, :]
                    causal = torch.tril(
                        torch.ones(
                            chunk_len,
                            chunk_len,
                            device=dA_block.device,
                            dtype=torch.bool,
                        )
                    )
                    segsum_block = torch.where(causal, row - col, -torch.inf)
                    segsum_block = torch.minimum(
                        segsum_block, torch.zeros_like(segsum_block)
                    )

                    block_out = (q_block @ state) * torch.exp(dA_block)[:, None]
                    block_out += (
                        (q_block @ k_block.transpose(0, 1))
                        * torch.exp(segsum_block)
                        * dt_block[None, :]
                    ) @ v_block
                    block_out += D[head].float() * v_block
                    z_block = z[stream, chunk_start:chunk_end, batch_idx, head].float()
                    block_out *= F.silu(z_block)
                    out[stream, chunk_start:chunk_end, batch_idx, head] = block_out

                    dA_last = dA_cs[folded_batch, head, chunk_idx, -1].float()
                    state = torch.exp(dA_last) * state + (
                        k_block * (torch.exp(dA_rev_block) * dt_block)[:, None]
                    ).transpose(0, 1) @ v_block
    return out


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads_qk", "nheads", "headdim_qk", "headdim_v"),
    (
        (70, 64, 2, 4, 13, 11),
        (35, 17, 1, 1, 16, 16),
    ),
)
def test_mamba2_get_final_state_matches_reference(
    dtype, seqlen, chunk_size, nheads_qk, nheads, headdim_qk, headdim_v
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton final-state kernel")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    streams = 2
    batch = 2
    dt_storage = torch.randn(
        streams, seqlen, batch, nheads + 3, device=device, dtype=dtype
    )
    dt_streams = dt_storage[..., :nheads]
    dt_streams[1].add_(1.5)
    assert not dt_streams.is_contiguous()
    k = 0.2 * torch.randn(
        streams * batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=dtype
    )
    v = 0.2 * torch.randn(
        streams * batch, seqlen, nheads, headdim_v, device=device, dtype=dtype
    )
    a = -torch.rand(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)

    expected_decay = _decay_reference(dt_streams, a, dt_bias, chunk_size)
    actual_decay = compute_dacs_segsum_triton(
        dt_streams, a, dt_bias, chunk_size, return_dt_input=True
    )
    for actual, expected_decay_tensor in zip(actual_decay, expected_decay):
        torch.testing.assert_close(actual, expected_decay_tensor, atol=2e-5, rtol=2e-5)
    actual_segsum = actual_decay[2]
    row = torch.arange(chunk_size, device=device)[:, None]
    col = torch.arange(chunk_size, device=device)[None, :]
    for chunk_idx, chunk_start in enumerate(range(0, seqlen, chunk_size)):
        chunk_len = min(chunk_size, seqlen - chunk_start)
        valid_causal = (row < chunk_len) & (col < chunk_len) & (row >= col)
        assert torch.isneginf(actual_segsum[:, :, chunk_idx, ~valid_causal]).all()
        diagonal = torch.diagonal(actual_segsum[:, :, chunk_idx], dim1=-2, dim2=-1)
        torch.testing.assert_close(
            diagonal[..., :chunk_len], torch.zeros_like(diagonal[..., :chunk_len])
        )
    decay_without_input = compute_dacs_segsum_triton(
        dt_streams, a, dt_bias, chunk_size, return_dt_input=False
    )
    assert decay_without_input[-1] is None
    for actual, expected_decay_tensor in zip(decay_without_input[:-1], expected_decay[:-1]):
        torch.testing.assert_close(actual, expected_decay_tensor, atol=2e-5, rtol=2e-5)

    dA_cs, dA_cs_rev, _, dt_out, dt_input = actual_decay
    expected_state = _final_state_reference(
        k, v, a, F.softplus(dt_input.float() + dt_bias[None, None, :])
    )
    actual_state, actual_log_alpha = mamba2_get_final_state_triton(
        k, v, dt_out, dA_cs, dA_cs_rev, chunk_size=chunk_size
    )
    expected_log_alpha = (
        F.softplus(dt_input.float() + dt_bias[None, None, :]) * a[None, None, :]
    ).sum(dim=1)

    assert actual_state.shape == expected_state.transpose(-1, -2).shape
    assert actual_state.dtype is torch.float32
    assert actual_state.device == k.device
    torch.testing.assert_close(
        actual_state, expected_state.transpose(-1, -2), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(actual_log_alpha, expected_log_alpha, atol=2e-5, rtol=2e-5)


def test_final_state_kernel_does_not_replace_full_forward_kernel():
    assert mamba2_fwd_kernel is not _mamba2_get_final_state_kernel


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads_qk", "nheads", "headdim_qk", "headdim_v"),
    (
        (35, 17, 1, 2, 13, 11),
        (70, 64, 2, 4, 16, 16),
        (269, 256, 1, 2, 128, 64),
    ),
)
def test_mamba2_fwd_uses_prepared_decay_and_stream_layout(
    dtype, seqlen, chunk_size, nheads_qk, nheads, headdim_qk, headdim_v
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton forward kernel")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(5678)
    device = torch.device("cuda")
    streams = 2
    batch = 2
    folded_batch = streams * batch
    q = 0.15 * torch.randn(
        folded_batch, seqlen, nheads_qk, headdim_qk, device=device, dtype=dtype
    )
    k = 0.15 * torch.randn_like(q)
    v = 0.15 * torch.randn(
        folded_batch, seqlen, nheads, headdim_v, device=device, dtype=dtype
    )
    dt_storage = torch.randn(
        streams, seqlen, batch, nheads + 2, device=device, dtype=dtype
    )
    dt_streams = dt_storage[..., :nheads]
    dt_streams[1].add_(0.75)
    assert not dt_streams.is_contiguous()
    a = -torch.rand(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_out, dA_cs, dA_cs_rev = compute_dacs_compact_triton(
        dt_streams, a, dt_bias, chunk_size
    )

    # Make the prepared values deliberately inconsistent with a re-derived
    # decay. The reference and kernel must consume these exact tensors.
    dA_cs = dA_cs.clone()
    dA_cs_rev = dA_cs_rev.clone()
    for chunk_idx, chunk_start in enumerate(range(0, seqlen, chunk_size)):
        chunk_len = min(chunk_size, seqlen - chunk_start)
        dA_cs_rev[:, :, chunk_idx, 0] -= 0.125
        if chunk_len > 1:
            dA_cs[:, :, chunk_idx, 1] -= 0.25

    initial_states = 0.1 * torch.randn(
        streams,
        batch,
        nheads,
        headdim_v,
        headdim_qk,
        device=device,
        dtype=torch.float32,
    )
    initial_states[1].add_(0.2)
    z_storage = torch.randn(
        streams,
        seqlen,
        batch,
        nheads,
        headdim_v + 3,
        device=device,
        dtype=dtype,
    )
    z = z_storage[..., :headdim_v]
    z[1].add_(0.5)
    assert not z.is_contiguous()
    D = torch.randn(nheads, device=device, dtype=torch.float32)

    expected = _forward_chunk_reference(
        q,
        k,
        v,
        dt_out,
        dA_cs,
        dA_cs_rev,
        initial_states,
        D,
        z,
        chunk_size,
    )
    actual = mamba2_fwd_triton(
        q,
        k,
        v,
        dt_out,
        dA_cs,
        dA_cs_rev,
        initial_states,
        D,
        z,
        chunk_size=chunk_size,
    )
    assert actual.shape == (streams, seqlen, batch, nheads, headdim_v)
    assert actual.is_contiguous()
    public_view = actual.view(streams * seqlen, batch, nheads * headdim_v)
    assert public_view.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()
    torch.testing.assert_close(actual.float(), expected, atol=7e-2, rtol=7e-2)

    out_storage = torch.empty(
        streams,
        seqlen,
        batch,
        nheads,
        headdim_v + 5,
        device=device,
        dtype=dtype,
    )
    strided_out = out_storage[..., :headdim_v]
    returned_out = mamba2_fwd_triton(
        q,
        k,
        v,
        dt_out,
        dA_cs,
        dA_cs_rev,
        initial_states,
        D,
        z,
        out=strided_out,
        chunk_size=chunk_size,
    )
    assert returned_out.data_ptr() == strided_out.data_ptr()
    torch.testing.assert_close(returned_out, actual, atol=0.0, rtol=0.0)
