# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.mamba2_utils.mamba2_bwd import (
    mamba2_get_initial_dstate_triton,
)
from megatron.core.ssm.mamba2_utils.mamba2_fwd import compute_dacs_segsum_triton


def _initial_dstate_reference(C, dout, z, dA_cs, chunk_size, terminal_dstates=None):
    streams, seqlen, batch, nheads, headdim_v = dout.shape
    folded_batch, _, nheads_qk, headdim_qk = C.shape
    assert folded_batch == streams * batch
    heads_per_qk = nheads // nheads_qk
    gamma = torch.empty(
        folded_batch,
        nheads,
        headdim_v,
        headdim_qk,
        device=C.device,
        dtype=torch.float32,
    )
    for stream in range(streams):
        for batch_idx in range(batch):
            folded_idx = stream * batch + batch_idx
            for head in range(nheads):
                qk_head = head // heads_per_qk
                if terminal_dstates is None:
                    acc = torch.zeros(
                        headdim_qk,
                        headdim_v,
                        device=C.device,
                        dtype=torch.float32,
                    )
                else:
                    acc = terminal_dstates[folded_idx, head].transpose(0, 1).float()
                for chunk_idx in reversed(range(dA_cs.shape[2])):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, seqlen)
                    chunk_len = chunk_end - chunk_start
                    c_block = C[folded_idx, chunk_start:chunk_end, qk_head].float()
                    dout_block = dout[
                        stream, chunk_start:chunk_end, batch_idx, head
                    ].float()
                    z_block = z[stream, chunk_start:chunk_end, batch_idx, head].float()
                    dpre = dout_block * F.silu(z_block)
                    dA_block = dA_cs[folded_idx, head, chunk_idx, :chunk_len].float()
                    gamma_chunk = c_block.transpose(0, 1) @ (
                        dpre * torch.exp(dA_block)[:, None]
                    )
                    chunk_log_alpha = dA_cs[folded_idx, head, chunk_idx, -1].float()
                    acc = torch.exp(chunk_log_alpha) * acc + gamma_chunk
                gamma[folded_idx, head] = acc.transpose(0, 1)
    return gamma


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads_qk", "nheads", "headdim_qk", "headdim_v"),
    (
        (35, 17, 1, 2, 13, 11),
        (70, 64, 2, 4, 16, 16),
    ),
)
def test_mamba2_get_initial_dstate_matches_adjoint_summary(
    dtype, seqlen, chunk_size, nheads_qk, nheads, headdim_qk, headdim_v
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton adjoint-summary kernel")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this CUDA device")

    torch.manual_seed(2468)
    device = torch.device("cuda")
    streams = 2
    batch = 2
    folded_batch = streams * batch

    c_storage = 0.15 * torch.randn(
        folded_batch,
        seqlen,
        nheads_qk,
        headdim_qk + 3,
        device=device,
        dtype=dtype,
    )
    C = c_storage[..., :headdim_qk]
    dout_storage = 0.1 * torch.randn(
        streams,
        seqlen,
        batch,
        nheads,
        headdim_v + 5,
        device=device,
        dtype=dtype,
    )
    z_storage = torch.randn_like(dout_storage)
    dout = dout_storage[..., :headdim_v]
    z = z_storage[..., :headdim_v]
    dout[1].add_(0.25)
    z[1].sub_(0.5)
    assert not C.is_contiguous()
    assert not dout.is_contiguous()
    assert not z.is_contiguous()

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
    dA_cs, _, _, _, _ = compute_dacs_segsum_triton(
        dt_streams,
        A,
        dt_bias,
        chunk_size,
    )

    expected_gamma = _initial_dstate_reference(C, dout, z, dA_cs, chunk_size)
    actual_gamma = mamba2_get_initial_dstate_triton(
        C,
        dout,
        z,
        dA_cs,
        chunk_size=chunk_size,
    )
    assert actual_gamma.shape == expected_gamma.shape
    assert actual_gamma.dtype is torch.float32
    assert actual_gamma.device == C.device
    torch.testing.assert_close(actual_gamma, expected_gamma, atol=5e-2, rtol=5e-2)

    terminal_dstates = 0.1 * torch.randn_like(actual_gamma)
    expected_with_terminal = _initial_dstate_reference(
        C,
        dout,
        z,
        dA_cs,
        chunk_size,
        terminal_dstates=terminal_dstates,
    )
    log_alpha = dA_cs[..., -1].sum(dim=2)
    composed_with_terminal = (
        torch.exp(log_alpha)[..., None, None] * terminal_dstates + actual_gamma
    )
    torch.testing.assert_close(
        composed_with_terminal,
        expected_with_terminal,
        atol=5e-2,
        rtol=5e-2,
    )
