# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm import mamba_true_context_parallel_v2 as true_cp_v2
from megatron.core.ssm import mamba_true_context_parallel_v3 as true_cp_v3


def test_payload_layout_is_dense_and_matches_contract():
    heads, headdim, state, conv_dim = 4, 8, 6, 56
    layout = true_cp_v3._payload_layout(heads, headdim, state, conv_dim)
    assert layout.body_log_alpha == 0
    assert layout.body_beta == heads
    assert layout.prefix_xbc == heads + heads * headdim * state
    assert layout.tail_xbc == layout.prefix_xbc + 3 * conv_dim
    assert layout.prefix_dt == layout.tail_xbc + 3 * conv_dim
    assert layout.width == heads + heads * headdim * state + 6 * conv_dim + 3 * heads


def test_backward_payload_layout_is_dense_and_matches_contract():
    heads, headdim, state, conv_dim = 4, 8, 6, 56
    layout = true_cp_v3._backward_payload_layout(heads, headdim, state, conv_dim)
    assert layout.body_log_alpha == 0
    assert layout.body_gamma == heads
    assert layout.prefix_xbc == heads + heads * headdim * state
    assert layout.tail_xbc == layout.prefix_xbc + 3 * conv_dim
    assert layout.prefix_dt == layout.tail_xbc + 3 * conv_dim
    assert layout.prefix_dpre == layout.prefix_dt + 3 * heads
    assert layout.width == (
        heads + heads * headdim * state + 6 * conv_dim + 3 * heads + 3 * heads * headdim
    )


def test_v3_spec_does_not_mutate_production_spec():
    from megatron.core.models.mamba.mamba_layer_specs import (
        mamba_stack_spec as production,
    )
    from megatron.core.models.mamba.mamba_layer_specs_true_cp_v3 import (
        mamba_stack_spec as v3_spec,
    )
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.ssm.mamba_mixer_true_cp_v3 import MambaMixerTrueCPV3

    assert production.submodules.mamba_layer.submodules.mixer.module is MambaMixer
    assert v3_spec.submodules.mamba_layer.submodules.mixer.module is MambaMixerTrueCPV3
    assert v3_spec is not production


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_body_convolution_matches_v2_after_prefix(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton convolution kernels")
    device = torch.device("cuda")
    torch.manual_seed(17)
    streams, segment_len, batch = 2, 11, 2
    heads, headdim, groups, state = 4, 8, 2, 6
    d_inner = heads * headdim
    conv_dim = d_inner + 2 * groups * state
    # Slice a larger allocation to exercise storage offsets and actual strides.
    storage = torch.randn(
        streams, segment_len + 2, batch, conv_dim + 3, device=device, dtype=dtype
    )
    xbc = storage[:, 1 : segment_len + 1, :, 1 : conv_dim + 1]
    weight = torch.randn(conv_dim, 1, 4, device=device, dtype=dtype) * 0.1
    bias = torch.randn(conv_dim, device=device, dtype=dtype) * 0.1
    halos = torch.randn(streams, 3, batch, conv_dim, device=device, dtype=dtype)

    full_x, full_B, full_C = true_cp_v2._causal_conv_format_streams_triton(
        xbc,
        halos,
        weight,
        bias,
        d_inner_local_tp=d_inner,
        ngroups_local_tp=groups,
        d_state=state,
        headdim=headdim,
    )
    body_x, body_B, body_C = true_cp_v3._causal_conv_body_width4_triton(
        xbc,
        weight,
        bias,
        d_inner_local_tp=d_inner,
        ngroups_local_tp=groups,
        d_state=state,
        headdim=headdim,
    )
    tolerance = 1e-6 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(body_x, full_x[:, 3:], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(body_B, full_B[:, 3:], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(body_C, full_C[:, 3:], atol=tolerance, rtol=tolerance)


def test_payload_packer_preserves_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton payload packer")
    device = torch.device("cuda")
    streams, segment_len, batch = 2, 7, 2
    heads, headdim, state, conv_dim = 3, 4, 5, 11
    log = torch.randn(streams, batch, heads, device=device)
    beta = torch.randn(streams, batch, heads, headdim, state, device=device)
    xbc = torch.randn(streams, segment_len, batch, conv_dim, device=device)
    dt = torch.randn(streams, segment_len, batch, heads, device=device)
    layout = true_cp_v3._payload_layout(heads, headdim, state, conv_dim)
    packed = true_cp_v3._pack_forward_payload_triton(log, beta, xbc, dt, layout)

    torch.testing.assert_close(
        packed[..., layout.body_log_alpha : layout.body_beta], log
    )
    torch.testing.assert_close(
        packed[..., layout.body_beta : layout.prefix_xbc], beta.flatten(2)
    )
    expected_prefix = xbc[:, :3].permute(0, 2, 1, 3).flatten(2)
    expected_tail = xbc[:, -3:].permute(0, 2, 1, 3).flatten(2)
    expected_dt = dt[:, :3].permute(0, 2, 1, 3).flatten(2)
    torch.testing.assert_close(
        packed[..., layout.prefix_xbc : layout.tail_xbc], expected_prefix
    )
    torch.testing.assert_close(
        packed[..., layout.tail_xbc : layout.prefix_dt], expected_tail
    )
    torch.testing.assert_close(packed[..., layout.prefix_dt :], expected_dt)


def test_backward_payload_packer_preserves_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the v3 backward payload")
    device = torch.device("cuda")
    streams, segment_len, batch = 2, 7, 2
    heads, headdim, state, conv_dim = 3, 4, 5, 17
    body_log = torch.randn(streams, batch, heads, device=device)
    body_gamma = torch.randn(
        streams, batch, heads, headdim, state, device=device
    )
    xbc = torch.randn(streams, segment_len, batch, conv_dim, device=device)
    dt = torch.randn(streams, segment_len, batch, heads, device=device)
    dout = torch.randn(
        streams, segment_len, batch, heads, headdim, device=device
    )
    z = torch.randn_like(dout)
    layout = true_cp_v3._backward_payload_layout(
        heads, headdim, state, conv_dim
    )
    packed = true_cp_v3._pack_backward_payload(
        body_log, body_gamma, xbc, dt, dout, z, layout
    )

    torch.testing.assert_close(
        packed[..., layout.body_log_alpha : layout.body_gamma], body_log
    )
    torch.testing.assert_close(
        packed[..., layout.body_gamma : layout.prefix_xbc], body_gamma.flatten(2)
    )
    torch.testing.assert_close(
        packed[..., layout.prefix_xbc : layout.tail_xbc],
        xbc[:, :3].permute(0, 2, 1, 3).flatten(2),
    )
    torch.testing.assert_close(
        packed[..., layout.tail_xbc : layout.prefix_dt],
        xbc[:, -3:].permute(0, 2, 1, 3).flatten(2),
    )
    torch.testing.assert_close(
        packed[..., layout.prefix_dt : layout.prefix_dpre],
        dt[:, :3].permute(0, 2, 1, 3).flatten(2),
    )
    expected_dpre = dout[:, :3].float() * z[:, :3].float() * torch.sigmoid(
        z[:, :3].float()
    )
    torch.testing.assert_close(
        packed[..., layout.prefix_dpre :],
        expected_dpre.permute(0, 2, 1, 3, 4).flatten(2),
    )


def test_full_replay_successor_halo_matches_autograd_reference():
    """The replayed successor prefix must own exactly the predecessor-tail gradient."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton successor replay")
    device = torch.device("cuda")
    torch.manual_seed(29)
    segments, batch = 4, 2
    heads, headdim, groups, state = 4, 2, 2, 3
    d_inner = heads * headdim
    conv_dim = d_inner + 2 * groups * state
    prefix = torch.randn(segments, batch, 3, conv_dim, device=device) * 0.2
    tail = torch.randn_like(prefix) * 0.2
    raw_dt = torch.randn(segments, batch, 3, heads, device=device) * 0.2
    dpre = torch.randn(segments, batch, 3, heads, headdim, device=device) * 0.2
    conv_weight = torch.randn(conv_dim, 1, 4, device=device) * 0.2
    conv_bias = torch.randn(conv_dim, device=device) * 0.1
    A = -torch.exp(torch.randn(heads, device=device) * 0.1)
    D = torch.randn(heads, device=device) * 0.2
    dt_bias = torch.randn(heads, device=device) * 0.1
    predecessor_final = (
        torch.randn(2, batch, heads, headdim, state, device=device) * 0.2
    )
    successor_terminal = torch.randn_like(predecessor_final) * 0.2
    layout = true_cp_v3._backward_payload_layout(
        heads, headdim, state, conv_dim
    )
    causal_payload = torch.zeros(
        segments, batch, layout.width, device=device, dtype=torch.float32
    )
    causal_payload[..., layout.prefix_xbc : layout.tail_xbc] = prefix.flatten(2)
    causal_payload[..., layout.tail_xbc : layout.prefix_dt] = tail.flatten(2)
    causal_payload[..., layout.prefix_dt : layout.prefix_dpre] = raw_dt.flatten(2)
    causal_payload[..., layout.prefix_dpre :] = dpre.flatten(2)
    # Global causal order for cp_size=2 is r0s0, r1s0, r1s1, r0s1.
    gathered = torch.stack(
        (
            torch.stack((causal_payload[0], causal_payload[3]), dim=0),
            torch.stack((causal_payload[1], causal_payload[2]), dim=0),
        ),
        dim=0,
    ).contiguous()
    actual = true_cp_v3._successor_halos_triton(
        gathered,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        D,
        prefix,
        predecessor_final,
        successor_terminal,
        layout,
        cp_rank=0,
        cp_size=2,
        d_inner=d_inner,
        ngroups=groups,
        d_state=state,
        headdim=headdim,
    )

    # Rank slot 0 owns segment 0, so replay segment 1 with segment 0's raw tail
    # as differentiable convolution context.  The scalar objective supplies the
    # same local output adjoint and resolved state-after-prefix adjoint as v3.
    tail_variable = tail[0].detach().clone().requires_grad_(True)
    context = torch.cat((tail_variable, prefix[1]), dim=1)
    preacts = []
    for token in range(3):
        value = conv_bias.clone()
        for tap in range(4):
            value = value + context[:, token + tap] * conv_weight[:, 0, tap]
        preacts.append(value)
    post = torch.stack(preacts, dim=1)
    post = post * torch.sigmoid(post)
    x = post[..., :d_inner].view(batch, 3, heads, headdim)
    qk_dim = groups * state
    B = post[..., d_inner : d_inner + qk_dim].view(batch, 3, groups, state)
    C = post[..., d_inner + qk_dim :].view(batch, 3, groups, state)
    head_to_group = torch.arange(heads, device=device) // (heads // groups)
    biased_dt = raw_dt[1] + dt_bias.view(1, 1, heads)
    dt = torch.where(
        biased_dt <= 20.0, torch.nn.functional.softplus(biased_dt), biased_dt
    )
    decay = torch.exp(dt * A.view(1, 1, heads))
    replay_state = predecessor_final[0]
    objective = torch.zeros((), device=device)
    for token in range(3):
        b = B[:, token].index_select(1, head_to_group)
        c = C[:, token].index_select(1, head_to_group)
        replay_state = decay[:, token][..., None, None] * replay_state
        replay_state = replay_state + dt[:, token][..., None, None] * (
            x[:, token][..., :, None] * b[..., None, :]
        )
        pre_gate = (replay_state * c[..., None, :]).sum(dim=-1)
        pre_gate = pre_gate + D.view(1, heads, 1) * x[:, token]
        objective = objective + (pre_gate * dpre[1, :, token]).sum()
    objective = objective + (replay_state * successor_terminal[0]).sum()
    objective.backward()

    torch.testing.assert_close(
        actual[0], tail_variable.grad.permute(1, 0, 2), atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))


def test_public_entrypoint_caches_boundaries_only_with_grad_enabled(monkeypatch):
    class FakeGroup:
        @staticmethod
        def size():
            return 2

    cache_flags = []

    def fake_apply(*args):
        cache_flags.append(args[-1])
        return args[0][..., :8]

    monkeypatch.setattr(
        true_cp_v3._MambaTrueContextParallelV3Fn,
        "apply",
        staticmethod(fake_apply),
    )
    segment_len, batch = 3, 1
    d_inner, headdim, groups, state = 8, 4, 1, 3
    heads = d_inner // headdim
    conv_dim = d_inner + 2 * groups * state
    projected_dim = 2 * d_inner + 2 * groups * state + heads
    tensors = (
        torch.randn(2 * segment_len, batch, projected_dim),
        torch.randn(conv_dim, 1, 4),
        torch.randn(conv_dim),
        torch.randn(heads),
        torch.randn(heads),
        torch.randn(heads),
    )

    def invoke():
        return true_cp_v3.mamba_true_context_parallel_v3_training(
            *tensors,
            cp_group=FakeGroup(),
            d_inner_local_tp=d_inner,
            ngroups_local_tp=groups,
            d_state=state,
            headdim=headdim,
            chunk_size=1,
            activation="silu",
            D_has_hdim=False,
            bwd_subsequence_size=0,
        )

    with torch.enable_grad():
        invoke()
    with torch.no_grad():
        invoke()
    assert cache_flags == [True, False]
