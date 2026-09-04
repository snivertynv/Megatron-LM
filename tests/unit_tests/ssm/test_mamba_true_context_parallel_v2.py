# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm import mamba_true_context_parallel_v2 as true_cp_v2


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _FakeContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def _sequential_boundaries(log_alpha, beta, gamma):
    initial_states = torch.empty_like(beta, dtype=torch.float32)
    state = torch.zeros_like(beta[0], dtype=torch.float32)
    for segment in range(log_alpha.shape[0]):
        initial_states[segment] = state
        state = (
            torch.exp(log_alpha[segment].float())[..., None, None] * state
            + beta[segment].float()
        )

    final_dstates = torch.empty_like(gamma, dtype=torch.float32)
    dstate = torch.zeros_like(gamma[0], dtype=torch.float32)
    for segment in reversed(range(log_alpha.shape[0])):
        final_dstates[segment] = dstate
        dstate = (
            torch.exp(log_alpha[segment].float())[..., None, None] * dstate
            + gamma[segment].float()
        )
    return initial_states, final_dstates


def test_local_segment_boundary_states_match_sequential_affine_chains(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Mamba's affine state-passing kernel")

    device = torch.device("cuda")
    cp_size = 3
    num_segments = 2 * cp_size
    batch, nheads, headdim, dstate = 2, 2, 3, 2

    values = torch.arange(
        num_segments * batch * nheads,
        device=device,
        dtype=torch.float32,
    ).reshape(num_segments, batch, nheads)
    global_log_alpha = -0.01 * (values + 1.0)
    global_beta = 0.02 * torch.arange(
        num_segments * batch * nheads * headdim * dstate,
        device=device,
        dtype=torch.float32,
    ).reshape(num_segments, batch, nheads, headdim, dstate)
    global_gamma = torch.flip(global_beta, dims=(0,)) * 0.75 + 0.125

    state_numel = headdim * dstate
    global_packed = torch.cat(
        [
            global_log_alpha.unsqueeze(-1),
            global_beta.reshape(num_segments, batch, nheads, state_numel),
            global_gamma.reshape(num_segments, batch, nheads, state_numel),
        ],
        dim=-1,
    )
    rank_slot_ids = [
        torch.tensor([rank, num_segments - rank - 1], device=device)
        for rank in range(cp_size)
    ]
    packed_by_rank_slot = torch.stack(
        [global_packed.index_select(0, segment_ids) for segment_ids in rank_slot_ids]
    ).contiguous()
    expected_initial, expected_final = _sequential_boundaries(
        global_log_alpha, global_beta, global_gamma
    )
    gather_calls = [0] * cp_size

    def fake_all_gather(tensor, cp_group):
        rank = cp_group.rank()
        gather_calls[rank] += 1
        assert tensor.dtype is torch.float32
        assert tensor.is_contiguous()
        assert tensor.shape == (2, batch, nheads, 1 + 2 * state_numel)
        torch.testing.assert_close(tensor, packed_by_rank_slot[rank])
        return packed_by_rank_slot

    monkeypatch.setattr(true_cp_v2, "_all_gather_rank_tensor", fake_all_gather)

    for rank in range(cp_size):
        segment_ids = rank_slot_ids[rank]
        local_initial, local_final = true_cp_v2._local_segment_boundary_states(
            global_log_alpha.index_select(0, segment_ids),
            global_beta.index_select(0, segment_ids),
            global_gamma.index_select(0, segment_ids),
            _FakeGroup(rank, cp_size),
        )
        assert local_initial.shape == (2, batch, nheads, headdim, dstate)
        assert local_final.shape == local_initial.shape
        assert local_initial.dtype is torch.float32
        assert local_final.dtype is torch.float32
        assert local_initial.is_contiguous()
        assert local_final.is_contiguous()
        torch.testing.assert_close(
            local_initial, expected_initial.index_select(0, segment_ids), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            local_final, expected_final.index_select(0, segment_ids), atol=1e-5, rtol=1e-5
        )

    assert gather_calls == [1, 1, 1]


def test_checkpoint_policy_resolves_before_autograd_forward(monkeypatch):
    resolved = []
    sentinel = torch.empty(0)

    def fake_apply(*args):
        resolved.append((args[-2], args[-1]))
        return sentinel

    monkeypatch.setattr(true_cp_v2._MambaTrueContextParallelFn, "apply", staticmethod(fake_apply))
    group = _FakeGroup(rank=0, size=2)
    tensor = torch.empty(1)
    omitted = object()

    def invoke(cache_conv_halos=None, cache_initial_states=omitted):
        kwargs = {}
        if cache_initial_states is not omitted:
            kwargs["cache_initial_states"] = cache_initial_states
        return true_cp_v2.mamba_true_context_parallel_training(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            cp_group=group,
            d_inner_local_tp=1,
            ngroups_local_tp=1,
            d_state=1,
            headdim=1,
            chunk_size=1,
            activation="silu",
            D_has_hdim=False,
            cache_conv_halos=cache_conv_halos,
            **kwargs,
        )

    with torch.enable_grad():
        assert invoke() is sentinel
        assert invoke(False) is sentinel
        assert invoke(cache_initial_states=False) is sentinel
    with torch.no_grad():
        assert invoke() is sentinel
        assert invoke(True) is sentinel
        assert invoke(cache_initial_states=False) is sentinel
    with torch.inference_mode():
        assert invoke() is sentinel
        assert invoke(cache_initial_states=False) is sentinel

    assert resolved == [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
        (True, False),
        (False, False),
        (False, False),
        (False, False),
    ]


@pytest.mark.parametrize("save_initial_states", (False, True))
@pytest.mark.parametrize("save_conv_halos", (False, True))
def test_autograd_forward_saves_only_requested_conv_halos(
    monkeypatch, save_conv_halos, save_initial_states
):
    local_seq, batch, d_inner, conv_dim = 6, 2, 4, 10
    zxBCdt = torch.randn(local_seq, batch, 1)
    parameter = torch.randn(1)
    conv_halos = torch.randn(2, 3, batch, conv_dim)
    initial_states = torch.randn(2, batch, 1, 4, 1)
    global_log_alpha = torch.randn(4, batch, 1)
    seen_return_flags = []

    monkeypatch.setattr(true_cp_v2, "_check_inputs", lambda *args, **kwargs: None)

    def fake_prepare(*args, **kwargs):
        seen_return_flags.append(
            (kwargs["return_conv_halos"], kwargs["return_initial_states"])
        )
        result = {"out": torch.randn(local_seq, batch, d_inner)}
        if kwargs["return_conv_halos"]:
            result["xBC_conv_halos"] = conv_halos
        if kwargs["return_initial_states"]:
            result["local_initial_states"] = initial_states
            result["global_log_alpha"] = global_log_alpha
        return result

    monkeypatch.setattr(true_cp_v2, "_prepare_folded_forward", fake_prepare)
    ctx = _FakeContext()
    true_cp_v2._MambaTrueContextParallelFn.forward(
        ctx,
        zxBCdt,
        parameter,
        parameter,
        parameter,
        parameter,
        parameter,
        _FakeGroup(rank=0, size=2),
        d_inner,
        1,
        1,
        1,
        1,
        64,
        save_conv_halos,
        save_initial_states,
    )

    assert seen_return_flags == [(save_conv_halos, save_initial_states)]
    expected_saved = 6 + (1 if save_conv_halos else 0) + (2 if save_initial_states else 0)
    assert len(ctx.saved_tensors) == expected_saved
    if save_initial_states:
        assert ctx.saved_tensors[-2] is initial_states
        assert ctx.saved_tensors[-1] is global_log_alpha
    if save_conv_halos:
        assert ctx.saved_tensors[6] is conv_halos


def test_resolve_conv_halos_reuses_cache_or_exchanges(monkeypatch):
    xBC_streams = torch.randn(2, 7, 2, 5)
    cached = torch.randn(2, 3, 2, 5)
    exchanged = torch.randn_like(cached)
    calls = []

    def fake_exchange(xBC, cp_group):
        calls.append((xBC, cp_group))
        return exchanged

    monkeypatch.setattr(true_cp_v2, "_exchange_conv_halos_streams", fake_exchange)
    group = _FakeGroup(rank=0, size=2)

    assert true_cp_v2._resolve_conv_halos_streams(xBC_streams, group, cached) is cached
    assert calls == []
    assert true_cp_v2._resolve_conv_halos_streams(xBC_streams, group, None) is exchanged
    assert calls == [(xBC_streams, group)]

    with pytest.raises(ValueError, match="must have shape"):
        true_cp_v2._resolve_conv_halos_streams(xBC_streams, group, cached[:, :2])


def test_route_conv_halo_grads_updates_only_owned_stream_tails(monkeypatch):
    cp_size, batch, conv_dim = 3, 2, 5
    gathered = torch.arange(
        cp_size * 2 * 3 * batch * conv_dim, dtype=torch.float32
    ).reshape(cp_size, 2, 3, batch, conv_dim)

    monkeypatch.setattr(
        true_cp_v2,
        "_all_gather_rank_tensor",
        lambda tensor, cp_group: gathered,
    )

    # Rank 1 owns global segments 1 and 4. Their successors are segment 2
    # (rank 2, slot 0) and segment 5 (rank 0, slot 1), respectively.
    dx = torch.randn(2, 7, batch, conv_dim)
    before = dx.clone()
    result = true_cp_v2._route_conv_halo_grads(
        dx, gathered[1], _FakeGroup(rank=1, size=cp_size)
    )
    assert result is dx
    torch.testing.assert_close(dx[:, :-3], before[:, :-3])
    torch.testing.assert_close(
        dx[:, -3:], before[:, -3:] + torch.stack((gathered[2, 0], gathered[0, 1]))
    )

    # Rank 0's second stream owns the final global segment, so it has no
    # successor contribution even though its synthetic gathered slot is set.
    dx = torch.randn(2, 7, batch, conv_dim)
    before = dx.clone()
    true_cp_v2._route_conv_halo_grads(
        dx, gathered[0], _FakeGroup(rank=0, size=cp_size)
    )
    torch.testing.assert_close(dx[0, -3:], before[0, -3:] + gathered[1, 0])
    torch.testing.assert_close(dx[1], before[1])


@pytest.mark.parametrize("save_initial_states", (False, True))
@pytest.mark.parametrize("save_conv_halos", (False, True))
def test_autograd_backward_routes_cached_tensors(
    monkeypatch, save_conv_halos, save_initial_states
):
    conv_halos = torch.randn(2, 3, 2, 10)
    initial_states = torch.randn(2, 2, 1, 4, 1)
    global_log_alpha = torch.randn(4, 2, 1)
    base = tuple(torch.randn(1) for _ in range(6))

    ctx = _FakeContext()
    saved = list(base)
    if save_conv_halos:
        saved.append(conv_halos)
    if save_initial_states:
        saved.extend([initial_states, global_log_alpha])
    ctx.saved_tensors = tuple(saved)
    ctx.save_conv_halos = save_conv_halos
    ctx.save_initial_states = save_initial_states
    ctx.cp_group = _FakeGroup(rank=0, size=2)
    ctx.d_inner_local_tp = 4
    ctx.ngroups_local_tp = 1
    ctx.d_state = 1
    ctx.headdim = 1
    ctx.chunk_size = 1
    ctx.bwd_subsequence_size = 64

    captured = {}

    def fake_backward_impl(grad_output, *args, **kwargs):
        captured.update(kwargs)
        return tuple(torch.randn(1) for _ in range(6))

    monkeypatch.setattr(true_cp_v2, "_true_context_backward_impl", fake_backward_impl)
    grads = true_cp_v2._MambaTrueContextParallelFn.backward(ctx, torch.randn(1))

    assert len(grads) == 15
    assert all(grad is None for grad in grads[6:])
    assert captured["cached_conv_halos"] is (conv_halos if save_conv_halos else None)
    assert captured["cached_initial_states"] is (
        initial_states if save_initial_states else None
    )
    assert captured["cached_global_log_alpha"] is (
        global_log_alpha if save_initial_states else None
    )


def test_scan_state_flag_validation():
    tensor = torch.empty(1)
    common = dict(
        cp_group=_FakeGroup(rank=0, size=2),
        d_inner_local_tp=1,
        ngroups_local_tp=1,
        d_state=1,
        headdim=1,
        chunk_size=1,
        corrected_states_out_dtype=None,
    )

    with pytest.raises(ValueError, match="skip_segment_summary requires"):
        true_cp_v2._prepare_folded_scan_state(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            return_backward_state=True,
            skip_segment_summary=True,
            **common,
        )
    with pytest.raises(ValueError, match="return_global_log_alpha requires"):
        true_cp_v2._prepare_folded_scan_state(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            return_backward_state=True,
            defer_boundary_exchange=True,
            return_global_log_alpha=True,
            **common,
        )
    with pytest.raises(ValueError, match="must be provided together"):
        true_cp_v2._true_context_backward_impl(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            cached_initial_states=tensor,
            cached_global_log_alpha=None,
            **{key: value for key, value in common.items() if key != "corrected_states_out_dtype"},
        )


def test_subsequence_state_passing_uses_view_layouts(monkeypatch):
    folded_batch, nsubsequences, nheads, headdim, dstate = 2, 3, 4, 5, 6
    summaries = torch.randn(
        folded_batch, nsubsequences, nheads, headdim, dstate, dtype=torch.float32
    )
    log_alpha = torch.randn(
        folded_batch, nheads, nsubsequences, dtype=torch.float32
    )
    initial_states = torch.randn(
        folded_batch, nheads, headdim, dstate, dtype=torch.float32
    )
    captured = {}

    def fake_state_passing(states_in, dA, **kwargs):
        captured["states_in"] = states_in
        captured["dA"] = dA
        captured.update(kwargs)
        incoming = states_in + 1.0
        captured["incoming"] = incoming
        final = torch.empty(
            folded_batch, nheads, headdim * dstate, dtype=torch.float32
        )
        return incoming, final

    monkeypatch.setattr(true_cp_v2, "_state_passing_fwd", fake_state_passing)
    actual = true_cp_v2._subsequence_state_passing(
        summaries, log_alpha, initial_states
    )

    assert captured["states_in"].shape == (
        folded_batch,
        nsubsequences,
        nheads,
        headdim * dstate,
    )
    assert captured["states_in"].untyped_storage().data_ptr() == (
        summaries.untyped_storage().data_ptr()
    )
    assert captured["dA"] is log_alpha
    assert captured["initial_states"].shape == (
        folded_batch,
        nheads,
        headdim * dstate,
    )
    assert captured["initial_states"].untyped_storage().data_ptr() == (
        initial_states.untyped_storage().data_ptr()
    )
    assert captured["seq_idx"] is None
    assert captured["chunk_size"] == 1
    assert captured["out_dtype"] is torch.float32
    assert actual.shape == summaries.shape
    assert actual.untyped_storage().data_ptr() == (
        captured["incoming"].untyped_storage().data_ptr()
    )
    torch.testing.assert_close(actual, summaries + 1.0)


def test_local_segment_final_dstates_matches_sequential_reverse_chain(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Mamba's affine state-passing kernel")

    device = torch.device("cuda")
    cp_size = 3
    num_segments = 2 * cp_size
    batch, nheads, headdim, dstate = 2, 2, 3, 2

    values = torch.arange(
        num_segments * batch * nheads,
        device=device,
        dtype=torch.float32,
    ).reshape(num_segments, batch, nheads)
    global_log_alpha = -0.01 * (values + 1.0)
    global_beta = 0.02 * torch.arange(
        num_segments * batch * nheads * headdim * dstate,
        device=device,
        dtype=torch.float32,
    ).reshape(num_segments, batch, nheads, headdim, dstate)
    global_gamma = torch.flip(global_beta, dims=(0,)) * 0.75 + 0.125

    state_numel = headdim * dstate
    global_gamma_flat = global_gamma.reshape(num_segments, batch, nheads, state_numel)
    rank_slot_ids = [
        torch.tensor([rank, num_segments - rank - 1], device=device)
        for rank in range(cp_size)
    ]
    gamma_by_rank_slot = torch.stack(
        [global_gamma_flat.index_select(0, segment_ids) for segment_ids in rank_slot_ids]
    ).contiguous()
    _, expected_final = _sequential_boundaries(global_log_alpha, global_beta, global_gamma)
    gather_calls = [0] * cp_size

    def fake_all_gather(tensor, cp_group):
        rank = cp_group.rank()
        gather_calls[rank] += 1
        assert tensor.dtype is torch.float32
        assert tensor.shape == (2, batch, nheads, state_numel)
        torch.testing.assert_close(tensor, gamma_by_rank_slot[rank])
        return gamma_by_rank_slot

    monkeypatch.setattr(true_cp_v2, "_all_gather_rank_tensor", fake_all_gather)

    for rank in range(cp_size):
        segment_ids = rank_slot_ids[rank]
        local_final = true_cp_v2._local_segment_final_dstates(
            global_log_alpha,
            global_gamma.index_select(0, segment_ids),
            _FakeGroup(rank, cp_size),
        )
        assert local_final.shape == (2, batch, nheads, headdim, dstate)
        assert local_final.dtype is torch.float32
        assert local_final.is_contiguous()
        torch.testing.assert_close(
            local_final, expected_final.index_select(0, segment_ids), atol=1e-5, rtol=1e-5
        )

    assert gather_calls == [1, 1, 1]


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize(
    ("segment_len", "headdim", "d_state"),
    (
        (70, 16, 16),  # One complete X=64 subsection and one masked tail.
        (128, 64, 128),  # Complete subsections at the primary exact P/N tile.
        (128, 48, 96),  # Complete subsections with padded P/N predicates.
    ),
)
def test_cached_initial_states_match_recomputed_backward(
    monkeypatch, dtype, segment_len, headdim, d_state
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the true-CP Triton kernels")

    device = torch.device("cuda")
    torch.manual_seed(123)
    cp_size = 2
    batch = 2
    nheads, ngroups, chunk_size = 2, 1, 64
    d_inner = nheads * headdim
    conv_dim = d_inner + 2 * ngroups * d_state
    projected_dim = 2 * d_inner + 2 * ngroups * d_state + nheads

    # Elementwise fake exchange: the synthetic peer-rank values must agree
    # between the packed and gamma-only gathers for bitwise comparison.
    def fake_all_gather(tensor, cp_group):
        return torch.stack([tensor, 0.5 * tensor]).contiguous()

    monkeypatch.setattr(true_cp_v2, "_all_gather_rank_tensor", fake_all_gather)

    real_get_final_state = true_cp_v2.mamba2_get_final_state_triton
    final_state_calls = []

    def counting_get_final_state(*args, **kwargs):
        final_state_calls.append(1)
        return real_get_final_state(*args, **kwargs)

    monkeypatch.setattr(
        true_cp_v2, "mamba2_get_final_state_triton", counting_get_final_state
    )

    zxBCdt_value = torch.randn(2 * segment_len, batch, projected_dim, device=device, dtype=dtype)
    conv_weight_value = torch.randn(conv_dim, 1, 4, device=device, dtype=dtype) * 0.2
    conv_bias_value = torch.randn(conv_dim, device=device, dtype=dtype) * 0.2
    dt_bias_value = torch.randn(nheads, device=device, dtype=torch.float32) * 0.2
    A_log_value = torch.randn(nheads, device=device, dtype=torch.float32) * 0.2
    D_value = torch.randn(nheads, device=device, dtype=torch.float32)
    grad_output = torch.randn(2 * segment_len, batch, d_inner, device=device, dtype=dtype)

    def run(cache_initial_states=None, bwd_subsequence_size=64):
        leaves = [
            zxBCdt_value.clone().requires_grad_(),
            conv_weight_value.clone().requires_grad_(),
            conv_bias_value.clone().requires_grad_(),
            dt_bias_value.clone().requires_grad_(),
            A_log_value.clone().requires_grad_(),
            D_value.clone().requires_grad_(),
        ]
        kwargs = {}
        if cache_initial_states is not None:
            kwargs["cache_initial_states"] = cache_initial_states
        out = true_cp_v2.mamba_true_context_parallel_training(
            *leaves,
            cp_group=_FakeGroup(rank=0, size=cp_size),
            d_inner_local_tp=d_inner,
            ngroups_local_tp=ngroups,
            d_state=d_state,
            headdim=headdim,
            chunk_size=chunk_size,
            activation="silu",
            D_has_hdim=False,
            bwd_subsequence_size=bwd_subsequence_size,
            **kwargs,
        )
        out.backward(grad_output)
        return out.detach(), [leaf.grad for leaf in leaves]

    out_recomputed, grads_recomputed = run(cache_initial_states=False)
    recomputed_calls = len(final_state_calls)
    out_cached, grads_cached = run()
    cached_calls = len(final_state_calls) - recomputed_calls
    out_unsplit, grads_unsplit = run(bwd_subsequence_size=0)
    unsplit_calls = len(final_state_calls) - recomputed_calls - cached_calls

    # The X-split backward derives its whole-segment summary from local
    # subsequence summaries, so the full summary kernel remains forward-only.
    assert recomputed_calls == 1
    assert cached_calls == 1
    assert unsplit_calls == 1
    assert torch.equal(out_cached, out_recomputed)
    assert torch.equal(out_cached, out_unsplit)
    # Splitting changes only the FP32 recurrence grouping. Its small reduction-
    # order difference can be amplified by BF16 convolution recomputation.
    cache_tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    for grad_cached, grad_recomputed in zip(grads_cached, grads_recomputed):
        torch.testing.assert_close(
            grad_cached.float(),
            grad_recomputed.float(),
            atol=cache_tolerance,
            rtol=cache_tolerance,
        )
    unsplit_atol = 5e-4 if dtype == torch.float32 else 1e-2
    unsplit_rtol = 1e-3 if dtype == torch.float32 else 1e-2
    for grad_cached, grad_unsplit in zip(grads_cached, grads_unsplit):
        torch.testing.assert_close(
            grad_cached.float(),
            grad_unsplit.float(),
            atol=unsplit_atol,
            rtol=unsplit_rtol,
        )
