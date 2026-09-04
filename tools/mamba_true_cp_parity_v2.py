#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare Mamba head-parallel CP against simplified true context-parallel CP v2.

Launch with torchrun on GPUs.  The harness compares the SSM training surface
directly:

    zxBCdt -> Mamba SSM output

The reference path is the existing head-parallel CP implementation:

    MambaContextParallel.pre_conv_ssm
    mamba_split_conv1d_scan_combined
    MambaContextParallel.post_conv_ssm

The v2 candidate path is:

    mamba_true_context_parallel_v2.mamba_true_context_parallel_training

Input gradients are compared rank-local.  Parameter gradients are summed across
the CP group before comparison, matching the DP+CP gradient-reduction semantics
used by training.

The candidate configuration matches the simplified v2 contract: width-4
convolution with bias, SiLU, head-level D, all-gather halo exchange, and the
specialized summary backward. Forward prefix states are cached by default so
backward skips the segment-summary recompute and gathers only the adjoint
summary. ``--recompute-initial-states`` opts into the recomputation path.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_mamba_modules() -> None:
    global MambaContextParallel
    global local_segment_boundary_states
    global local_segment_final_dstates
    global mamba_true_context_parallel_training
    global mamba_split_conv1d_scan_combined
    global rearrange

    from einops import rearrange as _rearrange
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_split_conv1d_scan_combined as _mamba_split_conv1d_scan_combined,
    )

    from megatron.core.ssm.mamba_context_parallel import (
        MambaContextParallel as _MambaContextParallel,
    )
    from megatron.core.ssm.mamba_true_context_parallel_v2 import (
        _local_segment_boundary_states,
        _local_segment_final_dstates,
        mamba_true_context_parallel_training as _mamba_true_context_parallel_training,
    )

    MambaContextParallel = _MambaContextParallel
    local_segment_boundary_states = _local_segment_boundary_states
    local_segment_final_dstates = _local_segment_final_dstates
    mamba_true_context_parallel_training = _mamba_true_context_parallel_training
    mamba_split_conv1d_scan_combined = _mamba_split_conv1d_scan_combined
    rearrange = _rearrange


def _cuda_summary() -> str:
    lines = [
        f"python={sys.executable}",
        f"torch={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"RANK={os.environ.get('RANK')}",
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}",
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}",
    ]
    try:
        lines.append(f"torch.cuda.is_available={torch.cuda.is_available()}")
        lines.append(f"torch.cuda.device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            lines.append(f"current_device={torch.cuda.current_device()}")
            lines.append(f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as error:  # pragma: no cover - diagnostic only
        lines.append(f"cuda_probe_error={error!r}")
    return "\n".join(lines)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


@dataclass(frozen=True)
class Case:
    name: str
    segment_len: int
    batch_size: int
    nheads: int
    headdim: int
    ngroups: int
    d_state: int
    chunk_size: int

    @property
    def local_seq_len(self) -> int:
        return 2 * self.segment_len

    @property
    def d_inner(self) -> int:
        return self.nheads * self.headdim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.ngroups * self.d_state

    @property
    def projected_dim(self) -> int:
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads


@dataclass
class ParamSet:
    conv: nn.Conv1d
    dt_bias: nn.Parameter
    A_log: nn.Parameter
    D: nn.Parameter

    def trainable_tensors(self) -> Iterable[Tuple[str, torch.Tensor]]:
        yield "conv_weight", self.conv.weight
        if self.conv.bias is not None:
            yield "conv_bias", self.conv.bias
        yield "dt_bias", self.dt_bias
        yield "A_log", self.A_log
        yield "D", self.D

    def zero_grad(self) -> None:
        for _, tensor in self.trainable_tensors():
            tensor.grad = None


def _init_distributed(args) -> Tuple[torch.device, int, int, torch.distributed.ProcessGroup]:
    if args.device != "cuda":
        raise SystemExit("This parity harness requires --device cuda.")
    if not torch.cuda.is_available():
        print("CUDA preflight failed.\n" + _cuda_summary(), flush=True)
        raise SystemExit(2)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=args.distributed_backend, init_method="env://")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        raise SystemExit("Run with at least 2 ranks so true CP and head CP can be compared.")
    return device, rank, world_size, torch.distributed.group.WORLD


def _default_cases(cp_size: int, args) -> List[Case]:
    nheads = cp_size * args.heads_per_cp_rank
    base_segment = max(args.segment_len, args.chunk_size * 2, 3)
    partial_segment = max(args.chunk_size + max(1, args.chunk_size // 3), 3)
    if args.large_global_seq_len % (2 * cp_size) != 0:
        raise SystemExit(
            "--large-global-seq-len must be divisible by 2 * context_parallel_size "
            f"({2 * cp_size}), got {args.large_global_seq_len}."
        )
    large_segment = args.large_global_seq_len // (2 * cp_size)
    cases = [
        Case(
            "groups_replicated",
            base_segment,
            args.batch_size,
            nheads,
            args.headdim,
            1,
            args.d_state,
            args.chunk_size,
        ),
        Case(
            "groups_sharded",
            base_segment,
            args.batch_size,
            nheads,
            args.headdim,
            cp_size,
            args.d_state,
            args.chunk_size,
        ),
        Case(
            "partial_chunk",
            partial_segment,
            args.batch_size,
            nheads,
            args.headdim,
            1,
            args.d_state,
            args.chunk_size,
        ),
    ]
    large_case = Case(
        "large_16k",
        large_segment,
        args.large_batch_size,
        args.large_nheads,
        args.large_headdim,
        args.large_ngroups,
        args.large_d_state,
        args.large_chunk_size,
    )
    wanted = {name.strip() for name in args.cases.split(",") if name.strip()}
    if "all" in wanted:
        selected = list(cases)
        if "large_16k" in wanted:
            selected.append(large_case)
        return selected
    available = {case.name: case for case in cases}
    available[large_case.name] = large_case
    selected = [available[name] for name in wanted if name in available]
    unknown = wanted - set(available)
    if unknown:
        raise SystemExit(f"Unknown --cases entries: {sorted(unknown)}")
    return selected


def _check_case_compat(case: Case, cp_size: int) -> None:
    if case.nheads % cp_size != 0:
        raise SystemExit(f"{case.name}: nheads must be divisible by cp_size.")
    if case.nheads % case.ngroups != 0:
        raise SystemExit(f"{case.name}: nheads must be divisible by ngroups.")
    if case.ngroups < cp_size:
        if cp_size % case.ngroups != 0:
            raise SystemExit(f"{case.name}: cp_size must be divisible by ngroups.")
    elif case.ngroups % cp_size != 0:
        raise SystemExit(f"{case.name}: ngroups must be divisible by cp_size.")
    if case.segment_len < 3:
        raise SystemExit(f"{case.name}: segment_len must be >= 3 for width-4 convolution.")


def _make_param_set(case: Case, device: torch.device, activation_dtype: torch.dtype) -> ParamSet:
    conv = nn.Conv1d(
        case.conv_dim,
        case.conv_dim,
        kernel_size=4,
        groups=case.conv_dim,
        padding=3,
        bias=True,
        device=device,
        dtype=activation_dtype,
    )
    dt_bias = nn.Parameter(torch.empty(case.nheads, device=device, dtype=activation_dtype))
    A_log = nn.Parameter(torch.empty(case.nheads, device=device, dtype=torch.float32))
    D = nn.Parameter(torch.empty(case.nheads, device=device, dtype=torch.float32))

    with torch.no_grad():
        conv.weight.normal_(mean=0.0, std=0.02)
        conv.bias.normal_(mean=0.0, std=0.01)
        log_dt_min = torch.log(torch.tensor(0.001, device=device))
        log_dt_max = torch.log(torch.tensor(0.1, device=device))
        dt = torch.exp(
            torch.rand(case.nheads, device=device, dtype=torch.float32) * (log_dt_max - log_dt_min)
            + log_dt_min
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        dt_bias.copy_(inv_dt.to(dtype=activation_dtype))
        A_log.copy_(torch.log(torch.empty(case.nheads, device=device).uniform_(1.0, 16.0)))
        D.copy_(1.0 + 0.05 * torch.randn(case.nheads, device=device))
    return ParamSet(conv, dt_bias, A_log, D)


def _clone_param_set(params: ParamSet) -> ParamSet:
    conv = nn.Conv1d(
        params.conv.in_channels,
        params.conv.out_channels,
        kernel_size=params.conv.kernel_size[0],
        groups=params.conv.groups,
        padding=params.conv.padding[0],
        bias=params.conv.bias is not None,
        device=params.conv.weight.device,
        dtype=params.conv.weight.dtype,
    )
    with torch.no_grad():
        conv.weight.copy_(params.conv.weight)
        if conv.bias is not None:
            conv.bias.copy_(params.conv.bias)
    return ParamSet(
        conv,
        nn.Parameter(params.dt_bias.detach().clone()),
        nn.Parameter(params.A_log.detach().clone()),
        nn.Parameter(params.D.detach().clone()),
    )


def _select_local_segments(global_tensor: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    segment_len = global_tensor.shape[0] // (2 * cp_size)
    first = rank
    second = 2 * cp_size - rank - 1
    pieces = [
        global_tensor[first * segment_len : (first + 1) * segment_len],
        global_tensor[second * segment_len : (second + 1) * segment_len],
    ]
    return torch.cat(pieces, dim=0).contiguous()


def _make_local_inputs(
    case: Case,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    cp_size: int,
    seed: int,
    make_grad_out: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    global_seq = case.local_seq_len * cp_size
    zx = 0.2 * torch.randn(
        global_seq,
        case.batch_size,
        case.projected_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if not make_grad_out:
        return _select_local_segments(zx, rank, cp_size), None
    grad_out = 0.1 * torch.randn(
        global_seq, case.batch_size, case.d_inner, device=device, dtype=dtype, generator=generator
    )
    return _select_local_segments(zx, rank, cp_size), _select_local_segments(
        grad_out, rank, cp_size
    )


def _run_boundary_check(device, rank, cp_size, cp_group) -> int:
    """Validate the packed forward/reverse affine exchange with a real collective."""

    num_segments = 2 * cp_size
    batch, nheads, headdim, dstate = 2, 2, 3, 2
    scalar_values = torch.arange(
        num_segments * batch * nheads, device=device, dtype=torch.float32
    ).reshape(num_segments, batch, nheads)
    global_log_alpha = -0.01 * (scalar_values + 1.0)
    global_beta = 0.02 * torch.arange(
        num_segments * batch * nheads * headdim * dstate,
        device=device,
        dtype=torch.float32,
    ).reshape(num_segments, batch, nheads, headdim, dstate)
    global_gamma = torch.flip(global_beta, dims=(0,)) * 0.75 + 0.125

    global_initial = torch.empty_like(global_beta)
    state = torch.zeros_like(global_beta[0])
    for segment in range(num_segments):
        global_initial[segment] = state
        state = (
            torch.exp(global_log_alpha[segment])[..., None, None] * state
            + global_beta[segment]
        )

    global_final = torch.empty_like(global_gamma)
    dstate_value = torch.zeros_like(global_gamma[0])
    for segment in reversed(range(num_segments)):
        global_final[segment] = dstate_value
        dstate_value = (
            torch.exp(global_log_alpha[segment])[..., None, None] * dstate_value
            + global_gamma[segment]
        )

    local_segment_ids = torch.tensor(
        [rank, num_segments - rank - 1], device=device, dtype=torch.long
    )
    actual_initial, actual_final = local_segment_boundary_states(
        global_log_alpha.index_select(0, local_segment_ids),
        global_beta.index_select(0, local_segment_ids),
        global_gamma.index_select(0, local_segment_ids),
        cp_group,
    )
    expected_initial = global_initial.index_select(0, local_segment_ids)
    expected_final = global_final.index_select(0, local_segment_ids)

    actual_final_cached = local_segment_final_dstates(
        global_log_alpha,
        global_gamma.index_select(0, local_segment_ids),
        cp_group,
    )

    failures = 0
    for name, actual, expected in (
        ("local_initial_states", actual_initial, expected_initial),
        ("local_final_dstates", actual_final, expected_final),
        ("local_final_dstates_cached", actual_final_cached, expected_final),
    ):
        max_abs = (actual - expected).abs().max()
        if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            failures += 1
            print(
                f"[rank {rank}] FAIL boundary {name}: max_abs={max_abs.item():.6e}",
                flush=True,
            )
        elif rank == 0:
            print(
                f"OK  boundary {name:<22} shape={tuple(actual.shape)} "
                f"max_abs={max_abs.item():.6e}",
                flush=True,
            )
    return failures


def _run_head_cp(
    zxBCdt: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    params: ParamSet,
    case: Case,
    cp_group: torch.distributed.ProcessGroup,
    collect_grads: bool = True,
    reduce_param_grads: bool = True,
    run_backward: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    params.zero_grad()
    zx = zxBCdt.detach().clone().requires_grad_(True)
    cp = MambaContextParallel(
        cp_group=cp_group,
        d_inner_local_tp=case.d_inner,
        nheads_local_tp=case.nheads,
        ngroups_local_tp=case.ngroups,
        d_state=case.d_state,
        conv1d_weight_cp1=params.conv.weight,
        conv1d_bias_cp1=params.conv.bias,
        conv1d_padding=params.conv.padding[0],
        dt_bias_cp1=params.dt_bias,
        A_log_cp1=params.A_log,
        D_cp1=params.D,
        D_has_hdim=False,
    )
    with torch.autograd.profiler.record_function("head_cp_pre_conv_ssm"):
        zx_pre = cp.pre_conv_ssm(zx)
    with torch.autograd.profiler.record_function("head_cp_mamba_split_conv1d_scan_combined"):
        y = mamba_split_conv1d_scan_combined(
            rearrange(zx_pre, "l b d -> b l d").contiguous(),
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            cp.get_dt_bias().float(),
            -torch.exp(cp.get_A_log().float()),
            D=cp.get_D(),
            chunk_size=case.chunk_size,
            activation="silu",
            headdim=case.headdim,
            ngroups=cp.ngroups_local_tpcp,
            norm_before_gate=False,
            seq_idx=None,
        )
        y = rearrange(y, "b l d -> l b d").contiguous()
    with torch.autograd.profiler.record_function("head_cp_post_conv_ssm"):
        y = cp.post_conv_ssm(y)
    if not run_backward:
        return y.detach(), {}
    if grad_out is None:
        raise ValueError("grad_out is required when run_backward=True.")
    loss = (y.float() * grad_out.float()).sum()
    with torch.autograd.profiler.record_function("head_cp_backward"):
        loss.backward()
    grads = _collect_grads(zx, params, cp_group, reduce_param_grads) if collect_grads else {}
    return y.detach(), grads


def _run_true_cp_v2(
    zxBCdt: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    params: ParamSet,
    case: Case,
    cp_group: torch.distributed.ProcessGroup,
    collect_grads: bool = True,
    reduce_param_grads: bool = True,
    run_backward: bool = True,
    cache_initial_states: bool = True,
    bwd_subsequence_size: int = 64,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    params.zero_grad()
    zx = zxBCdt.detach().clone().requires_grad_(True)
    with torch.autograd.profiler.record_function("true_cp_v2_mamba_training"):
        y = mamba_true_context_parallel_training(
            zx,
            params.conv.weight,
            params.conv.bias,
            params.dt_bias,
            params.A_log,
            params.D,
            cp_group=cp_group,
            d_inner_local_tp=case.d_inner,
            ngroups_local_tp=case.ngroups,
            d_state=case.d_state,
            headdim=case.headdim,
            chunk_size=case.chunk_size,
            activation="silu",
            D_has_hdim=False,
            cache_initial_states=cache_initial_states,
            bwd_subsequence_size=bwd_subsequence_size,
        )
    if not run_backward:
        return y.detach(), {}
    if grad_out is None:
        raise ValueError("grad_out is required when run_backward=True.")
    loss = (y.float() * grad_out.float()).sum()
    with torch.autograd.profiler.record_function("true_cp_v2_backward"):
        loss.backward()
    grads = _collect_grads(zx, params, cp_group, reduce_param_grads) if collect_grads else {}
    return y.detach(), grads


def _collect_grads(
    zx: torch.Tensor,
    params: ParamSet,
    cp_group: torch.distributed.ProcessGroup,
    reduce_param_grads: bool,
) -> Dict[str, torch.Tensor]:
    grads = {"zxBCdt": zx.grad.detach().clone()}
    for name, tensor in params.trainable_tensors():
        grad = torch.zeros_like(tensor) if tensor.grad is None else tensor.grad.detach().clone()
        if reduce_param_grads:
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=cp_group)
        grads[name] = grad
    return grads


def _max_mean(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float, bool]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    finite = torch.isfinite(actual_f).all() & torch.isfinite(expected_f).all()
    diff = (actual_f - expected_f).abs()
    return float(diff.max().item()), float(diff.mean().item()), bool(finite.item())


def _check_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    rank: int,
    cp_group: torch.distributed.ProcessGroup,
    fail_fast: bool,
) -> int:
    ok_local = torch.allclose(
        actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=True
    )
    max_abs, mean_abs, finite = _max_mean(actual, expected)
    stats = torch.tensor(
        [0 if ok_local and finite else 1, max_abs, mean_abs, 0 if finite else 1],
        device=actual.device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX, group=cp_group)
    failed = int(stats[0].item())
    if rank == 0:
        status = "OK " if failed == 0 else "BAD"
        print(
            f"{status} {name:<24} shape={tuple(actual.shape)!s:<20} "
            f"global_max_abs={stats[1].item():.6e} global_max_mean_abs={stats[2].item():.6e} "
            f"nonfinite={bool(stats[3].item())}",
            flush=True,
        )
    if failed and fail_fast:
        raise SystemExit(1)
    return failed


def _memory_line(prefix: str, device: torch.device) -> str:
    allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    return f"{prefix}: peak_allocated_mib={allocated:.1f} peak_reserved_mib={reserved:.1f}"


def _report_finiteness(
    label: str,
    output: torch.Tensor,
    grads: Dict[str, torch.Tensor],
    rank: int,
    cp_group: torch.distributed.ProcessGroup,
) -> int:
    """Report nonfinite values for one implementation without requiring a reference run."""
    failures = 0
    for name, tensor in [("output", output), *sorted(grads.items())]:
        tensor_f = tensor.detach().float()
        finite = torch.isfinite(tensor_f)
        nonfinite_count = (~finite).sum().to(dtype=torch.int64)
        finite_abs = torch.where(finite, tensor_f.abs(), torch.zeros_like(tensor_f))
        max_abs = finite_abs.max().to(dtype=torch.float64)
        torch.distributed.all_reduce(
            nonfinite_count, op=torch.distributed.ReduceOp.SUM, group=cp_group
        )
        torch.distributed.all_reduce(max_abs, op=torch.distributed.ReduceOp.MAX, group=cp_group)
        failed = int(nonfinite_count.item() != 0)
        failures += failed
        if rank == 0:
            print(
                f"FINITE {label:<10} {name:<16} ok={not failed} "
                f"nonfinite_count={int(nonfinite_count.item())} "
                f"finite_max_abs={max_abs.item():.6e}",
                flush=True,
            )
    return failures


def _benchmark_mode(
    label: str,
    run_fn: Callable[..., Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    base_params: ParamSet,
    zxBCdt: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    case: Case,
    cp_group: torch.distributed.ProcessGroup,
    device: torch.device,
    warmup: int,
    iters: int,
    include_grad_reduce: bool,
    run_backward: bool,
) -> Tuple[float, str]:
    params = _clone_param_set(base_params)
    times_ms: List[float] = []
    for idx in range(warmup + iters):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_fn(zxBCdt, grad_out, params, case, cp_group, False, False, run_backward)
        if include_grad_reduce:
            for _, tensor in params.trainable_tensors():
                if tensor.grad is not None:
                    torch.distributed.all_reduce(tensor.grad, group=cp_group)
        end.record()
        torch.cuda.synchronize(device)
        if idx >= warmup:
            times_ms.append(start.elapsed_time(end))
    mean_ms = sum(times_ms) / max(1, len(times_ms))
    return mean_ms, _memory_line(label, device)


def _profile_mode(
    label: str,
    run_fn: Callable[..., Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    base_params: ParamSet,
    zxBCdt: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    case: Case,
    cp_group: torch.distributed.ProcessGroup,
    device: torch.device,
    args,
    rank: int,
    run_backward: bool,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    trace_dir = Path(args.profile_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    params = _clone_param_set(base_params)
    torch.cuda.synchronize(device)
    for _ in range(args.profile_warmup):
        run_fn(zxBCdt, grad_out, params, case, cp_group, False, False, run_backward)
        if args.profile_include_grad_reduce and run_backward:
            for _, tensor in params.trainable_tensors():
                if tensor.grad is not None:
                    torch.distributed.all_reduce(tensor.grad, group=cp_group)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.profile_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.profile_stack,
    ) as prof:
        for _ in range(args.profile_iters):
            with torch.autograd.profiler.record_function(f"{label}_iteration"):
                run_fn(zxBCdt, grad_out, params, case, cp_group, False, False, run_backward)
                if args.profile_include_grad_reduce and run_backward:
                    with torch.autograd.profiler.record_function(f"{label}_param_grad_all_reduce"):
                        for _, tensor in params.trainable_tensors():
                            if tensor.grad is not None:
                                torch.distributed.all_reduce(tensor.grad, group=cp_group)
            prof.step()
    torch.cuda.synchronize(device)
    pass_suffix = "_fwd" if not run_backward else ""
    trace_path = trace_dir / f"mamba_true_cp_v2_{label}_{case.name}{pass_suffix}_rank{rank}.json"
    temp_trace_path = None
    export_trace_path = None
    if not args.no_profile_trace:
        export_trace_path = trace_path
    elif args.profile_summary_source == "trace" and args.profile_rows > 0:
        with tempfile.NamedTemporaryFile(
            prefix=f"mamba_true_cp_v2_{label}_{case.name}{pass_suffix}_rank{rank}_",
            suffix=".json",
            delete=False,
        ) as temp_trace:
            temp_trace_path = Path(temp_trace.name)
        export_trace_path = temp_trace_path

    if export_trace_path is not None:
        if export_trace_path == trace_path:
            partial_trace_path = trace_path.with_name(f"{trace_path.name}.tmp.{os.getpid()}")
            prof.export_chrome_trace(str(partial_trace_path))
            partial_trace_path.replace(trace_path)
        else:
            prof.export_chrome_trace(str(export_trace_path))
        if not args.no_profile_trace:
            print(f"PROFILE {label} rank={rank} trace={export_trace_path}", flush=True)

    if rank == 0 and args.profile_rows > 0 and args.profile_summary_source == "torch":
        print(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.profile_rows),
            flush=True,
        )
    elif rank == 0 and args.profile_rows > 0 and args.profile_summary_source == "trace":
        _print_trace_cuda_summary(export_trace_path, args.profile_rows)

    if temp_trace_path is not None:
        try:
            temp_trace_path.unlink()
        except FileNotFoundError:
            pass

    # Keep ranks aligned before the next profiled mode starts issuing collectives.
    torch.distributed.barrier(group=cp_group)


def _print_trace_cuda_summary(trace_path: Path, row_limit: int) -> None:
    with open(trace_path, "r", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    totals: Dict[str, List[float]] = {}
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        is_cuda_event = (
            "kernel" in category
            or "gpu_memcpy" in category
            or "gpu_memset" in category
            or name.startswith("nccl")
            or name.startswith("Memcpy")
            or name.startswith("Memset")
        )
        if not is_cuda_event:
            continue
        total_and_count = totals.setdefault(name, [0.0, 0.0])
        total_and_count[0] += float(event["dur"])
        total_and_count[1] += 1.0

    rows = sorted(totals.items(), key=lambda item: item[1][0], reverse=True)[:row_limit]
    print(
        "CUDA event summary parsed from Chrome trace "
        "(durations are CUDA event durations, not torch key_averages):",
        flush=True,
    )
    print(f"{'Name':<80} {'CUDA total':>12} {'CUDA avg':>12} {'# Calls':>8}", flush=True)
    for name, (total_us, count) in rows:
        avg_us = total_us / max(count, 1.0)
        print(
            f"{name[:80]:<80} {total_us / 1000.0:>10.3f}ms "
            f"{avg_us / 1000.0:>10.3f}ms {int(count):>8}",
            flush=True,
        )


def _run_case(args, case: Case, dtype: torch.dtype, device, rank, cp_size, cp_group) -> int:
    _check_case_compat(case, cp_size)
    torch.manual_seed(args.seed + 17)
    base_params = _make_param_set(case, device, dtype)
    run_backward = args.run_pass == "fwd-bwd"
    zxBCdt, grad_out = _make_local_inputs(
        case, device, dtype, rank, cp_size, args.seed + 1001, make_grad_out=run_backward
    )
    true_cp_v2_run = functools.partial(
        _run_true_cp_v2,
        cache_initial_states=args.cache_initial_states,
        bwd_subsequence_size=args.bwd_subsequence_size,
    )

    if rank == 0:
        print(
            f"\nCASE {case.name} dtype={dtype} cp={cp_size} local_seq={case.local_seq_len} "
            f"global_seq={case.local_seq_len * cp_size} segment={case.segment_len} "
            f"B={case.batch_size} H={case.nheads} P={case.headdim} "
            f"G={case.ngroups} N={case.d_state} chunk={case.chunk_size} "
            f"pass={args.run_pass} candidate=true_cp_v2 "
            f"cache_initial_states={args.cache_initial_states} "
            f"bwd_subsequence_size={args.bwd_subsequence_size}",
            flush=True,
        )

    failures = 0
    if not args.benchmark_only:
        if args.diagnostic_mode in ("head", "both"):
            head_params = _clone_param_set(base_params)
            torch.cuda.reset_peak_memory_stats(device)
            y_head, grads_head = _run_head_cp(
                zxBCdt, grad_out, head_params, case, cp_group, run_backward=run_backward
            )
            head_memory = _memory_line("head_cp", device)
            failures += _report_finiteness("head_cp", y_head, grads_head, rank, cp_group)
            if rank == 0:
                print(head_memory, flush=True)

        if args.diagnostic_mode in ("true", "both"):
            true_params = _clone_param_set(base_params)
            torch.cuda.reset_peak_memory_stats(device)
            y_true, grads_true = true_cp_v2_run(
                zxBCdt, grad_out, true_params, case, cp_group, run_backward=run_backward
            )
            true_memory = _memory_line("true_cp_v2", device)
            failures += _report_finiteness("true_cp_v3", y_true, grads_true, rank, cp_group)
            if rank == 0:
                print(true_memory, flush=True)

        if args.diagnostic_mode != "both":
            return failures

        atol = args.atol if args.atol is not None else (5e-2 if dtype != torch.float32 else 5e-4)
        rtol = args.rtol if args.rtol is not None else (5e-2 if dtype != torch.float32 else 5e-4)

        failures += _check_tensor(
            "output", y_true, y_head, atol, rtol, rank, cp_group, args.fail_fast
        )
        for name in sorted(grads_head):
            failures += _check_tensor(
                f"grad_{name}",
                grads_true[name],
                grads_head[name],
                atol,
                rtol,
                rank,
                cp_group,
                args.fail_fast,
            )

    if args.benchmark_iters > 0:
        head_ms, head_mem = _benchmark_mode(
            "head_cp",
            _run_head_cp,
            base_params,
            zxBCdt,
            grad_out,
            case,
            cp_group,
            device,
            args.benchmark_warmup,
            args.benchmark_iters,
            args.benchmark_include_grad_reduce and run_backward,
            run_backward,
        )
        true_ms, true_mem = _benchmark_mode(
            "true_cp_v2",
            true_cp_v2_run,
            base_params,
            zxBCdt,
            grad_out,
            case,
            cp_group,
            device,
            args.benchmark_warmup,
            args.benchmark_iters,
            args.benchmark_include_grad_reduce and run_backward,
            run_backward,
        )
        if rank == 0:
            ratio = true_ms / head_ms if head_ms else float("inf")
            print(
                f"BENCH case={case.name} dtype={dtype} pass={args.run_pass} head_ms={head_ms:.3f} "
                f"true_v2_ms={true_ms:.3f} true_v2/head={ratio:.3f}",
                flush=True,
            )
            print(f"BENCH {head_mem}", flush=True)
            print(f"BENCH {true_mem}", flush=True)

    if args.profile and (args.profile_case == "all" or args.profile_case == case.name):
        if args.profile_mode in ("head", "both"):
            _profile_mode(
                "head_cp",
                _run_head_cp,
                base_params,
                zxBCdt,
                grad_out,
                case,
                cp_group,
                device,
                args,
                rank,
                run_backward,
            )
        if args.profile_mode in ("true", "both"):
            _profile_mode(
                "true_cp_v2",
                true_cp_v2_run,
                base_params,
                zxBCdt,
                grad_out,
                case,
                cp_group,
                device,
                args,
                rank,
                run_backward,
            )

    return failures


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtypes", default="bf16,fp32", help="Comma-separated: fp32,bf16,fp16")
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "Comma-separated case names. 'all' runs the three quick cases; add "
            "large_16k explicitly. Available: groups_replicated, groups_sharded, "
            "partial_chunk, large_16k."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--segment-len", type=int, default=32)
    parser.add_argument("--heads-per-cp-rank", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--large-global-seq-len", type=int, default=16 * 1024)
    parser.add_argument("--large-batch-size", type=int, default=1)
    parser.add_argument("--large-nheads", type=int, default=128)
    parser.add_argument("--large-headdim", type=int, default=64)
    parser.add_argument("--large-ngroups", type=int, default=1)
    parser.add_argument("--large-d-state", type=int, default=128)
    parser.add_argument("--large-chunk-size", type=int, default=256)
    parser.add_argument(
        "--bwd-subsequence-size",
        type=int,
        default=64,
        help=(
            "True-CP-v2 backward subsection size. Zero selects the unsplit "
            "baseline; positive values must be multiples of 64."
        ),
    )
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--diagnostic-mode",
        choices=("head", "true", "both"),
        default="both",
        help="Run head CP, True CP, or both; each path reports output/gradient finiteness.",
    )
    parser.add_argument("--cuda-preflight-only", action="store_true")
    parser.add_argument(
        "--pass",
        dest="run_pass",
        choices=("fwd-bwd", "fwd", "boundary"),
        default="fwd-bwd",
        help=(
            "Select what each parity, benchmark, and profile run measures. "
            "fwd is the training forward pass with autograd tracking, not inference prefill. "
            "boundary runs only the FP32 packed boundary-collective check."
        ),
    )
    parser.add_argument(
        "--recompute-initial-states",
        dest="cache_initial_states",
        action="store_false",
        help="Recompute initial-state summaries in backward instead of caching forward results.",
    )
    parser.add_argument("--benchmark-iters", type=int, default=0)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-include-grad-reduce", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-mode", choices=("head", "true", "both"), default="both")
    parser.add_argument("--profile-case", default="groups_replicated")
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=0,
        help="Warmup iterations to run before starting torch profiler for each profiled mode.",
    )
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--profile-dir", default="outputs/mamba_true_cp_v2_profiles")
    parser.add_argument(
        "--profile-rows",
        type=int,
        default=30,
        help="Rows to print from the selected profiler summary. Use 0 to skip summary generation.",
    )
    parser.add_argument(
        "--profile-summary-source",
        choices=("torch", "trace", "none"),
        default="torch",
        help=(
            "Profiler summary source. torch uses prof.key_averages(); trace parses CUDA "
            "events from the Chrome trace and avoids key_averages(); none skips summaries."
        ),
    )
    parser.add_argument("--profile-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--profile-include-grad-reduce", action="store_true")
    parser.add_argument(
        "--no-profile-trace",
        action="store_true",
        help="Print profiler summary without exporting Chrome trace files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cuda_preflight_only:
        print(_cuda_summary(), flush=True)
        return 0

    device, rank, cp_size, cp_group = _init_distributed(args)
    _import_mamba_modules()

    if rank == 0:
        print(_cuda_summary(), flush=True)
        print(
            f"Running Mamba true-CP v2 parity with cp_size={cp_size} dtypes={args.dtypes} "
            f"cases={args.cases} pass={args.run_pass}",
            flush=True,
        )
        if args.run_pass == "fwd" and args.benchmark_include_grad_reduce:
            print("NOTE: --benchmark-include-grad-reduce is ignored with --pass fwd.", flush=True)
        if args.run_pass == "fwd" and args.profile_include_grad_reduce:
            print("NOTE: --profile-include-grad-reduce is ignored with --pass fwd.", flush=True)

    if args.run_pass == "boundary":
        failures = _run_boundary_check(device, rank, cp_size, cp_group)
        fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(
            fail_tensor, op=torch.distributed.ReduceOp.SUM, group=cp_group
        )
        failures = int(fail_tensor.item())
        if rank == 0:
            if failures:
                print(f"\nFAIL: {failures} boundary checks failed.", flush=True)
            else:
                print("\nPASS: packed Mamba true-CP v2 boundary exchange passed.", flush=True)
        torch.distributed.barrier(group=cp_group)
        return 1 if failures else 0

    failures = 0
    cases = _default_cases(cp_size, args)
    for dtype_name in args.dtypes.split(","):
        dtype_name = dtype_name.strip()
        dtype = _dtype(dtype_name)
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            if rank == 0:
                print(
                    "SKIP dtype=bf16 because this CUDA device does not report bf16 support.",
                    flush=True,
                )
            continue
        for case in cases:
            failures += _run_case(args, case, dtype, device, rank, cp_size, cp_group)

    fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(fail_tensor, op=torch.distributed.ReduceOp.SUM, group=cp_group)
    failures = int(fail_tensor.item())
    if rank == 0:
        if failures:
            print(f"\nFAIL: {failures} Mamba true-CP v2 parity checks failed.", flush=True)
        else:
            print("\nPASS: all Mamba true-CP v2 parity checks passed.", flush=True)

    torch.distributed.barrier(group=cp_group)
    return 1 if failures else 0


if __name__ == "__main__":
    start = time.time()
    status = main()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank == 0:
        print(f"elapsed_sec={time.time() - start:.2f}", flush=True)
    raise SystemExit(status)
