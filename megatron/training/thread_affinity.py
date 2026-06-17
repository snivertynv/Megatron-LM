# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for pinning only forward/backward dispatch threads."""

import os
import re
from contextlib import contextmanager, nullcontext
from typing import Iterable, Optional, Set

import torch

from megatron.training.utils import print_rank_0

_AUTOGRAD_THREAD_RE = re.compile(r"^pt_autograd_\d+$")
_FORWARD_BACKWARD_CPU_AFFINITY: Optional[Set[int]] = None

# Number of forward/backward calls to run with temporary per-step pinning before
# switching to permanent pinning.  By this point all lazy NCCL proxy threads
# (spawned on first pipeline send/recv or first all-reduce) will be running and
# can be explicitly swept back to HELPER CPUs.
_FINALIZE_AFTER_STEPS = 2
_STEP_COUNT = 0
_PERMANENTLY_PINNED = False


def _parse_cpu_list(cpu_list: str) -> Set[int]:
    """Parse Linux cpulist syntax, e.g. ``"0,1,8-11"``."""
    cpus = set()
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start)
            end = int(end)
            if end < start:
                raise ValueError(f"invalid CPU range '{part}' in '{cpu_list}'")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))

    if not cpus:
        raise ValueError(f"CPU affinity list '{cpu_list}' did not contain any CPUs")
    return cpus


def _format_cpu_list(cpus: Iterable[int]) -> str:
    """Format a CPU set using compact Linux cpulist syntax."""
    sorted_cpus = sorted(cpus)
    if not sorted_cpus:
        return ""

    ranges = []
    start = prev = sorted_cpus[0]
    for cpu in sorted_cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _iter_thread_ids_and_names():
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm", encoding="utf-8") as f:
                name = f.read().rstrip("\n")
        except FileNotFoundError:
            continue
        yield int(tid), name


def _pin_autograd_threads(cpus: Set[int]) -> int:
    """Pin already-created PyTorch CUDA autograd worker threads."""
    pinned = 0
    for tid, name in _iter_thread_ids_and_names():
        if _AUTOGRAD_THREAD_RE.match(name):
            os.sched_setaffinity(tid, cpus)
            pinned += 1
    return pinned


def _reset_non_pct_threads(pct_cpus: Set[int], helper_cpus: Set[int]) -> int:
    """Reset every live thread that is not the main thread and not a
    pt_autograd_* thread back to helper_cpus.  Called once at finalization
    to push any lazily-spawned NCCL proxy threads off PCT cores."""
    main_tid = os.getpid()
    reset = 0
    for tid, name in _iter_thread_ids_and_names():
        if tid == main_tid:
            continue
        if _AUTOGRAD_THREAD_RE.match(name):
            continue
        try:
            os.sched_setaffinity(tid, helper_cpus)
            reset += 1
        except OSError:
            pass
    return reset


def _force_autograd_thread_creation() -> None:
    """Create the CUDA autograd worker thread before the main thread is narrowed."""
    if not torch.cuda.is_available():
        return

    x = torch.zeros(1, device="cuda", requires_grad=True)
    (x * x).sum().backward()
    torch.cuda.synchronize()


def initialize_forward_backward_cpu_affinity(cpu_list: str) -> None:
    """Initialize opt-in CPU pinning for forward/backward dispatch threads.

    pt_autograd_* threads are pinned immediately.  The main thread is pinned
    permanently after _FINALIZE_AFTER_STEPS forward/backward calls, at which
    point all lazy NCCL proxy threads are already running and get explicitly
    reset to HELPER CPUs before the main thread is narrowed.
    """
    global _FORWARD_BACKWARD_CPU_AFFINITY

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "--forward-backward-cpu-affinity requires Linux sched affinity support"
        )
    if not os.path.isdir("/proc/self/task"):
        raise RuntimeError("--forward-backward-cpu-affinity requires /proc thread metadata")

    requested_cpus = _parse_cpu_list(cpu_list)
    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise RuntimeError("--forward-backward-cpu-affinity could not determine CPU count")
    online_cpus = set(range(cpu_count))
    if not requested_cpus.issubset(online_cpus):
        raise ValueError(
            "--forward-backward-cpu-affinity must be a subset of online CPUs. "
            f"requested={_format_cpu_list(requested_cpus)} "
            f"online_cpus={_format_cpu_list(online_cpus)}"
        )

    _force_autograd_thread_creation()
    pinned_autograd_threads = _pin_autograd_threads(requested_cpus)
    _FORWARD_BACKWARD_CPU_AFFINITY = requested_cpus

    print_rank_0(
        "Forward/backward CPU affinity enabled: "
        f"cpus={_format_cpu_list(requested_cpus)}, "
        f"pinned_autograd_threads={pinned_autograd_threads}. "
        f"Main thread will be permanently pinned after {_FINALIZE_AFTER_STEPS} steps "
        "(deferred to allow lazy NCCL proxy threads to spawn on HELPER CPUs first)."
    )


@contextmanager
def _forward_backward_cpu_affinity_context():
    global _STEP_COUNT, _PERMANENTLY_PINNED

    requested_cpus = _FORWARD_BACKWARD_CPU_AFFINITY
    if requested_cpus is None:
        yield
        return

    if _PERMANENTLY_PINNED:
        yield
        return

    _STEP_COUNT += 1

    if _STEP_COUNT >= _FINALIZE_AFTER_STEPS:
        # All lazy NCCL proxy threads should be spawned by now.
        # Snapshot helper CPUs while the main thread still has HELPER affinity.
        helper_cpus = os.sched_getaffinity(0)
        reset = _reset_non_pct_threads(requested_cpus, helper_cpus)
        os.sched_setaffinity(0, requested_cpus)
        _PERMANENTLY_PINNED = True
        print_rank_0(
            f"Forward/backward CPU affinity finalized: main thread permanently pinned to "
            f"cpus={_format_cpu_list(requested_cpus)}, "
            f"reset {reset} other threads to helper cpus={_format_cpu_list(helper_cpus)}."
        )
        yield
        return  # no restore — permanently pinned

    # Steps 1 .. _FINALIZE_AFTER_STEPS-1: temporary per-step pinning so the
    # main thread stays on HELPER CPUs when not in forward/backward, ensuring
    # any NCCL threads spawned during these steps inherit HELPER affinity.
    previous_cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, requested_cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous_cpus)


def forward_backward_cpu_affinity():
    """Return a context manager for the configured forward/backward CPU mask."""
    if _FORWARD_BACKWARD_CPU_AFFINITY is None:
        return nullcontext()
    return _forward_backward_cpu_affinity_context()
