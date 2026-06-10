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


def _force_autograd_thread_creation() -> None:
    """Create the CUDA autograd worker thread before the main thread is narrowed."""
    if not torch.cuda.is_available():
        return

    x = torch.zeros(1, device="cuda", requires_grad=True)
    (x * x).sum().backward()
    torch.cuda.synchronize()


def initialize_forward_backward_cpu_affinity(cpu_list: str) -> None:
    """Initialize opt-in CPU pinning for forward/backward dispatch threads.

    NCCL and other helper threads are intentionally left untouched so they keep
    the affinity mask inherited from process launch.
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
        "Other threads keep their launch-time affinity."
    )


@contextmanager
def _forward_backward_cpu_affinity_context():
    requested_cpus = _FORWARD_BACKWARD_CPU_AFFINITY
    if requested_cpus is None:
        yield
        return

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
