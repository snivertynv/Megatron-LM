#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run the existing distributed parity/profile matrix against CP v3.

It reuses the well-tested head-CP input construction and comparison harness
while replacing the candidate callable with CP v3 and asserting one combined
all-gather per executed pass.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.mamba_true_cp_parity_v2 as harness


_gather_calls = 0


def _install_v3_candidate() -> None:
    harness._import_mamba_modules_original()
    from megatron.core.ssm import mamba_true_context_parallel_v3 as v3

    original_gather = v3._all_gather_rank_tensor

    def counted_gather(tensor, cp_group):
        global _gather_calls
        _gather_calls += 1
        return original_gather(tensor, cp_group)

    v3._all_gather_rank_tensor = counted_gather

    def compatibility_entrypoint(*args, **kwargs):
        kwargs.pop("cache_initial_states", None)
        return v3.mamba_true_context_parallel_v3_training(*args, **kwargs)

    harness.mamba_true_context_parallel_training = compatibility_entrypoint


def _run_v3(*args, **kwargs):
    global _gather_calls
    run_backward = args[7] if len(args) > 7 else kwargs.get("run_backward", True)
    before = _gather_calls
    result = harness._run_true_cp_v2_original(*args, **kwargs)
    expected = 2 if run_backward else 1
    if _gather_calls - before != expected:
        raise AssertionError(
            f"CP v3 issued {_gather_calls - before} all-gathers instead of {expected}"
        )
    return result


def main() -> int:
    harness._import_mamba_modules_original = harness._import_mamba_modules
    harness._run_true_cp_v2_original = harness._run_true_cp_v2
    harness._import_mamba_modules = _install_v3_candidate
    harness._run_true_cp_v2 = _run_v3
    return harness.main()


if __name__ == "__main__":
    start = time.time()
    status = main()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(
            f"CP-v3 one-gather-per-pass assertions passed; elapsed_sec={time.time() - start:.2f}"
        )
    raise SystemExit(status)
