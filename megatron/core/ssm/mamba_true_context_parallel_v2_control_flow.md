# Mamba True Context-Parallel v2 Control Flow

The compact forward and its checkpointed state-free backward are both wired.
During training, forward saves the projected input, parameters, small
predecessor-convolution halos, compact incoming segment states, and global log
decays by default. Backward recomputes local scan intermediates but never saves
or materializes a full recurrence history.

```text
mamba_true_context_parallel_training(...)
│
├── Validate fixed fast-path contract
│   [_check_inputs + checks in mamba_true_context_parallel_training]
│   ├── CP size > 1
│   ├── Width-4 convolution with bias
│   ├── SiLU and head-level D
│   ├── All-gather halo exchange and specialized summary backward
│   └── Backward subsection size is zero or a multiple of 64
│
├── Resolve convolution-halo retention before autograd apply
│   ├── Default: save with grad mode enabled
│   ├── Default: skip under no-grad/inference prefill
│   └── Explicit cache_conv_halos=True/False overrides the default
│
├── Resolve initial-state retention before autograd apply
│   ├── Default training: save local_initial_states and global_log_alpha
│   ├── No-grad/inference: never retain them
│   └── Explicit cache_initial_states=False enables backward recomputation
│
└── _MambaTrueContextParallelFn.apply(...)
    │
    ├════════════════════════ FORWARD ════════════════════════
    │
    ├── _MambaTrueContextParallelFn.forward
    │   │
    │   ├── _prepare_folded_forward
    │   │   │
    │   │   ├── _prepare_folded_scan_state
    │   │   │   │
    │   │   │   ├── Split projected input  [_split_projected_streams]
    │   │   │   │   └── zxBCdt → z streams | xBC streams | dt streams
    │   │   │   │
    │   │   │   ├── Prepare width-4 convolution halos
    │   │   │   │   [_exchange_conv_halos_streams]
    │   │   │   │   ├── Take each local segment's final 3 xBC positions
    │   │   │   │   ├── All-gather rank/slot halos  [_all_gather_rank_tensor]
    │   │   │   │   └── Select the global predecessor  [_segment_indices]
    │   │   │   │       └── Optionally retain only [2, 3, B, conv_dim]
    │   │   │   │
    │   │   │   ├── Fused Triton causal convolution
    │   │   │   │   [_causal_conv_format_streams_triton]
    │   │   │   │   └── xBC + predecessor halo → x, B, C
    │   │   │   │       [_causal_conv_format_streams_kernel]
    │   │   │   │
    │   │   │   ├── Expose z head view without folding/copying  [_format_z_streams]
    │   │   │   │
    │   │   │   ├── Compact stream-aware decay preparation
    │   │   │   │   [compute_dacs_compact_triton]
    │   │   │   │   ├── Read the two strided windows without folding copies
    │   │   │   │   ├── dt_bias + softplus + dA cumulative sums
    │   │   │   │   └── Emit only dt_out, dA_cs, and dA_cs_rev
    │   │   │   │       └── Never allocate global chunk-square segsum
    │   │   │   │
    │   │   │   ├── Compact segment summaries
    │   │   │   │   [mamba2_get_final_state_triton]
    │   │   │   │   └── Emit state_out = exp(log α)·state_in + β summary
    │   │   │   │
    │   │   │   ├── Context-parallel affine prefix
    │   │   │   │   [_local_segment_initial_states]
    │   │   │   │   ├── Pack and all-gather (log α, β) summaries
    │   │   │   │   │   [_all_gather_rank_tensor]
    │   │   │   │   ├── Restore global causal order  [_segment_indices]
    │   │   │   │   ├── Exclusive prefix  [_exclusive_affine_prefix_states]
    │   │   │   │   └── Select this rank's 2 incoming states
    │   │   │   │
    │   │   │
    │   │   └── Fused state-free output scan  [mamba2_fwd_triton]
    │   │       ├── Map q=C, k=B, v=x
    │   │       ├── Initialize accumulator from each local incoming state
    │   │       ├── Consume prepared dt_out, dA_cs, and dA_cs_rev
    │   │       ├── Reconstruct causal log-decay only in program-local tiles
    │   │       ├── Add D·x and gate with z·sigmoid(z)
    │   │       └── Write [2, segment, batch, heads, headdim] directly
    │   │           ├── chunk <= 64: whole-chunk kernel
    │   │           └── chunk > 64: one-stage tiled row/key kernel
    │   │
    │   ├── Save projected input, parameters, selected conv halos, and default
    │   │   compact initial-state cache when enabled
    │   │   [ctx.save_for_backward]
    │   │   └── Never save out_x or a full recurrence history
    │   └── Return a no-copy [local_sequence, batch, hidden] view
        │
        ▼
      Training loss
        │
        ▼
    ┌════════════════════════ BACKWARD ═══════════════════════┐
    │
    └── _MambaTrueContextParallelFn.backward
        └── _true_context_backward_impl
            │
            ├── View grad output as [2, L, B, H, P] without copying
            │   [_format_grad_output_streams]
            │
            ├── Recompute local forward preparation
            │   [_prepare_folded_scan_state, defer_boundary_exchange=True]
            │   ├── Recreate z, xBC, and raw-dt stream views
            │   ├── Reuse saved predecessor halos by default
            │   │   └── Explicit training opt-out repeats the halo exchange
            │   ├── Recompute convolution outputs x, B, C
            │   ├── Recompute dt_out, dA_cs, and dA_cs_rev
            │   │   ├── Use the same compact preparation as forward
            │   │   └── Do not allocate segsum or folded dt_input
            │   ├── Unsplit default cache: skip compact beta/log_alpha recomputation
            │   ├── Unsplit recompute opt-in: rebuild compact beta and log_alpha
            │   ├── Split chunk-64 path: defer the whole-segment summary
            │   └── Defer the forward-only boundary gather
            │
            ├── Prepare backward recurrence summaries
            │   ├── Default chunk 64 / X=64 split path
            │   │   ├── Compute per-subsection (log_alpha, beta)
            │   │   │   [mamba2_get_subsequence_states_triton]
            │   │   ├── Compute reverse-ordered per-subsection gamma
            │   │   │   [mamba2_get_subsequence_dstates_triton]
            │   │   ├── Compose only one whole-segment beta/gamma for CP
            │   │   │   [compose_subsequence_summaries_triton]
            │   │   ├── Run complete subsections without tail predicates
            │   │   └── Run one general masked producer for a partial tail
            │   └── Unsplit fallback: compute one zero-terminal local gamma
            │       [mamba2_get_initial_dstate_triton]
            │
            ├── Resolve recurrence boundaries
            │   ├── Default cache: gamma-only reverse all-gather
            │   │   [_local_segment_final_dstates]
            │   │   ├── Reuse saved local_initial_states and global_log_alpha
            │   │   └── Reversed exclusive affine scan → local_final_dstates
            │   └── Recompute opt-in: packed forward/reverse all-gather
            │       [_local_segment_boundary_states]
            │       ├── Pack FP32 (log_alpha, beta, gamma)
            │       ├── Forward exclusive affine scan → local_initial_states
            │       └── Reversed exclusive affine scan → local_final_dstates
            │
            ├── Split-path rank-local subsection prefixes
            │   [_subsequence_state_passing → _state_passing_fwd]
            │   ├── Run Mamba's out-of-place affine scan with chunk_size=1
            │   ├── Release each zero-boundary summary after its scan
            │   ├── Return FP32 incoming beta and terminal-adjoint gamma states
            │   └── No subsection tensor enters a collective
            │
            ├── CP-native dC, dz, chunk starts, and current/next decay
            │   [mamba2_bwd_dq_cp_triton]
            │   ├── Default X=64: one CTA per logical chunk, seeded from its
            │   │   scanned incoming state; reuse that buffer as chunk starts
            │   ├── Other positive X: one CTA per subsection
            │   ├── Complete subsections use a mask-light exact-shape kernel
            │   ├── A partial final subsection uses the general masked kernel
            │   ├── X=0 or non-chunk-64 fallback: one CTA per local segment
            │   ├── Recompute pre-gate only in registers
            │   ├── Reconstruct only local causal segsum tiles from dA_cs
            │   ├── Fuse current dM reduction into dQ through chunk 64
            │   ├── Use a separate uniquely owned tiled reduction above 64
            │   ├── Cache compact grad_scale_intermediate through chunk 64
            │   ├── Map cached next-state decay with directional scans
            │   ├── Recompute next-state decay above chunk 64
            │   └── Emit compact processed_ddt and dA_chunk accumulators
            │
            ├── CP-native dB, dx, dD, and previous/state decay
            │   [mamba2_bwd_dk_dv_cp_triton]
            │   ├── Seed each split CTA from its scanned terminal adjoint
            │   ├── Use one full 64-row/full 64-key tile at default X=64
            │   ├── Split complete reverse subsections from the masked tail
            │   ├── Reuse transient chunk-start states from dQ
            │   ├── Recompute dpre without materializing pre-gate
            │   ├── Keep reverse accumulator register-local
            │   ├── Offload compact ddA_cs_rev from acc_dv_intermediate
            │   ├── Add direct previous and transition terms in the recurrence
            │   ├── Map cached previous decay with shifted directional scans
            │   └── Add directly to processed_ddt and dA_chunk
            │
            ├── Raw-dt and parameter-gradient mapping
            │   [mamba2_bwd_decay_map_cp_triton]
            │   ├── Consume the completed compact accumulators
            │   ├── Map chunked ddt directly to raw stream strides
            │   ├── Deterministically reduce dt_bias and dA_chunk
            │   └── Return raw dt, A_log, and dt_bias gradients
            │
            ├── Fused width-4 convolution backward
            │   [_causal_conv_format_streams_backward_triton]
            │   └── Return dxBC, halo grads, and local parameter grads
            │
            ├── One halo-gradient all-gather  [_route_conv_halo_grads]
            │   └── Add each successor's halo gradient to the owner's tail
            │
            └── Concatenate dz | dxBC | ddt and return all gradients
```

With the default saved-halo and initial-state-cache policies, backward issues
exactly two all-gathers: the gamma-only boundary exchange and the halo-gradient
exchange. Initial-state recomputation replaces the gamma-only exchange with one
packed `(log_alpha, beta, gamma)` exchange; the subsection decomposition does
not change either collective's count or rankwise payload. The forward pass has
its own predecessor-halo and affine-prefix exchanges. An explicit
`cache_conv_halos=False` training call adds one backward recomputation exchange
for predecessor halos.

Training caches the forward prefix states and causally ordered global log
decays by default. Backward therefore skips the whole-segment
`mamba2_get_final_state_triton` and global forward affine scan, and the boundary
exchange shrinks to the gamma-only payload of `_local_segment_final_dstates`.
The default chunk-64/X=64 path still computes rank-local subsection beta
summaries because dQ needs every chunk's incoming state; these summaries are
never communicated. The rank-local prefix uses Mamba's out-of-place
`_state_passing_fwd`; at X=64 its beta output is also the existing dK/dV
chunk-start handoff, so no third state buffer is allocated. Complete
subsections dispatch without sequence-tail predicates, while at most one
partial final subsection retains the general masked path. Pass
`cache_initial_states=False`, or use
`--mamba-true-cp-recompute-initial-states`, to opt into the packed recomputation
path. Set `--mamba-true-cp-bwd-subsequence-size 0` to restore the unsplit v2
backward; positive values are multiples of 64 and split only the logical
chunk-64 path.

Each CP rank owns two load-balanced sequence segments. Local computation produces affine
segment summaries; global all-gathers connect those segments causally for recurrent-state
prefixes and convolution halos.

## Function index

| Control-flow responsibility | Function |
| --- | --- |
| Public fixed-path entrypoint | `mamba_true_context_parallel_training` |
| Autograd boundary | `_MambaTrueContextParallelFn.forward`, `_MambaTrueContextParallelFn.backward` |
| Shape and fixed-path validation | `_check_inputs` |
| Forward orchestration and final scan | `_prepare_folded_forward` |
| Compact forward preparation / backward recomputation | `_prepare_folded_scan_state` |
| Split `zxBCdt` into projection streams | `_split_projected_streams` |
| Expose stream-major `z` head view | `_format_z_streams` |
| Expose stream-major grad-output head view | `_format_grad_output_streams` |
| Shared production compact decay preparation | `compute_dacs_compact_triton` |
| Full segsum preparation test/reference helper | `compute_dacs_segsum_triton` |
| Compact affine segment summary | `mamba2_get_final_state_triton` |
| Compact adjoint segment summary | `mamba2_get_initial_dstate_triton` |
| Per-subsection forward/reverse summaries | `mamba2_get_subsequence_states_triton`, `mamba2_get_subsequence_dstates_triton` |
| Compose subsection summaries for CP | `compose_subsequence_summaries_triton` |
| Rank-local parallel affine scan | `_subsequence_state_passing`, `_state_passing_fwd` |
| CP-native dC/dz, chunk-state, and current/next decay pass | `mamba2_bwd_dq_cp_triton` |
| CP-native dB/dx and previous/state decay pass | `mamba2_bwd_dk_dv_cp_triton` |
| Compact raw-dt/A_log/dt-bias mapper | `mamba2_bwd_decay_map_cp_triton` |
| State-free output scan | `mamba2_fwd_triton`, `mamba2_fwd_kernel`, `_mamba2_fwd_large_chunk_kernel` |
| Gather predecessor convolution halos | `_exchange_conv_halos_streams` |
| Reuse or gather predecessor halos | `_resolve_conv_halos_streams` |
| Launch fused convolution forward | `_causal_conv_format_streams_triton` |
| Convolution forward Triton kernel | `_causal_conv_format_streams_kernel` |
| State-free backward orchestration | `_true_context_backward_impl` |
| Forward global affine prefix | `_local_segment_initial_states` |
| Combined forward/reverse boundary exchange | `_local_segment_boundary_states` |
| Gamma-only reverse boundary exchange (cached initial states) | `_local_segment_final_dstates` |
| Associative affine scan | `_exclusive_affine_prefix_states` |
| Launch fused convolution backward | `_causal_conv_format_streams_backward_triton` |
| Convolution backward local/tiled work | `_causal_conv_backward_fused_width4_tiled_kernel` |
| Convolution backward boundary routing | `_causal_conv_backward_boundary_width4_kernel` |
| Convolution weight/bias reduction | `_causal_conv_backward_reduce_width4_kernel` |
| Return halo gradients to predecessor inputs | `_route_conv_halo_grads` |
| Differentiable CP all-gather | `_all_gather_rank_tensor` |
| Cached load-balanced segment mappings | `_segment_indices` |
