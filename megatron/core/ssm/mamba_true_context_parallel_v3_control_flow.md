# Mamba True Context-Parallel v3 Control Flow

True CP v3 is an additive training prototype. Production True CP v2 is
unchanged. For a width-4 convolution, every global segment is split into a
three-token prefix and a halo-independent body:

```text
segment j

S0 -- tokens 0,1,2 --> S3 -- tokens 3,...,L-1 --> SL
       prefix                        body
```

The implementation performs exactly one combined all-gather in forward and
one combined all-gather in backward. The forward gather replaces v2's
separate convolution-halo and forward-state exchanges. The backward gather
replaces v2's adjoint-state and halo-gradient exchanges.

## True-CP ownership

With context-parallel size `C`, rank `r` owns global segments `r` and
`2*C-r-1`. Its projected input is exposed as `[2, L, batch, ...]`; the two
streams are independent recurrence segments and are never scanned across
their local concatenation. Gathered rank-slot storage is restored to causal
segment order before either state stitch.

## Forward

```text
mamba_true_context_parallel_v3_training(...)
│
├── Split projected input into z, raw xBC, and raw dt stream views
│
├── Phase 1: local halo-independent work
│   ├── Convolve body positions 3:L from local raw xBC only
│   ├── Prepare compact body dt/decay tensors
│   ├── Summarize the body as
│   │       SL = exp(log_alpha_body) * S3 + beta_body
│   └── Pack body summary and boundary operands
│
├── Phase 2: one forward all-gather
│   └── log_alpha_body | beta_body |
│       prefix raw xBC | tail raw xBC | prefix raw dt
│
├── Prefix/state stitch
│   ├── Restore global causal segment order
│   ├── For each causally earlier segment:
│   │   ├── reconstruct its three convolved prefix tokens
│   │   ├── advance the FP32 state through the prefix
│   │   └── apply its gathered body affine summary
│   ├── Save the state S0 entering each locally owned segment during training
│   ├── Reconstruct and advance the current local prefix
│   ├── Write the three local prefix outputs
│   └── Return S3 as the local body-scan initial state
│
└── Phase 3: run the existing state-free forward scan over body tokens 3:L
```

For `H` value heads, value dimension `P`, state dimension `N`, and raw
convolution dimension `conv_dim`, the FP32 forward payload width per
rank-slot/batch item is

```text
H + H*P*N + 6*conv_dim + 3*H.
```

The fields are body log decay, body beta, first and last three raw xBC tokens,
and the first three raw dt tokens. Reconstructed convolution results are
rounded through the activation dtype before entering the recurrence, matching
the v2 convolution-store/scan-load boundary.

Forward prefix reconstruction and state replay are fused in the stride-aware
Triton `_prefix_state_replay_kernel`. One program owns one
`(folded_batch, value_head)` FP32 `[P,N]` state tile, loops over causally
earlier global segments inside the kernel, writes the three local prefix
outputs, and returns `S3`. There is no Python token/segment replay loop or
materialized global prefix tensor.

### Training-only forward cache

The custom autograd wrapper resolves its cache policy before entering
`autograd.Function.forward`:

- with grad mode enabled, retain local `S0` as FP32
  `[2, batch, H, P, N]` and the selected predecessor raw halo as activation
  dtype `[2, 3, batch, conv_dim]`;
- under `torch.no_grad()` or inference/prefill, retain neither tensor.

The predecessor halo is selected from the already gathered forward payload,
so caching it does not add communication. Only `S0`, rather than both `S0`
and `SL`, is retained. Backward recomputes local `SL` after its gather.

## Backward: full prefix replay

Let `lambda_L = dLoss/dSL` and `lambda_3 = dLoss/dS3`. The body has the
reverse affine form

```text
lambda_3 = exp(log_alpha_body) * lambda_L + gamma_body.
```

`gamma_body` is the body's input-state adjoint when its supplied terminal
adjoint is zero. No rank can know its true `lambda_L` locally because it
contains contributions from all causally later segments.

```text
custom backward
│
├── Phase 1: local pre-gather work
│   ├── Reconstruct the complete local convolution from the saved predecessor halo
│   ├── Prepare compact decay for body tokens 3:L
│   ├── Compute body log_alpha and gamma_body
│   ├── Compute prefix dpre = dout * z * sigmoid(z)
│   └── Pack the full-replay payload
│
├── Phase 2: one backward all-gather
│   └── log_alpha_body | gamma_body |
│       prefix raw xBC | tail raw xBC | prefix raw dt | prefix dpre
│
├── Reverse global state stitch
│   ├── Restore all 2*C segments to causal order
│   ├── Reconstruct all three-token convolution prefixes
│   ├── Start with zero adjoint after the final global segment
│   └── For segments 2*C-1,...,0:
│       ├── record the true lambda_L
│       ├── apply the body affine map to obtain lambda_3
│       └── reverse-replay prefix tokens 2,1,0 to obtain lambda_0
│
├── Recompute local forward boundary SL
│   ├── Prepare compact decay for the complete local segment
│   ├── X=0: compose one full-segment beta summary with cached S0
│   └── X=64: scan rank-local subsection beta summaries from cached S0
│
├── Replay only each local segment's immediate successor prefix
│   ├── Seed forward replay with the local predecessor's recomputed SL
│   ├── Seed reverse replay with the successor's resolved lambda_3
│   ├── Recompute successor prefix dC, dB, and dx
│   ├── map through SiLU backward
│   └── apply the tiny transpose convolution to produce the local raw-tail halo
│
├── Run the existing CP-native local backward stages
│   ├── dQ/dC, dz, forward chunk states, and decay contributions
│   ├── dK/dV, dD, reverse recurrence, and decay contributions
│   └── raw-dt, A_log, and dt_bias mapping
│
├── Run local convolution backward with the saved predecessor halo
│   ├── keep ordinary local input/weight/bias gradients
│   ├── discard its predecessor-halo output
│   └── add the locally replayed successor halo to raw xBC[-3:]
│
└── Assemble dz | dxBC | ddt in projected-input order
```

The FP32 backward payload width is

```text
H + H*P*N + 6*conv_dim + 3*H + 3*H*P.
```

The collective contains no true `lambda_L`, `lambda_3`, or final prefix
`dB/dx`: those values depend on later segments and are resolved only after the
gather.

### Triton replay ownership

All production replay after the backward gather is GPU-resident:

- `_backward_boundary_replay_kernel` assigns one program to each
  `(batch, value_head)`. It holds one FP32 `[P,N]` adjoint tile, traverses all
  `2*C` global segments and their three prefix tokens in reverse inside the
  kernel, and stores only the two local `lambda_L` values and two immediate-
  successor `lambda_3` values.
- `_successor_prefix_replay_kernel` assigns one program to each local
  `(stream, batch, value_head)`. It forward-replays the immediate successor
  from the recomputed predecessor `SL`, reverse-replays it from the resolved
  `lambda_3`, and emits tiny per-head three-token `dV/dB/dC` partials.
- `_successor_halo_reduce_kernel` deterministically reduces grouped-query
  heads, reconstructs the three convolution preactivations, applies SiLU
  backward, and performs the width-4 transpose-convolution tail update.

The backward path does not materialize globally ordered convolved prefixes
and has no Python token or global-segment replay loop. The only successor
workspace is three tokens of FP32 `dV` and per-value-head `dB/dC` partials.

### Why only the immediate successor produces a halo

A local segment's last three raw xBC tokens are read across a segment boundary
only by convolution outputs 0, 1, and 2 of its immediate successor. Later
segments affect the value of the successor's resolved `lambda_3`, but do not
directly read the predecessor's raw tokens. Therefore every rank scans all
segments for the adjoint boundary, then fully replays only the immediate
successor prefix for each of its two local segments.

The v2 convolution weight tensor stores width taps oldest-to-newest. Writing
`W[k] = conv_weight[:, 0, k]` and `g[t]` for the successor prefix gradient
with respect to convolution preactivation gives

```text
d_raw[-3] = W[0] * g[0]
d_raw[-2] = W[1] * g[0] + W[0] * g[1]
d_raw[-1] = W[2] * g[0] + W[1] * g[1] + W[0] * g[2].
```

The final global segment has no successor, so its synthetic halo is zeroed.

## Synchronization budget

```text
forward:   one body-summary + boundary-payload all-gather
backward:  one body-adjoint + full-replay-payload all-gather
total:     two synchronization barriers per training layer
```

There is no forward halo exchange, backward halo exchange, or separate state
exchange on the v3 path. Production v2 and its four-collective behavior remain
available unchanged.
