# Algorithms

This document describes the mechanisms implemented by ThetaScan v0.1.0. The
public names are used throughout: **GN memory** and **normalized kernel
memory**.

## 1. Shared scan view

For every token, ThetaScan forms a write key `k_t`, a payload `v_t`, and a read
query `q_t`. Token-local write contributions are combined with an associative
causal scan. A read at position `t` sees only positions `i <= t`. The *algebra*
needs no dense attention matrix — the state is a fixed-size accumulator — but
the portable `quad` backend deliberately evaluates it in the masked-matmul dual
form and does materialize the causal score matrix (O(T^2) memory) for
matmul-shaped throughput; `naive` and `fla` keep memory linear in `T`. Slow
model parameters are trained by normal end-to-end backpropagation. The fast
state is constructed inside each forward call and is not optimized by a
sequential inner training loop.

The public implementation currently recreates that state on every call. A
persistent state/streaming API is the first item in [ROADMAP.md](../ROADMAP.md).

### Projection layouts

- `share_key_query=True` learns one key projection and one query projection
  shared across heads, then adds zero-initialized per-head biases. Values and
  memory parameters remain per head.
- `key_value_heads=G` keeps all `H` query heads and learns `G` key/value groups,
  repeating each group across `H/G` contiguous heads. `H=8, G=4` is pairwise
  Transformer-style GQA.
- Leaving both options disabled gives independent per-head Q/K/V projections.

The first two options are mutually exclusive. `output_gate` independently
enables an input-dependent gate after the memory read.

### RoPE

Input-space RoPE is shared by both families. `mode="partial"` rotates the first
pair-aligned fraction of every key/query head, `"full"` rotates every paired
channel, and `"none"` leaves positional encoding to the host model. The same
rotation is used for writes and reads, so causality and scan associativity are
unchanged.

`placement="feature"` is an experimental signed-feature ablation. Rotating
after a non-negative feature map destroys its positive-kernel interpretation
and can drive a feature-mass denominator through zero. Active feature-space
RoPE is therefore rejected for every normalized GN or kernel read. It remains
available only for unnormalized GN, softmax-partition, and ReLU-squared reads;
projected B-splines reject it because it also breaks their partition structure.
Use input placement for ordinary experiments.

## 2. Gauss--Newton fast-weight memory

GN treats the parameters of a small memory MLP as fast state. At token `t`, it
linearizes the MLP at shared slow reference parameters `theta_0` and computes a
correction `Delta theta_t` that reduces the token's write residual. The scan
accumulates those token-local corrections and evaluates the corrected memory at
the current query.

`GNConfig.jacobian_steps` selects one or two local write steps. Two steps
recompute the residual after the first step's update and deposit a second
stream. Both W1 and W2 are updated in fast state.

The public nonlinearities are squared ReLU (`relu2`), squared ReLU with a
learned per-feature threshold (`relu2_threshold`), SiLU, and SwiGLU.

### GN numerical realization in v0.1

The public v0.1 forward always uses activation RMS normalization:

```text
nu(a) = 1 / sqrt(mean(a^2) + eps)
pre   = nu(a) * a
h     = sigma(pre - tau).
```

The exact RMSNorm derivative is `nu*I - (nu^3/m) a a^T`. The implemented GN
write deliberately drops the rank-one term and uses
`nu*diag(sigma'(pre-tau))` consistently in the hidden write and output-space
Gram. Note the direction of this approximation: the exact Jacobian
**suppresses** the radial direction (RMS normalization is scale-invariant
along `a`, and the rank-one term is exactly the correction that removes it),
so the scalar form **retains** that radial sensitivity rather than discarding
a small tail. What is dropped is a rank-one normalizing correction; the
compensation is that the write payload carries the same `nu` and the read
passes through the same RMS-normalized forward. The public solve is the
diagonal/Jacobi approximation to the resulting Gram. Consequently, v0.1 is a
shared-slow-reference, RMS-normalized approximate-GN realization; it is not a
literal exact full-Jacobian Gauss--Newton update. The ordinary unnormalized
read still evaluates its accumulated approximate parameter increments through
the full RMS-normalized forward. SwiGLU uses two independently RMS-normalized
branches and applies the same scalar-only derivative approximation to each.
For experimental `depth>1`, the local-Jacobian path treats downstream
residual blocks as identity and averages the block-local Gram contributions;
the public benchmark evidence uses `depth=1`.

### GN read normalization

Feature-mass normalization changes the read, not the GN payloads written by
each token. For the supported squared-ReLU modes, define a non-negative
slow-reference feature

```text
phi_i = ReLU^2(RMSNorm(W1 k_i) - tau)
phi_q = ReLU^2(RMSNorm(W1 q)   - tau),
```

where `tau=0` for ordinary `relu2`. Let `g_i` be the GN hidden-correction
payload and `lambda_i` the output-correction payload.

`read_normalization="none"` reads the accumulated W1/W2 corrections directly.
`"w2_feature_mass"` keeps the ordinary W1 read and normalizes the W2 read:

```text
L_t = sum_(i<=t) omega_(t,i) lambda_i outer phi_i
D_t = sum_(i<=t) omega_(t,i) phi_i
c2_t = L_t h_q / (h_q dot D_t + eps).
```

`"both_feature_mass"` also retrieves the hidden correction through a matched
statistic, then applies the nonlinearity before the normalized W2 read:

```text
G_t  = sum_(i<=t) omega_(t,i) g_i outer phi_i
c1_t = G_t phi_q / (phi_q dot D_t + eps)
h_q  = ReLU^2(RMSNorm(W1 q + c1_t) - tau)
c2_t = L_t h_q / (h_q dot D_t + eps).
```

This is a two-stage normalized GN memory. `G_t` is an associative
hidden-feature statistic, not the literal matrix `sum(Delta W1)`. It is more
structured than dividing deltas by token count and moves the read toward a
normalized kernel/attention-like estimator while preserving the original GN
write values `(g_i, lambda_i)`.

Both normalization modes require one Jacobian step and non-negative
`relu2`/`relu2_threshold` features. All three GN read modes compose with sum,
matched write-side EMA, and one or two recency branches.

## 3. Normalized kernel memory

Kernel memory fixes its nonlinear feature map within a sequence and accumulates
payload statistics directly. For a non-negative feature map `phi`, each
temporal view carries

```text
N_t = sum_(i<=t) omega_(t,i) phi(k_i) outer v_i
D_t = sum_(i<=t) omega_(t,i) phi(k_i).
```

The canonical key-mass read is

```text
R_t(q) = phi(q)^T N_t / (phi(q)^T D_t + eps).
```

The same weights `omega_(t,i)` must be used in `N_t` and `D_t`. This is the
normalized kernel-regression/Nadaraya--Watson estimator implemented by
`read_normalization="key_mass"`. `"none"` returns the unnormalized numerator
read. The experimental `"feature_mass"` mode first forms one mean payload per
feature, `N_t[j]/(D_t[j]+eps)`, then combines those means with the query
features; it is a diagonal per-feature estimator rather than the global ratio
above.

### Kernel kinds

`KernelConfig.feature_map` selects the address geometry:

| Kind | Feature geometry | Main controls |
|---|---|---|
| `softmax_partition` | `softmax(Γ ⊙ a(x))`, where `a(x)=RMSNorm(W1 L2Norm(x))` and `Γ` is the positive sharpness: normalized global competition between learned standardized linear scores. | Fixed, learned-per-head, or learned-per-feature positive sharpness; optional partition sparsity controls. |
| `relu2_ridge` | Normalize `ReLU(a(x)-tau)^2` across features; use a uniform address only when all features are zero. At `tau=0`, each pre-normalization feature has half-space support; learned `tau` gates the standardized scores. | Optional score bias or one zero-initialized learned threshold per head; these controls are mutually exclusive. |
| `projected_bspline` | Clamp scaled coordinates from `a(x)`, expand each learned direction in an open-uniform cubic B-spline basis, concatenate, then divide by the direction count. A cell is local along one projection and global in every orthogonal direction. | Basis count, cubic degree, coordinate bound, and fixed or learned-per-head projection scale; score bias is unsupported. |

### Why compare kernel maps with an MLP?

Evaluating `M` learned features has the same leading `O(M*d)` projection work
as an MLP layer with `M` hidden units, but equal leading cost does not imply
equal finite-width behavior. A normalized positive-feature read is a bounded
mixture of stored payloads; an unconstrained MLP can extrapolate outside that
mixture.

`softmax_partition` builds global competition between learned standardized
linear scores. At fixed normalization scale its decision boundaries inherit
the polyhedral geometry of linear-score competition, so it avoids the
full-dimensional center-coverage problem of a conventional radial-basis map.
Far along an input direction, one score can dominate and the read tends toward
that partition's prototype. `relu2_ridge` tests a global ridge bias while
retaining inexpensive numerator/mass writes; with its learned nonzero threshold,
the standardized-score boundary is not literally a fixed hyperplane.
`projected_bspline` is a middle point: locality is one-dimensional along learned
projection-pursuit directions and global in the orthogonal complement. A true
Euclidean radial-basis feature map is a possible future kernel feature map; it is not
what v0.1 calls `softmax_partition`. The current screen establishes that all
three published maps train, not that one geometry is universally best.

### Softmax-partition sharpness and sparsity

For `feature_map="softmax_partition"`, `kernel_sharpness_mode="fixed"` uses the configured scalar
`kernel_sharpness`. `"learned_per_head"` learns one positive value per head;
`"learned_per_feature"` learns positive diagonal logit calibration per head
and partition feature. The same controls are used for keys and queries.

Softmax-partition-only sparsity modes compare each normalized feature weight with a learned
fraction of the largest weight in that address. `relative_soft` uses a smooth
mask, `relative_st` uses an exactly sparse forward pass with a straight-through
gradient, and `relative_st_blend` learns a bounded interpolation between dense
and exactly sparse addresses. These are experimental controls; the safe blend
starts dense when `sparse_blend_init=0`.

For dense address `p`, learned relative threshold `rho`, and learned bounded
blend `alpha`, the safe-blend feature is

```text
phi = (1 - alpha) p + alpha SparseNormalize(p, rho).
```

`SparseNormalize` removes weights below `rho * max(p)` and renormalizes the
survivors. The dense path therefore remains available while the sparse path is
being learned.

### Values

`value_representation` selects the payload `v_i`:

- `raw` stores the projected value directly;
- `value_anchors` stores a distribution over a second learned value codebook;
- `value_mlp` transforms each value with a residual MLP before depositing it.

Value anchors require key-mass normalization. Freezing `feature_parameters_trainable`
freezes the internal kernel `W1` and learned address controls between optimizer
steps; the feature map is always fixed during an individual sequence scan. It
does not freeze the external token-to-key/query projections (`proj_k`, `proj_q`,
or their shared/grouped equivalents), which remain ordinary trainable model
parameters.

## 4. Temporal weighting

`TemporalConfig` is independent of the memory family:

- `mode="sum"` gives every prefix write equal weight;
- `mode="ema"` learns a data-dependent write-side decay;
- `mode="bank"` retains the sum state and adds one or two recency-weighted EMA
  views. `bank` is the API selector; **temporal-mode bank** and
  **recency branch** are the mechanism terms.

With `J=recency_branches`, a fast temporal-mode bank reads

```text
R_t = R_sum,t + sum_(j=1..J) eta_j (R_ema,j,t - R_sum,t).
```

Every blend starts at zero, so adding a branch does not change the initial
function. `blend_mode="free"` leaves `eta_j` signed and unconstrained;
`"tanh"` is the tanh-bounded variant with `eta_j in (-1,1)`. Retention can be
specified directly or by token half-life `H`, where
`alpha = 2^(-1/H)`.

For every normalized GN or kernel view, numerator and mass receive identical
temporal weights and are divided **before** completed reads are blended. A sum
view plus two recency branches therefore has three independently normalized
reads. It is not generally equivalent to blending numerators and denominators
first.

## 5. Random feature expansion

`feature_expansion` decouples the effective memory width from the trainable
memory parameter count. With effective width `m = memory_multiplier *
head_dim` and factor `f`, the trainable factors are stored at base width
`b = m / f` and composed with fixed non-trainable sign maps:

```text
U in {-1,+1}^(m x b) / sqrt(b),   D in {-1,+1}^(b x m) / sqrt(m)
W1_eff = U @ W1_trainable          # [m, d_in]
W2_eff = W2_trainable @ D          # [d_out, m].
```

Every write, read, and normalization above operates on the effective weights:
the GN Jacobian write deposits corrections against `W1_eff`/`W2_eff`, and the
ReLU-squared ridge kernel evaluates its features through `W1_eff`. Fast state,
addresses, evidence, and per-feature learned controls (thresholds) live at the
full width `m`; the trainable slow parameters stay at width `b`.

Each map's sign pattern is derived from `expansion_key` by SHAKE-256 over a
domain string that includes the map role and shape, then row-normalized to
unit Euclidean norm. The maps are derived when the target module is
constructed, are excluded from `state_dict()`, and never train. A persistent
40-byte fingerprint records the versioned derivation identity instead;
strict state-dict loading rejects a different key, map shape, or derivation
schema before it can silently pair learned factors with the wrong maps. Equal
keys give bitwise-equal maps; a per-layer key suffix gives independent maps per
layer. Legacy expanded checkpoints without a fingerprint require an explicit
`strict=False` load after the caller verifies the key. `f=1` bypasses the
mechanism and is bitwise identical to the dense parameterization.

The linear part of each effective factor has rank at most `b`; the per-feature
nonlinearity (and its full-width learned thresholds) is what distinguishes the
`m` expanded features. Supported combinations: GN with `relu2` or
`relu2_threshold`, and the kernel `relu2_ridge` feature map.

## 6. Scope of v0.1

The implemented axes above are research controls, not a claim that every cross
product is equally meaningful. Validation rejects incompatible combinations.
In particular, normalized GN requires one-step squared-ReLU features, kernel
memory currently requires `depth=1`, random feature expansion requires
squared-ReLU families and a divisible width, and feature-space RoPE is
experimental.

The benchmark evidence is single-seed architecture screening. It supports the
trainability of the named mechanisms and motivates further tests; it does not
establish a converged ranking, universal long-context behavior, or a final
capacity law.
