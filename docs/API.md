# API reference

ThetaScan is a PyTorch token mixer with one tensor contract:

```text
[batch, time, d_model] -> [batch, time, d_model]
```

The public API is intentionally configuration-first. Choose a memory family,
choose its small set of mechanism axes, construct `ThetaScan`, and add the
optional regularization loss to the host task loss.

## Minimal use

```python
import torch
from thetascan import ThetaScan, ThetaScanConfig

config = ThetaScanConfig(d_model=256, n_heads=4, family="gn")
mixer = ThetaScan(config)
x = torch.randn(2, 128, 256)
y = mixer(x)
assert y.shape == x.shape
```

For normalized kernel memory, select the kernel explicitly:

```python
from thetascan import KernelConfig, ThetaScan, ThetaScanConfig

config = ThetaScanConfig(
    d_model=256,
    n_heads=4,
    family="kernel",
    kernel=KernelConfig(feature_map="softmax_partition"),
)
mixer = ThetaScan(config)
```

`ThetaScan` deep-copies and validates its configuration at construction. The
module therefore has **snapshot semantics**: later mutation of the caller's
dataclass never partially reconfigures existing parameters or regularization.
Edit a config first, then construct a new module.

## What a user should choose

1. Set `d_model` and `n_heads` to match the host residual stream.
2. Choose `family="gn"` for a nonlinear MLP fast-weight memory or
   `family="kernel"` for normalized positive-feature regression.
3. For kernel memory, choose `feature_map="softmax_partition"`, `"relu2_ridge"`, or
   `"projected_bspline"`.
4. Start with one versioned preset or the defaults. Change one axis at a time.
5. To grow the effective memory width without growing trainable memory
   parameters, raise `memory_multiplier` and set `feature_expansion` (see
   [Random feature expansion](#random-feature-expansion)).
6. Compare `rope.mode="none"`, `"partial"`, and `"full"` for the target task.
7. Keep `depth=1` for v0.1 experiments.
8. Save `config.to_dict()`, the seed, package/source version, optimizer, data,
   and validation schedule with every result.

## `ThetaScanConfig`

```python
from thetascan import (
    GNConfig,
    KernelConfig,
    RegularizationConfig,
    RoPEConfig,
    RuntimeConfig,
    TemporalConfig,
    ThetaScanConfig,
)

ThetaScanConfig(
    d_model=512,
    n_heads=8,
    head_dim=None,
    memory_multiplier=2,
    feature_expansion=1,
    expansion_key="thetascan",
    depth=1,
    share_key_query=False,
    key_value_heads=None,
    output_gate=True,
    family="gn",
    gn=GNConfig(),
    kernel=KernelConfig(),
    rope=RoPEConfig(),
    temporal=TemporalConfig(),
    regularization=RegularizationConfig(),
    runtime=RuntimeConfig(),
)
```

| Field | Default | Accepted values | Meaning and action |
|---|---:|---|---|
| `d_model` | `512` | Positive integer divisible by `n_heads` | Host token width. Set it to the residual-stream width. |
| `n_heads` | `8` | Positive integer | Number of memory heads. |
| `head_dim` | `None` | `None` or positive integer with `head_dim*n_heads=d_model` | `None` derives `d_model/n_heads`. |
| `memory_multiplier` | `2` | Positive integer | Effective feature/hidden width relative to one head. It changes parameters, state, and compute. |
| `feature_expansion` | `1` | Integer `>=1` dividing `memory_multiplier*head_dim` | `1` is the ordinary dense memory. Above one, the trainable memory factors shrink to `memory_multiplier*head_dim / feature_expansion` and fixed random maps expand them back to the full width. See [Random feature expansion](#random-feature-expansion). |
| `expansion_key` | `"thetascan"` | Non-empty string | Deterministic namespace for the fixed expansion maps. Distinct keys give independent maps. |
| `depth` | `1` | Integer `>=1`; kernel requires `1` | Residual blocks inside the memory function. GN values above one are experimental and weakly validated; use `1` in v0.1. |
| `share_key_query` | `False` | Boolean | One shared key projection and one shared query projection plus per-head biases. Mutually exclusive with `key_value_heads`. |
| `key_value_heads` | `None` | `None` or divisor of `n_heads` | Transformer-GQA layout: all query heads, grouped K/V. Example: `4` for pairwise GQA with eight heads. |
| `output_gate` | `True` | Boolean | Enables the learned input-dependent output gate. |
| `family` | `"gn"` | `"gn"`, `"kernel"` | Primary algorithm choice. |
| `gn` | `GNConfig()` | See below | Active for `family="gn"`. |
| `kernel` | `KernelConfig()` | `KernelConfig` | Active for `family="kernel"`. Set `feature_map` explicitly in experiment manifests. |
| `rope` | `RoPEConfig()` | See below | Shared key/query positional axis. |
| `temporal` | `TemporalConfig()` | See below | Sum, write-side EMA, or a sum plus recency branches. |
| `regularization` | `RegularizationConfig()` | See below | Optional penalties returned by `regularization_loss()`. |
| `runtime` | `RuntimeConfig()` | See below | Backend selection. |

## `GNConfig`

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `nonlinearity` | `"relu2"` | `"relu2"`, `"relu2_threshold"`, `"silu"`, `"swiglu"` | Memory-MLP nonlinearity. Thresholded ReLU-squared learns a separate threshold for each memory unit. |
| `jacobian_steps` | `1` | `1`, `2` | Number of local Gauss--Newton write steps. Two recomputes the residual once and costs more. |
| `read_normalization` | `"none"` | `"none"`, `"w2_feature_mass"`, `"both_feature_mass"` | No normalization, output/W2 feature mass, or two-stage hidden-and-output feature mass. |

Rules:

- feature-mass modes require `jacobian_steps=1` and `relu2` or
  `relu2_threshold`.

Example:

```python
from thetascan import GNConfig, ThetaScanConfig

two_stage = ThetaScanConfig(
    family="gn",
    gn=GNConfig(
        nonlinearity="relu2_threshold",
        jacobian_steps=1,
        read_normalization="both_feature_mass",
    ),
)
```

## `KernelConfig`

Canonical construction is:

```python
from thetascan import KernelConfig, ThetaScanConfig

config = ThetaScanConfig(
    family="kernel",
    kernel=KernelConfig(feature_map="relu2_ridge"),
)
```

### Common fields

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `feature_map` | `"softmax_partition"` | `"softmax_partition"`, `"relu2_ridge"`, `"projected_bspline"` | Positive feature-map selector. Set it explicitly in experiment manifests. |
| `value_representation` | `"raw"` | `"raw"`, `"value_anchors"`, `"value_mlp"` | Payload representation stored in the numerator. |
| `value_anchors` | `8` | Integer `>=2` | Value-codebook size, active only for `value_anchors`. |
| `value_mlp_multiplier` | `1.0` | Positive float | Residual value-MLP hidden width relative to `head_dim`, active only for `value_mlp`. |
| `feature_parameters_trainable` | `True` | Boolean | Whether the internal kernel feature map (`W1` and learned address controls such as sharpness, thresholds, score biases, and spline scales) trains between optimizer steps. These parameters remain fixed inside each sequence scan. External token-to-key/query projections (`proj_k`, `proj_q`, or their shared/grouped equivalents) remain trainable when this flag is `False`. |
| `read_normalization` | `"key_mass"` | `"none"`, `"key_mass"`, `"feature_mass"` | Raw associative sum, global normalized ratio, or experimental per-feature/diagonal normalization. |

`key_mass` is the usual normalized kernel/Nadaraya--Watson read. Use it first.
`feature_mass` separately normalizes every feature slot before query mixing; it
is a different estimator and should be a labeled ablation. `value_anchors`
requires `key_mass`.

### Softmax-partition fields (`feature_map="softmax_partition"`)

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `kernel_sharpness` | `8.0` | Positive float | Initial positive scale for the learned linear partition logits. It is a temperature-like sharpness, not a radial bandwidth. |
| `kernel_sharpness_mode` | `"fixed"` | `"fixed"`, `"learned_per_head"`, `"learned_per_feature"` | Fixed scalar, one learned positive scale per head, or positive diagonal calibration for every partition feature. |
| `score_bias` | `False` | Boolean | Optional learned affine score bias. Not used by B-spline and mutually exclusive with the ReLU-squared threshold. |
| `sparsity` | `"none"` | `"none"`, `"relative_soft"`, `"relative_st"`, `"relative_st_blend"` | Softmax-partition-only learned threshold relative to each address's maximum feature weight. |
| `sparse_blend_init` | `0.0` | Float in `[0,1]`; blend mode only | Initial dense-to-sparse mixture. Zero reproduces the dense base exactly. |
| `relative_threshold_init` | `0.01` | Float in `(0,1)` | Initial threshold as a fraction of the maximum address weight. |
| `threshold_temperature` | `0.25` | Positive float | Smooth/straight-through threshold temperature. |

The learned sharpness modes and sparsity modes are currently defined only for
`softmax_partition`. The safe `relative_st_blend` path is preferable to interpreting a hard
threshold as a drop-in replacement for dense softmax; it can learn away from
the dense initial function.

### ReLU-squared ridge fields (`feature_map="relu2_ridge"`)

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `relu2_threshold_mode` | `"none"` | `"none"`, `"learned_per_head"` | Subtracts one learned, zero-initialized threshold from each head's ridge scores before squaring ReLU. |
| `score_bias` | `False` | Boolean | Alternative affine score offset. Do not enable it with a learned ReLU-squared threshold. |

ReLU-squared ridge features use the cheap kernel numerator/mass write rule. With the
zero threshold their pre-normalization supports are half-spaces. A learned
nonzero threshold is applied after score RMS normalization, so its boundary is
not necessarily a fixed hyperplane. Learned partition sharpness and partition
sparsity are not active for this feature map.

### Projected B-spline fields (`feature_map="projected_bspline"`)

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `bspline_basis_count` | `8` | Integer greater than degree | Number of fixed cells per learned scalar direction. `memory_multiplier*head_dim` must be divisible by it. |
| `bspline_degree` | `3` | `3` in v0.1 | Cubic B-spline basis. |
| `bspline_bound` | `3.0` | Positive float | Symmetric coordinate range covered by the open-uniform partition. |
| `bspline_scale` | `1.0` | Positive float | Initial coordinate scale. |
| `bspline_scale_mode` | `"fixed"` | `"fixed"`, `"learned_per_head"` | Fixed scale or one learned positive scale per head. |

Projected B-spline rejects score bias, feature-space RoPE, and random feature
expansion.

## `RoPEConfig`

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `mode` | `"none"` | `"none"`, `"partial"`, `"full"` | No rotation, a leading pair-aligned fraction, or all paired channels. |
| `fraction` | `0.5` | Float in `(0,1]` | Rotated fraction for partial mode. |
| `base` | `10000.0` | Positive float | Rotary frequency base. |
| `placement` | `"input"` | `"input"`, `"feature"` | Rotate projected keys/queries before the nonlinear map, or experimentally rotate nonlinear features before scanning. |

Input placement is the established path. An active feature-space rotation can
make a positive feature map signed, so it is rejected whenever the read uses a
feature-mass denominator: GN `w2_feature_mass`/`both_feature_mass` and kernel
`key_mass`/`feature_mass` require input-space RoPE or `mode="none"`.
`projected_bspline` rejects active feature-space RoPE in every normalization
mode because the rotation also breaks its partition structure. Feature-space
RoPE remains available only as an experimental ablation for unnormalized GN,
softmax-partition, and ReLU-squared reads. Compare RoPE modes with every other
axis fixed.

## `TemporalConfig`

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `mode` | `"sum"` | `"sum"`, `"ema"`, `"bank"` | One unweighted prefix sum; one exponentially decaying state with a learned per-head data-dependent retention, and no sum alongside it; or a temporal-mode bank, the sum plus one or two recency reads blended into it. |
| `bank_mode` | `"fast"` | `"fast"`, `"stale"` | `fast` blends toward each EMA read; `stale` is the complementary historical view. |
| `retention_init` | `0.9` | Float in `(0,1)` | Initial retention for a one-branch temporal-mode bank. |
| `recency_branches` | `1` | `1`, `2` | Number of recency branches. Two branches plus the sum give three temporal views. |
| `retention_inits` | `None` | Tuple/list with one value in `(0,1)` per branch | Explicit retentions. Mutually exclusive with half-lives. |
| `half_life_inits` | `None` | Tuple/list of positive token half-lives | Converted by `alpha=2**(-1/half_life)`. Mutually exclusive with retentions. |
| `blend_mode` | `"free"` | `"free"`, `"tanh"` | Unconstrained signed blend or tanh-bounded signed blend. Raw blends start at zero. |

For two branches with no explicit tuple, retentions resolve to `(0.9, 0.99)`.
Distinct timescales are required. In normalized memory, every view filters its
numerator and denominator with the same weights and divides before blending.

**Choosing a mode.** The modes differ in how many times the same write streams
are read, which is the dominant cost of the mixer. `sum` and `ema` are one read
each. A bank is the sum plus one read per recency branch, so the two-branch
reference recipe is three reads. `ema` therefore buys forgetting at one read
where a bank buys a learned mixture of timescales at three; if one timescale is
enough for the task, `ema` is the cheaper way to get it. `sum` additionally
compiles more cleanly than either retained mode -- see
[Compilation](#compilation) -- and is the only mode the optional `fla` backend
supports.

Example two-recency-branch temporal-mode bank:

```python
from thetascan import TemporalConfig

temporal = TemporalConfig(
    mode="bank",
    bank_mode="fast",
    recency_branches=2,
    half_life_inits=(8.0, 64.0),
    blend_mode="free",
)
```

## Random feature expansion

`feature_expansion` decouples the effective memory width from the trainable
memory parameter count. `memory_multiplier` always states the **effective**
width. At `feature_expansion=1` the trainable factors have that full width. At
`f >= 2` the trainable input factor `W1` and output factor `W2` shrink to the
base width `memory_multiplier * head_dim / f`, and two fixed non-trainable
sign maps expand them back:

```text
W1_effective = expand_map @ W1_trainable    # [width, base] @ [base, input]
W2_effective = W2_trainable @ reduce_map    # [output, base] @ [base, width]
```

Fast state, write addresses, evidence, and per-feature learned controls (for
example ReLU-squared thresholds) stay at the full effective width; the
trainable slow parameters do not grow with it.

The maps are Rademacher sign matrices with unit-norm rows, derived
deterministically from `expansion_key` by a cryptographic extendable-output
hash. They are built from the key when a module is constructed, are **not**
stored in `state_dict()`, and never train. Expanded modules instead store a
40-byte, versioned `_expansion_fingerprint` that identifies every derived map.
`load_state_dict(..., strict=True)` validates this fingerprint and rejects a
checkpoint produced with a different key, map shape, or derivation schema.
The same key always gives the same maps; distinct keys (for example a per-layer
suffix such as `"thetascan:layer-4"`) give independent maps.

Expanded checkpoints created before the fingerprint was introduced have no
such field. They fail closed under `strict=True`; load one with `strict=False`
only after independently verifying that its `expansion_key` matches the target,
then resave it in the current format. A present but malformed or mismatched
fingerprint is rejected even under `strict=False`.

```python
from thetascan import GNConfig, ThetaScanConfig

expanded = ThetaScanConfig(
    d_model=512,
    n_heads=8,
    memory_multiplier=6,      # effective width: 6 * 64 = 384 features
    feature_expansion=2,      # trainable factors at 192 features
    family="gn",
    gn=GNConfig(
        nonlinearity="relu2_threshold",
        read_normalization="both_feature_mass",
    ),
)
```

Rules:

- `feature_expansion` must divide `memory_multiplier * head_dim`;
- GN requires `nonlinearity` `relu2` or `relu2_threshold`;
- kernel requires `feature_map="relu2_ridge"`;
- `expansion_key` must be a non-empty string; and
- `feature_expansion=1` is bitwise identical to the ordinary dense memory.

## `RegularizationConfig`

| Field | Default | Applies to | Meaning |
|---|---:|---|---|
| `feature_weight` | `0.0` | GN and kernel | Mean squared off-diagonal row-cosine penalty within memory feature matrices. |
| `head_weight` | `0.0` | GN and kernel | Cosine penalty between flattened head-specific memory matrices. Shared matrices with no head pairs are skipped. |
| `value_mlp_weight` | `0.0` | Kernel `value_mlp` | Row-correlation penalty for the value MLP. No-op elsewhere. |

All weights must be non-negative. The complete penalty is computed inside the
core and returned by `mixer.regularization_loss()`.

## `RuntimeConfig`

| Field | Default | Accepted values | Meaning |
|---|---:|---|---|
| `backend` | `"auto"` | `"auto"`, `"naive"`, `"quad"`, `"chunk"`, `"cumsum"`, `"fla"` | `auto` always selects `chunk`. The explicit backends are useful for parity tests, profiling, and the one case where `fla` wins. |
| `scan_chunk` | `512` | integer `>= 1` | Time tile of the `chunk` backend. Other backends ignore it. |

Backend choice is execution, not an algorithm axis. Every backend computes the
same scan, and each is covered against the sequential `naive` oracle in float64.

`chunk` is the default. It forms the causal score one `[scan_chunk, scan_chunk]`
tile at a time and carries an explicit `[Dk, Dv]` state across tiles, so its
activation footprint grows linearly in sequence length where `quad` grows
quadratically.

Peak memory measured on one H100 in bfloat16 against `quad`, same process,
backend order reversed to cancel drift, at the default `scan_chunk=512`:

| Preset | T=1024 | T=2048 | T=4096 |
|---|---|---|---|
| `gn_reference_v0_1` | 1.42x lighter | 2.20x | 3.82x |
| `gn_expanded_reference_v0_1` | 1.17x | 1.62x | 2.51x |
| `kernel_expanded_reference_v0_1` | 1.50x | 2.02x | 2.80x |

No wall-time comparison is published for any backend. Which one is faster
depends on sequence length, on whether the model is compiled, and on the
device, and the differences are small enough at the reference length that a
single number would mislead. Measure your own configuration with
`benchmarks/scan_backends.py`, which runs the backends in one process and
replays the list in reverse so ordering cannot masquerade as an effect; pass
`--compile` to match a compiled training loop.

`scan_chunk` trades the two terms of the footprint, `T * scan_chunk` for the
score tiles against `(T / scan_chunk) * Dk * Dv` for the carried states, so the
lightest footprint is near `sqrt(Dk * Dv)` — roughly 128 at the reference
widths. Going that low is not free: small tiles are launch-bound, compilation
does not rescue them, and because the tile loop unrolls statically a small tile
also raises compile time. The default of `512` keeps the footprint reduction
without paying either cost.

### Compilation

`torch.compile` is applied by the host, not by this library: there is no
configuration field for it, and any backend can be compiled. It is worth
enabling, and what it delivers depends on the **temporal mode** rather than on
the backend.

`temporal.mode="sum"` traces to a single graph and compiles cleanly. A mode that
carries a retention -- `"ema"` and `"bank"` -- currently does not. The read hands
the engine an accumulator object that owns a freshly allocated decay stream;
TorchDynamo guards that tensor by identity, the guard is invalidated on every
step, and the frame eventually falls back. Results stay correct -- only the
speed-up is lost. This is a limitation of the current implementation rather than
of the algorithm, and it is independent of the selected backend.

Two consequences worth knowing before benchmarking. A compiled measurement is
only meaningful with a non-zero blend: at the default zero-initialized
`blend_mode` weights the recency branches contribute nothing and may be
optimized away, so the figure does not describe the configured model. And the
`chunk` tile loop unrolls statically, so a smaller `scan_chunk` raises compile
time in proportion to the tile count.

`quad` is the masked-matmul dual form. It materializes the full `[B, H, T, T]`
causal score, and a second tensor of that shape when a retention is active. It
remains the reference implementation and the backend the published v0.1.0
measurements used; `chunk` with `scan_chunk >= T` reproduces it bit for bit.

`cumsum` supports `temporal.mode="sum"` only and validation rejects it with
`ema` or `bank`: its decayed form divides by a clamped cumulative product and
silently loses precision once a long sequence's accumulated decay leaves the
guard range. `auto` never selects `cumsum`.

`fla` needs the optional `flash-linear-attention` dependency and CUDA tensors,
applies to `temporal.mode="sum"` only, and is never selected by `auto`. Two
measured reasons.

Its kernel wrapper is closed to the compiler, so a compiled model breaks into
several graphs around each call. It does still gain from compilation -- the work
around the scan is fused -- but it gains the least of the three backends, because
the scan itself cannot be, and measured on one H100 it ends up the slowest of the
three once compiled, at every length tried.

Without a retention and **without compilation** it is the fastest option at long
context, at some cost in peak memory. That is the one case worth selecting it for
by hand, and it is narrow: a compiled portable tile measured comfortably faster
than an uncompiled FLA at the same length, so compiling is the better move
whenever it is available.

It also cannot run a retention at all on the stack tested. On Hopper with
Triton `>= 3.4`, FLA refuses its own gated backward, so `temporal.mode="ema"`
and `"bank"` fail outright:

```
RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for
gated chunk_bwd_dqkwg (see #640). Please install tilelang
```

That failure is raised from backward, so it appears only once a step is
propagated, and installing `tilelang` as the message suggests does not help.
Selecting `fla` explicitly for a retained view is therefore expected to fail on
such a stack. Test a forward *and* a backward on the target
Torch/CUDA/Triton/FLA combination before forcing `fla`.

## Configuration serialization

`to_dict()` returns a detached, fully expanded nested manifest. `from_dict()`
reconstructs and validates it. Standard JSON encoders serialize tuple-valued
half-life/retention fields as arrays, which `from_dict()` accepts.

```python
import json
from thetascan import ThetaScanConfig

config = ThetaScanConfig.kernel_expanded_reference_v0_1()

with open("thetascan-config.json", "w", encoding="utf-8") as handle:
    json.dump(config.to_dict(), handle, indent=2)

with open("thetascan-config.json", encoding="utf-8") as handle:
    restored = ThetaScanConfig.from_dict(json.load(handle))
```

`from_dict()` also supports a versioned preset plus nested overrides:

```python
config = ThetaScanConfig.from_dict({
    "preset": "kernel_expanded_reference_v0_1",
    "d_model": 512,
    "n_heads": 8,
    "rope": {"mode": "none"},
    "temporal": {"half_life_inits": [16.0, 128.0]},
})
```

Supported preset names are listed below. Save the **expanded** `to_dict()` result
with experiment artifacts so later preset changes cannot make a manifest
ambiguous.

## Presets

| Preset | Family | Purpose |
|---|---|---|
| `ThetaScanConfig.gn_reference_v0_1()` | GN | Two-stage feature mass, ordinary ReLU-squared, partial RoPE, and one free-blend recency branch. Cap-compliant benchmark reference. |
| `ThetaScanConfig.gn_expanded_reference_v0_1()` | GN | The strongest measured arm: 2x random feature expansion at doubled effective width, thresholded ReLU-squared, two-stage feature mass, one recency branch. |
| `ThetaScanConfig.kernel_expanded_reference_v0_1()` | Kernel ReLU-squared ridge | 2x random feature expansion, learned per-head threshold, key-mass read, two recency branches at 8- and 64-token half-lives. |

Presets are starting points. In particular, RoPE is task-dependent; compare
none, partial, and full while holding the rest fixed.

## `ThetaScan` methods

| Method | Contract |
|---|---|
| `forward(x)` | Accepts and returns `[batch,time,d_model]`. Each call begins with fresh fast state. |
| `regularization_loss()` | Returns a scalar tensor containing every enabled regularizer. Add it to task loss. |
| `zero_output_projection_()` | Zeros the final projection in place and returns the module; useful for an identity-starting residual branch. |

Training example:

```python
prediction = mixer(x)
task_loss = criterion(prediction, target)
loss = task_loss + mixer.regularization_loss()
loss.backward()
optimizer.step()
```

## Streaming and state limitation

There is no public cache/state argument in v0.1. `forward()` recreates fast
state, so two calls on adjacent chunks are **not** equivalent to one call on
their concatenation. Do not use repeated calls as streaming continuation. The
required typed persistent-state contract and parity tests are specified in
[ROADMAP.md](../ROADMAP.md).

## Support summary

| Configuration | `sum` | `ema` | `bank`, one/two branches | Random feature expansion |
|---|---:|---:|---:|---|
| GN, unnormalized | yes | yes | yes | `relu2`/`relu2_threshold` only |
| GN ReLU-squared, W2 feature mass | yes | yes | yes | yes (one Jacobian step) |
| GN ReLU-squared, both feature masses | yes | yes | yes | yes (one Jacobian step) |
| Kernel softmax partition, key mass | yes | yes | yes | no |
| Kernel ReLU-squared ridge, key mass | yes | yes | yes | yes |
| Kernel value MLP, key mass | yes | yes | yes | `relu2_ridge` feature map only |
| Kernel value anchors | yes | yes | yes | `relu2_ridge` feature map only; key-mass reads |
| Kernel projected B-spline | yes | yes | yes | no; input-space RoPE only |

Validation is fail-loud. If a combination is rejected, treat that as an API
boundary rather than bypassing the public config with private `_core` fields.
