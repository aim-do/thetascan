# User guide: what to do first

ThetaScan is a token mixer, not a tokenizer, optimizer, complete language
model, or persistent streaming cache. Put it inside a residual sequence block,
train it with the host model, and record the complete configuration for every
experiment.

## 1. Install and verify

From the source checkout:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The default installation requires PyTorch. FLA is optional; first establish a
portable CPU/CUDA reference before adding it:

```bash
python -m pip install -e '.[fla]'
```

## 2. Start from one versioned configuration

Do not begin by combining every axis. Pick one reference, verify a forward and
backward pass, then change one field at a time.

```python
import torch
from thetascan import ThetaScan, ThetaScanConfig

configs = {
    "gn_reference_v0_1": ThetaScanConfig.gn_reference_v0_1(),
    "gn_expanded_reference_v0_1": (
        ThetaScanConfig.gn_expanded_reference_v0_1()
    ),
    "kernel_expanded_reference_v0_1": (
        ThetaScanConfig.kernel_expanded_reference_v0_1()
    ),
}

config = configs["gn_expanded_reference_v0_1"]
mixer = ThetaScan(config)
x = torch.randn(2, 128, config.d_model, requires_grad=True)
y = mixer(x)
(y.square().mean() + mixer.regularization_loss()).backward()
assert y.shape == x.shape
```

The two expanded references use 2x random feature expansion — a doubled
effective memory width at the dense reference's trainable parameter count —
because those arms produced the strongest completed 7,500-step measurements.
They remain research starting points, not universal defaults.

## 3. Choose a memory family

### GN memory

GN writes local Jacobian updates into a small nonlinear memory MLP. Use it
when the fast state itself should represent changes to a nonlinear function.

```python
from thetascan import GNConfig, ThetaScanConfig

gn = ThetaScanConfig(
    d_model=512,
    n_heads=8,
    family="gn",
    gn=GNConfig(
        nonlinearity="relu2",
        jacobian_steps=1,
        read_normalization="none",
    ),
)
```

The first controlled GN sequence should be:

1. `read_normalization="none"`;
2. `"w2_feature_mass"` with the same seed and schedule;
3. `"both_feature_mass"` with the same seed and schedule.

Feature-mass normalization requires one-step ReLU-squared features. Keep
`jacobian_steps=2`, SiLU, and SwiGLU as separate ablations rather than mixing
them into the first comparison.

### Normalized kernel memory

Kernel memory accumulates a payload numerator and matching feature mass. Select
the address geometry explicitly:

```python
from thetascan import KernelConfig, ThetaScanConfig

partition = ThetaScanConfig(
    family="kernel",
    kernel=KernelConfig(feature_map="softmax_partition"),
)

relu2 = ThetaScanConfig(
    family="kernel",
    kernel=KernelConfig(
        feature_map="relu2_ridge",
        relu2_threshold_mode="learned_per_head",
    ),
)

bspline = ThetaScanConfig(
    family="kernel",
    kernel=KernelConfig(
        feature_map="projected_bspline",
        bspline_basis_count=8,
        bspline_degree=3,
        bspline_bound=3.0,
        bspline_scale_mode="learned_per_head",
    ),
)
```

Use `read_normalization="key_mass"` first. It is the matched global
Nadaraya--Watson ratio. `feature_mass` is a different, experimental per-feature
normalization; `none` is an unnormalized sum and may grow with context.

The three feature maps test different inductive biases:

- `softmax_partition` gives normalized global competition between learned,
  standardized linear scores;
- `relu2_ridge` gives positive global ridge features. With zero threshold the
  active regions are half-spaces; a learned threshold instead gates the
  standardized score and need not retain a literal hyperplane boundary; and
- `projected_bspline` gives local cubic cells along learned one-dimensional
  projections while remaining global in orthogonal directions.

## 4. Choose the value representation

Start with raw values:

```python
KernelConfig(feature_map="softmax_partition", value_representation="raw")
```

Only then test:

```python
KernelConfig(
    feature_map="softmax_partition",
    value_representation="value_anchors",
    value_anchors=16,
)

KernelConfig(
    feature_map="softmax_partition",
    value_representation="value_mlp",
    value_mlp_multiplier=1.0,
)
```

Value anchors require key-mass normalization. The value MLP can be
regularized.

For softmax-partition addressing, compare fixed sharpness, one learned positive
scale per head, and learned per-feature diagonal calibration in separate runs.
Relative partition sparsity is also independent; the safest experimental form is
`sparsity="relative_st_blend"` with `sparse_blend_init=0.0`, which begins as the
dense address.

## 5. Add temporal views

First establish the plain sum, then write-side EMA, then a temporal-mode bank:

```python
from thetascan import TemporalConfig

plain = TemporalConfig(mode="sum")

write_ema = TemporalConfig(mode="ema")

dual_recency = TemporalConfig(
    mode="bank",
    bank_mode="fast",
    recency_branches=2,
    half_life_inits=(8.0, 64.0),
    blend_mode="free",
)
```

`mode="bank"` selects the mechanism, a **temporal-mode
bank**. Two recency branches plus the retained sum produce three reads. Each
branch begins with zero blend, and every normalized read uses matched temporal
weights in its numerator and denominator.

Compare `blend_mode="free"` with the tanh-bounded mode using the same
timescales and seed. Do not call either coefficient a probability.

## 6. Grow state with random feature expansion

When a family works at its dense width, the next capacity experiment is to
grow the effective width without growing trainable memory parameters:

```python
from thetascan import GNConfig, ThetaScanConfig

expanded = ThetaScanConfig(
    d_model=512,
    n_heads=8,
    memory_multiplier=6,     # effective width doubles versus the dense 3
    feature_expansion=2,     # trainable factors stay at the dense width
    family="gn",
    gn=GNConfig(
        nonlinearity="relu2_threshold",
        read_normalization="both_feature_mass",
    ),
)
```

`memory_multiplier` always states the effective width; `feature_expansion=f`
stores the trainable `W1`/`W2` factors at `1/f` of it and expands them with
fixed key-derived sign maps (see [ALGORITHMS.md](ALGORITHMS.md) section 5).
State, addresses, and per-feature thresholds live at the full width, so
recurrent-state bytes and compute grow while the trainable parameter count
stays at the dense reference. Compare a dense arm and an expanded arm at the
same trainable budget and report the state difference explicitly.

The maps are deterministic in `expansion_key`: equal keys reproduce equal
maps across constructions and checkpoint loads, and per-layer suffixes (as
used by the benchmark adapter) give each layer independent maps. GN requires
`relu2`/`relu2_threshold` features; kernel requires `feature_map="relu2_ridge"`.

## 7. Test RoPE independently

```python
from thetascan import RoPEConfig

no_rope = RoPEConfig(mode="none")
partial = RoPEConfig(mode="partial", fraction=0.5)
full = RoPEConfig(mode="full")
```

Run all three with the same seed and every other field fixed. Input placement
is the normal option. Active feature-space placement is an experimental
signed-feature ablation supported only by unnormalized GN,
softmax-partition, and ReLU-squared reads. It is rejected for normalized reads
because signed feature mass can cancel through zero, and it is always rejected
for projected B-splines because it breaks their partition structure.

## 8. Put the mixer in a residual block

A typical pre-normalized block is:

```python
import torch.nn as nn
from thetascan import ThetaScan

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model)
        self.mixer = ThetaScan(config)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
```

Use `mixer.zero_output_projection_()` when the substituted residual branch must
start as an exact identity. This changes initialization only.


## 9. Train and regularize

```python
from thetascan import RegularizationConfig

config.regularization = RegularizationConfig(
    feature_weight=1e-4,
    head_weight=1e-4,
    value_mlp_weight=0.0,
)

# Construct after all config edits: ThetaScan snapshots the config.
mixer = ThetaScan(config)

prediction = mixer(x)
task_loss = criterion(prediction, target)
loss = task_loss + mixer.regularization_loss()
loss.backward()
optimizer.step()
```

ThetaScan does not prescribe AdamW, Muon, or a training schedule. Published
Parameter Golf ThetaScan runs use Muon for eligible two-dimensional memory
matrices and Adam for small control tensors; reproduce that routing only when
comparing with those runs.

## 10. Save a reproducible manifest

```python
import json
from thetascan import ThetaScanConfig

with open("config.json", "w", encoding="utf-8") as handle:
    json.dump(config.to_dict(), handle, indent=2)

with open("config.json", encoding="utf-8") as handle:
    restored = ThetaScanConfig.from_dict(json.load(handle))
```

Save the expanded manifest, not only a preset name. Also save:

- ThetaScan version, Git revision, and source hash;
- seed and all optimizer parameter-group rules;
- data/tokenizer identifiers and revisions;
- model parameter count and artifact byte count;
- batch tokens, validation schedule, warmup, and warmdown; and
- backend plus Torch/CUDA/FLA versions.

`from_dict()` accepts a versioned preset with nested overrides for convenient
experiment generation, but `to_dict()` should be the persisted record.

## 11. Run controlled comparisons

For cheap selection, the public 1,000-step protocol validates every 250 steps
and warms down over the last 200. For longer evidence, declare the 7,500-step
target and 750-step warmdown even when stopping at a resumable intermediate
checkpoint. The reviewed long runs completed the full schedule through the
step-6,750 warmdown boundary to terminally cooled step-7,500 endpoints.

A cooled 1,000-step endpoint and step 1,000 of the long schedule are not the
same experiment. Compare only runs sharing seed, data, whole-model parameter
allocation, optimizer policy, and schedule. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the
[Parameter Golf adapter](../benchmarks/parameter-golf/README.md).

## 12. Known limits

- Every `forward()` call starts a fresh fast state. There is no public
  persistent state/streaming API yet; adjacent chunks passed in separate calls
  do not continue one memory. See [ROADMAP.md](../ROADMAP.md).
- Kernel memory requires `depth=1`. GN depth above one is accepted but remains
  experimental and is not established by the public result set.
- Portable reference backends establish semantics; a dedicated fused kernel is
  still needed for strong wall-clock scaling claims.
- Many cross-axis combinations are intentionally rejected. Do not bypass those
  checks through private `_core` fields.
- Published results are single-seed architecture screens, not a completed
  parameter search or a statistical ranking.
