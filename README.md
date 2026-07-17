# ThetaScan

**Scan-parallel nonlinear memory for sequence models.**

Modern language models increasingly pair attention with fixed-state recurrent
token mixers — Jamba, Samba, and Kimi Linear are production-scale examples —
because a fixed-size state keeps memory and compute per token constant at long
context. ThetaScan is a fixed-state mixer built from a small nonlinear
network: its trained parameters act as a **slow memory** (a dictionary of
token directions) and a **fast memory**, accumulated by an exact associative
prefix scan, records where the current sequence's tokens sit in that
dictionary. No token-record cache grows with context, and there is no
sequential inner optimizer. In the completed parameter-matched 17M
language-model league (7,500 steps through terminal warmdown), both ThetaScan
GN arms finished **below the attention control on raw and quantized
validation loss**, and every ThetaScan arm finished far ahead of the Mamba-3
control (see [the snapshot and its caveats](#current-research-snapshot)).

**Research preview v0.1.0; snapshot updated 24 July 2026.** The implementation
and theory are public; architecture search, replication, streaming inference,
and optimized kernels are still at an early stage.

[Paper: *ThetaScan: Scan-Parallel Nonlinear Memory* (PDF)](paper/ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf) ·
[Paper source and versions](paper/README.md) ·
[User guide](docs/USER_GUIDE.md) ·
[API reference](docs/API.md) ·
[Algorithms](docs/ALGORITHMS.md) ·
[Terminology](docs/TERMINOLOGY.md) ·
[Reproduce experiments](docs/REPRODUCIBILITY.md) ·
[Roadmap](ROADMAP.md) ·
[Release notes](CHANGELOG.md)

## News

- **2026-07-24 — v0.1.0, first public release.** Two nonlinear memory
  families (Gauss--Newton fast weights and normalized positive-feature
  kernels), random feature expansion (doubled effective memory width at the
  dense trainable budget), temporal-mode banks, and versioned reference
  presets. Finished the 7,500-step post-warmdown parameter-matched league:
  the best arm ends below attention on raw and exact-int8 validation BPB.
  Paper *ThetaScan: Scan-Parallel Nonlinear Memory*, Public Preview v0.1.

Each future snapshot adds a dated entry here; details land in
[Release notes](CHANGELOG.md).

## Why ThetaScan?

Self-attention preserves a separate key--value record for every token. That
gives precise access to history, but attention is quadratic during training and
its inference cache grows with context. Fixed-state recurrent mixers compress
history into a fixed-size state, which is why production hybrids carry most of
their layers as recurrent mixers and keep only a few attention layers. The
quality of such a hybrid rests on two properties of its fixed-state layer:
how well nearby tokens stay **separated** inside the state, and how quickly
**cross-talk** between stored records degrades reads as context grows.

ThetaScan's answer is nonlinearity under a strict parallelism constraint. The
memory is a small network. Its slow, backpropagation-trained parameters decide
which token directions exist and how sharply nearby tokens are pulled apart;
its fast, per-sequence state records what the sequence wrote at those
directions. Every write is computed from the current token and the shared slow
parameters — never from the evolving fast state — so writes commute and the
whole memory is one exact associative prefix scan. The read stays nonlinear
and normalized: many stored tokens contribute, with sharp query-dependent
weights, the way softmax attention mixes values.

The library provides two complementary families:

- **ThetaScan-GN** writes damped local Gauss--Newton/Jacobian updates into a
  small nonlinear fast-weight network.
- **ThetaScan-Kernel** makes a Hebbian kernel deposit: each token adds its
  positive feature vector bound to its value, and the read is a normalized
  numerator/mass scan over those deposits. Its public feature maps are learned
  softmax partitions, global ReLU-squared ridges, and projected cubic
  B-splines.

Both families share public axes for projection layout, RoPE, temporal
accumulation, learned recency, read normalization, random feature expansion,
and regularization. A compact typed configuration replaces the research
prototype's large collection of interdependent flags.

**Random feature expansion** decouples state capacity from the trainable
budget: the trainable memory factors are stored at a base width and expanded
to the full effective width by fixed key-derived sign maps, so the state and
addresses grow while trainable memory parameters do not. The strongest
measured arm is the dense reference's trainable budget at doubled effective
width.

```text
(key, value) -> nonlinear write at the slow reference -> associative prefix scan
             -> normalized / multiscale read(query) -> token-mixer output
```

## Current research snapshot

### How to read the benchmark

We did not build a bespoke benchmark. An existing, fully specified open
training project is used as the host: its two middle attention layers are
replaced by the mixer under test at the same whole-model parameter count, and
the validation loss is compared. If the loss stays close, the layer is
replaceable — that is the question the benchmark answers. The harness trims
only the two adjacent FFNs to compensate for each mixer's parameter count,
reports the remaining exact delta, and fails if a model no longer matches its
declared allocation. The quantized ("exact-int8") column is the same
validation loss measured again after an int8 quantization round trip of the
weights.

Two schedules serve different purposes and should not be mixed in one ranking:

- **1,000-step screens** use validation every 250 steps and a 200-step terminal
  learning-rate warmdown. They cheaply reject broken ideas and identify
  mechanisms worth retesting. A cooled 1,000-step endpoint is not a prediction
  of a full-learning-rate checkpoint from a longer schedule.
- **The long league** declares a 7,500-step target with a 750-step warmdown
  and ran to completion in three exact checkpoint continuations
  (0→3,000→4,000→7,500). Its endpoints are terminally cooled measurements.

The protocol controls parameter count, data, seed within a comparison, and
validation schedule. Optimizer policy is part of each arm: the attention and
Mamba-3 baselines use the host `muon-2d` policy, while every ThetaScan arm
uses `muon-2d+theta` (batched per-head Muon on the memory factors), so rows
compare architecture-plus-policy pairs. The protocol does not remove seed
variance or prove that any architecture has reached its optimal
hyperparameters.

In the completed single-seed, matched-parameter 17M league, two middle
attention sublayers of a nine-layer decoder were replaced under one common
benchmark protocol. At the terminally cooled step-7,500 endpoint:

| Mixer in the two replaced layers | Raw bpb @ 7,500 | Exact-int8 bpb | Delta vs attention (int8) |
|---|---:|---:|---:|
| **ThetaScan-GN, 2x random feature expansion** | **1.2327** | **1.23830852** | **-0.00240866** |
| ThetaScan-GN reference (dense) | 1.2342 | 1.23984702 | -0.00087016 |
| Attention | 1.2349 | 1.24071718 | - |
| ThetaScan-Kernel ReLU-squared ridge, 2x random feature expansion | 1.2361 | 1.24215211 | +0.00143493 |
| Parameter-matched Mamba-3 hybrid control | 1.3194 | 1.32677632 | +0.08605914 |

The replacement goal is met: both GN arms ended below Attention on raw
**and** exact-int8 BPB, the expanded kernel arm finished close behind
Attention, and every ThetaScan arm finished far ahead of the Mamba-3 control
under this shared schedule. The expanded GN arm also improved on its dense
counterpart by 0.0015 int8 BPB at the same trainable memory budget — the
direct capacity observation behind random feature expansion. Residual
qualifications — seed count, artifact sizes, optimizer pairing — are recorded
in the league record below and in the paper's limitations section.

Full trajectories, configuration manifests, parameter contracts, and honest
qualifications are in the
[7,500-step league record](benchmarks/parameter-golf/results/2026-07-21-h100-continuation-7500-v1/SUMMARY.md)
and the [paper](paper/ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf).

The reported benchmark uses the portable quadratic reference backend. The
update algebra is scan-compatible, but linear wall-clock scaling still requires
a dedicated fused or FLA-backed length benchmark.

## Install and first forward pass

```bash
python -m pip install .
```

Use `python -m pip install -e .` instead when developing from the checkout.

```python
import torch
from thetascan import ThetaScan, ThetaScanConfig

config = ThetaScanConfig(
    d_model=256,
    n_heads=4,
    family="gn",
)
mixer = ThetaScan(config)
y = mixer(torch.randn(2, 128, 256))
assert y.shape == (2, 128, 256)
```

Versioned reference presets reproduce the architecture choices used by the
published screens. They are starting points for a user's own search, not claims
that one configuration is universally best:

```python
from thetascan import ThetaScan, ThetaScanConfig

gn_reference_v0_1 = ThetaScan(ThetaScanConfig.gn_reference_v0_1())
gn_expanded_reference_v0_1 = ThetaScan(
    ThetaScanConfig.gn_expanded_reference_v0_1()
)
kernel_expanded_reference_v0_1 = ThetaScan(
    ThetaScanConfig.kernel_expanded_reference_v0_1()
)
```

To print and run the documented configuration recipes without training:

```bash
python examples/configure_mixer.py --recipe gn-reference-v0.1
python examples/configure_mixer.py --recipe gn-expanded-reference-v0.1
python examples/configure_mixer.py --recipe kernel-expanded-reference-v0.1
```

For a controlled RoPE comparison, run the same recipe with `--rope none`,
`--rope partial`, and `--rope full`; do not change another axis between those
three runs.

## Public configuration hierarchy

```text
ThetaScanConfig
+-- family: gn | kernel
+-- memory_multiplier (effective width) / feature_expansion / expansion_key
+-- share_key_query / key_value_heads / output_gate
+-- gn
|   +-- nonlinearity: relu2 | relu2_threshold | silu | swiglu
|   +-- jacobian_steps: 1 | 2
|   +-- read_normalization: none | w2_feature_mass | both_feature_mass
+-- kernel
|   +-- feature_map: softmax_partition | relu2_ridge | projected_bspline
|   +-- value_representation: raw | value_anchors | value_mlp
|   +-- softmax partition: fixed/learned sharpness and optional relative sparsity
|   +-- ReLU2 ridge: optional learned per-head threshold
|   +-- B-spline: basis count/bound and fixed/learned projection scale
|   +-- feature_parameters_trainable: true | false
|   +-- read_normalization: none | key_mass | feature_mass
+-- rope: none | partial | full
+-- temporal
|   +-- mode: sum | ema | bank
|   +-- recency_branches: 1 | 2
|   +-- retention or half-life initialization and blend mode
+-- regularization
+-- runtime: auto | naive | quad | cumsum | fla
```

The release terminology separates mechanism names from compact configuration
fields: the trainable dictionary is the **slow memory** and the per-sequence
scan state is the **fast memory**; a **temporal-mode bank** is selected by
`temporal.mode="bank"`, its EMA components are **recency branches**
(`recency_branches`); learned softmax partition `kernel_sharpness` is
**partition-feature sharpness**; and `feature_expansion` selects **random
feature expansion**, whose fixed sign maps are derived from `expansion_key`.
These are the canonical v0.1 public names; pre-release laboratory aliases are
not part of the publication API.

Presets return validated instances of the same hierarchy:
`ThetaScanConfig.gn_reference_v0_1()`, `.gn_expanded_reference_v0_1()`, and
`.kernel_expanded_reference_v0_1()` are versioned release-reference presets.
The returned configuration objects remain editable for controlled ablations.

Configurations are serializable and validated on load:

```python
import json
from thetascan import ThetaScanConfig

manifest = ThetaScanConfig.gn_expanded_reference_v0_1().to_dict()
with open("config.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
with open("config.json", encoding="utf-8") as handle:
    restored = ThetaScanConfig.from_dict(json.load(handle))
```

`ThetaScan` snapshots a deep copy of the config when constructed. Mutating the
original dataclass later does not partially reconfigure a live module.

`runtime.backend="auto"` is intentional: it chooses FLA on a supported CUDA
input when the package is available, but retains a portable PyTorch fallback
for CPU, CI, and parity tests. The selection follows the current input device,
including after a module is moved between CPU and CUDA. FLA is an accelerator,
not the only definition of the algorithm.

`rope=RoPEConfig()` is a shared positional axis for both families. Its default
is `mode="none"`; the versioned reference presets provide partial-RoPE
starting points.
RoPE is task- and length-dependent, so a comparison should run all three
supported modes (`none`, `partial`, and `full`) while holding the other fields
fixed. See the `RoPEConfig` section of [API.md](docs/API.md) for `fraction` and
`base`. Input-space placement is the supported path for normalized reads;
feature-space rotation is restricted to explicitly unnormalized experimental
ablations because signed feature mass can cancel through zero.

## Library versus examples

The installed `thetascan` package contains only the mixer, its configuration,
and its regularization API. It deliberately does **not** contain a language
model, tokenizer, dataset, task, or optimizer recipe.

The single example, `examples/configure_mixer.py`, prints and runs the three
versioned reference recipes through the documented public configuration only.
Training integration lives in the parameter-golf benchmark adapter, not in a
toy demo.

## Possible combinations

Every public axis composes with both families unless a rule below says
otherwise, so the useful summary is which values each axis takes, not a
feature-support table.

Axes shared by both families:

- **Positional RoPE**: `none`, `partial`, `full` (input-space placement).
- **Projection layout**: independent per-head Q/K/V, `share_key_query`, or
  Transformer-GQA `key_value_heads` (the last two are mutually exclusive).
- **Output gating**: `output_gate` on or off.
- **Temporal axis**: `sum`, write-side `ema`, or `bank` (retained sum plus one
  or two EMA recency branches); every view keeps a matched numerator/mass
  read before blending.
- **Random feature expansion**: `feature_expansion` with key-derived fixed
  sign maps (`expansion_key`) — squared-ReLU families only (`relu2` /
  `relu2_threshold` for GN, `relu2_ridge` for kernel).
- **Orthogonality regularization** and the runtime backend selection.

Family-specific axes:

- **GN**: memory nonlinearity `relu2`, `relu2_threshold`, `silu`, `swiglu`;
  one or two Jacobian steps; read normalization `none`, `w2_feature_mass`, or
  two-stage `both_feature_mass`.
- **Kernel**: feature map `softmax_partition` (fixed or learned sharpness,
  per head or per feature, optional relative sparsity), `relu2_ridge`
  (optional learned per-head threshold), or `projected_bspline`; value
  representation `raw`, `value_anchors`, or residual `value_mlp`; read
  normalization `none`, matched `key_mass`, or experimental `feature_mass`.

The following limitations are explicit rather than silently accepted:

1. GN feature-mass normalization requires one Jacobian step and non-negative
   ReLU-squared features.
2. Random feature expansion requires the squared-ReLU families (`relu2` or
   `relu2_threshold` for GN, `relu2_ridge` for kernel) and an effective width
   divisible by the expansion factor. The fixed maps are derived from
   `expansion_key`, never trained, and never stored in checkpoints.
3. `runtime.backend="cumsum"` supports `temporal.mode="sum"` only; its decayed
   form loses precision on long sequences, and validation rejects it with
   `ema` or `bank`. `auto` never selects it.
4. Signed temporal-blend coefficients are deliberately not probabilities. Use
   the documented support matrix when composing them.

`KernelConfig(value_representation="value_anchors")` additionally requires
`read_normalization="key_mass"`. The full support matrix is in [API.md](docs/API.md).

Normalized kernel memory also supports `temporal.mode="ema"`: the same learned decay
weights are applied to the associative value numerator and the key-mass
denominator, so the normalized read remains a matched weighted ratio.

The temporal-mode bank (`temporal.mode="bank"`) is different from write-side
EMA: it retains the sum
state and adds one or two parallel EMA recency branches. This yields two or
three temporal views in total. Each normalized view has its own matched
numerator/mass read before the completed reads are blended. See
[Algorithms](docs/ALGORITHMS.md) for the ordering and equations.

`forward()` currently creates fresh fast state for each call. There is no
persistent streaming/cache API in v0.1, so separate chunks do not continue one
memory history. The intended state contract and parity requirements are the
first item in [ROADMAP.md](ROADMAP.md). Kernel memory requires `depth=1`; GN
depth above one is accepted but remains experimental.

## Scope and benchmark integration

The library does not vendor an experimental `parameter-golf` fork. A runnable
source-only bootstrap in
[benchmarks/parameter-golf](benchmarks/parameter-golf/README.md) starts from a
pinned official `openai/parameter-golf` revision, verifies its source blob, and
generates a clean attention/ThetaScan/Mamba-3 layer-swap harness. The generated
harness contains no checkpoints or dataset shards. Reviewed configurations,
sanitized logs, runtime provenance, and compact result records are published
under `benchmarks/parameter-golf/results/`.

The benchmark adapter is included in the Git/source distribution, while the
runtime wheel intentionally contains only the library and required license
notices. This public repository is a curated research snapshot; exploratory lab
work is kept separate and only reviewed configurations and results are brought
back here.

## Paper and citation

The preliminary **Versioned Technical Paper / Public Preview v0.1**,
*ThetaScan: Scan-Parallel Nonlinear Memory*,
(2026-07-24) is available as
[Markdown](paper/ThetaScan-Scan-Parallel-Nonlinear-Memory.md) and as a
[rendered PDF](paper/ThetaScan-Scan-Parallel-Nonlinear-Memory.pdf). It is a
citable snapshot of an ongoing parameter search, not an archival or final
paper; later numbered previews may incorporate new data and algorithms while
the prior version record remains visible. See [paper versions](paper/VERSIONS.md).
It uses the
collective byline **The ThetaScan Project**, allowing later versions to credit
substantive research collaborators without an anonymous author entry.

For software citation metadata, use [CITATION.cff](CITATION.cff) or GitHub's
**Cite this repository** action and cite the exact repository version used. The
software release is `0.1.0`; the accompanying paper is independently Public
Preview `v0.1`.

## Research collaboration

ThetaScan welcomes focused research collaboration on optimizer and learning-rate
design, faster convergence, persistent streaming state, fused scan/FLA kernels,
systematic parameter selection, replication, and long-context/load testing. Reproducibility reports
and design discussions may be opened as issues. Contact
[hi@aim.do](mailto:hi@aim.do) before beginning a substantial code contribution;
the patent and source-available licensing context requires a contributor
agreement before code can be merged. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

ThetaScan software and its accompanying software documentation are
source-available under the [PolyForm Small Business License 1.0.0](LICENSE), not
an OSI-approved Open Source license. For company use, both license conditions
must be met: fewer than 100 employees plus independent contractors, and less
than USD 1,000,000 (2019) total prior-tax-year revenue, adjusted for inflation
as specified in the license. Uses outside that grant require separate written
permission or another legal basis.

The copyright holder and software licensor is **Ultimamind SRL, Belgium**.
Licensing correspondence: [hi@aim.do](mailto:hi@aim.do).

See [LICENSING.md](LICENSING.md) for scope and commercial licensing,
[PATENTS.md](PATENTS.md) for the patent notice, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party boundaries.
Issues and reproducibility reports are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md) before submitting code.
