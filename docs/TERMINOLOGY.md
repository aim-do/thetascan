# Terminology

ThetaScan uses the following mechanism-level terms across the library,
documentation, paper, and new experiment manifests.

**ThetaScan** is the ASCII project, package, and citation spelling. The Greek
display form **ΘetaScan** may appear in mathematical prose or artwork; it does
not name a separate algorithm or software package.

| Canonical term | Meaning | Public API |
|---|---|---|
| **slow memory** | The trainable dictionary of a memory block (`W1`, `W2`, thresholds, controls): which token directions exist and how nearby tokens are separated. Trained by ordinary backpropagation, fixed while one sequence is scanned. | Trainable module parameters |
| **fast memory** | The per-sequence state accumulated by the associative scan: where the current sequence's tokens sit in the slow dictionary. Rebuilt for every forward call. | The scan state; no persistent public handle in v0.1 |
| **GN memory** | Gauss--Newton/Jacobian writes into a nonlinear fast-weight MLP. | `family="gn"` |
| **normalized kernel memory** | Positive feature-map numerator/mass scan with normalized regression reads. | `family="kernel"` |
| **softmax-partition kernel** | Normalized global competition between learned standardized linear scores. | `KernelConfig(feature_map="softmax_partition")` |
| **partition-feature sharpness** | Positive scalar, per-head, or per-feature calibration of softmax-partition logits; these features are not radial centers or anchors. | `kernel_sharpness_mode="fixed"`, `"learned_per_head"`, or `"learned_per_feature"` |
| **feature parameters** | Internal slow kernel-map weights (`W1`) and learned address controls, fixed while one sequence is scanned. External token-to-key/query projections are not included. | `feature_parameters_trainable` controls only their outer-loop training. |
| **ReLU-squared ridge kernel** | Positive global ridge features; zero threshold gives half-space support, while learned thresholds gate standardized scores. | `feature_map="relu2_ridge"` |
| **projected B-spline kernel** | Cubic spline cells along learned scalar projection directions; a projection-structured normalized-score map. | `feature_map="projected_bspline"` |
| **random feature expansion** | Trainable memory factors stored at a base width and expanded to the full effective width by fixed key-derived sign maps; state grows, trainable parameters do not. | `feature_expansion`, `expansion_key` |
| **expansion key** | Deterministic string namespace from which the fixed expansion maps are derived; equal keys give equal maps. | `expansion_key` |
| **numerator `N_t` / mass `D_t`** | Matched normalized-kernel statistics. | Avoid switching to `S_t/z_t` in new descriptions. |
| **temporal-mode bank** | Sum state plus one or two recency-weighted EMA views. | `temporal.mode="bank"` |
| **recency branch** | One EMA view in a temporal-mode bank. | Counted by `recency_branches`. |
| **free blend** | Unconstrained signed temporal blend coefficient. | `blend_mode="free"` |
| **tanh-bounded blend** | Signed blend mapped through `tanh`, hence in `(-1,1)`. | `blend_mode="tanh"` |
| **per-head biases** | Zero-initialized head-specific offsets added after shared K/Q projections. | Use “biases,” not “offsets,” in public descriptions. |
| **tiled scan** | Execution strategy, not a mechanism: the causal score is formed in square time tiles, issued together, and an explicit carried state links consecutive tiles, so activations grow linearly rather than quadratically in sequence length. Every backend computes the same memory. | `runtime.backend="chunk"` |
| **scan tile** | The time extent of one such tile. Call it a tile, not a block or a window: “block” already names a memory depth level and “window” a temporal restriction. | `runtime.scan_chunk` |

Result summaries and public manifests use these terms consistently. Internal
research provenance is represented by source hashes rather than by exposing
discarded pre-release names in the public API.
