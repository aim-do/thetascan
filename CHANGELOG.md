# Changelog

ThetaScan uses semantic versions for the software API. Paper versions are
tracked separately in the paper itself.

## 0.1.0 - 2026-07-24

First public research release.

- Added GN fast-weight memory (Gauss--Newton/Jacobian writes, one or two
  steps) and the canonical normalized-kernel memory family.
- Added learned softmax-partition, global ReLU-squared ridge, and projected
  cubic B-spline kernel maps under `family="kernel"` and `KernelConfig`.
- Added output-stage and two-stage feature-mass normalization for GN, and
  global key-mass plus experimental per-feature normalization for kernels,
  with fixed or learned partition sharpness, learned ReLU-squared thresholds,
  and learned B-spline coordinate scales.
- Added **random feature expansion**: trainable memory factors stored at a
  base width and expanded to the effective width by fixed key-derived sign
  maps (`feature_expansion`, `expansion_key`); state and addresses grow while
  trainable memory parameters do not.
- Added sum, selective EMA, and a temporal-mode bank with one or two learned
  recency branches.
- Added shared-QK, grouped-query, and independent projection layouts plus
  optional partial or full RoPE.
- Added configuration snapshot semantics, `to_dict()`/`from_dict()` manifests,
  centralized regularization, and explicit streaming-state roadmap scope.
- Added versioned reference presets `gn_reference_v0_1`,
  `gn_expanded_reference_v0_1`, and `kernel_expanded_reference_v0_1`.
- Restricted the `cumsum` backend to undecayed sums: its decayed form loses
  precision on long sequences, and validation now rejects it with `ema` or
  `bank` temporal modes.
- Added a pinned Parameter Golf bootstrap, staged runpod configurations for
  the full 7,500-step schedule, exact parameter contracts, and a preliminary
  versioned theory report (Public Preview v0.1).
- Added the completed single-seed five-arm 7,500-step league (attention,
  Mamba-3 parity, dense GN reference, GN expanded 2x, kernel expanded 2x)
  through terminal warmdown as a compact reviewed research record.

The reported experiments are single-seed measurements. This release does not
claim a converged architecture ranking or a completed parameter search.
