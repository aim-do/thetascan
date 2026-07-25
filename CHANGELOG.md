# Changelog

ThetaScan uses semantic versions for the software API. Paper versions are
tracked separately in the paper itself.

## Unreleased

Execution-only work. No algorithm, parameterization, preset, or checkpoint
contract changed, and no result in `benchmarks/parameter-golf/results/` is
affected.

- **Fixed long-context rotary positions.** The RoPE table formed its position
  indices in the activation dtype. A bfloat16 `arange` stops resolving adjacent
  integers above 256, so past that length neighbouring positions shared a phase
  before the sine was evaluated and the rotation carried no relative
  information at exactly the lengths it exists for. Positions and trigonometry
  are now formed in float32 (float64 for a float64 caller) and only the
  finished rotations are cast. The table is also memoized, since one forward
  requests the same immutable table once per query, key, layer and checkpoint
  recomputation.
- **The `chunk` backend issues every tile in one batched call.** It previously
  walked its tiles in Python, so the operation count grew as `T / scan_chunk`
  and, at long context with small per-tile tensors, wall time tracked that
  count rather than the arithmetic. The tiles are now a leading batch axis of
  one `[B, H, T / scan_chunk, scan_chunk, scan_chunk]` score, and the
  cross-chunk state is computed in parallel for both promoted temporal views:
  the plain sum is an exclusive cumulative sum over the tile axis, and a static
  per-head retention is the same sum against a fixed geometric weight over that
  axis. A per-token retention stream keeps its sequential state, because the
  parallel alternative materializes the reciprocal decay this backend refuses
  to form. Semantics are unchanged -- bitwise identical for the sum and
  per-token views, `1.5e-16` for a static retention, worst gradient `3.6e-16`,
  and `scan_chunk >= T` still reproduces `quad` bit for bit. Two consequences:
  the largest retained tensor is now the whole tiled score rather than one tile
  (the same total bytes, still linear in `T`, in one allocation), and raising
  `scan_chunk` no longer buys anything, because it was never a launch-count
  knob after this and the `T * scan_chunk` score term makes a larger tile cost
  both time and memory.
- **A normalized read no longer forms a separate key mass.** The denominator
  `q . sum_j omega_j k_j` was built as a running `[B, H, T, m]` cumulative sum
  over a non-innermost axis, which scans the whole sequence over only
  `B * H * m` lanes -- and duplicates a prefix the scan beside it already
  computes. It now rides that scan as one extra unit-valued payload column, so
  every temporal view keeps exactly the mass it had: sum, EMA, the stale
  difference of a bank, and zero for the zero-memory base read. Per-feature
  normalization divides the query by the mass before the scan and therefore
  cannot fold; its prefix sum now moves `T` to the innermost axis so it uses
  the parallel per-row scan. Verified against the brute-force oracle in
  float64 for the sum, EMA, and both bank views. In reduced precision the
  denominator's rounding changes: it accumulates through the scan's float32
  cross-chunk state and bfloat16 tile matmul instead of a separate float32
  prefix followed by a bfloat16 dot.
- Documented that `torch.compile` and `torch.utils.checkpoint` do not compose
  here. Compiling a model whose residual blocks are activation checkpointed
  leaves the elementwise work unfused and measured as a complete no-op; the two
  levers only pay together. Nothing in the library changed for this -- it is a
  property of how a caller assembles the surrounding model -- but the previous
  note claiming compilation could not close the tile-loop gap is no longer the
  relevant statement, because there is no tile loop.
- Added the `chunk` scan backend and the `RuntimeConfig.scan_chunk` tile size.
  It forms the causal score one `[scan_chunk, scan_chunk]` tile at a time and
  carries an explicit `[Dk, Dv]` state between tiles, so activations grow
  linearly rather than quadratically in sequence length. It covers exactly the
  two promoted temporal views, the plain causal sum and the static per-head
  recency retention, needs no compiled extension, and runs on CPU and CUDA.
  Peak memory measured against `quad` on one H100 in bfloat16 across all three
  reference presets: 1.17x-1.54x lighter at `T=1024` and 2.80x-3.82x lighter at
  `T=4096`. No wall-time comparison is published: which backend is faster
  depends on sequence length, on whether the model is compiled, and on the
  device. Use `benchmarks/scan_backends.py` to measure a specific
  configuration.
- **Changed the `auto` backend.** It now always selects `chunk`, on every
  device and for every temporal mode, and never selects `fla`. Previously `auto`
  selected `fla` whenever the package was importable, including for every
  reference preset, all of which use a recency bank. On Hopper with Triton
  `>= 3.4` that combination raises from FLA's own gated backward
  (`chunk_bwd_dqkwg`, upstream issue #640), so a preset that trained under
  `auto` with `flash-linear-attention` installed failed on the first backward;
  installing `tilelang` as the error suggests does not help. And for the one
  case FLA does support, the ungated plain sum, its compiler-disabled kernel
  wrapper breaks a compiled model into several graphs and measured slower than
  the portable tile at every length tried. `fla` remains available as an
  explicit choice for an eager long-sequence plain sum. The published v0.1.0
  measurements are unaffected: they ran without that package, where `auto`
  resolved to `quad`.
- Reproducing a published v0.1.0 number bit for bit now requires
  `backend="quad"` explicitly, or equivalently `scan_chunk >= T`. `chunk`
  computes tile-local retention exponents, which is a different summation order
  from the global cumulative sum in `quad`; parity is bounded closeness in
  reduced precision and exact in float64.
- Replaced the unused private `chunk_size` field with `scan_chunk`, which now
  drives both the tiled scan and the retention-weighted mass.
- Added `benchmarks/scan_backends.py`, a same-process backend A/B that replays
  the backend list in reverse so ordering cannot be mistaken for an effect, and
  skips a backend that fails at build, forward or backward instead of losing
  the whole run.
- **Removed every wall-time figure from the released text**, including the
  step-time column of the step-7,500 league table in the paper and in the
  evidence record. Two reasons, either one sufficient. The arms were not run
  under a matched execution and compilation policy, so a per-step figure
  compared implementations rather than architectures. And a compiled measurement
  taken with the default zero-initialized recency blend is not a measurement of
  the configured model: with a zero blend the recency branches contribute
  nothing and can be optimized away, so such a figure understates the work the
  model actually does. Raw per-step values remain in the collected training
  logs. Nothing was restated in their place, because a replacement needs every
  arm re-measured together under one policy.
- **Fixed one cause of a compilation defect found while measuring the above.**
  The mixer resolved its scan backend lazily and cached it on the module, so the
  first traced forward guarded on `self._backend is None` and the second
  invalidated that guard -- one recompile of the forward per compiled model, in
  every temporal mode. Resolution now happens each call; it is pure Python over
  config fields and the input device, so a compiler folds it away. A regression
  test counts traced graphs for the sum, EMA and bank modes and requires one.
- A second cause remains and is **not** fixed: with an active retention the read
  closure reads a list of freshly allocated retention tensors as closure state,
  which TorchDynamo weak-references and invalidates every step. Measured on an
  H100 in fresh processes, compilation is worth substantially less with a
  retention active than without one. Fixing it means passing the temporal state
  as explicit arguments rather than free variables.
- Split the roadmap: [ROADMAP.md](ROADMAP.md) keeps public direction, and the
  measured execution internals moved to an untracked private file.
- Added a multi-view read capability -- `scan_chunk.linattn_views`,
  `interface.scan_views`, `engine.dual_read_views` -- which forms one score tile
  for every temporal view of a recency bank instead of one per view, and states
  each view as a signed combination of plain scans so a stale view needs no scan
  of its own. `linattn` and `dual_read` are now one-view wrappers over it. The
  mixer does **not** yet use the grouped form, because handing the engine a
  container of view objects triggers the recompile fallback noted above.

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
