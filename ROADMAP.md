# ThetaScan roadmap

ThetaScan v0.1.0 is a research preview. This roadmap records important public
API work without promising a release date or a particular implementation.

## Priority: persistent streaming state

The current `ThetaScan.forward(x)` call creates a fresh fast-memory state for
the supplied sequence. It does not return a reusable state and cannot continue
the same causal sequence across multiple calls. This is the largest missing
library feature for long-context inference.

A public streaming API should:

- expose an explicit, typed state rather than private implementation tensors;
- support `state=None` initialization, continuation, and deliberate reset;
- preserve numerical parity with a single full-sequence call for every
  supported family, normalization, temporal view, and expansion setting;
- document batch-size, device, dtype, detach, cloning, and serialization
  semantics;
- make state shape and memory cost inspectable without depending on private
  fields;
- support chunk boundaries that do not change causal results; and
- include CPU reference tests before optimized CUDA/FLA state updates are
  treated as stable.

Until this API exists, split a sequence into separate calls only when separate
memory histories are intended. Do not interpret repeated `forward` calls as
streaming continuation.

## Later work

- fused scan kernels and length-scaling benchmarks;
- replicated parameter searches and long-context/load tests;
- a numerically honest decayed `cumsum` path (the current prefix-scan form is
  restricted to undecayed sums because its clamped cumulative-product division
  loses precision on long sequences);
- a W2-less memory variant that binds addresses directly to values;
- a stable checkpoint/config manifest schema around `to_dict()` and
  `from_dict()`;
- broader validation of depth greater than one; and
- additional normalized feature maps and mixtures of feature maps.

When the repository is published, the persistent-state section above should be
opened as a tracked GitHub issue and linked back here. Reproducible design
proposals are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
