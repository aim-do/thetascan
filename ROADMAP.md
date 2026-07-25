# ThetaScan roadmap

ThetaScan v0.1.0 is a research preview. This roadmap records the directions that
matter for the public API and for the science, without promising a release date
or a particular implementation.

## Priority: persistent streaming state

`ThetaScan.forward(x)` creates a fresh fast-memory state for the sequence it is
given. It does not return a reusable state and cannot continue the same causal
sequence across calls. This is the largest missing library feature for
long-context inference.

A public streaming API should expose an explicit, typed state rather than private
tensors; support initialization, continuation, and deliberate reset; preserve
numerical parity with a single full-sequence call for every family,
normalization, temporal view, and expansion setting; document device, dtype,
detach, cloning, and serialization semantics; make the state's memory cost
inspectable; and keep chunk boundaries from changing causal results.

Until it exists, split a sequence into separate calls only when separate memory
histories are intended. Repeated `forward` calls are not streaming continuation.

## Science

- **Other nonlinearities.** SiLU and SwiGLU memory networks are implemented and
  exposed but unscreened at scale. Squared ReLU carried every reported result;
  whether the sharpening exponent or the shape is what matters is open.
- **More feature maps, and mixtures of them.** The three published maps are
  points in a larger space, and nothing forces a single map per head.
- **Separating state size from read mathematics.** The GN arms lead the kernel
  arm, and two explanations are confounded: the GN state carries an extra
  address-correction channel, and its read re-derives the query address through
  the state. A targeted pair of arms would separate them.
- **Replication.** Additional seeds on the same staged protocol. Every published
  ranking is single-seed.
- **Long context.** Controlled recall under varied record count, similarity,
  rewrite rate, and query distance, plus length sweeps beyond the published
  context.
- **Optimizer.** A policy-crossed study separating the architecture from
  per-head Muon; the baselines were not retuned with it.
- **Depth beyond one.** Accepted but weakly validated.
- **A memory variant without the slow value decoder**, binding addresses
  directly to values, which composes naturally with random feature expansion.

## Execution

- **A fused scan kernel.** The portable evaluator is a loop of PyTorch matmuls.
  A kernel that keeps a score tile and the carried state on chip, and generates
  the fixed expansion in place rather than materializing it, is the standing
  engineering item. No wall-time claim is published for any backend; measure a
  specific configuration with `benchmarks/scan_backends.py`.
- **Compilation behaviour.** Whether `torch.compile` helps depends on the
  temporal mode, and closing that gap is worth more than any backend choice.
- **A separate decoding path.** Everything above concerns training.
  Autoregressive inference should update persistent state once per token, which
  is the streaming API above rather than a scan tile size.
- **A numerically honest decayed prefix-scan path.** The `cumsum` backend is
  restricted to undecayed sums because its clamped cumulative-product division
  loses precision on long sequences.

## Contributing

The persistent-state section should be tracked as an issue and linked back here.
Reproducible design proposals are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
