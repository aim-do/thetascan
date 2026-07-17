# Reproducibility

## CPU smoke test

Create a clean virtual environment, install the package, then run:

```bash
python -m pip install -e '.[benchmark]'
python -m unittest discover -s tests -v
python examples/configure_mixer.py --recipe gn-reference-v0.1
python examples/configure_mixer.py --recipe gn-expanded-reference-v0.1
python examples/configure_mixer.py --recipe kernel-expanded-reference-v0.1
```

The `benchmark` extra supplies the Python 3.10 `tomllib` compatibility package
used by the harness and RunPod launcher; on newer Python versions it adds no
runtime dependency.

The test suite exercises the public GN and normalized-kernel translators,
shared RoPE options, a two-step GN forward and backward pass,
both GN feature-mass reads under sum/write-side EMA/temporal-mode banks, all
three kernel kinds, matched kernel normalization with EMA, fixed/per-head/per-feature
softmax-partition sharpness, one- and two-recency-branch banks, random feature
expansion (shapes, key determinism, state-dict round trips, dense equivalence
at factor one, backend parity), value anchors, the value MLP, frozen key/query
feature parameters, config serialization/snapshot semantics, regularization,
explicit unsupported-combination errors, the benchmark presets, and agreement
of the portable `naive`/`quad` backends (`cumsum` participates for the
undecayed sum path it supports). The listed configuration examples exercise
the versioned publication recipes.


## GPU acceleration

Install the optional FLA dependency only after selecting a version compatible
with the installed Torch, CUDA, and Triton stack:

```bash
python -m pip install -e '.[fla]'
```

Use `RuntimeConfig(backend="auto")` first.  It provides a portable PyTorch
fallback; forcing `backend="fla"` is appropriate only after a local forward and
backward smoke test on the target GPU.

## Full language-model benchmark

### Why there are two training schedules

The public experiments use parameter matching to isolate architecture choices:
the harness keeps the complete model near 17.06M trainable parameters and
adjusts only the FFNs adjacent to the two replaced mixers. Exact parameter
counts and residual deltas are asserted at runtime. An int8+zlib artifact-size
figure is recorded because the host project's report format includes it, but
compressed bytes are not used as a proxy for model capacity.

The 1,000-step protocol is a low-cost screen. It validates at steps 250, 500,
750, and 1,000 and uses a 200-step terminal warmdown. Use it to detect numerical
failures and nominate configurations, not to claim a final architecture
ranking.

The long protocol declares `iterations=7500` and `warmdown_iters=750`, then
saves resumable segments. The reviewed public comparison completed the full
schedule in three stages (0 to 3,000, to 4,000, to 7,500), passing the
step-6,750 warmdown boundary to terminally cooled endpoints. Intermediate
full-learning-rate points are deliberately not numerically comparable to the
cooled 1,000-step screen or to the cooled endpoints. Each continuation resumed
the exact model, optimizer, RNG, gradient-accumulation count, and data cursor
from a verified checkpoint. The saved training-shard prefix binds every shard's
full-file SHA-256, name, byte count, and token count.

The source-only parameter-golf adapter is runnable and the repository includes
reviewed compact result sets and their exact suite configurations; raw pod
exports and checkpoints are deliberately excluded. The adapter prepares a
clean checkout from a pinned official revision, generates a small layer-swap
overlay, retains upstream license/notices, installs ThetaScan as a normal
package, pins the upstream dataset revision, records the ThetaScan source hash,
and generates a fresh shared seed at launch.
It is distributed with the repository/source archive rather than the minimal
runtime wheel.

```bash
python benchmarks/parameter-golf/prepare_harness.py ../thetascan-parameter-golf
python benchmarks/parameter-golf/run_benchmark.py ../thetascan-parameter-golf \
  --arm thetascan --recipe gn-expanded-reference-v0.1 --rope partial --dry-run
```

Remove `--dry-run` after installing the GPU dependencies and downloading the
upstream `sp1024` data. The default 20 steps are only a mechanism smoke test.
Users choose the actual iteration budget and obtain their own validation result.

The adapter also provides attention and strict official-Mamba-3 arms. The
comparison keeps the host project's official model profile: ordinary
FFNs use hidden width 1024. Mamba-3 block parity preserves both full mixers and
reduces only their two surrounding FFNs to 938. The generated harness prints
and validates the exact parameter counts and delta before training.

ThetaScan exposes three projection-layout arms through `--projection-layout`:
`mamba-shared`, `transformer-gqa`, and `independent`. Their two swapped FFNs use
hidden widths 1023, 832, and 576 respectively. The base GN arm has
17,059,928 parameters, exactly 16 more than the 17,059,912 attention baseline;
the small remainder cannot be removed with integer FFN-width granularity.
Reference variants add small learned controls on top of that base: the
expanded kernel arm (per-head thresholds, two recency branches) totals
17,059,976, while the expanded GN arm's full-width threshold vector is
returned exactly by a 1,020-wide swapped FFN, keeping 17,059,928. The harness
validates each declared count. Projection layouts remain an experimental axis.
The publication copy retains only the most informative long-run configurations
rather than the full early allocation matrix.

Every ThetaScan arm trains under the `muon-2d+theta` optimizer policy, which
routes the memory factors W1/W2 through batched per-head Muon; the attention
and Mamba-3 baselines use the host benchmark's `muon-2d` policy. League rows
therefore compare architecture-plus-policy pairs, not a pure mixer swap under
one optimizer.

RoPE is an explicit experimental axis. Run separate `none`, `partial`, and
`full` invocations while keeping the recipe and all other settings fixed; it is
not implied that the recommended RoPE preset wins every workload.

Published measurements, their qualifications, validation schedules,
and artifact-size records are indexed in
[`benchmarks/parameter-golf/results`](../benchmarks/parameter-golf/results/README.md).

The reviewed five-arm league — attention, Mamba-3 parity, the dense GN
reference, and the two 2x random-feature-expansion arms — completed through
the warmdown and is retained as a compact record in
[`2026-07-21-h100-continuation-7500-v1`](../benchmarks/parameter-golf/results/2026-07-21-h100-continuation-7500-v1/SUMMARY.md).
The two expanded arms exceed the host benchmark's compressed-artifact limit,
so their runpod configs carry `research_allow_oversize_artifact` at every
stage; the exact size figures are in the record.

The July 2026 historical exports bind the executed Python trees by SHA-256 but
report no clean Git revision, and their exact source snapshots are not
distributed. The current release implements and tests the same named recipes,
so it is suitable for new repetitions; it is not claimed to reproduce every
historical artifact bit for bit. New published runs should start from a clean,
tagged revision and retain both its Git identifier and computed source hash.

See the full setup, dependency, data, parity, and fair-comparison instructions
in [../benchmarks/parameter-golf](../benchmarks/parameter-golf/README.md).
