# Parameter-golf benchmark adapter

This directory contains a runnable, source-only adapter, a fail-closed RunPod
launcher, exact published configurations, and reviewed benchmark results. It
ships no model checkpoints, credentials, private orchestration, or dataset. The
benchmark is kept outside the installable `thetascan` package and imports
ThetaScan through its documented public API.

## What the bootstrap prepares

`prepare_harness.py` creates a new checkout from the pinned official
`openai/parameter-golf` revision recorded in `adapter_manifest.json`. That
repository supplies the MIT-licensed benchmark, data downloader, tokenizer, and
notices. The bootstrap verifies the immutable `train_gpt.py` Git blob, applies a
narrow, fail-loud layer-swap transformation, and places generated files under
`ours/` in the checkout. It never fetches an experimental fork, so none of that
fork's history, private result archives, or orchestration utilities enter the
prepared checkout. The pinned official checkout itself remains intact. SHA-256
hashes of the generated harness, public-API bridge, and revision-pinned data
downloader are written to
`ours/ADAPTER-MANIFEST.json`.

## Prepare a clean checkout

From the ThetaScan repository and the Python environment that will run training:

```bash
python -m pip install -e '.[benchmark]'
python benchmarks/parameter-golf/prepare_harness.py ../thetascan-parameter-golf
python -m pip install -r ../thetascan-parameter-golf/requirements.txt
python ../thetascan-parameter-golf/ours/download_data.py \
  --variant sp1024 --train-shards 1
```

The `benchmark` extra supplies the Python 3.10 `tomllib` compatibility package;
on newer Python versions it adds no runtime dependency.

The bootstrap enables Git long-path support for its clone, which is required by
some upstream paths on Windows. A short destination such as `C:\\pg-theta` is
still preferable on Windows if the surrounding workspace path is already deep.

The data command downloads the full validation split and one training shard for
a smoke run. The wrapper pins the Hugging Face dataset revision recorded in
`ADAPTER-MANIFEST.json`; it writes into the official checkout's `data/`
directory. Increase `--train-shards` for a longer run. Data remains in the
prepared checkout and is governed by the upstream dataset terms. The pinned
FineWeb export is identified by its dataset card as ODC-By 1.0 and is also
subject to the referenced Common Crawl terms; the exact license and terms URLs
are recorded in `adapter_manifest.json` and copied into the generated
`ADAPTER-MANIFEST.json`.

FLA acceleration is optional for ThetaScan:

```bash
python -m pip install -e '.[fla]'
```

The Mamba-3 arm is strict and contains no fallback. Install the official module
and kernels compatible with the target Torch/CUDA environment. The commit is
pinned by the install command and recorded in the manifest; runtime verifies
the installed `mamba3.py` SHA-256 against that commit:

```bash
python -m pip install \
  'mamba-ssm @ git+https://github.com/state-spaces/mamba.git@a14b1dff0454a3bc27d9eb31355dc01e4b2490ec'
```

## Run the arms

### Screening versus long trajectories

The benchmark is parameter-matched at the **whole-model** level. Mixer-specific
parameters are compensated by trimming only the FFNs around the two substituted
layers; the runner asserts every expected total. The separate 16 MB compressed
artifact contract is retained for Parameter Golf compliance, not treated as a
measure of architectural capacity.

Use the two published schedules for different decisions:

- a 1,000-step screen validates every 250 steps and warms the learning rate down
  over the final 200 steps. It is intended for inexpensive mechanism selection;
- a long run declares a 7,500-step target and 750-step warmdown, with resumable
  checkpoints at intermediate stops. The reviewed long comparison ran the full
  schedule: staged continuations at steps 3,000 and 4,000, then through the
  step-6,750 warmdown boundary to terminally cooled step-7,500 endpoints.

Therefore a cooled 1,000-step endpoint and a step-1,000 point extracted from the
7,500-step schedule answer different questions. Compare arms only within the
same seed, data, parameter allocation, optimizer policy, and schedule.

The default is a 20-step mechanism smoke run. A fresh seed is generated at
launch and printed by both the runner and harness; runnable configurations do
not prescribe a seed. Reviewed result records include the seed that produced
the measurement. An explicit `SEED` environment variable may be used when a
user needs to repeat a specific local run. `--rope preset` is the default: it
preserves each recipe's declared starting point. Versioned reference recipes
use partial RoPE; explicit modes override the recipe.

```bash
python benchmarks/parameter-golf/run_benchmark.py ../thetascan-parameter-golf \
  --arm attention

python benchmarks/parameter-golf/run_benchmark.py ../thetascan-parameter-golf \
  --arm thetascan --recipe gn-expanded-reference-v0.1 --rope partial \
  --backend auto --projection-layout mamba-shared

python benchmarks/parameter-golf/run_benchmark.py ../thetascan-parameter-golf \
  --arm mamba3
```

The runner intentionally executes one arm and one RoPE mode per invocation.
For a fair multi-arm comparison, pass the same explicit `SEED` environment
value to each invocation. Attention and Mamba-3 are external arms and do not
have a ThetaScan recipe. The only ThetaScan recipe values are
`gn-reference-v0.1`, `gn-expanded-reference-v0.1`, and
`kernel-expanded-reference-v0.1`.

The reference recipes are starting points, not hidden conclusions. Compare
`--rope none`, `--rope partial`, and `--rope full` in separate invocations with
the same recipe, seed, and all other settings held fixed.

For a resumable parameter-selection run, declare the final learning-rate target
separately from the current segment. The following trains only through step
3,000, but keeps the schedule defined through step 7,500 and saves a lossless
research checkpoint:

```bash
python benchmarks/parameter-golf/run_benchmark.py ../thetascan-parameter-golf \
  --arm thetascan --recipe gn-reference-v0.1 --rope partial --backend quad \
  --projection-layout mamba-shared --optimizer-policy muon-2d+theta \
  --iterations 7500 --segment-start-step 0 --segment-end-step 3000 \
  --val-start-step 1000 --val-every 250 --warmup-steps 20 \
  --warmdown-iters 750 --grad-accum-steps 8 --expected-train-shards 16 \
  --checkpoint-output research-checkpoint.pt
```

At 524,288 train tokens per step, 3,000 steps consume approximately 15.73
100-million-token shards. Download 16 shards for this segment; additional shards
would not be read. Before continuing to 7,500, append shards until 40 are
available. The checkpoint stores model and optimizer states, RNG states, the
global step, resolved gradient-accumulation count, and the exact shard cursor.
Every file in the stored training-shard prefix is authenticated by full-file
SHA-256 in addition to its name, byte count, and token count. Resume rejects a
changed source, architecture, optimizer contract, accumulation count, seed, or
original shard content/order. The LR warmdown begins at global step 6,750, not
at the temporary step-3,000 stop.

After verifying the checkpoint metadata and downloading 40 shards, continue the
same arm with the same seed and immutable configuration:

```bash
python benchmarks/parameter-golf/run_benchmark.py ../thetascan-parameter-golf \
  --arm thetascan --recipe gn-reference-v0.1 --rope partial --backend quad \
  --projection-layout mamba-shared --optimizer-policy muon-2d+theta \
  --iterations 7500 --segment-start-step 3000 --segment-end-step 7500 \
  --val-start-step 1000 --val-every 250 --warmup-steps 20 \
  --warmdown-iters 750 --grad-accum-steps 8 --expected-train-shards 40 \
  --resume-checkpoint research-checkpoint.pt \
  --resume-checkpoint-sha256 <sha256-from-research-checkpoint.json> \
  --resume-checkpoint-bytes <bytes-from-research-checkpoint.json> \
  --checkpoint-output research-checkpoint-7500.pt
```

Use `--dry-run` to inspect the command and material environment overrides,
including FFN and mixer widths, without loading data or starting GPUs.
Use `--data-path` and `--tokenizer-path` when the assets are stored elsewhere.
The external harness writes model artifacts into the prepared checkout. The safe
RunPod collector copies only the explicitly requested research checkpoint into
the gitignored `results/_local/` directory and verifies its SHA-256 and size.

## Mamba-3 block parity

The parity arm swaps exactly zero-based layers `4,5`. It retains the official
Parameter Golf 16 MB profile: ordinary block FFNs have absolute hidden width
`1024`. Only the FFNs around those two Mamba-3 mixers use `938`. Therefore each
swapped FFN removes `(1024 - 938) * 512 * 2 = 88,064` parameters.

The Mamba-3 mixers retain their full comparison shape (`d_state=64`,
`headdim=64`, `expand=1`, `rope_fraction=0.5`, `chunk_size=64`) and use the
official implementation unchanged. Failure to import the pinned source is
fatal. The harness checks that exactly two layers were swapped and requires the
exact expected mixer delta, FFN delta, net delta, and total model counts; a
dependency or environment override cannot silently pass on approximate parity.

The expected reference arithmetic is: two full Mamba-3 mixers add `+175,376`,
the two FFN trims remove `2 * 88,064 = -176,128`, and the net is `-752`
parameters. With the pinned official baseline, the expected totals are
`17,059,912` (attention) and `17,059,160` (Mamba-3), or `-0.004408%`. Runtime
counts are authoritative: the harness prints the exact percentage and fails if
a dependency change invalidates near parity.

## ThetaScan projection layouts and parity

`--projection-layout` is a separate GN/kernel benchmark axis:

- `mamba-shared`: one Q and one K projection shared by all eight heads, with
  per-head V and zero-initialized per-head Q/K biases;
- `transformer-gqa`: eight Q heads and four K/V groups, each K/V group repeated
  over a pair of heads, matching the official attention GQA layout;
- `independent`: eight independent Q, K, and V heads.

The reference memory shape is `memory_multiplier=3` at expansion 1 (dense), or
`memory_multiplier=6` at `feature_expansion=2` — the same trainable factor
width with a doubled effective state. The reference recipes use a fast
temporal-mode bank. Only the FFNs around swapped layers `4,5` are trimmed to
compensate for each projection layout: hidden widths are respectively `1023`,
`832`, and `576`. The base GN and kernel arms (one recency branch, no further
learned controls) then each contain exactly `17,059,928` trainable parameters,
only `+16` (`+0.000094%`) versus attention. Zero delta is impossible using an
integer FFN width: each one-channel two-layer trim changes 1,024 parameters per
swapped layer, while the two ThetaScan mixers leave an eight-parameter control
remainder per layer. Reference variants add their own small learned controls
on top of that base: the expanded kernel arm (per-head thresholds, two recency
branches) reaches `17,059,976` (`+64` versus attention), and the expanded GN
arm's full-width threshold vector costs `+3,072` per block, returned exactly
by its `1020`-wide swapped FFN for the same `17,059,928` total. The harness
prints and validates each arm's declared count and fails closed on any silent
shape change. The table below lists the base arms per projection layout.

| Arm/layout | Mixer parameters per swapped block | Swapped FFN hidden | Full-model parameters |
|---|---:|---:|---:|
| Attention | 786,440 | 1,024 | 17,059,912 |
| Mamba-3 | 874,128 | 938 | 17,059,160 |
| ThetaScan Mamba-shared (GN reference) | 787,472 | 1,023 | 17,059,928 |
| ThetaScan Mamba-shared (GN expanded 2x) | 790,544 | 1,020 | 17,059,928 |
| ThetaScan Mamba-shared (kernel expanded 2x) | 787,496 | 1,023 | 17,059,976 |
| ThetaScan Transformer-GQA (GN reference) | 983,056 | 832 | 17,059,928 |
| ThetaScan independent Q/K/V (GN reference) | 1,245,200 | 576 | 17,059,928 |

The fifteen retained files under `configs/runpod/` have three roles. Their
stable stage identifiers are intentionally sparse rather than renumbered:

- `01`, `02`, and `05`-`07` run attention, parameter-matched Mamba-3, and the
  three reference ThetaScan arms continuously from step 0 to 3,000 on the
  7,500-step schedule;
- `08`-`12` reproduce the exact step-3,000 to step-4,000 continuation for
  attention, Mamba-3, and the three reference ThetaScan arms; and
- `13`-`17` continue the verified local step-4,000 checkpoints of the same
  five arms through the step-6,750 warmdown boundary to terminally cooled
  step-7,500 endpoints.

The step-0 configurations download 16 shards, run a 20-step smoke stage, then
execute the `20 -> 3000` production segment. Smoke writes a full checkpoint and
production immediately reloads it, so the paid smoke verifies the real resume
path and the combined trajectory remains continuous. The continuation
configurations expose 21 and then 40 shards and require the corresponding
local checkpoint plus its recorded SHA-256 and byte size; checkpoints are
deliberately not distributed in this repository. The reviewed terminally
cooled measurements of the five-arm league are in the
[step-7,500 continuation set](results/2026-07-21-h100-continuation-7500-v1/SUMMARY.md).
The two expanded arms carry `research_allow_oversize_artifact` at every stage:
their int8 artifacts exceed the 16 MB submission cap (by 36,604 and 12,318
bytes) and are research evidence rather than valid capped submissions.

Every comparison must pass one explicit shared seed to all arms. Process
timeouts are safety limits and do not alter the learning-rate schedule.

The retained evidence records a SHA-256 of the Python source tree used by each
reviewed run. No retained run identifies a clean public Git revision: older
records use `git_revision: null`, while newer records preserve the dirty base
revision when it was available. The present source implements and tests the
same named recipes, yet it is not a byte-for-byte archive of every executed
source tree. Treat the measurements as compact audited evidence and the current
configs as repeatable new-run recipes; exact bitwise replay of the original
runs is not claimed.

## Safe RunPod launch

The launcher accepts one file from `configs/runpod/` per pod and requests the
official Parameter Golf container by its historical source tag:
`runpod/parameter-golf:5e377cdd76814bd8d13488af05a237795957be13`
(OCI index digest
`sha256:74af8ca6ea79dde333038cd824055fc1fbac06b9d3b69360222d3353c5482d1f`).
RunPod's creation API accepts the image name rather than an OCI digest, so this
digest is recorded as expected provenance and is not claimed as a
platform-attested runtime digest. The resolved Python/CUDA package inventory is
exported with each result.
This is the same Python 3.12 / PyTorch 2.9.1+cu128 / Triton 3.5.1 runtime used by
the earlier 1,000-step runs. It is addressed directly because the historical
public template object `y5cejece4j` now returns 404 even though the official
OpenAI README still links to it. The image's preinstalled Python dependencies
are verified, not reinstalled, and the exact resolved runtime is exported with
every result. Creation is intentionally minimal and declares a platform
`terminateAfter` first; the source snapshot and bootstrap are then sent over
account-key SSH, avoiding RunPod control-plane environment-size limits.
First inspect the payload locally; `--dry-run` neither needs credentials nor
creates a pod:

```bash
python benchmarks/parameter-golf/run_runpod.py launch \
  --config benchmarks/parameter-golf/configs/runpod/01-attention-official.json \
  --seed 184726391 --dry-run
```

Generate one fresh seed, then pass that same value to every comparison launch.
Each reviewed config carries its own boot-absolute timeout, export grace period,
and maximum per-pod budget under `launcher`; the dry-run prints the effective
guards and remaining setup headroom. The corresponding CLI options are explicit
overrides, not required launch boilerplate.
Remove `--dry-run` and add `--yes` only when ready to authorize billing. Keep the
launcher attached so it streams the checkpoint before terminating the pod. Each
pod has a boot-absolute watchdog and two self-delete paths. If `--detach` is
necessary, start the collector immediately in another persistent process; do
not wait for the whole matrix, because the pod deletes itself after the result
grace window:

```bash
python benchmarks/parameter-golf/run_runpod.py collect \
  --state benchmarks/parameter-golf/results/_local/<run>/launch.json
```

The collector streams the approximately 140 MB lossless research checkpoint to
`results/_local/` rather than buffering it in memory, verifies its SHA-256 and
byte count, downloads sanitized logs/results, and verifies pod termination.
Every smoke and production stage must print an exact int8+zlib round-trip line.
Submission-valid runs must remain at or below 16,000,000 bytes. The two retained
expanded research recipes explicitly set `research_allow_oversize_artifact`;
their measured oversize artifacts are accepted as research evidence but never
reported as cap-compliant submissions. The runner records the size and cap,
then deletes `final_model.int8.ptz` before exporting results.
Research checkpoints are opt-in local operational artifacts. Keep them in the
gitignored `_local/` area or the lab workspace; never commit them as curated
public benchmark results.

Reviewed public result sets are indexed in
[results/README.md](results/README.md).

## Fair-comparison checklist

- Use the same prepared checkout, dataset, tokenizer, iteration budget, batch
  tokens, and environment for every arm.
- The runner clears inherited model/training hyperparameter environment
variables before applying its controlled settings; use the documented CLI
  flags rather than ambient shell overrides.
- Confirm that the strict Mamba-3 run prints exactly `mixer_impl:Mamba3`; any
  other implementation invalidates that arm.
- Record the runtime-generated seed if a result will later be replicated.
- Archive `ours/ADAPTER-MANIFEST.json`; it records generated-file hashes, the
  ThetaScan version/source hash, pinned dataset revision, and upstream commits.
- The harness records Python, PyTorch, and `nvidia-smi`. Archive `pip freeze`
  and record CUDA and Triton separately; GPU dependency versions remain
  environment-specific.
- Treat `auto`, `fla`, and portable ThetaScan backends as execution choices, not
  separate algorithms; smoke-test forward and backward on the target GPU.
- Report the resolved adapter manifest and actual parameter counts alongside any
  result generated by the user.

The generated checkout retains the official parameter-golf `LICENSE` and
`THIRD_PARTY_NOTICES.md`. Under `ours/thetascan-license/`, the bootstrap also
copies ThetaScan's complete PolyForm license/notices with their original names
and working relative links. The generated `train_gpt_hybrid.py` and pinned
downloader remain derivatives of the official MIT-licensed harness.

The adapter files live in the Git/source distribution, not in the minimal
runtime wheel. Start these commands from a ThetaScan source checkout or unpacked
source archive; installing only the wheel intentionally installs just the
library.
