# Experiment evidence map

This index maps the numerical claims in Public Preview v0.1 of the ThetaScan
report to compact public records. The retained July 2026 experiments used one
seed, and the historical source snapshots are not distributed. They support
audit of reported measurements and new repetitions with the current
implementation; they do not support bitwise replay or statistical
significance claims.

| Claim or protocol detail | Public evidence | Qualification |
|---|---|---|
| Seed `1661305741`; 7,500-step schedule completed through the 6,750-to-7,500 warmdown; 40 shards for the final continuation | [7,500-step continuation](2026-07-21-h100-continuation-7500-v1/SUMMARY.md) | Compact reviewed record; one seed |
| Width 512, eight query/four key-value attention heads, context/vocabulary 1,024, 524,288 tokens per step, two substituted layers, FFN width 1,024, Mamba-3 parity FFN width 938, per-arm swapped-FFN widths 1,023/1,020 | [`experiment-config.json`](2026-07-21-h100-continuation-7500-v1/experiment-config.json) | Matched whole-model trainable parameter allocation, not matched FLOPs or wall time |
| GN expanded 2x `1.2327` raw / `1.23830852` exact-int8 at step 7,500, below Attention `1.2349` / `1.24071718` on both metrics | [7,500-step continuation](2026-07-21-h100-continuation-7500-v1/SUMMARY.md) | Terminally cooled endpoint; one seed; artifact exceeds the 16 MB cap by 36,604 bytes (research-only) |
| GN reference (dense) `1.2342` raw / `1.23984702` exact-int8 at step 7,500, below Attention on both metrics, cap-compliant | [7,500-step continuation](2026-07-21-h100-continuation-7500-v1/SUMMARY.md) | Terminally cooled endpoint; one seed |
| Kernel expanded 2x `1.2361` raw / `1.24215211` exact-int8 at step 7,500, within `0.0015` int8 BPB of Attention | [7,500-step continuation](2026-07-21-h100-continuation-7500-v1/SUMMARY.md) | One seed; artifact exceeds the cap by 12,318 bytes (research-only) |
| Mamba-3 parity `1.3194` raw / `1.32677632` exact-int8 at step 7,500 | [7,500-step continuation](2026-07-21-h100-continuation-7500-v1/SUMMARY.md) | Official module under the shared schedule; volatile at full learning rate |
| Optimizer policy: baselines `muon-2d`, all ThetaScan arms `muon-2d+theta` (per-head Muon on the memory factors) | [`experiment-config.json`](2026-07-21-h100-continuation-7500-v1/experiment-config.json) | Rows compare architecture-plus-policy pairs, not a pure mixer swap |
| Step-time and artifact-size comparisons | Endpoint table in the linked summary | Portable materialized evaluator at context 1,024, not a fused linear-scaling kernel benchmark |
| Random-expansion trainable-vs-effective width accounting (trainable factors at base width, state at expanded width) | Per-mixer contracts in `tests/` and the staged configs under `configs/runpod/` | Verified by fail-closed parameter contracts at model build time |

## Deliberately excluded claims

Pre-release allocation sweeps, broad mechanism matrices, failed diagnostics,
earlier partial-schedule screens, and exact internal recipe identifiers
remain in the lab workspace. The public report may discuss the corresponding
mechanisms as hypotheses, but it should not quote a deleted exploratory
number as publication evidence.
