# H100 matched continuation from step 4,000 through the 7,500-step warmdown

This compact public record retains five exact checkpoint continuations from
step 4,000 to the end of the 7,500-step schedule, through the final warmdown
(6,750 to 7,500). Every arm used seed `1661305741`, FineWeb10B `sp1024`,
sequence length 1,024, 524,288 tokens per step, and one NVIDIA H100 80GB. The
earlier 16- and 21-shard prefixes were preserved and 40 shards were exposed
for this continuation. A 20-step resume smoke check ended at 4,020;
production was then validated every 250 steps through 7,500.

Unlike the earlier public screens, these endpoints are terminally cooled:
the learning-rate warmdown completed before step 7,500. Lower bits per byte
(BPB) is better. Step 4,020 is an integrity check, not a ranking point.

## Raw validation trajectory

| Step | Attention | GN expanded 2x | GN reference | Kernel expanded 2x | Mamba-3 parity |
|---:|---:|---:|---:|---:|---:|
| 4,020 (resume smoke) | 1.2844 | 1.2825 | 1.2844 | 1.2850 | 1.3616 |
| 4,250 | 1.2818 | 1.2802 | 1.2820 | 1.2825 | 1.3916 |
| 4,500 | 1.2799 | 1.2779 | 1.2800 | 1.2811 | 1.4136 |
| 4,750 | 1.2779 | 1.2762 | 1.2781 | 1.2792 | 1.3799 |
| 5,000 | 1.2740 | 1.2724 | 1.2742 | 1.2752 | 1.3616 |
| 5,250 | 1.2741 | 1.2725 | 1.2739 | 1.2749 | 1.3671 |
| 5,500 | 1.2716 | 1.2698 | 1.2718 | 1.2727 | 1.3523 |
| 5,750 | 1.2689 | 1.2669 | 1.2687 | 1.2700 | 1.3436 |
| 6,000 | 1.2680 | 1.2666 | 1.2679 | 1.2692 | 1.3548 |
| 6,250 | 1.2683 | 1.2666 | 1.2679 | 1.2693 | 1.3786 |
| 6,500 | 1.2651 | 1.2634 | 1.2653 | 1.2664 | 1.3843 |
| 6,750 (warmdown start) | 1.2627 | 1.2610 | 1.2621 | 1.2641 | 1.3708 |
| 7,000 | 1.2537 | 1.2521 | 1.2534 | 1.2550 | 1.3560 |
| 7,250 | 1.2429 | 1.2408 | 1.2421 | 1.2441 | 1.3320 |
| **7,500** | 1.2349 | **1.2327** | 1.2342 | 1.2361 | 1.3194 |

## Step-7,500 endpoint (terminally cooled)

| Arm | Public recipe | Raw BPB | Exact-int8 BPB | Delta vs Attention (int8) | Parameters | Artifact bytes |
|---|---|---:|---:|---:|---:|---:|
| GN expanded reference (2x random feature expansion) | `gn-expanded-reference-v0.1` | **1.2327** | **1.23830852** | -0.00240866 | 17,059,928 | 16,036,604 * |
| GN reference | `gn-reference-v0.1` | 1.2342 | 1.23984702 | -0.00087016 | 17,059,928 | 15,953,281 |
| Attention | (host baseline) | 1.2349 | 1.24071718 | 0.00000000 | 17,059,912 | 15,890,122 |
| Kernel expanded reference (2x random feature expansion) | `kernel-expanded-reference-v0.1` | 1.2361 | 1.24215211 | +0.00143493 | 17,059,976 | 16,012,318 * |
| Mamba-3 parity | (official module) | 1.3194 | 1.32677632 | +0.08605914 | 17,059,160 | 15,883,475 |

Step times are deliberately absent: the arms were not run under a matched
execution and compilation policy, so a per-step figure would compare
implementations rather than architectures. The raw values remain in the
per-arm training logs collected with this run.

\* Research-only artifacts: the two expanded arms exceed the 16,000,000-byte
submission cap by 36,604 and 12,318 bytes respectively. They are valid
measurements at matched trainable parameters but are not valid capped
submissions. The dense GN reference is cap-compliant.

All five endpoint artifacts passed exact int8 round-trip validation. Both GN
arms ended below Attention on raw and exact-int8 BPB, and the expanded kernel
arm ended within `0.0015` int8 BPB of Attention; one seed cannot establish
statistical superiority or a tie. Mamba-3 remained volatile at full learning
rate and improved sharply only during warmdown.

## Honest accounting

- **One seed.** These are measurements, not estimates of run-to-run variance.
- **Optimizer policy.** The baselines use the host `muon-2d` policy; every
  ThetaScan arm uses `muon-2d+theta` (batched per-head Muon on the memory
  factors). Rows compare architecture-plus-policy pairs.
- **Step time.** ThetaScan arms run a portable materialized evaluator at
  context 1,024; this is not a fused-kernel wall-time benchmark.
- **Fixed expansion maps.** The expanded arms were measured with maps drawn
  from the research key namespace. The public `expansion_key` namespace draws
  statistically equivalent but not bitwise-identical maps, so a new
  repetition reproduces the protocol, not the exact bit pattern.

## Public reconstruction and provenance

[`experiment-config.json`](experiment-config.json) records the canonical
public recipes and the continuation protocol. The corresponding staged runpod
configurations are checked in under `configs/runpod/` (`01`, `02`, and
`05`-`07` for 0-to-3,000; `08`-`12` for 3,000-to-4,000; `13`-`17` for
4,000-to-7,500).
Research checkpoints, pod metadata, credentials, and the original raw lab
logs are not published, so this compact record supports audit and new
repetition rather than bitwise replay.
