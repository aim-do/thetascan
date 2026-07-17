# Curated benchmark results

This directory contains compact, reviewed measurements chosen to explain the
public v0.1 API and the report. Lower bits per byte (BPB) is better. Every
summary states its seed, schedule, parameter count, artifact status, and
replay limitations.

The publication copy deliberately excludes exploratory matrices, failed
diagnostics, raw lab logs, pod state, credentials, datasets, prepared upstream
checkouts, and resumable checkpoints. Those belong in the separate research
workspace. Internal pre-release recipe identifiers are also omitted from the
public namespace.

## Evidence levels

- **Compact reviewed record:** exact reviewed metrics plus canonical public
  configurations. The historical source snapshot is not distributed, so this
  supports audit and a new repetition, not bitwise replay.
- **Clean reproducible record:** the same files plus a clean tagged source
  revision and complete sanitized run export. New publication runs should aim
  for this level.

Single-seed results are measurements, not estimates of run-to-run variance.
Small differences must not be described as statistical ties or definitive
architecture rankings.

## Published sets

- [Matched continuation from 4,000 steps through the completed 7,500-step
  warmdown](2026-07-21-h100-continuation-7500-v1/SUMMARY.md) — five arms
  (attention, Mamba-3 parity, GN reference, GN expanded 2x, kernel expanded
  2x), terminally cooled endpoints.
- [Report claim-to-artifact evidence map](EXPERIMENT_EVIDENCE.md)

## Publishing a new result

1. Start from a clean tagged source revision.
2. Run smoke validation before production.
3. Record the expanded public config, seed, data manifest, optimizer policy,
   schedule, parameter count, and source hash.
4. Verify exact int8 round-trip and record the artifact size. The upstream
   Parameter Golf cap is 16,000,000 bytes; a result above it may still be
   useful research evidence but is not a valid capped submission.
5. Publish compact metrics and canonical configurations. Do not commit
   credentials, launch state, datasets, or research checkpoints.

`run_runpod.py` exports local runs under ignored `_local/` directories. Review
and curate them manually before publication.
