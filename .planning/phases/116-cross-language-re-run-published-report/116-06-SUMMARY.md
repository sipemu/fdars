---
phase: 116-cross-language-re-run-published-report
plan: 06
subsystem: benchmarks
tags: [rerun, criterion, scikit-fda, R, comparison, noise-gate]
requirements: [RPT-01]
provides: [after-optimization-comparison-csv, rerun-log, refreshed-env]
key-files:
  created:
    - benchmarks/results/rerun_log.json
  modified:
    - benchmarks/results/comparison.csv
    - benchmarks/results/rust.json
    - benchmarks/results/env.json
    - benchmarks/results/python_n*_m*.json
    - benchmarks/results/r_n*_m*.json
    - fdars-core/benches/crosslang_grid.rs
key-decisions:
  - "7 heavy single-threaded elastic chunks recorded as timeout / skipped-inferred-timeout (final, never re-queued) — report lists them as 'not run (timeout)'"
  - "crosslang_fpcr noise remediation round 1 (60 samples / 20 s) sufficed; no round 2"
  - "fdars-vs-python FPCA cos=0.9816 @ n500_m50 is the Phase-113 user-approved DISCLOSED (not hard-gated) quadrature divergence; fdars-vs-R hard gate cos=1.000 on every cell"
duration: ~3.2 h (2.93 h measured wall-clock in rerun_log)
completed: 2026-09-25
---

# Plan 116-06 Summary: Full post-optimization re-run

Executed INLINE by the orchestrator (20 budgeted `--rust-next` calls of ≤540 s + 6 per-cell grid calls).

## Task 1 — fdars criterion (67 chunks, all *_NUM_THREADS=1)
- `--rerun-init` archived stale criterion output to `target/criterion.pre116-20260924T210514Z`; grid CSVs generated; prebuild ok (20 s, cached).
- Final `--rust-status` (verbatim):
```
Chunk status counts:
  ok: 60
  skipped-inferred-timeout: 3
  timeout: 4

Non-ok chunks (idx, group, filter, status):
  [8] audit_p3_elastic_cross_banded ^audit_p3_elastic_cross_banded/n100_m200$ -> timeout
  [9] audit_p3_elastic_cross_banded ^audit_p3_elastic_cross_banded/n500_m50$ -> skipped-inferred-timeout
  [26] audit_p3_elastic_cross ^audit_p3_elastic_cross/n100_m200$ -> timeout
  [27] audit_p3_elastic_cross ^audit_p3_elastic_cross/n500_m50$ -> skipped-inferred-timeout
  [29] audit_p3_elastic_self ^audit_p3_elastic_self/n100_m200$ -> timeout
  [30] audit_p3_elastic_self ^audit_p3_elastic_self/n500_m50$ -> skipped-inferred-timeout
  [37] audit_p3_karcher ^audit_p3_karcher/n500_m200$ -> timeout
```
- `collect_rust.py` → rust.json: 36 crosslang ids (6 ops × 6 cells), thread env RAYON_NUM_THREADS=1 in log meta.

## Task 2 — scikit-fda + R grid
- All six cells (100/300/500 × 50/200) `GRID_CELL_STATUS=ok`, `[GATE PASS]` before timing on every cell. gate.py unchanged.
- FPCA fdars-vs-R (hard gate): subspace cos = 1.000000 on every cell. fdars-vs-python FPCA disclosed (Phase 113 decision `e7f7239a`): 0.9816 at n500_m50 (DISCLOSED-BELOW), ≥0.9986 elsewhere. FPCR disclosed only (Python-vs-R |corr| 0.8585 at n500_m50).
- `fpca_native_r` retained in every r_n*_m*.json. env.json refreshed (CPU i9-13900H, 20 logical cores; Python 3.14.7 scipy-openblas; R 4.6.1).

## Task 3 — assemble, remediate, commit
- `assemble_comparison.py --gate-passed` → comparison.csv (schema unchanged). Rust noise-gate distribution: PASS 74 / FAIL-NOISY 63 (the non-headline legacy audit groups are noisier; they are flagged, never headlined).
- Initial `--noise-report`: crosslang_fpcr/n500_m200 FAIL-NOISY (RSD 11.79%, CIW 8.11%). Remediation round 1 (sample_size 60, measurement_time 20 s, doc-commented) → re-ran whole group (logged as remediation).
- Final `--noise-report` (verbatim):
```
NOISE criterion:crosslang_fpca/n500_m200 rsd=0.77 ciw=0.54 PASS
NOISE criterion:crosslang_basis_smoothing/n500_m200 rsd=2.53 ciw=1.72 PASS
NOISE criterion:crosslang_depth/n500_m200 rsd=4.72 ciw=3.30 PASS
NOISE criterion:crosslang_functional_boxplot/n500_m200 rsd=3.95 ciw=2.77 PASS
NOISE criterion:crosslang_fpcr/n500_m200 rsd=3.72 ciw=1.86 PASS
NOISE criterion:crosslang_elastic/n500_m200 rsd=2.46 ciw=1.73 PASS
```
- Footprint: target/ = 57.83 GiB, target/criterion/ = 0.02 GiB, target/release/ = 0.99 GiB.
- Freshness verify: 36 crosslang rows, no orphan-group rust rows, no stale-identical Python/R rows → `freshness OK`. pytest 192 passed; cargo fmt check ok; clippy --benches clean.

## Deviations / disclosures (carry into the report — 116-07)
- **Machine load during the fdars re-run was NOT uniformly quiet.** The pre-run quiet check passed (load < 2 after the user freed the machine), but unrelated desktop workloads (node, baloo indexer, browser) returned during the run: 1-min load average at the start of each `--rust-next` call ranged 2.8–15.0 on a 20-logical-core machine. All measurements are single-threaded and the six headline cells pass the 5% noise gate, but the report's methodology section must disclose that background load was present during part of the fdars run (it can inflate absolute fdars times, i.e. it biases AGAINST fdars, not in its favor).
- Budget-interrupted chunks were auto-requeued by the scheduler (status `interrupted` → pending) and completed in later calls — expected behavior, not a deviation.
- A raw `GATE FAIL [fdars-vs-python] FPCA` stderr line appears in the 500x50 log; it is emitted by the shared `check_agreement` helper for the disclosed (non-gating) fdars-vs-python comparison and does not affect the hard gate.

## Commits
- `06a0878c` feat(116-06): full post-optimization cross-language re-run (after-data)
