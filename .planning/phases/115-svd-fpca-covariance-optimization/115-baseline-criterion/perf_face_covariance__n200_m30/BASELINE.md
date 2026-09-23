# BEFORE-Baseline: perf_face_covariance/n200_m30

## Overview

This is the committed BEFORE-optimization criterion baseline for the `perf_face_covariance/n200_m30`
benchmark cell (bench binary: `perf_hotpaths`, group: `perf_face_covariance`, function: `n200_m30`).
Captured in Phase 115, Plan 01 (Wave 1) from the UN-OPTIMIZED `fdars-core` code, before any
`fdars-core/src` change. Plan 02 re-runs the identical command and compares against this baseline
to verify the OPT-02 `sym_cov_irreg` speedup.

## Bench Command

```sh
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench -p fdars-core --features linalg,parallel --bench perf_hotpaths -- "perf_face_covariance/n200_m30"
```

## Measurement Results

| Metric            | Value (ns)              |
|-------------------|------------------------|
| mean point_est    | 268,166,197.56         |
| mean CI lower (95%) | 260,142,509.90       |
| mean CI upper (95%) | 275,883,662.23       |
| mean std_error    | 4,000,112.36           |
| slope point_est   | 260,026,606.58         |
| slope CI lower    | 250,184,599.42         |
| slope CI upper    | 272,051,057.71         |
| median point_est  | 268,850,616.50         |
| std_dev           | 16,030,848.88          |
| sample_size       | 15                     |
| outliers          | 4 of 15 (1 low-severe, 1 low-mild, 2 high-mild) |

**Mean (human-readable): ~268 ms** (slope: ~260 ms)

## Provenance

- **Git commit SHA:** `328215e762f7b37f6727762af5a528164eeaa767`
- **Features:** `linalg,parallel`
- **Criterion version:** 0.5 (existing dev-dep, no new dependency)
- **Captured:** 2026-09-23

## Machine

- **OS:** Linux simon-xps159530 6.18.45-1-MANJARO x86_64
- **CPU:** 13th Gen Intel(R) Core(TM) i9-13900H
- **Kernel:** 6.18.45-1-MANJARO #1 SMP PREEMPT_DYNAMIC

## Note on Orphaned Bench Group

The on-disk criterion results directory at
`target/criterion/face_covariance/{n50_m30,n50_m60,n200_m30}/` is an **orphaned Phase-113 probe**:
there is no source benchmark currently emitting a group named `face_covariance` in any `.rs` bench
file. It cannot be re-run reproducibly.

The reproducible BEFORE/AFTER cell is `perf_face_covariance/n200_m30` (bench binary `perf_hotpaths`,
group `perf_face_covariance`, source-committed in `fdars-core/benches/perf_hotpaths.rs`).
Plan 02 MUST use `perf_face_covariance` — not the orphaned `face_covariance` group — for the
speedup comparison.

## Speedup Target (OPT-02)

The `sym_cov_irreg` upper-triangle+mirror refactor eliminates ~50% of the `cov_irreg` computation
(the inner `(s, t)` double loop computes only `i <= j` cells; the `j > i` half is filled by
bit-copy). Expected speedup: **~1.5-2x** on this n=200, m=30 cell.

Plan 02 re-runs the same command and reports: `after_mean_ns / before_mean_ns` (point estimates).
A ratio < 1.0 confirms the speedup. The hard gate is: `after < before` (any positive speedup).
