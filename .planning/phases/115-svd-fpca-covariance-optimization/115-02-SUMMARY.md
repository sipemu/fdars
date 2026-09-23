---
phase: 115-svd-fpca-covariance-optimization
plan: "02"
subsystem: irreg_fdata / covariance optimization
tags: [performance, optimization, numeric-identity, criterion, svd]
status: complete

dependency_graph:
  requires: [115-01]
  provides: [sym_cov_irreg, after-criterion-evidence, svd-near-optimal-verdict]
  affects: [fdars-core/src/irreg_fdata/smoothing.rs, fdars-core/src/irreg_fdata/face.rs]

tech_stack:
  added: []
  patterns:
    - "Upper-triangle + bit-copy mirror for symmetric covariance estimation"
    - "Single kernel-weight table when s_grid == t_grid (halves exp() evaluations)"

key_files:
  created:
    - .planning/phases/115-svd-fpca-covariance-optimization/115-after-criterion/perf_face_covariance__n200_m30/estimates.json
    - .planning/phases/115-svd-fpca-covariance-optimization/115-after-criterion/perf_face_covariance__n200_m30/COMPARISON.md
    - .planning/phases/115-svd-fpca-covariance-optimization/115-svd-rebench/svd_n500_m50_estimates.json
    - .planning/phases/115-svd-fpca-covariance-optimization/115-svd-rebench/svd_n1000_m200_estimates.json
    - .planning/phases/115-svd-fpca-covariance-optimization/115-svd-rebench/svd_n500_m500_estimates.json
    - .planning/phases/115-svd-fpca-covariance-optimization/115-svd-rebench/SVD-VERDICT.md
  modified:
    - fdars-core/src/irreg_fdata/smoothing.rs
    - fdars-core/src/irreg_fdata/face.rs

decisions:
  - "sym_cov_irreg marked pub(crate) (not private) so face.rs can import it via crate::irreg_fdata::smoothing::sym_cov_irreg without a pub re-export"
  - "Single kernel table w (instead of w_s + w_t) since s_grid == t_grid — halves table-build cost and memory, contributing extra speedup beyond the triangle shortcut"
  - "No diagonal special-case in the mirror (si==ti self-copies harmlessly) per RESEARCH Pitfall 3"
  - "SVD documented near-optimal with no source change — faer thin SVD already production path under linalg feature"

metrics:
  duration: "~7 minutes"
  completed: "2026-09-23"
  tasks: 2
  commits: 2
  plan_head_before: 7cde79b24ea259318f358e4d57957bb2435f8995

actuals:
  tokens: 8400
  tasks: 2
  commits: 2
---

# Phase 115 Plan 02: SVD/FPCA & Covariance Optimization (Wave 2) Summary

Implemented the numeric-identity-safe `sym_cov_irreg` upper-triangle optimization (OPT-02), proved bit-identical output in debug and release, measured a ~3.45x speedup on `face_covariance`, and documented SVD as near-optimal.

## What Was Built

### Task 1: sym_cov_irreg (OPT-02)

Added a new `pub(crate) fn sym_cov_irreg(ifd: &IrregFdata, grid: &[f64], bandwidth: f64) -> FdMatrix` in `fdars-core/src/irreg_fdata/smoothing.rs` immediately after `cov_irreg`. The key design decisions:

- **Single kernel table**: Since `s_grid == t_grid`, the two separate tables `w_s` and `w_t` that `cov_irreg` builds are byte-identical. `sym_cov_irreg` builds one table `w` of size `total_obs * m`, halving the `kernel_gaussian` (exp) call count and memory allocation.
- **Upper-triangle loop**: `for si in 0..m { for ti in si..m { ... } }` — visits only `m*(m+1)/2` cells instead of `m*m`.
- **Verbatim accumulation body**: The inner `i → j1 → j2` loop body is word-for-word identical to `cov_irreg`, preserving FP summation order. Upper-triangle values are bit-identical.
- **Bit-copy mirror**: `cov[ti + si*m] = val` after writing `cov[si + ti*m] = val`. The diagonal (`si==ti`) self-copies harmlessly.

Changed `face.rs:148` from `cov_irreg(ifd, grid, grid, bandwidth)` to `sym_cov_irreg(ifd, grid, bandwidth)`. Removed the now-unused `cov_irreg` import from face.rs. Public `cov_irreg` is unchanged.

### Task 2: Numeric Identity Proof + AFTER Criterion + SVD Re-benchmark

- **Release numeric identity**: `cargo test -p fdars-core --release --features linalg -- face` — 21 tests passed (0 failed). Golden fixture `equivalence_phase115` passed in both debug and release.
- **AFTER criterion**: Non-overlapping CI — statistically significant ~3.45x speedup.
- **SVD re-benchmark**: CI widths well under 5% on all cells; faer thin SVD confirmed as the production path.

## Performance Results

### Covariance: BEFORE vs AFTER

| Metric   | BEFORE (328215e7)       | AFTER (c0e84bd8)         |
|----------|-------------------------|--------------------------|
| Mean     | 268.166 ms              | 77.786 ms                |
| CI lower | 260.1 ms                | 76.710 ms                |
| CI upper | 275.9 ms                | 79.372 ms                |

**Speedup: ~3.45x (71.0% reduction). Non-overlapping CIs — statistically significant.**

The RESEARCH predicted ~2x; the observed ~3.45x reflects an additional benefit: the single kernel table halves both the exp() evaluations (table-build step) and memory pressure on top of the triangle shortcut.

### SVD Re-benchmark (higher sample count)

| Cell           | Mean      | CI lower  | CI upper   | CI width |
|----------------|-----------|-----------|------------|----------|
| n500_m50       | 1.061 ms  | 1.057 ms  | 1.070 ms   | 1.2%     |
| n1000_m200     | 33.699 ms | 33.674 ms | 33.724 ms  | 0.15%    |
| n500_m500      | 126.36 ms | 126.23 ms | 126.48 ms  | 0.20%    |

All CI widths well below 5%. The change reports show ~60-64% improvement vs the nalgebra baseline — confirming faer thin SVD (already active under `--features linalg`) is the production path and near-optimal.

**SVD verdict: near-optimal. No source change warranted.** Production `fdata_to_pc` already uses faer thin SVD under `linalg` (`regression.rs:455-499`). No numeric-identity-safe optimization with meaningful leverage exists without a new crate dependency or API change.

## Numeric Identity

- Golden fixture `equivalence_phase115.rs`: 7 constants (incl. symmetry pairs COV_0_5 / COV_5_0) at ≤1e-10 — **UNCHANGED**, passes in debug and release.
- After the optimization, symmetry pairs are now exactly equal (bit-copy mirror) rather than differing at ~1e-18.
- All 21 face module unit tests green in `--release --features linalg`.
- clippy `--all-targets --features linalg,parallel -- -D warnings`: clean.

## Deviations from Plan

None — plan executed exactly as written. The observed speedup (~3.45x) exceeded the research prediction (~2x) but this is an honest measurement, not a deviation.

## Known Stubs

None.

## Threat Flags

None — private numeric-loop refactor, no I/O, no network, no public API change.

## Self-Check

- [x] `fdars-core/src/irreg_fdata/smoothing.rs` contains `sym_cov_irreg` (pub(crate))
- [x] `fdars-core/src/irreg_fdata/face.rs` line 149 calls `sym_cov_irreg`
- [x] Task 1 committed: c0e84bd8
- [x] Task 2 committed: 74b9ab0d
- [x] estimates.json and COMPARISON.md exist under 115-after-criterion/
- [x] SVD estimates + SVD-VERDICT.md exist under 115-svd-rebench/
- [x] 2 commits measured from ledger (7cde79b2..HEAD)
