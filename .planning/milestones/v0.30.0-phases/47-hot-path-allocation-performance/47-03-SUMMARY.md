---
phase: 47-hot-path-allocation-performance
plan: 03
subsystem: irreg_fdata
tags: [perf, compute, face_covariance, kernel-smoothing, exp-reduction]
requires:
  - phase: 47-01
    provides: golden/criterion proof harness
provides:
  - "OPT-E: face_covariance −80.7% wall-time via kernel-weight-table precompute"
affects: [47-04, 51-benchmark-coverage-regression-guards]
actuals:
  tokens: 8000
  tasks: 3
  commits: 1
tech-stack:
  added: []
  patterns: [separable-kernel weight-table precompute out of grid loop]
key-files:
  created: []
  modified: [fdars-core/src/irreg_fdata/smoothing.rs, fdars-core/tests/equivalence_phase47.rs, .planning/phases/47-hot-path-allocation-performance/PERF-RESULTS.md]
key-decisions:
  - "cov_irreg precomputes w_s/w_t tables (only exp() calls); grid loop becomes lookups. Same i→j1→j2 summation order preserved → byte-equivalent."
  - "accumulate_cov_at_point deleted (sole caller was cov_irreg)."
patterns-established:
  - "Factor separable-kernel weights out of the grid loop when the factor depends on only one grid axis."
requirements-completed: [PERF-01]
coverage:
  - id: D1
    description: "OPT-E face_covariance compute win (−80.7% wall-time) behavior-preserving"
    requirement: PERF-01
    verification:
      - kind: integration
        ref: "golden_face_covariance_n40 pass rel 1e-12 (captured pre-edit); perf_face_covariance/n200_m30 983.8→189.8ms; 45 irreg tests pass; clippy clean"
        status: pass
    human_judgment: false
duration: 30min
completed: 2026-08-31
status: complete
---

# Phase 47 / Plan 03: OPT-E face_covariance Summary

**cov_irreg precomputes per-observation Gaussian kernel-weight tables (w_s/w_t) once instead of recomputing them per (s,t) grid cell — ~98% fewer exp() calls, cutting face_covariance wall-time 80.7% (983.8→189.8ms) with byte-equivalent output.**

## Accomplishments
- **OPT-E:** restructured `cov_irreg` to precompute `w_s`/`w_t` weight tables before the grid loop (the only `exp()` calls now); the `(si,ti)` loop is pure lookups, preserving the exact `i→j1→j2` per-cell summation order. Removed `accumulate_cov_at_point`.
- **face_covariance @ n200_m30: 983.8ms → 189.8ms [167.8, 217.3] (−80.7%)** — non-overlapping CIs, far exceeds the ≥15% PERF-01 bar.
- Golden `golden_face_covariance_n40` captured from pre-edit code, passes at rel 1e-12; all 45 irreg_fdata tests + full suite green; clippy `--all-targets` clean; signature unchanged.

## Deviations
None.

---
*Phase: 47-hot-path-allocation-performance · Completed: 2026-08-31*
