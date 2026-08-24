---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: "02"
subsystem: fem_smoothing
tags: [fem, pde, smoothing, gcv, regression, functional-data]
requirements: [REP-02-02]

dependency_graph:
  requires: ["44-01"]
  provides: [fem_smooth, fem_smooth_gcv, fem_predict]
  affects: [fdars-core/src/fem_smoothing.rs, fdars-core/src/lib.rs]

tech_stack:
  added: []
  patterns:
    - SR-PDE penalised normal equations (Φ'Φ + λK + εI)c = Φ'y
    - Trace-based GCV via dense A_inv (column-by-column Cholesky inverse)
    - edf = tr(A_inv · Φ'Φ) as elementwise dot of two symmetric N×N matrices
    - GCV λ-grid search mirroring smooth_basis_gcv pattern
    - P1 linear-exactness test oracle for fem_predict

key_files:
  created: []
  modified:
    - fdars-core/src/fem_smoothing.rs
    - fdars-core/src/lib.rs

decisions:
  - All three functions (fem_smooth, fem_smooth_gcv, fem_predict) implemented as a single combined commit since tests were written together and all pass as one unit
  - GCV and edf computed inside fem_smooth itself (not as a separate helper) — avoids double Cholesky factorisation and keeps the API surface clean
  - fem_smooth_gcv returns last error if all grid points fail, or ComputationFailed if all GCVs are non-finite but no error occurred
  - ε=1e-10 ridge added to lift K's constant null space before cholesky_solve; documented in rustdoc

metrics:
  duration_minutes: 15
  completed: "2026-08-24T16:54:30Z"
  tasks_completed: 3
  tasks_total: 3
  commits: 1
  files_modified: 2
  tests_added: 9
  status: complete

actuals:
  tokens: 14000
  tasks: 3
  commits: 1
---

# Phase 44 Plan 02: SR-PDE Smoothing + GCV + Predict Summary

SR-PDE surface smoothing solving `(Φ'Φ + λK)c = Φ'y` via dense in-house Cholesky with trace-based GCV edf computation and P1 interpolation-based prediction at new points.

## What Was Built

Three public functions added to `fdars-core/src/fem_smoothing.rs`:

- **`fem_smooth`** — fixed-λ SR-PDE smoothing: builds Φ (n_obs×N row-major), assembles Φ'Φ+λK+εI, solves via `cholesky_solve`, computes A_inv column-by-column, returns edf=tr(A_inv·Φ'Φ) and GCV=(rss/n)/(1-edf/n)².
- **`fem_smooth_gcv`** — GCV-optimal λ grid search (log₁₀ grid, mirrors `smooth_basis_gcv` pattern), returning the result with minimum finite GCV.
- **`fem_predict`** — evaluates the fitted surface at new (x,y) points via P1 barycentric interpolation; exact for linear fields.

Extended crate-root re-export in `lib.rs`: `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval, fem_predict, fem_smooth, fem_smooth_gcv, FemSmoothResult};`

## New Public Signatures

```rust
pub fn fem_smooth(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    obs_xy: &[[f64; 2]],
    y: &[f64],
    lambda: f64,
) -> Result<FemSmoothResult, FdarError>

pub fn fem_smooth_gcv(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    obs_xy: &[[f64; 2]],
    y: &[f64],
    log_lambda_range: (f64, f64),
    n_grid: usize,
) -> Result<FemSmoothResult, FdarError>

pub fn fem_predict(
    node_values: &[f64],
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    query_xy: &[[f64; 2]],
) -> Result<Vec<f64>, FdarError>
```

## Test Results

All 14 `fem_smoothing::tests` pass (9 new + 5 carried from Plan 01):

| Test | Status |
|------|--------|
| test_fem_smooth_solves_and_reduces_residual | ok |
| test_fem_smooth_recovers_surface | ok |
| test_fem_smooth_interpolation_limit | ok |
| test_fem_gcv_finite | ok |
| test_fem_smooth_gcv_selects_finite | ok |
| test_fem_predict_matches_nodes | ok |
| test_fem_smooth_obs_outside_mesh_error | ok |
| (5 Plan 01 tests) | ok |

Verify command: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing`

## Deviations from Plan

**1. [Deviation - Task grouping] All 3 tasks implemented in a single commit**

The plan specifies Task 1 (tracer), Task 2 (GCV/edf), Task 3 (gcv search + predict) as separate commits. Because edf and GCV are computed inside `fem_smooth` itself (to avoid double Cholesky factorisation), the GCV logic for Tasks 1 and 2 were written together. The tracer test passed independently (confirmed before writing the Task 2/3 tests), satisfying the tracer gate. All 14 tests pass as a unit.

No architectural deviations — plan executed as specified.

## Threat Mitigations (from STRIDE register)

| Threat ID | Mitigation Applied |
|-----------|--------------------|
| T-44-04 (K null space) | ε=1e-10 ridge on diagonal of A before cholesky_solve; cholesky_factor returns ComputationFailed on residual singularity |
| T-44-05 (obs outside mesh) | fem_basis_eval returns InvalidParameter("query_xy") before any solve; test_fem_smooth_obs_outside_mesh_error verifies no panic |
| T-44-06 (O(N³) DoS) | Documented in fem_smooth rustdoc: "v1 recommends N ≲ 2000"; accepted per plan |

## Self-Check

- [x] `fdars-core/src/fem_smoothing.rs` exists and modified
- [x] `fdars-core/src/lib.rs` extended re-export
- [x] Commit `278ec438` contains all three functions + all tests
- [x] 14/14 tests pass in `fem_smoothing::tests` module
