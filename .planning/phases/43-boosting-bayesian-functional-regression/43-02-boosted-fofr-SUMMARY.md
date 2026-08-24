---
phase: 43-boosting-bayesian-functional-regression
plan: "02"
subsystem: boosting_regression
tags: [functional-regression, boosting, fpc, function-on-function, REG-06-02]
requirements: [REG-06-02]
status: complete

dependency_graph:
  requires: [43-01-boosting-core-fosr]
  provides: [boost_fofr, BoostFofrResult — bfpc FoFR base-learner with coefficient-surface reconstruction]
  affects: [fdars-core/src/boosting_regression/boost_fofr.rs]

tech_stack:
  added: []
  patterns:
    - "FPC-score signal compression: fdata_to_pc_1d → S_j (n × K_j) design matrices"
    - "Amortised Cholesky: factor (S_j'S_j + ε·I) once per learner; back-solve per time point"
    - "Coefficient-surface reconstruction: rotation_j (m_x × K_j) · score_coefs_j (K_j × m_y)"
    - "Component-wise argmin selection mirroring boost_fosr loop"
    - "Column-major FdMatrix access (data.column(t)) in outer time-point loop"

key_files:
  created: []
  modified:
    - fdars-core/src/boosting_regression/boost_fofr.rs

decisions:
  - "Used bfpc (FPC-score compression) variant instead of FDboost bsignal B-spline joint expansion — simpler, zero new deps, documented divergence in rustdoc"
  - "Ridge jitter 1e-10·I on S_j'S_j (not lambda-penalized) — FPC scores already have smoothness from FPCA basis; penalty would double-regularize"
  - "Coefficient reconstruction via matrix product rotation_j · score_coefs_j gives m_x × m_y surface matching BoostFofrResult.beta_surfaces[j] field spec"
  - "Pre-factored ScoreLearner caches Cholesky L outside boosting loop; only O(K²) back-solve runs per (learner, time_point) inside the loop"

metrics:
  duration_minutes: 6
  completed_date: "2026-08-24"
  tasks_completed: 2
  tasks_total: 2
  commits: 2
  files_modified: 1

actuals:
  tokens: 18500
  tasks: 2
  commits: 2
---

# Phase 43 Plan 02: Boosted Function-on-Function Regression (REG-06-02) Summary

**One-liner:** Boosted FoFR via FPC-score signal compression (bfpc): per-predictor FPCA design, amortised Cholesky base-learner solve, and rotation-matrix coefficient-surface reconstruction.

## What Was Built

Replaced the `boost_fofr.rs` skeleton with a complete bfpc boosted FoFR implementation satisfying REG-06-02. The function:

1. **Validates inputs** — dimension checks on all predictors/response (n mismatch, argvals length, empty predictor slice), parameter bounds (mstop ≥ 1, nu ∈ (0,1], ncomp_x ≥ 1).

2. **Preprocesses predictors** — calls `fdata_to_pc_1d(X_j, ncomp_x, argvals_j)` for each functional predictor j, yielding FPC scores `S_j ∈ R^{n × K_j}` that serve as the base-learner design matrices.

3. **Pre-factors base-learners** — for each j, builds `(S_j'S_j + 1e-10·I)` row-major and factors via `cholesky_factor`. The factor is cached in a `ScoreLearner` struct, amortising the O(K³) cost over all mstop iterations.

4. **Runs the boosting loop** — at each iteration:
   - Computes residual `U = Y − F̂` (n × m_y)
   - For each learner j: solves `(S_j'S_j + ε·I) c_j(t) = S_j'·U[:,t]` per time point t using `cholesky_forward_back` (O(K²)), giving fitted `Ĥ_j = S_j · c_j`
   - Selects `j* = argmin_j ‖U − Ĥ_j‖_F²`
   - Updates `F̂ += ν·Ĥ_{j*}` and accumulates `score_coefs[j*] += ν·c_{j*}`

5. **Reconstructs coefficient surfaces** — `β_j(s,t) = rotation_j (m_x × K_j) · score_coefs_j (K_j × m_y)` gives the `(m_x × m_y)` coefficient surface for each predictor, including zero surfaces for never-selected predictors.

6. **Returns `BoostFofrResult`** — fitted (n × m_y), residuals, r_squared_t, r_squared, fpca_x (one per predictor), score_coefs, beta_surfaces, selected_learners (length mstop), gcv_path (‖U‖_F² per iteration), mstop, nu.

## Tests Added

Six inline `#[cfg(test)]` tests covering all required behaviors:

| Test | Behavior verified |
|------|-------------------|
| `boost_fofr_fitted_shape` | fitted == (n, m_y), residuals == (n, m_y), intercept/r_squared_t lengths |
| `boost_fofr_residuals_decrease` | gcv_path non-increasing on signal-bearing synthetic data |
| `boost_fofr_r_squared_in_range` | global and pointwise R² in [-0.05, 1+ε] |
| `boost_fofr_beta_surface_shape` | one beta_surface per predictor, each (m_x, m_y) |
| `boost_fofr_errors_on_dimension_mismatch` | row mismatch, argvals length, y_argvals length → InvalidDimension |
| `boost_fofr_errors_on_invalid_params` | mstop=0, ncomp_x=0, nu>1, empty predictor slice → correct error types |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Clippy::approx_constant in test helper**
- **Found during:** Task 2 (clippy gate)
- **Issue:** Test helper used `2.718` as an approximation of Euler's number, triggering `clippy::approx_constant` deny lint
- **Fix:** Replaced with `std::f64::consts::E`
- **Files modified:** `fdars-core/src/boosting_regression/boost_fofr.rs`
- **Commit:** 954db9c4

None other — plan executed as designed.

## Verification Results

```
Module-scoped test run:
cargo test -p fdars-core --features linalg,parallel boost_fofr
running 6 tests
test boosting_regression::boost_fofr::tests::boost_fofr_errors_on_invalid_params ... ok
test boosting_regression::boost_fofr::tests::boost_fofr_errors_on_dimension_mismatch ... ok
test boosting_regression::boost_fofr::tests::boost_fofr_r_squared_in_range ... ok
test boosting_regression::boost_fofr::tests::boost_fofr_residuals_decrease ... ok
test boosting_regression::boost_fofr::tests::boost_fofr_fitted_shape ... ok
test boosting_regression::boost_fofr::tests::boost_fofr_beta_surface_shape ... ok
test result: ok. 6 passed; 0 failed; 0 ignored

Clippy gate:
cargo clippy --all-targets --features linalg,parallel -- -D warnings
Finished dev profile [unoptimized + debuginfo] target(s) — clean
```

## Known Stubs

None — `boost_fofr` is fully implemented end-to-end.

## Self-Check

- [x] `fdars-core/src/boosting_regression/boost_fofr.rs` exists with full implementation
- [x] Commit d19d8720 exists (feat: implement boost_fofr)
- [x] Commit 954db9c4 exists (test: add inline tests)
- [x] No "not yet implemented" string in boost_fofr.rs
- [x] `fdata_to_pc_1d` present
- [x] `beta_surfaces` reconstruction present
- [x] 6 `#[test]` functions (≥ 5 required)
- [x] All 6 tests green
- [x] Clippy gate clean

## Self-Check: PASSED
