---
phase: 43-boosting-bayesian-functional-regression
plan: "01"
subsystem: boosting_regression
status: complete
tags: [boosting, fosr, functional-regression, bspline, cholesky, REG-06-01]
completed: "2026-08-24"
duration_minutes: 13

dependency_graph:
  requires: []
  provides:
    - fdars-core/src/boosting_regression/mod.rs (BoostingConfig, BayesianConfig, StabilityConfig, all 5 result structs)
    - fdars-core/src/boosting_regression/boost_fosr.rs (boost_fosr, boost_fosr_one_step)
    - fdars-core/src/boosting_regression/boost_fofr.rs (skeleton for Plan 02)
    - fdars-core/src/boosting_regression/gamlss.rs (skeleton for Plan 03)
    - fdars-core/src/boosting_regression/bayesian.rs (skeleton for Plan 04)
    - fdars-core/src/boosting_regression/stability.rs (skeleton for Plan 05)
  affects:
    - fdars-core/src/lib.rs (new pub mod + re-exports)
    - fdars-core/src/prelude.rs (BayesianFosrResult, BoostFosrResult added)

tech_stack:
  added: []
  patterns:
    - penalized B-spline base-learners (cholesky_factor + cholesky_forward_back amortized per-learner)
    - component-wise boosting (argmin RSS base-learner selection)
    - column-major FdMatrix column() access for cache-efficient inner loops
    - GCV path tracking (gcv_path as ‖U‖_F² per iteration)

key_files:
  created:
    - fdars-core/src/boosting_regression/mod.rs
    - fdars-core/src/boosting_regression/boost_fosr.rs
    - fdars-core/src/boosting_regression/boost_fofr.rs
    - fdars-core/src/boosting_regression/gamlss.rs
    - fdars-core/src/boosting_regression/bayesian.rs
    - fdars-core/src/boosting_regression/stability.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

decisions:
  - Beta matrix stores mean fitted-value proxy per predictor/time-point (not full K-vector of B-spline coefficients), keeping the result struct at (p × m_t) regardless of nbasis
  - BaseLearner struct holds pre-factored Cholesky L (K × K) + design matrix Φ (n × K, column-major); factorization amortized across all mstop iterations (only back-solves per iteration)
  - Four skeleton files return FdarError::ComputationFailed("not yet implemented (Plan NN)") — clean clippy, no dead_code warnings
  - All base-learners share nbasis/order/lambda to equalise effective df (Pitfall 4 in RESEARCH.md)

metrics:
  duration_minutes: 13
  tasks_completed: 3
  commits: 3
  files_created: 6
  files_modified: 2
  lines_added: 1215

actuals:
  tokens: 18750
  tasks: 3
  commits: 3
---

# Phase 43 Plan 01: Boosting Core + FOSR + Module Scaffold Summary

**One-liner:** Component-wise B-spline-boosted FOSR (REG-06-01) with penalized Cholesky base-learners, plus full module scaffold (all 5 config structs, all 5 result structs, 4 compiling skeletons) registered in lib.rs and prelude.rs.

## What Was Built

### Task 1: Module Scaffold + boost_fosr Tracer (feat 95ece8ae)

Created `fdars-core/src/boosting_regression/` with 6 files:

**`mod.rs`** (283 lines) — module barrel containing:
- `BoostingConfig` (mstop, nu, nbasis, order, lfd_order, lambda, ncomp_x, seed)
- `BayesianConfig` (ncomp, tau2, ig_a0, ig_b0, n_iter, burn_in, thin, seed)
- `StabilityConfig` (n_resamples, pi_thr, seed)
- `BoostFosrResult`, `BoostFofrResult`, `GamlssResult`, `BayesianFosrResult`, `StabilityResult`
- All `#[derive(Debug, Clone, PartialEq)]`; result structs also `#[non_exhaustive]`
- Barrel `pub use self::*::*` re-exports for all 5 public functions

**`boost_fosr.rs`** (707 lines) — full implementation of REG-06-01:
- Input validation (dimension + parameter checks → FdarError)
- `build_bspline_design` helper: evaluates B-spline basis at arbitrary scalar predictor values via `bspline_basis`
- `BaseLearner` struct: pre-factored Cholesky L (K × K) + column-major Φ (n × K)
- `boost_fosr_one_step` (pub crate): single boosting iteration; called by boost_fosr loop and available for gamlss.rs
- `boost_fosr` (public): full mstop-iteration boosting loop with gcv_path, selected_learners
- `pointwise_r_squared` helper (copied from function_on_scalar.rs pattern)

**Four skeletons**: `boost_fofr.rs`, `gamlss.rs`, `bayesian.rs`, `stability.rs` — each with the correct public signature returning `Err(FdarError::ComputationFailed { operation: "<name>", detail: "not yet implemented (Plan NN)" })`.

**`src/lib.rs`**: added `pub mod boosting_regression;` and crate-root re-export block for all 8 public types + 5 functions.

**`src/prelude.rs`**: added `BayesianFosrResult` and `BoostFosrResult`.

### Task 2: TDD Tests — boost_fosr (test 8d9af855)

Added 6 inline `#[test]` functions to `boost_fosr.rs`:
- `boost_fosr_reduces_rss_monotonically` — gcv_path non-increasing
- `boost_fosr_recovers_known_beta` — R² > 0.8 on Y = x·sin(πt) + noise
- `boost_fosr_r_squared_in_range` — all r_squared_t in [-0.05, 1+ε]
- `boost_fosr_selected_learners_valid` — len==mstop, all indices < p
- `boost_fosr_errors_on_dimension_mismatch` — FdarError::InvalidDimension
- `boost_fosr_errors_on_invalid_params` — FdarError::InvalidParameter for mstop=0, nu<0, lambda=0, nbasis<4

All 6 tests pass.

### Task 3: Clippy + Full Test Gate (chore c50eb195)

Fixed `clippy::manual_memcpy` in test helper (replaced manual loop with `copy_from_slice`).

Gate results:
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: **clean** (0 warnings)
- `cargo test -p fdars-core --features linalg,parallel`: **3275 tests, 0 failures**

## Verification

Module-scoped test run:
```
test boosting_regression::boost_fosr::tests::boost_fosr_errors_on_dimension_mismatch ... ok
test boosting_regression::boost_fosr::tests::boost_fosr_errors_on_invalid_params ... ok
test boosting_regression::boost_fosr::tests::boost_fosr_r_squared_in_range ... ok
test boosting_regression::boost_fosr::tests::boost_fosr_recovers_known_beta ... ok
test boosting_regression::boost_fosr::tests::boost_fosr_reduces_rss_monotonically ... ok
test boosting_regression::boost_fosr::tests::boost_fosr_selected_learners_valid ... ok
test result: ok. 6 passed; 0 failed; 0 ignored
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `FdMatrix::from_column_major` returns `Result<_, FdarError>`**
- **Found during:** Task 2 (test compilation)
- **Issue:** Test code used `FdMatrix::from_column_major(data, n, m)` without `.unwrap()` — compile error.
- **Fix:** Added `.unwrap()` in test helper `make_synthetic` and fixed doctest example (changed p from 2 to 1 to match x_vals length).
- **Files modified:** `boost_fosr.rs`
- **Commit:** 8d9af855

**2. [Rule 1 - Bug] `clippy::manual_memcpy` in test helper**
- **Found during:** Task 3 (clippy gate)
- **Issue:** Manual loop `pred_data[i] = x1[i]` flagged by clippy.
- **Fix:** Replaced with `copy_from_slice`.
- **Files modified:** `boost_fosr.rs`
- **Commit:** c50eb195

### Design Choice: Beta Matrix as Mean-Effect Proxy

The `beta` field in `BoostFosrResult` stores the running mean fitted-value (averaged over observations) rather than B-spline coefficient vectors. This keeps `beta` as (p × m_t) regardless of `nbasis`, matching the `FosrResult.beta` convention. The full K-dimensional B-spline coefficients are available via `_coefs_star` in `boost_fosr_one_step` for callers (e.g., `gamlss.rs`) that need them.

## Known Stubs

The four skeleton files (`boost_fofr.rs`, `gamlss.rs`, `bayesian.rs`, `stability.rs`) return `FdarError::ComputationFailed("not yet implemented (Plan NN)")`. These are intentional and tracked:
- `boost_fofr.rs` → Plan 02
- `gamlss.rs` → Plan 03
- `bayesian.rs` → Plan 04
- `stability.rs` → Plan 05

## Self-Check: PASSED

All 6 created/modified files verified on disk. All 3 task commits verified in git log.
