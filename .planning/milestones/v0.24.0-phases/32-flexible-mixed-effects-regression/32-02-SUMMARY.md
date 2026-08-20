---
phase: 32-flexible-mixed-effects-regression
plan: "02"
subsystem: fof_regression
status: complete
tags: [functional-data, mixed-effects, function-on-function, fpca, rust]
requirements: [REG-05]

dependency_graph:
  requires: ["32-01"]
  provides: [fof_re_regression, predict_fof_re, FofReConfig, FofReResult]
  affects: [fdars-core/src/fof_regression.rs, fdars-core/src/lib.rs]

tech_stack:
  added: []
  patterns:
    - double-FPCA + per-Y-score scalar LMM via famm::fit_scalar_mixed_model
    - recover_random_effects from famm to reconstruct subject-level curve effects
    - fixed-effect-only predict_fof_re (no RE for new/unseen subjects)

key_files:
  created: []
  modified:
    - fdars-core/src/fof_regression.rs
    - fdars-core/src/lib.rs

decisions:
  - "Pass x_scores directly as covariates to fit_scalar_mixed_model WITHOUT h.sqrt() rescaling (Pitfall 2: L2 weighting already embedded in fpca_x.project)"
  - "Transpose u_hat_per_component from [component][subject] to [subject][component] before calling recover_random_effects to match its expected layout"
  - "predict_fof_re is fixed-effect-only, matching fmm_predict / predict_fof convention"
  - "FofReConfig has no #[non_exhaustive] (callers may use struct literal); FofReResult has #[non_exhaustive]"

metrics:
  duration: "~15 minutes"
  completed: "2026-08-20"
  tasks_completed: 2
  tasks_total: 2
  commits: 1

actuals:
  tokens: 17500
  tasks: 2
  commits: 1
---

# Phase 32 Plan 02: FoF Random-Effects Estimator Summary

Flexible random-effects function-on-function regression (`fof_re_regression` + `predict_fof_re`) wired into `fof_regression.rs` as the fourth REG-05 estimator, reusing `famm::fit_scalar_mixed_model` and `famm::recover_random_effects` from Plan 01.

## What Was Built

### `FofReConfig` (no `#[non_exhaustive]`)
Config struct with `ncomp_x` (default 3), `ncomp_y` (default 3), `max_iter` (default 50), `tol` (default 1e-10). Implements `Default`; callers may use struct-literal syntax.

### `FofReResult` (`#[non_exhaustive]`)
Result struct carrying all base `FofResult` fields plus: `random_effects` (n_subjects × m_y), `sigma2_u` (per Y-score component variance, length ncomp_y), `sigma2_eps` (mean residual variance), `n_subjects`.  Also carries `fpca_x` and `fpca_y` so `predict_fof_re` can reuse the same projection path.

### `fof_re_regression`
Three-step algorithm:
1. **Double FPCA** — same as `fof_regression` (fdata_to_pc_1d on X and Y, project to score space).
2. **Per-Y-score scalar LMM** — for each Y-score component `l`, call `famm::fit_scalar_mixed_model(y_scores[:,l], subject_map, n_subjects, Some(&x_scores), ncomp_x)`. X-scores are passed WITHOUT h.sqrt() rescaling (Pitfall 2 avoidance).
3. **Reconstruction** — build beta_surface from coef_matrix + FPCA rotations; compute fitted as mean_y + fixed-score contribution + subject random-effect contribution; recover random_effects via `famm::recover_random_effects` (with [component][subject] → [subject][component] transpose).

Input validation: n_x == n_y, n >= 3, argvals lengths, ncomp >= 1, subject_ids.len() == n — all return `FdarError` never panic.

### `predict_fof_re`
Fixed-effect-only prediction: project new_x onto fit.fpca_x, compute predicted Y-scores via coef_matrix, reconstruct through fpca_y rotation and mean. No random effects added for new/unseen subjects (same convention as `fmm_predict`).

### lib.rs re-exports
Extended `pub use fof_regression::{...}` to include `fof_re_regression, predict_fof_re, FofReConfig, FofReResult`.

## Tests (7 new, all inline in `fof_regression.rs`)

| Test | Assertion |
|------|-----------|
| `test_fof_re_regression_dims` | beta_surface m_y×m_x, fitted n×m_y, residuals n×m_y, random_effects n_subjects×m_y, coef_matrix 3×3, sigma2_u len 3 |
| `test_fof_re_regression_invariant` | fitted[(i,j)] + residuals[(i,j)] == y_data[(i,j)] within 1e-6 for all (i,j) |
| `test_fof_re_regression_re_nonzero` | L2 norm of random_effects > 0 under grouped data with subject-level shifts |
| `test_fof_re_regression_ids_mismatch` | subject_ids.len() != n → FdarError::InvalidDimension { parameter: "subject_ids" } |
| `test_fof_re_reexport` | smoke test: fof_re_regression accessible via super::* |
| `test_predict_fof_re_shape` | output shape (n_new, m_y), all entries finite |
| `test_predict_fof_re_training_matches_fixed` | on training X, predict_fof_re == mean_y + X_scores * coef_matrix reconstructed (fixed-effect part only) within 1e-6 |

All 16 `fof_regression` tests (9 original + 7 new) pass. Base `fof_regression`, `predict_fof`, `fof_cv` signatures unchanged.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Transposed u_hat layout for recover_random_effects**
- **Found during:** Task 1 — first test run after GREEN implementation
- **Issue:** `u_hat_per_component` was accumulated as `[component_index][subject_index]` but `recover_random_effects` expects `[subject_index][component_index]`
- **Fix:** Added an explicit transpose loop before passing to `recover_random_effects`
- **Files modified:** `fdars-core/src/fof_regression.rs`
- **Commit:** 0f97c7c4

**2. [Rule 1 - Bug] Closure capture borrow in test_fof_re_regression_re_nonzero**
- **Found during:** Task 1 RED compile
- **Issue:** `flat_map` closure tried to `move` `fit.random_effects` (which is not Copy) across multiple iterations
- **Fix:** Replaced with explicit nested for-loop accumulating `l2_sq`
- **Files modified:** `fdars-core/src/fof_regression.rs`
- **Commit:** 0f97c7c4

## Known Stubs

None — all fields are wired to real computed values.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. The only new trust boundary is `subject_ids: &[usize]` at function entry — validated (`len == n`) with `FdarError::InvalidDimension`, matching T-32-03 mitigation in the plan's threat register.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| `fdars-core/src/fof_regression.rs` | FOUND |
| `fdars-core/src/lib.rs` | FOUND |
| `32-02-SUMMARY.md` | FOUND |
| Commit `0f97c7c4` | FOUND |
| `fof_re_regression` symbol | FOUND (line 675) |
| `predict_fof_re` symbol | FOUND (line 943) |
| `FofReConfig` symbol | FOUND (line 528) |
| `FofReResult` symbol | FOUND (line 568) |
| lib.rs re-exports | FOUND (lines 247-248) |
| All 16 fof_regression tests | PASSED |
