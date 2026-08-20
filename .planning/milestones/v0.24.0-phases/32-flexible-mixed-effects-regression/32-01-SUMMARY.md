---
plan: 32-01
phase: 32
title: "denseFLMM + multiFAMM + fastFMM (famm.rs extensions)"
status: complete
completed: 2026-08-20
tasks_total: 3
tasks_completed: 3
---

# Plan 32-01 Summary — Wave 1

## What was built

Extended `fdars-core/src/famm.rs` with three flexible functional mixed-effects estimators
(REG-05), all crate-root re-exported and reusing the existing REML-EM machinery.

### 1. `dense_flmm` (denseFLMM) — tracer
Dense functional linear mixed model over FPC scores. Pipeline: `build_subject_map` →
`fdata_to_pc_1d` → per-component `fit_scalar_mixed_model` (REML-EM) → back-project fixed
effects through `fpca.rotation` → `recover_random_effects`. Returns `DenseFlmmResult`
(mean function, fixed-effect functions, random-effect functions, fitted/residuals,
per-time random variance, `sigma2_u`/`sigma2_eps`, and an always-present zero-filled
`sigma2_slope` — random-slope estimation is a documented functionality gap this release,
not an architectural stub) + `DenseFlmmConfig`.

### 2. `multi_famm` (multiFAMM)
D independent per-dimension `dense_flmm` calls, stacked row-wise. `MultiFammResult` +
`MultiFammConfig`. **Documented divergence:** fdars uses D independent univariate FPCAs,
NOT R's joint multivariate FPCA — cross-dimension covariance kernels are not modelled.

### 3. `fast_fmm` (fastFMM)
Per-gridpoint massively-univariate scalar mixed-model fit, running-mean smoothing along
the grid axis (`smooth_window` config), Wald-only (standard-normal) pointwise inference.
`FastFmmResult` (smoothed beta matrix, t-stats, p-values, per-gridpoint variances) +
`FastFmmConfig`. **Documented divergence:** running-mean instead of mgcv splines,
Wald-only instead of bootstrap.

### Supporting change
6 private `famm.rs` helpers promoted private→`pub(crate)` (non-breaking visibility
widening) so Plan 32-02's `fof_regression.rs` path can reuse them:
`fit_scalar_mixed_model`, `build_subject_map`, `SubjectStructure` (+`new`),
`ScalarMixedResult`, `recover_random_effects`.

## Tests (8 new, inline `#[cfg(test)]`)
- `test_dense_flmm_basic` — dimensions, ncomp, zero-filled sigma2_slope
- `test_dense_flmm_fitted_plus_residuals_equals_data` — reconstruction invariant (<1e-6)
- `test_dense_flmm_recovers_signal_and_positive_variance` — residuals <0.5× mean-only baseline; positive random-intercept variance
- `test_dense_flmm_invalid_inputs` — empty / mismatched subject_ids / ncomp==0 → FdarError
- `test_multi_famm_basic` — 2-dim stacked shapes
- `test_multi_famm_invalid_inputs` — empty dims / grid-size mismatch → FdarError
- `test_fast_fmm_basic` — p-values ∈ [0,1], finite t-stats, shapes
- `test_fast_fmm_invalid_inputs` — smooth_window==0 / mismatched subject_ids → FdarError

## Gate status
- `cargo test -p fdars-core --features linalg,parallel --lib famm` — 33/33 pass
- `cargo test --doc famm` — doctest green (E0639-safe: doctest uses `Default::default()` + field mutation)
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean
- `cargo fmt` applied
- No new crate dependency; no existing public signature changed

## Execution note
The initial executor dispatch stalled on the slow `cargo clippy --all-targets` build (compiles
all 28 examples + 8 benches) and dropped its connection twice. Recovery: the `pub(crate)`
promotions and the three estimator bodies (written by the dispatched executors) were preserved;
the orchestrator finished the inline tests + `lib.rs` re-exports + full gate inline, avoiding the
slow-build stall by scoping iteration to `--lib`. All work committed.

## Commits
- `refactor(32-01): promote 6 famm.rs helpers to pub(crate)`
- `feat(32-01): denseFLMM + multiFAMM + fastFMM mixed-model estimators`
