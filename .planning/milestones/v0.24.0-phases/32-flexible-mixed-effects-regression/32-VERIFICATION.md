---
phase: 32-flexible-mixed-effects-regression
verified: 2026-08-20T21:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 32: Flexible Mixed-Effects Regression Verification Report

**Phase Goal:** Extend `fdars-core` with a functional mixed-effects regression family (REG-05): `dense_flmm`, `multi_famm`, `fast_fmm` in `famm.rs` and `fof_re_regression`/`predict_fof_re` in `fof_regression.rs`, all crate-root re-exported, covering fixed + random effects, variance components, and fitted functional curves; plus a code-review fix pass (CR-01 + WR-01..04) confirmed in-code.

**Verified:** 2026-08-20T21:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1 | `dense_flmm`, `multi_famm`, `fast_fmm` are `Result`-returning, `#[must_use]`, crate-root re-exported; each returns fixed-effect estimates, random-effect/variance-component estimates, and fitted functional curves | VERIFIED | All three exist at `famm.rs:1036,1337,1521`; `lib.rs:234-236` re-exports all nine symbols; all carry `#[must_use = "..."]`; `DenseFlmmResult` fields include `beta_functions`, `random_effects`, `sigma2_u`, `fitted` |
| 2 | `fof_re_regression` is wired into `fof_regression.rs` extending only the RE variant; returns structured result over functional response; base FoF signatures untouched | VERIFIED | `fof_re_regression` at `fof_regression.rs:675`; calls `crate::famm::build_subject_map`, `crate::famm::fit_scalar_mixed_model`, `crate::famm::recover_random_effects`; `FofReResult` carries all required fields; `fof_regression/predict_fof/fof_cv` signatures unchanged |
| 3 | Inline `#[cfg(test)]` tests recover fixed effects and random-effect/variance-component structure from synthetic grouped data within documented tolerances | VERIFIED | `test_dense_flmm_recovers_signal_and_positive_variance` (resid_ss < 0.5 * base_ss, at least one sigma2_u > 0); `test_dense_flmm_fitted_plus_residuals_equals_data` (1e-6 tol); `test_fof_re_regression_invariant` (1e-6 tol); `test_fof_re_regression_re_nonzero` (L2 norm > 0 under grouping); all 41 famm + 16 fof_regression tests pass |
| 4 | Mixed-model family reuses existing `famm.rs` fixed-effect machinery; no new crate dependency; invalid inputs return `FdarError` rather than panicking | VERIFIED | Six helpers promoted to `pub(crate)` (commit 96ea9f62); `fof_re_regression` calls `crate::famm::*` helpers; no diff to `fdars-core/Cargo.toml`; dimension/parameter guards in all three estimators tested (`test_dense_flmm_invalid_inputs`, `test_multi_famm_invalid_inputs`, `test_fast_fmm_invalid_inputs`, `test_fast_fmm_empty_data_error`, `test_fof_re_regression_ids_mismatch`); `random_slopes=true` now returns `InvalidParameter`; `max_iter=0` returns `InvalidParameter` |
| 5 | Existing `fmm/fmm_predict/fmm_test_fixed` and base `fof_regression.rs` signatures unchanged; scoped tests green; clippy clean (orchestrator-confirmed) | VERIFIED | All base entry points confirmed present at original file positions; 41 famm tests include all legacy fmm/fmm_predict/fmm_test_fixed tests (all pass); 16 fof_regression tests include all legacy fof_regression/predict_fof/fof_cv tests (all pass); clippy `--all-targets --features linalg,parallel -- -D warnings` confirmed clean by orchestrator (not re-run per instructions) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/famm.rs` | Extended with `dense_flmm`, `DenseFlmmConfig`, `DenseFlmmResult`, `multi_famm`, `MultiFammConfig`, `MultiFammResult`, `fast_fmm`, `FastFmmConfig`, `FastFmmResult` + inline tests | VERIFIED | 2625-line file; all nine symbols present; 41 tests in `#[cfg(test)] mod tests` |
| `fdars-core/src/fof_regression.rs` | Extended with `fof_re_regression`, `predict_fof_re`, `FofReConfig`, `FofReResult` + inline tests | VERIFIED | 1376-line file; all four symbols present; 16 tests covering RE and legacy paths |
| `fdars-core/src/lib.rs` | Extended famm and fof_regression re-export lines | VERIFIED | Lines 234-236: famm re-exports (9 symbols); lines 247-248: fof_regression re-exports (all including `fof_re_regression`, `predict_fof_re`, `FofReConfig`, `FofReResult`) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `dense_flmm` | `fdata_to_pc_1d` → `fit_scalar_mixed_model_tracked` | per-component parallel loop + `recover_beta_functions`/`recover_random_effects` | WIRED | `famm.rs:1082-1161`; uses `iter_maybe_parallel!` collect pattern; CR-01 fix confirmed — `fit_scalar_mixed_model_tracked` with `config.max_iter`/`config.tol` at line 1097-1104 |
| `multi_famm` | `dense_flmm` (per dimension) | `DenseFlmmConfig` built from `MultiFammConfig`; stacked fitted/residuals assembled | WIRED | `famm.rs:1382-1430`; delegates each dimension to `dense_flmm`; stacks rows via column-major `FdMatrix` |
| `fast_fmm` | `fit_scalar_mixed_model_tracked` (per gridpoint) | `iter_maybe_parallel!(0..m).map(...).collect()` pattern; WR-01 odd-window fix | WIRED | `famm.rs:1568-1586`; column-major zero-copy `data.column(t).to_vec()`; odd-window guard at line 1606-1610 |
| `fof_re_regression` | `fdata_to_pc_1d` (x and y) → `crate::famm::build_subject_map` → `crate::famm::fit_scalar_mixed_model` (per Y-score) → `crate::famm::recover_random_effects` | cross-module `pub(crate)` | WIRED | `fof_regression.rs:749,769,816`; `pub(crate)` promotions in `famm.rs:185,264,272,438,674` confirmed |
| Six `pub(crate)` helpers | Sibling module `fof_regression.rs` | visibility widening (non-breaking) | WIRED | Commit 96ea9f62; `build_subject_map`, `ScalarMixedResult`, `SubjectStructure`, `fit_scalar_mixed_model`, `recover_random_effects` all at `pub(crate)` |
| Crate-root re-exports | `lib.rs` | `pub use famm::{...}` and `pub use fof_regression::{...}` | WIRED | `lib.rs:234-236,247-248`; smoke test `test_fof_re_reexport` passes |

### Code-Review Fix Verification (CR-01 / WR-01..04)

| Finding | Fix Required | Fix Present | Evidence |
|---------|-------------|-------------|----------|
| CR-01: `fast_fmm` ignores `config.max_iter` and `config.tol` | Replace `fit_scalar_mixed_model` with `fit_scalar_mixed_model_tracked` passing config fields | VERIFIED | `famm.rs:1571-1578`; `test_fast_fmm_max_iter_takes_effect` passes (1-iter vs 100-iter sigma2_eps differ) |
| WR-01: Running-mean smoother wrong width for even `smooth_window` | Round even window up to nearest odd | VERIFIED | `famm.rs:1606-1610`; `test_fast_fmm_even_smooth_window` passes (w=4 == w=5 output) |
| WR-02: `random_slopes=true` silently ignored | Return `InvalidParameter` | VERIFIED | `famm.rs:1070-1076`; `test_dense_flmm_random_slopes_errors` passes |
| WR-03: `max_iter=0` silently wrong | Return `InvalidParameter` in `dense_flmm` and `fast_fmm` | VERIFIED | `famm.rs:1064-1068` (dense), `1549-1553` (fast); both tests pass |
| WR-04: Missing tests REG-05-G, REG-05-K, REG-05-L | Add three tests | VERIFIED | `test_dense_flmm_converged` (line 2423), `test_fast_fmm_detects_effect` (line 2452), `test_fast_fmm_empty_data_error` (line 2483) all present and passing |
| IN-03: `sigma2_eps` doc comment | Expand field doc | VERIFIED | `famm.rs:978-984`; notes L²-normalized scale and non-comparability to R's `lmer()` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 41 famm tests (new + legacy) | `cargo test -p fdars-core --features linalg,parallel --lib -- famm` | 41 passed; 0 failed; finished in 2.24s | PASS |
| All 16 fof_regression tests (new + legacy) | `cargo test -p fdars-core --features linalg,parallel --lib -- fof_regression` | 16 passed; 0 failed; finished in 0.68s | PASS |
| famm doc tests | `cargo test -p fdars-core --features linalg,parallel --doc -- famm` | 1 passed; 0 failed | PASS |
| fof doc tests | `cargo test -p fdars-core --features linalg,parallel --doc -- fof` | 4 passed; 0 failed | PASS |

### Anti-Patterns Found

No TBD, FIXME, or XXX markers in `famm.rs` or `fof_regression.rs`. The three occurrences of "not yet implemented" in `famm.rs` are part of the documented API contract for `random_slopes` (the function returns an error when that field is `true` — the text is in rustdoc and an error message string, not a deferred-work marker). No empty implementations, no hardcoded-empty data returned from public paths.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| REG-05 | 32-01-PLAN.md, 32-02-PLAN.md | Flexible mixed-effects regression: dense FLMM, multivariate FAMM, fast FMM, and RE function-on-function estimator | SATISFIED | Five new public functions, nine config/result structs, 57 total tests passing (41 famm + 16 fof_regression) |

---

## Summary

All five roadmap success criteria are verified in the codebase:

1. `dense_flmm`, `multi_famm`, and `fast_fmm` exist in `famm.rs` as `Result`-returning, `#[must_use]`, crate-root re-exported functions returning fixed effects, random effects, variance components, and fitted curves. The code-review critical fix (CR-01) is confirmed: `fast_fmm` calls `fit_scalar_mixed_model_tracked` passing `config.max_iter` and `config.tol` at lines 1571-1578.

2. `fof_re_regression` is wired into `fof_regression.rs` exclusively as a RE variant, calling the six `pub(crate)` helpers promoted in Plan 01. Base `fof_regression`/`predict_fof`/`fof_cv` signatures are unchanged and their tests pass.

3. Inline tests with synthetic grouped data confirm: fixed-effect function recovery (residuals < 50% of baseline SS), fitted+residuals == data (tol 1e-6), random effects non-zero under grouping, and variance components positive. All 57 tests pass under `--features linalg,parallel`.

4. No new crate dependency (zero diff to `fdars-core/Cargo.toml`); all invalid-input paths (empty data, mismatched subject_ids, mismatched grid, `random_slopes=true`, `max_iter=0`) return `FdarError`.

5. Legacy entry points (`fmm`, `fmm_predict`, `fmm_test_fixed`, `fof_regression`, `predict_fof`, `fof_cv`) exist unchanged; their tests all pass; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` confirmed clean by orchestrator.

Phase goal achieved. No gaps, no deferred items, no human verification required.

---

_Verified: 2026-08-20T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
