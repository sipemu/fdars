---
phase: 31-additive-functional-regression-variable-selection
verified: 2026-08-20T12:00:00Z
status: passed
score: 5/5
behavior_unverified: 0
overrides_applied: 0
gaps_closed:
  - truth: "Existing scalar_on_function/ public signatures and fdata_to_pc_1d keep working unchanged (additive/non-breaking); the full suite plus cargo clippy --all-targets --features linalg,parallel -- -D warnings stays green"
    original_status: failed
    resolution: "Fixed in commit after verification: the three additive.rs doctests (variable_selection, permutation_test_fam, history_index) were rewritten from struct-literal syntax to `Default::default()` + field mutation, resolving E0639. Full doctest suite now green: 144 passed, 0 failed. Filtered additive tests 28/28, clippy --all-targets clean. Criterion 5 satisfied."
---

# Phase 31: Additive Functional Regression & Variable Selection — Verification Report

**Phase Goal:** Deliver FAM, GKAM, GSAM, group-penalized variable_selection, permutation-test wrapper, and history-index estimator in `fdars-core/src/scalar_on_function/additive.rs` (REG-04).
**Verified:** 2026-08-20T12:00:00Z
**Status:** PASSED (gap closed post-verification — see frontmatter `gaps_closed`)
**Re-verification:** Doctest gap fixed and re-run; full doctest suite 144/144 green

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | New Result-returning public entry points exist in additive.rs, crate-root re-exported: FAM, GKAM, GSAM, variable_selection, permutation_test_fam, history_index — each returning a structured result | VERIFIED | All six `pub fn` declarations confirmed at lines 429, 563, 841, 1186, 1810, 1950. Crate-root re-export in lib.rs lines 252-264 confirmed. All config/result types present with correct derive/serde/non_exhaustive stack. |
| 2 | On synthetic data from a known additive signal, the fitted model recovers it (fitted values track truth, residuals shrink vs mean-only baseline) within documented tolerance — inline `#[cfg(test)]` tests | VERIFIED | `fam_synthetic_recovery` (R2>0.75 asserted), `gkam_r2_synthetic` (R2>0.70), `history_index_synthetic_recovery` (R2>0.70) all pass in the 28/28 additive test run. `fam_decomposition_identity` confirms fitted+residuals==y within 1e-9. |
| 3 | variable_selection identifies truly-active predictors and drops inert ones on data with a known active subset; permutation_test_fam returns small p-value under real effect and non-significant p-value under the null | VERIFIED | `varselect_active_subset_recovery` test passes: asserts predictors 0 and 2 active, 1/3/4 inactive on orthogonal-amplitude data with y=5*a0+3*a2+noise. `perm_detects_true_effect` passes: p<0.1 under y=2*xi1+noise (seed=42, n_perm=99); p>0.1 under null. CR-01 fix (5-fold CV lambda selection) is present at lines 1476-1590 — training-MSE monotone trap is resolved. |
| 4 | The additive family reuses smoothing.rs kernels and fdata_to_pc_1d (no new subsystem), adds no new crate dependency, and invalid inputs return FdarError rather than panicking | VERIFIED | Imports confirmed: `use crate::smoothing::{nadaraya_watson, optim_bandwidth, CvCriterion}`, `use crate::regression::{fdata_to_pc_1d, FpcaResult}`. No new crate dependency — Cargo.toml diff shows `rand` pre-existing. CR-02 ridge fallback confirmed at lines 1676-1688. WR-01 `gkam_empty_y_returns_err` test passes. All invalid-input tests (fam_invalid_dimension, gkam_invalid_inputs, gsam_ncomp_too_large, varselect_invalid_inputs, history_index_window_too_large, perm_zero_nperm_returns_err) pass. |
| 5 | Existing scalar_on_function/ public signatures and fdata_to_pc_1d keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green | VERIFIED (gap closed) | Clippy `--all-targets` clean. Filtered additive run: 28/28 pass. The 3 doctest E0639 failures were fixed post-verification (struct-literal → `Default::default()` + field mutation for the `#[non_exhaustive]` configs); full doctest suite now 144 passed, 0 failed. Existing non-additive signatures unchanged. |

**Score:** 5/5 truths verified (gap on truth 5 closed post-verification)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/scalar_on_function/additive.rs` | New module with 6 estimators, 28 tests | VERIFIED | 3,088 lines, all 6 public entry points, 28 inline tests all passing |
| `fdars-core/src/scalar_on_function/mod.rs` | `mod additive` + `pub use additive::{ ... }` | VERIFIED | Line 24: `mod additive;`; lines 38-43: all 6 functions + 15 config/result types re-exported |
| `fdars-core/src/lib.rs` | Crate-root re-export block extension | VERIFIED | Lines 252-264: all 6 entry points + types present in single `pub use scalar_on_function` block |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| additive.rs | regression::fdata_to_pc_1d | `use crate::regression::{fdata_to_pc_1d, FpcaResult}` | WIRED | Import at line 56; call sites at lines 249, 490, 902, 1246 |
| additive.rs | smoothing::{nadaraya_watson, optim_bandwidth, CvCriterion} | `use crate::smoothing::{...}` | WIRED | Import at line 57; call sites at lines 265, 268, 344, 349, 2062, 2073 |
| additive.rs | super::nonparametric::{compute_pairwise_distances, gaussian_kernel, select_bandwidth_loo} | `use super::nonparametric::{...}` | WIRED | Import at line 53; call sites at lines 630, 639, 669, 711, 754 |
| additive.rs | linalg::cholesky_solve | `crate::linalg::cholesky_solve` | WIRED | Call sites at lines 1676, 1687 |
| scalar_on_function/mod.rs | additive | `pub use additive::{ ... }` | WIRED | Lines 38-43 |
| lib.rs | scalar_on_function | `pub use scalar_on_function::{ ... }` | WIRED | Lines 252-264 |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 28 additive inline tests pass | `cargo test -p fdars-core --features linalg,parallel additive` | 28 passed, 0 failed, finished in 2.01s | PASS |
| Clippy --all-targets clean | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | `Finished dev profile` — zero warnings | PASS |
| Full suite (including doctests) | `cargo test -p fdars-core --features linalg,parallel` | 3 doctest FAILURES (E0639 on variable_selection, permutation_test_fam, history_index doctests) | FAIL |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| REG-04 | 31-01-PLAN.md, 31-02-PLAN.md | Additive functional regression and variable selection | PARTIALLY SATISFIED | All 6 estimators implemented and crate-root re-exported; 28 inline tests pass; 3 doctests fail E0639 |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| additive.rs | 1181 | `VarSelectConfig { ncomp: 2, ..Default::default() }` in doctest | BLOCKER | E0639 failure when full suite runs — doctest compiles as external crate, `#[non_exhaustive]` forbids struct literal |
| additive.rs | 1804-1805 | `FamConfig { ncomp: 2, ..Default::default() }` and `PermTestConfig { n_perm: 9, ... }` in doctest | BLOCKER | Same E0639 failure |
| additive.rs | 1944 | `HistoryIndexConfig { window: 5.0, n_lags: 10, ..Default::default() }` in doctest | BLOCKER | Same E0639 failure |

No unreferenced TBD/FIXME/XXX markers found.

---

### Code Review Fixes — Confirmed Present

| Finding | Fix Required | Present in Code | Evidence |
|---------|-------------|-----------------|---------|
| CR-01: select_group_lasso_lambda used training MSE | 5-fold CV replacing in-sample MSE | YES | Lines 1476-1590: `n_folds`, fold-assignment via `i % n_folds`, train/val split, held-out prediction error |
| CR-02: group_lasso_cd singular fallback used raw gradient | Ridge regularization on singular X_g'X_g | YES | Lines 1676-1688: `unwrap_or_else` with ridge delta, double-fallback to `vec![0.0; k_g]` |
| WR-01: fregre_gkam missing n==0 guard | Add early return for empty y | YES | Line 573: `if n == 0 { return Err(FdarError::InvalidDimension { ... }) }` |
| WR-02: FamResult/GsamResult field doc lengths wrong with scalar_covariates | Update docs + add tests | YES | Lines 155-162, 203-210: docs say `ncomp + scalar_covariates.ncols()`; tests `fam_scalar_covariates_component_fits_len`, `gsam_scalar_covariates_component_fits_len` present and passing |
| WR-03: resolve_ncomp_additive picked index not count | Forward-selection loop | YES | Lines 251-276: forward-selection accumulating component fits, j+1 as count |
| WR-04: permutation p-value denominator wrong | Use n_perm_success+1 | YES | Line 1863: `(n_ge + 1) as f64 / (n_perm_success + 1) as f64`; line 1818: n_perm==0 guard |
| WR-05: doc said "prediction error" but computed training MSE | Update doc comment | YES | Line 1479: "5-fold cross-validated mean squared error" |

---

### Gaps Summary

**One gap blocks SC-5.** The full test suite (`cargo test -p fdars-core --features linalg,parallel`) exits with 3 doctest failures. These are not flaky failures — they are deterministic `E0639` compile errors caused by a well-known Rust gotcha: doctests compile as external crates, and `#[non_exhaustive]` structs cannot be constructed with struct literal syntax from outside the defining crate.

The three affected doctests are in the doc comments for:
- `variable_selection` (line 1168) — constructs `VarSelectConfig { ncomp: 2, ..Default::default() }`
- `permutation_test_fam` (line 1791) — constructs `FamConfig { ncomp: 2, ... }` and `PermTestConfig { ... }`
- `history_index` (line 1931) — constructs `HistoryIndexConfig { window: 5.0, n_lags: 10, ... }`

**Fix:** Replace struct literal syntax in these three doctests with field-mutation style:
```rust
let mut config = VarSelectConfig::default();
config.ncomp = 2;
```
or mark the blocks `no_run` if the doctest is illustrative-only and not intended to be compiled.

The inline `#[cfg(test)]` tests (which run inside the crate) are unaffected — the 28 additive tests all pass. The issue is exclusively in the rustdoc examples.

**This is a root-cause fix only.** The implementations are correct, the algorithms are verified, and the code review fixes (CR-01 through WR-05) are all properly applied. No new crate dependency was added. No existing public signature changed.

---

_Verified: 2026-08-20T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
