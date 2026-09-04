---
phase: 71-prediction-diagnostics-integration
verified: 2026-09-04T00:00:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 71: Prediction, Diagnostics & Integration Verification Report

**Phase Goal:** Both fitted regressors predict on new curves and expose their coefficient function and fitted values; the full public wavelet surface (DWT + `wcr` + `wnet` + config/result types + `predict`) is reachable from the crate root and prelude, with a running end-to-end doctest — all additive and non-breaking.
**Verified:** 2026-09-04
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `WcrResult::predict` and `WnetResult::predict` re-pass training curves and reproduce stored `fitted_values` within 1e-8 abs (SC1, WAV-05) | ✓ VERIFIED | `wcr_predict_reproduces_training_fitted` passes (tolerance enforced in source as `<= 1e-8`); `wnet_predict_reproduces_training_fitted` passes. Both run in `fdars-core/src/wavelet/regression.rs`. |
| 2 | `predict` on fresh curves returns finite, NaN-free values; training-grid-length mismatch returns `FdarError`, never panics (SC1, WAV-05) | ✓ VERIFIED | `wcr_predict_on_new_curves_is_finite_and_rejects_grid_mismatch` PASS; `wnet_predict_on_new_curves_is_finite_and_rejects_grid_mismatch` PASS. Both assert `Err(FdarError::InvalidDimension { .. })` on wrong ncols. |
| 3 | `beta_t()`/`coefficient_function()` and `fitted_values()` accessors on both structs return stored slices with expected lengths (SC2, WAV-05) | ✓ VERIFIED | `accessors_return_stored_slices` PASS. Confirms `beta_t()` == `.beta_t` field, `coefficient_function()` == `.beta_t` field, `fitted_values()` == `.fitted_values` field for both `WcrResult` and `WnetResult`; lengths m and n. |
| 4 | Full wavelet surface reachable via BOTH `fdars_core::prelude::*` AND crate-root paths (SC3, WAV-06) | ✓ VERIFIED | `lib.rs` lines 604–608: `pub use wavelet::regression::{wcr, wnet, WcrConfig, WcrMethod, WcrResult, WnetConfig, WnetResult};` + `pub use wavelet::{decompose, decompose_matrix, max_level, reconstruct, BoundaryMode, WaveletCoeffs, WaveletFamily};`. `prelude.rs` lines 112–118: mirrored groups. 14-symbol surface confirmed present in both locations. |
| 5 | Running end-to-end module doctest (fit→predict→beta_t/fitted via prelude) passes under `cargo test --doc` (SC3, WAV-06) | ✓ VERIFIED | `cargo test -p fdars-core --features linalg,parallel --doc wavelet` → `wavelet::regression (line 27) ... ok`, 1 passed 0 failed 0 ignored. Doctest uses `use fdars_core::prelude::*;`, not `ignore`/`no_run`. |
| 6 | Whole-crate `cargo test` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check` green; 28 examples + R/WASM bindings unaffected; no existing public signature changed; no `Cargo.toml` change (SC4, WAV-06) | ✓ VERIFIED | `cargo test -p fdars-core --features linalg,parallel --lib` → 2794 passed 0 failed; `--doc` → 199 passed 0 failed 4 ignored; clippy → `Finished` with no warnings; `fmt --check` → empty output (exit 0). `Cargo.toml` version still `0.36.0`; no `Cargo.toml` appears in any of the 4 phase commits. |

**Score:** 6/6 truths verified (0 present-but-behavior-unverified)

---

## Deviation Assessment: wcr Affine-Intercept Re-expression

The SUMMARY documents one auto-fixed deviation: `wcr` re-expresses its stored intercept in the affine coefficient-space convention (`intercept_stored = intercept_ols − Σ col_mean_j · w_j`) after the centered recovery of `coeff_weights`. Assessment:

- **Does NOT loosen the 1e-8 self-consistency tolerance.** The test `wcr_predict_reproduces_training_fitted` asserts `(p - f).abs() <= 1e-8` — the original plan tolerance. It passes.
- **Does NOT change beta_t numerically.** Inspection of `regression.rs` lines 566–607 confirms: `fitted_values` is computed on line 554, `coeff_weights` recovered on line 566, then the intercept is re-expressed — but `beta_t = coeff_weights_to_beta_t(&coeff_weights, &layout)` on line 587 is computed from the unchanged `coeff_weights`, not from the updated intercept. Slope is numerically identical.
- **Does NOT change residuals.** `residuals = y − fitted_values` on lines 589–593 uses the `fitted_values` computed before the intercept re-expression.
- **Pre-existing recovery tests still pass.** `wcr_pcr_recovers_known_beta_t_on_spanning_design` (1e-6 relative L2 tolerance) and `wcr_pls_recovers_known_beta_t_on_spanning_design` both PASS — tolerance not loosened.

**Verdict: deviation is correct and non-regressive.**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/wavelet/regression.rs` | `predict` + accessor methods on both result structs; self-consistency + grid-mismatch tests; module doctest | ✓ VERIFIED | Substantive: inherent `predict` methods (lines 629–657 for `WcrResult`, 1255–1283 for `WnetResult`); `beta_t()`, `coefficient_function()`, `fitted_values()` accessors on both impl blocks; 5 new tests (lines 1617–2258); `//!` doctest at lines 27–54. All wired (used from tests + doctest). |
| `fdars-core/src/lib.rs` | Crate-root `pub use wavelet::...` block (14 symbols) | ✓ VERIFIED | Lines 604–608: two `pub use` lines, 14 symbols total, not feature-gated. No existing `pub use` line modified. |
| `fdars-core/src/prelude.rs` | Mirrored wavelet group after PEER group | ✓ VERIFIED | Lines 112–118: `// Wavelet-domain regression (v0.37.0)` group with exact mirror. No existing line modified. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `WcrResult::predict` | `curves_to_coeff_design` seam | calls `curves_to_coeff_design(new, self.family.clone(), self.mode, Some(self.level))` | ✓ WIRED | Source line 641; stored config passed, not reconstructed from beta_t. |
| `WnetResult::predict` | `curves_to_coeff_design` seam | calls `curves_to_coeff_design(new, self.family.clone(), self.mode, Some(self.level))` | ✓ WIRED | Source line 1267; mirrors WcrResult::predict. |
| crate-root `pub use wavelet::regression::...` | `fdars_core::wcr`, `fdars_core::WcrResult`, etc. | direct `pub use` in lib.rs line 605 | ✓ WIRED | Build clean; whole-crate test exercises the surface. |
| prelude `pub use crate::wavelet::regression::...` | `fdars_core::prelude::*` | direct `pub use crate::...` in prelude.rs line 113 | ✓ WIRED | Module doctest uses `use fdars_core::prelude::*;` and compiles + runs. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| WcrResult::predict self-consistency (≤1e-8) | `cargo test --lib -- wavelet::regression::tests::wcr_predict_reproduces_training_fitted` | 1 passed 0 failed | ✓ PASS |
| WnetResult::predict self-consistency (≤1e-8) | `cargo test --lib -- wavelet::regression::tests::wnet_predict_reproduces_training_fitted` | 1 passed 0 failed | ✓ PASS |
| Grid-mismatch → FdarError for wcr | `cargo test --lib -- wavelet::regression::tests::wcr_predict_on_new_curves_is_finite_and_rejects_grid_mismatch` | 1 passed 0 failed | ✓ PASS |
| Grid-mismatch → FdarError for wnet | `cargo test --lib -- wavelet::regression::tests::wnet_predict_on_new_curves_is_finite_and_rejects_grid_mismatch` | 1 passed 0 failed | ✓ PASS |
| Accessor correctness (both structs) | `cargo test --lib -- wavelet::regression::tests::accessors_return_stored_slices` | 1 passed 0 failed | ✓ PASS |
| Pre-existing wcr PCR recovery (1e-6 tol, unchanged) | `cargo test --lib -- wavelet::regression::tests::wcr_pcr_recovers_known_beta_t_on_spanning_design` | 1 passed 0 failed | ✓ PASS |
| Pre-existing wcr PLS recovery (1e-6 tol, unchanged) | `cargo test --lib -- wavelet::regression::tests::wcr_pls_recovers_known_beta_t_on_spanning_design` | 1 passed 0 failed | ✓ PASS |
| End-to-end module doctest via prelude | `cargo test -p fdars-core --features linalg,parallel --doc wavelet` | 1 passed 0 failed 0 ignored | ✓ PASS |
| Whole-crate lib tests | `cargo test -p fdars-core --features linalg,parallel --lib` | 2794 passed 0 failed | ✓ PASS |
| Whole-crate doc tests | `cargo test -p fdars-core --features linalg,parallel --doc` | 199 passed 0 failed 4 ignored | ✓ PASS |
| clippy --all-targets | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished with no warnings | ✓ PASS |
| fmt --check | `cargo fmt --check` | empty output (exit 0) | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| WAV-05 | 71-01-PLAN.md | Out-of-sample predict + coefficient/fitted accessors on WcrResult/WnetResult | ✓ SATISFIED | `predict` methods on both structs; 5 targeted tests pass; self-consistency ≤1e-8 verified |
| WAV-06 | 71-01-PLAN.md | Full crate-root + prelude re-exports; running end-to-end doctest | ✓ SATISFIED | 14-symbol surface at both `lib.rs` and `prelude.rs`; doctest at wavelet::regression line 27 passes |

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers in any file modified by this phase. No empty implementations, no return stubs.

### Human Verification Required

None. All success criteria verified programmatically.

---

## Summary

Phase 71 goal is **achieved**. All 6 must-have truths are VERIFIED with concrete test evidence:

1. Both `WcrResult::predict` and `WnetResult::predict` are self-consistent to ≤1e-8 on training curves, return finite predictions on new curves, and return `FdarError::InvalidDimension` on grid mismatch — confirmed by 4 dedicated tests.
2. `beta_t()`/`coefficient_function()`/`fitted_values()` accessors exist on both structs and return the stored field slices — confirmed by `accessors_return_stored_slices`.
3. All 14 wavelet surface symbols are reachable at both the crate root and prelude; the end-to-end module doctest passes under `cargo test --doc` (not `ignore`/`no_run`).
4. Whole-crate `cargo test` (2794 lib + 199 doc = 2993 total, 0 failures), `clippy --all-targets -D warnings` (clean), and `fmt --check` (clean) are all green. No `Cargo.toml` version change. No existing public signature modified.

The wcr affine-intercept deviation is confirmed non-regressive: `beta_t` and `residuals` are numerically unchanged; the pre-existing 1e-6-tolerance recovery tests pass; the new 1e-8 self-consistency test is not loosened.

---

_Verified: 2026-09-04_
_Verifier: Claude (gsd-verifier)_
