---
phase: 26-pace-sparse-fpca
verified: 2026-08-19
status: passed
score: 5/5
verifier: orchestrator (filesystem-fallback — spawned gsd-verifier stalled on the slow full-suite run; verdict backed by independently-run test + gate evidence)
---

# Phase 26 — PACE Sparse FPCA — Verification

**Status:** ✅ PASSED — 5/5 must-haves verified against the codebase.

## Requirement Coverage

| Requirement | Plan | Status | Evidence |
|-------------|------|--------|----------|
| FPCA-01 | 26-01 | ✓ SATISFIED | `pace_fpca` + `PaceFpcaConfig` + `PaceFpcaResult` implemented in `fdars-core/src/pace_fpca.rs`, crate-root re-exported; all 5 SC verified; REQUIREMENTS.md marks FPCA-01 complete |

## Success Criteria

**SC1 — Entry point + result shape:** `pace_fpca(&IrregFdata, &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError>` in `pace_fpca.rs`, re-exported at the crate root (`lib.rs`, 2 references). `PaceFpcaResult` carries mean, eigenvalues, eigenfunctions, conditional-expectation scores, fitted trajectories, and `fitted_lower`/`fitted_upper` bands. Verified by `test_pace_shape_smoke`, `test_crate_root_reexport` (pass).

**SC2 — Recovery from known model:** `test_pace_synthetic_recovery` (√2 sin/cos eigenfunctions, known λ, Gaussian scores, per-curve 3–8 random points, σ² noise → eigenfunction/score recovery within tolerance) and `test_blup_scores_known` pass.

**SC3 — Reuse-only, no new dependency:** built by orchestrating `irreg_fdata::mean_irreg`/`cov_irreg` + nalgebra symmetric eigendecomposition + `helpers::linear_interp` + `linalg::cholesky_solve`; band z-quantile is a local helper (statrs NOT used). `Cargo.toml` diff over the milestone is the version line only — no new dependency.

**SC4 — Error guards + no panic + fitted within bands:** invalid inputs return `FdarError` without panicking — verified by `test_empty_data`, `test_too_few_points`, `test_one_point_curve_rejected` (n_i≥2 guard), `test_zero_ncomp`, `test_invalid_bandwidth`, `test_invalid_sigma2`, `test_invalid_alpha`, `test_short_work_grid`, and `test_narrow_bandwidth_returns_err_not_nan` (the code-review CR-01 NaN-mean guard). `test_fitted_within_bands` confirms `fitted ∈ [fitted_lower, fitted_upper]`.

**SC5 — Additive/non-breaking + full gate green:** existing FPCA APIs unchanged (only additive `pub mod`/`pub use` lines in `lib.rs`); full lib suite **2096 tests pass**, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits **0**.

## Code Review

1 blocker (NaN-mean → all-NaN result) + 4 warnings found and **all fixed** with regression tests (single-ridge factorization shared by BLUP + band solves; band-solve error propagation; n_i≥2 guard; NaN-mean guard). Re-gated green.

## Verdict

All 5 success criteria verified with concrete passing tests + a green full suite and clippy gate. Phase goal achieved.
