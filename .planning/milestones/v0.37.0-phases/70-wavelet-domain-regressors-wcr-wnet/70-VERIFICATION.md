---
phase: 70-wavelet-domain-regressors-wcr-wnet
verified: 2026-09-04T20:45:00Z
status: passed
score: 7/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 70: Wavelet-Domain Regressors (`wcr` + `wnet`) Verification Report

**Phase Goal:** Users can fit two wavelet-domain scalar-on-function regressors on Gaussian responses — `wcr` (PCR/PLS in wavelet-coefficient space) and `wnet` (elastic-net with cross-validated λ) — each transforming curves to wavelet coefficients via the Phase 69 DWT, then reusing fdars' existing FPCR/PLS and coordinate-descent machinery.
**Verified:** 2026-09-04
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `wcr` fits via PCR on wavelet coefficients and recovers a known β(t) on spanning full-rank synthetic data within tolerance; result carries β(t), intercept, fitted values | ✓ VERIFIED | `wcr_pcr_recovers_known_beta_t_on_spanning_design` PASSES — n=120, m=32 LCG spanning design, ncomp=min(n,P), rel L2 < 1e-6; `WcrResult` carries `beta_t`, `intercept`, `fitted_values` |
| 2 | `wcr` fits via PLS on wavelet coefficients and recovers the same known β(t) within tolerance | ✓ VERIFIED | `wcr_pls_recovers_known_beta_t_on_spanning_design` PASSES — same spanning design, same 1e-6 tolerance; PLS branch separately reachable via `WcrMethod::Pls` |
| 3 | `wnet` recovers a sparse coefficient pattern — selected/nonzero coeffs concentrate on true support; result carries β(t), intercept, fitted values, selected coefficients | ✓ VERIFIED | `wnet_elastic_net_cd_recovers_sparse_support` PASSES — all true-support indices selected; `selected.len() < P/2`; `WnetResult` carries `selected`, `coeff_weights`, `beta_t`, `intercept`, `fitted_values` |
| 4 | `wnet` CV-λ is deterministic across runs and at selected λ yields a non-degenerate fit tracking the injected signal | ✓ VERIFIED | `wnet_cv_lambda_is_deterministic_across_runs` PASSES — `fit1.lambda == fit2.lambda` and `l1 == l2` via direct helper; `wnet_recovers_beta_t_on_snr_data` PASSES — 300×32 spanning design, ~20:1 SNR, nonzero fit, rel L2 < 0.35 |
| 5 | Both regressors validate inputs → descriptive FdarError (never panic); finite/NaN-free β(t) + fitted values | ✓ VERIFIED | 13 rejection tests pass: wcr rejects too-few-rows, mismatched-y-len, zero-ncomp, unsupported-family, level-out-of-range; wnet rejects too-few-rows, zero-cols, mismatched-y-len, alpha-out-of-range, too-few-folds, empty-lambda-grid, unsupported-family, level-out-of-range; finite-output tests pass for both (`wcr_finite_outputs_both_methods`, `wnet_finite_outputs_on_larger_snr_design`) |
| 6 | NEW per-coefficient elastic-net CD adapter (NOT a call to `additive.rs` `variable_selection`) | ✓ VERIFIED | `grep variable_selection regression.rs` returns empty; `elastic_net_cd` is a new function (lines 697–798) with its own cyclic coordinate-descent loop, soft-threshold, and ridge denominator — modeled on the pattern from `additive.rs` but independently implemented |
| 7 | NO crate-root or prelude re-exports of wcr/wnet/wavelet symbols (deferred to Phase 71) | ✓ VERIFIED | `grep wcr\|wnet lib.rs` empty; `grep wcr\|wnet prelude.rs` empty; `pub mod wavelet;` exists in `lib.rs` (Phase 69) but no `pub use wavelet::regression::...` anywhere |

**Score:** 7/7 truths verified (0 present, behavior-unverified)

### Deviation Assessment (SC1 — direct beta-coeff recovery)

The plan specified recovering β_coeff via the score-projection method (`Σ_k coeff_k * rotation/weight_k`). The executor instead used `recover_coeff_weights()` — regressing the centered fitted contribution back onto the centered coefficient design via normal equations. This deviation **satisfies SC1 more correctly** than the plan's suggested approach: the plan's projection recovers β in each method's internal integration-weighted inner product (sqrt-weighted for PCR's SVD, int-weighted for PLS's NIPALS), which does not equal the plain functional dot product β(t) must act by against the concatenated-coefficient design. The direct regression approach yields the exact plain-dot β_coeff independently of the method, achieving the sub-1e-6 relative L2 recovery that SC1 requires. The deviation is mathematically sound and positively auditable.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/wavelet/regression.rs` | New submodule with WcrConfig, WcrMethod, WcrResult, wcr(), CoeffLayout, curves_to_coeff_design(), coeff_weights_to_beta_t() | ✓ VERIFIED | File exists; 1746 lines; all symbols present and substantive |
| `fdars-core/src/wavelet/mod.rs` | Adds `pub mod regression;` | ✓ VERIFIED | Line 37: `pub mod regression;` confirmed |
| `fdars-core/src/wavelet/regression.rs` | Adds WnetConfig, WnetResult, wnet(), elastic_net_cd(), wnet_cv_lambda() | ✓ VERIFIED | All symbols implemented at lines 573–1066; 27 inline tests |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `curves_to_coeff_design()` | `decompose_matrix` | calls `decompose_matrix(data, family, mode, level)` at line 177 | ✓ WIRED | Fully wired; per-curve coefficient rows assembled into n×P FdMatrix |
| `coeff_weights_to_beta_t()` | `reconstruct` | constructs `WaveletCoeffs` shell then calls `reconstruct(&coeffs)` at line 262 | ✓ WIRED | Fully wired; round-trip test confirms ≤1e-10 rel L2 |
| `wcr()` → `fdata_to_pc_1d` (PCR path) | PCR fit on coefficient design | line 506 | ✓ WIRED | Uses uniform 0..P argvals grid (abstract basis, documented) |
| `wcr()` → `fdata_to_pls_1d` (PLS path) | PLS fit on coefficient design | line 511 | ✓ WIRED | Reuses existing PLS from `scalar_on_function/pls.rs` |
| `wnet()` → `curves_to_coeff_design` | Shared curves→coeff seam | line 1021 | ✓ WIRED | Reuses Plan 01's shared seam verbatim |
| `wnet()` → `wnet_cv_lambda` | CV-λ selection | line 1024 | ✓ WIRED | Deterministic via `crate::cv::create_folds(n, n_folds, seed)` |
| `wnet()` → `elastic_net_cd` | Coordinate descent at selected λ | line 1025 | ✓ WIRED | New per-coefficient CD adapter (NOT variable_selection) |
| `wnet()` → `coeff_weights_to_beta_t` | Shared inverse-DWT β(t) seam | line 1051 | ✓ WIRED | Reuses Plan 01's shared seam verbatim |
| `wnet_cv_lambda` → `crate::cv::create_folds` | Deterministic fold partition | line 891 | ✓ WIRED | Fixed `config.seed` ensures determinism across runs |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `WcrResult.beta_t` | Inverse DWT of `coeff_weights` | `recover_coeff_weights` → `coeff_weights_to_beta_t` → `reconstruct` | Yes — traced through normal equations on actual fitted values | ✓ FLOWING |
| `WcrResult.fitted_values` | `compute_fitted(&ols_design, &coeffs)` | PCR/PLS scores + OLS solve | Yes | ✓ FLOWING |
| `WnetResult.coeff_weights` | `elastic_net_cd` output `beta` vector | CD solver on centered coefficient design | Yes — sparse, L1-thresholded | ✓ FLOWING |
| `WnetResult.selected` | `filter(|b| b != 0.0)` over `coeff_weights` | Derived from CD output | Yes | ✓ FLOWING |
| `WnetResult.lambda` | `wnet_cv_lambda` return value | K-fold CV-MSE minimization | Yes — not a static fallback | ✓ FLOWING |
| `WnetResult.beta_t` | `coeff_weights_to_beta_t(&coeff_weights, &layout)` | Inverse DWT of sparse `coeff_weights` | Yes | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| wcr PCR β(t) recovery < 1e-6 rel L2 | `cargo test ... wcr_pcr_recovers_known_beta_t_on_spanning_design` | 1 passed, 0 failed | ✓ PASS |
| wcr PLS β(t) recovery < 1e-6 rel L2 | `cargo test ... wcr_pls_recovers_known_beta_t_on_spanning_design` | 1 passed, 0 failed | ✓ PASS |
| wnet sparse support recovery | `cargo test ... wnet_elastic_net_cd_recovers_sparse_support` | 1 passed, 0 failed | ✓ PASS |
| wnet CV-λ determinism | `cargo test ... wnet_cv_lambda_is_deterministic_across_runs` | 1 passed, 0 failed | ✓ PASS |
| wnet SNR β(t) recovery | `cargo test ... wnet_recovers_beta_t_on_snr_data` | 1 passed, 0 failed | ✓ PASS |
| All wavelet lib tests | `cargo test -p fdars-core --features linalg,parallel --lib wavelet` | 74 passed, 0 failed | ✓ PASS |
| Clippy `--all-targets` | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Clean (no output) | ✓ PASS |
| Fmt check | `cargo fmt --check` | No diff | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| WAV-03 | 70-01-PLAN.md | `wcr` wavelet-domain PCR+PLS scalar-on-function regressor | ✓ SATISFIED | `wcr()` implemented; both PCR and PLS paths recover β(t) < 1e-6 rel L2 on spanning full-rank design |
| WAV-04 | 70-02-PLAN.md | `wnet` wavelet-domain elastic-net with deterministic CV-λ | ✓ SATISFIED | `wnet()` implemented with per-coefficient CD, K-fold CV, sparse support recovery, finite β(t) |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | No TBD/FIXME/XXX markers; no stub returns; no hardcoded empty data in production paths |

---

### Human Verification Required

None. All truths are verifiable programmatically and all behavioral tests passed.

---

### Gaps Summary

No gaps. All 7 must-have truths are VERIFIED:

- Both PCR and PLS paths for `wcr` are separately implemented, reachable, and covered by dedicated β(t)-recovery tests on spanning LCG designs (not phase-shifted sinusoids), with tolerance NOT loosened (< 1e-6 rel L2).
- The documented deviation from the plan (direct regression vs score-projection for β_coeff recovery) is a correctness improvement, not a scope reduction — it achieves tighter recovery and is thoroughly documented in the code and summaries.
- `wnet` sparse-support recovery, CV-λ determinism, and SNR β(t)-tracking are all individually tested and green.
- A NEW per-coefficient elastic-net CD adapter (`elastic_net_cd`) was written, NOT a call to `variable_selection`.
- No crate-root or prelude re-exports added (correctly deferred to Phase 71).
- Clippy (`--all-targets`) and `cargo fmt --check` are clean.

---

_Verified: 2026-09-04T20:45:00Z_
_Verifier: Claude (gsd-verifier)_
