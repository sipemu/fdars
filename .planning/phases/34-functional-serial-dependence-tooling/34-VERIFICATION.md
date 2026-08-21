---
phase: 34-functional-serial-dependence-tooling
verified: 2026-08-21T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 34: Functional Serial Dependence Tooling Verification Report

**Phase Goal:** Add L2-norm functional ACF/PACF with strong-white-noise confidence bands, a functional stationarity test, a long-run-covariance kernel-sandwich estimator, and a functional differencing operator in a new `fdars-core/src/fts/acf.rs`, reusing `helpers` quadrature + `covariance.rs`, without any existing code changing. Numeric outputs only, additive/non-breaking, no new crate dependency.
**Verified:** 2026-08-21
**Status:** PASSED
**Re-verification:** No — initial verification

## FTS-02 Requirement Traceability

FTS-02 appears in `REQUIREMENTS.md` under milestone v0.25.0, mapped to Phase 34 with status "Pending".
The requirement text matches the phase goal exactly.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Five Result-returning crate-root entry points exist (functional_acf, functional_pacf, stationarity_test, long_run_covariance, functional_difference) consuming a column-major FdMatrix | ✓ VERIFIED | `pub use fts:{ functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test, FacfResult, LongRunCovResult, StationarityResult }` at lib.rs:235-238 |
| 2 | fACF/fPACF return L2-norm autocorrelation + strong-white-noise bands; seeded tests: white-noise inside bands, injected lag-1 dependence exceeds band | ✓ VERIFIED | Tests `facf_whitenoise_inside_band` and `facf_ar1_exceeds_band` both pass (24/24 fts tests green); MC χ²-mixture band via eigendecomposition of C_0 (acf.rs:319-364) |
| 3 | functional_difference produces N-1 series that round-trips vs cumulative sum; stationarity_test rejects trended, accepts stationary | ✓ VERIFIED | Tests `diff_roundtrip` (tolerance 1e-10), `stat_test_nonstationary` (p ≤ 0.05 on i*t trend), `stat_test_stationary` (p > 0.05 on i.i.d. GP) all pass |
| 4 | long_run_covariance returns symmetric operator reducing to lag-0 covariance at bandwidth 0; reuses autocovariance helper; FdarError on invalid input; no new crate dependency | ✓ VERIFIED | Tests `lrc_bandwidth_zero` (element-wise within 1e-12), `lrc_symmetric` (within 1e-10), `lrc_default_bandwidth` pass; `autocovariance_matrix` called 10 times in acf.rs; `git diff --exit-code fdars-core/Cargo.toml` exits 0 |
| 5 | Existing public signatures unchanged (additive/non-breaking); full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green | ✓ VERIFIED | `git diff HEAD~5 --name-only` shows only `fts/acf.rs`, `fts/mod.rs`, `lib.rs`, and `.planning/` files changed; clippy exits 0 (zero warnings); 24/24 fts tests green |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/fts/mod.rs` | FTS module barrel with all three result structs | ✓ VERIFIED | Exists, 76 lines; declares `FacfResult`, `StationarityResult`, `LongRunCovResult` with required derives (`Debug, Clone, PartialEq`, `#[non_exhaustive]`, serde-gated); `pub use acf::{...}` re-exports all 5 entry points |
| `fdars-core/src/fts/acf.rs` | All 5 entry points + helpers + 24 inline tests | ✓ VERIFIED | Exists, 1283 lines; all 5 public functions present and substantive; 24 tests present and all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fdars-core/src/lib.rs` | `fts::{functional_acf, functional_pacf, stationarity_test, long_run_covariance, functional_difference, FacfResult, StationarityResult, LongRunCovResult}` | `pub mod fts;` (lib.rs:90) + `pub use fts::{...}` (lib.rs:235-238) | ✓ WIRED | All 5 functions and all 3 result structs re-exported at crate root |
| `fts/acf.rs` `autocovariance_matrix` | Shared by `functional_acf` and `long_run_covariance` | `pub(crate) fn autocovariance_matrix(...)` defined once, called at lines 305, 312, 692, 711 | ✓ WIRED | Helper is the single spine; `grep` count = 10 (1 define + multiple call sites); confirms reuse-first requirement |
| `fts/acf.rs` | `helpers::{simpsons_weights, trapz, NUMERICAL_EPS}` | `use crate::helpers::{simpsons_weights, trapz, NUMERICAL_EPS}` (acf.rs:15) | ✓ WIRED | Reuses existing quadrature helpers as specified |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `functional_acf` → `FacfResult.acf` | `acf_vals` | `hs_norm_sq(autocovariance_matrix(...))` computed from input `FdMatrix` | Yes — from real column-major row access | ✓ FLOWING |
| `functional_acf` → `FacfResult.upper_band` | `band` | MC χ²-mixture via `nalgebra::SymmetricEigen` of C_0, then `mc_band_threshold` | Yes — from eigenvalues of real data covariance | ✓ FLOWING |
| `functional_acf` → `FacfResult.pacf` | `pacf` | `durbin_levinson_pacf(&acf_vals)` — scalar recursion over the real fACF sequence | Yes — derived from real data | ✓ FLOWING |
| `stationarity_test` → `StationarityResult` | `observed_t`, `p_value` | KPSS partial-sum + seeded Fisher-Yates permutation p-value over real data rows | Yes | ✓ FLOWING |
| `long_run_covariance` → `LongRunCovResult.cov_matrix` | `acc` | Bartlett-weighted accumulation of `autocovariance_matrix` for h=0..bandwidth | Yes | ✓ FLOWING |
| `functional_difference` → `FdMatrix` | `out[(i,j)]` | `data[(i+1,j)] - data[(i,j)]` — direct FdMatrix read | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 24 fts:: inline tests pass | `cargo test -p fdars-core --features linalg,parallel "fts::"` | 24 passed, 0 failed | ✓ PASS |
| White-noise inside band (SC2, seeded) | `facf_whitenoise_inside_band` test (n=80, seed=7, n_sim=1000) | ok | ✓ PASS |
| AR(1) lag-1 exceeds band (SC2, seeded) | `facf_ar1_exceeds_band` test (n=120, seed=13) | ok | ✓ PASS |
| Differencing round-trips within 1e-10 (SC3) | `diff_roundtrip` test | ok | ✓ PASS |
| Stationarity test rejects trended series (SC3) | `stat_test_nonstationary` test | ok | ✓ PASS |
| Stationarity test does not reject stationary series (SC3) | `stat_test_stationary` test | ok | ✓ PASS |
| LRC at bandwidth=0 equals C_0 within 1e-12 (SC4) | `lrc_bandwidth_zero` test | ok | ✓ PASS |
| LRC is symmetric within 1e-10 (SC4) | `lrc_symmetric` test | ok | ✓ PASS |
| Clippy gate (SC5) | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished with 0 warnings | ✓ PASS |
| No new crate dependency (SC4/SC5) | `git diff --exit-code fdars-core/Cargo.toml` | exit code 0 — clean | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FTS-02 | 34-01-PLAN.md, 34-02-PLAN.md, 34-03-PLAN.md | L2-norm functional ACF/PACF + white-noise bands, stationarity test, LRC estimator, differencing operator in `fts/acf.rs` | ✓ SATISFIED | All 5 entry points exist, are crate-root re-exported, and pass all seeded behavioral tests; no new crate dependency; additive only |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | — |

No `TBD`, `FIXME`, `XXX`, `TODO`, `PLACEHOLDER`, or stub patterns detected in `fdars-core/src/fts/acf.rs` or `fdars-core/src/fts/mod.rs`. No `return null`, `return {}`, or empty implementation patterns. No `unwrap()` calls in non-test production code.

### Human Verification Required

None. All success criteria are verifiable programmatically. Tests are seeded and deterministic. The stationarity test divergence from the HKR 2014 normalized statistic is explicitly documented in the rustdoc as a known "DIVERGENCE / ASSUMED" note — the permutation p-value is explicitly stated to be valid regardless of normalization.

### Gaps Summary

No gaps found. Phase goal is fully achieved.

---
_Verified: 2026-08-21_
_Verifier: Claude (gsd-verifier)_
