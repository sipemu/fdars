---
phase: 24-concurrent-varying-coefficient-regression
verified: 2026-08-17T10:12:08Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 24: Concurrent / Varying-Coefficient Regression — Verification Report

**Phase Goal:** Users can fit a dense functional concurrent (varying-coefficient) regression relating a functional response to one or more functional predictors sampled on the same shared grid, recovering a smooth time-varying coefficient curve β(t).
**Verified:** 2026-08-17T10:12:08Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1: Public entry point `concurrent_regression` re-exported at crate root, returns `Result<ConcurrentRegrResult, FdarError>` with fields `{ beta_curve, intercept, fitted, residuals, argvals }` | ✓ VERIFIED | `lib.rs:81` `pub mod concurrent_regression;`, `lib.rs:234` `pub use concurrent_regression::{concurrent_regression, ConcurrentRegrResult};`; struct with all 5 fields at `concurrent_regression.rs:36-47`; `test_shape_smoke` passes |
| 2 | SC2: On synthetic data from a known β(t) with low noise, recovered `beta_curve` reproduces the true coefficient curve within tolerance at interior grid points | ✓ VERIFIED | `test_recovery_known_beta` passes: true_beta=sin(πt), bw=0.15, interior j∈5..45, tol<0.15; behavioral test executed and exit 0 confirmed |
| 3 | SC3: β(t) estimated by penalized pointwise / local-linear least squares; increasing roughness penalty produces demonstrably smoother `beta_curve` (monotone roughness/curvature check) | ✓ VERIFIED | `test_monotone_roughness` passes: roughness(bw=0.05) > roughness(bw=0.15) > roughness(bw=0.35) on sin(2πt) synthetic data; behavioral test executed and exit 0 confirmed |
| 4 | SC4: `fitted` reconstructs response; `residuals == response − fitted` pointwise (<1e-10); invalid inputs return `FdarError`, no panic | ✓ VERIFIED | `test_residuals_consistency` passes (abs diff <1e-10 every cell); `test_invalid_inputs` covers 6 scenarios; `test_nan_inf_bandwidth_returns_error` and `test_underdetermined_system_returns_error` cover post-review regressions; all 9 concurrent_regression tests pass |
| 5 | SC5: No existing public signature changed; full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green; no new crate dependency | ✓ VERIFIED | Full lib suite: 2070 tests pass (0 failed); 137 doctests pass; clippy exits 0; Cargo.lock diff shows no new package entries; commit `5480ee25` touches only `concurrent_regression.rs` (new) and `lib.rs` (2 additive lines: `pub mod` + `pub use`) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/concurrent_regression.rs` | New module: `concurrent_regression` fn + `ConcurrentRegrResult` struct + inline `#[cfg(test)]` tests | ✓ VERIFIED | 871-line file; struct at lines 33-47; fn at lines 92-290; 9 inline tests at lines 296-870 |
| `fdars-core/src/lib.rs` (additive change) | `pub mod concurrent_regression;` + `pub use concurrent_regression::{concurrent_regression, ConcurrentRegrResult};` | ✓ VERIFIED | Lines 81 and 234; confirmed by `git show 5480ee25 -- fdars-core/src/lib.rs` showing only 3 additive lines (mod, comment, use), no existing lines removed |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `concurrent_regression` | `smoothing::solve_gaussian_pub` | pointwise OLS per grid column (line 226) | ✓ WIRED | `smoothing::solve_gaussian_pub(&mut xtx, &mut xty, q)` called inside `iter_maybe_parallel!(0..m)` closure |
| `concurrent_regression` | `smoothing::local_linear` | β(t) sequence smoothing; bandwidth = roughness knob (lines 249-261) | ✓ WIRED | Called twice: once for intercept (line 249) and once per predictor k (line 261) |
| `lib.rs` | `concurrent_regression::{concurrent_regression, ConcurrentRegrResult}` | `pub mod concurrent_regression;` + `pub use` re-export | ✓ WIRED | Both symbols accessible at crate root; confirmed by test_shape_smoke calling `concurrent_regression` (resolved via re-export) and returning `ConcurrentRegrResult` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `ConcurrentRegrResult.beta_curve` | `beta_curve[(k,j)]` | Computed by `local_linear` over `raw_beta[k*m+j]` from per-column OLS | Yes — flows from input `predictors` and `response` through real OLS + kernel smoothing | ✓ FLOWING |
| `ConcurrentRegrResult.intercept` | `intercept[j]` | `local_linear(&argvals, &raw_intercept, ...)` | Yes — flows from real OLS intercept coefficient per column | ✓ FLOWING |
| `ConcurrentRegrResult.fitted` | `fitted[(i,j)]` | `intercept[j] + Σk beta_curve[(k,j)] * pred[(i,j)]` (line 277) | Yes — computed from smoothed coefficients applied to real predictor values | ✓ FLOWING |
| `ConcurrentRegrResult.residuals` | `residuals[(i,j)]` | `response[(i,j)] - fitted[(i,j)]` (line 278) | Yes — direct arithmetic from response and computed fitted | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SC1 shape smoke (struct fields, re-export) | `cargo test --lib concurrent_regression::tests::test_shape_smoke -- --exact` | 1 passed, 0 failed | ✓ PASS |
| SC2 recovery known β(t) | `cargo test --lib concurrent_regression::tests::test_recovery_known_beta -- --exact` | 1 passed, 0 failed | ✓ PASS |
| SC3 monotone roughness | `cargo test --lib concurrent_regression::tests::test_monotone_roughness -- --exact` | 1 passed, 0 failed | ✓ PASS |
| SC4 residuals + invalid inputs (all 9 concurrent_regression tests) | `cargo test --lib "concurrent_regression::tests"` | 9 passed, 0 failed | ✓ PASS |
| SC5 full suite non-breaking | `cargo test -p fdars-core --features linalg,parallel` | 2070 lib + 137 doc tests passed, 0 failed | ✓ PASS |
| SC5 clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished dev profile, exit 0 | ✓ PASS |

### Probe Execution

No probes declared or applicable for this phase (pure Rust library implementation, no migration scripts).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| REG-01 | 24-01-PLAN.md | Dense functional concurrent (varying-coefficient) regression via new public entry point | ✓ SATISFIED | `concurrent_regression` and `ConcurrentRegrResult` implemented and re-exported; all 5 SC verified; REQUIREMENTS.md line 12 marked COMPLETED 2026-08-17 |

No orphaned requirements: REQUIREMENTS.md maps exactly REG-01 to Phase 24 and REG-02 to Phase 25 (pending). REG-01 is the only ID declared in 24-01-PLAN.md and the only one scoped to this phase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | — |

No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, `PLACEHOLDER`, or stub patterns found in `fdars-core/src/concurrent_regression.rs`. All fields carry real computed data (no hardcoded empty returns, no `return null`, no empty handlers).

### Human Verification Required

None. All success criteria are verifiable programmatically and all behavioral tests were executed and passed. No UI behavior, real-time interaction, or external service integration is involved.

### Gaps Summary

No gaps. All 5 ROADMAP Success Criteria are verified against the actual codebase:

- SC1: entry point exists, is properly structured, and is wired to the crate root
- SC2: behavioral recovery test executes and passes (not just symbol presence)
- SC3: behavioral monotone-roughness test executes and passes (not just symbol presence)
- SC4: residual consistency and all invalid-input guard tests execute and pass
- SC5: full 2070-test suite and clippy --all-targets gate confirmed green; commit touches exactly 2 files with only additive changes; no new Cargo dependency

---

_Verified: 2026-08-17T10:12:08Z_
_Verifier: Claude (gsd-verifier)_
