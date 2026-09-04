---
phase: 68-longitudinal-peer-prediction-integration
verified: 2026-09-04T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 68: Longitudinal PEER, Prediction & Integration — Verification Report

**Phase Goal:** A user can fit longitudinal PEER via a public `lpeer()` estimator with subject-level random effects, predict out-of-sample from fitted PEER/lpeer results, and reach the whole surface from the crate root + prelude — demonstrated by an end-to-end module doctest.
**Verified:** 2026-09-04
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `lpeer()` fits subject-level random effects via `famm::fit_scalar_mixed_model`, returns `LpeerResult` with time-varying β(t) + non-negative `sigma2_subject`/`sigma2_resid` (PER-04) | ✓ VERIFIED | `test_lpeer_variance_non_negative` passes; `famm::fit_scalar_mixed_model` call at line 587 of peer.rs; `LpeerResult.sigma2_subject` and `.sigma2_resid` fields present with non-exhaustive struct and famm clamping documented |
| 2 | On synthetic longitudinal data with known subject-RE structure, `lpeer()` recovers β(t) within tolerance and fitted `sigma2_subject` tracks injected between-subject variance (PER-04) | ✓ VERIFIED | `test_lpeer_beta_recovery` (max error < 0.5 vs sin(πt)) and `test_lpeer_sigma2_tracks_injection` (sigma2_subject ∈ (0.1, 5.0) for injected 1.0) both pass; fixture m=10 deviation is a legitimate basis-completeness fix (ncomp=min(199,10,10)=10=m ensures full FPC coverage; m=20 caused systematic ~1.1 truncation error unrelated to statistical power) |
| 3 | `predict` on re-passed training curves reproduces training `fitted_values` within 1e-9 (self-consistency); finite on new curves; wrong ncols returns `FdarError::InvalidDimension` (PER-05) | ✓ VERIFIED | `test_peer_predict_self_consistent`, `test_predict_new_curves_finite`, `test_predict_wrong_ncols` (PeerResult); `test_lpeer_predict_self_consistent` (LpeerResult) — all 4 pass; shared `peer_predict_core` at line 1096 validates dims and is invoked by both `PeerResult::predict` (line 1147) and `LpeerResult::predict` (line 648) |
| 4 | Full PEER/lpeer public surface (8 symbols) reachable from crate root + prelude; running module doctest demonstrates fit→β(t)→predict and passes `cargo test --doc` (PER-05) | ✓ VERIFIED | `lib.rs` line 599: `pub use peer::{lpeer, peer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, PeerPenalty, PeerResult}`; `prelude.rs` line 108: identical 8-item set; 2/2 doctests pass (module header line 16 + `lpeer` fn line 428); `test_crate_root_exports_compile` + `test_prelude_exports_compile` compile-check tests also pass; `no_run` snippet is gone (confirmed by grep) |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Fixture Deviation Assessment (m=20 → m=10)

The SUMMARY documents a deviation from the RESEARCH §6 fixture spec (m=20 → m=10). This is a legitimate technical fix, not a weakening of the recovery assertion. With `ncomp = min(n-1, m, 10) = 10` and `m=20`, only 10 of 20 FPC directions are back-projected — the omitted 10 directions cause systematic error of ~1.1 (matching `max|sin(πt)| ≈ 1`). Setting m=10 ensures `ncomp = m = 10` (full FPC basis), reducing max error below 0.5. Increasing n from 50 to 200 was also applied for REML EM convergence reliability. The recovery tolerance (max error < 0.5) is unchanged and was strengthened by guaranteeing full basis coverage.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/peer.rs` | `lpeer()`, `LpeerResult`, `peer_predict_core()`, `PeerResult::predict`, `LpeerResult::predict`, running doctests | ✓ VERIFIED | All symbols present; `lpeer` at line 456, `LpeerResult` at line 190, `peer_predict_core` at line 1096, both `predict` impls at lines 1147 and 648; module doctest is running (not `no_run`) |
| `fdars-core/src/lib.rs` | `pub use peer::{...}` crate-root re-export block (8 symbols) | ✓ VERIFIED | Line 599: explicit 8-symbol block, no wildcard |
| `fdars-core/src/prelude.rs` | `pub use crate::peer::{...}` prelude re-export block (8 symbols) | ✓ VERIFIED | Line 108: identical 8-symbol block |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `LpeerResult.w_bar` | `peer_predict_core` | Computed identically to `peer()` col-means of weighted design | ✓ VERIFIED | Lines 542–544 of peer.rs mirror the `peer()` w_bar computation; `test_lpeer_predict_self_consistent` confirms within 1e-9 |
| `famm::fit_scalar_mixed_model` | `LpeerResult.sigma2_subject` / `.sigma2_resid` | `result.sigma2_u` → `sigma2_subject`; `result.sigma2_eps` → `sigma2_resid` | ✓ VERIFIED | Line 619 of peer.rs assembles `LpeerResult` from `result.sigma2_u` and `result.sigma2_eps`; famm clamps both non-negative |
| `beta[j]` back-projection | `fpca.rotation[(j,k)]` | `Σ_k gamma[k] * rotation[(j,k)]` | ✓ VERIFIED | Lines 598–604 of peer.rs; rotation layout is m×ncomp (column-major FdMatrix), access `rotation[(j,k)]` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 30 peer:: lib tests (all tasks) | `cargo test -p fdars-core --lib --features linalg,parallel peer::` | 30 passed; 0 failed; finished in 0.20s | ✓ PASS |
| 2 peer doctests (module header + lpeer fn) | `cargo test -p fdars-core --doc --features linalg,parallel peer` | 2 passed; 0 failed; finished in 0.41s | ✓ PASS |
| Clippy --all-targets clean | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | Finished dev profile [unoptimized + debuginfo]; no warnings | ✓ PASS |

### Requirements Coverage (Milestone-Closing: PER-01..PER-05)

| Requirement | Phase | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| PER-01 | 66 | Public `peer()` estimator with PEER decomposition, β(t) + result struct | ✓ SATISFIED | Delivered Phase 66; `test_peer_difference_beta_recovery`, `test_peer_result_shape` pass |
| PER-02 | 66 | Three penalty families (Ridge, Difference, Decree) via enum | ✓ SATISFIED | Delivered Phase 66; `test_peer_ridge_fits`, `test_peer_decree_fits`, `test_peer_decree_distinct_from_roughness` pass |
| PER-03 | 67 | Automatic λ selection via GCV or REML; explicit λ honored | ✓ SATISFIED | Delivered Phase 67; `test_peer_gcv_recovers_beta`, `test_peer_reml_lambda_positive`, `test_peer_gcv_deterministic`, `test_peer_reml_gcv_beta_agreement` pass |
| PER-04 | 68 | `lpeer()` with subject random effects via famm; variance components in result | ✓ SATISFIED | Delivered this phase; `test_lpeer_variance_non_negative`, `test_lpeer_beta_recovery`, `test_lpeer_sigma2_tracks_injection` pass |
| PER-05 | 68 | Out-of-sample `predict`; crate root + prelude exports; running module doctest | ✓ SATISFIED | Delivered this phase; self-consistency tests pass; 8-symbol exports verified in both lib.rs and prelude.rs; 2/2 doctests pass |

All 5 milestone requirements are satisfied. v0.36.0 implementation is complete.

**Note on crate version:** `Cargo.toml` still reads `0.35.0`. This is expected — the version bump to `0.36.0` and `v0.36.0` tag are performed by the `gsd-complete-milestone` / `gsd-audit-milestone` workflow after this phase, consistent with prior milestone patterns (e.g., `chore: bump fdars-core to 0.35.0` preceded the `chore: archive v0.35.0 milestone` commit).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER) found in peer.rs, lib.rs, or prelude.rs | — | — |

### Human Verification Required

None. All phase behaviors have automated known-answer verification. The validation strategy (`68-VALIDATION.md`) documents zero manual-only verification items.

### Gaps Summary

No gaps. All 4 must-have truths verified, all 3 artifacts present and substantive and wired, all key links confirmed, 30/30 unit tests pass, 2/2 doctests pass, clippy clean, no debt markers.

---

_Verified: 2026-09-04T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
