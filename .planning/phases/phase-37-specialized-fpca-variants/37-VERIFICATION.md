---
phase: 37-specialized-fpca-variants
verified: 2026-08-21T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 37: Specialized FPCA Variants Verification Report

**Phase Goal:** A user can run the specialized FPCA variants that `fdapace`/`refund` expose but fdars was missing — FPCA of curve derivatives (`fpca_der`), a functional SVD / cross-FPCA between two functional samples (`fsvd`), a cross-covariance surface between two samples (`cross_covariance`), a dynamical/functional correlation scalar (`dynamical_correlation`), and a sandwich-smoother / sparse-SVD (ssvd) FPCA path.
**Verified:** 2026-08-21T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All five entry points exist, are Result-returning, consume column-major FdMatrix, and are crate-root re-exported | VERIFIED | `pub mod fpca_variants;` at lib.rs:91; `pub use fpca_variants::{cross_covariance, dynamical_correlation, fpca_der, fsvd, ssvd, FsvdResult}` at lib.rs:439-441; all five functions return `Result<_, FdarError>` in fpca_variants.rs |
| 2 | `fpca_der` returns derivative loadings/scores of the differentiated process; leading component reconstructs a known mode of variation; `nderiv=0` equals `fdata_to_pc_1d` | VERIFIED | `test_fpca_der` (mode-of-variation, relative reconstruction err < 1e-6) and `test_fpca_der_nderiv0` (sv/rotation equality within 1e-12) both pass; `test_fpca_der_errors` green |
| 3 | `fsvd` recovers a known rank-1 cross-covariance; `cross_covariance` agrees with hand-computed reference and with `functional_covariance` on the self-case; `dynamical_correlation` is 1 for x==x, -1 for x==-x, in [-1,1] for random | VERIFIED | `test_fsvd_rank1` (cross-cov reconstruction rel err < 1e-6), `test_cross_cov_hand_computed` (within 1e-12), `test_cross_cov_self` (within 1e-12), `test_dyncorr_identical` (within 1e-10), `test_dyncorr_negated` (within 1e-10), `test_dyncorr_range` all pass |
| 4 | `ssvd` matches `fdata_to_pc_1d` in the dense/near-zero-bandwidth limit within ~1e-4; all variants reuse `fdata_to_pc_1d` + covariance/fdata helpers; no new crate dependency; invalid inputs return `FdarError` | VERIFIED | `test_ssvd_dense_limit` (relative tolerance 1e-4) passes; `fdars-core/Cargo.toml` has zero diff vs HEAD~4; `test_ssvd_errors`, `test_fsvd_errors`, `test_dyncorr_errors`, `test_cross_cov_errors`, `test_fpca_der_errors` all pass |
| 5 | Existing public signatures unchanged (additive/non-breaking); full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green | VERIFIED | `git diff HEAD~4 HEAD` shows only additions (new `fpca_variants.rs` + 5 lines added to lib.rs); regression.rs and all other existing modules untouched; clippy clean; 2403 lib + 751 integration/doc tests, 0 failed |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/fpca_variants.rs` | New module with all five functions + FsvdResult | VERIFIED | 1193-line substantive implementation; no stubs; all functions present and wired |
| `pub mod fpca_variants;` in `fdars-core/src/lib.rs` | Module declaration | VERIFIED | lib.rs line 91 |
| `pub use fpca_variants::{cross_covariance, dynamical_correlation, fpca_der, fsvd, ssvd, FsvdResult};` | Crate-root re-export of all six symbols | VERIFIED | lib.rs lines 438-441 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fpca_der` | `fdata::deriv_1d` + `regression::fdata_to_pc_1d` | Call after input validation | VERIFIED | fpca_variants.rs:231-232: `fdata::deriv_1d` then `fdata_to_pc_1d` |
| `cross_covariance` | `fdata::center_1d` | Per-grid centering of X and Y separately | VERIFIED | fpca_variants.rs:126-127 |
| `fsvd` | `cross_covariance` (Plan 01) | Calls `cross_covariance(x, y)?` then Gram-matrix SVD | VERIFIED | fpca_variants.rs:451 |
| `ssvd` | `fdata::functional_covariance` + `helpers::simpsons_weights` | Empirical cov + sandwich eigendecompose | VERIFIED | fpca_variants.rs:719, 729 |
| `FsvdResult` (crate root) | `fpca_variants::FsvdResult` | `pub use` re-export | VERIFIED | smoke_reexports test exercises `crate::FsvdResult` type annotation |

### Data-Flow Trace (Level 4)

All five entry points compute from the caller-supplied `FdMatrix` data — no static returns or hardcoded fallbacks in production paths:

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `cross_covariance` | `cov` (p×q FdMatrix) | Centered FdMatrix inputs via `fdata::center_1d` | Yes | FLOWING |
| `fpca_der` | `FpcaResult` | `fdata::deriv_1d` + `fdata_to_pc_1d` on caller data | Yes | FLOWING |
| `dynamical_correlation` | `f64` scalar | Caller X/Y FdMatrix after 4-step Dubin-Muller centering | Yes | FLOWING |
| `fsvd` | `FsvdResult` | `cross_covariance` result → Gram SVD → scores from caller data | Yes | FLOWING |
| `ssvd` | `FpcaResult` | `functional_covariance` → Gaussian-smoothed → sandwich eigen | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 18 fpca_variants inline tests pass | `cargo test -p fdars-core --features linalg,parallel -- fpca_variants` | 18 passed; 0 failed | PASS |
| All 5 fpca_variants doc-tests pass | Included in above run | 5 passed; 0 failed | PASS |
| Full crate suite green | `cargo test -p fdars-core --features linalg,parallel` | 2403 lib + 751 integration/doc tests, 0 failed | PASS |
| Clippy clean | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished with no warnings | PASS |

Named test verification (per criterion):

| Test | Exercises | Status |
|------|-----------|--------|
| `test_cross_cov_shape` | cross_covariance p×q output shape | PASS |
| `test_cross_cov_self` | cross_covariance == functional_covariance on self-case | PASS |
| `test_cross_cov_hand_computed` | hand-computed reference (1e-12) | PASS |
| `test_cross_cov_errors` | mismatched n, n<2, zero cols | PASS |
| `test_fpca_der_nderiv0` | nderiv=0 equals fdata_to_pc_1d (1e-12) | PASS |
| `test_fpca_der` | mode-of-variation reconstruction (rel err < 1e-6) | PASS |
| `test_fpca_der_errors` | empty matrix, argvals mismatch, ncomp<1, m<2 with nderiv>0 | PASS |
| `test_dyncorr_identical` | dyncorr(x,x) = 1 within 1e-10 | PASS |
| `test_dyncorr_negated` | dyncorr(x,-x) = -1 within 1e-10 | PASS |
| `test_dyncorr_range` | result in [-1,1] for random data | PASS |
| `test_dyncorr_errors` | mismatched n/cols/argvals, n<2 | PASS |
| `test_fsvd_unit_norm` | left/right functions unit functional L2 norm (1e-8) | PASS |
| `test_fsvd_rank1` | rank-1 cross-cov reconstruction (rel err < 1e-6) | PASS |
| `test_fsvd_errors` | mismatched sample size, ncomp<1, argvals mismatch | PASS |
| `test_ssvd_dense_limit` | ssvd vs fdata_to_pc_1d (relative 1e-4) | PASS |
| `test_ssvd_orthonormality` | W-orthonormal eigenfunctions (1e-6) | PASS |
| `test_ssvd_errors` | ncomp<1, empty matrix, argvals mismatch, negative bandwidth | PASS |
| `smoke_reexports` | all five variants + FsvdResult reachable from crate root | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| FPCA-02-01 | 37-01-PLAN.md | `fpca_der` — FPCA of curve derivatives | SATISFIED | `fpca_der` implemented, 3 tests green |
| FPCA-02-02 | 37-02-PLAN.md | `fsvd` — functional SVD / cross-FPCA | SATISFIED | `fsvd` + FsvdResult implemented, 3 tests green |
| FPCA-02-03 | 37-01-PLAN.md | `cross_covariance` — cross-covariance surface | SATISFIED | `cross_covariance` implemented, 4 tests green |
| FPCA-02-04 | 37-02-PLAN.md | `dynamical_correlation` — scalar in [-1,1] | SATISFIED | `dynamical_correlation` implemented, 4 tests green |
| FPCA-02-05 | 37-02-PLAN.md | `ssvd` — sandwich-smoother FPCA; crate-root re-exports | SATISFIED | `ssvd` implemented, 3 tests + smoke_reexports green |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | No debt markers (TBD/FIXME/XXX), no stubs, no placeholder returns found in fpca_variants.rs |

No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, or `PLACEHOLDER` markers present in `fpca_variants.rs`. No empty implementations (`return null/{}`) or hardcoded-data anti-patterns found.

### Additional Verification Notes

**Git diff additive-only confirmation:** `git diff HEAD~4 HEAD --name-only` shows exactly four files changed: the two SUMMARY.md planning docs plus `fdars-core/src/fpca_variants.rs` (new) and `fdars-core/src/lib.rs` (five lines added). `fdars-core/Cargo.toml` is entirely absent from the diff — zero new crate dependencies added.

**Deviation from plan (accepted):** Plan 37-02 specified using `nalgebra::SVD::new(Cw.to_dmatrix(), true, true)` for `fsvd`. The implementation instead uses symmetric eigendecomposition of the Gram matrix (`Cwᵀ·Cw` or `Cw·Cwᵀ`) because nalgebra's general SVD failed to converge reliably on near-rank-deficient cross-covariance inputs (recompose error ~0.10). The plan intent (weighted cross-cov → unit-L2 singular functions → paired sign fix → scores → FsvdResult) is fully satisfied; only the SVD engine differs. The deviation is sound and well-documented in both the rustdoc and the SUMMARY. No override block is required as the plan itself acknowledged clamp vs. error as a design choice and the deviation is of the same character (an implementation-detail substitution, not a missing capability).

### Human Verification Required

(none — all success criteria are verifiable programmatically and all tests are green)

---

_Verified: 2026-08-21T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
