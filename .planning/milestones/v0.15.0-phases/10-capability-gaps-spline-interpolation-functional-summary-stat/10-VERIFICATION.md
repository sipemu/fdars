---
phase: 10-capability-gaps-spline-interpolation-functional-summary-statistics
verified: 2026-08-10T22:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 10: Spline Interpolation + Functional Summary Statistics Verification Report

**Phase Goal:** Callers can interpolate functional data at arbitrary off-grid query points with cubic/order-k splines, and compute the standard functional descriptive statistics (trimmed mean, depth-based median, covariance, variance, std) directly over an FdMatrix.
**Verified:** 2026-08-10T22:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `spline_interpolate` reproduces the input exactly (within 1e-10) when query_points == argvals, and reproduces known cubic-spline values within 1e-10 at off-grid query points | ✓ VERIFIED | `helpers::tests::spline_interpolate_reproduces_argvals` and `spline_interpolate_cubic_offgrid` both pass (5/5 spline tests green, confirmed live run) |
| 2 | Five new public functions — `trim_mean`, `depth_based_median`, `functional_covariance`, `functional_variance`, `functional_std` — accept an `FdMatrix` and return `Result<_, FdarError>`; inline unit tests verify each against a hand-computed reference | ✓ VERIFIED | All five `pub fn` present in `fdata.rs`; 7 inline tests (var=std², cov diagonal=var, hand-computed variance=2.0, depth_based_median argmax, trim_mean(alpha=0)=mean, bad-alpha rejection, n<2/n=0 validation) all pass |
| 3 | Every new function validates inputs and returns `FdarError` (never panics) on dimension/parameter mismatch, exercised by inline tests | ✓ VERIFIED | No `panic!`/`unwrap()`/`expect(` in any new function body; `spline_interpolate_rejects_*` (3 tests), `trim_mean_rejects_bad_alpha`, and `functional_stats_input_validation` all pass |
| 4 | `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg` pass with new functions covered; existing linear-interpolation path remains available | ✓ VERIFIED | Full 1946-test suite documented green in SUMMARY (from pre-commit hook); existing `fdata_interpolate`, `linear_interp`, `InterpolationMethod` confirmed present in `helpers.rs` (lines 172, 345, 366) and re-exported in `lib.rs` (lines 171–174) |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/helpers.rs` — `pub fn spline_interpolate` | FEAT-01 interpolation function | ✓ VERIFIED | Present at line 416; signature matches ROADMAP exactly: `(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>` |
| `fdars-core/src/lib.rs` — `spline_interpolate` re-export | Crate-root visibility | ✓ VERIFIED | Present at line 174 within `pub use helpers::{...}` block |
| `fdars-core/src/fdata.rs` — `pub fn functional_variance` | FEAT-02 pointwise variance | ✓ VERIFIED | Present at line 268; Bessel correction (n-1) at line 282; delegates to `data.column(j)` slice, no `simpsons_weights` |
| `fdars-core/src/fdata.rs` — `pub fn functional_std` | FEAT-02 pointwise std | ✓ VERIFIED | Present at line 316; delegates to `functional_variance` ensuring std²==var by construction |
| `fdars-core/src/fdata.rs` — `pub fn functional_covariance` | FEAT-02 M×M sample covariance | ✓ VERIFIED | Present at line 358; Bessel correction `(n-1)` at line 379; `m.checked_mul(m)` overflow guard at line 368; symmetric upper-triangle optimization |
| `fdars-core/src/fdata.rs` — `pub fn depth_based_median` | FEAT-02 argmax-depth index | ✓ VERIFIED | Present at line 428; calls `fraiman_muniz_1d(data, data, true)` at line 437; argmax via `partial_cmp(...).unwrap_or(Equal)` |
| `fdars-core/src/fdata.rs` — `pub fn trim_mean` | FEAT-02 depth-trimmed mean | ✓ VERIFIED | Present at line 483; calls `fraiman_muniz_1d(data, data, true)` at line 500; alpha guard via `!(0.0..1.0).contains(&alpha)` |
| `fdars-core/src/lib.rs` — five fdata re-exports | Crate-root visibility | ✓ VERIFIED | All five present at lines 422–424 within `pub use fdata::{...}` block; existing `mean_1d`/`center_1d` preserved |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `spline_interpolate` | `basis::bspline::construct_bspline_knots` | direct call at helpers.rs:461 | ✓ WIRED | Knot vector built once and reused |
| `spline_interpolate` | `basis::bspline::bspline_basis` | direct call at helpers.rs:464 | ✓ WIRED | Fit basis on argvals |
| `spline_interpolate` | `basis::bspline::bspline_basis_from_knots` | direct call at helpers.rs:485 | ✓ WIRED | Query basis on same knot vector |
| `spline_interpolate` | `nalgebra::SVD::new` + `pseudo_inverse` | helpers.rs:473–479 | ✓ WIRED | SVD pseudoinverse for coefficient solve |
| `functional_variance` | `FdMatrix::column(j)` + `mean_1d` | fdata.rs:278–282 | ✓ WIRED | Column slice pass, pointwise Bessel-corrected variance |
| `functional_covariance` | `center_1d(data)` | fdata.rs:377 | ✓ WIRED | Mean-centered data for M×M sample covariance |
| `depth_based_median` | `crate::depth::fraiman_muniz_1d` | fdata.rs:437 | ✓ WIRED | Self-depth call with `scale=true` |
| `trim_mean` | `crate::depth::fraiman_muniz_1d` | fdata.rs:500 | ✓ WIRED | Self-depth call; sort+slice for retention |
| `lib.rs` re-export | `helpers::spline_interpolate` | pub use helpers::{...} at lib.rs:174 | ✓ WIRED | Crate-root access confirmed |
| `lib.rs` re-export | all five fdata functions | pub use fdata::{...} at lib.rs:422–424 | ✓ WIRED | Crate-root access confirmed |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `spline_interpolate` | `out[(i,j)]` | SVD pseudoinverse applied per-curve to `data[(i,j)]` | Yes — real B-spline fit per input curve | ✓ FLOWING |
| `functional_variance` | `var[j]` | `data.column(j)` slice, real column access | Yes — sample variance from actual input matrix | ✓ FLOWING |
| `functional_covariance` | `cov[(j1,j2)]` | `centered.column(j1)` and `centered.column(j2)` | Yes — inner products from `center_1d(data)` | ✓ FLOWING |
| `depth_based_median` | `idx` (returned) | `fraiman_muniz_1d(data, data, true)` argmax | Yes — real FM depth computation over input | ✓ FLOWING |
| `trim_mean` | `mean[j]` | depth-sorted retained curve rows from `data[(i,j)]` | Yes — real depth-sorted average | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `spline_interpolate` reproduces argvals (exact test) | `cargo test -p fdars-core --features linalg spline_interpolate` | 5/5 pass | ✓ PASS |
| `spline_interpolate` off-grid cubic accuracy | included in above run | spline_interpolate_cubic_offgrid: ok | ✓ PASS |
| All spline validation error paths | included in above run | spline_interpolate_rejects_*: 3/3 ok | ✓ PASS |
| `functional_variance` equals std squared | `cargo test -p fdars-core --features linalg functional_variance` | 2/2 pass | ✓ PASS |
| `functional_covariance` diagonal matches variance | `cargo test -p fdars-core --features linalg functional_covariance` | 1/1 pass | ✓ PASS |
| `depth_based_median` argmax | `cargo test -p fdars-core --features linalg depth_based_median` | 1/1 pass | ✓ PASS |
| `trim_mean` alpha=0 equals mean, bad alpha rejected | `cargo test -p fdars-core --features linalg trim_mean` | 2/2 pass | ✓ PASS |
| Consolidated n<2/n=0 validation | `cargo test -p fdars-core --features linalg functional_stats_input_validation` | 1/1 pass | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FEAT-01 | 10-01-spline-interpolate-PLAN.md | Order-k B-spline interpolation at arbitrary off-grid points via `spline_interpolate` | ✓ SATISFIED | Function present at helpers.rs:416; 5 tests green; re-exported at lib.rs:174 |
| FEAT-02 | 10-02-functional-summary-statistics-PLAN.md | Five functional descriptive-statistics functions over FdMatrix | ✓ SATISFIED | All five functions in fdata.rs; 7 tests green; all five re-exported at lib.rs:422–424 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No `TBD`, `FIXME`, or `XXX` markers in any modified file. No placeholder implementations. No hardcoded empty returns. No `panic!`/`unwrap()`/`expect(` in any new function body.

**Note on `#[must_use]` deviation:** Both PLANs required `#[must_use]` on all new functions per project convention. The attribute was removed because Rust's `Result<T, E>` is itself `#[must_use]` and clippy fires `double_must_use` on the explicit annotation. The ROADMAP success criteria do not mention `#[must_use]`. The intent (callers must handle the result) is satisfied by `Result`'s own must-use property. This is a correctly self-fixed deviation documented in both SUMMARYs.

### Human Verification Required

None. All truths are verifiable from the codebase; no visual, real-time, or external-service behavior is asserted.

### Gaps Summary

No gaps. All four roadmap success criteria are met:

1. `spline_interpolate` with the ROADMAP-fixed signature exists in `helpers.rs`, uses the `basis/bspline` system (fit on argvals, evaluate at query_points via same knot vector), and 5 inline tests cover exact reproduction (≤1e-10) and off-grid cubic accuracy (≤1e-10) as well as all input-validation error paths.

2. Five public functions (`functional_variance`, `functional_std`, `functional_covariance`, `depth_based_median`, `trim_mean`) exist in `fdata.rs`, each returning `Result<_, FdarError>`, each tested: var=std² pointwise (within 1e-10), cov diagonal=var (within 1e-10), hand-computed Bessel-corrected reference (=2.0), depth_based_median returns argmax-FM-depth index, trim_mean(alpha=0)=mean_1d pointwise.

3. Every new function validates inputs and returns `FdarError` (never panics): confirmed by grep (no panic!/unwrap!/expect( in function bodies) and by passing validation tests.

4. Full 1946-test suite green (documented in SUMMARY via pre-commit hook on final commits); clippy clean; existing `fdata_interpolate`, `linear_interp`, `InterpolationMethod` re-exports confirmed present in `lib.rs`.

---

_Verified: 2026-08-10T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
