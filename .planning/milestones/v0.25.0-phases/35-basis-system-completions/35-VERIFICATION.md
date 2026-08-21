---
phase: 35-basis-system-completions
verified: 2026-08-21T11:00:00Z
status: passed
score: 13/13
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 35: Basis System Completions — Verification Report

**Phase Goal:** Add `monomial_basis`/`exponential_basis`/`power_basis`/`polygonal_basis` factories (with penalty matrices) to `basis/`, a composable `MultiFunData` multi-domain container in new `multi_fdata.rs`, a composable `Lfd`/linear-differential-operator object, and a `principal_differential_analysis` estimator — extending `basis/` additively without changing existing basis code. Result-returning, no new dependency, numeric outputs only.

**Verified:** 2026-08-21T11:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Four Result-returning basis factories exist (monomial/exponential/power/polygonal) at crate root, each producing eval matrix + penalty matrix | ✓ VERIFIED | `basis/monomial.rs`, `basis/exponential.rs`, `basis/power.rs`, `basis/polygonal.rs` all exist and are re-exported via `basis/mod.rs` and `src/lib.rs` line 493-497 |
| 2 | Factories evaluate to closed-form references within tolerance | ✓ VERIFIED | 46 inline tests pass: monomial closed-form, exponential closed-form, power integer-exponents match monomial, polygonal hat-peak and partition-of-unity all confirmed green |
| 3 | BasisSystem bundles eval_matrix + penalty_matrix, both Result-returning | ✓ VERIFIED | `basis_system.rs` defines `BasisSystem { eval_matrix, penalty_matrix, nbasis, n_eval, lfd_order }` with `#[non_exhaustive]`, `#[derive(Debug, Clone, PartialEq)]`, conditional serde; all four factories return `Result<BasisSystem, FdarError>` |
| 4 | MultiFunData multi-domain container with constructors + accessors preserving per-component argvals, enforcing consistent observation count | ✓ VERIFIED | `multi_fdata.rs` — `MultiFunData::new` enforces `data.nrows()==n_obs` and `argvals.len()==data.ncols()`; accessors `n_obs()`, `n_components()`, `component(k)`, `argvals(k)` all implemented; 15 inline tests pass |
| 5 | MultiFunData allows different domains per component | ✓ VERIFIED | `test_two_component_different_grids_ok` exercises 5 obs × 10 cols + 5 obs × 4 cols construction, returns Ok |
| 6 | Lfd::apply applies to functional data returning FdMatrix of same shape | ✓ VERIFIED | `pda.rs` — `Lfd { coefs: Vec<Vec<f64>> }` with `apply(&FdMatrix, &[f64]) -> Result<FdMatrix, FdarError>`; constant-operator test + shape-preserved test pass |
| 7 | principal_differential_analysis recovers harmonic-oscillator coefficients within tolerance | ✓ VERIFIED | `pda::tests::pda_recovers_harmonic_oscillator` passes: w=2π, 20 curves, 101 pts; β₀ ≈ ω² ≈ 39.478 and β₁ ≈ 0, both within tolerance 1.0 at every interior grid point |
| 8 | New items reuse existing infrastructure, add no new crate dependency | ✓ VERIFIED | `git diff --exit-code fdars-core/Cargo.toml` returns exit 0 (clean); exponential/power/polygonal reuse `crate::smooth_basis::{differentiate_basis_columns, integrate_symmetric_penalty}` and `crate::helpers::simpsons_weights`; pda reuses `crate::helpers::gradient` and existing `nalgebra` SVD |
| 9 | Invalid inputs return FdarError, not panic — all four factories + Lfd + PDA | ✓ VERIFIED | Tests confirmed: argvals.len()<2 → InvalidDimension; nbasis/rates/exponents empty → InvalidParameter; power with non-positive domain + non-integer exponent → InvalidParameter; polygonal non-monotone/duplicate knots → InvalidParameter; Lfd argvals mismatch → InvalidDimension; Lfd bad coefs length → InvalidDimension; PDA n<order+1 → InvalidDimension; PDA order=0 → InvalidParameter |
| 10 | Existing `basis/` public signatures unchanged (additive/non-breaking) | ✓ VERIFIED | `bspline_basis`, `fourier_basis`, `constant_basis`, `pspline_fit_1d` retain original signatures returning bare `Vec<f64>`; only additive `pub mod` declarations and `pub use` lines added to `mod.rs`/`lib.rs` |
| 11 | smooth_basis private helpers promoted to pub(crate) for downstream reuse | ✓ VERIFIED | `smooth_basis.rs` line 550: `pub(crate) fn differentiate_basis_columns`; line 573: `pub(crate) fn integrate_symmetric_penalty`; no body or signature changes |
| 12 | MultiFunData + FdComponent + Lfd + PdaResult + principal_differential_analysis are crate-root re-exported | ✓ VERIFIED | `lib.rs` line 130: `pub mod multi_fdata;`; line 131: `pub mod pda;`; line 138: `pub use multi_fdata::{FdComponent, MultiFunData};`; line 141: `pub use pda::{Lfd, PdaResult, principal_differential_analysis};` |
| 13 | Full suite (2359 tests) + clippy --all-targets --features linalg,parallel -- -D warnings clean | ✓ VERIFIED | `cargo test`: 2359 passed, 0 failed; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: 0 warnings, finished cleanly |

**Score:** 13/13 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|---------|--------|---------|
| `fdars-core/src/basis/basis_system.rs` | BasisSystem struct | ✓ VERIFIED | 75 lines; `pub struct BasisSystem { eval_matrix, penalty_matrix, nbasis, n_eval, lfd_order }` with `#[non_exhaustive]` + derives |
| `fdars-core/src/basis/monomial.rs` | monomial_basis factory | ✓ VERIFIED | 286 lines; factory + analytic Gram penalty + 7 inline tests |
| `fdars-core/src/basis/exponential.rs` | exponential_basis factory | ✓ VERIFIED | 262 lines; factory + numeric Gram via promoted helpers + 8 inline tests |
| `fdars-core/src/basis/power.rs` | power_basis factory | ✓ VERIFIED | 381 lines; factory + analytic/numeric Gram branch on integer vs non-integer exponents + 10 inline tests |
| `fdars-core/src/basis/polygonal.rs` | polygonal_basis factory | ✓ VERIFIED | 368 lines; hat-function eval + 1st-order numeric Gram + 10 inline tests |
| `fdars-core/src/multi_fdata.rs` | MultiFunData container | ✓ VERIFIED | 384 lines; `FdComponent` + `MultiFunData` + invariant enforcement + accessors + 15 inline tests |
| `fdars-core/src/pda.rs` | Lfd + PDA estimator | ✓ VERIFIED | 571 lines; `Lfd::apply` + `PdaResult` + `principal_differential_analysis` + 9 inline tests incl. harmonic-oscillator |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `basis/mod.rs` | `basis_system::BasisSystem` | `pub mod basis_system; pub use basis_system::BasisSystem;` | ✓ WIRED | Confirmed in mod.rs lines 8, 29 |
| `basis/mod.rs` | `monomial::monomial_basis` | `pub mod monomial; pub use monomial::monomial_basis;` | ✓ WIRED | mod.rs lines 14, 34 |
| `basis/mod.rs` | exponential/power/polygonal factories | `pub mod` + `pub use` for each | ✓ WIRED | mod.rs lines 10/12/16 + 31/35/36 |
| `src/lib.rs` | basis factories + BasisSystem | `pub use basis::{..., monomial_basis, polygonal_basis, power_basis, ..., BasisSystem}` | ✓ WIRED | lib.rs lines 493-497 |
| `src/lib.rs` | `multi_fdata::{MultiFunData, FdComponent}` | `pub mod multi_fdata; pub use multi_fdata::{...}` | ✓ WIRED | lib.rs lines 130, 138 |
| `src/lib.rs` | `pda::{Lfd, PdaResult, principal_differential_analysis}` | `pub mod pda; pub use pda::{...}` | ✓ WIRED | lib.rs lines 131, 141 |
| `exponential.rs`/`power.rs`/`polygonal.rs` | `smooth_basis::{differentiate_basis_columns, integrate_symmetric_penalty}` | `use crate::smooth_basis::{...}` | ✓ WIRED | Import confirmed in each file; helpers confirmed `pub(crate)` at smooth_basis.rs lines 550, 573 |
| `pda.rs` | `crate::helpers::gradient` | `crate::helpers::gradient(&row, argvals)` in Lfd::apply and PDA derivative loop | ✓ WIRED | Confirmed at pda.rs lines 134, 275 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `monomial.rs` | `eval_matrix` | `t.powi(j as i32)` over argvals | Yes — analytic closed form | ✓ FLOWING |
| `monomial.rs` | `penalty_matrix` | `monomial_penalty_analytic` (falling-factorial Gram) | Yes — analytic formula | ✓ FLOWING |
| `exponential.rs` | `eval_matrix` | `(rates[j] * t).exp()` over argvals | Yes | ✓ FLOWING |
| `exponential.rs` | `penalty_matrix` | `differentiate_basis_columns` + `integrate_symmetric_penalty` on fine grid | Yes — numeric quadrature | ✓ FLOWING |
| `polygonal.rs` | `eval_matrix` | `hat_function(t, knots, j)` | Yes — piecewise linear formula | ✓ FLOWING |
| `pda.rs` | `PdaResult.coefficients` | SVD pseudoinverse of per-point design matrices built from `helpers::gradient` derivatives | Yes — real ODE recovery | ✓ FLOWING |
| `multi_fdata.rs` | `components` | Caller-supplied `FdComponent` blocks validated in `new()` | Yes — validated through constructor | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| monomial closed-form eval | `cargo test -- monomial_eval_matrix_closed_form` | ok | ✓ PASS |
| monomial penalty P[2,2]=4.0 on [0,1] | `cargo test -- monomial_penalty_p22_standard_domain` | ok | ✓ PASS |
| exponential rate=0 is constant 1.0 | `cargo test -- exp_rate_zero_is_constant_one` | ok | ✓ PASS |
| power integer exponents match monomial | `cargo test -- power_integer_exponents_match_monomial` | ok | ✓ PASS |
| polygonal partition-of-unity | `cargo test -- poly_partition_of_unity` | ok | ✓ PASS |
| polygonal hat-peaks at knot | `cargo test -- poly_hat_peaks_at_knot` | ok | ✓ PASS |
| PDA recovers harmonic oscillator | `cargo test -- pda_recovers_harmonic_oscillator` | ok | ✓ PASS |
| MultiFunData two-component different grids | `cargo test -- test_two_component_different_grids_ok` | ok | ✓ PASS |
| Full suite + clippy | `cargo test` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | 2359 passed, 0 warnings | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REP-01 SC1 | 35-01, 35-02 | Four basis factories with eval + penalty matrices | ✓ SATISFIED | All four factories exist, are crate-root accessible, produce correct eval and penalty matrices |
| REP-01 SC2 | 35-03 | MultiFunData multi-domain container | ✓ SATISFIED | `multi_fdata.rs` with validated invariants, accessors, crate-root re-exports |
| REP-01 SC3 | 35-04 | Lfd + PDA estimator recovering ODE coefficients | ✓ SATISFIED | `pda.rs` with `Lfd::apply`, `principal_differential_analysis`, harmonic-oscillator test passes |
| REP-01 SC4 | 35-01 through 35-04 | Invalid inputs return FdarError, no panic | ✓ SATISFIED | 15+ validation tests across all new items, all green |
| REP-01 SC5 | 35-01, 35-02 | Existing public signatures unchanged; suite + clippy green | ✓ SATISFIED | Bare-Vec factories untouched; 2359 tests pass; clippy clean; Cargo.toml unchanged |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None found |

Debt-marker scan (TBD, FIXME, XXX, TODO, HACK, PLACEHOLDER) across all 7 new files returned no matches.

### Human Verification Required

None. All observable truths are fully verifiable from the codebase and test results.

### Gaps Summary

No gaps. All 13 must-haves from the four PLANs are verified in the actual codebase:

- **SC1 (four factories):** `monomial.rs`, `exponential.rs`, `power.rs`, `polygonal.rs` all exist, produce closed-form eval matrices and penalty matrices, and are crate-root re-exported.
- **SC2 (MultiFunData):** `multi_fdata.rs` with validated invariants and per-component argvals preservation exists and is wired.
- **SC3 (Lfd + PDA):** `pda.rs` with `Lfd::apply` (iterated `helpers::gradient` derivatives, constant-coefficient broadcast) and `principal_differential_analysis` (pointwise SVD pseudoinverse) exist. The harmonic-oscillator recovery test (`pda_recovers_harmonic_oscillator`) passes.
- **SC4 (error handling):** All input-validation paths return `FdarError` variants; no panics; confirmed by targeted tests.
- **SC5 (non-breaking + green):** Existing factories return bare `Vec<f64>` (signatures unchanged); `Cargo.toml` is clean (no new dependency); 2359 tests pass; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` produces zero warnings.

---

_Verified: 2026-08-21T11:00:00Z_
_Verifier: Claude (gsd-verifier)_
