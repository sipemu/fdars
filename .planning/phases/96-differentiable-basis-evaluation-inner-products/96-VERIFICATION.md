---
phase: 96-differentiable-basis-evaluation-inner-products
verified: 2026-09-11T16:00:00Z
status: passed
score: 3/3 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 96: Differentiable Basis Evaluation & Inner Products (DOP-01) Verification Report

**Phase Goal:** Basis evaluation (B-spline / Fourier) and functional inner products are generic over Scalar and differentiable, with f64 numerics preserved (family 1 of DIF-F2).
**Verified:** 2026-09-11T16:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | B-spline basis evaluation (`bspline_basis_from_knots<T: Scalar>`, `evaluate_order_zero<T>`, `bspline_recurrence_step<T>`) is generic over Scalar; at f64 it reproduces known structural properties (partition-of-unity 1e-12, reflection symmetry 1e-12). | VERIFIED | `bspline.rs:22-95` — three generic functions present with `T: Scalar` bounds; `test_bspline_basis_from_knots_f64_known_properties` passes (independent oracle: partition-of-unity + reflection symmetry, not self-comparison). |
| 2 | Functional inner products are generic over Scalar and differentiable through their inputs (satisfied by Phase 95 `inner_product<T>`; this phase exercises them in the validation objective). | VERIFIED | `utility.rs` generic `inner_product<T: Scalar>` supplied by Phase 95; exercised in six named DOP-01 tests that compose basis eval → inner_product → scalar differentiability objective. All six pass. |
| 3 | Gradients of a basis-eval / inner-product objective match finite differences within tolerance at both Dual (forward-mode) and reverse-mode Var(vjp) — for B-spline AND Fourier. | VERIFIED | Four gradient tests pass individually: `test_bspline_inner_product_objective_dual` (PASS), `test_bspline_inner_product_objective_var` (PASS), `test_fourier_inner_product_objective_dual` (PASS), `test_fourier_inner_product_objective_var` (PASS). Tolerance: `1e-6 * (1.0 + fd.abs())` (absolute-floor form, consistent across both bases after WR-02 fix). Central FD with h=1e-6. |

**Score:** 3/3 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/basis/bspline.rs` | Generic `bspline_basis_from_knots<T: Scalar>`, `evaluate_order_zero<T>`, `bspline_recurrence_step<T>` | VERIFIED | All three functions present with `T: Scalar` bounds at lines 22, 43, 74. Knots stay `f64`; only evaluation points and basis values carry `T`. `bspline_basis` f64 wrapper unchanged. |
| `fdars-core/src/basis/fourier.rs` | NEW generic `fourier_basis_eval<T: Scalar>`; `fourier_basis` / `fourier_basis_with_period` delegate with unchanged signatures | VERIFIED | `fourier_basis_eval<T: Scalar>` at lines 58-87; `fourier_basis_with_period` delegates at line 45; `fourier_basis` unchanged at lines 23-28. Public signatures byte-identical to pre-phase-96 forms. |
| `fdars-core/src/basis/tests.rs` | B-spline f64 known-properties + Dual FD + Var FD tests; Fourier known-answer + Dual FD + Var FD tests | VERIFIED | 6 new DOP-01 tests present and individually passing. WR-01/02 code review findings addressed: tautological parity tests replaced with independent oracles (partition-of-unity+symmetry for bspline; hand-computed known-answer for Fourier); absolute-floor tolerance `1e-6*(1.0+fd.abs())` used uniformly. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `bspline_basis_from_knots<T>` | `crate::utility::inner_product<T>` | T-valued basis column extracted and fed to inner_product in `test_bspline_inner_product_objective_dual/var` | WIRED | Both Dual and Var tests compose the full chain: bspline eval → extract column j → inner_product → scalar; gradient flows confirmed by FD match. |
| `fourier_basis_eval<T>` | `crate::utility::inner_product<T>` | T-valued Fourier column extracted and fed to inner_product in `test_fourier_inner_product_objective_dual/var` | WIRED | Same chain for Fourier; FD match confirmed. |
| `fourier_basis_with_period` (f64) | `fourier_basis_eval<T>` | Line 45: `fourier_basis_eval(t, nbasis, period, t_min)` with t_min derived from t | WIRED | Delegation confirmed by reading `fourier.rs:43-46`. f64 wrappers produce identical output because `f64::from_f64` is the identity and operation order preserved. |
| `bspline_basis_from_knots<T>` | callers (`helpers.rs:523`, `helpers.rs:671`, `basis/pspline.rs:168`) | T=f64 inferred at all call sites | WIRED | grep confirms all callers pass `&[f64]` slices — T=f64 is inferred, zero source churn. All callers compile unchanged. |
| `fourier_basis` / `fourier_basis_with_period` | callers (seasonal/strength.rs, smooth_basis.rs, seasonal/period.rs, basis/projection.rs, auto_select.rs, fourier_fit.rs, elastic_regression/scalar_on_shape.rs) | Unchanged f64 public signatures | WIRED | grep confirms all 7+ call sites use the unchanged 2- or 3-arg f64 signatures. No caller modified. |
| `fourier_basis_eval` | `basis/mod.rs` re-export + `lib.rs` re-export | `pub use fourier::{fourier_basis, fourier_basis_eval, fourier_basis_with_period}` (WR-03 fix) | WIRED | `basis/mod.rs:32` and `lib.rs:680` both re-export `fourier_basis_eval`. Barrel symmetry with `bspline_basis_from_knots` achieved. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `bspline_basis_from_knots<T>` | basis matrix (Vec<T>) | Cox-de-Boor recurrence over caller-supplied `t: &[T]` and `knots: &[f64]` | Yes — pure computation over caller inputs, no static returns | FLOWING |
| `fourier_basis_eval<T>` | basis matrix (Vec<T>) | Fourier series expansion over caller-supplied `t: &[T]`, `period: f64`, `t_min: f64` | Yes — pure computation, DC and sin/cos harmonics derived from t | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| B-spline known-properties test (partition-of-unity + reflection symmetry at f64) | `cargo test -p fdars-core --lib --features linalg,parallel "basis::tests::test_bspline_basis_from_knots_f64_known_properties"` | 1 passed | PASS |
| B-spline Dual FD gradient match (forward-mode) | `cargo test -p fdars-core --lib --features linalg,parallel "basis::tests::test_bspline_inner_product_objective_dual"` | 1 passed | PASS |
| B-spline Var FD gradient match (reverse-mode vjp) | `cargo test -p fdars-core --lib --features linalg,parallel "basis::tests::test_bspline_inner_product_objective_var"` | 1 passed | PASS |
| Fourier known-answer test (hand-computed DC/sin/cos at t=0,¼,½,¾) | `cargo test -p fdars-core --lib --features linalg,parallel "basis::tests::test_fourier_basis_eval_f64_known_answer"` | 1 passed | PASS |
| Fourier Dual FD gradient match (forward-mode) | `cargo test -p fdars-core --lib --features linalg,parallel "basis::tests::test_fourier_inner_product_objective_dual"` | 1 passed | PASS |
| Fourier Var FD gradient match (reverse-mode vjp) | `cargo test -p fdars-core --lib --features linalg,parallel "basis::tests::test_fourier_inner_product_objective_var"` | 1 passed | PASS |
| Full test suite (2916 lib + integration + doctests) | `cargo test -p fdars-core --features linalg,parallel` | 2916 lib: 0 failed; 209 doctests: 0 failed | PASS |
| Clippy --all-targets | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | clean (Finished with no warnings) | PASS |
| 28 examples build | `cargo build -p fdars-core --examples --features linalg,parallel` | clean | PASS |
| Serde build | `cargo build -p fdars-core --features serde` | clean | PASS |
| WASM build | `cargo build -p fdars-core --target wasm32-unknown-unknown --features js` | clean | PASS |

---

### Probe Execution

No probe scripts declared for this phase. Gate commands run directly as behavioral spot-checks above.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOP-01 | Plans 01, 02, 03 | Basis evaluation (B-spline / Fourier) and functional inner products are generic over Scalar and differentiable; f64 path reproduces current numerics. | SATISFIED | Generic signatures present in bspline.rs and fourier.rs; 6 DOP-01 tests pass; non-breaking gate (all builds, all tests) clean; churn confined to basis/{bspline,fourier,tests}.rs + barrel re-exports in mod.rs and lib.rs. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No TBD/FIXME/XXX markers. No stub patterns. No empty implementations. No hardcoded empty data in non-test code.

---

### Churn Confinement Check

`git diff fccb4c82..3ef58900 -- fdars-core/src/` touches:

- `fdars-core/src/basis/bspline.rs` — algorithm generalization
- `fdars-core/src/basis/fourier.rs` — additive generic core + delegating wrappers
- `fdars-core/src/basis/tests.rs` — 6 new DOP-01 tests + WR-01/02 fixes
- `fdars-core/src/basis/mod.rs` — WR-03: added `fourier_basis_eval` to barrel re-export
- `fdars-core/src/lib.rs` — WR-03: added `fourier_basis_eval` to crate-root re-export

The `mod.rs` and `lib.rs` changes are additive barrel re-exports only (WR-03 code-review fix for API symmetry). No algorithm call-site was changed anywhere in `fdars-core/src/`. No `Cargo.toml` change (no new dependency). This matches the expected scope.

---

### Code Review Findings Status

The phase underwent a post-execution code review that identified 3 warnings. All were addressed in commit `3ef58900` before this verification:

- **WR-01 (tautological parity tests):** Replaced with independent oracles — bspline uses partition-of-unity + reflection symmetry; Fourier uses hand-computed known-answer at t ∈ {0, ¼, ½, ¾}. RESOLVED.
- **WR-02 (fragile absolute-floor tolerance):** B-spline FD tests updated to `1e-6 * (1.0 + fd.abs())` (same as Fourier). RESOLVED.
- **WR-03 (fourier_basis_eval missing from barrel re-export):** Added to `basis/mod.rs:32` and `lib.rs:680`. RESOLVED.

Review status: **clean** (0 critical, 0 open warnings).

---

### Human Verification Required

None. All must-haves are verified via compilation, structural code reading, and individual named behavioral tests. No visual, UI, or external-service dependencies.

---

## Gaps Summary

No gaps. All three success criteria are met:

1. B-spline and Fourier basis evaluation are generic over Scalar with independent f64 oracle tests passing (structural properties, not self-comparison).
2. Functional inner products from Phase 95 are exercised differentiably in the validation objective.
3. Gradients match finite differences within `1e-6 * (1.0 + fd.abs())` at both Dual (forward) and Var (reverse-mode vjp) for both basis families.

The whole-crate non-breaking gate is clean: 2916 lib tests pass, 209 doctests pass, clippy --all-targets clean, 28 examples build, serde build clean, WASM build clean, no new dependency.

---

_Verified: 2026-09-11T16:00:00Z_
_Verifier: Claude (gsd-verifier)_
