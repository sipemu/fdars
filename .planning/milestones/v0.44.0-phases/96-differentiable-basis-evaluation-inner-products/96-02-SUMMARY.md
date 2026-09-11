---
phase: 96-differentiable-basis-evaluation-inner-products
plan: "02"
subsystem: basis
tags: [basis, fourier, generics, scalar, dual, var, autodiff, DOP-01, non-breaking]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: Scalar, Dual, Var+vjp — for the DOP-01 Dual/Var FD objective tests
  - phase: 95-generic-scalar-hot-path-signatures
    provides: inner_product<T> — reused unchanged in the combined objective
  - phase: 96-differentiable-basis-evaluation-inner-products
    plan: "01"
    provides: the tracer pattern (generic basis eval → inner_product → FD-checked objective)

provides:
  - "fourier_basis_eval<T: Scalar>(t: &[T], nbasis, period: f64, t_min: f64) -> Vec<T> — new additive generic Fourier basis core (differentiable through t)"
  - "fourier_basis / fourier_basis_with_period f64 wrappers refactored to derive t_min and delegate — PUBLIC SIGNATURES UNCHANGED"
  - "f64 bit-parity test (assert_eq!) + Dual and Var(vjp) FD objective tests"

affects: [96-03-non-breaking-gate]

actuals:
  tokens: 0
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Additive generic core when in-place is impossible: fourier_basis_with_period derives t_min from t internally, so a generic core takes t_min: f64 explicitly and the f64 wrappers delegate — keeps public signatures frozen (strictly non-breaking) instead of adding a param to the public fn"
    - "Bit-parity: preserve the EXACT f64 op order 2*PI*(ti - t_min)/period — do NOT precompute a 2*PI/period scale (reorders FP ops, breaks parity)"
    - "Robust FD tolerance 1e-6*(1.0 + fd.abs()) with an absolute floor — the relative-only 1e-6*fd.abs().max(1e-10) collapses to ~1e-16 at Fourier grid points where the derivative is exactly 0 (sin(2πt)=0), producing false failures"

key-files:
  created: []
  modified:
    - fdars-core/src/basis/fourier.rs
    - fdars-core/src/basis/tests.rs

key-decisions:
  - "ADDITIVE core (not in-place param-add): RESEARCH/PATTERNS suggested threading t_min into the public fourier_basis_with_period signature; the orchestrator overrode that (4+ callers + public/crate-root re-exported signature => breaking). New fourier_basis_eval<T> core + delegating f64 wrappers is strictly non-breaking."
  - "Executed INLINE by the orchestrator after two consecutive agent connection-drops in this phase (the planner and the 96-01 executor both dropped mid-run); the change is bounded and well-specified, so inline was more reliable than a third dispatch."

commits:
  - "b6daff85: feat(96-02): add generic fourier_basis_eval<T> core; f64 wrappers delegate (DOP-01)"

gates:
  - "cargo test --lib basis:: — 214 passed, 0 failed (incl. test_fourier_basis_eval_f64_parity, test_fourier_inner_product_objective_dual, test_fourier_inner_product_objective_var)"
  - "cargo clippy --all-targets --features linalg,parallel -- -D warnings — clean"
  - "cargo fmt — clean"
  - "public signatures of fourier_basis / fourier_basis_with_period unchanged (grep-confirmed)"
---

# Phase 96 Plan 02 — Generic Fourier Basis Core (DOP-01)

## What was built

A new **additive** generic-over-`Scalar` Fourier basis core, `fourier_basis_eval<T: Scalar>(t: &[T], nbasis, period: f64, t_min: f64) -> Vec<T>`, differentiable through the evaluation points `t`. The existing f64 wrappers `fourier_basis` and `fourier_basis_with_period` keep their **exact public signatures** — they derive `t_min` in f64 (as before) and delegate to the core. This delivers "Fourier basis evaluation generic over Scalar" (DOP-01 criterion #1) with **zero caller churn** and no public-API change.

## Why additive (not in-place)

`fourier_basis_with_period` derives `t_min` internally by folding over `t`. Generalizing it in place would force either (a) adding a `t_min: f64` parameter to a public, crate-root-re-exported function with 4+ in-crate callers — a breaking change violating the milestone's non-breaking invariant — or (b) a `Scalar::to_f64` (not adding one). The additive core sidesteps both.

## Verification (DOP-01)

1. **Generic + f64 bit-parity:** `fourier_basis_eval::<f64>` is bit-identical to `fourier_basis_with_period` (`assert_eq!`), guaranteed by `f64::from_f64` identity and preserving the exact `2π·(ti − t_min)/period` operation order.
2. **Differentiable + FD-checked:** a combined objective (Fourier eval at `t` → harmonic column → generic `inner_product` against a fixed curve → scalar) matches central FD (h=1e-6) at both `Dual` and `Var(vjp)`, using a robust absolute-floor tolerance to handle grid points where the Fourier derivative is exactly zero.

## Non-breaking

`git grep` confirms `fourier_basis` and `fourier_basis_with_period` signatures are byte-identical to pre-change; all callers (seasonal/strength.rs, smooth_basis.rs, seasonal/period.rs, fourier.rs wrapper, tests) compile unchanged. Verified by the whole-crate gate in Plan 03.
