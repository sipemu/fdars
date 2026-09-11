---
phase: 96-differentiable-basis-evaluation-inner-products
plan: "01"
subsystem: basis
tags: [basis, bspline, generics, scalar, dual, var, autodiff, tdd, DOP-01]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: Scalar trait, Dual (forward-mode), Var+vjp (reverse-mode) — needed for the DOP-01 Dual/Var FD objective tests
  - phase: 95-generic-scalar-hot-path-signatures
    provides: inner_product<T> (generic functional inner product) — the combined basis-eval → inner-product objective reuses it unchanged

provides:
  - "bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order) -> Vec<T> — in-place generic B-spline basis evaluation; f64 call sites unchanged (T=f64 inferred)"
  - "evaluate_order_zero<T: Scalar> and bspline_recurrence_step<T: Scalar> — generic Cox-de-Boor helpers"
  - "f64 bit-parity test (assert_eq!), partition-of-unity preserved (1e-10)"
  - "Combined basis-eval → inner_product objective differentiable w.r.t. t; FD-checked at Dual and Var(vjp) within 1e-6"

affects: [96-02-fourier-generic-core, 96-03-non-breaking-gate, 97-differentiable-regression]

actuals:
  tokens: 0
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "In-place generic basis eval: rewrite bspline_basis_from_knots<T> + Cox-de-Boor helpers over T rather than adding _generic companions"
    - "Recurrence bit-parity: ((t_val - T::from_f64(knots[j])) / T::from_f64(d1)) * b[j] — division THEN multiply, operation order preserved"
    - "Repeated-knot guard d1.abs() > 1e-10 stays f64; fallback 0.0 -> T::zero()"
    - "Span-search comparison lifts knots: t_val >= T::from_f64(knots[j]) (Scalar PartialOrd is T-vs-T, value-only for Dual/Var)"
    - "Order-zero indicator 1.0 -> T::one(); f64::from_f64 is identity so T=f64 is bit-identical"

key-files:
  created: []
  modified:
    - fdars-core/src/basis/bspline.rs
    - fdars-core/src/basis/tests.rs

key-decisions:
  - "Plain <T: Scalar> (no `= f64` default) — Rust 1.97 rejects invalid_type_param_default on free functions (future-incompatible); non-breaking holds via T=f64 argument inference"
  - "construct_bspline_knots and the auto-deriving bspline_basis wrapper stay f64-only (they derive knots/bounds from t; no Scalar::to_f64)"
  - "All bspline_basis_from_knots callers (helpers.rs spline_interpolate x2, basis/pspline.rs) compile unchanged at inferred T=f64; crate-root + basis/mod.rs re-exports intact"

recovery-note: >
  The dispatched gsd-executor committed the RED tests (Task 1, commit 0e2bc6a3) and
  generalized both Cox-de-Boor helpers, but its connection dropped mid-Task-2 before
  generalizing bspline_basis_from_knots itself, committing, or writing this SUMMARY.
  The orchestrator recovered inline (MEMORY: "execute inline + commit --no-verify after
  out-of-band gates" is the documented fallback when executors drop): generalized the
  main entry point (commit 16d9803a), ran gates foreground (basis tests 211/211 incl. the
  3 new parity/Dual/Var tests; clippy --all-targets clean; callers unchanged), and wrote
  this SUMMARY. No duplicate work; the helpers' partial edits were sound and retained.

commits:
  - "0e2bc6a3: test(96-01): add RED tests for bspline_basis_from_knots<T> parity and Dual/Var FD objectives"
  - "16d9803a: feat(96-01): generalize bspline_basis_from_knots to <T: Scalar> (tracer GREEN)"

gates:
  - "cargo test --lib basis:: — 211 passed, 0 failed (incl. test_bspline_basis_from_knots_f64_parity, test_bspline_inner_product_objective_dual, test_bspline_inner_product_objective_var)"
  - "cargo clippy --all-targets --features linalg,parallel -- -D warnings — clean"
  - "cargo fmt — clean"
  - "caller-churn grep — bspline_basis_from_knots callers unchanged (T=f64 inferred)"
---

# Phase 96 Plan 01 — Tracer: Generic B-spline Basis Evaluation (DOP-01)

## What was built

The B-spline evaluation path is now generic over `T: Scalar` with the **evaluation points `t` as the differentiable input**, proving the DOP-01 loop end-to-end on B-spline before Fourier (Plan 02):

- `bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order) -> Vec<T>` (in place)
- `evaluate_order_zero<T: Scalar>` and `bspline_recurrence_step<T: Scalar>` (Cox-de-Boor helpers)
- Knots stay `f64`; only the evaluation points and the resulting basis values carry `T`.

## Verification (DOP-01 criteria)

1. **Generic + f64 bit-parity:** `bspline_basis_from_knots::<f64>` reproduces the current numerics bit-identically (`assert_eq!` parity test) and preserves partition-of-unity within 1e-10. Guaranteed by `f64::from_f64` being the identity and the recurrence preserving operation order.
2. **Differentiable through inputs:** a combined objective — evaluate the basis at `t`, take a basis column, `inner_product` (Phase 95 generic) against a fixed lifted curve, return a scalar — is differentiable w.r.t. `t`.
3. **FD match at Dual and Var:** central finite differences (h=1e-6, tol 1e-6) match the gradients at both forward-mode `Dual` and reverse-mode `Var` (via `vjp`).

## Non-breaking

All existing `bspline_basis_from_knots` callers pass `&[f64]` → `T = f64` inferred → zero call-site churn; the auto-deriving `bspline_basis` wrapper and `construct_bspline_knots` are unchanged; crate-root and `basis/mod.rs` re-exports intact.

## Next

Plan 02 mirrors this on Fourier via a new additive generic core `fourier_basis_eval<T>` (public wrappers unchanged). Plan 03 is the whole-crate non-breaking gate.
