---
phase: 97-differentiable-regression-prediction-smoothing-penalties
plan: "02"
subsystem: smooth_basis
tags: [smoothing, roughness-penalty, generics, scalar, dual, var, autodiff, DOP-03]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: Scalar, Dual, Var+vjp

provides:
  - "penalty_value_generic<T: Scalar>(coef: &[T], penalty: &FdMatrix, lambda: f64) -> T — differentiable roughness-penalty value λ·cᵀRc"
  - "bit-identical f64 parity to direct λ·cᵀRc; Dual + Var(vjp) gradients FD-checked w.r.t. coefficients"
  - "additive crate-root + prelude re-exports"

affects: [97-03-non-breaking-gate, 99-end-to-end-autodiff]

actuals: { tokens: 0, tasks: 3, commits: 1 }

tech-stack:
  added: []
  patterns:
    - "Additive scalar penalty-value evaluator — does NOT alter the smoothing normal-equations solve"
    - "Differentiable input = coefficients c; penalty matrix R + lambda stay f64 lifted via T::from_f64; grad 2λRc for symmetric R"

key-files:
  created: []
  modified:
    - fdars-core/src/smooth_basis.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

key-decisions:
  - "DOP-03 differentiable input = spline coefficients c (user decision), NOT raw curve through the solve nor λ. penalty_value_generic is a NEW standalone evaluation; f64 parity is against a direct λ·cᵀRc reference (bit-identical, assert_eq!)."
  - "Penalty-matrix carrier is &FdMatrix (tests build it via FdMatrix::from_column_major of bspline_penalty_matrix's Vec<f64>). Penalty-matrix constructors unchanged."
  - "Executed inline by the orchestrator."

commits:
  - "891af090: feat(97-02): add differentiable penalty_value_generic<T> (DOP-03)"

gates:
  - "smooth_basis::tests::test_penalty_value_* — 3 pass (bit-identical parity + Dual + Var FD)"
  - "clippy --lib clean"
---

# Phase 97 Plan 02 — Differentiable Roughness Penalty (DOP-03)

`penalty_value_generic<T: Scalar>` evaluates `λ·cᵀRc` for a coefficient vector against a fixed f64 penalty matrix, differentiable w.r.t. the coefficients (`grad = 2λRc` for symmetric R). Additive — does not touch the smoothing fit. f64 parity is bit-identical to a direct reference (`assert_eq!`); Dual + Var(vjp) gradients match central FD.
