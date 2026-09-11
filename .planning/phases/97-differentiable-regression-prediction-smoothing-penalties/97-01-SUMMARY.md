---
phase: 97-differentiable-regression-prediction-smoothing-penalties
plan: "01"
subsystem: scalar_on_function
tags: [regression, fregre_lm, fpcr, generics, scalar, dual, var, autodiff, DOP-02]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: Scalar, Dual, Var+vjp
  - phase: 95-generic-scalar-hot-path-signatures
    provides: generic hot-path substrate
  - phase: 96-differentiable-basis-evaluation-inner-products
    provides: Phase-96 robust FD tolerance pattern

provides:
  - "predict_curve_generic<T: Scalar>(curve: &[T], fit: &FregreLmResult) -> T — differentiable per-curve FPCR prediction (project_scores_generic + intercept + Σ coef_k·score_k)"
  - "f64 parity vs predict_fregre_lm within 1e-6; Dual + Var(vjp) gradients FD-checked w.r.t. the curve"
  - "additive crate-root + prelude re-exports"

affects: [97-03-non-breaking-gate, 99-end-to-end-autodiff]

actuals: { tokens: 0, tasks: 3, commits: 1 }

tech-stack:
  added: []
  patterns:
    - "Independent additive autodiff entry point — predict_fregre_lm left byte-unchanged (its 3-mul batch kernel NOT refactored, per RESEARCH Option 1)"
    - "Differentiable input = curve; intercept/coefficients/fpca params stay f64 lifted via T::from_f64"

key-files:
  created: []
  modified:
    - fdars-core/src/scalar_on_function/fregre_lm.rs
    - fdars-core/src/scalar_on_function/mod.rs
    - fdars-core/src/scalar_on_function/tests.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

key-decisions:
  - "DOP-02 parity tolerance is 1e-6, NOT bit-identical — project_scores_generic pre-folds rotation·weights while the batch kernel does 3 separate mults (~1e-14 divergence). predict_fregre_lm NOT refactored (avoids rippling a 1e-14 change to conformal/FregreLmResult::predict callers)."
  - "Executed inline by the orchestrator (small phase; executors dropped on connection errors earlier in this milestone)."

commits:
  - "977a92db: feat(97-01): add differentiable predict_curve_generic<T> (DOP-02 tracer)"

gates:
  - "scalar_on_function:: tests — 9 predict tests pass incl. 3 new + existing predict_fregre_lm guard (1e-6)"
  - "clippy --lib clean; predict_fregre_lm signature unchanged"
---

# Phase 97 Plan 01 — Differentiable FPCR Prediction (DOP-02)

`predict_curve_generic<T: Scalar>` composes Phase 94's `project_scores_generic` with the fitted linear combination, giving a differentiable-w.r.t.-curve scalar prediction. `predict_fregre_lm` is left byte-unchanged (independent additive path). f64 parity within 1e-6 (accumulation-order divergence ~1e-14); Dual + Var(vjp) gradients match central FD (tol `1e-6*(1+|fd|)`). Re-exported at crate root + prelude.
