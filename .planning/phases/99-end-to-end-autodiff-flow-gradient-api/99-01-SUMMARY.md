---
phase: 99-end-to-end-autodiff-flow-gradient-api
plan: "01"
subsystem: autodiff
tags: [autodiff, composition, grad, vjp, gen-02, api-01, doctest, integration]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: grad/jacobian/vjp unified API + re-exports
  - phase: 95-generic-scalar-hot-path-signatures
    provides: inner_product / l2_distance
  - phase: 97-differentiable-regression-prediction-smoothing-penalties
    provides: predict_curve_generic
  - phase: 98-differentiable-depth-curve-distances
    provides: modal_depth_generic

provides:
  - "tests/differentiable_composition.rs — end-to-end composed objective (predict + inner_product + modal depth) FD-checked at Dual and Var (GEN-02)"
  - "autodiff/mod.rs module doctest demonstrating grad + vjp agreement (API-01)"

affects: [100-release-preparation]

actuals: { tokens: 0, tasks: 2, commits: 1 }

tech-stack:
  added: []
  patterns:
    - "One Scalar-generic objective, both modes: grad (forward) and vjp (reverse) validated against each other + central FD"
    - "End-to-end composition spans phases 95/97/98; the unified API + re-exports (Phase 94) required no changes"

key-files:
  created:
    - fdars-core/tests/differentiable_composition.rs
  modified:
    - fdars-core/src/autodiff/mod.rs

key-decisions:
  - "GEN-02 + API-01 were largely enabled already: the grad/jacobian/vjp entry points and all generic-op re-exports existed from phases 94-98. Phase 99 adds the integration proof (composition demo) + the required running gradient-API doctest; no new public API."
  - "Composition uses an off-reference query so modal depth's L2 sqrt stays differentiable. Executed inline."

commits:
  - "(feat commit): feat(99): end-to-end autodiff composition demo + gradient-API doctest (GEN-02, API-01)"

gates:
  - "composition integration test: 1 pass (Dual + Var FD, all coords)"
  - "autodiff doctests: 8 pass (incl. new grad/vjp demo)"
  - "clippy --all-targets clean; full suite 0 failed; 213 doctests; no new dep"
---

# Phase 99 — End-to-End Autodiff Flow & Gradient API (GEN-02, API-01)

The broadened differentiable subset composes end-to-end: `tests/differentiable_composition.rs` builds one `Scalar`-generic objective from `predict_curve_generic` (97) + `inner_product` (95) + `modal_depth_generic` (98) and confirms its gradient w.r.t. the curve matches central finite differences at both forward-mode `Dual` and reverse-mode `Var` (vjp). A module doctest in `autodiff/mod.rs` demonstrates the unified gradient API (`grad` vs `vjp`, analytic-checked). The `grad`/`jacobian`/`vjp` entry points and all new generic-op re-exports were already in place (phases 94–98), so this phase is the integration proof + required doctest — no new public API, no new dependency.
