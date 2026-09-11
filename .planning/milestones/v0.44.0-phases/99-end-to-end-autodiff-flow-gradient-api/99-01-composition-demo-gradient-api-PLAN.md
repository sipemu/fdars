---
wave: 1
depends_on: []
autonomous: true
requirements: [GEN-02, API-01]
files_modified:
  - fdars-core/src/autodiff/mod.rs
  - fdars-core/tests/differentiable_composition.rs
---

# Phase 99 Plan 01 — End-to-End Autodiff Flow & Gradient API (GEN-02, API-01)

<objective>
Prove the broadened differentiable subset composes end-to-end into a FD-checked scalar objective (GEN-02) and that the unified gradient API is exposed with a running doctest (API-01). Executed inline.
</objective>

<tasks>
<task><name>Task 1: composition demo integration test (GEN-02)</name>
<action>Add tests/differentiable_composition.rs: a Scalar-generic `objective(curve)` composing predict_curve_generic (97) + inner_product (95) + modal_depth_generic (98); FD-check its gradient w.r.t. the curve at Dual and Var(vjp), tol 1e-6*(1+|fd|), off-reference query.</action>
<verify><automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel --test differentiable_composition 2>&1 | tail -4</automated>
<fails_when>the composition test fails or does not run</fails_when></verify>
<acceptance_criteria>Composed objective's Dual tangent and Var vjp gradient both match central FD for every curve coordinate; reverse-mode primal matches f64 within 1e-9.</acceptance_criteria>
</task>

<task><name>Task 2: gradient-API module doctest (API-01)</name>
<action>Add a running module doctest to autodiff/mod.rs demonstrating grad + vjp on a Scalar-generic objective, analytic-checked. Confirm unified grad/jacobian/vjp + all new generic fns are crate-root + prelude re-exported.</action>
<verify><automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel --doc autodiff 2>&1 | tail -3</automated>
<fails_when>autodiff doctests fail</fails_when></verify>
<acceptance_criteria>The module doctest passes under cargo test; grad and vjp agree and match the analytic gradient.</acceptance_criteria>
</task>
</tasks>

<must_haves>
truths:
  - "Both Dual and Var flow through the generalized hot-paths end-to-end; a composed objective built from the broadened subset yields FD-correct gradients (GEN-02)."
  - "Ergonomic grad/jacobian/vjp entry points are exposed; full crate-root + prelude re-exports cover the new public surface (API-01)."
  - "A worked end-to-end composition demo exists and is finite-difference-checked."
  - "A running module doctest demonstrates the gradient API and passes under cargo test."
prohibitions:
  - statement: "No new crate dependency added to Cargo.toml"
    status: enforced

# Artifacts this phase produces
# - tests/differentiable_composition.rs (composed objective + Dual/Var FD checks)
# - autodiff/mod.rs module doctest (grad + vjp demo)
</must_haves>

<threat_model>
No attack surface — pure numeric autodiff demo/test over caller-provided data; no I/O, network, deserialization, or privilege boundary.
</threat_model>
