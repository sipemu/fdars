---
wave: 1
depends_on: []
autonomous: true
requirements: [DOP-04]
files_modified:
  - fdars-core/src/depth/modal.rs
  - fdars-core/src/depth/mod.rs
  - fdars-core/src/depth/tests.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
---

# Phase 98 Plan 01 — Differentiable Depth & Curve Distances (DOP-04)

<objective>
Deliver ≥1 differentiable functional depth (modal) and confirm ≥1 differentiable curve distance (l2_distance, already generic from Phase 95), generic over `Scalar`, f64 parity preserved. FD-checked at Dual and Var. Executed inline by the orchestrator (small, well-specified phase; conserving context).
</objective>

<tasks>
<task><name>Task 1: modal_depth_generic + tests</name>
<action>Add `pub fn modal_depth_generic<T: Scalar>(curve: &[T], reference: &FdMatrix, h: f64) -> T` to depth/modal.rs = `(1/nref)·Σ_j exp(-0.5·(dist_j/h)²)`, `dist_j = sqrt(Σ_t (curve[t]-ref[j,t])²/n)`. Reference/h stay f64 (from_f64 lift); match modal_1d op order for parity. Re-export at depth/mod.rs + crate root + prelude. Add f64-parity (~1e-12 vs modal_1d) + Dual + Var(vjp) FD tests (off-reference query so dist>0). Add a running doctest.</action>
<verify><automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel --lib modal_depth_generic 2>&1 | tail -5</automated>
<fails_when>fewer than 3 tests pass, or "FAILED" in output</fails_when></verify>
<acceptance_criteria>
- modal_depth_generic exists, generic over Scalar, differentiable w.r.t. curve; modal_1d/modal unchanged.
- f64 parity vs modal_1d within 1e-12; Dual + Var gradients match central FD within 1e-6*(1+|fd|).
- l2_distance<T> (Phase 95) documented as satisfying the curve-distance criterion.
</acceptance_criteria>
</task>

<task><name>Task 2: non-breaking gate</name>
<action>clippy --all-targets, full test, doctests, 28 examples, serde, wasm; churn confined to depth/ + additive re-exports; no new dep.</action>
<verify><automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -2</automated>
<fails_when>clippy prints any warning/error</fails_when></verify>
<acceptance_criteria>All whole-crate gates green; no new dependency; public depth signatures unchanged.</acceptance_criteria>
</task>
</tasks>

<must_haves>
truths:
  - "At least one functional depth measure (modal) is generic over Scalar and differentiable w.r.t. curve values; f64 depth values unchanged (within 1e-12 of modal_1d)."
  - "At least one curve distance beyond soft-DTW (l2_distance, Phase 95) is generic over Scalar and differentiable."
  - "Gradients of the modal-depth path match central FD at both Dual and reverse-mode Var (1e-6*(1+|fd|))."
prohibitions:
  - statement: "No new crate dependency added to Cargo.toml"
    status: enforced
  - statement: "modal_1d/modal and other depth public signatures unchanged; churn confined to depth/ + additive re-exports"
    status: enforced

# Artifacts this phase produces
# - modal_depth_generic<T: Scalar> (depth/modal.rs) + its parity/Dual/Var tests + doctest
# - additive re-exports in depth/mod.rs, lib.rs, prelude.rs
</must_haves>

<threat_model>
No attack surface — pure numeric depth evaluation over caller-provided scalar/f64 slices; no I/O, network, deserialization, or privilege boundary.
</threat_model>
