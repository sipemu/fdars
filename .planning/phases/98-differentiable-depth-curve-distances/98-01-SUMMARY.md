---
phase: 98-differentiable-depth-curve-distances
plan: "01"
subsystem: depth
tags: [depth, modal, generics, scalar, dual, var, autodiff, DOP-04]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: Scalar, Dual, Var+vjp
  - phase: 95-generic-scalar-hot-path-signatures
    provides: l2_distance<T> (satisfies the curve-distance criterion)

provides:
  - "modal_depth_generic<T: Scalar>(curve: &[T], reference: &FdMatrix, h: f64) -> T — differentiable kernel-density modal depth"
  - "f64 parity vs modal_1d ~1e-12; Dual + Var(vjp) gradients FD-checked; running doctest; additive crate-root + prelude re-exports"

affects: [99-end-to-end-autodiff]

actuals: { tokens: 0, tasks: 2, commits: 1 }

tech-stack:
  added: []
  patterns:
    - "Additive generic per-curve depth; modal_1d/modal batch f64 path unchanged"
    - "Differentiable input = object curve; reference + bandwidth stay f64 (from_f64 lift); match modal_1d op order for parity"
    - "Modal depth non-differentiable exactly at query==reference (dist=0 sqrt kink) — FD tests use off-reference queries; documented"

key-files:
  created: []
  modified:
    - fdars-core/src/depth/modal.rs
    - fdars-core/src/depth/mod.rs
    - fdars-core/src/depth/tests.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

key-decisions:
  - "Modal depth is the ONLY smooth kernel-density depth (Gaussian + L2); all rank/CDF/combinatorial depths (FM, band, RP/tukey, rpd) are non-differentiable and deferred permanently."
  - "Curve-distance criterion #2 satisfied by l2_distance<T> (Phase 95) — not re-generalized; documented + its existing Dual/Var tests referenced. No bonus Lp distance (scope discipline)."
  - "Executed inline by the orchestrator (small phase; conserving context after the milestone's earlier agent connection-drops)."

commits:
  - "(feat commit): feat(98): add differentiable modal_depth_generic<T> (DOP-04)"

gates:
  - "modal_depth_generic tests — 3 pass (parity 1e-12 + Dual + Var FD)"
  - "clippy --all-targets clean; full test 0 failed; 212 doctests; 28 examples; serde; wasm; no new dep"
---

# Phase 98 Plan 01 — Differentiable Depth & Curve Distances (DOP-04)

`modal_depth_generic<T: Scalar>` gives a differentiable (Gaussian-kernel + L2) modal depth of a curve against a reference set — the single smooth depth in the crate. Differentiable w.r.t. the object curve; reference + bandwidth stay f64. f64 parity to modal_1d within ~1e-12; Dual + Var(vjp) gradients match central FD at off-reference queries (the query==reference self-match is an expected sqrt kink, like `abs` at 0). The curve-distance criterion is already satisfied by `l2_distance<T>` from Phase 95 (generic, differentiable, with Dual/Var tests). All whole-crate gates green; no new dependency; `modal_1d`/`modal` signatures unchanged.
