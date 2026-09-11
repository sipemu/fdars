# Phase 99: End-to-End Autodiff Flow & Gradient API - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary
Prove autodiff types flow end-to-end through the generalized hot-paths into a composed scalar objective, exposed via the unified gradient API, with a FD-checked worked demo + running module doctest (GEN-02 + API-01).

**In scope:** a worked end-to-end composition demo (integration test) chaining the broadened v0.44.0 subset into one `Scalar`-generic objective, FD-checked at Dual and Var; a running module doctest of the gradient API; confirm full crate-root + prelude re-exports cover all new public surface.

**Already in place (confirm, don't rebuild):** unified entry points `grad`/`jacobian`/`diff`/`directional_derivative` (forward.rs) + `vjp` (reverse.rs), all prelude+crate-root re-exported (Phase 94); the generalized ops `bspline_basis_from_knots`/`fourier_basis_eval` (96), `predict_curve_generic`/`penalty_value_generic` (97), `modal_depth_generic` (98), `l2_distance`/`inner_product` (95), `soft_dtw_distance_generic` — all re-exported.
</domain>

<decisions>
- Composition demo = integration test `tests/differentiable_composition.rs`: `objective(curve) = predict_curve_generic(curve, fit) + inner_product(curve, ref, argvals) + modal_depth_generic(curve, refmat, h)`, generic over `T: Scalar`, FD-checked (h=1e-6, tol 1e-6*(1+|fd|)) at Dual and Var(vjp). Off-reference query so modal depth's sqrt is smooth. Composes phases 95/97/98.
- Module doctest in autodiff/mod.rs: `grad` vs `vjp` on `f(x)=Σ sin(x_i)·exp(x_i)` (analytic-checked) — demonstrates the gradient API, runs under `cargo test --doc`.
- API surface: the unified entry points + all new generic fns are already re-exported (verified). No new public API needed; this phase is the integration proof + doctest.
- Executed inline (small; conserving context). No new dependency.
</decisions>

<code_context>
- Entry points: forward.rs (grad/jacobian/diff/directional_derivative), reverse.rs (vjp); re-exported autodiff/mod.rs:96-97 + prelude:21.
- Broadened subset re-exported at crate root + prelude across phases 95-98.
- Integration tests auto-discovered from fdars-core/tests/*.rs.
</code_context>

<deferred>
- REL-01 (version bump, CHANGELOG, 1.0-checklist, release gates) → Phase 100.
</deferred>
