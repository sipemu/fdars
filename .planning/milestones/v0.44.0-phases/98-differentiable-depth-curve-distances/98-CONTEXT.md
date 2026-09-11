# Phase 98: Differentiable Depth & Curve Distances - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Make ≥1 functional DEPTH measure and ≥1 CURVE DISTANCE (beyond soft-DTW) generic over `Scalar` and differentiable, f64 parity preserved (family 4 of DIF-F2 / DOP-04). FD-checked at `Dual` and `Var`.

**In scope:**
- **Depth (the genuine new work):** a new additive generic per-curve **modal depth** `modal_depth_generic<T: Scalar>(curve: &[T], reference: &FdMatrix, h: f64) -> T` = `(1/nref)·Σ_j exp(-0.5·(dist_j/h)²)` where `dist_j = sqrt(Σ_t (curve[t]-ref[j,t])² / n)`. Differentiable w.r.t. the object curve; reference + bandwidth stay f64. Modal is the ONLY smooth kernel-density depth (Gaussian kernel + L2, no rank/sort/partition ops).
- **Distance:** criterion #2 is **already satisfied** by `l2_distance<T: Scalar>` (helpers.rs, Phase 95) — a generic, differentiable curve distance beyond soft-DTW, with existing Dual/Var FD tests. Phase 98 documents this and does not re-generalize it. (soft_dtw + amplitude_at_warp are also already generic.)
- Tests: f64 parity (modal generic vs modal_1d), Dual + Var FD on modal depth; reference the existing l2_distance Dual/Var tests for the distance criterion.

**Out of scope / stays f64:** Fraiman-Muniz, band/MBD, random-projection/tukey, rpd — all rank/CDF/combinatorial (non-differentiable). `modal_1d`/`modal` public signatures unchanged. No new dependency.
</domain>

<decisions>
## Implementation Decisions
- **New additive `modal_depth_generic<T: Scalar>`** in depth/modal.rs (per-curve; differentiable input = the object curve `&[T]`). `modal_1d`/`modal` (batch, f64, FdMatrix) stay unchanged.
- **f64/generic boundary:** reference curves (via `reference[(j,t)]`), `h`, `n` stay f64, lifted via `T::from_f64` only where mixed with `T`. Match `modal_1d`'s exact arithmetic order for parity: `dist_sq = Σ_t (curve[t]-from_f64(ref))²`; `dist = (dist_sq / from_f64(n)).sqrt()`; `u = dist/from_f64(h)`; `kernel = (from_f64(-0.5) * (u*u)).exp()` (association matches `-0.5*(dist/h).powi(2)`); `depth = Σ_j kernel / from_f64(nref)`.
- **f64 parity tolerance:** modal_1d uses `f64::powi(2)` which is not bit-guaranteed equal to `u*u`, so parity is asserted within **1e-12** (not assert_eq!). Empty reference → `T::zero()`.
- Plain `<T: Scalar>` (no `=f64` default). Re-export `modal_depth_generic` at crate root + prelude (additive), beside `modal`.
- FD tolerance `1e-6*(1.0+fd.abs())`, h_fd=1e-6 (Phase 96/97 robust form), at Dual + Var(vjp).

## Claude's Discretion
- Exact test fixture; whether to also add a bonus generic Lp distance (SKIP — l2_distance covers the criterion; conserve scope).
</decisions>

<code_context>
- `modal_1d(data_obj, data_ori, h) -> Vec<f64>` — depth/modal.rs:18 (kernel at 32-38). `modal` dispatcher at 55 (unchanged).
- `l2_distance<T: Scalar>` — helpers.rs:66 (already generic; Dual/Var tests helpers.rs:1204-1300) → satisfies distance criterion.
- `Scalar` trait autodiff/mod.rs:38-85 (exp, sqrt, from_f64, arithmetic). `FdMatrix[(j,t)]` column-major indexing.
- depth/mod.rs re-exports `modal`; crate root + prelude re-export `modal`.
- depth/tests.rs:66 modal tests (structural: centrality/positivity).
</code_context>

<specifics>
- Modal depth is smooth (Gaussian + L2); the differentiable input is the object curve. This is the single differentiable depth in the crate.
- The distance criterion is already met by Phase 95's l2_distance — do not duplicate; document + reference its tests.
</specifics>

<deferred>
- Rank/CDF/combinatorial depths (FM, band, RP/tukey, rpd) — non-differentiable, deferred permanently.
- Bonus Lp/derivative distances — not needed; l2_distance suffices.
- Unified gradient API + composition demo → Phase 99.
</deferred>
