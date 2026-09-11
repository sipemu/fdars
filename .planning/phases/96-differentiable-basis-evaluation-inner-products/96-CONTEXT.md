# Phase 96: Differentiable Basis Evaluation & Inner Products - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Make **B-spline / Fourier basis evaluation** generic over `Scalar` and differentiable through its evaluation points, with f64 numerics preserved bit-for-bit (family 1 of DIF-F2 / DOP-01). The functional inner products (`inner_product`, `inner_product_l2`) were **already generalized in Phase 95** — so this phase's genuinely-new work is basis evaluation plus a combined basis-eval → inner-product objective whose gradients are finite-difference-checked at both forward-mode `Dual` and reverse-mode `Var`.

**In scope:** In-place generalization over `T: Scalar` of the core basis evaluators — `bspline_basis_from_knots` and its `pub(super)` helpers `evaluate_order_zero` + `bspline_recurrence_step` (basis/bspline.rs), and `fourier_basis_with_period` (basis/fourier.rs) — with the **evaluation points `t` as the differentiable input** (`t: &[T]`). f64-parity tests + a combined basis-eval/inner-product objective FD-checked at `Dual` and `Var`.

**Out of scope (deferred):** The auto-deriving wrappers `bspline_basis` / `fourier_basis` stay f64-only (they derive `t_min`/`t_max`/`period` from `t`, which would require a `T::to_f64` — NOT adding one). Knot / period *construction* (`construct_bspline_knots`) stays f64. Regression prediction + roughness penalties → Phase 97. Depth + curve distances → Phase 98. The functional inner products themselves — done in Phase 95; not re-touched here (only *exercised* in the validation objective). `FdMatrix` genericization — out of milestone scope.
</domain>

<decisions>
## Implementation Decisions

### Differentiation semantics (roadmap-literal)
- **The differentiable input is the evaluation points `t`** — `t: &[T]`. This is the literal reading of "basis evaluation generic over `Scalar`" and enables differentiating a downstream objective w.r.t. where the basis is evaluated (e.g. a warping/registration parameter that shifts the grid). Knots, period, `nbasis`, `order` stay `f64`/`usize` (fixed basis structure, not curve observations).
- The complementary "differentiate w.r.t. curve data / coefficients" path is **already supported** — the basis matrix at `f64` times a generic (`Dual`/`Var`) curve flows through Phase 95's generic inner products. Phase 96 does not need to re-enable it; the validation objective can exercise both.

### Generalization pattern & scope (continues Phase 95's locked decision)
- **In-place generic `<T: Scalar>`** (NOT `_generic` companions), consistent with Phase 95. Use plain `<T: Scalar>` — Rust 1.97 rejects a literal `= f64` default on free functions (`invalid_type_param_default`, future-incompatible, confirmed in Phase 95); non-breaking holds via argument inference `T = f64`.
- **Generalize the core evaluators that take EXPLICIT f64 knots/period** (so `t` can be `T` with a clean boundary):
  - `bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order: usize) -> Vec<T>` (basis/bspline.rs:62)
  - `evaluate_order_zero<T: Scalar>(t_val: T, knots: &[f64], t_max_knot_idx: usize) -> Vec<T>` (bspline.rs:20, `pub(super)`)
  - `bspline_recurrence_step<T: Scalar>(b: &[T], knots: &[f64], t_val: T, k: usize) -> Vec<T>` (bspline.rs:37, `pub(super)`)
  - `fourier_basis_with_period<T: Scalar>(t: &[T], nbasis: usize, period: f64) -> Vec<T>` (basis/fourier.rs:42)
- **Keep the auto-deriving wrappers `bspline_basis` and `fourier_basis` f64-only** — they compute `t_min`/`t_max`/`period` by folding over `t`, which needs f64 bounds. Do NOT add `T::to_f64` to the `Scalar` trait; do NOT generalize these wrappers. Generic callers pass explicit knots/period (via the generic core entry points).
- Every existing f64 caller (`helpers.rs` spline_interpolate, `basis/pspline.rs`, `basis/projection.rs`, `seasonal/strength.rs`, `elastic_regression/scalar_on_shape.rs`) must compile unchanged — they call the generic entry points at inferred `T = f64`.

### f64 / generic boundary mechanics (from Phase 95)
- **Knots/period stay f64.** In the B-spline recurrence, knot differences `d1 = knots[j+k-1]-knots[j]`, `d2 = knots[j+k]-knots[j+1]` are computed in f64; the division numerator `(t_val - knots[j])` mixes `T` and f64. **Fold the f64 sub-expression before the single lift** for bit-parity: e.g. compute the f64 `d1` first, then `(t_val - T::from_f64(knots[j])) / T::from_f64(d1)` (or precompute an f64 factor and multiply once). Preserve the existing repeated-knot guard (`d1.abs() > 1e-10 else 0.0` → `T::zero()`).
- **Partition of unity:** `evaluate_order_zero`'s single indicator `1.0` becomes `T::one()`; the zero fills become `T::zero()`.
- **Span-search comparisons** (`t_val >= knots[j] && t_val < knots[j+1]`) use value-only `PartialOrd` on `T` — correct for `Dual`/`Var` (a discrete control-flow decision, not differentiated), exactly like soft-DTW's DP.
- **Fourier phase:** `x = 2π·(t_i - t_min)/period` with `t_i: T`, `t_min`/`period` f64. Precompute the f64 scale `2π/period`, then `x = T::from_f64(scale) * (t_i - T::from_f64(t_min))`; DC term `T::one()`; harmonics `(T::from_f64(freq as f64) * x).sin()/.cos()`. `t_min` here is a fixed f64 grid-origin passed/derived once (the generic `_with_period` takes `t_min` implicitly as it does today — keep it f64; if it is currently derived from `t` inside `_with_period`, derive it from the f64 primal or accept it as an explicit f64 param — Claude's discretion, but it MUST stay f64).

### Tests & validation
- **f64 parity:** `bspline_basis_from_knots::<f64>` and `fourier_basis_with_period::<f64>` reproduce current numerics bit-identically (or within the existing 1e-10 partition-of-unity tolerance the current tests use). Keep/extend the existing basis tests (basis/tests.rs: dimensions, partition-of-unity, non-negativity, boundary, DC-constant, sin/cos range).
- **Autodiff FD checks:** a combined objective — evaluate the basis at `t`, form an inner product (Phase 95's generic `inner_product`/`inner_product_l2`) against a fixed curve, sum to a scalar — differentiated w.r.t. `t`, gradients match central finite differences (tol 1e-6, h per Phase 94/95 convention) at BOTH `Dual` and `Var` (via `vjp`).
- Mirror the Phase 94/95 tiered pattern; RED-first per the project's TDD-ish executor flow.

### Claude's Discretion
- Exact placement of the combined validation objective/test (basis/tests.rs vs a new inline module), whether `evaluate_order_zero` returns `T::one()` vs `T::from_f64(1.0)` (equivalent; prefer `T::one()`), how `t_min` is threaded into `fourier_basis_with_period` if it is currently t-derived, and the precise f64-fold form in the recurrence — all Claude's discretion, guided by the Phase 95 kernels and the no-new-dependency / non-breaking constraints.
</decisions>

<code_context>
## Existing Code Insights

### Generalization targets (basis eval — NEW work this phase)
- `bspline_basis_from_knots(t: &[f64], knots: &[f64], order: usize) -> Vec<f64>` — basis/bspline.rs:62. Drives `evaluate_order_zero` (bspline.rs:20) then `bspline_recurrence_step` (bspline.rs:37) per point; column-major flatten.
- `fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64>` — basis/fourier.rs:42. DC + sin/cos harmonics.
- Repeated-knot guard: `if d1.abs() > 1e-10 { ... } else { 0.0 }` (bspline.rs:37 body). Preserve.

### Already generic (Phase 95 — REUSE, do not re-touch)
- `inner_product<T: Scalar>(c1: &[T], c2: &[T], argvals: &[f64]) -> T` — utility.rs.
- `inner_product_l2<T: Scalar>(psi1: &[T], psi2: &[T], time: &[f64]) -> T` — warping.rs.
- `Scalar` trait — autodiff/mod.rs:38–85. `Dual` (forward.rs), `Var`/`vjp` (reverse.rs).

### Stay f64 (do NOT generalize)
- `bspline_basis(t: &[f64], nknots, order)` — bspline.rs:100 (auto-derives knots from t bounds).
- `fourier_basis(t: &[f64], nbasis)` — fourier.rs:22 (auto-derives period from t bounds).
- `construct_bspline_knots(t_min: f64, t_max: f64, nknots, order)` — bspline.rs:4.

### Downstream callers (must compile unchanged at inferred T=f64)
- `helpers.rs` spline_interpolate (~520, ~671); `basis/pspline.rs` (~168); `basis/projection.rs` (~111); `seasonal/strength.rs` (~52); `elastic_regression/scalar_on_shape.rs` (~116).

### Tests
- `basis/tests.rs` — B-spline: dimensions/partition-of-unity(1e-10)/non-negative/boundary; Fourier: dimensions/DC-constant/sin-cos-range/period. Single-point edge case (~604). Mirror for parity; add Dual/Var FD objective test.

### Module structure
- `basis/mod.rs`: submodules `bspline`, `fourier` are `pub`; re-exports `bspline_basis`, `bspline_basis_from_knots`, `construct_bspline_knots`, `fourier_basis`, `fourier_basis_with_period`. Crate-root re-exports at lib.rs:678–680.

</code_context>

<specifics>
## Specific Ideas

- Do NOT re-generalize the functional inner products — Phase 95 already did. Phase 96 = basis evaluation + a combined objective that exercises basis-eval feeding an inner product, FD-checked at Dual and Var.
- Keep the f64 auto-deriving wrappers untouched; generalize only the explicit-knots/period core evaluators. This avoids adding `T::to_f64` to `Scalar`.
- Bit-identical f64 parity is the non-breaking guard — fold f64 knot/period arithmetic before the single `T::from_f64` lift; `f64::from_f64` is the identity.
</specifics>

<deferred>
## Deferred Ideas

- Generalizing the auto-deriving `bspline_basis`/`fourier_basis` wrappers (would need a `Scalar::to_f64` or a bounds-extraction step) — deferred; not needed for DOP-01.
- Differentiating w.r.t. knots/period (adaptive basis tuning) — out of scope; basis structure is a fixed design choice.
- Regression prediction + roughness penalties → Phase 97 (DOP-02/03). Depth + curve distances → Phase 98 (DOP-04).
</deferred>
