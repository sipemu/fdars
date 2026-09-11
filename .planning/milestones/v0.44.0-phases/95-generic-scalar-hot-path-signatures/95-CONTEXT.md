# Phase 95: Generic Scalar Hot-Path Signatures - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Generalize the **shared numeric hot-path kernels** over the scalar type via in-place generic type parameters (`T: Scalar`, defaulted `= f64` where it type-checks), so every existing f64 call site, all 28 examples, the R + WASM binding surfaces, and the `--features serde` build compile **unchanged**. This is GEN-01 — the compile-time proof of non-breakingness AND the enabling substrate the DOP families (Phases 96/97/98) are written against.

**In scope:** In-place generalization of the SHARED, low-level numeric kernels that multiple DOP families call — L2 curve distance, functional inner products, and the weighted-dot / trapezoidal integration kernels they compose. Plus a non-breaking compile-gate proof and f64-parity + forward-mode `Dual` (and reverse-mode `Var`) flow tests on the generalized kernels.

**Out of scope (deferred to the DOP phase that owns them):** B-spline / Fourier basis evaluation (Phase 96), regression-prediction / roughness-penalty kernels (Phase 97 — note `project_scores_generic` already exists from Phase 94), functional-depth + curve-distance-beyond-soft-DTW algorithm kernels (Phase 98). `FdMatrix` row-op genericization / any `FdMatrix` data-structure redesign (`row_dot`/`row_l2_sq`/`row_to_buf`/`column` stay f64 — a generic `FdMatrix` is a major redesign, explicitly out of scope). Integration-weight *construction* (`simpsons_weights`) stays f64 — weights are quadrature constants, not curve data.
</domain>

<decisions>
## Implementation Decisions

### Generalization Pattern (user decision)
- **In-place generic `<T: Scalar = f64>`** — rewrite the existing shared-kernel functions to be generic over `T`, NOT add `_generic` companions. E.g. `l2_distance<T: Scalar>(c1: &[T], c2: &[T], w: &[f64]) -> T`. Matches the ROADMAP wording ("defaulted type params (`T = f64`)"). One function per kernel — no API-surface doubling — a cleaner substrate for the DOP phases to build on.
- **Why non-breaking:** for a free function, `T` is inferred from the `&[T]` arguments, so an existing call `l2_distance(&a, &b, &w)` with `a: Vec<f64>` infers `T = f64` and returns `f64` exactly as before. The `= f64` default is near-cosmetic for free functions (kept for intent/documentation and any turbofish-free ambiguous site). The **compile-gate over all 28 examples + R/WASM + serde is the authoritative non-breaking proof** — the executor must run it.
- **Residual risk (low, must be checked):** exotic call sites — passing a kernel as a `fn` pointer with a concrete `fn(&[f64],...)->f64` type, or an explicit type annotation that now needs a turbofish — could churn. The executor greps for such sites and the full example/binding compile-gate catches any breakage. If a genuinely unavoidable break appears, surface it (do not silently change a call site's public behavior).
- **Phase 94's two `_generic` companions** (`soft_dtw_distance_generic`, `project_scores_generic`) **stay as-is** — they are already generic and validated; converting them to in-place would be needless churn. Naming reconciliation (companion vs in-place) is a Phase 99 (API-01) concern, noted as deferred; do not rename them here.

### Scope — the "targeted" shared-kernel set
- **L2 curve distance** — `l2_distance` (`helpers.rs:56`): `(c1[i]-c2[i])^2 * w[i]` accumulate + `sqrt`. Generalize to `<T: Scalar>(c1: &[T], c2: &[T], weights: &[f64]) -> T`. Update `l2_distance_matrix` (`distance.rs:67`) to call it at `T = f64` (unchanged behavior).
- **Functional inner products** — `inner_product` (`utility.rs:34`) and `inner_product_l2` (`warping.rs:83`): `∫ c1·c2 dt` via Simpson's / trapezoidal weights. Generalize curves to `&[T]`, keep `argvals`/`time`/weights `f64`.
- **The weighted-dot / trapezoidal integration kernels** these compose (`trapz` and any shared accumulate helper in `helpers.rs`) — generalize the *accumulation over curve values* to `T` where the inputs are curve data; keep the quadrature-weight vector `f64`.
- Do NOT expand into basis/penalty/depth/regression algorithm functions — those are owned by their DOP phases.

### f64 / generic boundary (the invariant, from Phase 94)
- **Only curve INPUT DATA becomes `T`.** Integration weights, `argvals`/`time` grids, `gamma`, model parameters, and basis knots stay `f64`. Mixed arithmetic lifts an f64 constant via `T::from_f64(v)` only where it must combine with a `T` value. This matches `soft_dtw_distance_generic` (gamma f64) and `project_scores_generic` (mean/rotation/weights f64).

### Tests & non-breaking proof
- **f64 parity:** each generalized kernel at `T = f64` reproduces the pre-change numeric result bit-for-bit (or within 1e-12) — add parity tests, mirroring the Phase 94 f64-parity tier.
- **Autodiff flow:** each generalized kernel accepts `Dual` and `Var` and produces finite-difference-correct gradients (a light FD spot-check per kernel; the heavy validation is Phase 94's already-passing subset + the DOP phases).
- **Compile-gate (the GEN-01 deliverable):** `cargo build` of the crate, ALL 28 examples (`cargo build --examples` / per-example), the `--features serde` build, `cargo test`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and doctests must all be green with NO call-site edits outside the generalized function bodies. R/WASM binding surfaces: the WASM build target (`wasm32-unknown-unknown`) compiles; R bindings are an external package (`fdars-r`) — verify the fdars-core public signatures they depend on are unchanged (source-level check; cannot build the external R package here).

### Claude's Discretion
- Exact list of the shared accumulate/`trapz` helpers to lift, whether to introduce a tiny private generic helper vs generalizing each in place, module placement of any shared generic kernel, and the precise parity-test values — all Claude's discretion, guided by the scout's Tier-1 ranking and the no-new-dependency / non-breaking constraints.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets / Template
- **`Scalar` trait** — `fdars-core/src/autodiff/mod.rs:38–85` (12 methods + Copy/arith supertraits). The generalization target bound.
- **Template functions (Phase 94, already generic — mirror their f64/generic boundary):**
  - `soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S` — `metric/soft_dtw.rs:158` (gamma stays f64).
  - `project_scores_generic<S: Scalar>(curve: &[S], mean: &[f64], rotation: &FdMatrix, weights: &[f64], ncomp) -> Vec<S>` — `regression.rs:232` (model params stay f64).

### Generalization Targets (Tier-1 shared kernels)
- `l2_distance(curve1: &[f64], curve2: &[f64], weights: &[f64]) -> f64` — `helpers.rs:56`. Kernel: `Σ (c1-c2)² · w`, then `sqrt`. Called by `l2_distance_matrix` (`distance.rs:67`) and across clustering/depth/classification.
- `inner_product(curve1: &[f64], curve2: &[f64], argvals: &[f64]) -> f64` — `utility.rs:34` (Simpson's weights).
- `inner_product_l2(psi1: &[f64], psi2: &[f64], time: &[f64]) -> f64` — `warping.rs:83` (`trapz` of product).
- `trapz(...)` and shared accumulate helpers — `helpers.rs` (used by the above).

### Deferred (NOT this phase)
- `FdMatrix` row ops (`row_dot`/`row_l2_sq`/`row_to_buf`/`column`) — `matrix.rs`, f64-hardcoded, `#[inline]`; generic FdMatrix is a major redesign → out of scope.
- `simpsons_weights` (`helpers.rs:76`) — quadrature constants, stay f64.
- Basis eval (`basis/bspline.rs`, `basis/fourier.rs`) → Phase 96. Penalty/regression kernels → Phase 97. Depth kernels (`streaming_depth/fraiman_muniz.rs`) → Phase 98.

### Integration Points
- `autodiff/mod.rs` exports `Scalar`; import it where kernels are generalized.
- `l2_distance_matrix` and other f64 callers must keep compiling unchanged (they call at inferred `T = f64`).
- 28 `[[example]]` entries in `fdars-core/Cargo.toml`; WASM target `wasm32-unknown-unknown`; external `fdars-r` package (source-level signature check only).

</code_context>

<specifics>
## Specific Ideas

- The whole point of GEN-01 is the **compile-time non-breaking proof** — the executor MUST compile all 28 examples + the serde build + (where feasible) the WASM target, and confirm zero call-site churn outside the generalized function bodies. That gate IS the deliverable as much as the generalized signatures are.
- Keep the f64/generic boundary identical to Phase 94: curve data → `T`, everything else (weights, grids, params) → `f64`.
- Do not over-reach: this phase is the shared substrate, not the DOP algorithms. Resist generalizing basis/penalty/depth functions here — they belong to 96/97/98.
</specifics>

<deferred>
## Deferred Ideas

- Naming reconciliation between Phase 94's `_generic` companion functions and Phase 95's in-place generalization — Phase 99 (API-01) / unified gradient API.
- Generic `FdMatrix` (row ops over `T`) — major data-structure redesign, out of the milestone's additive/non-breaking scope.
- Basis-eval genericization → Phase 96; regression-prediction + roughness-penalty genericization → Phase 97; depth + curve-distance-beyond-soft-DTW → Phase 98.
</deferred>
