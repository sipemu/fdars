# Phase 35: Basis-System Completions - Context

**Gathered:** 2026-08-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver REP-01 — complete the basis-system family and representation layer. Scope: `monomial_basis`, `exponential_basis`, `power_basis`, and a named `polygonal_basis` (piecewise-linear) factory in `basis/`, each with a penalty matrix; a composable `MultiFunData` multivariate/multi-domain container in a new `multi_fdata.rs`; a composable `Lfd`/linear-differential-operator object; and a `principal_differential_analysis` (PDA, linear-ODE estimation) estimator. Additive/non-breaking — the existing `bspline`/`fourier`/`constant` bases and `smooth_basis`/`pspline` penalties are UNTOUCHED (the constant basis is already handled by T-01). No new crate dependency. Numeric outputs only. Independent of Phases 34/36. R baseline: `fda` (monomial/exponential/power/polygonal bases, Lfd/PDA) / `funData` (multiFunData) / `tf`.

</domain>

<decisions>
## Implementation Decisions

### Basis factory API & return type
- New factories return `Result<BasisSystem, FdarError>` (a new struct bundling the column-major eval matrix + penalty matrix + metadata), per ROADMAP SC1 ("produces a basis-evaluation matrix over supplied argvals plus a penalty matrix"). This departs from the bare `-> Vec<f64>` of `bspline_basis`/`fourier_basis`/`constant_basis` deliberately — the new factories validate inputs and carry their penalty.
- Signatures: `monomial_basis(argvals, nbasis)`, `exponential_basis(argvals, rates: &[f64])`, `power_basis(argvals, exponents: &[f64])`, `polygonal_basis(argvals, knots)` (final param names at planner discretion).
- Eval-matrix layout: column-major flat `Vec<f64>` of shape `(n × nbasis)`, matching `bspline_basis`'s `basis[ti + j*n]` convention.
- Existing `bspline`/`fourier`/`constant` factories are LEFT UNTOUCHED (additive; zero signature changes).

### Penalty matrices
- Analytic penalty where a closed form exists (monomial/power: Gram of the `lfd_order`-th derivative of the polynomial/power basis); otherwise a numeric Gram of the `lfd_order`-th derivative on a fine grid, mirroring `smooth_basis::fourier_penalty_matrix` / `differentiate_basis_columns`.
- Default penalty is the 2nd-derivative (curvature) roughness; expose an `lfd_order` parameter consistent with existing penalty fns.
- Penalty is carried INSIDE the `BasisSystem` result (SC1: "plus a penalty matrix"), not a separate function.
- Polygonal (piecewise-linear) basis uses a 1st-order roughness penalty (the 2nd derivative is 0 a.e. for piecewise-linear), documented in rustdoc.

### MultiFunData container
- New `fdars-core/src/multi_fdata.rs`: `MultiFunData { components: Vec<FdComponent> }` where each `FdComponent` holds an `FdMatrix` plus its own `argvals` (multi-domain — components may live on different domains/grids).
- Invariant: equal observation count (rows) across all components; each component's argvals length must match its `FdMatrix` column count. Constructor `MultiFunData::new(components) -> Result<Self, FdarError>` validates both.
- Accessors: `n_obs()`, `n_components()`, `component(k)`, `argvals(k)` (Result or Option on out-of-range index at planner discretion).

### Lfd object & PDA
- `Lfd { coefs: Vec<Vec<f64>> }` — grid-sampled weight functions β₀(t)…β_{m-1}(t) for the linear differential operator `Lx = x^{(m)} + Σ_{k=0}^{m-1} βₖ(t)·x^{(k)}`. Method `apply(data, argvals) -> Result<FdMatrix, FdarError>` forms Lx via finite-difference derivatives. Constant-coefficient Lfd is the special case (each βₖ a length-1 or constant vector).
- `principal_differential_analysis(data, argvals, order) -> Result<PdaResult, FdarError>` — pointwise least-squares recovering the βₖ(t) of a linear ODE from the sampled curves and their finite-difference derivatives (the `fda` `pda.fd` capability by outcome).
- `PdaResult { coefficients: Vec<Vec<f64>>, ... }` — the recovered βₖ(t) weight functions (+ any diagnostics).
- Test: PDA recovers a known constant-coefficient linear ODE (harmonic oscillator x'' = −ω²x → β₀(t) ≈ ω², β₁(t) ≈ 0) within tolerance from synthetic solution curves.

### Claude's Discretion
- Exact struct/field names, the `BasisSystem`/`FdComponent`/`PdaResult` shapes, finite-difference derivative scheme, and the numeric-Gram grid density are at the planner/executor's discretion, guided by `smooth_basis.rs`/`pspline.rs` conventions and the `fda` reference.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `basis/bspline.rs` `bspline_basis(t, nknots, order) -> Vec<f64>` — the column-major flat-matrix factory convention (`basis[ti + j*n]`) the new factories mirror for their eval matrix.
- `basis/fourier.rs`, `basis/constant.rs` — more factory precedents (`constant_basis` from T-01).
- `smooth_basis.rs` `fourier_penalty_matrix(nbasis, period, lfd_order) -> Vec<f64>` + `differentiate_basis_columns(...)` — the numeric derivative-Gram penalty pattern to reuse.
- `basis/pspline.rs` `difference_matrix(n, order)` + `penalty = Dᵀ D` — difference-penalty precedent.
- `helpers.rs` `simpsons_weights`, `trapz` — quadrature for numeric Gram penalties and Lfd/PDA integration.
- `matrix.rs` `FdMatrix` (column-major) — the component type for `MultiFunData` and the return type for `Lfd::apply`.

### Established Patterns
- Column-major `FdMatrix` / flat `Vec<f64>` basis matrices; `data[(i,j)] = i + j*nrows`.
- `Result<T, FdarError>` on public fns (the new factories/estimators follow this; existing bare-`Vec` basis factories are the documented exception, left untouched).
- Public types derive `Debug, Clone, PartialEq` + conditional serde; `#[non_exhaustive]` on result structs.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.

### Integration Points
- New factories + `BasisSystem` registered in `basis/mod.rs` `pub use` block and re-exported at crate root.
- New `pub mod multi_fdata;` in `src/lib.rs` + crate-root re-export of `MultiFunData`.
- `Lfd`, `principal_differential_analysis`, `PdaResult` re-exported at crate root (module placement — `basis/` or a new `pda.rs` — at planner discretion).

</code_context>

<specifics>
## Specific Ideas

- Test discipline (from ROADMAP SC): each factory evaluates to its closed-form basis functions on hand-computed reference points within tolerance; `MultiFunData` preserves per-component argvals + enforces consistent n-obs; `Lfd` applies to data and PDA recovers a known linear ODE's coefficients within tolerance; the constant basis (T-01) stays untouched.
- Invalid inputs return `FdarError` (never panic): empty/mismatched argvals, non-monotone knots for polygonal, invalid degree/rate/exponent parameters, mismatched `MultiFunData` observation counts, singular PDA design.

</specifics>

<deferred>
## Deferred Ideas

- Full `tidyfun`-style tidy multi-representation vector semantics beyond the `MultiFunData` container — REP-02, deferred (L-effort).
- Retrofitting the existing `bspline`/`fourier`/`constant` factories to Result-returning — out of scope (breaking; they stay as-is).
- General non-linear differential operators / PDA with basis-expanded coefficients beyond pointwise-LS — future extension.

</deferred>
