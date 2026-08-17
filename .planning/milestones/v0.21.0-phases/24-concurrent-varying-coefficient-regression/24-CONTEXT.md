# Phase 24: Concurrent / Varying-Coefficient Regression - Context

**Gathered:** 2026-08-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a **dense** functional concurrent (varying-coefficient) regression as a new public
entry point in `fdars-core/src/concurrent_regression.rs`, re-exported at the crate root.
The model relates a functional response y(t) to one or more functional predictors x_k(t)
evaluated on the *same shared grid*, recovering a smooth time-varying coefficient curve
β(t) (plus a time-varying intercept β₀(t)). Estimation is penalized pointwise / local-linear
least squares over the dense grid, reusing `smoothing.rs` kernels. Additive/non-breaking:
zero changes to existing public signatures.

**Explicitly out of scope:** the sparse/PACE variant (deferred — no FPCA-01 dependency),
plotting/visualization of β(t), and any change to existing regression APIs.

</domain>

<decisions>
## Implementation Decisions

### Estimation Method & Smoothing Convention
- β(t) estimated by **pointwise OLS per grid column, then smoothing the resulting β(t)
  sequence** with `smoothing.rs` kernels (two-step pointwise-then-smooth; honors the
  "reuse smoothing.rs kernels" + "pointwise/local-linear LS" mandate).
- Roughness penalty is parameterized as the **kernel bandwidth** fed to the smoother
  (larger bandwidth → demonstrably smoother β(t); satisfies the monotone-smoothness
  success criterion). Reuses `local_linear` / `smoothing_matrix_nw` directly.
- Include a **time-varying intercept β₀(t)** by default (standard varying-coefficient
  model; mirrors `FofResult.intercept`).
- Default smoothing kernel is **"gaussian"** (the default across `smoothing.rs` doctests).

### API Shape & Signature
- Public entry-point name: **`concurrent_regression`** (matches module name; `fof_regression`
  naming precedent).
- Multiple predictors passed as **`predictors: &[FdMatrix]`** (each n×m on the shared grid;
  single predictor = 1-element slice). No separate overloads.
- **Flat params** — `(response, predictors, argvals, bandwidth, kernel)` — mirroring
  `fof_regression` / `smoothing.rs` flat style (few params, no config struct).
- `argvals: Option<&[f64]>` — uniform 0..1 grid computed if `None` (project-wide convention).

### Result Struct Contents
- `beta_curve`: **`FdMatrix`, rows = predictor coefficients, cols = grid points** (p × m;
  column-major convention; scales to any predictor count).
- **Separate `intercept: Vec<f64>`** field for β₀(t); `beta_curve` holds only predictor
  coefficients (mirrors `FofResult`).
- Result fields: the mandated **`{ beta_curve, fitted, residuals }`** plus **`intercept`**
  and **`argvals`** (metadata for reproducibility). No R² diagnostics this phase (scoped minimal).
- `fitted` / `residuals`: **`FdMatrix` (n×m)** matching the response layout;
  `residuals == response − fitted` pointwise.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `smoothing.rs`: `local_linear(x, y, x_new, bandwidth, kernel)`,
  `smoothing_matrix_nw(x, bandwidth, kernel)`, `nadaraya_watson`, `local_polynomial`,
  all `Result<Vec<f64>, FdarError>`-returning with gaussian/epanechnikov kernels and
  bandwidth/empty-input validation already implemented — direct reuse for β(t) smoothing.
- `fof_regression.rs` (`FofResult`): closest structural analog — carries `intercept`,
  `beta_surface` (FdMatrix), `fitted` (FdMatrix), `residuals` (FdMatrix). Mirror its
  field conventions and column-major layout.
- `FdMatrix` (`src/matrix.rs`): column-major storage, `data[(i, j)]` = obs i at point j,
  row helpers (`row_to_buf`, `row_dot`). Rows = observations, cols = grid points.

### Established Patterns
- All public fns return `Result<T, FdarError>`; dimension checks at entry
  (`InvalidDimension`), parameter-range checks (`InvalidParameter`, e.g. bandwidth > 0).
- Public result types derive `Debug, Clone, PartialEq`; `#[non_exhaustive]` on public
  result structs; conditional serde via `cfg_attr`.
- Inline `#[cfg(test)] mod tests` per module; `uniform_grid(n)` from `test_helpers.rs`.
- Crate-root re-export in `lib.rs` (`pub mod` + `pub use`).

### Integration Points
- Add `pub mod concurrent_regression;` and `pub use concurrent_regression::{...}` in
  `src/lib.rs` (alongside `fof_regression` / `smoothing` re-exports).
- Reuse `helpers.rs` uniform-grid computation for the `argvals: None` default.

</code_context>

<specifics>
## Specific Ideas

- Recovery test (SC2): synthetic data from a known β(t) with low noise — recovered
  `beta_curve` reproduces the true coefficient curve within a stated tolerance.
- Monotone smoothness test (SC3): increasing bandwidth yields a demonstrably smoother
  `beta_curve` (roughness/curvature check).
- Consistency test (SC4): `residuals == response − fitted` pointwise; invalid inputs
  (mismatched grids/dimensions, empty data) return the appropriate `FdarError`, no panic.
- R baselines matched by capability (`fdaconcur` local-linear convention), not exact
  R signatures; document the chosen pointwise-then-smooth convention in rustdoc.

</specifics>

<deferred>
## Deferred Ideas

- Sparse/PACE kernel-weighted concurrent-regression variant (needs FPCA-01 PACE infra).
- R²(t) / overall R² and other GLM-style diagnostics on the result struct.
- λ difference-penalty (basis-penalty / refund-style) estimation alternative.

</deferred>
