# Phase 40: Fréchet / Object-Data Regression - Context

**Gathered:** 2026-08-22
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey areas proposed in tables, all accepted as recommended

<domain>
## Phase Boundary

Deliver metric-space (object-data) regression and statistics (FRE-01) as a new `fdars-core/src/frechet/` module: a metric-space abstraction (distance + weighted-Fréchet-mean solver) with a 1D-Wasserstein (density-response) backend, the Fréchet mean and variance of a sample, the 1D 2-Wasserstein distance, global and local (kernel-weighted) Fréchet regression over Euclidean predictors, density-response Fréchet regression, and a Fréchet ANOVA group-difference test. All entry points are `Result`-returning, crate-root re-exported, consume numeric inputs, and return structured numeric output. Share DENS-01's (`density_fda.rs`) quantile/Wasserstein machinery. Additive/non-breaking — zero changes to existing public signatures, no new crate dependency. Numeric outputs only. R baseline: `frechet` (Petersen, Müller, et al.).

</domain>

<decisions>
## Implementation Decisions

### Metric-Space Abstraction
- Provide a `MetricSpace` trait with `distance(&a, &b)` and `weighted_frechet_mean(objects, weights)`; the regression / statistics routines are generic over it.
- Provide ONE concrete backend this phase — `WassersteinDensitySpace` (1D-Wasserstein / density response). FRE-02 (deferred) adds more spaces (covariance/spherical/network/point-process).
- Objects in the density backend are densities on a shared strictly-increasing grid (`FdMatrix` rows), reusing `density_fda` conventions.
- The density-space weighted Fréchet-mean solver delegates to `density_fda::wasserstein_barycenter` (the quantile-average Wasserstein Fréchet mean, which already accepts weights).

### Fréchet Mean/Variance & Wasserstein Distance
- Fréchet variance = mean squared distance to the Fréchet mean (weighted variant where weights apply).
- The 1D 2-Wasserstein distance = L2 distance between quantile functions: `W₂(F,G) = (∫₀¹ (Q_F(t) − Q_G(t))² dt)^{1/2}`, computed by reusing `density_fda`'s density→quantile machinery.
- Reuse `wasserstein_barycenter` for the weighted barycenter rather than re-deriving the quantile average.

### Global & Local Fréchet Regression
- Global Fréchet regression uses the Petersen–Müller global linear weight scheme `wᵢ(x) = 1 + (x − x̄)ᵀ Σ⁻¹ (xᵢ − x̄)` (Σ = predictor covariance), then computes the weighted Fréchet mean of the response objects.
- Local Fréchet regression uses local-linear kernel weights (Gaussian kernel, user bandwidth parameter).
- Predictors are Euclidean, supplied as an `FdMatrix` (n × p).
- The density-response variant predicts a conditional density (the weighted barycenter under the regression weights) in 2-Wasserstein space — same regression entry points specialized to the density backend.

### Fréchet ANOVA, Determinism, Layout
- Fréchet ANOVA uses the Dubey–Müller statistic (between-group vs pooled Fréchet variance / dispersion contrast).
- p-value is permutation-based, per-thread seeded (`StdRng::seed_from_u64(seed + k)`), default 999 permutations (mirrors INF-01's convention). An asymptotic statistic is also returned.
- Invalid inputs return `FdarError` (never panic): empty sample, mismatched predictor/response counts, non-monotone or mismatched grids, invalid bandwidth, fewer than two groups for ANOVA, degenerate objects.
- Module layout: new `fdars-core/src/frechet/` directory with focused submodules (e.g. `space`/backend, `mean`, `regression`, `anova`) and a `mod.rs` that declares result structs and re-exports the public API; crate-root re-exports in `src/lib.rs`.

### Claude's Discretion
- Exact submodule filenames, trait method signatures, result-struct field names, kernel/bandwidth defaults, documented tolerance constants, and whether distance is computed via stored quantile functions or recomputed per call are at Claude's discretion, guided by the `frechet`/`fdadensity` references and codebase style.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (DENS-01 — `fdars-core/src/density_fda.rs`)
- `wasserstein_barycenter(density_matrix: &FdMatrix, argvals: &[f64], weights: Option<&[f64]>) -> Result<Vec<f64>, FdarError>` — the weighted quantile-average Wasserstein Fréchet mean of densities. This IS the density-space weighted-Fréchet-mean solver.
- `lqd_transform(density, argvals, n_quantile_pts) -> Result<Vec<f64>, FdarError>` and `inverse_lqd(...)` — density↔quantile machinery (LQD framework); internal density→quantile-function conversion is the basis for the 2-Wasserstein quantile-L2 distance.
- `normalize_density(vals, argvals)` — normalize a density to integrate to 1.
- `lqd_fpca`, `LqdFpcaResult` — density FPCA (not required this phase but same module conventions).
- Module validates strictly-increasing `argvals`, strictly-positive densities; mirror these validation patterns.

### Established Patterns
- Column-major `FdMatrix` (`src/matrix.rs`); all public fns return `Result<T, FdarError>`; validate at entry.
- Result structs derive `Debug, Clone, PartialEq`, serde-gated (`#[cfg_attr(feature = "serde", ...)]`), `#[non_exhaustive]`; `#[must_use]` on expensive computations.
- Permutation/bootstrap tests use per-thread seeded RNG `StdRng::seed_from_u64(seed + k)`, default 999 replications (INF-01 convention); feature-gated rayon via `parallel.rs` macros.
- Multi-file modules use a `mod.rs` barrel with explicit `pub use` (see `fts/`, `depth/`, `classification/`).
- `helpers.rs`: `simpsons_weights`, `trapz`, `NUMERICAL_EPS`; linear algebra via nalgebra (Σ⁻¹ for the global weight scheme).

### Integration Points
- New directory `fdars-core/src/frechet/` with `mod.rs` + submodules; wire `pub mod frechet;` (or `mod frechet; pub use frechet::{...}`) into `src/lib.rs` with crate-root re-exports.
- Reuse (do not modify): `density_fda.rs`, `helpers.rs`, `matrix.rs`.

</code_context>

<specifics>
## Specific Ideas

- R baseline is `frechet` (Petersen & Müller). Match by capability, not exact R signatures; document any divergence in rustdoc.
- Fréchet mean of an identical-object sample must have variance ≈ 0 (within tolerance), and variance must grow with dispersion.
- In the 1D-Wasserstein density space, the Fréchet mean must agree with DENS-01's `wasserstein_barycenter` within a documented tolerance.
- 2-Wasserstein distance must be 0 (within tolerance) for identical distributions and match a hand-computed shift/scale reference.
- Global/local Fréchet regression must track a known predictor→object relationship within a documented tolerance on synthetic data.
- Fréchet ANOVA must flag a genuine between-group difference and not flag a homogeneous sample.

</specifics>

<deferred>
## Deferred Ideas

- FRE-02: additional object-data Fréchet spaces (covariance/correlation matrices, spherical, network/graph-Laplacian, point-process) — separate v2 backlog item depending on this phase's solver framework.
- Non-density metric-space backends beyond 1D-Wasserstein.
- Plotting/rendering of Fréchet fits or regression surfaces (numeric outputs only).

</deferred>
