# Phase 43: Boosting / Bayesian Functional Regression - Context

**Gathered:** 2026-08-23
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey areas proposed in batch, all accepted as recommended

<domain>
## Phase Boundary

Deliver a new estimation subsystem `fdars-core/src/boosting_regression/` implementing REG-06: (1) component-wise gradient boosting with functional base-learners for a function-on-scalar response (boosted FOSR, one base-learner per iteration); (2) the same boosting framework for function-on-function predictor/response (boosted FoFR); (3) GAMLSS-style distributional functional regression modelling >1 distributional parameter (location + scale); (4) a Bayesian function-on-scalar regression Gibbs sampler producing coefficient posterior summaries (mean + credible bands); (5) FDboost-style stability selection over the boosting base-learners (per-learner selection frequencies + stable predictor set).

Additive/non-breaking only: `Result`-returning public fns, inline `#[cfg(test)]` tests, crate-root + prelude re-exports, zero changes to existing public signatures. Numeric outputs only — no plotting/rendering of boosting paths. No new crate dependency. R baseline: `FDboost` 1.1-4 + `refund` (matched by capability, not exact signatures; document divergences in rustdoc).

</domain>

<decisions>
## Implementation Decisions

### Boosting Core (REG-06-01, REG-06-02)
- Functional base-learner family: penalized B-spline base-learners, reusing `smooth_basis`/`FdPar` penalty-matrix machinery (`bspline_penalty_matrix`) — matches FDboost `bbs()`.
- Per-iteration base-learner selection: component-wise — select the single base-learner minimizing residual sum of squares per boosting iteration (FDboost standard).
- Step size ν (learning rate): fixed ν = 0.1 (FDboost default), configurable via `BoostingConfig`.
- Stopping rule: fixed `mstop` from config, with GCV/AIC tracked along the boosting path (early-stopping optional/deferred).
- Same boosting framework serves both boosted FOSR (function-on-scalar response) and boosted FoFR (function-on-function) base-learners.

### GAMLSS Distributional Regression (REG-06-03)
- Distribution family: Gaussian, modelling location μ(t) + scale σ(t) (the "location + scale" milestone target).
- Fitting scheme: component-wise boosting cycling over the distributional parameters (gamboostLSS style) — reuses the boosting core.
- Link functions: identity for μ, log for σ (guarantees positivity of scale).
- Output: per-parameter functional coefficients + fitted μ(t) and σ(t) + log-likelihood.

### Bayesian FOSR (REG-06-04)
- Sampler: Gibbs sampler on FPC-score coefficients (conjugate, deterministic + seeded via `StdRng::seed_from_u64`).
- Prior structure: Normal prior on coefficients + Inverse-Gamma on variances (conjugate, weakly-informative).
- Credible bands: pointwise credible bands from posterior quantiles.
- Output summaries: posterior mean β(t) + pointwise credible bands + retained thinned posterior draws.

### Stability Selection & API Packaging (REG-06-05)
- Resampling scheme: subsampling ⌊n/2⌋ without replacement, B resamples (Meinshausen–Bühlmann / FDboost default), seeded per replicate (`seed.wrapping_add(b)`).
- Selection output: per-base-learner selection frequencies + stable predictor set at threshold π (default 0.9).
- Config/result API: `BoostingConfig` builder struct (mstop, nu, basis spec, seed, …) + per-method Result structs following the existing `FosrResult` field convention (fitted, residuals, r_squared, coefficient structures).
- Module layout: folder `src/boosting_regression/` with submodules (e.g. boost_fosr, boost_fofr, gamlss, bayesian, stability) + `mod.rs` barrel — fits the heavy multi-plan scope and the ~500-line factoring convention.

### Claude's Discretion
- Exact submodule split, internal helper structure, per-struct field naming, and plan decomposition are at the planner's discretion, provided they follow the accepted decisions and existing conventions.
- Precise credible-band / ICL-style diagnostic numerics beyond posterior mean + pointwise bands are at implementer discretion (full mgcv/BayesX-grade diagnostics are explicitly out of scope).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdata_to_pc_1d(data, ncomp, argvals) -> Result<FpcaResult>` (`src/regression.rs:287`); `FpcaResult` fields: `singular_values, rotation, scores, mean, centered, weights`; methods `project()`, `reconstruct()`. Use for FPC-score dimension reduction in boosted FOSR and Bayesian FOSR.
- `fosr(data, predictors, lambda) -> Result<FosrResult>` (`src/function_on_scalar.rs:275`) and `FosrResult` (intercept, beta, fitted, residuals, r_squared_t, r_squared, beta_se, lambda, gcv) — the field/naming template for new boosting result structs.
- `fof_regression(x_data, y_data, x_argvals, y_argvals, ncomp_x, ncomp_y) -> Result<FofResult>` (`src/fof_regression.rs:59`) — double-FPCA → regress scores → reconstruct pattern to mirror for boosted FoFR.
- `fregre_lm(data, y, scalar_covariates: Option<&FdMatrix>, ncomp) -> Result<FregreLmResult>` (`src/scalar_on_function/mod.rs`) — FPC-based regression + optional scalar covariates.
- `smooth_basis`/`FdPar` (`src/smooth_basis.rs`): `bspline_penalty_matrix(argvals, nbasis, order, lfd_order)`, `smooth_basis_gcv(...)` — penalized B-spline base-learners + GCV.
- `simpsons_weights(argvals) -> Vec<f64>` (`src/helpers.rs:57`) — integration weights for functional inner products.
- `cholesky_factor`/`cholesky_solve`/`compute_xtx` (`src/linalg.rs`, `linalg` feature) — penalized normal-equation solves for base-learner fits and Gibbs conditional draws.

### Established Patterns
- RNG seeding: `StdRng::seed_from_u64(seed.wrapping_add(b as u64))` per replicate (`src/scalar_on_function/bootstrap.rs:89`) — apply to Gibbs sampler and stability-selection resampling for determinism.
- Parallelism: `iter_maybe_parallel!` / `slice_maybe_parallel!` macros (`src/parallel.rs`) gate rayon by feature; use for resample loops.
- Column-major `FdMatrix`; all public fns return `Result<T, FdarError>`.
- Tests: inline `#[cfg(test)] mod tests { use super::*; use crate::test_helpers::uniform_grid; ... }` with shape + validity + recovery + error-path assertions.

### Integration Points
- Register `pub mod boosting_regression;` in `src/lib.rs` (module list ~line 90) + crate-root `pub use boosting_regression::{...}` re-exports (~line 287) + add key result types to `src/prelude.rs`.

</code_context>

<specifics>
## Specific Ideas

- R baseline capability parity: FDboost `bbs()` penalized base-learners, component-wise `mstop`/`nu` boosting, gamboostLSS distributional boosting, and Meinshausen–Bühlmann stability selection (`stabsel`). Bayesian FOSR mirrors `refund`-style FOSR with a Gibbs sampler. Document any numeric divergence from the R baseline in rustdoc.
- Determinism is mandatory wherever randomness appears (Gibbs draws, stability-selection subsampling) — everything seeded.

</specifics>

<deferred>
## Deferred Ideas

- Variational Bayes (VB) alternative to Gibbs; simultaneous (rather than pointwise) credible bands; horseshoe/g-prior alternatives — all deferred; conjugate Gibbs + pointwise bands is the v1 scope.
- Line-search optimal boosting step and CV-selected `mstop` — deferred; fixed ν + config `mstop` with GCV/AIC path tracking is v1.
- Full mgcv/BayesX-grade sampler diagnostics (multiple chains, R̂, convergence tests) — explicitly out of milestone scope.
- Additional GAMLSS distribution families / shape parameters beyond Gaussian location+scale.

</deferred>
