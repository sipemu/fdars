# Phase 42: Object-Data Fréchet Regression - Context

**Gathered:** 2026-08-22
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — all 4 grey areas accepted as recommended

<domain>
## Phase Boundary

Extend the shipped FRE-01 `frechet/` module (FRE-02) with pluggable non-density `MetricSpace`
backends and generic regression/ANOVA entry points that consume them — additively, with **zero
changes to existing public signatures**:

1. SPD covariance-matrix response space (Frobenius / power-α / log-Cholesky metrics) — FRE-02-01.
2. Correlation-matrix response space — FRE-02-02.
3. Spherical-data response space (geodesic exp/log + Karcher mean) — FRE-02-03.
4. Network response space (graph Laplacian + Frobenius) — FRE-02-04.
5. Point-process response space (intensity/count + L2) — FRE-02-05.
6. Generic global + local Fréchet regression over Euclidean predictors for at least one non-density
   backend, reusing the FRE-01 weight machinery — FRE-02-06.
7. Generic Fréchet-ANOVA over at least one non-density space, reusing the FRE-01 Tₙ machinery — FRE-02-07.

Numeric outputs only. No new crate dependency (reuse `nalgebra` + in-crate linear algebra). The
existing density backend (`WassersteinDensitySpace`, `frechet_global_reg`, `frechet_local_reg`,
`frechet_anova`) and their signatures are preserved bit-for-bit.

</domain>

<decisions>
## Implementation Decisions

### SPD & Correlation Matrix Backends (FRE-02-01/02)
- **SPD object representation:** flat column-major `Vec<f64>` + dimension `d` (a d×d matrix),
  matching the crate matrix convention.
- **SPD metrics:** all three from FRE-02-01, selected via an `SpdMetric` enum on `SpdMatrixSpace::new`:
  - **Frobenius:** `d(A,B)=‖A−B‖_F`; weighted Fréchet mean = weighted average `Σwᵢaᵢ / Σwᵢ`.
  - **Power-α:** matrix power `A^α` via `nalgebra::SymmetricEigen` (eigenvalues^α); distance
    `‖A^α − B^α‖_F / α`; weighted mean `(Σwᵢ Aᵢ^α / Σwᵢ)^{1/α}`.
  - **Log-Cholesky:** average in log-Cholesky coordinates (Cholesky factor with log-diagonal) then map back.
- **Correlation-matrix backend:** Frobenius distance on correlation matrices; weighted mean =
  weighted average re-normalized to unit diagonal (documented projection / divergence).
- **Eigendecomposition path:** reuse `nalgebra::SymmetricEigen` for matrix powers/logs — no new dep.

### Spherical, Network & Point-Process Backends (FRE-02-03/04/05)
- **Spherical space:** object = unit vector `Vec<f64>`; geodesic distance `arccos(⟨a,b⟩)` (clamped);
  exp/log maps; weighted Fréchet mean = intrinsic Karcher mean via gradient descent, initialized at
  the normalized extrinsic (weighted-average) mean, iter-capped.
- **Network space:** object = graph Laplacian (flat d×d); Frobenius distance; weighted mean =
  weighted average (stays a valid Laplacian on the convex cone of Laplacians).
- **Point-process space:** object = intensity/count vector on a grid; L2 distance; weighted mean =
  weighted average of intensities.
- **Iterative-mean validation:** fixed max-iters + tolerance (sensible defaults, config-overridable);
  validate non-empty inputs, matching dims, weight-length match; return `ComputationFailed` on
  non-convergence (never panic).

### Generic Solver Reuse (FRE-02-06/07)
- **Reuse strategy:** extract the Petersen–Müller global-linear-weight computation and the
  Dubey–Müller Tₙ / Fréchet-variance logic into `pub(crate)` helpers. The existing density
  entry points delegate to those helpers — **their public signatures stay untouched.**
- **Generic regression entry points:**
  `frechet_global_reg_space<S: MetricSpace>(space, predictors, responses: &[S::Object], xout) -> Result<Vec<S::Object>, FdarError>`
  plus a `frechet_local_reg_space` variant (kernel-weighted).
- **Generic ANOVA entry point:**
  `frechet_anova_space<S: MetricSpace>(space, objects: &[S::Object], labels, n_perm, seed) -> Result<FrechetAnovaResult, FdarError>`
  reusing the seeded-permutation Tₙ machinery (999-perm default, `StdRng::seed_from_u64(seed + k)`).
- **Predicted-object output:** `Vec<S::Object>` (one predicted object per `xout` row) — no
  density-specific `FdMatrix` in the generic path.

### API Surface, Result Types & Module Layout
- **Layout:** new `frechet/spaces/` submodule (or a cohesive `frechet/object_spaces.rs`) for the five
  backends; the generic solver functions are added to the existing `frechet/regression.rs` and
  `frechet/anova.rs`.
- **Backend types:** one struct per space — `SpdMatrixSpace`, `CorrelationMatrixSpace`,
  `SphericalSpace`, `NetworkSpace`, `PointProcessSpace` — each `impl MetricSpace`.
- **Metric-selection enum:** `SpdMetric { Frobenius, Power(f64), LogCholesky }` passed to
  `SpdMatrixSpace::new(...)`.
- **Re-exports:** all new space structs, the `SpdMetric` enum, and the generic regression/ANOVA
  functions re-exported from `frechet/mod.rs` and the crate root.

### Claude's Discretion
- Exact default max-iters/tolerance for the Karcher mean; the precise log-Cholesky coordinate
  convention; whether the local generic regression shares a helper with the density local path or a
  parallel one — all at the planner's discretion within these conventions.
- Whether backends live under `frechet/spaces/*.rs` (one file per space) or a single
  `object_spaces.rs` is a planner layout choice.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `frechet/space.rs`: `pub trait MetricSpace: Send + Sync { type Object; fn distance(..) -> Result<f64>; fn weighted_frechet_mean(objects, weights) -> Result<Object>; }` and the reference `WassersteinDensitySpace` impl to mirror. Also `signed_quantile_average` (density-specific).
- `frechet/regression.rs`: `frechet_global_reg`/`frechet_local_reg` — currently **density-specific** (take `&FdMatrix`, return `FrechetGlobalRegResult`/`FrechetLocalRegResult` with `predicted: FdMatrix`). The Petersen–Müller weight computation (predictor means, covariance Σ̂ + ridge, per-point linear weights) lives here and must be extracted into a `pub(crate)` helper the generic path shares.
- `frechet/anova.rs`: `frechet_anova` + private `compute_tn` (Tₙ statistic, per-group + pooled Fréchet variances, σ̂ₗ² estimator, seeded permutation). Extract the object-generic core so `frechet_anova_space` reuses it.
- `frechet/mean.rs`: `frechet_mean`/`frechet_variance` patterns.
- `nalgebra::SymmetricEigen` (used in `fts/acf.rs`, `regression.rs`) for eigen-based matrix power/log.
- Crate conventions: column-major flat matrices, `Result<T, FdarError>`, `#[non_exhaustive]` serde-gated `*Result` structs, `StdRng::seed_from_u64(seed + k)` per-thread seeding.

### Established Patterns
- `frechet/mod.rs`: `mod anova; mod mean; mod regression; mod space;` + explicit `pub use`. New backends module + generic fns follow the same shape; result structs defined in `mod.rs`.
- `FrechetAnovaResult` is already object-generic in its fields (statistic, p-values, group variances) — the generic ANOVA can return the same struct.

### Integration Points
- New public items re-exported in `frechet/mod.rs` and crate root `lib.rs` (there is an existing
  `pub use frechet::{...}` block to EXTEND, not replace).

</code_context>

<specifics>
## Specific Ideas

- R baseline: `frechet` 0.3.0 (`covariance` / `correlation` / `sphere` / `network` / `point process`
  responses, each with a Fréchet-regression + ANOVA analog). Match by capability; document any
  divergence (Frobenius-vs-affine-invariant correlation geometry, log-Cholesky convention, extrinsic
  Karcher init) in rustdoc.
- **Hand-computable test oracles to plan:** Frobenius Fréchet mean of identical SPD matrices recovers
  the matrix; power-α with α=1 equals Frobenius; geodesic distance of antipodal unit vectors = π;
  Karcher mean of two nearby unit vectors lies on the great-circle midpoint; network Laplacian mean of
  identical graphs recovers the graph; generic regression with constant response predicts that
  constant; `frechet_anova_space` on a homogeneous sample yields a non-significant statistic and is
  seed-reproducible.
- The existing density path must remain a passing regression (its tests unchanged) after the
  weight/Tₙ helper extraction.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Plotting/rendering of object-space Fréchet fits is an
explicit milestone Out-of-Scope item; FTS-03 spectral FTS was Phase 41.)

</deferred>
