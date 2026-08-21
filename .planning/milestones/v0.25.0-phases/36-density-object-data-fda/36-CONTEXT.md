# Phase 36: Density Object-Data FDA - Context

**Gathered:** 2026-08-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver DENS-01 — density-valued functional data analysis in a new `fdars-core/src/density_fda.rs`. Scope: the log-quantile-density (LQD) transform and its inverse, LQD-FPCA for probability densities (reuse `fdata_to_pc_1d` in LQD space, with fraction-of-variance-explained), a 1D Wasserstein Fréchet mean (quantile-average barycenter) of densities, and density normalization/regularization. Numeric outputs only — no plotting. Additive/non-breaking, `Result`-returning, crate-root re-exported, inline `#[cfg(test)]` tests. Zero changes to existing public signatures, no new crate dependency. Independent of Phases 34/35. The tractable 1D-density subset of R-audit Area 7 (simpler than the general Fréchet items FRE-01/FRE-02). R baseline: `fdadensity` (0.1.4).

</domain>

<decisions>
## Implementation Decisions

### Module layout & API surface
- New `fdars-core/src/density_fda.rs`; `pub mod density_fda;` in `src/lib.rs` + crate-root re-exports.
- Named public entry points: `lqd_transform`, `inverse_lqd`, `lqd_fpca`, `wasserstein_barycenter`, `normalize_density` (final names at planner discretion but this surface).
- Result types: `LqdFpcaResult` embedding the reused `FpcaResult` + an `fve: Vec<f64>` (+ optional density-space modes); the transforms return `Vec<f64>` / `FdMatrix` numeric output.
- Crate-root `pub use density_fda::{...}` for all entry points and result types.

### LQD transform conventions
- Input: densities sampled on a common evaluation grid (`argvals`). `lqd_transform(density, argvals)`.
- LQD definition (`fdadensity` convention): ψ(t) = −log f(Q(t)) on a uniform quantile grid t ∈ [0,1], via numeric CDF (trapz-integrated) → quantile inversion (interpolate F⁻¹) → log. Pin the exact numeric chain during research/planning.
- Quantile grid: uniform on [0,1] with configurable resolution (default ~101 points).
- `inverse_lqd(psi, t_grid)` reconstructs a normalized density on a target grid (exp + cumulative-integral back-map), always integrating to 1.

### LQD-FPCA & FVE
- Transform each density to LQD space on the common quantile grid → assemble the LQD `FdMatrix` → `fdata_to_pc_1d` on it.
- FVE = `cumsum(singular_values²) / sum(singular_values²)` — monotone non-decreasing, reaches 1 at full rank.
- `LqdFpcaResult { fpca: FpcaResult, fve: Vec<f64>, ... }` — embed the reused `FpcaResult`.
- Optionally inverse-LQD the mean±component to expose density-space variation modes (nice-to-have; document in rustdoc if deferred).

### Wasserstein barycenter, normalization & errors
- `wasserstein_barycenter` — 1D Wasserstein Fréchet mean = quantile-average: average the quantile functions (uniform weights, optional weight vector), invert to a density. Reduces to the input density on a single-density sample; lies quantile-between inputs on a two-density sample.
- `normalize_density(vals, argvals)` — scale a non-negative curve to integrate to 1 via Simpson/trapz (`helpers::simpsons_weights`/`trapz`); reject all-zero / negative.
- Return `FdarError` (never panic) on: negative or all-zero density, non-monotone/duplicate grid, argvals/values length mismatch, empty sample.
- Document any divergence from `fdadensity` (grid conventions, boundary handling, quantile-inversion interpolation) in rustdoc, per prior-milestone convention.

### Claude's Discretion
- Exact struct/field names, quantile-inversion interpolation scheme, quantile-grid default resolution, and whether density-space modes ship this phase are at the planner/executor's discretion, guided by the `fdadensity` reference and codebase conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `regression.rs` `fdata_to_pc_1d(data, ncomp, argvals) -> Result<FpcaResult, FdarError>` — the FPCA engine to reuse in LQD space; `FpcaResult { singular_values, rotation, scores, mean, centered, weights }` (FVE derives from `singular_values`).
- `helpers.rs` `simpsons_weights`, `trapz`, `cumulative_trapz` (CDF integration + normalization), `quantile_sorted(sorted, p)` (quantile ops for the Wasserstein barycenter).
- `matrix.rs` `FdMatrix` (column-major) — the LQD matrix passed to `fdata_to_pc_1d`, and density-set container.

### Established Patterns
- Column-major `FdMatrix`; rows = observations/curves (densities), columns = evaluation points.
- `Result<T, FdarError>` on all public fns; validation at entry.
- Public types derive `Debug, Clone, PartialEq` + conditional serde; `#[non_exhaustive]` on result structs.
- No new dependency; full clippy gate `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.

### Integration Points
- `pub mod density_fda;` in `src/lib.rs` + crate-root `pub use density_fda::{...}`.
- LQD-FPCA delegates to `fdata_to_pc_1d` (do not re-implement SVD).

</code_context>

<specifics>
## Specific Ideas

- Test discipline (from ROADMAP SC): LQD transform → inverse round-trips a valid density within tolerance; inverse always returns a normalized non-negative density; LQD-FPCA FVE is monotone non-decreasing and reaches 1 at full rank, with a single-mode synthetic family captured (near-)entirely by the leading component; the 1D Wasserstein barycenter equals the quantile-average (reduces to the input on a singleton sample, lies between two inputs); normalization forces integral-to-1. All numeric, seeded where randomness is used.

</specifics>

<deferred>
## Deferred Ideas

- General Fréchet regression / object-data statistics on arbitrary metric spaces (FRE-01/FRE-02) — deferred (L-effort); DENS-01 covers only the tractable 1D-density subset of Area 7.
- Multivariate density FPCA / general metric-space barycenters — out of scope (1D densities only, quantile geometry).
- Bandwidth-selection / smoothing subsystems for raw-sample density estimation — this phase takes densities-on-a-grid as input; kernel density estimation from raw samples is out of scope.

</deferred>
