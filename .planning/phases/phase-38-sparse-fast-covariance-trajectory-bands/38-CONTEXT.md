# Phase 38: Sparse Fast Covariance & Trajectory Bands - Context

**Gathered:** 2026-08-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the FACE fast-sandwich sparse/irregular covariance tooling that `face`/`mfaces` expose but fdars was missing, as additive, `Result`-returning, crate-root-re-exported entry points in a new `fdars-core/src/irreg_fdata/face.rs`:

- `face_covariance` — FACE fast-sandwich covariance surface for sparse/irregular data
- `mface_covariance` — multivariate (`mfaces`) block covariance across multiple simultaneously-observed sparse variables
- a fitted-trajectory entry point returning per-curve continuous trajectories with pointwise confidence bands, integrated with the FACE path

All build on the existing `irreg_fdata::cov_irreg` (kernel-smoothed sparse covariance) and the shipped PACE `pace_fpca` (FPCA-01, which already yields BLUP fitted trajectories + pointwise bands). **Zero changes to existing public signatures**, **no new crate dependency**. Numeric outputs only.
</domain>

<decisions>
## Implementation Decisions

### FACE covariance (`face_covariance`)
- Sandwich construction: sandwich-smooth the `cov_irreg` surface (symmetric smoother S on both sides, S·Cov·Sᵀ), reusing the existing kernel-smoothing machinery — the SAME approach as Phase 37's `ssvd`. This is a kernel-sandwich approximation of `refund::face`'s penalized tensor-product spline FACE; document the divergence in rustdoc (capability match, not exact).
- Validity: symmetrize by construction and clip negative eigenvalues to 0 so the returned surface is a valid PSD covariance (documented).
- Signature: `face_covariance(ifd, grid, bandwidth) -> Result<FdMatrix, FdarError>` — mirrors `cov_irreg` but `Result`-returning with input validation (empty sample, non-monotone/mismatched argvals, invalid bandwidth → `FdarError`).
- Correctness: dense-limit test — on densely-sampled synthetic curves it recovers a known covariance surface within a documented tolerance.

### Multivariate `mface_covariance`
- Input: `&[IrregFdata]` — P variables observed on the same n subjects, one argvals grid per variable.
- Block layout: a (P·G)×(P·G) block covariance where diagonal blocks are the per-variable FACE covariance and off-diagonal blocks are the cross-variable covariance.
- Cross-block estimation: kernel-smoothed cross-covariance between paired sparse variables (extend the `cov_irreg` accumulation to cross-variable point pairs).
- Return: a struct carrying the block covariance matrix + per-variable grids + a block accessor (documented block layout). `#[non_exhaustive]`.

### Fitted trajectories + pointwise bands
- Engine: reuse `pace_fpca`'s BLUP scores → fitted trajectory → pointwise Gaussian band machinery (it already produces `fitted`, `fitted_lower`, `fitted_upper`).
- Band type: reuse `pace_fpca`'s alpha-controlled pointwise Gaussian bands.
- Return: reuse `PaceFpcaResult` (or a thin wrapper exposing fitted + lower/upper + grid). `face_covariance` remains available separately for the FACE surface.
- Correctness: dense-curve test — the fitted trajectory tracks the true curve within its pointwise bands within a documented tolerance.

### Packaging
- Module placement: new `fdars-core/src/irreg_fdata/face.rs`, re-exported via `irreg_fdata/mod.rs` and the crate root.
- Dependencies: no new crate — reuse `irreg_fdata::{kernels, smoothing, cov_irreg, IrregFdata, mean_irreg}` and `pace_fpca`.
- Result types: standard derives (`Debug, Clone, PartialEq`) + conditional serde + `#[non_exhaustive]`.
- Divergence docs: rustdoc documents FACE here as a kernel-sandwich approximation of `refund::face` / `mfaces` (matched by capability, not R's exact penalized-spline internals).

### Claude's Discretion
- Exact sandwich smoother kernel/bandwidth defaults, PSD-clipping eigenvalue threshold, tolerance constants for the dense-limit tests, and the precise wrapper-vs-reuse shape for the trajectory entry point are at Claude's discretion, guided by the `pace_fpca`/`cov_irreg` conventions and the R baselines.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `irreg_fdata::cov_irreg(ifd, s_grid, t_grid, bandwidth) -> FdMatrix` (`smoothing.rs:111`) — kernel-smoothed sparse-data empirical covariance surface; the base FACE sandwiches.
- `irreg_fdata::IrregFdata` (`mod.rs:38`) — sparse container (`offsets`, `argvals`, `values`; `from_lists`, `from_flat`, `n_obs`, `get_obs`, `obs_counts`).
- `irreg_fdata::mean_irreg` (`kernels.rs:58`), `integrate_irreg`, `to_regular_grid` (`smoothing.rs`), `KernelType` — mean/smoothing helpers.
- `pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult}` (`pace_fpca.rs`) — BLUP FPCA that ALREADY returns `fitted`, `fitted_lower`, `fitted_upper`, eigenfunctions, scores, on a work grid; `PaceFpcaConfig { ncomp, bandwidth, sigma2, work_grid, alpha }`. Reuse for the trajectory-band entry point.
- Phase 37's `ssvd` (`fpca_variants.rs`) — reference for the W^{1/2}·Cov·W^{1/2} sandwich + symmetric eigendecompose + sign-fix pattern, and for the PSD/eigenvalue handling.
- `fdars-core/src/covariance.rs` (`CovKernel`) — kernel definitions if a smoother matrix is needed.

### Established Patterns
- Column-major `FdMatrix`; all public fns return `Result<T, FdarError>`; dimension/validity checks at entry.
- `irreg_fdata` submodule split (`mod.rs`, `kernels.rs`, `smoothing.rs`, `tests.rs`); add `face.rs` and re-export via `mod.rs` `pub use` + crate root.
- `#[non_exhaustive]` public result structs; `#[must_use]` on expensive computations; nalgebra `symmetric_eigen` for PSD projection.

### Integration Points
- `irreg_fdata/mod.rs` re-exports + a crate-root `pub use` block for the new symbols.
- Reuses `cov_irreg`, `pace_fpca`, `mean_irreg` — no new module dependencies, no new crate.

</code_context>

<specifics>
## Specific Ideas

- R baselines matched by **capability**: `face` (FACE fast-sandwich covariance), `mfaces` (multivariate FACE), `fdapace` (trajectory bands). Document divergence from R's exact penalized-spline internals in rustdoc (as prior milestones did).
- Invalid inputs must return `FdarError` (never panic): empty sample, mismatched variable/observation counts for `mface_covariance`, non-monotone or mismatched argvals, degenerate/all-missing curves, invalid bandwidth.
- `cov_irreg` gives a kernel-smoothed empirical covariance, NOT the FACE sandwich specifically — FACE is the new sandwich estimator built on top of it.
- Full gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must stay green; use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for build/doctest linking.

</specifics>

<deferred>
## Deferred Ideas

- Plotting/rendering of covariance surfaces or trajectory bands (out of scope — numeric outputs only).
- Automatic σ² estimation for `pace_fpca` (already deferred upstream; caller supplies σ²).
- FPCA-02 (Phase 37) is a separate, already-completed disjoint area (`fpca_variants.rs`).

</deferred>
