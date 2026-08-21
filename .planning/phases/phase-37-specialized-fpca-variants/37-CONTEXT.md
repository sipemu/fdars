# Phase 37: Specialized FPCA Variants - Context

**Gathered:** 2026-08-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the specialized FPCA variants that `fdapace`/`refund` expose but fdars was missing, as additive, `Result`-returning, crate-root-re-exported public entry points that consume column-major `FdMatrix` and return structured numeric output:

- `fpca_der` — FPCA of curve derivatives
- `fsvd` — functional SVD / cross-FPCA between two functional samples
- `cross_covariance` — cross-covariance surface between two samples
- `dynamical_correlation` — scalar dynamical/functional correlation
- a sandwich-smoother / sparse-SVD (`ssvd`) FPCA path

All reuse the shipped dense FPCA (`fdata_to_pc_1d`) + `covariance.rs`. **Zero changes to existing public signatures**, no new crate dependency. Numeric outputs only — plotting/rendering out of scope.
</domain>

<decisions>
## Implementation Decisions

### Derivative FPCA (`fpca_der`)
- Derivative estimator: reuse the existing `fdata::deriv_1d` finite-difference helper (no new estimator).
- Decompose the **differentiated curves**: differentiate each curve first, then run `fdata_to_pc_1d` on the derivatives — matches the `fdapace::FPCAder` "derivatives of the process" convention.
- Expose a `nderiv` parameter (default 1), delegating order handling to `deriv_1d`.
- Result type: reuse the existing `FpcaResult` struct (its loadings/scores are understood to be of the differentiated process). No new struct.

### Functional SVD (`fsvd`) + `cross_covariance`
- `fsvd` inputs: two **paired** samples X (n×p) and Y (n×q) with matched sample size n, each with its own argvals grid.
- Weighting: weight the cross-covariance by √(Simpson integration weights) on both grids before thin-SVD, then rescale the resulting singular functions back to unit functional (L2) norm.
- Sign convention: deterministic — fix signs so the largest-magnitude element of each left singular function is positive (mirrors the existing `fix_svd_signs` pattern).
- `cross_covariance`: sample-centered empirical estimator with 1/(n−1) divisor, C(s,t) = 1/(n−1) Σ (xᵢ(s)−x̄(s))(yᵢ(t)−ȳ(t)), returned as a p×q `FdMatrix` symmetric-in-construction over the two argument grids.

### Dynamical Correlation (`dynamical_correlation`)
- Definition: Dubin–Müller / `fdapace::DynCorr` — standardize each curve (subtract its functional mean, divide by integrated-L2 functional sd), then take the integrated inner product averaged over the sample.
- Documented range: [−1, 1]; ≈1 for perfectly co-varying samples, ≈0 for independent samples.
- Grid handling: require X and Y to share the same argvals grid (same-domain definition); mismatched grids return `FdarError`.
- Return: scalar `f64`.

### Sandwich-smoother / ssvd path + packaging
- ssvd approach: smooth the empirical covariance surface with a sandwich (S·Cov·Sᵀ) then eigendecompose, reusing `covariance.rs` machinery — an alternative to the raw thin-SVD decomposition.
- Smoother choice: reuse existing kernel smoothing already available (no new penalized-spline dependency, **no new crate**).
- Dense-limit guarantee: with ~zero smoothing bandwidth, the ssvd path must agree with `fdata_to_pc_1d` within a documented tolerance (inline `#[cfg(test)]` test).
- Module placement: put all five variants in a **new `fpca_variants.rs`** module (keeps `regression.rs` under the ~500-line factoring guideline), crate-root re-exported.

### Claude's Discretion
- Exact tolerance constants for the synthetic-reconstruction tests, the specific kernel/bandwidth default for the ssvd smoother, and internal helper structure are at Claude's discretion, guided by codebase conventions and the R baselines.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `regression::fdata_to_pc_1d(data, ncomp, argvals) -> Result<FpcaResult, FdarError>` — the dense FPCA engine to reuse (SVD via nalgebra, Simpson-weighted inner product, sign fixing).
- `FpcaResult { singular_values, rotation (m×ncomp loadings), scores (n×ncomp), mean, centered, weights }` — the result struct to reuse for `fpca_der`.
- `fdata::deriv_1d(data, argvals, nderiv) -> FdMatrix` — finite-difference derivative for `fpca_der`.
- `helpers::simpsons_weights(argvals)` / `simpsons_weights_2d(...)` — integration weights for weighting + functional norms.
- `fdata::center_1d` / `center_columns` — mean-centering pattern used by `fdata_to_pc_1d`.
- `fdata::functional_covariance(data)` and `covariance.rs` (`CovKernel`, `covariance_matrix`) — covariance surface + kernel smoothing for the ssvd sandwich.
- `pace_fpca::{pace_fpca, PaceFpcaResult}` — reference for how FPCA-family results carry mean/eigenvalues/eigenfunctions/scores in this codebase.

### Established Patterns
- Column-major `FdMatrix` (`row + col*nrows`); all public fns return `Result<T, FdarError>`; dimension checks at entry (empty matrix, argvals length, sample-size match).
- SVD routed through `nalgebra::SVD` after `to_dmatrix()`; deterministic sign fixing (`fix_svd_signs`).
- Crate-root re-exports in `lib.rs` (e.g. `pub use regression::{fdata_to_pc_1d, FpcaResult}`), one `pub use` line per module.
- Inline `#[cfg(test)] mod tests` with synthetic data + documented tolerances; `#[must_use]` on expensive computations.

### Integration Points
- New module declared in `lib.rs` (`pub mod fpca_variants;`) + a `pub use fpca_variants::{...}` re-export block.
- Reuses `regression`, `covariance`, `fdata`, `helpers` — no new module dependencies, no new crate.

</code_context>

<specifics>
## Specific Ideas

- R baselines matched by **capability**, not exact signatures: `fdapace` (FPCAder, FSVD, GetCrCov, DynCorr/FCCor) and `refund` (fpca.sc sandwich, fpca.ssvd). Document any divergence from the R baseline in rustdoc (as prior milestones did).
- Invalid inputs must return `FdarError` (never panic): empty matrix, mismatched argvals vs values, mismatched sample sizes between the two samples for `fsvd`/`cross_covariance`/`dynamical_correlation`, `ncomp` out of range, degenerate columns.
- Full gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must stay green; use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for build/doctest linking to avoid /tmp exhaustion.

</specifics>

<deferred>
## Deferred Ideas

- Plotting/rendering of FPCA loadings or cross-covariance surfaces (out of milestone scope — numeric outputs only).
- SPARSE-01 (FACE sparse covariance) is Phase 38, a separate disjoint area (`irreg_fdata/`).

</deferred>
