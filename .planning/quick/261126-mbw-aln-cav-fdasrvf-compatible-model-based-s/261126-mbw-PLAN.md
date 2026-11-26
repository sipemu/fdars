# Quick 261126-mbw: fdasrvf-compatible model-based shape CI (ALN-CAV, GH #71)

## Goal
`curve_shape_confidence_interval` (#54) bootstraps the Karcher mean. fdasrvf
`shape_CI(beta, a, no, Nsamp, mode, rotated, scale, lambda)` instead fits a Gaussian
model in the tangent space at the Karcher **median**, samples `Nsamp` shapes
(`sample_shapes`: shape PCA scores + random warps + random rotations) and returns the
`curve_boxplot` of the samples (`Q1a`, `Q3a`). Add that as an option, default unchanged.

## Public API (additive)
- `enum CurveShapeCiMethod { Bootstrap (default), ModelBased }` (`#[non_exhaustive]`).
- `CurveShapeCiConfig` new fields: `method`, `n_components` (fdasrvf `no`, 5),
  `n_samples` (`Nsamp`, 100), `n_warp_components` (fdasrvf `randomGamma` fixed 5).
- `CurveShapeCiResult::model: Option<CurveShapeModelCi>` (None for Bootstrap).
- `CurveShapeModelCi { q1a, q3a, boxplot, pca, scores, shape_samples, shape_samples_srvf,
  samples, samples_srvf, warps, rotations }`.

## Algorithm (port of fdasrvf 2.6.0 `shape_CI` / `sample_shapes`)
1. `curve_shape_karcher_median(curves, shape, max_iter, tol)`.
2. `curve_shape_pca(median, n_components)` (L² trapezoid PCA of the shooting vectors,
   equivalent to fdasrvf's SVD of `multivariate_karcher_cov` up to quadrature).
3. Per sample `i` (rng `seed + i`): `v = Σ_k z_k √λ_k φ_k` (zero-mean model, as fdasrvf);
   open: `q = exp_μ(v)`; closed: fdasrvf's 4-step geodesic shooting with closure
   projection + parallel translation. Curve `β = srvf⁻¹(q)` centred (betans, qns).
4. Warps (if `reparam`): horizontal FPCA of the median's warps (`n_warp_components`),
   `γ = psi_to_gam(exp_{μψ}(Σ z √λ φ))`, inverted, applied `β∘γ⁻¹` (fdasrvf randomGamma +
   invertGamma + group_action_by_gamma_coord).
5. Rotations (if `rotation`, planar only — fdasrvf errors for d ≠ 2): von Mises
   rejection sampler with mean/sd of the alignment rotation angles.
6. Box plot of the posed samples around the median (`curve_boxplot(samples, alpha = 1−a)`):
   reuse `boxplot_impl`.
Errors as fdasrvf: `scale = false` and rotation with d ≠ 2 are rejected.

## Validation
- Default method bit-identical (existing bootstrap test untouched; explicit equality test).
- Structure/dims; deterministic with fixed seed; thread-count independent.
- Distributional: with rotation/warps off, log maps of sampled shapes projected on the
  PCA components have variances ≈ λ_k and are in the component span (N = 2000).
- Golden vs fdasrvf 2.6.0 (`golden_shape_ci.R/.out`): RNG-independent quantities — median
  shape, distances to median, tangent covariance eigenvalues — within documented tolerance
  of fdasrvf's own resolution drift. RNG streams differ (R Mersenne/inversion vs rand StdRng).

## Tasks
1. model_ci.rs implementation + config/result fields + re-exports. 2. tests + golden.
3. CHANGELOGs, gates, commit `closes #71`. 4. SUMMARY.
