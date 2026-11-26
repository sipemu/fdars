---
quick_id: 261126-mbw
status: complete
issue: 71
branch: cav-71
commits:
  - 1a4e5418 feat(quick-261126-mbw): fdasrvf-compatible model-based shape CI for R^d curves (closes #71)
---
# Summary: fdasrvf-compatible model-based shape CI (ALN-CAV, GH #71)

## Public API (additive, non-breaking)
- `CurveShapeCiMethod { Bootstrap (default), ModelBased }` (`#[non_exhaustive]`).
- `CurveShapeCiConfig` new fields: `method`, `n_components` (fdasrvf `no`, 5),
  `n_samples` (`Nsamp`, 100), `n_warp_components` (fdasrvf randomGamma's fixed 5).
- `CurveShapeCiResult::model: Option<CurveShapeModelCi>` (`None` for Bootstrap).
- `CurveShapeModelCi { q1a, q3a, boxplot, pca, scores, shape_samples, shape_samples_srvf,
  samples, samples_srvf, warps, rotations }` (fdasrvf `Q1a`, `Q3a`, `box`, `betans`, `qns`,
  `betas`, `qs`, `gams`, `R`). Re-exported from `alignment`, crate root, prelude (result).
- For ModelBased the generic result fields describe the model: `mean` = Karcher median,
  `bootstrap_means` = unposed samples, `bootstrap_distances` = their distances to the
  median, `radius`/bands from those.

## Implementation (`fdars-core/src/alignment/curve_shape/model_ci.rs`)
Port of fdasrvf 2.6.0 `shape_CI` -> `sample_shapes` -> `curve_boxplot`, reusing
`curve_shape_karcher_median`, `curve_shape_pca`, the karcher `exp_map`, `boxplot_impl`,
`horiz_fpca_from_alignment` + `exp_map_sphere`/`psi_to_gam`/`invert_gamma` (the generative
model's phase machinery). Open curves: one exact exp-map step; closed: fdasrvf's 4-step
shooting with closure projection + `parallel_translate`. Rotations: fdasrvf's von Mises
rejection sampler on the alignment angles (sd = 0 guarded -> mean angle; fdasrvf would
divide by zero). Errors like fdasrvf for `scale = false` and rotation with d != 2. Per-sample
RNG `seed + i`, parallel via `iter_maybe_parallel!` (thread-count independent).

## Validation (`model_ci_tests.rs`, 9 tests)
- Default is the unchanged bootstrap (model None; model-only knobs ignored; centre = mean).
- Structure, determinism (same seed equal, other seed differs), 1-thread == parallel.
- Warps are diffeomorphisms, rotations proper, samples centred unit-length; aligning a posed
  sample recovers its unposed shape (< 0.05).
- Distributional: N = 2000 samples, log maps projected on the model components have
  variances = lambda_k within 4 SE, ~0 means and correlations, lie in the component span,
  recover the drawn scores; d(sample, median) = ||v||.
- Closed contours: samples closed; distance = ||v|| up to the measured SRVF round-trip error.
- von Mises sampler: E cos = I1/I0(kappa) (asymptotic series), E sin = 0.
- **Golden vs fdasrvf 2.6.0** (installed into `~/.cache/R-lib` from CRAN; scripts
  `golden_shape_ci.R/.out`, `golden_shape_ci_moments.R/.out` in this dir), 6 open curves of
  #54's depth golden, T = 60, `rotated = FALSE`:
  - median shape: d(fdars median, fdasrvf median) = 0.019 (< 10% of mean spread 0.198), and
    fdars attains the smaller sum of distances;
  - covariance eigenvalues 1-4 within 2 x fdasrvf's own T 60->240 drift + 2% (observed
    4.8%, 5.8%, 21%, 2.4% vs drift 2.2%, 3.8%, 15%, 3.6%);
  - RNG-dependent sample moments (2000 samples each side): mean/sd of unposed and posed
    (box-plot) distances agree within 4 combined SE after scaling by sqrt(sum lambda ratio)
    (observed <= 1.7 SE).
  - RNG differences documented: R Mersenne-Twister/inversion vs rand StdRng, so individual
    samples and hence Q1a/Q3a are not reproducible across libraries.
  - Quadrature differences: fdars trapezoid L2 + L2-orthonormal PCA vs fdasrvf Riemann sum/T
    + Euclidean SVD of K (equivalent up to quadrature); fdasrvf's median is unstable across
    T (at T = 240 it collapses onto curve 4), which bounds achievable agreement.

## Gates
- `cargo fmt --all --check`: clean.
- `cargo test -p fdars-core --lib alignment::curve_shape`: 100 passed (+ new 9 incl. von
  Mises), 1 ignored (pre-existing). `cargo test --doc curve_shape`: 24 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean.
- No serde types added.

## Deviations
- STATE.md not touched and `quick-tasks-append` not run (writes STATE.md) per orchestrator
  override; commit-only, no push.
- Model mean is zero (fdasrvf) rather than the PCA mean tangent; fdasrvf's `normv < 1e-4 ->
  q2 = mu` quirk in closed shooting replaced by stopping at the current point.
- `scale = false` rejected (fdasrvf aborts too) rather than extended.

## Remaining
None for #71.
