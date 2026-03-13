# Changelog

All notable changes to fdars-core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Robust regression** (`scalar_on_function/robust.rs`): L1 (LAD) regression via IRLS (`fregre_l1`), Huber M-estimation (`fregre_huber`), prediction (`predict_fregre_robust`), and `FregreRobustResult` struct
- **Predict/project methods on result types**: `FpcaResult::project()`/`reconstruct()`, `PlsResult::project()`, `KmeansResult::predict()`, `FuzzyCmeansResult::predict()`
- **`FdMatrix::iter_rows()`/`iter_columns()`**: row iterator (yields `Vec<f64>`), column iterator (zero-copy `&[f64]`)
- **Builder configs for smooth_basis**: `SmoothBasisGcvConfig`, `BasisNbasisCvConfig` with `Default` impls and `_with_config()` entry points
- **3 new benchmark files**: `smoothing_benchmarks.rs` (33 cases), `basis_benchmarks.rs` (22 cases), `matrix_benchmarks.rs` (45 cases)
- 119 new `explain_generic` tests, 72 new `smooth_basis` tests, 14 new `famm` tests, and more across regression, clustering, matrix modules

### Changed

- **Smoothing API Result migration**: `nadaraya_watson`, `local_linear`, `local_polynomial`, `knn_smoother`, `smoothing_matrix_nw` now return `Result<Vec<f64>, FdarError>` with input validation
- **`famm.rs` parallelism**: per-component scalar mixed model fitting now uses `iter_maybe_parallel!`
- **`#[non_exhaustive]`** on 33 public enums and 102 public structs for forward-compatible API evolution
- **Actionable error diagnostics**: 30 `ComputationFailed` error messages across 20 files now include "what to try" hints (e.g., SVD → "try reducing ncomp", Cholesky → "try increasing lambda", zero variance → "check your data")
- Replaced last `.unwrap()` in library code (`seasonal/mod.rs`) with graceful fallback
- Clippy pedantic cleanup (82 warnings fixed across 41 files)

## [0.8.4] - 2026-03-13

### Added

- **2D Function-on-Scalar Regression** (`fosr_2d`): surface-valued functional responses Y(s,t) with anisotropic tensor-product penalty (Kronecker-structured), GCV selection for both smoothing parameters, and convenience reshape methods (`beta_surface`, `r_squared_surface`, `residual_surface`). New types: `Grid2d`, `FosrResult2d`
- **`prelude` module**: `use fdars_core::prelude::*` imports ~30 most common types
- **`elastic` module**: unified re-export of all elastic_* modules
- **`ProjectionBasisType` enum**: replaces `basis_type: i32` magic numbers in GMM and basis modules
- **Builder config structs**: `GmmClusterConfig`, `StlConfig`, `ConformalConfig` with `Default` impls and `_with_config()` entry points
- **`FdMatrix` row methods**: `row_to_buf`, `row_dot`, `row_l2_sq` for zero-allocation row access
- **`lib.rs` re-exports** for 8 modules (regression, clustering, metric, depth, outliers, utility, fdata, basis)
- **`#[must_use]`** on 74 expensive public functions
- **`Debug + Clone`** on 65 public result types; `PartialEq` on 97 public types
- **Doc examples**: 37 new `# Examples` doc tests on public functions (alignment, basis, classification, conformal, depth, explain, metric, regression, scalar_on_function, smoothing, tolerance, etc.)
- Shared `test_helpers` module with `uniform_grid` replacing 17 duplicated definitions
- 37 new tests (conformal elastic, elastic regression configs, tricube kernel) and 12 new 2D FOSR tests

### Changed

- **Module splits**: 13 monolithic files split into submodule directories (classification, depth, streaming_depth, irreg_fdata, explain/helpers, metric, conformal, explain_generic, elastic_regression, basis, gmm, detrend, tolerance)
- Integration tests consolidated from 10 versioned files into 5 topically-named files
- Benchmark suite expanded with 4 new criterion benchmarks (classification, alignment, regression, explainability)
- Shared linalg helpers consolidated from duplicated Cholesky/Mahalanobis code into `linalg` module
- Complete `Result<T, FdarError>` migration: all public functions now return `Result`, including tolerance module
- NaN-safe sorting: replaced 46 inline patterns with `helpers::sort_nan_safe`
- Applied rustfmt to all source, bench, test, and example files
- **Clippy pedantic cleanup** (82 warnings fixed across 41 files): `let...else` patterns, explicit imports replacing wildcard `use super::*`, digit-separated numeric literals, `usize::from(bool)`, `sort_unstable`, `for x in &mut container`, inclusive ranges, `needless_pass_by_value` (Vec → &[T])

### Fixed

- Changepoint p-values: removed dead `CovKernel` enum and unused parameters; permutation-based p-values now correctly calibrated
- Replaced `.unwrap()` calls with `Result<T, FdarError>` in clustering, `cholesky_factor`, and VIF diagnostics

### Performance

- kNN partial sort: `select_nth_unstable` replaces full sort (O(n) vs O(n log n))
- SHAP sparse coalitions: skip zero entries in ATA accumulation
- Pre-allocated buffers in SHAP and PDP inner loops
- SRSF caching: pre-compute transforms once in elastic distance matrices
- Eliminated `to_row_major()` copies in DTW, soft-DTW, Fourier, hshift metrics
- Parallelized hot loops in depth projections, k-means++, outlier variance/trimmed stats, FOSR fitted values

## [0.8.3] - 2026-03-11

### Added

- **Model selection framework**: `aic` and `bic` fields on `FregreLmResult` and `FunctionalLogisticResult`; new `SelectionCriterion` enum, `ModelSelectionResult` struct, and `model_selection_ncomp()` utility
- **Predict methods**: standalone `predict_functional_logistic`, `predict_elastic_regression`, `predict_elastic_logistic`; `.predict()` convenience methods on all result structs
- **Gradient functions**: `gradient_nonuniform()` for non-uniform grids, `gradient()` auto-dispatch wrapper
- 9 new R-validated tests (conformal, cross-validation, explainability consistency, elastic changepoint)
- Inline documentation of variance convention choices at 4 key locations

### Changed

- **Breaking**: renamed `covariates` to `scalar_covariates` in all classification functions for consistency with `scalar_on_function`
- Split `explain.rs` (6516 lines, 44 public functions) into 10 focused submodules under `src/explain/` with zero downstream breakage
- Tightened R validation test tolerances (L2 norm, distance matrix, SRSF roundtrip, local polynomial)

## [0.8.2] - 2026-03-11

### Fixed

- **29 correctness bugs** from deep validation rounds 3-7:
  - Conformal prediction: Jackknife+ lower quantile uses floor per Barber et al. 2021; CV+ calls fit_predict once per fold
  - 2D integration weights use column-major layout matching FdMatrix
  - AIC/BIC use total EDF for multi-curve smoothing
  - `fregre_basis_cv` applies roughness penalty instead of ridge
  - `local_polynomial` degree=0 delegates to Nadaraya-Watson
  - `compute_coefficient_se` uses covariance diagonal, not L2 norm
  - Logistic and generic SHAP `base_value` corrected
  - QDA covariance uses Bessel's correction (n-1 divisor)
  - CV passes covariates through to LDA/QDA/kNN; fixes double remap_labels
  - Hilbert transform branches on odd/even signal length
  - Equivalence test p-value divides by nb (was 2*nb)
  - Karcher mean resets convergence flag before fine iterations
  - Soft-DTW self-distance computes diagonal explicitly
  - Edge case guards for `deriv_1d`, `deriv_2d`, `compute_step_sizes`, `create_folds`

## [0.8.1] - 2026-03-11

### Added

- **Conformal prediction module** (`conformal`): split conformal inference for regression (`conformal_fregre_lm`, `conformal_elastic_regression`) and classification (LDA, kNN, elastic logistic) with Adaptive Prediction Sets scoring; calibration utilities
- **`outliers_threshold_lrt_with_dist`**: returns full bootstrap null distribution alongside threshold for per-curve p-value computation
- **Generic explainability framework** (`explain_generic`): `FpcPredictor` trait unifying regression, binary, and multiclass models; 15 model-agnostic functions (PDP, SHAP, ALE, LIME, permutation importance, Sobol, Friedman H, anchors, counterfactuals, prototype/criticism, saliency, VIF, domain selection, stability)
- Cross-validation fold utilities (`cv` module)
- `ClassifFit` struct with `FpcPredictor` impl for LDA/QDA/kNN classifiers

### Fixed

- Changepoint p-values always returning 1.0: normalization mismatch between observed CUSUM statistic and Brownian bridge simulation
- Panic on extreme `trim` values in outlier detection: underflow and OOB now clamped safely

## [0.8.0] - 2026-03-10

### Added

- **Elastic FPCA**: vertical, horizontal, and joint functional PCA after SRSF alignment
- **Elastic regression**: alignment-integrated scalar-on-function regression, PCR, and logistic classification
- **Elastic changepoint detection**: amplitude and phase changepoint tests with permutation inference
- **Elastic attribution**: amplitude vs phase importance decomposition for elastic models
- **Smooth basis**: B-spline and Fourier basis representation with smoothing penalty and GCV selection
- **14 new explainability features** (total: 28): LOO-CV/PRESS diagnostics, Sobol indices, calibration diagnostics (Brier, log loss, Hosmer-Lemeshow), functional saliency maps, domain selection, conditional permutation importance, counterfactual explanations, prototype/criticism selection (MMD), LIME, ECE/MCE/ACE, regression depth diagnostics, stability/robustness analysis, anchor explanations
- 26 runnable examples with comprehensive documentation

### Changed

- Refactored all complexity hotspots: max cyclomatic 11 (from 27), max cognitive 17 (from 73)

## [0.7.2] - 2026-03-06

### Added

- 22 new R-validated tests covering irregular functional data, detrending, seasonal analysis
- `irreg_fdata` module for irregular/sparse functional data
- Seeded random projections (`random_projection_1d_seeded`, `random_tukey_1d_seeded`) for reproducible depth ordering
- Comprehensive `VALIDATION.md` documenting observed accuracy for every validated metric

### Changed

- **Breaking**: `functional_spatial_1d()` gains a third parameter `argvals: Option<&[f64]>` (pass `None` for previous behavior)

### Fixed

- **Simpson's 1/3 integration** replaces trapezoidal rule in `simpsons_weights()` and cumulative integration, reducing inner product and norm errors by 3+ orders of magnitude
- **5-point central difference gradient** (O(h^4)) replaces 3-point stencil in `gradient_uniform()`, improving SRSF round-trip fidelity
- Functional spatial depth now uses L2 integration weights via Simpson's rule
- Modified epigraph index matches R's `roahd::MEI` using `<=` comparison
- Bessel's correction (n-1 divisor) in tolerance band variance estimation
- REML divisor (n-p) in FAMM variance component updates

### Performance

- Tri-cube kernel added to smoothing; LOESS now uses tri-cube by default (matching R's `loess()`)

## [0.7.1] - 2026-03-06

### Fixed

- **TSRVF tangent vector smoothing**: DP alignment kinks propagated into SRSF and dominated tangent vectors. Added Nadaraya-Watson smoothing (Gaussian kernel, bandwidth = 2 grid spacings) to aligned SRSFs and mean SRSF in `tsrvf_from_alignment`
- Fixed Karcher mean premature convergence after 2 iterations
- Fixed `tsrvf_inverse` using mean initial value for all curves

## [0.7.0] - 2026-03-05

### Added

- **Scalar-on-function regression** (`scalar_on_function`): FPC-based linear model (`fregre_lm`), nonparametric kernel regression (`fregre_np_mixed`), functional logistic regression, cross-validated component selection (`fregre_cv`)
- **Function-on-scalar regression** (`function_on_scalar`): penalized pointwise OLS (`fosr`), FPC-based FOSR (`fosr_fpc`) matching R's `fda.usc`, functional ANOVA with permutation tests
- **Gaussian mixture model clustering** (`gmm`): EM algorithm with full/diagonal/spherical covariance, automatic K selection via BIC/ICL, prediction for new observations
- **Functional classification** (`classification`): LDA, QDA, k-NN, kernel, and DD-classifier with cross-validation support and mixed scalar/functional predictors
- **Functional mixed effects models** (`famm`): FPCA-based functional mixed model with iterative GLS + REML variance estimation, subject-level random effects, prediction, and permutation hypothesis tests
- 5 new runnable examples covering all new modules

### Changed

- Reduced max cognitive complexity from 82 to 29; max cyclomatic from 22 to 11
- FOSR FPC implementation now matches R reference (`fda.usc::fregre.pc`)
- FAMM variance estimation uses iterative REML, closing gap with R's `lmer`

## [0.6.2] - 2026-03-05

### Added

- **Landmark registration** (`landmark`): align functional data using known landmark features (peaks, valleys, zero-crossings) with monotone cubic Hermite interpolation (Fritsch-Carlson)
- **TSRVF transform**: transported square-root velocity function for projecting elastically-aligned curves into tangent space; `tsrvf_transform`, `tsrvf_from_alignment`, `tsrvf_inverse`
- **Warping utilities** (`warping`): `gam_to_psi`/`psi_to_gam`, Karcher mean on the sphere, gamma inversion, geodesic shooting
- **Constrained elastic alignment** (`elastic_align_pair_constrained`): landmark-constrained DP alignment
- ~100 new tests (edge cases + R validation)

### Changed

- Reduced max cognitive complexity from 61 to 15 across alignment, metric, and landmark modules

## [0.6.1] - 2026-03-04

### Added

- **Functional equivalence testing** (TOST): `equivalence_test` and `equivalence_test_one_sample` with multiplier and percentile bootstrap methods, sup-norm test statistic
- R cross-validation scripts and JSON fixtures for alignment, tolerance bands, and equivalence tests
- 17 new integration tests validated against R (total: 54)

### Fixed

- **Elastic alignment rewrite**: DP optimizer now matches R's fdasrvf with 35 coprime-pair neighborhood, slope-corrected edge weights, full m*m cost table, and SRSF L2 normalization. Pairwise elastic distance now exact-matches R
- **Karcher mean**: closest-observed-SRSF template selection, SqrtMeanInverse pre-centering and post-convergence centering. Relative L2 error reduced from 13.2% to 1.8%

## [0.6.0] - 2026-03-04

### Added

- **Elastic alignment module** (`alignment`): SRSF transforms (`srsf_transform`, `srsf_inverse`), pairwise alignment via DP (`elastic_align_pair`), Fisher-Rao distance (`elastic_distance`), `align_to_target` (parallelized), distance matrices, Karcher mean with convergence tracking, warping utilities (`reparameterize_curve`, `compose_warps`)
- **Tolerance bands module** (`tolerance`): `fpca_tolerance_band`, `conformal_prediction_band`, `exponential_family_tolerance_band`, `scb_mean_degras`, `elastic_tolerance_band`
- 2 new examples: tolerance bands and elastic alignment

## [0.5.0] - 2026-03-03

### Added

- Criterion benchmarks for `kfsd_1d`, `kmeans_fd`, `fuzzy_cmeans_fd`, `silhouette_score`, `calinski_harabasz`, `norm_lp_irreg`

### Changed

- `IrregFdata` is now the primary API for irregular functional data

### Performance

- **depth.rs**: KFSD kernel matrix replaced `Vec<Vec<f64>>` with `FdMatrix`; random projection depth flattened to stride-based access
- **irreg_fdata.rs**: Lp norm specialization for p=1 and p=2 avoids `powf` (50-100x faster)
- **clustering.rs**: complete flat-buffer rewrite; all 9 internal helpers converted from `&[Vec<f64>]` to `(curves: &[f64], n, m)`; cluster centers stored as flat `Vec<f64>`; `compute_fuzzy_membership_into` writes into caller-provided buffer
- **metric.rs**: extracted `self_distance_matrix` / `cross_distance_matrix` helpers; flat upper-triangle buffers replace `Vec<(usize, usize, f64)>` tuples; Lp distance inlined to preserve auto-vectorization; DTW precomputed band bounds
- **Overall**: ~8x faster for `norm_lp_irreg` p=1 vs p=3; 18% faster `modified_band_1d`; 16% faster `random_projection_1d`

## [0.4.0] - 2026-02-19

### Changed

- **Breaking: FdMatrix API migration** — All 12 public modules now accept `&FdMatrix` instead of `(&[f64], n: usize, m: usize)`. The new `FdMatrix` type wraps column-major `Vec<f64>` with safe `(i, j)` indexing, zero-copy column access, nalgebra interop, and `Send + Sync` for rayon parallelism
- Eliminated all complexity violations (max cyclomatic: 10, max cognitive: 15)
- Replaced 11 risky `.unwrap()` calls with safe alternatives

## [0.3.2] - 2026-02-17

### Added

- **R validation suite**: 44 integration tests validating against R reference packages (fda, fda.usc, roahd, cluster, fpc, dtw, pls, glmnet, KernSmooth, FNN, lomb, pracma) across 12 modules
- **Streaming depth module**: online depth computation
- 14 documented examples covering the full API
- 54 new tests (line coverage 78% to 94%)

### Changed

- Eliminated all 16 remaining library-code complexity violations by extracting ~60 helpers
- Consolidated two READMEs into single root README

## [0.3.0] - 2026-01-05

### Added

- **Seasonal Analysis Methods**
  - `autoperiod()`: Hybrid FFT + ACF period detection with gradient ascent refinement (Vlachos et al. 2005)
  - `cfd_autoperiod()`: Differencing-based detrending with density clustering (Puech et al. 2020)
  - `sazed()`: Parameter-free ensemble period detection combining 5 methods
  - All methods return confidence scores for detected periods

- **WASM Compatibility**
  - Conditional compilation for `wasm32-unknown-unknown` target
  - New feature flags: `parallel`, `linalg`, `js`
  - Sequential fallbacks when `parallel` feature is disabled
  - CI workflow for WASM build verification

### Changed

- `rayon` dependency is now optional (enabled by `parallel` feature, on by default)
- `faer` and `anofox-regression` dependencies are now optional (enabled by `linalg` feature)
- Parallel iteration now uses macros (`iter_maybe_parallel!`, `slice_maybe_parallel!`) for conditional compilation

### Fixed

- SAZED algorithm optimized to reduce false positive rate from 64% to 3%

## [0.2.0] - 2026-01-03

### Changed

- Improved test coverage to 84%+
- Added pre-commit hooks for cargo fmt and clippy
- Refactored seasonal analysis code to remove duplication

## [0.1.0] - 2025-12-01

### Added

- Initial release
- Functional data operations (mean, centering, derivatives, Lp norms, geometric median)
- Depth measures (Fraiman-Muniz, modal, band, modified band, random projection, etc.)
- Distance metrics (Lp, Hausdorff, DTW, Fourier semimetric)
- Basis representations (B-splines, Fourier, P-splines)
- Clustering (K-means, fuzzy c-means)
- Smoothing (Nadaraya-Watson, local linear, local polynomial, k-NN)
- Regression (functional PCA, PLS, ridge)
- Outlier detection (LRT-based)
- Seasonal analysis (FFT, ACF period detection, seasonal strength)
