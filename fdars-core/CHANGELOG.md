# Changelog

All notable changes to fdars-core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.1]

### Added

- **`ExplainLayer.extra` → `serde_json::Value`** (#29): Arbitrary JSON support behind `serde` feature, unblocking 8+ app node types. `serde_json` added as optional dependency.
- **`RegressionLayer.fpca`** (#29): Stores FPCA decomposition. `From<&FregreLmResult>` and `From<&PlsRegressionResult>` conversions preserve internal state for explain functions.
- **`vert/horiz/joint_fpca_from_alignment`** (#29): Accept raw fields (aligned_data, gammas, mean_srsf, argvals) instead of `KarcherMeanResult`.
- **`spm_monitor_from_fields`** (#29): Phase II monitoring from raw FPCA fields (mean, rotation, weights, eigenvalues, UCLs) instead of `SpmChart`. Bitwise-identical to `spm_monitor`.
- **`SpmChartLayer.fpca_rotation`**: Stores FPCA rotation matrix for field-based monitoring.

## [0.12.0]

### Added

- **`AlignmentOutput` trait** (#26): Common interface (`mean`, `mean_srsf`, `aligned_data`, `gammas`, `converged`, `n_iter`) implemented for `KarcherMeanResult` and `RobustKarcherResult`. `From<RobustKarcherResult> for KarcherMeanResult` conversion. Enables interchangeable alignment methods in downstream code.
- **`FdaData` layer enhancements** (#28): Multiple grouping variables (`Vec<GroupVar>`), outliergram parabola coefficients, regression model metadata (`model_name`, `n_obs`), alignment convergence info, SPM FPCA state.
- **Shared `linalg.rs` module**: Unified `cholesky_factor`, `cholesky_solve`, `cholesky_forward_back`, `compute_xtx` — eliminates 4x Cholesky duplication.
- **Shared `distance.rs` module**: Generic `pairwise_distance_matrix` builder with closure-based distance + parallelism. Presets: `l2_distance_matrix`, `euclidean_distance_matrix`, `cross_distance_matrix`.
- **Shared `validation.rs` module**: `validate_fdata`, `validate_response`, `validate_labels`, `validate_dist_mat`, `validate_ncomp`.
- **Shared helpers**: `gaussian_kernel`, `bandwidth_candidates_from_dists`, `quantile_sorted`, `r_squared`, `r_squared_adj`, `aic`, `bic`.
- **`CvSelectionResult<T>`**: Generic CV hyperparameter selection result type.

### Fixed

- **Deduplicated `quantile_sorted`** (4 implementations → 1 canonical in `helpers.rs`). Fixed unsafe `floor() as usize` without bounds check in `scalar_on_function/bootstrap.rs`.
- **Deduplicated `subset_rows`** (3 implementations → 1 in `cv.rs`).
- **Removed dead code**: `_srsf_single_wrapper` in generative.rs, dead `quantile_sorted` in conformal/mod.rs.
- **Consistent conventions**: Added `#[non_exhaustive]` to 17 public structs, `PartialEq` to 6 structs missing it.

## [0.11.0]

### Breaking Changes

- **Clustering refactored to distance-matrix-driven**: Removed `elastic_kmeans`, `elastic_hierarchical`, `ElasticClusterConfig`, `ElasticClusterMethod`, `ElasticClusterResult`, `ElasticDendrogram`. Replaced with generic `kmedoids_from_distances`, `hierarchical_from_distances`, `Dendrogram`, `KMedoidsConfig`, `KMedoidsResult`, `Linkage` — all work with any precomputed distance matrix.

### Added

- **PLS regression** (`scalar_on_function/pls.rs`): `fregre_pls` and `predict_fregre_pls` for scalar-on-function regression using PLS components with optional scalar covariates. New type `PlsRegressionResult`.
- **Pluggable CV metrics** (`cv.rs`): Standalone metric functions (`metric_rmse`, `metric_mae`, `metric_r_squared`, `metric_accuracy`, `metric_precision`, `metric_recall`, `metric_f1`) and `cv_fdata_with_metrics` for user-defined scoring. Convenience sets `regression_metrics()`, `classification_metrics()`.
- **OOF predictions in fregre_cv** (#25): `FregreCvResult` now includes `oof_predictions`, `fold_assignments`, and `fold_errors` at the optimal K.
- **Distance-matrix variants**: `knn_classify_from_distances`, `kernel_classify_from_distances`, `fregre_np_from_distances`, `predict_fregre_np_from_distances`, `silhouette_score_from_distances`, `calinski_harabasz_from_distances` — all accept any precomputed distance matrix.
- **FdaData wire module** (`wire.rs`): Unified layered data container for pipeline interchange with 15 layer types (Fpca, Pls, Alignment, Distances, Depth, Outliers, Clusters, Regression, Tolerance, Spm, Explain, etc.). Composable, serde-enabled, designed for visual pipeline apps.
- **Berkeley growth example** (example 28): CV experiment predicting adult height from growth curves with P-spline smoothing, GCV model selection, and 10-fold CV comparing `fregre_lm` vs `fregre_pls`.
- **Reference documentation**: `documentation/spm_references.md` and `documentation/alignment_references.md` mapping 100+ functions to source papers with DOI links.

## [0.10.0]

### Fixed

- **B-spline cross-grid evaluation** (issue #21): `pspline_fit_1d` now stores the knot vector and order in `PsplineFitResult`. New `pspline_evaluate` function evaluates fitted P-splines on arbitrary grids using stored knots. New `bspline_basis_from_knots` evaluates B-spline basis from pre-computed knots. `construct_bspline_knots` is now public. (4 regression tests)
- **FPCA grid-density invariance** (issue #22): `fdata_to_pc_1d` now accepts `argvals` and uses Simpson's-rule integration weights in the SVD, making scores invariant to grid resolution. `FpcaResult` stores weights; `project()` uses weighted inner products. `fdata_to_pls_1d` similarly updated. All 13 callers, 3 examples, and `FpcPredictor` trait updated. (6 regression tests)

### Added (issue #23 features)

- **`pspline_fit_gcv`** (`basis/pspline.rs`): Automatic lambda selection via GCV minimization over a 25-point log-spaced grid. (2 tests)
- **`fdata_interpolate`** (`helpers.rs`): Resample functional data to a new grid with `InterpolationMethod::Linear` or `CubicHermite` (Fritsch-Carlson monotone C1). (3 tests)
- **`outliergram`** (`outliers.rs`): Outliergram combining MEI and MBD with parabolic threshold and IQR-based outlier detection. New type `OutligramResult`. (3 tests)
- **`magnitude_shape_outlyingness`** (`outliers.rs`): Magnitude-shape decomposition of outlyingness (1 - MBD for magnitude, normalized direction distance for shape). New type `MagnitudeShapeResult`. (2 tests)
- **Serde support** (issue #24): Optional `serde` feature flag adds `Serialize`/`Deserialize` derives to `FdMatrix`, `FpcaResult`, `PlsResult`, `SpmChart`, `SpmConfig`, `ControlLimit`, `ControlLimitMethod`, `MfSpmChart`, `MfpcaResult`, `MfpcaConfig`, `SpmMonitorResult`. Enables persistent Phase I/II monitoring workflows. (1 roundtrip test)

### Added

#### Alignment Module — 6 new advanced features

- **Multivariate curve Karcher mean** (`alignment/nd.rs`): iterative Karcher mean, covariance estimation, and PCA for R^d curves with shared warping. New types: `KarcherMeanResultNd`, `PcaNdResult`. Functions: `karcher_mean_nd`, `karcher_covariance_nd`, `pca_nd`
- **Gaussian generative model** (`alignment/generative.rs`): sample synthetic curves from fitted amplitude (vertical FPCA) and phase (horizontal FPCA) Gaussian models; joint model preserves amplitude-phase correlation. New type: `GenerativeModelResult`. Functions: `gauss_model`, `joint_gauss_model`
- **Bayesian alignment** (`alignment/bayesian.rs`): pairwise alignment via preconditioned Crank-Nicolson (pCN) MCMC on the Hilbert sphere with posterior warping function samples, credible bands, and acceptance diagnostics. New types: `BayesianAlignConfig`, `BayesianAlignmentResult`. Function: `bayesian_align_pair`
- **Curve geodesic interpolation** (`alignment/geodesic.rs`): geodesic paths between curves in elastic space — amplitude interpolation in SRSF space, phase interpolation on the Hilbert sphere; 1-D and N-D variants. New types: `GeodesicPath`, `GeodesicPathNd`. Functions: `curve_geodesic`, `curve_geodesic_nd`
- **Peak persistence diagram** (`alignment/persistence.rs`): topology-based automatic lambda selection by tracking peak birth/death across a lambda sweep of Karcher means. New type: `PersistenceDiagramResult`. Function: `peak_persistence`
- **Horizontal FPNS** (`alignment/fpns.rs`): Functional Principal Nested Spheres for warping functions — nonlinear PCA on the Hilbert sphere via iterative geodesic principal direction extraction. New type: `FpnsResult`. Function: `horiz_fpns`
- 25 new alignment tests

#### Alignment Module — 9 additional gaps

- **Elastic depth** (`alignment/elastic_depth.rs`): amplitude + phase decomposed functional depth via inverse-average-distance on elastic distance matrices. New type: `ElasticDepthResult`. Function: `elastic_depth`
- **Robust Karcher mean** (`alignment/robust_karcher.rs`): Karcher median via iterative Weiszfeld algorithm on the elastic manifold, and trimmed Karcher mean that removes the most distant curves. New types: `RobustKarcherConfig`, `RobustKarcherResult`. Functions: `karcher_median`, `robust_karcher_mean`
- **SRVF outlier detection** (`alignment/outlier.rs`): detect outlier curves via elastic distances from a reference (mean or median) with Tukey fence thresholding and amplitude/phase decomposition. New types: `ElasticOutlierConfig`, `ElasticOutlierResult`. Function: `elastic_outlier_detection`
- **Closed curve alignment** (`alignment/closed.rs`): alignment for periodic/closed curves with coarse-to-fine rotation search over circular starting-point shifts, plus Karcher mean for closed curves. New types: `ClosedAlignmentResult`, `ClosedKarcherMeanResult`. Functions: `elastic_align_pair_closed`, `elastic_distance_closed`, `karcher_mean_closed`
- **Elastic partial matching** (`alignment/partial_match.rs`): find the best-aligned subcurve of a longer target curve via sliding variable-length window search. New types: `PartialMatchConfig`, `PartialMatchResult`. Function: `elastic_partial_match`
- **Multi-resolution alignment** (`alignment/multires.rs`): coarse DP on a subsampled grid + fine gradient refinement on the original resolution for faster alignment of long curves. New type: `MultiresConfig`. Function: `elastic_align_pair_multires`
- **Shape confidence intervals** (`alignment/shape_ci.rs`): bootstrap confidence bands for the elastic Karcher mean via resampling and alignment of bootstrap means. New types: `ShapeCiConfig`, `ShapeCiResult`. Function: `shape_confidence_interval`
- **Transfer alignment** (`alignment/transfer.rs`): align curves from a target population to a source population's coordinate system via bridging warps composed with within-population warps. New types: `TransferAlignConfig`, `TransferAlignResult`. Function: `transfer_alignment`
- 31 new alignment tests; total: 2,492 tests (1,808 unit + 684 integration/doc)
- All new types and functions re-exported in `lib.rs` and `prelude.rs`

## [0.9.0] - 2026-03-17

### Added

#### SPM Module — 12 new submodules

- **Automatic ncomp selection** (`spm/ncomp.rs`): `select_ncomp` with cumulative variance, elbow detection, and fixed methods. New type: `NcompMethod`
- **Per-PC T² contributions** (`spm/contrib.rs`): `t2_pc_contributions` returns an n × ncomp matrix whose rows sum to the Hotelling T² value, enabling per-component fault attribution
- **Runs/zone rules** (`spm/rules.rs`): Western Electric (WE1–WE4) and Nelson (5–7) pattern detection rules for control charts; custom run rules. New types: `ChartRule`, `RuleViolation`. Functions: `evaluate_rules`, `western_electric_rules`, `nelson_rules`
- **Bootstrap/robust control limits** (`spm/bootstrap.rs`): empirical quantile, bootstrap resampling, and Gaussian KDE alternatives to parametric chi-squared limits. New type: `ControlLimitMethod`. Functions: `t2_limit_robust`, `spe_limit_robust`
- **ARL computation** (`spm/arl.rs`): Monte Carlo average run length simulation for T², EWMA-T², and SPE charts with parallelized replicates. New types: `ArlConfig`, `ArlResult`. Functions: `arl0_t2`, `arl1_t2`, `arl0_ewma_t2`, `arl0_spe`
- **MEWMA monitoring** (`spm/mewma.rs`): multivariate EWMA with asymptotic or exact time-dependent covariance and chi-squared UCL. New types: `MewmaConfig`, `MewmaMonitorResult`. Function: `spm_mewma_monitor`
- **Profile monitoring** (`spm/profile.rs`): rolling-window FOSR coefficient monitoring via FPCA and T² for detecting changes in predictor–response relationships. New types: `ProfileMonitorConfig`, `ProfileChart`, `ProfileMonitorResult`. Functions: `profile_phase1`, `profile_monitor`
- **Partial-domain monitoring** (`spm/partial.rs`): monitor incomplete functional observations using conditional expectation (BLUP), partial projection, or zero-padding. New types: `DomainCompletion`, `PartialDomainConfig`, `PartialMonitorResult`. Functions: `spm_monitor_partial`, `spm_monitor_partial_batch`
- **Phase-aware (elastic) SPM** (`spm/elastic_spm.rs`): separates amplitude and phase variation via Karcher mean alignment, then monitors each component independently. New types: `ElasticSpmConfig`, `ElasticSpmChart`, `ElasticSpmMonitorResult`. Functions: `elastic_spm_phase1`, `elastic_spm_monitor`
- **CUSUM monitoring** (`spm/cusum.rs`): multivariate (Crosier's MCUSUM) and per-component univariate CUSUM charts for detecting small sustained shifts; optional automatic restart after alarms. New types: `CusumConfig`, `CusumMonitorResult`. Functions: `spm_cusum_monitor`, `spm_cusum_monitor_with_restart`
- **Adaptive EWMA (AMFEWMA)** (`spm/amewma.rs`): dynamically adjusts smoothing parameter λ_t based on prediction error magnitude — small λ for persistent shifts, large λ for sudden shifts. New types: `AmewmaConfig`, `AmewmaMonitorResult`. Function: `spm_amewma_monitor`
- **Iterative Phase I** (`spm/iterative.rs`): repeatedly builds charts, removes out-of-control observations, and re-estimates until convergence for robust chart construction from contaminated data. New types: `IterativePhase1Config`, `IterativePhase1Result`. Function: `spm_phase1_iterative`

#### Alignment Module — 8 new features

- **Lambda cross-validation** (`alignment/lambda_cv.rs`): K-fold CV for optimal alignment regularization parameter selection. New types: `LambdaCvConfig`, `LambdaCvResult`. Function: `lambda_cv`
- **Warp statistics** (`alignment/warp_stats.rs`): pointwise mean, standard deviation, confidence bands, and Karcher mean warp on the Hilbert sphere with geodesic distances. New type: `WarpStatistics`. Function: `warp_statistics`
- **Phase box plots** (`alignment/phase_boxplot.rs`): functional box plots for warping functions using modified band depth, with central region, whiskers, and outlier detection. New type: `PhaseBoxplot`. Function: `phase_boxplot`
- **Elastic clustering** (`alignment/clustering.rs`): k-means++ with Karcher mean centers and agglomerative hierarchical clustering (single/complete/average linkage) using elastic distances. New types: `ElasticClusterConfig`, `ElasticClusterMethod`, `ElasticClusterResult`, `ElasticDendrogram`. Functions: `elastic_kmeans`, `elastic_hierarchical`, `cut_dendrogram`
- **Registration diagnostics** (`alignment/diagnostics.rs`): detect registration failures via warp complexity, smoothness, and amplitude improvement metrics with configurable thresholds. New types: `AlignmentDiagnostic`, `AlignmentDiagnosticSummary`, `DiagnosticConfig`. Functions: `diagnose_alignment`, `diagnose_pairwise`
- **Elastic shape analysis** (`alignment/shape.rs`): quotient space operations under reparameterization, translation, and scale invariance — orbit representatives, shape distances, shape means, and distance matrices. New types: `ShapeQuotient`, `OrbitRepresentative`, `ShapeDistanceResult`, `ShapeMeanResult`. Functions: `orbit_representative`, `shape_distance`, `shape_mean`, `shape_self_distance_matrix`
- **Warp inversion** (`alignment/srsf.rs`): `invert_warp` computes γ⁻¹ via monotone interpolation; `warp_inverse_error` measures ‖γ∘γ⁻¹ − id‖∞
- **Penalized alignment** (`alignment/pairwise.rs`): `WarpPenaltyType` enum (first-order, second-order, combined) and `elastic_align_pair_penalized` for curvature-penalized registration

#### Test & documentation

- 98 new tests across SPM (55) and alignment (43); total: 2,436 tests (1,752 unit + 684 integration/doc)
- A-grade documentation on all 19 SPM files: equation/page citations, error bounds, `#[must_use]` attributes
- All new types and functions re-exported in `lib.rs` and `prelude.rs`

## [0.8.5] - 2026-03-14

### Added

- **Statistical Process Monitoring module** (`spm/`): complete FPCA-based control chart framework for functional data — Hotelling T² and SPE statistics (`stats.rs`), chi-squared and moment-matched control limits (`control.rs`), Phase I/II univariate and multivariate monitoring (`phase.rs`), EWMA smoothing with adjusted eigenvalues (`ewma.rs`), Functional Regression Control Chart via FOSR residuals (`frcc.rs`), per-variable T²/SPE contribution diagnostics (`contrib.rs`), self-contained chi-squared CDF/quantile implementation (`chi_squared.rs`). New types: `SpmChart`, `MfSpmChart`, `SpmMonitorResult`, `ControlLimit`, `EwmaConfig`, `EwmaMonitorResult`, `FrccChart`, `FrccConfig`, `FrccMonitorResult`, `MfpcaConfig`, `MfpcaResult`, `SpmConfig`
- **Multivariate FPCA** (`spm/mfpca.rs`): standardize p functional variables, stack, joint SVD with `project()` and `reconstruct()` methods on `MfpcaResult`
- **Scalar-on-Shape regression** (`elastic_regression/scalar_on_shape.rs`): phase-invariant scalar-on-function regression using Fisher-Rao inner product and DP alignment; three index function methods via `IndexMethod` enum (`Identity`, `Polynomial`, `NadarayaWatson`); alternating estimation of β, h, g with Fourier basis representation and roughness penalty. New types: `ScalarOnShapeConfig`, `ScalarOnShapeResult`, `IndexMethod`
- **Phase tolerance bands** (`tolerance/elastic.rs`): `phase_tolerance_band` maps warping functions to tangent space of the Hilbert sphere via shooting vectors, computes FPCA tolerance bands, and maps bounds back to warping functions (γ). `elastic_tolerance_band_with_config` computes both amplitude and phase bands in a single Karcher mean pass. New types: `PhaseToleranceBand`, `ElasticToleranceBandResult`, `ElasticToleranceConfig`
- 159 new tests: 55 SPM unit, 32 scalar-on-shape unit, 34 SPM integration/validation, 12 phase tolerance band unit, 16 phase band integration/validation, 7 conformal generic, 3 elastic changepoint
- SPM, scalar-on-shape, and tolerance band re-exports in `lib.rs` and `prelude.rs`
- `#[must_use]` on `elastic_tolerance_band`
- **Andrews curves** (`andrews.rs`): `andrews_transform` maps p-dimensional observations to Fourier curves on [-π,π]; `andrews_loadings` visualizes FPCA loading vectors as Andrews curves; `AndrewsResult`, `AndrewsLoadings` types
- **Covariance kernels & GP** (`covariance.rs`): `CovKernel` enum with 8 kernel types (Gaussian, Exponential, Matérn, Brownian, Periodic, Linear, Polynomial, White Noise) and Sum/Product kernel algebra; `covariance_matrix` builds m×m kernel matrices; `generate_gaussian_process` produces n GP sample paths via Cholesky decomposition; `GaussianProcessResult` type
- **RPD depth** (`depth/rpd.rs`): Random Projection with Derivatives depth — enriches random projections with finite-difference derivative information for shape-sensitive depth ordering; `rpd_depth_1d`, `rpd_depth_1d_seeded`
- **Three distance semimetrics**: PCA-based (`metric/pca.rs`, L2 in PC score space via SVD), derivative-based (`metric/deriv.rs`, L2 between k-th finite differences), basis coefficient (`metric/basis_coef.rs`, Euclidean distance in B-spline/Fourier coefficient space); each with `_self_1d` and `_cross_1d` variants
- **KL divergence** (`metric/kl.rs`): symmetrized Kullback-Leibler divergence treating curves as probability densities with epsilon regularization; `kl_self_1d`, `kl_cross_1d`
- **Robust regression** (`scalar_on_function/robust.rs`): L1 (LAD) regression via IRLS (`fregre_l1`), Huber M-estimation (`fregre_huber`), prediction (`predict_fregre_robust`), and `FregreRobustResult` struct
- **Predict/project methods on result types**: `FpcaResult::project()`/`reconstruct()`, `PlsResult::project()`, `KmeansResult::predict()`, `FuzzyCmeansResult::predict()`
- **`FdMatrix::iter_rows()`/`iter_columns()`**: row iterator (yields `Vec<f64>`), column iterator (zero-copy `&[f64]`)
- **Builder configs for smooth_basis**: `SmoothBasisGcvConfig`, `BasisNbasisCvConfig` with `Default` impls and `_with_config()` entry points
- **3 new benchmark files**: `smoothing_benchmarks.rs` (33 cases), `basis_benchmarks.rs` (22 cases), `matrix_benchmarks.rs` (45 cases)
- 89 new tests for Andrews curves (17), covariance/GP (35), RPD depth (7), semimetrics (22), KL divergence (8)
- 119 new `explain_generic` tests, 72 new `smooth_basis` tests, 14 new `famm` tests, and more across regression, clustering, matrix modules

### Changed

- **Smoothing API Result migration**: `nadaraya_watson`, `local_linear`, `local_polynomial`, `knn_smoother`, `smoothing_matrix_nw` now return `Result<Vec<f64>, FdarError>` with input validation
- **`famm.rs` parallelism**: per-component scalar mixed model fitting now uses `iter_maybe_parallel!`
- **`#[non_exhaustive]`** on 33 public enums and 102 public structs for forward-compatible API evolution
- **Actionable error diagnostics**: 30 `ComputationFailed` error messages across 20 files now include "what to try" hints (e.g., SVD → "try reducing ncomp", Cholesky → "try increasing lambda", zero variance → "check your data")
- Replaced last `.unwrap()` in library code (`seasonal/mod.rs`) with graceful fallback
- **Breaking**: `conformal_generic_regression` and `conformal_generic_classification` gain `calibration_indices: Option<&[usize]>` parameter for held-out calibration (pass `None` for previous random-split behavior)
- **Breaking**: `conformal_generic_classification` now rejects multiclass models with an error (previously produced degenerate one-hot conformal sets silently)
- **Breaking**: `elastic_amp_changepoint` and `elastic_ph_changepoint` remove unused `_cov_kernel` and `_cov_bandwidth` parameters
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

- Changepoint p-values: removed dead `CovKernel` enum and unused parameters; replaced Brownian bridge with permutation testing (#18)
- Generic conformal data leakage: `calibration_indices` parameter allows held-out calibration to preserve finite-sample coverage guarantee (#20)
- Generic conformal multiclass: reject multiclass models whose `predict_from_scores` returns class labels instead of probabilities (#20)
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
