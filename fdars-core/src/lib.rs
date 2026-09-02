//! # fdars-core
//!
//! Core algorithms for Functional Data Analysis in Rust.
//!
//! This crate provides pure Rust implementations of various FDA methods including:
//! - Functional data operations (mean, derivatives, norms)
//! - Depth measures (Fraiman-Muniz, modal, band, random projection, etc.)
//! - Distance metrics (Lp, Hausdorff, DTW, Fourier, etc.)
//! - Basis representations (B-splines, P-splines, Fourier)
//! - Clustering (k-means, fuzzy c-means)
//! - Smoothing (Nadaraya-Watson, local linear/polynomial regression)
//! - Outlier detection
//! - Regression (PCA, PLS, ridge)
//! - Seasonal analysis (period estimation, peak detection, seasonal strength)
//! - Detrending and decomposition for non-stationary data
//!
//! ## Imports
//!
//! Items are organized into domain-specific submodules. Prefer importing from
//! the submodule for clarity:
//!
//! ```rust,no_run
//! use fdars_core::matrix::FdMatrix;
//! use fdars_core::alignment::{karcher_mean, elastic_align_pair, AlignmentOutput};
//! use fdars_core::spm::{spm_phase1, spm_monitor, SpmConfig};
//! use fdars_core::regression::fdata_to_pc_1d;
//! use fdars_core::scalar_on_function::{fregre_lm, fregre_pls};
//! use fdars_core::cv::{cv_fdata_with_metrics, regression_metrics};
//! use fdars_core::distance::pairwise_distance_matrix;
//! ```
//!
//! All public items are also re-exported at the crate root for convenience:
//! ```rust,no_run
//! use fdars_core::{FdMatrix, karcher_mean, spm_phase1, fdata_to_pc_1d};
//! ```
//!
//! The [`prelude`] module provides the most commonly used types:
//! ```rust,no_run
//! use fdars_core::prelude::*;
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `parallel` | **yes** | Enables rayon-based parallelism via `iter_maybe_parallel!` macro |
//! | `linalg` | no | Enables `faer` and `anofox-regression` dependencies (requires Rust 1.84+). Gates `ridge_regression_fit`. Not WASM-compatible. |
//! | `serde` | no | Adds `Serialize`/`Deserialize` to core types (`FdMatrix`, `FpcaResult`, `SpmChart`, etc.) and enables `serde_json::Value` in `ExplainLayer.extra`. |
//! | `js` | no | Enables `getrandom/js` for WASM builds. |
//!
//! ## Data Layout
//!
//! Functional data is represented using the [`FdMatrix`] type, a column-major matrix
//! wrapping a flat `Vec<f64>` with safe `(i, j)` indexing and dimension tracking:
//! - For n observations with m evaluation points: `data[(i, j)]` gives observation i at point j
//! - 2D surfaces (n observations, m1 x m2 grid): stored as n x (m1*m2) matrices
//! - Zero-copy column access via `data.column(j)`, row gather via `data.row(i)`
//! - nalgebra interop via `to_dmatrix()` / `from_dmatrix()` for SVD operations

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

pub mod error;
pub(crate) mod linalg;
pub mod matrix;
pub mod parallel;

pub use error::FdarError;

#[cfg(test)]
pub(crate) mod test_helpers;

pub mod alignment;
pub mod andrews;

// Shared utility modules
pub mod basis;
pub mod boosting_regression;
pub mod classification;
pub mod clustering;
pub mod clustering_advanced;
pub mod coclustering;
pub mod concurrent_regression;
pub mod cv;
pub mod density_fda;
pub mod depth;
pub mod detrend;
pub mod dim;
pub mod distance;
pub(crate) mod distributions;
pub mod famm;
pub mod fdata;
pub mod fem_smoothing;
pub mod fof_regression;
pub mod fpca_variants;
pub mod frechet;
pub mod fts;
pub mod function_on_scalar;
pub mod function_on_scalar_2d;
pub mod gmm;
pub mod helpers;
pub mod inference;
pub mod irreg_fdata;
pub mod landmark;
pub mod metric;
pub mod outliers;
pub mod regression;
pub mod scalar_on_function;
pub mod seasonal;
pub mod simulation;
pub mod smoothing;
pub mod streaming_depth;
pub mod tolerance;
pub mod utility;
pub mod validation;
pub mod warping;
pub mod wire;

// Covariance kernels and Gaussian processes
pub mod covariance;

// Statistical Process Monitoring
pub mod spm;

// Elastic analysis modules
pub mod conformal;
pub mod elastic;
pub mod elastic_changepoint;
pub mod elastic_explain;
pub mod elastic_fpca;
pub mod elastic_regression;
pub mod explain;
pub mod explain_generic;
pub mod multi_fdata;
pub mod pace_fpca;
pub mod pda;
pub(crate) mod permutation_test;
pub mod prelude;
pub mod scoring;
pub mod smooth_basis;

/// Hidden test-support surface used ONLY by the crate's integration
/// equivalence tests (`tests/equivalence_phase49.rs`). NOT part of the public
/// API — it exists solely so the behavior-preservation goldens can reach the
/// `pub(crate)` numerical primitives and the current `inference`/`spm` tail
/// functions from an external test crate. `#[doc(hidden)]` keeps it out of the
/// rendered docs; these are thin forwarding `fn`s (not `pub use` re-exports,
/// which cannot escape `pub(crate)`), so no existing signature is widened and
/// no stable guarantee is made about this module.
#[doc(hidden)]
pub mod __equivalence_test_support {
    /// Forwards to the CURRENT (pre-consolidation) `inference`/`spm` tail
    /// functions — the goldens assert these stay bit-identical across the
    /// call-site migration.
    pub mod current {
        #[inline]
        pub fn chi_square_sf(x: f64, k: usize) -> f64 {
            crate::inference::dist::chi_square_sf(x, k)
        }
        #[inline]
        pub fn chi_square_sf_df(x: f64, df: f64) -> f64 {
            crate::inference::dist::chi_square_sf_df(x, df)
        }
        #[inline]
        pub fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
            crate::inference::dist::f_sf(f, d1, d2)
        }
        #[inline]
        pub fn chi2_cdf(x: f64, k: usize) -> f64 {
            crate::spm::chi_squared::chi2_cdf(x, k)
        }
        #[inline]
        pub fn chi2_quantile(p: f64, k: usize) -> f64 {
            crate::spm::chi_squared::chi2_quantile(p, k)
        }
        #[inline]
        pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
            crate::spm::chi_squared::regularized_gamma_p(a, x)
        }
    }

    /// Forwards to the NEW consolidated `crate::distributions::*` surface — the goldens assert the
    /// shared module reproduces the pre-refactor bits directly.
    pub mod distributions {
        #[inline]
        pub fn chi2_sf(x: f64, df: f64) -> f64 {
            crate::distributions::chi2_sf(x, df)
        }
        #[inline]
        pub fn chi2_cdf(x: f64, k: usize) -> f64 {
            crate::distributions::chi2_cdf(x, k)
        }
        #[inline]
        pub fn chi2_quantile(p: f64, k: usize) -> f64 {
            crate::distributions::chi2_quantile(p, k)
        }
        #[inline]
        pub fn reg_gamma_p(a: f64, x: f64) -> f64 {
            crate::distributions::reg_gamma_p(a, x)
        }
        #[inline]
        pub fn reg_gamma_q(a: f64, x: f64) -> f64 {
            crate::distributions::reg_gamma_q(a, x)
        }
        #[inline]
        pub fn ln_gamma(x: f64) -> f64 {
            crate::distributions::ln_gamma(x)
        }
    }

    /// Forwards to the `pub(crate)` per-thread RNG-seeding helper (CONS-02) so
    /// the `rng_stream` golden can prove the consolidated
    /// `helpers::seed_for_thread` draws are bit-identical to the pre-refactor
    /// `StdRng::seed_from_u64(seed + k)` formula from an external test crate.
    pub mod helpers {
        /// Returns the first `n` `u64` draws produced by
        /// `crate::helpers::seed_for_thread(seed, k)`. Returning drawn `u64`s
        /// (rather than the `pub(crate)` `StdRng` itself) keeps the crate-private
        /// RNG type from escaping while still exercising the exact stream.
        #[inline]
        pub fn seed_for_thread_draws(seed: u64, k: usize, n: usize) -> Vec<u64> {
            use rand::RngCore;
            let mut rng = crate::helpers::seed_for_thread(seed, k);
            (0..n).map(|_| rng.next_u64()).collect()
        }
    }
}

// Re-export matrix types
pub use matrix::{FdCurveSet, FdMatrix};

// Re-export multi-domain functional data container
pub use multi_fdata::{FdComponent, MultiFunData};

// Re-export linear differential operator and principal differential analysis
pub use pda::{principal_differential_analysis, Lfd, PdaResult};

// Re-export density-valued FDA entry points
pub use density_fda::{
    inverse_lqd, lqd_fpca, lqd_transform, normalize_density, wasserstein_barycenter, LqdFpcaResult,
};

// Re-export Fréchet / object-data regression types (FRE-01 + FRE-02 object spaces)
pub use frechet::{
    frechet_anova, frechet_anova_space, frechet_global_reg, frechet_global_reg_space,
    frechet_local_reg, frechet_local_reg_space, frechet_mean, frechet_variance,
    wasserstein2_distance, CorrelationMatrixSpace, FrechetAnovaResult, FrechetGlobalRegResult,
    FrechetLocalRegResult, MetricSpace, NetworkSpace, PointProcessSpace, SpdMatrixSpace, SpdMetric,
    SphericalSpace, WassersteinDensitySpace,
};

// Re-export Andrews curves types
pub use andrews::{andrews_loadings, andrews_transform, AndrewsLoadings, AndrewsResult};

// Re-export covariance kernel types
pub use covariance::{
    covariance_matrix, generate_gaussian_process, CovKernel, GaussianProcessResult,
};

// Re-export alignment types and functions
pub use alignment::{
    align_to_target, alignment_quality, amplitude_distance, amplitude_self_distance_matrix,
    bayesian_align_pair, compose_warps, curve_geodesic, curve_geodesic_nd, cut_dendrogram,
    diagnose_alignment, diagnose_pairwise, elastic_align_pair, elastic_align_pair_closed,
    elastic_align_pair_constrained, elastic_align_pair_multires, elastic_align_pair_nd,
    elastic_align_pair_penalized, elastic_align_pair_with_landmarks, elastic_cross_distance_matrix,
    elastic_cross_distance_matrix_banded, elastic_cross_distance_matrix_with_band,
    elastic_decomposition, elastic_depth, elastic_distance, elastic_distance_closed,
    elastic_distance_nd, elastic_outlier_detection, elastic_partial_match,
    elastic_self_distance_matrix, elastic_self_distance_matrix_banded,
    elastic_self_distance_matrix_with_band, gauss_model, hierarchical_from_distances, horiz_fpns,
    invert_warp, joint_gauss_model, karcher_covariance_nd, karcher_mean, karcher_mean_banded,
    karcher_mean_closed, karcher_mean_nd, karcher_mean_with_band, karcher_median,
    kmedoids_from_distances, lambda_cv, least_squares_score, least_squares_shift_registration,
    orbit_representative, pairwise_consistency, pairwise_correlation_score, pca_nd,
    peak_persistence, phase_boxplot, phase_distance_pair, phase_self_distance_matrix,
    reparameterize_curve, robust_karcher_mean, shape_confidence_interval, shape_distance,
    shape_mean, shape_self_distance_matrix, sobolev_least_squares_score, srsf_inverse,
    srsf_inverse_nd, srsf_transform, srsf_transform_nd, transfer_alignment, tsrvf_from_alignment,
    tsrvf_from_alignment_with_method, tsrvf_inverse, tsrvf_transform, tsrvf_transform_with_method,
    warp_complexity, warp_inverse_error, warp_smoothness, warp_statistics, AlignmentDiagnostic,
    AlignmentDiagnosticSummary, AlignmentQuality, AlignmentResult, AlignmentResultNd,
    AlignmentSetResult, BayesianAlignConfig, BayesianAlignmentResult, ClosedAlignmentResult,
    ClosedKarcherMeanResult, ConstrainedAlignmentResult, DecompositionResult, Dendrogram,
    DiagnosticConfig, ElasticDepthResult, ElasticOutlierConfig, ElasticOutlierResult, FpnsResult,
    GenerativeModelResult, GeodesicPath, GeodesicPathNd, KMedoidsConfig, KMedoidsResult,
    KarcherMeanResult, KarcherMeanResultNd, LambdaCvConfig, LambdaCvResult, Linkage,
    MultiresConfig, OrbitRepresentative, PartialMatchConfig, PartialMatchResult, PcaNdResult,
    PersistenceDiagramResult, PhaseBoxplot, RobustKarcherConfig, RobustKarcherResult,
    ShapeCiConfig, ShapeCiResult, ShapeDistanceResult, ShapeMeanResult, ShapeQuotient,
    ShiftRegistrationResult, TransferAlignConfig, TransferAlignResult, TransportMethod,
    TsrvfResult, WarpPenaltyType, WarpStatistics,
};

// Re-export commonly used items
pub use helpers::{
    aic, bandwidth_candidates_from_dists, bic, cumulative_trapz, extract_curves, fdata_interpolate,
    fdata_interpolate_with_policy, gaussian_kernel, gradient, gradient_nonuniform,
    gradient_uniform, impute_missing_values, l2_distance, linear_interp, quantile_sorted,
    r_squared, r_squared_adj, simpsons_weights, simpsons_weights_2d, spline_interpolate,
    spline_interpolate_with_policy, trapz, ExtrapolationPolicy, ImputationMethod,
    InterpolationMethod, DEFAULT_CONVERGENCE_TOL, NUMERICAL_EPS,
};

// Re-export warping utilities
pub use warping::{
    exp_map_sphere, gam_to_psi, gam_to_psi_smooth, inner_product_l2, inv_exp_map_sphere,
    invert_gamma, l2_norm_l2, normalize_warp, phase_distance, psi_to_gam,
};

// Re-export seasonal analysis types
pub use seasonal::{
    autoperiod, autoperiod_fdata, cfd_autoperiod, cfd_autoperiod_fdata, hilbert_transform, sazed,
    sazed_fdata, AutoperiodCandidate, AutoperiodResult, CfdAutoperiodResult, ChangeDetectionResult,
    ChangePoint, ChangeType, DetectedPeriod, InstantaneousPeriod, Peak, PeakDetectionResult,
    PeriodEstimate, SazedComponents, SazedResult, StrengthMethod,
};

// Re-export landmark registration types
pub use landmark::{
    detect_and_register, detect_landmarks, landmark_register, Landmark, LandmarkKind,
    LandmarkResult,
};

// Re-export detrending types
pub use detrend::{DecomposeResult, StlConfig, StlResult, TrendResult};

// Re-export simulation types (incl. FTS-03-04/05 functional VAR/VMA + FARMA simulators, plan 41-02)
pub use simulation::{sim_farma, sim_fvarma, EFunType, EValType, FarmaResult, FvarmaResult};

// Re-export irregular fdata types
pub use irreg_fdata::{IrregFdata, KernelType};
// Re-export FACE sparse covariance (Phase 38)
pub use irreg_fdata::{face_covariance, face_trajectory, mface_covariance, MfaceCovResult};

// Re-export tolerance band types
pub use tolerance::{
    conformal_prediction_band, elastic_tolerance_band, elastic_tolerance_band_with_config,
    equivalence_test, equivalence_test_one_sample, exponential_family_tolerance_band,
    fpca_tolerance_band, phase_tolerance_band, scb_mean_degras, BandType,
    ElasticToleranceBandResult, ElasticToleranceConfig, EquivalenceBootstrap,
    EquivalenceTestResult, ExponentialFamily, MultiplierDistribution, NonConformityScore,
    PhaseToleranceBand, ToleranceBand,
};

// Re-export functional inference types and two-sample tests
pub use inference::{
    f_perm_test, flm_f_test, flm_gof_test, itp_flm, itp_one_pop, itp_two_pop, mean_scb,
    oneway_anova_vstat, scb_two_sample_test, t_perm_test, two_sample_mean_test, ItpResult,
    TestResult, DEFAULT_N_PERM,
};

// Re-export functional time series serial-dependence types (FTS-02) and
// spectral / dynamic-FPCA types (FTS-03, plan 41-01)
pub use fts::{
    dpca, dpca_reconstruct, fplsr, ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update,
    functional_acf, functional_difference, functional_pacf, long_run_covariance, spectral_density,
    stationarity_test, ArModelResult, DpcaReconstruction, DpcaResult, FacfResult, FplsrResult,
    FtsmForecastResult, FtsmResult, LongRunCovResult, SpectralDensityResult, StationarityResult,
};

// Re-export FAMM types
pub use famm::{
    dense_flmm, fast_fmm, fmm, fmm_predict, fmm_test_fixed, multi_famm, DenseFlmmConfig,
    DenseFlmmResult, FastFmmConfig, FastFmmResult, FmmResult, FmmTestResult, MultiFammConfig,
    MultiFammResult,
};

// Re-export concurrent (varying-coefficient) regression types
pub use concurrent_regression::{concurrent_regression, ConcurrentRegrResult};

// Re-export PACE sparse FPCA types
pub use pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult};

// Re-export function-on-function regression types
pub use fof_regression::{
    fof_cv, fof_re_regression, fof_regression, predict_fof, predict_fof_re, FofCvResult,
    FofReConfig, FofReResult, FofResult,
};

// Re-export function-on-scalar regression types.
// `fanova` is deprecated (delegates to `fanova_seeded`) but must stay re-exported for external
// back-compat; this compiler warns on re-exporting a deprecated item, so allow it on this line only.
#[allow(deprecated)]
pub use function_on_scalar::{
    fanova, fanova_seeded, fosr, fosr_fpc, predict_fosr, FanovaResult, FosrFpcResult, FosrResult,
};
pub use function_on_scalar_2d::{fosr_2d, predict_fosr_2d, FosrResult2d, Grid2d};

// Re-export scalar-on-function regression types
pub use scalar_on_function::{
    bootstrap_ci_fregre_lm, bootstrap_ci_functional_logistic, fam, fregre_basis_cv, fregre_cv,
    fregre_gkam, fregre_gsam, fregre_huber, fregre_l1, fregre_lm, fregre_lm_multi,
    fregre_lm_multi_cv, fregre_np_cv, fregre_np_from_distances, fregre_np_mixed, fregre_pls,
    functional_glm, functional_logistic, history_index, model_selection_ncomp,
    permutation_test_fam, predict_fregre_lm, predict_fregre_lm_multi, predict_fregre_np,
    predict_fregre_np_from_distances, predict_fregre_pls, predict_fregre_robust,
    predict_functional_glm, predict_functional_logistic, variable_selection, BootstrapCiResult,
    FamConfig, FamResult, FregreBasisCvResult, FregreCvResult, FregreLmResult, FregreNpCvResult,
    FregreNpResult, FregreRobustResult, FunctionalGlmResult, FunctionalLogisticResult, GkamConfig,
    GkamResult, GlmFamily, GsamConfig, GsamResult, HistoryIndexConfig, HistoryIndexResult,
    ModelSelectionResult, MultiCvResult, MultiFregreLmResult, PermTestConfig, PermTestResult,
    PermTestStatistic, PlsRegressionResult, SelectionCriterion, VarSelectConfig, VarSelectPenalty,
    VarSelectResult,
};

// Re-export generic explainability types
pub use explain_generic::{
    generic_ale, generic_anchor, generic_conditional_permutation_importance,
    generic_counterfactual, generic_domain_selection, generic_friedman_h, generic_lime,
    generic_pdp, generic_permutation_importance, generic_prototype_criticism, generic_saliency,
    generic_shap_values, generic_sobol_indices, generic_stability, generic_vif, FpcPredictor,
    TaskType,
};

// Re-export explainability types
pub use elastic_explain::{elastic_pcr_attribution, ElasticAttributionResult};
pub use explain::{
    anchor_explanation, anchor_explanation_logistic, beta_decomposition,
    beta_decomposition_logistic, calibration_diagnostics, conditional_permutation_importance,
    conditional_permutation_importance_logistic, conformal_prediction_residuals,
    counterfactual_logistic, counterfactual_regression, dfbetas_dffits, domain_selection,
    domain_selection_logistic, expected_calibration_error, explanation_stability,
    explanation_stability_logistic, fpc_ale, fpc_ale_logistic, fpc_permutation_importance,
    fpc_permutation_importance_logistic, fpc_shap_values, fpc_shap_values_logistic, fpc_vif,
    fpc_vif_logistic, friedman_h_statistic, friedman_h_statistic_logistic, functional_pdp,
    functional_pdp_logistic, functional_saliency, functional_saliency_logistic,
    influence_diagnostics, lime_explanation, lime_explanation_logistic, loo_cv_press,
    pointwise_importance, pointwise_importance_logistic, prediction_intervals, prototype_criticism,
    regression_depth, regression_depth_logistic, significant_regions, significant_regions_from_se,
    sobol_indices, sobol_indices_logistic, AleResult, AnchorCondition, AnchorResult, AnchorRule,
    BetaDecomposition, CalibrationDiagnosticsResult, ConditionalPermutationImportanceResult,
    ConformalPredictionResult, CounterfactualResult, DepthType, DfbetasDffitsResult,
    DomainSelectionResult, EceResult, FpcPermutationImportance, FpcShapValues, FriedmanHResult,
    FunctionalPdpResult, FunctionalSaliencyResult, ImportantInterval, InfluenceDiagnostics,
    LimeResult, LooCvResult, PointwiseImportanceResult, PredictionIntervalResult,
    PrototypeCriticismResult, RegressionDepthResult, SignificanceDirection, SignificantRegion,
    SobolIndicesResult, StabilityAnalysisResult, VifResult,
};

// Re-export classification types
pub use classification::{
    fclassif_cv, fclassif_cv_with_config, fclassif_dd, fclassif_kernel, fclassif_knn,
    fclassif_knn_fit, fclassif_lda, fclassif_lda_fit, fclassif_qda, fclassif_qda_fit,
    kernel_classify_from_distances, knn_classify_from_distances, ClassifCvConfig, ClassifCvResult,
    ClassifFit, ClassifMethod, ClassifResult,
};

// Re-export conformal prediction types
pub use conformal::{
    conformal_classif, conformal_elastic_logistic, conformal_elastic_pcr,
    conformal_elastic_pcr_with_config, conformal_elastic_regression,
    conformal_elastic_regression_with_config, conformal_fregre_lm, conformal_fregre_np,
    conformal_generic_classification, conformal_generic_regression, conformal_logistic,
    cv_conformal_classification, cv_conformal_regression, jackknife_plus_regression,
    ClassificationScore, ConformalClassificationResult, ConformalConfig, ConformalMethod,
    ConformalRegressionResult,
};

// Re-export GMM clustering types
pub use gmm::{
    funhddC_cluster, gmm_cluster, gmm_cluster_with_config, gmm_em, predict_gmm, CovType,
    FunHddcConfig, FunHddcResult, GmmClusterConfig, GmmClusterResult, GmmResult,
};

// Re-export streaming depth types
pub use streaming_depth::{
    FullReferenceState, RollingReference, SortedReferenceState, StreamingBd, StreamingDepth,
    StreamingFraimanMuniz, StreamingMbd,
};

// Re-export FEM smoothing types (wave-1 + wave-2: mesh assembly, basis evaluation, SR-PDE smoothing)
pub use fem_smoothing::{
    assemble_fem_matrices, fem_basis_eval, fem_predict, fem_smooth, fem_smooth_gcv, FemSmoothResult,
};

// Re-export smooth basis types
pub use smooth_basis::{
    basis_nbasis_cv, basis_nbasis_cv_with_config, bspline_penalty_matrix, fourier_penalty_matrix,
    smooth_basis, smooth_basis_aic, smooth_basis_gcv, smooth_basis_gcv_with_config,
    smooth_monotone, smooth_positive, BasisCriterion, BasisNbasisCvConfig, BasisNbasisCvResult,
    BasisType, FdPar, SmoothBasisGcvConfig, SmoothBasisResult, SmoothMonotoneResult,
    SmoothPositiveResult,
};

// Re-export elastic FPCA types
pub use elastic_fpca::{
    horiz_fpca, horiz_fpca_from_alignment, joint_fpca, joint_fpca_from_alignment, vert_fpca,
    vert_fpca_from_alignment, HorizFpcaResult, JointFpcaResult, VertFpcaResult,
};

// Re-export elastic regression types
pub use elastic_regression::{
    elastic_logistic, elastic_logistic_with_config, elastic_multinomial, elastic_pcr,
    elastic_pcr_with_config, elastic_regression, elastic_regression_with_config,
    predict_elastic_logistic, predict_elastic_multinomial, predict_elastic_regression,
    predict_scalar_on_shape, scalar_on_shape, ElasticConfig, ElasticLogisticResult,
    ElasticMultinomialResult, ElasticPcrConfig, ElasticPcrResult, ElasticRegressionResult,
    IndexMethod, PcaMethod, ScalarOnShapeConfig, ScalarOnShapeResult,
};

// Re-export SPM types
pub use spm::{
    arl0_ewma_t2, arl0_spe, arl0_t2, arl1_t2, elastic_spm_monitor, elastic_spm_phase1,
    evaluate_rules, ewma_scores, frcc_monitor, frcc_phase1, hotelling_t2, hotelling_t2_regularized,
    mf_spm_monitor, mf_spm_phase1, mfpca, nelson_rules, profile_monitor, profile_phase1,
    select_ncomp, spe_contributions, spe_control_limit, spe_limit_robust,
    spe_moment_match_diagnostic, spe_multivariate, spe_univariate, spm_amewma_monitor,
    spm_cusum_monitor, spm_cusum_monitor_with_restart, spm_ewma_monitor, spm_mewma_monitor,
    spm_monitor, spm_monitor_from_fields, spm_monitor_partial, spm_monitor_partial_batch,
    spm_phase1, spm_phase1_iterative, t2_contributions, t2_contributions_mfpca, t2_control_limit,
    t2_limit_robust, t2_pc_contributions, t2_pc_significance, western_electric_rules, AmewmaConfig,
    AmewmaMonitorResult, ArlConfig, ArlResult, ChartRule, ControlLimit, ControlLimitMethod,
    CusumConfig, CusumMonitorResult, DomainCompletion, ElasticSpmChart, ElasticSpmConfig,
    ElasticSpmMonitorResult, EwmaConfig, EwmaMonitorResult, FrccChart, FrccConfig,
    FrccMonitorResult, IterativePhase1Config, IterativePhase1Result, MewmaConfig,
    MewmaMonitorResult, MfSpmChart, MfpcaConfig, MfpcaResult, NcompMethod, PartialDomainConfig,
    PartialMonitorResult, ProfileChart, ProfileMonitorConfig, ProfileMonitorResult, RuleViolation,
    SpmChart, SpmConfig, SpmMonitorResult,
};

// Re-export elastic changepoint types
pub use elastic_changepoint::{
    elastic_amp_changepoint, elastic_fpca_changepoint, elastic_ph_changepoint, ChangepointResult,
    ChangepointType, FpcaChangepointMethod,
};

// Re-export cross-validation utilities
pub use cv::{
    classification_metrics, create_folds, create_stratified_folds, cv_fdata, cv_fdata_with_metrics,
    fold_indices, metric_accuracy, metric_f1, metric_mae, metric_precision, metric_r_squared,
    metric_recall, metric_rmse, regression_metrics, subset_rows, subset_vec, CvFdataResult,
    CvMetrics, CvSelectionResult, CvType, MetricFn,
};

// Re-export distance utilities
pub use distance::{
    cross_distance_matrix, euclidean_distance_matrix, l2_distance_matrix, pairwise_distance_matrix,
};

// Re-export validation utilities
pub use validation::{
    validate_dist_mat, validate_fdata, validate_labels, validate_ncomp, validate_response,
};

// Re-export smoothing CV types
pub use smoothing::{
    aic_smoother, cv_smoother, gcv_smoother, knn_gcv, knn_lcv, optim_bandwidth, CvCriterion,
    KnnCvResult, OptimBandwidthResult,
};

// Re-export regression types
pub use regression::{fdata_to_pc_1d, fdata_to_pls_1d, FpcaResult, PlsResult};
// Re-export specialized FPCA variants (Phase 37)
pub use fpca_variants::{
    cross_covariance, dynamical_correlation, fpca_der, fsvd, ssvd, FsvdResult,
};
#[cfg(feature = "linalg")]
pub use regression::{ridge_regression_fit, RidgeResult};

// Re-export co-clustering types (funLBM latent block model)
pub use coclustering::{
    co_cluster, co_cluster_select, BlockParams, CoClusterConfig, CoClusterResult,
    CoClusterSelectResult,
};

// Re-export clustering types
pub use clustering::{
    calinski_harabasz, calinski_harabasz_from_distances, fuzzy_cmeans_fd, kmeans_fd,
    silhouette_score, silhouette_score_from_distances, FuzzyCmeansResult, KmeansResult,
};

// Re-export advanced clustering types (DBSCAN, kCFC, funFEM, align-cluster)
pub use clustering_advanced::{
    align_cluster_fd, dbscan_fd, funfem_cluster, kcfc_cluster, AlignClusterConfig,
    AlignClusterResult, DbscanConfig, DbscanResult, FunFemConfig, FunFemResult, KcfcConfig,
    KcfcResult,
};

// Re-export distance metric types and functions
pub use metric::{
    dtw_cross_1d, dtw_distance, dtw_self_1d, fourier_cross_1d, fourier_self_1d, gak,
    gak_gram_matrix, hausdorff_3d, hausdorff_cross_1d, hausdorff_cross_2d, hausdorff_self_1d,
    hausdorff_self_2d, hshift_cross_1d, hshift_self_1d, lp_cross_1d, lp_cross_2d, lp_self_1d,
    lp_self_2d, sigma_gak, soft_dtw_barycenter, soft_dtw_cross_1d, soft_dtw_distance,
    soft_dtw_div_cross_1d, soft_dtw_div_self_1d, soft_dtw_divergence, soft_dtw_self_1d, GakConfig,
    SoftDtwBarycenterResult,
};

// Re-export depth measure functions
// Re-export the shared dimensionality selector for the unified depth/fdata dispatchers.
pub use dim::Dim;

// The deprecated `_2d` shims are still re-exported for back-compat (API-03); re-exporting a
// deprecated item emits a deprecation warning, so allow it on this back-compat re-export block.
#[allow(deprecated)]
pub use depth::{
    band_1d, epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d, fraiman_muniz,
    fraiman_muniz_1d, fraiman_muniz_2d, functional_boxplot, functional_depth,
    functional_spatial_1d, functional_spatial_2d, half_region_depth_1d, hypograph_index_1d,
    kernel_functional_spatial_1d, kernel_functional_spatial_2d, linfinity_depth_1d, modal,
    modal_1d, modal_2d, modified_band_1d, modified_epigraph_index_1d,
    modified_half_region_depth_1d, modified_hypograph_index_1d, random_projection,
    random_projection_1d, random_projection_1d_seeded, random_projection_2d, random_tukey,
    random_tukey_1d, random_tukey_1d_seeded, random_tukey_2d, total_variation_depth_1d,
    DepthMethod, FunctionalBoxplotResult, TvdMssResult,
};

// Re-export outlier detection functions
pub use outliers::{
    depthgram, detect_outliers_lrt, magnitude_shape_outlyingness, muod, outliergram,
    outliers_threshold_lrt, outliers_threshold_lrt_with_dist, sequential_transform_outliers,
    tvdmss, DepthgramConfig, DepthgramResult, MagnitudeShapeResult, MuodConfig, MuodResult,
    OutligramResult, SeqTransform, SeqTransformConfig, SeqTransformOutliers, TvdMssConfig,
    TvdMssOutliers,
};

// Re-export utility functions
pub use utility::{
    compute_adot, inner_product, inner_product_matrix, integrate_simpson, knn_loocv, knn_predict,
    pcvm_statistic, rp_stat, RpStatResult,
};

// Re-export functional data operation types and functions.
// `mean_2d` is a deprecated back-compat shim (API-03); allow the deprecation on this re-export block.
#[allow(deprecated)]
pub use fdata::{
    center_1d, depth_based_median, deriv_1d, deriv_2d, functional_covariance, functional_std,
    functional_variance, geometric_median_1d, geometric_median_2d, mean, mean_1d, mean_2d,
    norm_lp_1d, normalize, normalize_with_argvals, trim_mean, Deriv2DResult, NormalizationMethod,
};

// Re-export basis representation types and functions
pub use basis::{
    basis_to_fdata, basis_to_fdata_1d, bspline_basis, bspline_basis_from_knots, constant_basis,
    construct_bspline_knots, difference_matrix, exponential_basis, fdata_to_basis,
    fdata_to_basis_1d, fourier_basis, fourier_basis_with_period, fourier_fit_1d, monomial_basis,
    polygonal_basis, power_basis, pspline_evaluate, pspline_fit_1d, pspline_fit_gcv,
    select_basis_auto_1d, select_fourier_nbasis_gcv, BasisAutoSelectionResult,
    BasisProjectionResult, BasisSystem, FourierFitResult, ProjectionBasisType, PsplineFitResult,
    SingleCurveSelection,
};

// Re-export functional scoring metrics
pub use scoring::{
    functional_explained_variance, functional_mae, functional_mape, functional_mse, functional_msle,
};

// Re-export boosting and Bayesian functional regression types (Phase 43 REG-06)
pub use boosting_regression::{
    bayesian_fosr, boost_fofr, boost_fosr, gamlss_fosr, stability_selection, BayesianConfig,
    BayesianFosrResult, BoostFofrResult, BoostFosrResult, BoostingConfig, GamlssResult,
    StabilityConfig, StabilityResult,
};
