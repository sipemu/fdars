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

pub mod matrix;
pub mod parallel;

pub mod alignment;
pub mod basis;
pub mod classification;
pub mod clustering;
pub mod depth;
pub mod detrend;
pub mod famm;
pub mod fdata;
pub mod function_on_scalar;
pub mod gmm;
pub mod helpers;
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
pub mod warping;

// Re-export matrix types
pub use matrix::{FdCurveSet, FdMatrix};

// Re-export alignment types and functions
pub use alignment::{
    align_to_target, alignment_quality, amplitude_distance, amplitude_self_distance_matrix,
    compose_warps, elastic_align_pair, elastic_align_pair_constrained, elastic_align_pair_nd,
    elastic_align_pair_with_landmarks, elastic_cross_distance_matrix, elastic_decomposition,
    elastic_distance, elastic_distance_nd, elastic_self_distance_matrix, karcher_mean,
    pairwise_consistency, phase_distance_pair, phase_self_distance_matrix, reparameterize_curve,
    srsf_inverse, srsf_inverse_nd, srsf_transform, srsf_transform_nd, tsrvf_from_alignment,
    tsrvf_from_alignment_with_method, tsrvf_inverse, tsrvf_transform, tsrvf_transform_with_method,
    warp_complexity, warp_smoothness, AlignmentQuality, AlignmentResult, AlignmentResultNd,
    AlignmentSetResult, ConstrainedAlignmentResult, DecompositionResult, KarcherMeanResult,
    TransportMethod, TsrvfResult,
};

// Re-export commonly used items
pub use helpers::{
    cumulative_trapz, extract_curves, gradient_uniform, l2_distance, linear_interp,
    simpsons_weights, simpsons_weights_2d, trapz, DEFAULT_CONVERGENCE_TOL, NUMERICAL_EPS,
};

// Re-export warping utilities
pub use warping::{
    exp_map_sphere, gam_to_psi, inner_product_l2, inv_exp_map_sphere, invert_gamma, l2_norm_l2,
    normalize_warp, phase_distance, psi_to_gam,
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
pub use detrend::{DecomposeResult, TrendResult};

// Re-export simulation types
pub use simulation::{EFunType, EValType};

// Re-export irregular fdata types
pub use irreg_fdata::{IrregFdata, KernelType};

// Re-export tolerance band types
pub use tolerance::{
    conformal_prediction_band, elastic_tolerance_band, equivalence_test,
    equivalence_test_one_sample, exponential_family_tolerance_band, fpca_tolerance_band,
    scb_mean_degras, BandType, EquivalenceBootstrap, EquivalenceTestResult, ExponentialFamily,
    MultiplierDistribution, NonConformityScore, ToleranceBand,
};

// Re-export FAMM types
pub use famm::{fmm, fmm_predict, fmm_test_fixed, FmmResult, FmmTestResult};

// Re-export function-on-scalar regression types
pub use function_on_scalar::{fanova, fosr, predict_fosr, FanovaResult, FosrResult};

// Re-export scalar-on-function regression types
pub use scalar_on_function::{
    fregre_cv, fregre_lm, fregre_np_mixed, functional_logistic, predict_fregre_lm,
    predict_fregre_np, FregreCvResult, FregreLmResult, FregreNpResult, FunctionalLogisticResult,
};

// Re-export classification types
pub use classification::{
    fclassif_cv, fclassif_dd, fclassif_kernel, fclassif_knn, fclassif_lda, fclassif_qda,
    ClassifCvResult, ClassifResult,
};

// Re-export GMM clustering types
pub use gmm::{gmm_cluster, gmm_em, predict_gmm, CovType, GmmClusterResult, GmmResult};

// Re-export streaming depth types
pub use streaming_depth::{
    FullReferenceState, RollingReference, SortedReferenceState, StreamingBd, StreamingDepth,
    StreamingFraimanMuniz, StreamingMbd,
};
