//! Convenience re-exports for common fdars-core types.
//!
//! # Usage
//! ```rust
//! use fdars_core::prelude::*;
//! ```

// Core types
pub use crate::andrews::{AndrewsLoadings, AndrewsResult};
pub use crate::covariance::{CovKernel, GaussianProcessResult};
pub use crate::error::FdarError;
pub use crate::matrix::{FdCurveSet, FdMatrix};

// Regression results
pub use crate::function_on_scalar::FosrResult;
#[cfg(feature = "linalg")]
pub use crate::regression::RidgeResult;
pub use crate::regression::{project_scores_generic, FpcaResult, PlsResult};

// Automatic-differentiation core: forward-mode (v0.39.0 DIF-04) + reverse-mode (v0.44.0 RAD)
pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};
pub use crate::scalar_on_function::{
    predict_curve_generic, FregreLmResult, FunctionalLogisticResult,
};

// Boosting and Bayesian functional regression results (Phase 43 REG-06)
pub use crate::boosting_regression::{BayesianFosrResult, BoostFosrResult};

// Classification
pub use crate::classification::{ClassifCvResult, ClassifFit, ClassifMethod, ClassifResult};

// Shapelet transform & classification
pub use crate::shapelet::{
    shapelet_classifier_fit, ShapeletClassifier, ShapeletClassifierConfig, ShapeletClassifierFit,
    ShapeletDiscoveryConfig, ShapeletSet, ShapeletTransformFit,
};

// Explainability
pub use crate::explain_generic::{FpcPredictor, TaskType};

// Depth functions.
pub use crate::depth::{
    band, fraiman_muniz, functional_spatial, modal, modal_depth_generic, modified_band,
    random_projection, random_tukey, rpd_depth,
};

// Metric functions
pub use crate::metric::{
    dtw_distance, lp_cross, lp_self, sbd, sbd_distance_matrix, soft_dtw_distance_generic, LpDomain,
    SbdResult,
};

// k-Shape clustering + SBD-backed k-medoids
pub use crate::kshape::{kshape_fd, sbd_kmedoids, KShapeConfig, KShapeResult};

// Optimal experimental design (FOptDes, v0.35.0)
pub use crate::optimal_design::{
    design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind,
};

// Smoothing
pub use crate::smoothing::{CvCriterion, OptimBandwidthResult};

// Basis types
pub use crate::basis::BasisProjectionResult;
pub use crate::smooth_basis::{
    penalty_value_generic, BasisType, SmoothBasisResult, SmoothMonotoneResult, SmoothPositiveResult,
};

// FEM smoothing (wave-1 foundation)
pub use crate::fem_smoothing::FemSmoothResult;

// Elastic analysis
pub use crate::elastic_fpca::{HorizFpcaResult, JointFpcaResult, VertFpcaResult};
pub use crate::elastic_regression::{
    ElasticLogisticResult, ElasticPcrResult, ElasticRegressionResult, ScalarOnShapeResult,
};
pub use crate::{
    elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric, PrincipalDirections,
    VeesaPipelineResult,
};
pub use crate::{jfpca_fit, JfpcaModel, JfpcaTransform};

// Statistical Process Monitoring
pub use crate::spm::{
    AmewmaMonitorResult, ArlResult, ControlLimit, CusumMonitorResult, ElasticSpmChart,
    ElasticSpmMonitorResult, EwmaMonitorResult, FrccChart, FrccMonitorResult,
    IterativePhase1Result, MewmaMonitorResult, MfSpmChart, MfpcaResult, PartialMonitorResult,
    ProfileChart, ProfileMonitorResult, SpmChart, SpmMonitorResult,
};

// Tolerance bands
pub use crate::tolerance::{
    ConformalAnomalyConfig, ConformalAnomalyResult, ElasticToleranceBandResult,
    ElasticToleranceConfig, PhaseToleranceBand, ToleranceBand,
};

// Cross-validation
pub use crate::cv::{CvMetrics, CvType};

// Alignment
pub use crate::alignment::amplitude_distance_at_warp_generic;
pub use crate::alignment::{
    AlignmentOutput, AlignmentResult, BayesianAlignmentResult, ClosedKarcherMeanResult, Dendrogram,
    ElasticDepthResult, ElasticOutlierResult, FpnsResult, GenerativeModelResult, GeodesicPath,
    KMedoidsResult, KarcherMeanResult, KarcherMeanResultNd, LambdaCvResult, PartialMatchResult,
    PcaNdResult, PersistenceDiagramResult, PhaseBoxplot, RobustKarcherResult, ShapeCiResult,
    ShapeMeanResult, TransferAlignResult, WarpStatistics,
};

// Irregular functional data
pub use crate::irreg_fdata::IrregFdata;

// Co-clustering (funLBM latent block model)
pub use crate::coclustering::{CoClusterConfig, CoClusterResult, CoClusterSelectResult};

// PEER regression (v0.36.0)
pub use crate::peer::{
    lpeer, peer, LambdaChoice, LambdaMethod, LocalPeerResult, PeerConfig, PeerPenalty, PeerResult,
};

// Wavelet-domain regression (v0.37.0)
pub use crate::wavelet::regression::{
    wcr, wnet, WcrConfig, WcrMethod, WcrResult, WnetConfig, WnetResult,
};
pub use crate::wavelet::{
    decompose, decompose_matrix, max_level, reconstruct, BoundaryMode, WaveletCoeffs, WaveletFamily,
};
