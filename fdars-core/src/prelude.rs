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
pub use crate::regression::{FpcaResult, PlsResult};
pub use crate::scalar_on_function::{FregreLmResult, FunctionalLogisticResult};

// Classification
pub use crate::classification::{ClassifCvResult, ClassifFit, ClassifMethod, ClassifResult};

// Explainability
pub use crate::explain_generic::{FpcPredictor, TaskType};

// Depth functions
pub use crate::depth::{
    band_1d, fraiman_muniz_1d, fraiman_muniz_2d, functional_spatial_1d, functional_spatial_2d,
    modal_1d, modal_2d, modified_band_1d, random_projection_1d, random_tukey_1d, rpd_depth_1d,
};

// Metric functions
pub use crate::metric::{dtw_distance, lp_cross_1d, lp_self_1d};

// Smoothing
pub use crate::smoothing::{CvCriterion, OptimBandwidthResult};

// Basis types
pub use crate::basis::BasisProjectionResult;
pub use crate::smooth_basis::{BasisType, SmoothBasisResult};

// Elastic analysis
pub use crate::elastic_fpca::{HorizFpcaResult, JointFpcaResult, VertFpcaResult};
pub use crate::elastic_regression::{
    ElasticLogisticResult, ElasticPcrResult, ElasticRegressionResult, ScalarOnShapeResult,
};

// Statistical Process Monitoring
pub use crate::spm::{
    AmewmaMonitorResult, ArlResult, ControlLimit, CusumMonitorResult, ElasticSpmChart,
    ElasticSpmMonitorResult, EwmaMonitorResult, FrccChart, FrccMonitorResult,
    IterativePhase1Result, MewmaMonitorResult, MfSpmChart, MfpcaResult, PartialMonitorResult,
    ProfileChart, ProfileMonitorResult, SpmChart, SpmMonitorResult,
};

// Tolerance bands
pub use crate::tolerance::{
    ElasticToleranceBandResult, ElasticToleranceConfig, PhaseToleranceBand, ToleranceBand,
};

// Cross-validation
pub use crate::cv::{CvMetrics, CvType};

// Alignment
pub use crate::alignment::{
    AlignmentResult, BayesianAlignmentResult, ClosedKarcherMeanResult, ElasticClusterResult,
    ElasticDepthResult, ElasticOutlierResult, FpnsResult, GenerativeModelResult, GeodesicPath,
    KarcherMeanResult, KarcherMeanResultNd, LambdaCvResult, PartialMatchResult, PcaNdResult,
    PersistenceDiagramResult, PhaseBoxplot, RobustKarcherResult, ShapeCiResult, ShapeMeanResult,
    TransferAlignResult, WarpStatistics,
};

// Irregular functional data
pub use crate::irreg_fdata::IrregFdata;
