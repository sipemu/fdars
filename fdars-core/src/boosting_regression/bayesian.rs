//! Bayesian function-on-scalar regression via conjugate Gibbs sampler (REG-06-04).
//!
//! Compresses the functional response and predictors via FPCA score projection
//! (`fdata_to_pc_1d`), then runs a conjugate Normal-Inverse-Gamma Gibbs sampler
//! in FPC-score regression space. Posterior summaries (mean, pointwise credible bands)
//! are reconstructed in the original functional domain.
//!
//! **Status:** Skeleton — implementation delivered in Plan 04.
//!
//! # References
//!
//! Jiang et al. (2025). Bayesian Function-on-Scalar Regression. arXiv:2505.05633.
//! Goldsmith et al. (2015). Smooth scalar-on-image regression via spatial Bayesian
//! variable selection. *JCGS*, 23(1).
//!
//! # Divergences from refund
//!
//! Uses FPCA score compression (`fdata_to_pc_1d`) rather than spline basis priors.
//! Pointwise credible bands only (no simultaneous bands). Seeded for determinism.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use super::{BayesianConfig, BayesianFosrResult};

/// Bayesian function-on-scalar regression via conjugate Gibbs sampler.
///
/// Runs a conjugate Normal-Inverse-Gamma Gibbs sampler on FPC-score regression
/// coefficients, returning posterior mean and pointwise credible bands for each
/// predictor coefficient function β_j(t).
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t).
/// * `predictors` — Scalar predictor matrix (n × p).
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `config` — [`BayesianConfig`] controlling the Gibbs sampler.
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] with `detail: "not yet implemented (Plan 04)"`.
/// Full implementation arrives in Plan 04.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn bayesian_fosr(
    _data: &FdMatrix,
    _predictors: &FdMatrix,
    _argvals: &[f64],
    _config: &BayesianConfig,
) -> Result<BayesianFosrResult, FdarError> {
    // filled in Plan 04
    Err(FdarError::ComputationFailed {
        operation: "bayesian_fosr",
        detail: "not yet implemented (Plan 04)".to_string(),
    })
}
