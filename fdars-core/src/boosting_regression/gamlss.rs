//! GAMLSS-style distributional functional regression — location + scale (REG-06-03).
//!
//! Models a Gaussian functional response Y(t) with separate boosting models for
//! location μ(t) (identity link) and scale σ(t) (log link). The cyclic gamboostLSS
//! algorithm alternates one boosting step per distributional parameter per iteration,
//! calling `boost_fosr_one_step` from the sibling module.
//!
//! **Status:** Skeleton — implementation delivered in Plan 03.
//!
//! # References
//!
//! Hofner et al. (2016). gamboostLSS: An R Package for Model-Based Boosting for
//! Simultaneous Estimation of Noncrossing Quantile Curves. *Journal of Statistical
//! Software*, 74(1). DOI:10.18637/jss.v074.i01.
//!
//! # Divergences from gamboostLSS
//!
//! Uses cyclic rather than noncyclic (per-iteration) parameter selection. Only the
//! Gaussian family (location + scale) is implemented. Link functions: identity for μ,
//! log for σ.

use super::{BoostingConfig, GamlssResult};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// GAMLSS-style distributional functional regression.
///
/// Fits separate boosted models for location μ(t) and scale σ(t) of a Gaussian
/// functional response, cycling over distributional parameters each iteration.
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t).
/// * `predictors` — Scalar predictor matrix (n × p).
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `config` — [`BoostingConfig`] controlling boosting iterations and B-spline spec.
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] with `detail: "not yet implemented (Plan 03)"`.
/// Full implementation arrives in Plan 03.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gamlss_fosr(
    _data: &FdMatrix,
    _predictors: &FdMatrix,
    _argvals: &[f64],
    _config: &BoostingConfig,
) -> Result<GamlssResult, FdarError> {
    // filled in Plan 03
    Err(FdarError::ComputationFailed {
        operation: "gamlss_fosr",
        detail: "not yet implemented (Plan 03)".to_string(),
    })
}
