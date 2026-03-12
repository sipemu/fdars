//! ALE (Accumulated Local Effects) and LIME (Local Surrogate).

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{FregreLmResult, FunctionalLogisticResult};

// ===========================================================================
// ALE (Accumulated Local Effects)
// ===========================================================================

/// Result of Accumulated Local Effects analysis.
pub struct AleResult {
    /// Bin midpoints (length n_bins_actual).
    pub bin_midpoints: Vec<f64>,
    /// ALE values centered to mean zero (length n_bins_actual).
    pub ale_values: Vec<f64>,
    /// Bin edges (length n_bins_actual + 1).
    pub bin_edges: Vec<f64>,
    /// Number of observations in each bin (length n_bins_actual).
    pub bin_counts: Vec<usize>,
    /// Which FPC component was analyzed.
    pub component: usize,
}

/// ALE plot for an FPC component in a linear functional regression model.
///
/// ALE measures the average local effect of varying one FPC score on predictions,
/// avoiding the extrapolation issues of PDP.
///
/// # Errors
///
/// See [`crate::explain_generic::generic_ale`] for error conditions.
pub fn fpc_ale(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Result<AleResult, FdarError> {
    crate::explain_generic::generic_ale(fit, data, scalar_covariates, component, n_bins)
}

/// ALE plot for an FPC component in a functional logistic regression model.
///
/// # Errors
///
/// See [`crate::explain_generic::generic_ale`] for error conditions.
pub fn fpc_ale_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Result<AleResult, FdarError> {
    crate::explain_generic::generic_ale(fit, data, scalar_covariates, component, n_bins)
}

// ===========================================================================
// LIME (Local Surrogate)
// ===========================================================================

/// Result of a LIME local surrogate explanation.
pub struct LimeResult {
    /// Index of the observation being explained.
    pub observation: usize,
    /// Local FPC-level attributions, length ncomp.
    pub attributions: Vec<f64>,
    /// Local intercept.
    pub local_intercept: f64,
    /// Local R^2 (weighted).
    pub local_r_squared: f64,
    /// Kernel width used.
    pub kernel_width: f64,
}

/// LIME explanation for a linear functional regression model.
///
/// # Errors
///
/// See [`crate::explain_generic::generic_lime`] for error conditions.
pub fn lime_explanation(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Result<LimeResult, FdarError> {
    crate::explain_generic::generic_lime(
        fit,
        data,
        scalar_covariates,
        observation,
        n_samples,
        kernel_width,
        seed,
    )
}

/// LIME explanation for a functional logistic regression model.
///
/// # Errors
///
/// See [`crate::explain_generic::generic_lime`] for error conditions.
pub fn lime_explanation_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Result<LimeResult, FdarError> {
    crate::explain_generic::generic_lime(
        fit,
        data,
        scalar_covariates,
        observation,
        n_samples,
        kernel_width,
        seed,
    )
}
