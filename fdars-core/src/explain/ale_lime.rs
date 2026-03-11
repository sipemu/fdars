//! ALE (Accumulated Local Effects) and LIME (Local Surrogate).

use super::helpers::*;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};

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
pub fn fpc_ale(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Option<AleResult> {
    let (n, m) = data.shape();
    if n < 2 || m != fit.fpca.mean.len() || n_bins == 0 || component >= fit.ncomp {
        return None;
    }
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    // Prediction function for linear model
    let predict = |obs_scores: &[f64], obs_scalar: Option<&[f64]>| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * obs_scores[k];
        }
        if let Some(z) = obs_scalar {
            for j in 0..p_scalar {
                eta += fit.gamma[j] * z[j];
            }
        }
        eta
    };

    compute_ale(
        &scores,
        scalar_covariates,
        n,
        ncomp,
        p_scalar,
        component,
        n_bins,
        &predict,
    )
}

/// ALE plot for an FPC component in a functional logistic regression model.
pub fn fpc_ale_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Option<AleResult> {
    let (n, m) = data.shape();
    if n < 2 || m != fit.fpca.mean.len() || n_bins == 0 || component >= fit.ncomp {
        return None;
    }
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    // Prediction function for logistic model
    let predict = |obs_scores: &[f64], obs_scalar: Option<&[f64]>| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * obs_scores[k];
        }
        if let Some(z) = obs_scalar {
            for j in 0..p_scalar {
                eta += fit.gamma[j] * z[j];
            }
        }
        sigmoid(eta)
    };

    compute_ale(
        &scores,
        scalar_covariates,
        n,
        ncomp,
        p_scalar,
        component,
        n_bins,
        &predict,
    )
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
pub fn lime_explanation(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Option<LimeResult> {
    let (n, m) = data.shape();
    if observation >= n || m != fit.fpca.mean.len() || n_samples == 0 || kernel_width <= 0.0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();

    // Score standard deviations
    let mut score_sd = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_sd[k] = (ss / (n - 1).max(1) as f64).sqrt().max(1e-10);
    }

    // Predict for linear model
    let predict = |s: &[f64]| -> f64 {
        let mut yhat = fit.coefficients[0];
        for k in 0..ncomp {
            yhat += fit.coefficients[1 + k] * s[k];
        }
        yhat
    };

    compute_lime(
        &obs_scores,
        &score_sd,
        ncomp,
        n_samples,
        kernel_width,
        seed,
        observation,
        &predict,
    )
}

/// LIME explanation for a functional logistic regression model.
pub fn lime_explanation_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Option<LimeResult> {
    let (n, m) = data.shape();
    if observation >= n || m != fit.fpca.mean.len() || n_samples == 0 || kernel_width <= 0.0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();

    let mut score_sd = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_sd[k] = (ss / (n - 1).max(1) as f64).sqrt().max(1e-10);
    }

    let predict = |s: &[f64]| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * s[k];
        }
        sigmoid(eta)
    };

    compute_lime(
        &obs_scores,
        &score_sd,
        ncomp,
        n_samples,
        kernel_width,
        seed,
        observation,
        &predict,
    )
}
