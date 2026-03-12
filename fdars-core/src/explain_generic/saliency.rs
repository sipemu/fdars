use crate::error::FdarError;
use crate::explain::{
    compute_column_means, compute_domain_selection, compute_saliency_map, mean_absolute_column,
    DomainSelectionResult, FunctionalSaliencyResult,
};
use crate::matrix::FdMatrix;

use super::shap::generic_shap_values;
use super::FpcPredictor;

/// Generic functional saliency maps via SHAP-weighted rotation.
///
/// Lifts FPC-level attributions to the function domain.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if the model has zero components
/// or `n_samples` is zero (propagated from [`generic_shap_values`]).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_saliency(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<FunctionalSaliencyResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }

    // Get SHAP values first
    let shap = generic_shap_values(model, data, scalar_covariates, n_samples, seed)?;

    // Compute per-observation saliency: saliency[(i,j)] = Σ_k shap[(i,k)] × rotation[(j,k)]
    let scores = model.project(data);
    let mean_scores = compute_column_means(&scores, ncomp);

    // Weights = mean |SHAP_k| / mean |score_k - mean_k| ≈ effective coefficient magnitude
    let mut weights = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut sum_shap = 0.0;
        let mut sum_score_dev = 0.0;
        for i in 0..n {
            sum_shap += shap.values[(i, k)].abs();
            sum_score_dev += (scores[(i, k)] - mean_scores[k]).abs();
        }
        weights[k] = if sum_score_dev > 1e-15 {
            sum_shap / sum_score_dev
        } else {
            0.0
        };
    }

    let saliency_map = compute_saliency_map(
        &scores,
        &mean_scores,
        &weights,
        model.fpca_rotation(),
        n,
        m,
        ncomp,
    );
    let mean_absolute_saliency = mean_absolute_column(&saliency_map, n, m);

    Ok(FunctionalSaliencyResult {
        saliency_map,
        mean_absolute_saliency,
    })
}

/// Generic domain selection using SHAP-based functional importance.
///
/// Computes pointwise importance from the model's effective β(t) reconstruction
/// via SHAP weights, then finds important intervals via sliding window.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if the model has zero components
/// or `n_samples` is zero (propagated from [`generic_shap_values`]).
/// Returns [`FdarError::ComputationFailed`] if the domain selection
/// computation fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_domain_selection(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    window_width: usize,
    threshold: f64,
    n_samples: usize,
    seed: u64,
) -> Result<DomainSelectionResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }

    // Reconstruct effective β(t) = Σ_k w_k × φ_k(t) using SHAP-derived weights
    let shap = generic_shap_values(model, data, scalar_covariates, n_samples, seed)?;
    let scores = model.project(data);
    let mean_scores = compute_column_means(&scores, ncomp);

    let mut effective_weights = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut sum_shap = 0.0;
        let mut sum_score_dev = 0.0;
        for i in 0..n {
            sum_shap += shap.values[(i, k)].abs();
            sum_score_dev += (scores[(i, k)] - mean_scores[k]).abs();
        }
        effective_weights[k] = if sum_score_dev > 1e-15 {
            sum_shap / sum_score_dev
        } else {
            0.0
        };
    }

    // Reconstruct β(t) = Σ_k w_k × φ_k(t)
    let rotation = model.fpca_rotation();
    let mut beta_t = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            beta_t[j] += effective_weights[k] * rotation[(j, k)];
        }
    }

    compute_domain_selection(&beta_t, window_width, threshold).ok_or_else(|| {
        FdarError::ComputationFailed {
            operation: "generic_domain_selection",
            detail: "compute_domain_selection returned None".into(),
        }
    })
}
