//! Counterfactual explanations and prototype/criticism selection.

use super::helpers::{
    compute_kernel_mean, compute_witness, gaussian_kernel_matrix, greedy_prototype_selection,
    median_bandwidth, project_scores, reconstruct_delta_function,
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::FpcaResult;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};

// ===========================================================================
// Counterfactual Explanations
// ===========================================================================

/// Result of a counterfactual explanation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CounterfactualResult {
    /// Index of the observation.
    pub observation: usize,
    /// Original FPC scores.
    pub original_scores: Vec<f64>,
    /// Counterfactual FPC scores.
    pub counterfactual_scores: Vec<f64>,
    /// Score deltas: counterfactual - original.
    pub delta_scores: Vec<f64>,
    /// Counterfactual perturbation in function domain: sum_k delta_xi_k phi_k(t), length m.
    pub delta_function: Vec<f64>,
    /// L2 distance in score space: ||delta_xi||.
    pub distance: f64,
    /// Original model prediction.
    pub original_prediction: f64,
    /// Counterfactual prediction.
    pub counterfactual_prediction: f64,
    /// Whether a valid counterfactual was found.
    pub found: bool,
}

/// Counterfactual explanation for a linear functional regression model (analytical).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `observation >= n` or `fit.ncomp`
/// is zero.
/// Returns [`FdarError::InvalidDimension`] if `data` column count does not match
/// `fit.fpca.mean`.
/// Returns [`FdarError::ComputationFailed`] if the coefficient norm is near zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn counterfactual_regression(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    target_value: f64,
) -> Result<CounterfactualResult, FdarError> {
    let (n, m) = data.shape();
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    let scores = project_scores(
        data,
        &fit.fpca.mean,
        &fit.fpca.rotation,
        ncomp,
        &fit.fpca.weights,
    );

    let original_prediction = fit.fitted_values[observation];
    let gap = target_value - original_prediction;

    // gamma = [coef[1], ..., coef[ncomp]]
    let gamma: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let gamma_norm_sq: f64 = gamma.iter().map(|g| g * g).sum();

    if gamma_norm_sq < 1e-30 {
        return Err(FdarError::ComputationFailed {
            operation: "counterfactual_regression",
            detail: "coefficient norm is near zero; the model has no predictive signal — try increasing ncomp or check data quality".into(),
        });
    }

    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let delta_scores: Vec<f64> = gamma.iter().map(|&gk| gap * gk / gamma_norm_sq).collect();
    let counterfactual_scores: Vec<f64> = original_scores
        .iter()
        .zip(&delta_scores)
        .map(|(&o, &d)| o + d)
        .collect();

    let delta_function = reconstruct_delta_function(&delta_scores, &fit.fpca.rotation, ncomp, m);
    let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();
    let counterfactual_prediction = original_prediction + gap;

    Ok(CounterfactualResult {
        observation,
        original_scores,
        counterfactual_scores,
        delta_scores,
        delta_function,
        distance,
        original_prediction,
        counterfactual_prediction,
        found: true,
    })
}

/// Counterfactual explanation for a functional logistic regression model (gradient descent).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `observation >= n` or `fit.ncomp`
/// is zero.
/// Returns [`FdarError::InvalidDimension`] if `data` column count does not match
/// `fit.fpca.mean`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn counterfactual_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    max_iter: usize,
    step_size: f64,
) -> Result<CounterfactualResult, FdarError> {
    let (n, m) = data.shape();
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    let scores = project_scores(
        data,
        &fit.fpca.mean,
        &fit.fpca.rotation,
        ncomp,
        &fit.fpca.weights,
    );

    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let original_prediction = fit.probabilities[observation];
    let original_class = usize::from(original_prediction >= 0.5);
    let target_class = 1 - original_class;

    let (current_scores, current_pred, found) = logistic_counterfactual_descent(
        fit.intercept,
        &fit.coefficients,
        &original_scores,
        target_class,
        ncomp,
        max_iter,
        step_size,
    );

    let delta_scores: Vec<f64> = current_scores
        .iter()
        .zip(&original_scores)
        .map(|(&c, &o)| c - o)
        .collect();

    let delta_function = reconstruct_delta_function(&delta_scores, &fit.fpca.rotation, ncomp, m);
    let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();

    Ok(CounterfactualResult {
        observation,
        original_scores,
        counterfactual_scores: current_scores,
        delta_scores,
        delta_function,
        distance,
        original_prediction,
        counterfactual_prediction: current_pred,
        found,
    })
}

/// Run logistic counterfactual gradient descent: returns (scores, prediction, found).
fn logistic_counterfactual_descent(
    intercept: f64,
    coefficients: &[f64],
    initial_scores: &[f64],
    target_class: usize,
    ncomp: usize,
    max_iter: usize,
    step_size: f64,
) -> (Vec<f64>, f64, bool) {
    let mut current_scores = initial_scores.to_vec();
    let mut current_pred =
        logistic_predict_from_scores(intercept, coefficients, &current_scores, ncomp);

    for _ in 0..max_iter {
        current_pred =
            logistic_predict_from_scores(intercept, coefficients, &current_scores, ncomp);
        let current_class = usize::from(current_pred >= 0.5);
        if current_class == target_class {
            return (current_scores, current_pred, true);
        }
        for k in 0..ncomp {
            // Cross-entropy gradient: dL/ds_k = (p - target) * coef_k
            let grad = (current_pred - target_class as f64) * coefficients[1 + k];
            current_scores[k] -= step_size * grad;
        }
    }
    (current_scores, current_pred, false)
}

/// Logistic predict from FPC scores.
fn logistic_predict_from_scores(
    intercept: f64,
    coefficients: &[f64],
    scores: &[f64],
    ncomp: usize,
) -> f64 {
    let mut eta = intercept;
    for k in 0..ncomp {
        eta += coefficients[1 + k] * scores[k];
    }
    sigmoid(eta)
}

// ===========================================================================
// Prototype / Criticism Selection (MMD-based)
// ===========================================================================

/// Result of prototype/criticism selection.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PrototypeCriticismResult {
    /// Indices of selected prototypes.
    pub prototype_indices: Vec<usize>,
    /// Witness function values for prototypes.
    pub prototype_witness: Vec<f64>,
    /// Indices of selected criticisms.
    pub criticism_indices: Vec<usize>,
    /// Witness function values for criticisms.
    pub criticism_witness: Vec<f64>,
    /// Bandwidth used for the Gaussian kernel.
    pub bandwidth: f64,
}

/// Select prototypes and criticisms from FPCA scores using MMD-based greedy selection.
///
/// Takes an `FpcaResult` directly -- works with both linear and logistic models
/// (caller passes `&fit.fpca`).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `fpca.scores` has zero rows.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero, `n_prototypes`
/// is zero, or `n_prototypes > n`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn prototype_criticism(
    fpca: &FpcaResult,
    ncomp: usize,
    n_prototypes: usize,
    n_criticisms: usize,
) -> Result<PrototypeCriticismResult, FdarError> {
    let n = fpca.scores.nrows();
    let actual_ncomp = ncomp.min(fpca.scores.ncols());
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "fpca.scores",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if actual_ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    if n_prototypes == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_prototypes",
            message: "must be > 0".into(),
        });
    }
    if n_prototypes > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_prototypes",
            message: format!("n_prototypes {n_prototypes} > n {n}"),
        });
    }
    let n_crit = n_criticisms.min(n.saturating_sub(n_prototypes));

    let bandwidth = median_bandwidth(&fpca.scores, n, actual_ncomp);
    let kernel = gaussian_kernel_matrix(&fpca.scores, actual_ncomp, bandwidth);
    let mu_data = compute_kernel_mean(&kernel, n);

    let (selected, is_selected) = greedy_prototype_selection(&mu_data, &kernel, n, n_prototypes);
    let witness = compute_witness(&kernel, &mu_data, &selected, n);
    let prototype_witness: Vec<f64> = selected.iter().map(|&i| witness[i]).collect();

    let mut criticism_candidates: Vec<(usize, f64)> = (0..n)
        .filter(|i| !is_selected[*i])
        .map(|i| (i, witness[i].abs()))
        .collect();
    criticism_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let criticism_indices: Vec<usize> = criticism_candidates
        .iter()
        .take(n_crit)
        .map(|&(i, _)| i)
        .collect();
    let criticism_witness: Vec<f64> = criticism_indices.iter().map(|&i| witness[i]).collect();

    Ok(PrototypeCriticismResult {
        prototype_indices: selected,
        prototype_witness,
        criticism_indices,
        criticism_witness,
        bandwidth,
    })
}
