use crate::error::FdarError;
use crate::explain::{reconstruct_delta_function, CounterfactualResult};
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Compute gradient of model prediction w.r.t. FPC scores via finite differences.
fn compute_gradient_finite_diff(
    model: &dyn FpcPredictor,
    scores: &[f64],
    ncomp: usize,
) -> Vec<f64> {
    let eps = 1e-5;
    let base = model.predict_from_scores(scores, None);
    let mut grad = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut s_plus = scores.to_vec();
        s_plus[k] += eps;
        grad[k] = (model.predict_from_scores(&s_plus, None) - base) / eps;
    }
    grad
}

/// Build a CounterfactualResult from original/final scores.
fn build_counterfactual_result(
    model: &dyn FpcPredictor,
    observation: usize,
    original_scores: Vec<f64>,
    final_scores: Vec<f64>,
    original_prediction: f64,
    ncomp: usize,
    m: usize,
    found: bool,
) -> CounterfactualResult {
    let delta_scores: Vec<f64> = final_scores
        .iter()
        .zip(&original_scores)
        .map(|(&c, &o)| c - o)
        .collect();
    let delta_function = reconstruct_delta_function(&delta_scores, model.fpca_rotation(), ncomp, m);
    let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();
    let counterfactual_prediction = model.predict_from_scores(&final_scores, None);

    CounterfactualResult {
        observation,
        original_scores,
        counterfactual_scores: final_scores,
        delta_scores,
        delta_function,
        distance,
        original_prediction,
        counterfactual_prediction,
        found,
    }
}

/// Gradient descent search for a counterfactual in classification models.
fn counterfactual_gd_search(
    model: &dyn FpcPredictor,
    original_scores: &[f64],
    max_iter: usize,
    ncomp: usize,
    converged: impl Fn(f64) -> bool,
    update: impl Fn(&mut [f64], &[f64], f64),
) -> (Vec<f64>, f64, bool) {
    let mut current_scores = original_scores.to_vec();
    let mut current_pred = model.predict_from_scores(&current_scores, None);
    let mut found = false;
    for _ in 0..max_iter {
        current_pred = model.predict_from_scores(&current_scores, None);
        if converged(current_pred) {
            found = true;
            break;
        }
        let grads = compute_gradient_finite_diff(model, &current_scores, ncomp);
        update(&mut current_scores, &grads, current_pred);
    }
    if !found {
        current_pred = model.predict_from_scores(&current_scores, None);
        found = converged(current_pred);
    }
    (current_scores, current_pred, found)
}

/// Generic counterfactual explanation for any FPC-based model.
///
/// For regression: uses analytical projection toward target_value.
/// For classification: uses gradient descent toward the opposite class.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `observation >= n` or the
/// model has zero components.
/// Returns [`FdarError::InvalidDimension`] if `data` columns do not match
/// the model.
/// Returns [`FdarError::ComputationFailed`] if the gradient norm is near
/// zero (regression only).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_counterfactual(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    target_value: f64,
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
    let scores = model.project(data);
    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let original_prediction = model.predict_from_scores(&original_scores, None);

    match model.task_type() {
        super::TaskType::Regression => {
            let grad = compute_gradient_finite_diff(model, &original_scores, ncomp);
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-30 {
                return Err(FdarError::ComputationFailed {
                    operation: "generic_counterfactual",
                    detail: "gradient norm is near zero; the model is locally flat at this observation — try a different observation or increase ncomp".into(),
                });
            }
            let gap = target_value - original_prediction;
            let delta_scores: Vec<f64> = grad.iter().map(|&gk| gap * gk / grad_norm_sq).collect();
            let final_scores: Vec<f64> = original_scores
                .iter()
                .zip(&delta_scores)
                .map(|(&o, &d)| o + d)
                .collect();
            Ok(build_counterfactual_result(
                model,
                observation,
                original_scores,
                final_scores,
                original_prediction,
                ncomp,
                m,
                true,
            ))
        }
        super::TaskType::BinaryClassification => {
            let original_class = if original_prediction >= 0.5 { 1.0 } else { 0.0 };
            let target_class = 1.0 - original_class;
            let (final_scores, _pred, found) = counterfactual_gd_search(
                model,
                &original_scores,
                max_iter,
                ncomp,
                |pred: f64| {
                    let pc: f64 = if pred >= 0.5 { 1.0 } else { 0.0 };
                    (pc - target_class).abs() < 1e-10
                },
                |scores, grads, pred| {
                    for k in 0..ncomp {
                        scores[k] -= step_size * (pred - target_class) * grads[k];
                    }
                },
            );
            Ok(build_counterfactual_result(
                model,
                observation,
                original_scores,
                final_scores,
                original_prediction,
                ncomp,
                m,
                found,
            ))
        }
        super::TaskType::MulticlassClassification(_) => {
            let original_class = original_prediction.round();
            let (final_scores, _pred, found) = counterfactual_gd_search(
                model,
                &original_scores,
                max_iter,
                ncomp,
                |pred| (pred.round() - original_class).abs() > 0.5,
                |scores, grads, _pred| {
                    let grad_norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
                    for k in 0..ncomp {
                        scores[k] += step_size * grads[k] / grad_norm;
                    }
                },
            );
            Ok(build_counterfactual_result(
                model,
                observation,
                original_scores,
                final_scores,
                original_prediction,
                ncomp,
                m,
                found,
            ))
        }
    }
}
