//! Elastic logistic regression for binary classification.

use crate::alignment::{dp_alignment_core, srsf_transform};
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

use super::{
    apply_warps_to_srsfs, beta_converged, init_identity_warps, srsf_fitted_values, ElasticConfig,
};

/// Result of elastic logistic regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticLogisticResult {
    /// Intercept.
    pub alpha: f64,
    /// Regression function β(t), length m.
    pub beta: Vec<f64>,
    /// Predicted probabilities, length n.
    pub probabilities: Vec<f64>,
    /// Predicted class labels (0 or 1), length n.
    pub predicted_classes: Vec<usize>,
    /// Classification accuracy.
    pub accuracy: f64,
    /// Logistic loss.
    pub loss: f64,
    /// Final warping functions (n × m).
    pub gammas: FdMatrix,
    /// Aligned SRSFs (n × m).
    pub aligned_srsfs: FdMatrix,
    /// Number of iterations used.
    pub n_iter: usize,
}

/// Elastic logistic regression for binary classification.
///
/// Labels should be -1 or 1. Uses gradient descent with Armijo line search.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Binary labels (-1 or 1), length n
/// * `argvals` — Evaluation points (length m)
/// * `ncomp_beta` — Number of B-spline basis functions for β
/// * `lambda` — Roughness penalty on β
/// * `max_iter` — Maximum iterations
/// * `tol` — Convergence tolerance
///
/// # Errors
///
/// Returns [`crate::FdarError::InvalidDimension`] if `n < 2`, `m < 2`,
/// `y.len() != n`, or `argvals.len() != m`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_logistic(
    data: &FdMatrix,
    y: &[i8],
    argvals: &[f64],
    _ncomp_beta: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ElasticLogisticResult, crate::FdarError> {
    let (n, m) = data.shape();
    if n < 2 || m < 2 || y.len() != n || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data/y/argvals",
            expected: "n >= 2, m >= 2, y.len() == n, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, y.len()={}, argvals.len()={}",
                n,
                m,
                y.len(),
                argvals.len()
            ),
        });
    }

    let weights = simpsons_weights(argvals);
    let q_all = srsf_transform(data, argvals);
    let mut gammas = init_identity_warps(n, argvals);
    let mut beta = vec![0.0; m];
    let mut alpha = 0.0;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let q_aligned = apply_warps_to_srsfs(&q_all, &gammas, argvals);
        let (grad_a, grad_beta, prob) =
            logistic_gradients(&q_aligned, &beta, &weights, alpha, y, lambda);

        let loss_current = logistic_loss(&prob, y, &beta, lambda);
        let grad_norm_sq: f64 = grad_a * grad_a + grad_beta.iter().map(|&g| g * g).sum::<f64>();

        let step = armijo_line_search_logistic(
            &q_aligned,
            alpha,
            &beta,
            grad_a,
            &grad_beta,
            &weights,
            y,
            lambda,
            loss_current,
            grad_norm_sq,
        );

        let beta_new: Vec<f64> = beta
            .iter()
            .zip(grad_beta.iter())
            .map(|(&b, &g)| b - step * g)
            .collect();
        let alpha_new = alpha - step * grad_a;

        if beta_converged(&beta_new, &beta, tol) && iter > 0 {
            beta = beta_new;
            alpha = alpha_new;
            break;
        }

        beta = beta_new;
        alpha = alpha_new;

        update_logistic_warps(&mut gammas, &q_all, &beta, y, argvals, lambda * 0.01);
    }

    // Final predictions
    let aligned_srsfs = apply_warps_to_srsfs(&q_all, &gammas, argvals);
    let (probabilities, predicted_classes, accuracy, loss) =
        compute_logistic_predictions(&aligned_srsfs, &beta, &weights, alpha, y, lambda);

    Ok(ElasticLogisticResult {
        alpha,
        beta,
        probabilities,
        predicted_classes,
        accuracy,
        loss,
        gammas,
        aligned_srsfs,
        n_iter,
    })
}

/// Elastic logistic regression using a configuration struct.
///
/// Equivalent to [`elastic_logistic`] but bundles method parameters in [`ElasticConfig`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_logistic_with_config(
    data: &FdMatrix,
    y: &[i8],
    argvals: &[f64],
    config: &ElasticConfig,
) -> Result<ElasticLogisticResult, crate::FdarError> {
    elastic_logistic(
        data,
        y,
        argvals,
        config.ncomp_beta,
        config.lambda,
        config.max_iter,
        config.tol,
    )
}

/// Predict probabilities for new data using a fitted elastic logistic model.
///
/// Transforms new curves to SRSFs and applies the fitted logistic
/// coefficients to produce P(Y=1).
///
/// # Arguments
/// * `fit` — A fitted [`ElasticLogisticResult`]
/// * `new_data` — New functional data (n_new × m)
/// * `argvals` — Evaluation points (length m)
pub fn predict_elastic_logistic(
    fit: &ElasticLogisticResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Vec<f64> {
    let weights = simpsons_weights(argvals);
    let q_new = srsf_transform(new_data, argvals);
    let eta = srsf_fitted_values(&q_new, &fit.beta, &weights, fit.alpha);
    eta.iter().map(|&e| 1.0 / (1.0 + (-e).exp())).collect()
}

impl ElasticLogisticResult {
    /// Predict probabilities for new data. Delegates to [`predict_elastic_logistic`].
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
        predict_elastic_logistic(self, new_data, argvals)
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Compute logistic loss with L2 penalty.
fn logistic_loss(prob: &[f64], y: &[i8], beta: &[f64], lambda: f64) -> f64 {
    let n = prob.len();
    let mut loss = 0.0;
    for i in 0..n {
        let target = if y[i] == 1 { 1.0 } else { 0.0 };
        let p = prob[i].clamp(1e-15, 1.0 - 1e-15);
        loss -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
    }
    loss /= n as f64;
    // L2 penalty
    loss += 0.5 * lambda * beta.iter().map(|&b| b * b).sum::<f64>();
    loss
}

/// Compute logistic gradients for α and β, returning (grad_a, grad_beta, probabilities).
fn logistic_gradients(
    q_aligned: &FdMatrix,
    beta: &[f64],
    weights: &[f64],
    alpha: f64,
    y: &[i8],
    lambda: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let (n, m) = q_aligned.shape();
    let eta = srsf_fitted_values(q_aligned, beta, weights, alpha);
    let prob: Vec<f64> = eta.iter().map(|&e| 1.0 / (1.0 + (-e).exp())).collect();

    let mut grad_a = 0.0;
    for i in 0..n {
        let target = if y[i] == 1 { 1.0 } else { 0.0 };
        grad_a += prob[i] - target;
    }
    grad_a /= n as f64;

    let mut grad_beta = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            let target = if y[i] == 1 { 1.0 } else { 0.0 };
            grad_beta[j] += (prob[i] - target) * q_aligned[(i, j)] * weights[j];
        }
        grad_beta[j] /= n as f64;
        grad_beta[j] += lambda * beta[j];
    }

    (grad_a, grad_beta, prob)
}

/// Armijo line search for logistic regression. Returns optimal step size.
fn armijo_line_search_logistic(
    q_aligned: &FdMatrix,
    alpha: f64,
    beta: &[f64],
    grad_a: f64,
    grad_beta: &[f64],
    weights: &[f64],
    y: &[i8],
    lambda: f64,
    loss_current: f64,
    grad_norm_sq: f64,
) -> f64 {
    let mut step = 1.0;
    for _ in 0..20 {
        let alpha_trial = alpha - step * grad_a;
        let beta_trial: Vec<f64> = beta
            .iter()
            .zip(grad_beta.iter())
            .map(|(&b, &g)| b - step * g)
            .collect();
        let eta_trial = srsf_fitted_values(q_aligned, &beta_trial, weights, alpha_trial);
        let prob_trial: Vec<f64> = eta_trial
            .iter()
            .map(|&e| 1.0 / (1.0 + (-e).exp()))
            .collect();
        let loss_trial = logistic_loss(&prob_trial, y, &beta_trial, lambda);
        if loss_trial <= loss_current - 1e-4 * step * grad_norm_sq {
            break;
        }
        step *= 0.5;
    }
    step
}

/// Update warping functions for all curves in elastic logistic regression.
fn update_logistic_warps(
    gammas: &mut FdMatrix,
    q_all: &FdMatrix,
    beta: &[f64],
    y: &[i8],
    argvals: &[f64],
    lambda: f64,
) {
    let (n, m) = q_all.shape();
    for i in 0..n {
        let qi: Vec<f64> = (0..m).map(|j| q_all[(i, j)]).collect();
        let beta_signed: Vec<f64> = beta.iter().map(|&b| b * f64::from(y[i])).collect();
        let new_gam = dp_alignment_core(&beta_signed, &qi, argvals, lambda);
        for j in 0..m {
            gammas[(i, j)] = new_gam[j];
        }
    }
}

/// Compute final logistic predictions: probabilities, classes, accuracy, loss.
fn compute_logistic_predictions(
    aligned_srsfs: &FdMatrix,
    beta: &[f64],
    weights: &[f64],
    alpha: f64,
    y: &[i8],
    lambda: f64,
) -> (Vec<f64>, Vec<usize>, f64, f64) {
    let n = y.len();
    let eta = srsf_fitted_values(aligned_srsfs, beta, weights, alpha);
    let probabilities: Vec<f64> = eta.iter().map(|&e| 1.0 / (1.0 + (-e).exp())).collect();
    let predicted_classes: Vec<usize> = probabilities
        .iter()
        .map(|&p| usize::from(p >= 0.5))
        .collect();
    let accuracy = predicted_classes
        .iter()
        .zip(y.iter())
        .filter(|(&p, &t)| p == usize::from(t == 1))
        .count() as f64
        / n as f64;
    let loss = logistic_loss(&probabilities, y, beta, lambda);
    (probabilities, predicted_classes, accuracy, loss)
}
