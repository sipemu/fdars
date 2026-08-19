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

// ─── Elastic Multinomial ─────────────────────────────────────────────────────

/// Result of elastic multinomial logistic regression (one-vs-rest, K ≥ 2 classes).
///
/// Each field corresponds to the joint outcome of fitting K binary
/// [`ElasticLogisticResult`] models — one per class — and then normalizing the
/// per-class sigmoid outputs to produce class posteriors.
///
/// # Probability convention
///
/// `train_probabilities` is an *n × K* matrix whose row *i* is the row-normalised
/// vector of OvR sigmoid scores, so each row sums to 1. Predicted labels are the
/// per-row argmax mapped through `classes`.
///
/// # K = 2 agreement
///
/// When K = 2, the predicted labels agree with the binary [`elastic_logistic`] on
/// separable data (the OvR construction reduces to the binary case).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElasticMultinomialResult {
    /// Number of classes K.
    pub n_classes: usize,
    /// Sorted distinct class labels (always `0..K`).
    pub classes: Vec<usize>,
    /// One OvR binary model per class, length K.
    pub class_models: Vec<ElasticLogisticResult>,
    /// Row-normalised OvR probabilities (n × K); each row sums to 1.
    pub train_probabilities: FdMatrix,
    /// Predicted class labels for training data, length n.
    pub predicted_classes: Vec<usize>,
    /// Fraction of training curves correctly classified.
    pub train_accuracy: f64,
}

/// Elastic multinomial logistic regression for K ≥ 2 classes (one-vs-rest).
///
/// Fits K binary [`elastic_logistic`] models — class *k* labelled +1, all other
/// classes labelled −1 — using the existing SRSF/warping/IRLS machinery unchanged.
/// Row-normalises the K sigmoid scores to obtain class posteriors.
///
/// Mirrors the binary signature: labels are `&[usize]` in `0..K` (contiguous).
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Class labels in `0..K` (contiguous, length n)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp_beta` — Number of B-spline basis functions for β per OvR model
/// * `lambda` — Roughness penalty on β
/// * `max_iter` — Maximum iterations per OvR binary fit
/// * `tol` — Convergence tolerance
///
/// # Errors
///
/// Returns [`crate::FdarError::InvalidDimension`] if `n == 0` or `y.len() != n`.
/// Returns [`crate::FdarError::InvalidParameter`] if fewer than 2 distinct classes
/// or the label set is not the contiguous range `0..K`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_multinomial(
    data: &FdMatrix,
    y: &[usize],
    argvals: &[f64],
    ncomp_beta: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ElasticMultinomialResult, crate::FdarError> {
    let (n, _m) = data.shape();

    // ── Input guards (T-27-01) ───────────────────────────────────────────────
    if n == 0 || y.len() != n {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data/y",
            expected: "n >= 1, y.len() == n".to_string(),
            actual: format!("n={}, y.len()={}", n, y.len()),
        });
    }

    let mut sorted_labels: Vec<usize> = y.to_vec();
    sorted_labels.sort_unstable();
    sorted_labels.dedup();
    let k = sorted_labels.len();

    if k < 2 {
        return Err(crate::FdarError::InvalidParameter {
            parameter: "y",
            message: format!(
                "at least 2 distinct classes required; found {} distinct label(s)",
                k
            ),
        });
    }
    // Labels must form contiguous 0..K
    for (idx, &label) in sorted_labels.iter().enumerate() {
        if label != idx {
            return Err(crate::FdarError::InvalidParameter {
                parameter: "y",
                message: format!(
                    "labels must form the contiguous range 0..{} but found label {} at position {}",
                    k, label, idx
                ),
            });
        }
    }

    // ── Fit K binary OvR models ──────────────────────────────────────────────
    let classes = sorted_labels;
    let mut class_models: Vec<ElasticLogisticResult> = Vec::with_capacity(k);
    for &class_k in &classes {
        let labels_k: Vec<i8> = y
            .iter()
            .map(|&lbl| if lbl == class_k { 1i8 } else { -1i8 })
            .collect();
        let model_k =
            elastic_logistic(data, &labels_k, argvals, ncomp_beta, lambda, max_iter, tol)?;
        class_models.push(model_k);
    }

    // ── Build n×K probability matrix and row-normalise (T-27-02) ────────────
    let mut train_probabilities = FdMatrix::zeros(n, k);
    for (col_k, model_k) in class_models.iter().enumerate() {
        for row_i in 0..n {
            train_probabilities[(row_i, col_k)] = model_k.probabilities[row_i];
        }
    }
    // Row-normalise so each row sums to 1; guard zero-sum row → uniform 1/K
    for row_i in 0..n {
        let row_sum: f64 = (0..k).map(|col| train_probabilities[(row_i, col)]).sum();
        let scale = if row_sum < 1e-15 {
            1.0 / k as f64
        } else {
            1.0 / row_sum
        };
        for col in 0..k {
            train_probabilities[(row_i, col)] *= scale;
        }
        if row_sum < 1e-15 {
            // Assign uniform probability
            for col in 0..k {
                train_probabilities[(row_i, col)] = 1.0 / k as f64;
            }
        }
    }

    // ── Predicted classes and accuracy ───────────────────────────────────────
    let predicted_classes: Vec<usize> = (0..n)
        .map(|row_i| {
            let mut best_k = 0;
            let mut best_p = train_probabilities[(row_i, 0)];
            for col in 1..k {
                let p = train_probabilities[(row_i, col)];
                if p > best_p {
                    best_p = p;
                    best_k = col;
                }
            }
            classes[best_k]
        })
        .collect();

    let train_accuracy = predicted_classes
        .iter()
        .zip(y.iter())
        .filter(|(&pred, &true_lbl)| pred == true_lbl)
        .count() as f64
        / n as f64;

    Ok(ElasticMultinomialResult {
        n_classes: k,
        classes,
        class_models,
        train_probabilities,
        predicted_classes,
        train_accuracy,
    })
}

/// Predict class labels for new curves using a fitted elastic multinomial model.
///
/// For each class model, calls [`predict_elastic_logistic`] to obtain P(Y=1) for
/// that class, assembles an *n\_new × K* matrix, row-normalises (same zero-guard
/// as fitting), and returns the per-row argmax mapped through `fit.classes`.
///
/// # Arguments
/// * `fit` — A fitted [`ElasticMultinomialResult`]
/// * `new_data` — New functional data (n\_new × m)
/// * `argvals` — Evaluation points (length m), same grid used for fitting
pub fn predict_elastic_multinomial(
    fit: &ElasticMultinomialResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Vec<usize> {
    let n_new = new_data.nrows();
    let k = fit.n_classes;

    // Collect OvR probabilities column by column
    let mut prob_matrix = FdMatrix::zeros(n_new, k);
    for (col_k, model_k) in fit.class_models.iter().enumerate() {
        let probs_k = predict_elastic_logistic(model_k, new_data, argvals);
        for row_i in 0..n_new {
            prob_matrix[(row_i, col_k)] = probs_k[row_i];
        }
    }

    // Row-normalise
    for row_i in 0..n_new {
        let row_sum: f64 = (0..k).map(|col| prob_matrix[(row_i, col)]).sum();
        if row_sum < 1e-15 {
            for col in 0..k {
                prob_matrix[(row_i, col)] = 1.0 / k as f64;
            }
        } else {
            let scale = 1.0 / row_sum;
            for col in 0..k {
                prob_matrix[(row_i, col)] *= scale;
            }
        }
    }

    // Argmax → class label
    (0..n_new)
        .map(|row_i| {
            let mut best_k = 0;
            let mut best_p = prob_matrix[(row_i, 0)];
            for col in 1..k {
                let p = prob_matrix[(row_i, col)];
                if p > best_p {
                    best_p = p;
                    best_k = col;
                }
            }
            fit.classes[best_k]
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    /// Build a synthetic FdMatrix where each class has a bump centred at a distinct location.
    /// Rows 0..n_per_class → class 0, next n_per_class rows → class 1, etc.
    fn make_class_data(
        n_per_class: usize,
        k_classes: usize,
        m: usize,
    ) -> (FdMatrix, Vec<usize>, Vec<f64>) {
        let argvals = uniform_grid(m);
        let n = n_per_class * k_classes;
        let mut data_col_major = vec![0.0f64; n * m];
        let mut y = vec![0usize; n];

        for cls in 0..k_classes {
            // Bump centre at a different location for each class
            let centre = (cls as f64 + 1.0) / (k_classes as f64 + 1.0);
            let width = 0.08;
            for obs in 0..n_per_class {
                let row = cls * n_per_class + obs;
                y[row] = cls;
                // tiny per-obs noise to avoid identical curves (scale: 0.03)
                let noise_seed = (row * 17 + 3) as f64 * 0.001;
                for col in 0..m {
                    let t = argvals[col];
                    let val =
                        (-((t - centre) / width).powi(2)).exp() + noise_seed * (col as f64).sin();
                    // column-major: index = row + col * n
                    data_col_major[row + col * n] = val;
                }
            }
        }

        let mat = FdMatrix::from_column_major(data_col_major, n, m).unwrap();
        (mat, y, argvals)
    }

    // ── Task 1: shape smoke ───────────────────────────────────────────────────

    #[test]
    fn elastic_multinomial_shape_smoke() {
        let (data, y, argvals) = make_class_data(2, 3, 20);
        let n = data.nrows();

        let result = elastic_multinomial(&data, &y, &argvals, 4, 0.01, 5, 1e-3)
            .expect("elastic_multinomial should succeed on valid K=3 input");

        assert_eq!(result.n_classes, 3, "n_classes must be 3");
        assert_eq!(result.classes, vec![0, 1, 2], "classes must be [0,1,2]");
        assert_eq!(result.class_models.len(), 3, "must have 3 OvR models");
        assert_eq!(
            result.train_probabilities.shape(),
            (n, 3),
            "train_probabilities must be (n, 3)"
        );
        for row_i in 0..n {
            let row_sum: f64 = (0..3)
                .map(|col| result.train_probabilities[(row_i, col)])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "row {} sum = {} (expected 1.0)",
                row_i,
                row_sum
            );
        }
        assert_eq!(
            result.predicted_classes.len(),
            n,
            "predicted_classes must have length n"
        );
        assert!(
            (0.0..=1.0).contains(&result.train_accuracy),
            "train_accuracy must be in [0,1]"
        );
    }

    // ── Task 2: predict + recovery + K=2 binary agreement ────────────────────

    #[test]
    fn elastic_multinomial_recovers_separated_classes() {
        // K=3, well-separated bumps, modest size for speed
        let (data, y, argvals) = make_class_data(3, 3, 24);

        let result =
            elastic_multinomial(&data, &y, &argvals, 4, 0.01, 8, 1e-3).expect("fit should succeed");

        // Train accuracy threshold: >= 0.8 (documented)
        assert!(
            result.train_accuracy >= 0.8,
            "train_accuracy {} < 0.8 threshold",
            result.train_accuracy
        );

        // Predict on held-out template curves (one per class, no noise)
        let n_per_class = 3usize;
        let k = 3usize;
        let m = 24;
        let argvals2 = uniform_grid(m);
        let mut new_col_major = vec![0.0f64; k * m];
        let mut expected_labels = vec![0usize; k];
        for cls in 0..k {
            let centre = (cls as f64 + 1.0) / (k as f64 + 1.0);
            let width = 0.08;
            expected_labels[cls] = cls;
            for col in 0..m {
                let t = argvals2[col];
                new_col_major[cls + col * k] = (-((t - centre) / width).powi(2)).exp();
            }
        }
        let new_data = FdMatrix::from_column_major(new_col_major, k, m).unwrap();
        let preds = predict_elastic_multinomial(&result, &new_data, &argvals2);
        assert_eq!(preds.len(), k, "predict must return k labels");
        // Each template should recover its class
        for (i, (&pred, &exp)) in preds.iter().zip(expected_labels.iter()).enumerate() {
            assert_eq!(
                pred, exp,
                "class {} template predicted as {} (expected {})",
                i, pred, exp
            );
        }
        let _ = n_per_class; // suppress unused warning
    }

    #[test]
    fn elastic_multinomial_k2_agrees_with_binary() {
        // Well-separated 2-class data: class 0 has bump near 0.25, class 1 near 0.75
        let m = 20;
        let argvals = uniform_grid(m);
        let n_per = 3usize;
        let n = n_per * 2;

        let mut data_col = vec![0.0f64; n * m];
        let mut y_multi = vec![0usize; n];
        let mut y_bin = vec![0i8; n];

        for obs in 0..n_per {
            let centre = 0.25;
            let w = 0.1;
            y_multi[obs] = 0;
            y_bin[obs] = -1;
            for col in 0..m {
                let t = argvals[col];
                data_col[obs + col * n] = (-((t - centre) / w).powi(2)).exp();
            }
        }
        for obs in 0..n_per {
            let row = n_per + obs;
            let centre = 0.75;
            let w = 0.1;
            y_multi[row] = 1;
            y_bin[row] = 1;
            for col in 0..m {
                let t = argvals[col];
                data_col[row + col * n] = (-((t - centre) / w).powi(2)).exp();
            }
        }
        let data = FdMatrix::from_column_major(data_col, n, m).unwrap();

        let ncomp_beta = 4;
        let lambda = 0.01;
        let max_iter = 8;
        let tol = 1e-3;

        let multi_fit =
            elastic_multinomial(&data, &y_multi, &argvals, ncomp_beta, lambda, max_iter, tol)
                .expect("multinomial K=2 should succeed");
        let bin_fit = elastic_logistic(&data, &y_bin, &argvals, ncomp_beta, lambda, max_iter, tol)
            .expect("binary logistic should succeed");

        // Map binary predicted (0=class0 if p<0.5, 1=class1 if p>=0.5) to usize labels
        // binary predicted_classes: 0 → y==-1 (class 0), 1 → y==1 (class 1)
        let bin_preds: Vec<usize> = bin_fit.predicted_classes.clone();

        assert_eq!(
            multi_fit.predicted_classes, bin_preds,
            "K=2 multinomial predictions must agree with binary elastic_logistic"
        );
    }

    // ── Task 3: input guards ──────────────────────────────────────────────────

    #[test]
    fn elastic_multinomial_rejects_count_mismatch() {
        let (data, _, argvals) = make_class_data(2, 2, 10);
        // y has 1 fewer element than n
        let bad_y: Vec<usize> = vec![0; data.nrows() - 1];
        let result = elastic_multinomial(&data, &bad_y, &argvals, 4, 0.0, 5, 1e-3);
        assert!(result.is_err(), "should return Err on y.len() != n");
    }

    #[test]
    fn elastic_multinomial_rejects_single_class() {
        let (data, _, argvals) = make_class_data(2, 2, 10);
        let all_zero: Vec<usize> = vec![0; data.nrows()];
        let result = elastic_multinomial(&data, &all_zero, &argvals, 4, 0.0, 5, 1e-3);
        assert!(result.is_err(), "should return Err for K<2");
    }

    #[test]
    fn elastic_multinomial_rejects_noncontiguous_labels() {
        let (data, _, argvals) = make_class_data(2, 2, 10);
        // Labels {0, 2} — gap at 1
        let mut bad_y: Vec<usize> = vec![0; data.nrows()];
        bad_y[data.nrows() - 1] = 2;
        bad_y[data.nrows() - 2] = 2;
        let result = elastic_multinomial(&data, &bad_y, &argvals, 4, 0.0, 5, 1e-3);
        assert!(
            result.is_err(),
            "should return Err for non-contiguous labels"
        );
    }

    #[test]
    fn elastic_multinomial_rejects_empty() {
        let data = FdMatrix::zeros(0, 10);
        let y: Vec<usize> = vec![];
        let argvals = uniform_grid(10);
        let result = elastic_multinomial(&data, &y, &argvals, 4, 0.0, 5, 1e-3);
        assert!(result.is_err(), "should return Err for empty input");
    }
}
