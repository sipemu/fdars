//! Elastic regression models (alignment-integrated regression).
//!
//! These models from fdasrvf align curves during the regression fitting process,
//! jointly optimizing alignment and regression coefficients.
//!
//! Key capabilities:
//! - [`elastic_regression`] — Scalar-on-function regression with elastic alignment
//! - [`elastic_logistic`] — Binary classification with elastic alignment
//! - [`elastic_pcr`] — Principal component regression after elastic alignment

use crate::alignment::{
    dp_alignment_core, karcher_mean, reparameterize_curve, sqrt_mean_inverse, srsf_transform,
    KarcherMeanResult,
};
use crate::basis::bspline_basis;
use crate::elastic_fpca::{
    horiz_fpca, joint_fpca, vert_fpca, HorizFpcaResult, JointFpcaResult, VertFpcaResult,
};
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::smooth_basis::bspline_penalty_matrix;
use nalgebra::{DMatrix, DVector};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of elastic scalar-on-function regression.
#[derive(Debug, Clone)]
pub struct ElasticRegressionResult {
    /// Intercept.
    pub alpha: f64,
    /// Regression function β(t), length m.
    pub beta: Vec<f64>,
    /// Fitted values, length n.
    pub fitted_values: Vec<f64>,
    /// Residuals, length n.
    pub residuals: Vec<f64>,
    /// Residual sum of squares.
    pub sse: f64,
    /// Coefficient of determination.
    pub r_squared: f64,
    /// Final warping functions (n × m).
    pub gammas: FdMatrix,
    /// Aligned SRSFs (n × m).
    pub aligned_srsfs: FdMatrix,
    /// Number of iterations used.
    pub n_iter: usize,
}

/// Result of elastic logistic regression.
#[derive(Debug, Clone)]
pub struct ElasticLogisticResult {
    /// Intercept.
    pub alpha: f64,
    /// Regression function β(t), length m.
    pub beta: Vec<f64>,
    /// Predicted probabilities, length n.
    pub probabilities: Vec<f64>,
    /// Predicted class labels (-1 or 1), length n.
    pub predicted_classes: Vec<i8>,
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

/// PCA method for elastic PCR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcaMethod {
    Vertical,
    Horizontal,
    Joint,
}

/// Result of elastic principal component regression.
#[derive(Debug, Clone)]
pub struct ElasticPcrResult {
    /// Intercept.
    pub alpha: f64,
    /// Regression coefficients on PC scores, length ncomp.
    pub coefficients: Vec<f64>,
    /// Fitted values, length n.
    pub fitted_values: Vec<f64>,
    /// Residual sum of squares.
    pub sse: f64,
    /// Coefficient of determination.
    pub r_squared: f64,
    /// PCA method used.
    pub pca_method: PcaMethod,
    /// Karcher mean result.
    pub karcher: KarcherMeanResult,
    /// Vertical FPCA result (stored when method is Vertical or Joint).
    pub vert_fpca: Option<VertFpcaResult>,
    /// Horizontal FPCA result (stored when method is Horizontal or Joint).
    pub horiz_fpca: Option<HorizFpcaResult>,
    /// Joint FPCA result (stored when method is Joint).
    pub joint_fpca: Option<JointFpcaResult>,
}

// ─── Elastic Regression ─────────────────────────────────────────────────────

/// Alternating alignment + penalized regression for scalar-on-function.
///
/// Iterates:
/// 1. Align SRSFs by current warps
/// 2. Build basis inner products Φ\[i,j\] = ∫ q_aligned_i · B_j dt
/// 3. Penalized OLS for β
/// 4. Find optimal warps via `regression_warp`
/// 5. Check convergence
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Scalar responses (length n)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp_beta` — Number of B-spline basis functions for β
/// * `lambda` — Roughness penalty on β
/// * `max_iter` — Maximum iterations (default: 20)
/// * `tol` — Convergence tolerance (default: 1e-4)
pub fn elastic_regression(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    ncomp_beta: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Option<ElasticRegressionResult> {
    let (n, m) = data.shape();
    if n < 2 || m < 2 || y.len() != n || argvals.len() != m || ncomp_beta < 2 {
        return None;
    }

    let weights = simpsons_weights(argvals);
    let q_all = srsf_transform(data, argvals);

    let (b_mat, r_trimmed, actual_nbasis) = build_basis_and_penalty(argvals, ncomp_beta, m);

    let mut gammas = init_identity_warps(n, argvals);
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut beta = vec![0.0; m];
    let mut alpha = y_mean;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let (beta_new, alpha_new) = regression_iteration_step(
            &q_all,
            &gammas,
            argvals,
            &b_mat,
            &r_trimmed,
            &weights,
            y,
            alpha,
            lambda,
            n,
            m,
            actual_nbasis,
        )?;

        if beta_converged(&beta_new, &beta, tol) && iter > 0 {
            beta = beta_new;
            alpha = alpha_new;
            break;
        }

        beta = beta_new;
        alpha = alpha_new;

        update_regression_warps(&mut gammas, &q_all, &beta, argvals, alpha, y, lambda * 0.01);
        center_warps(&mut gammas, argvals);
    }

    // Final fitted values
    let aligned_srsfs = apply_warps_to_srsfs(&q_all, &gammas, argvals);
    let fitted_values = srsf_fitted_values(&aligned_srsfs, &beta, &weights, alpha);
    let (residuals, sse, r_squared) = compute_regression_residuals(y, &fitted_values, y_mean);

    Some(ElasticRegressionResult {
        alpha,
        beta,
        fitted_values,
        residuals,
        sse,
        r_squared,
        gammas,
        aligned_srsfs,
        n_iter,
    })
}

// ─── Elastic Logistic Regression ────────────────────────────────────────────

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
pub fn elastic_logistic(
    data: &FdMatrix,
    y: &[i8],
    argvals: &[f64],
    _ncomp_beta: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Option<ElasticLogisticResult> {
    let (n, m) = data.shape();
    if n < 2 || m < 2 || y.len() != n || argvals.len() != m {
        return None;
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

    Some(ElasticLogisticResult {
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

// ─── Elastic PCR ────────────────────────────────────────────────────────────

/// Elastic principal component regression.
///
/// Performs Karcher mean alignment, then FPCA (vert/horiz/joint), then OLS
/// on the PC scores.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Scalar responses (length n)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of PCs to use
/// * `pca_method` — Which FPCA variant to use
/// * `lambda` — Alignment penalty (passed to karcher_mean)
/// * `max_iter` — Maximum iterations for karcher_mean
/// * `tol` — Convergence tolerance for karcher_mean
pub fn elastic_pcr(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    ncomp: usize,
    pca_method: PcaMethod,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Option<ElasticPcrResult> {
    let (n, _m) = data.shape();
    if n < 2 || y.len() != n || ncomp < 1 {
        return None;
    }

    // Karcher mean alignment
    let km = karcher_mean(data, argvals, max_iter, tol, lambda);

    // FPCA
    let mut stored_vert: Option<VertFpcaResult> = None;
    let mut stored_horiz: Option<HorizFpcaResult> = None;
    let mut stored_joint: Option<JointFpcaResult> = None;

    let scores_mat = match pca_method {
        PcaMethod::Vertical => {
            let fpca = vert_fpca(&km, argvals, ncomp)?;
            let scores = fpca.scores.clone();
            stored_vert = Some(fpca);
            scores
        }
        PcaMethod::Horizontal => {
            let fpca = horiz_fpca(&km, argvals, ncomp)?;
            let scores = fpca.scores.clone();
            stored_horiz = Some(fpca);
            scores
        }
        PcaMethod::Joint => {
            let fpca = joint_fpca(&km, argvals, ncomp, None)?;
            let scores = fpca.scores.clone();
            stored_joint = Some(fpca);
            scores
        }
    };

    let actual_ncomp = scores_mat.ncols();
    let (coefs, alpha, fitted_values, sse, r_squared) =
        ols_on_scores(&scores_mat, y, n, actual_ncomp)?;

    Some(ElasticPcrResult {
        alpha,
        coefficients: coefs,
        fitted_values,
        sse,
        r_squared,
        pca_method,
        karcher: km,
        vert_fpca: stored_vert,
        horiz_fpca: stored_horiz,
        joint_fpca: stored_joint,
    })
}

// ─── Internal: Regression Warp ──────────────────────────────────────────────

/// Find optimal warp for a single curve in elastic regression.
///
/// Aligns q_i to both +β and -β via DP, then binary searches between the
/// two extreme warps to find the one giving predicted y closest to actual.
fn regression_warp(
    q_i: &[f64],
    beta: &[f64],
    argvals: &[f64],
    alpha: f64,
    y_i: f64,
    lambda: f64,
) -> Vec<f64> {
    let weights = simpsons_weights(argvals);

    // Align to +β
    let gam_pos = dp_alignment_core(beta, q_i, argvals, lambda);

    // Align to -β
    let neg_beta: Vec<f64> = beta.iter().map(|&b| -b).collect();
    let gam_neg = dp_alignment_core(&neg_beta, q_i, argvals, lambda);

    // Compute predicted y for each extreme
    let y_pos = compute_predicted_y(q_i, beta, &gam_pos, argvals, alpha, &weights);
    let y_neg = compute_predicted_y(q_i, beta, &gam_neg, argvals, alpha, &weights);

    // If already close enough, return the nearest extreme
    if let Some(gam) = check_extreme_warps(&gam_pos, &gam_neg, y_pos, y_neg, y_i) {
        return gam;
    }

    // Binary search between the two warps
    let (gam_lo, gam_hi) = order_warps_by_prediction(gam_pos, gam_neg, y_pos, y_neg);
    binary_search_warps(gam_lo, gam_hi, q_i, beta, argvals, alpha, y_i, &weights)
}

/// Compute predicted y for a warped curve.
fn compute_predicted_y(
    q_i: &[f64],
    beta: &[f64],
    gam: &[f64],
    argvals: &[f64],
    alpha: f64,
    weights: &[f64],
) -> f64 {
    let m = argvals.len();
    let q_warped = reparameterize_curve(q_i, argvals, gam);
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_deriv = crate::helpers::gradient_uniform(gam, h);

    let mut y_hat = alpha;
    for j in 0..m {
        let q_aligned_j = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
        y_hat += q_aligned_j * beta[j] * weights[j];
    }
    y_hat
}

// ─── Shared Helpers ────────────────────────────────────────────────────────

/// Apply warping functions to SRSFs, producing aligned SRSFs with sqrt(γ') factor.
fn apply_warps_to_srsfs(q_all: &FdMatrix, gammas: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let (n, m) = q_all.shape();
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let mut q_aligned = FdMatrix::zeros(n, m);
    for i in 0..n {
        let qi: Vec<f64> = (0..m).map(|j| q_all[(i, j)]).collect();
        let gam: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let q_warped = reparameterize_curve(&qi, argvals, &gam);
        let gam_deriv = crate::helpers::gradient_uniform(&gam, h);
        for j in 0..m {
            q_aligned[(i, j)] = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
        }
    }
    q_aligned
}

/// Initialize warping functions to identity (γ_i(t) = t).
fn init_identity_warps(n: usize, argvals: &[f64]) -> FdMatrix {
    let m = argvals.len();
    let mut gammas = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            gammas[(i, j)] = argvals[j];
        }
    }
    gammas
}

/// Trim or pad penalty matrix to match actual basis dimension.
fn trim_penalty_to_basis(
    r_mat: &DMatrix<f64>,
    penalty_k: usize,
    actual_nbasis: usize,
) -> DMatrix<f64> {
    if penalty_k >= actual_nbasis {
        r_mat
            .view((0, 0), (actual_nbasis, actual_nbasis))
            .into_owned()
    } else {
        let mut r = DMatrix::zeros(actual_nbasis, actual_nbasis);
        let dim = penalty_k.min(actual_nbasis);
        for i in 0..dim {
            for j in 0..dim {
                r[(i, j)] = r_mat[(i, j)];
            }
        }
        r
    }
}

/// Build design matrix Φ[i,k] = ∫ q_aligned_i · B_k · w dt.
fn build_phi_matrix(
    q_aligned: &FdMatrix,
    b_mat: &DMatrix<f64>,
    weights: &[f64],
    n: usize,
    m: usize,
    actual_nbasis: usize,
) -> DMatrix<f64> {
    let mut phi = DMatrix::zeros(n, actual_nbasis);
    for i in 0..n {
        for k in 0..actual_nbasis {
            let mut val = 0.0;
            for j in 0..m {
                val += q_aligned[(i, j)] * b_mat[(j, k)] * weights[j];
            }
            phi[(i, k)] = val;
        }
    }
    phi
}

/// Solve penalized OLS: (Φ'Φ + λR)c = Φ'y.
fn solve_penalized_ols(
    phi: &DMatrix<f64>,
    r_trimmed: &DMatrix<f64>,
    y_centered: &[f64],
    lambda: f64,
) -> Option<Vec<f64>> {
    let y_vec = DVector::from_vec(y_centered.to_vec());
    let phi_t_phi = phi.transpose() * phi;
    let system = &phi_t_phi + lambda * r_trimmed;
    let rhs = phi.transpose() * &y_vec;
    let coefs = if let Some(chol) = system.clone().cholesky() {
        chol.solve(&rhs)
    } else {
        let svd = nalgebra::SVD::new(system, true, true);
        svd.solve(&rhs, 1e-10).ok()?
    };
    Some(coefs.iter().cloned().collect())
}

/// Reconstruct β(t) = Σ c_k B_k(t) from B-spline coefficients.
fn reconstruct_beta_from_coefs(
    coefs: &[f64],
    b_mat: &DMatrix<f64>,
    m: usize,
    actual_nbasis: usize,
) -> Vec<f64> {
    let mut beta = vec![0.0; m];
    for j in 0..m {
        for k in 0..actual_nbasis {
            beta[j] += coefs[k] * b_mat[(j, k)];
        }
    }
    beta
}

/// Compute intercept: α̂ = mean(y - ∫ q·β·w dt).
fn compute_alpha_from_residuals(
    q_aligned: &FdMatrix,
    beta: &[f64],
    weights: &[f64],
    y: &[f64],
) -> f64 {
    let (n, m) = q_aligned.shape();
    let mut alpha = 0.0;
    for i in 0..n {
        let mut y_hat_i = 0.0;
        for j in 0..m {
            y_hat_i += q_aligned[(i, j)] * beta[j] * weights[j];
        }
        alpha += y[i] - y_hat_i;
    }
    alpha / n as f64
}

/// Compute fitted values: ŷ_i = α + ∫ q_aligned_i · β · w dt.
fn srsf_fitted_values(q_aligned: &FdMatrix, beta: &[f64], weights: &[f64], alpha: f64) -> Vec<f64> {
    let (n, m) = q_aligned.shape();
    let mut fitted = vec![0.0; n];
    for i in 0..n {
        fitted[i] = alpha;
        for j in 0..m {
            fitted[i] += q_aligned[(i, j)] * beta[j] * weights[j];
        }
    }
    fitted
}

/// Check relative convergence of β.
fn beta_converged(beta_new: &[f64], beta_old: &[f64], tol: f64) -> bool {
    let diff: f64 = beta_new
        .iter()
        .zip(beta_old.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = beta_old
        .iter()
        .map(|&b| b * b)
        .sum::<f64>()
        .sqrt()
        .max(1e-10);
    diff / norm < tol
}

/// Center warping functions using Karcher mean.
fn center_warps(gammas: &mut FdMatrix, argvals: &[f64]) {
    let (n, m) = gammas.shape();
    let gam_mu = sqrt_mean_inverse(gammas, argvals);
    for i in 0..n {
        let gam_i: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let composed = crate::alignment::compose_warps(&gam_i, &gam_mu, argvals);
        for j in 0..m {
            gammas[(i, j)] = composed[j];
        }
    }
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

/// OLS regression on PC scores: returns (coefs, alpha, fitted, sse, r²).
fn ols_on_scores(
    scores_mat: &FdMatrix,
    y: &[f64],
    n: usize,
    ncomp: usize,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64, f64)> {
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut score_means = vec![0.0; ncomp];
    for k in 0..ncomp {
        for i in 0..n {
            score_means[k] += scores_mat[(i, k)];
        }
        score_means[k] /= n as f64;
    }

    let mut x_cen = DMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            x_cen[(i, k)] = scores_mat[(i, k)] - score_means[k];
        }
    }
    let y_cen: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let y_vec = DVector::from_vec(y_cen);

    let xtx = x_cen.transpose() * &x_cen;
    let xty = x_cen.transpose() * &y_vec;
    let coefficients = if let Some(chol) = xtx.clone().cholesky() {
        chol.solve(&xty)
    } else {
        let svd = nalgebra::SVD::new(xtx, true, true);
        svd.solve(&xty, 1e-10).ok()?
    };
    let coefs: Vec<f64> = coefficients.iter().cloned().collect();

    let alpha = y_mean
        - coefs
            .iter()
            .zip(score_means.iter())
            .map(|(&c, &sm)| c * sm)
            .sum::<f64>();

    let mut fitted_values = vec![0.0; n];
    for i in 0..n {
        fitted_values[i] = alpha;
        for k in 0..ncomp {
            fitted_values[i] += coefs[k] * scores_mat[(i, k)];
        }
    }

    let sse: f64 = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &yh)| (yi - yh).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - sse / ss_tot
    } else {
        0.0
    };

    Some((coefs, alpha, fitted_values, sse, r_squared))
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

/// One iteration step of elastic regression: align, solve OLS, return new (β, α).
fn regression_iteration_step(
    q_all: &FdMatrix,
    gammas: &FdMatrix,
    argvals: &[f64],
    b_mat: &DMatrix<f64>,
    r_trimmed: &DMatrix<f64>,
    weights: &[f64],
    y: &[f64],
    alpha: f64,
    lambda: f64,
    n: usize,
    m: usize,
    actual_nbasis: usize,
) -> Option<(Vec<f64>, f64)> {
    let q_aligned = apply_warps_to_srsfs(q_all, gammas, argvals);
    let phi = build_phi_matrix(&q_aligned, b_mat, weights, n, m, actual_nbasis);
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - alpha).collect();
    let coefs = solve_penalized_ols(&phi, r_trimmed, &y_centered, lambda)?;
    let beta_new = reconstruct_beta_from_coefs(&coefs, b_mat, m, actual_nbasis);
    let alpha_new = compute_alpha_from_residuals(&q_aligned, &beta_new, weights, y);
    Some((beta_new, alpha_new))
}

/// Build B-spline basis matrix and roughness penalty for β representation.
fn build_basis_and_penalty(
    argvals: &[f64],
    ncomp_beta: usize,
    m: usize,
) -> (DMatrix<f64>, DMatrix<f64>, usize) {
    let nknots = ncomp_beta.saturating_sub(4).max(2);
    let basis_flat = bspline_basis(argvals, nknots, 4);
    let actual_nbasis = basis_flat.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis_flat);

    let penalty_flat = bspline_penalty_matrix(argvals, ncomp_beta, 4, 2);
    let penalty_k = (penalty_flat.len() as f64).sqrt() as usize;
    let r_mat = DMatrix::from_column_slice(penalty_k, penalty_k, &penalty_flat);
    let r_trimmed = trim_penalty_to_basis(&r_mat, penalty_k, actual_nbasis);

    (b_mat, r_trimmed, actual_nbasis)
}

/// Update warping functions for all curves in elastic regression.
fn update_regression_warps(
    gammas: &mut FdMatrix,
    q_all: &FdMatrix,
    beta: &[f64],
    argvals: &[f64],
    alpha: f64,
    y: &[f64],
    lambda: f64,
) {
    let (n, m) = q_all.shape();
    for i in 0..n {
        let qi: Vec<f64> = (0..m).map(|j| q_all[(i, j)]).collect();
        let new_gam = regression_warp(&qi, beta, argvals, alpha, y[i], lambda);
        for j in 0..m {
            gammas[(i, j)] = new_gam[j];
        }
    }
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
        let beta_signed: Vec<f64> = beta.iter().map(|&b| b * y[i] as f64).collect();
        let new_gam = dp_alignment_core(&beta_signed, &qi, argvals, lambda);
        for j in 0..m {
            gammas[(i, j)] = new_gam[j];
        }
    }
}

/// Compute residuals, SSE, and R² from y and fitted values.
fn compute_regression_residuals(
    y: &[f64],
    fitted_values: &[f64],
    y_mean: f64,
) -> (Vec<f64>, f64, f64) {
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let sse: f64 = residuals.iter().map(|&r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - sse / ss_tot
    } else {
        0.0
    };
    (residuals, sse, r_squared)
}

/// Compute final logistic predictions: probabilities, classes, accuracy, loss.
fn compute_logistic_predictions(
    aligned_srsfs: &FdMatrix,
    beta: &[f64],
    weights: &[f64],
    alpha: f64,
    y: &[i8],
    lambda: f64,
) -> (Vec<f64>, Vec<i8>, f64, f64) {
    let n = y.len();
    let eta = srsf_fitted_values(aligned_srsfs, beta, weights, alpha);
    let probabilities: Vec<f64> = eta.iter().map(|&e| 1.0 / (1.0 + (-e).exp())).collect();
    let predicted_classes: Vec<i8> = probabilities
        .iter()
        .map(|&p| if p >= 0.5 { 1 } else { -1 })
        .collect();
    let accuracy = predicted_classes
        .iter()
        .zip(y.iter())
        .filter(|(&p, &t)| p == t)
        .count() as f64
        / n as f64;
    let loss = logistic_loss(&probabilities, y, beta, lambda);
    (probabilities, predicted_classes, accuracy, loss)
}

/// Check if either extreme warp is already close enough to target y.
fn check_extreme_warps(
    gam_pos: &[f64],
    gam_neg: &[f64],
    y_pos: f64,
    y_neg: f64,
    y_i: f64,
) -> Option<Vec<f64>> {
    if (y_pos - y_i).abs() <= (y_neg - y_i).abs() {
        if (y_pos - y_i).abs() < 1e-10 {
            return Some(gam_pos.to_vec());
        }
    } else if (y_neg - y_i).abs() < 1e-10 {
        return Some(gam_neg.to_vec());
    }
    None
}

/// Order warps so gam_lo gives lower prediction and gam_hi gives higher.
fn order_warps_by_prediction(
    gam_pos: Vec<f64>,
    gam_neg: Vec<f64>,
    y_pos: f64,
    y_neg: f64,
) -> (Vec<f64>, Vec<f64>) {
    if y_pos < y_neg {
        (gam_pos, gam_neg)
    } else {
        (gam_neg, gam_pos)
    }
}

/// Binary search between two warps to find one giving predicted y closest to target.
fn binary_search_warps(
    mut gam_lo: Vec<f64>,
    mut gam_hi: Vec<f64>,
    q_i: &[f64],
    beta: &[f64],
    argvals: &[f64],
    alpha: f64,
    y_i: f64,
    weights: &[f64],
) -> Vec<f64> {
    for _ in 0..15 {
        let gam_mid: Vec<f64> = gam_lo
            .iter()
            .zip(gam_hi.iter())
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect();
        let y_mid = compute_predicted_y(q_i, beta, &gam_mid, argvals, alpha, weights);
        if (y_mid - y_i).abs() < 1e-6 {
            return gam_mid;
        }
        if y_mid < y_i {
            gam_lo = gam_mid;
        } else {
            gam_hi = gam_mid;
        }
    }
    gam_lo
        .iter()
        .zip(gam_hi.iter())
        .map(|(&lo, &hi)| 0.5 * (lo + hi))
        .collect()
}

/// Predict new responses using a fitted elastic regression model.
///
/// Transforms new curves to SRSFs, aligns them using the training warps as
/// a template (identity alignment for new data), then applies the fitted
/// regression coefficients.
///
/// # Arguments
/// * `fit` — A fitted [`ElasticRegressionResult`]
/// * `new_data` — New functional data (n_new × m)
/// * `argvals` — Evaluation points (length m)
pub fn predict_elastic_regression(
    fit: &ElasticRegressionResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Vec<f64> {
    let weights = simpsons_weights(argvals);
    let q_new = srsf_transform(new_data, argvals);
    srsf_fitted_values(&q_new, &fit.beta, &weights, fit.alpha)
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

impl ElasticRegressionResult {
    /// Predict responses for new data. Delegates to [`predict_elastic_regression`].
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
        predict_elastic_regression(self, new_data, argvals)
    }
}

impl ElasticLogisticResult {
    /// Predict probabilities for new data. Delegates to [`predict_elastic_logistic`].
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
        predict_elastic_logistic(self, new_data, argvals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];

        for i in 0..n {
            let amp = 1.0 + 0.5 * (i as f64 / n as f64);
            let shift = 0.1 * (i as f64 - n as f64 / 2.0);
            for j in 0..m {
                data[(i, j)] = amp * (2.0 * PI * (t[j] + shift)).sin();
            }
            y[i] = amp; // Response related to amplitude
        }
        (data, y, t)
    }

    #[test]
    fn test_elastic_regression_basic() {
        let (data, y, t) = generate_test_data(15, 51);
        let result = elastic_regression(&data, &y, &t, 10, 1e-3, 5, 1e-3);
        assert!(result.is_some(), "elastic_regression should succeed");

        let res = result.unwrap();
        assert_eq!(res.fitted_values.len(), 15);
        assert_eq!(res.beta.len(), 51);
        assert_eq!(res.gammas.shape(), (15, 51));
        assert!(res.n_iter > 0);
    }

    #[test]
    fn test_elastic_logistic_basic() {
        let n = 20;
        let m = 51;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0_i8; n];

        for i in 0..n {
            let label = if i < n / 2 { -1_i8 } else { 1_i8 };
            y[i] = label;
            let amp = if label == 1 { 2.0 } else { 1.0 };
            for j in 0..m {
                data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
            }
        }

        let result = elastic_logistic(&data, &y, &t, 10, 1e-2, 5, 1e-3);
        assert!(result.is_some(), "elastic_logistic should succeed");

        let res = result.unwrap();
        assert_eq!(res.probabilities.len(), n);
        assert_eq!(res.predicted_classes.len(), n);
        assert!(res.accuracy >= 0.0 && res.accuracy <= 1.0);
    }

    #[test]
    fn test_elastic_pcr_vertical() {
        let (data, y, t) = generate_test_data(15, 51);
        let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3);
        assert!(result.is_some(), "elastic_pcr (vertical) should succeed");

        let res = result.unwrap();
        assert_eq!(res.fitted_values.len(), 15);
        assert_eq!(res.coefficients.len(), 3);
    }

    #[test]
    fn test_elastic_pcr_horizontal() {
        let (data, y, t) = generate_test_data(15, 51);
        let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Horizontal, 0.0, 5, 1e-3);
        assert!(result.is_some(), "elastic_pcr (horizontal) should succeed");
    }

    #[test]
    fn test_elastic_regression_invalid() {
        let data = FdMatrix::zeros(1, 10);
        let y = vec![1.0];
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        assert!(elastic_regression(&data, &y, &t, 5, 1e-3, 5, 1e-3).is_none());
    }
}
