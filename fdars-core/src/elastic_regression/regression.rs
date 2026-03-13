//! Elastic scalar-on-function regression.

use crate::alignment::{dp_alignment_core, reparameterize_curve, sqrt_mean_inverse};
use crate::basis::bspline_basis;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::smooth_basis::bspline_penalty_matrix;
use nalgebra::{DMatrix, DVector};

use super::{
    apply_warps_to_srsfs, beta_converged, init_identity_warps, srsf_fitted_values, ElasticConfig,
};

use crate::alignment::srsf_transform;

/// Result of elastic scalar-on-function regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
///
/// # Errors
///
/// Returns [`crate::FdarError::InvalidDimension`] if `n < 2`, `m < 2`,
/// `y.len() != n`, `argvals.len() != m`, or `ncomp_beta < 2`.
/// Returns [`crate::FdarError::ComputationFailed`] if a regression iteration fails
/// to converge (e.g., singular penalized system).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_regression(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    ncomp_beta: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ElasticRegressionResult, crate::FdarError> {
    let (n, m) = data.shape();
    if n < 2 || m < 2 || y.len() != n || argvals.len() != m || ncomp_beta < 2 {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data/y/argvals",
            expected: "n >= 2, m >= 2, y.len() == n, argvals.len() == m, ncomp_beta >= 2"
                .to_string(),
            actual: format!(
                "n={}, m={}, y.len()={}, argvals.len()={}, ncomp_beta={}",
                n,
                m,
                y.len(),
                argvals.len(),
                ncomp_beta
            ),
        });
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
        )
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "regression_iteration",
            detail: format!("iteration {} failed", iter + 1),
        })?;

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

    Ok(ElasticRegressionResult {
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

/// Elastic scalar-on-function regression using a configuration struct.
///
/// Equivalent to [`elastic_regression`] but bundles method parameters in [`ElasticConfig`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_regression_with_config(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    config: &ElasticConfig,
) -> Result<ElasticRegressionResult, crate::FdarError> {
    elastic_regression(
        data,
        y,
        argvals,
        config.ncomp_beta,
        config.lambda,
        config.max_iter,
        config.tol,
    )
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

impl ElasticRegressionResult {
    /// Predict responses for new data. Delegates to [`predict_elastic_regression`].
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
        predict_elastic_regression(self, new_data, argvals)
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

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
    Some(coefs.iter().copied().collect())
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
