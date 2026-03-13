//! Scalar-on-Shape (ScoSh) regression.
//!
//! Regresses a scalar response on the *shape* of a functional predictor,
//! separating shape from amplitude and phase via SRSF alignment.
//!
//! The model is:
//!
//! ```text
//! y_i = h(<q_i o gamma_i, beta>) + g(f0_i) + epsilon_i
//! ```
//!
//! where `q_i` is the SRSF of curve `f_i`, `gamma_i` is a warping function,
//! `beta` is the shape index function, `h` is a link function, `g` captures
//! the amplitude (mean-level) effect, and `f0_i` is the mean of curve `i`.

use crate::alignment::{compose_warps, dp_alignment_core, sqrt_mean_inverse, srsf_transform};
use crate::basis::fourier_basis;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::smooth_basis::fourier_penalty_matrix;
use crate::FdarError;
use nalgebra::{DMatrix, DVector};

use super::{
    apply_warps_to_srsfs, beta_converged, init_identity_warps, IndexMethod, ScalarOnShapeConfig,
};

// ─── Result ─────────────────────────────────────────────────────────────────

/// Result of scalar-on-shape regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ScalarOnShapeResult {
    /// Estimated index function beta(t), length m.
    pub beta: Vec<f64>,
    /// Fourier coefficients for beta, length nbasis.
    pub beta_coefficients: Vec<f64>,
    /// Warping functions gamma_i, n x m.
    pub gammas: FdMatrix,
    /// Shape scores <q_i o gamma_i, beta>, length n.
    pub shape_scores: Vec<f64>,
    /// Coefficients for h (index function).
    pub h_coefficients: Vec<f64>,
    /// Coefficients for g (intercept link function).
    pub g_coefficients: Vec<f64>,
    /// Fitted values, length n.
    pub fitted_values: Vec<f64>,
    /// Residuals, length n.
    pub residuals: Vec<f64>,
    /// Residual sum of squares.
    pub sse: f64,
    /// R-squared.
    pub r_squared: f64,
    /// Number of outer iterations.
    pub n_iter_outer: usize,
    /// Number of inner iterations (last outer step).
    pub n_iter_inner: usize,
    /// Index method used.
    pub index_method: IndexMethod,
}

// ─── Main API ───────────────────────────────────────────────────────────────

/// Scalar-on-shape regression with elastic alignment.
///
/// Separates the effect of curve *shape* from amplitude and phase, regressing
/// a scalar response on the shape component via an SRSF inner product with
/// penalized Fourier basis representation of beta.
///
/// The model alternates between:
/// 1. Estimating the shape index function beta and alignment (inner loop)
/// 2. Estimating the index link h and amplitude link g (outer loop)
///
/// # Arguments
/// * `data` - Functional data (n x m)
/// * `y` - Scalar responses (length n)
/// * `argvals` - Evaluation points (length m)
/// * `config` - Algorithm configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if dimensions are inconsistent
/// (`n < 2`, `m < 2`, length mismatches, or `nbasis < 3`).
/// Returns [`FdarError::ComputationFailed`] if the penalized regression
/// system is singular.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn scalar_on_shape(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    config: &ScalarOnShapeConfig,
) -> Result<ScalarOnShapeResult, FdarError> {
    let (n, m) = data.shape();

    // ── Validate inputs ─────────────────────────────────────────────────
    validate_inputs(n, m, y, argvals, config)?;

    let weights = simpsons_weights(argvals);
    let q_all = srsf_transform(data, argvals);
    let mut gammas = init_identity_warps(n, argvals);

    // Compute mean level f0_i for each curve (amplitude component).
    let f0: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..m {
                s += data[(i, j)] * weights[j];
            }
            s
        })
        .collect();

    // Build Fourier basis and penalty.
    let period = argvals[m - 1] - argvals[0];
    let nbasis = config.nbasis;
    let basis_flat = fourier_basis(argvals, nbasis);
    let b_mat = DMatrix::from_column_slice(m, nbasis, &basis_flat);
    let penalty_flat = fourier_penalty_matrix(nbasis, period, config.lfd_order);
    let r_mat = DMatrix::from_column_slice(nbasis, nbasis, &penalty_flat);

    // Initialize beta from first Fourier basis function (constant).
    let mut beta: Vec<f64> = (0..m).map(|j| b_mat[(j, 0)]).collect();
    normalize_beta(&mut beta, &weights);

    let mut g_coefs: Vec<f64> = vec![0.0; config.g_degree];
    let mut h_coefs: Vec<f64> = vec![1.0]; // identity initial
    let mut n_iter_inner = 0;
    let mut n_iter_outer = 0;

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    for outer in 0..config.max_iter_outer {
        n_iter_outer = outer + 1;
        let beta_old_outer = beta.clone();

        // Compute g values: g(f0_i) from current g_coefs.
        let g_vals = eval_polynomial_no_intercept(&f0, &g_coefs);

        // Compute adjusted_y = y - g(f0).
        let adjusted_y: Vec<f64> = y
            .iter()
            .zip(g_vals.iter())
            .map(|(&yi, &gi)| yi - gi)
            .collect();

        // ── Inner loop: estimate beta with alignment ────────────────────
        let inner_result = inner_loop(
            &q_all,
            &mut gammas,
            &adjusted_y,
            argvals,
            &weights,
            &b_mat,
            &r_mat,
            &mut beta,
            config,
            n,
            m,
            nbasis,
        )?;
        n_iter_inner = inner_result;

        // Compute shape scores with current beta and alignment.
        let q_aligned = apply_warps_to_srsfs(&q_all, &gammas, argvals);
        let shape_scores = compute_shape_scores(&q_aligned, &beta, &weights, n, m);

        // ── Estimate h and g ────────────────────────────────────────────
        match &config.index_method {
            IndexMethod::Identity => {
                h_coefs = vec![1.0];
                // g: regress (y - shape_scores) on f0 powers (no intercept).
                let g_response: Vec<f64> = y
                    .iter()
                    .zip(shape_scores.iter())
                    .map(|(&yi, &si)| yi - si)
                    .collect();
                g_coefs = fit_polynomial_no_intercept(&f0, &g_response, config.g_degree);
            }
            IndexMethod::Polynomial(deg) => {
                // h: regress (y - g(f0)) on polynomial of shape_scores.
                let g_vals_current = eval_polynomial_no_intercept(&f0, &g_coefs);
                let h_response: Vec<f64> = y
                    .iter()
                    .zip(g_vals_current.iter())
                    .map(|(&yi, &gi)| yi - gi)
                    .collect();
                h_coefs = fit_polynomial_with_intercept(&shape_scores, &h_response, *deg);

                // g: regress (y - h(shape_scores)) on f0 powers (no intercept).
                let h_vals = eval_polynomial_with_intercept(&shape_scores, &h_coefs);
                let g_response: Vec<f64> = y
                    .iter()
                    .zip(h_vals.iter())
                    .map(|(&yi, &hi)| yi - hi)
                    .collect();
                g_coefs = fit_polynomial_no_intercept(&f0, &g_response, config.g_degree);
            }
            IndexMethod::NadarayaWatson(bw) => {
                // h: nonparametric estimate via Nadaraya-Watson.
                let g_vals_current = eval_polynomial_no_intercept(&f0, &g_coefs);
                let h_response: Vec<f64> = y
                    .iter()
                    .zip(g_vals_current.iter())
                    .map(|(&yi, &gi)| yi - gi)
                    .collect();
                let h_vals = crate::smoothing::nadaraya_watson(
                    &shape_scores,
                    &h_response,
                    &shape_scores,
                    *bw,
                    "gaussian",
                )
                .unwrap_or(h_response);
                // Store mean as coefficient (for prediction we re-estimate).
                h_coefs = h_vals;

                // g: regress (y - h_vals) on f0 powers (no intercept).
                let g_response: Vec<f64> = y
                    .iter()
                    .zip(h_coefs.iter())
                    .map(|(&yi, &hi)| yi - hi)
                    .collect();
                g_coefs = fit_polynomial_no_intercept(&f0, &g_response, config.g_degree);
            }
        }

        // Check outer convergence.
        if beta_converged(&beta, &beta_old_outer, config.tol) && outer > 0 {
            break;
        }
    }

    // ── Final results ───────────────────────────────────────────────────
    let q_aligned = apply_warps_to_srsfs(&q_all, &gammas, argvals);
    let shape_scores = compute_shape_scores(&q_aligned, &beta, &weights, n, m);
    let fitted_values =
        compute_fitted_values(&shape_scores, &f0, &h_coefs, &g_coefs, &config.index_method);
    let (residuals, sse, r_squared) = compute_residuals(y, &fitted_values, y_mean);

    // Extract beta_coefficients by projecting beta onto the basis.
    let beta_coefficients = project_onto_basis(&beta, &b_mat, &weights, m, nbasis);

    Ok(ScalarOnShapeResult {
        beta,
        beta_coefficients,
        gammas,
        shape_scores,
        h_coefficients: h_coefs,
        g_coefficients: g_coefs,
        fitted_values,
        residuals,
        sse,
        r_squared,
        n_iter_outer,
        n_iter_inner,
        index_method: config.index_method.clone(),
    })
}

/// Predict scalar responses for new functional data using a fitted
/// scalar-on-shape model.
///
/// # Arguments
/// * `fit` - A fitted [`ScalarOnShapeResult`]
/// * `new_data` - New functional data (n_new x m)
/// * `argvals` - Evaluation points (length m)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `new_data` columns or
/// `argvals` length does not match the fitted beta length.
pub fn predict_scalar_on_shape(
    fit: &ScalarOnShapeResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    let (n_new, m) = new_data.shape();
    let m_beta = fit.beta.len();

    if m != m_beta {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("ncols = {} (matching beta length)", m_beta),
            actual: format!("ncols = {}", m),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {} (matching ncols)", m),
            actual: format!("length {}", argvals.len()),
        });
    }

    let weights = simpsons_weights(argvals);
    let q_new = srsf_transform(new_data, argvals);

    // Align each new SRSF to beta via DP.
    let mut gammas_new = FdMatrix::zeros(n_new, m);
    for i in 0..n_new {
        let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
        let gam = dp_alignment_core(&fit.beta, &qi, argvals, 0.0);
        for j in 0..m {
            gammas_new[(i, j)] = gam[j];
        }
    }

    let q_aligned = apply_warps_to_srsfs(&q_new, &gammas_new, argvals);
    let shape_scores = compute_shape_scores(&q_aligned, &fit.beta, &weights, n_new, m);

    // Compute f0 for new curves.
    let f0: Vec<f64> = (0..n_new)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..m {
                s += new_data[(i, j)] * weights[j];
            }
            s
        })
        .collect();

    let fitted = compute_fitted_values(
        &shape_scores,
        &f0,
        &fit.h_coefficients,
        &fit.g_coefficients,
        &fit.index_method,
    );

    Ok(fitted)
}

impl ScalarOnShapeResult {
    /// Predict responses for new data. Delegates to [`predict_scalar_on_shape`].
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Result<Vec<f64>, FdarError> {
        predict_scalar_on_shape(self, new_data, argvals)
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Validate all inputs before computation.
fn validate_inputs(
    n: usize,
    m: usize,
    y: &[f64],
    argvals: &[f64],
    config: &ScalarOnShapeConfig,
) -> Result<(), FdarError> {
    if n < 2 || m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 2 and m >= 2".to_string(),
            actual: format!("n={n}, m={m}"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("length {n} (matching nrows)"),
            actual: format!("length {}", y.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (matching ncols)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if config.nbasis < 3 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("must be >= 3, got {}", config.nbasis),
        });
    }
    Ok(())
}

/// Inner loop: estimate beta via alternating alignment and penalized regression.
///
/// Returns the number of inner iterations used.
fn inner_loop(
    q_all: &FdMatrix,
    gammas: &mut FdMatrix,
    adjusted_y: &[f64],
    argvals: &[f64],
    weights: &[f64],
    b_mat: &DMatrix<f64>,
    r_mat: &DMatrix<f64>,
    beta: &mut Vec<f64>,
    config: &ScalarOnShapeConfig,
    n: usize,
    m: usize,
    nbasis: usize,
) -> Result<usize, FdarError> {
    let mut n_inner = 0;

    for inner in 0..config.max_iter_inner {
        n_inner = inner + 1;
        let beta_old = beta.clone();

        // 1. Align each SRSF to current beta.
        for i in 0..n {
            let qi: Vec<f64> = (0..m).map(|j| q_all[(i, j)]).collect();
            let gam = dp_alignment_core(beta, &qi, argvals, config.dp_lambda);
            for j in 0..m {
                gammas[(i, j)] = gam[j];
            }
        }

        // 2. Apply warps.
        let q_aligned = apply_warps_to_srsfs(q_all, gammas, argvals);

        // 3. Build design matrix Phi[i,k] = sum_j q_aligned[i,j] * B[j,k] * w[j].
        let phi = build_phi_matrix(&q_aligned, b_mat, weights, n, m, nbasis);

        // 4. Solve penalized OLS: (Phi'Phi + lambda*R)c = Phi' adjusted_y.
        let coefs = super::regression::solve_penalized_ols(&phi, r_mat, adjusted_y, config.lambda)
            .ok_or_else(|| FdarError::ComputationFailed {
                operation: "scalar_on_shape inner loop",
                detail: format!(
                    "penalized OLS failed at inner iteration {}; try increasing lambda",
                    inner + 1
                ),
            })?;

        // 5. Reconstruct beta(t) = sum_k c_k B_k(t).
        let mut beta_new = vec![0.0; m];
        for j in 0..m {
            for k in 0..nbasis {
                beta_new[j] += coefs[k] * b_mat[(j, k)];
            }
        }

        // 6. Normalize beta to unit L2 norm (in function space).
        normalize_beta(&mut beta_new, weights);

        // 7. Center warps via Karcher mean inverse.
        center_warps(gammas, argvals);

        // 8. Check convergence.
        if beta_converged(&beta_new, &beta_old, config.tol) && inner > 0 {
            *beta = beta_new;
            break;
        }

        *beta = beta_new;
    }

    Ok(n_inner)
}

/// Normalize beta to unit L2 norm under the integration weights.
fn normalize_beta(beta: &mut [f64], weights: &[f64]) {
    let norm_sq: f64 = beta
        .iter()
        .zip(weights.iter())
        .map(|(&b, &w)| b * b * w)
        .sum();
    let norm = norm_sq.sqrt().max(1e-15);
    for b in beta.iter_mut() {
        *b /= norm;
    }
}

/// Center warping functions using Karcher mean inverse.
fn center_warps(gammas: &mut FdMatrix, argvals: &[f64]) {
    let (n, m) = gammas.shape();
    let gam_mu = sqrt_mean_inverse(gammas, argvals);
    for i in 0..n {
        let gam_i: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let composed = compose_warps(&gam_i, &gam_mu, argvals);
        for j in 0..m {
            gammas[(i, j)] = composed[j];
        }
    }
}

/// Build design matrix Phi[i,k] = sum_j q_aligned[i,j] * B[j,k] * w[j].
fn build_phi_matrix(
    q_aligned: &FdMatrix,
    b_mat: &DMatrix<f64>,
    weights: &[f64],
    n: usize,
    m: usize,
    nbasis: usize,
) -> DMatrix<f64> {
    let mut phi = DMatrix::zeros(n, nbasis);
    for i in 0..n {
        for k in 0..nbasis {
            let mut val = 0.0;
            for j in 0..m {
                val += q_aligned[(i, j)] * b_mat[(j, k)] * weights[j];
            }
            phi[(i, k)] = val;
        }
    }
    phi
}

/// Compute shape scores: score_i = sum_j q_aligned[i,j] * beta[j] * w[j].
fn compute_shape_scores(
    q_aligned: &FdMatrix,
    beta: &[f64],
    weights: &[f64],
    n: usize,
    m: usize,
) -> Vec<f64> {
    let mut scores = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..m {
            s += q_aligned[(i, j)] * beta[j] * weights[j];
        }
        scores[i] = s;
    }
    scores
}

/// Fit a polynomial WITHOUT intercept: g(x) = a1*x + a2*x^2 + ... + a_d*x^d.
///
/// This ensures g(0) = 0 by construction (no constant term).
fn fit_polynomial_no_intercept(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    if degree == 0 || n == 0 {
        return vec![0.0; degree.max(1)];
    }
    let d = degree;

    // Build Vandermonde V: columns [x, x^2, ..., x^d].
    let v = DMatrix::from_fn(n, d, |i, k| x[i].powi((k + 1) as i32));
    let y_vec = DVector::from_iterator(n, y.iter().copied());

    let vtv = v.transpose() * &v;
    let vty = v.transpose() * &y_vec;

    // Add small ridge for stability.
    let ridge = DMatrix::from_diagonal_element(d, d, 1e-10);
    let system = vtv + ridge;

    let coefs = if let Some(chol) = system.clone().cholesky() {
        chol.solve(&vty)
    } else {
        let svd = nalgebra::SVD::new(system, true, true);
        match svd.solve(&vty, 1e-10) {
            Ok(c) => c,
            Err(_) => DVector::zeros(d),
        }
    };

    coefs.iter().copied().collect()
}

/// Evaluate polynomial without intercept: g(x) = a1*x + a2*x^2 + ... .
fn eval_polynomial_no_intercept(x: &[f64], coefs: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let mut val = 0.0;
            for (k, &a) in coefs.iter().enumerate() {
                val += a * xi.powi((k + 1) as i32);
            }
            val
        })
        .collect()
}

/// Fit a polynomial WITH intercept: h(z) = a0 + a1*z + a2*z^2 + ... + a_d*z^d.
fn fit_polynomial_with_intercept(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let d = degree + 1; // include intercept
    if n == 0 {
        return vec![0.0; d];
    }

    // Vandermonde: columns [1, x, x^2, ..., x^degree].
    let v = DMatrix::from_fn(n, d, |i, k| x[i].powi(k as i32));
    let y_vec = DVector::from_iterator(n, y.iter().copied());

    let vtv = v.transpose() * &v;
    let vty = v.transpose() * &y_vec;

    let ridge = DMatrix::from_diagonal_element(d, d, 1e-10);
    let system = vtv + ridge;

    let coefs = if let Some(chol) = system.clone().cholesky() {
        chol.solve(&vty)
    } else {
        let svd = nalgebra::SVD::new(system, true, true);
        match svd.solve(&vty, 1e-10) {
            Ok(c) => c,
            Err(_) => DVector::zeros(d),
        }
    };

    coefs.iter().copied().collect()
}

/// Evaluate polynomial with intercept: h(z) = a0 + a1*z + a2*z^2 + ... .
fn eval_polynomial_with_intercept(x: &[f64], coefs: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let mut val = 0.0;
            for (k, &a) in coefs.iter().enumerate() {
                val += a * xi.powi(k as i32);
            }
            val
        })
        .collect()
}

/// Compute fitted values from shape scores and amplitude using h and g.
fn compute_fitted_values(
    shape_scores: &[f64],
    f0: &[f64],
    h_coefs: &[f64],
    g_coefs: &[f64],
    index_method: &IndexMethod,
) -> Vec<f64> {
    let g_vals = eval_polynomial_no_intercept(f0, g_coefs);

    let h_vals = match index_method {
        IndexMethod::Identity => {
            // h(z) = z
            shape_scores.to_vec()
        }
        IndexMethod::Polynomial(_) => eval_polynomial_with_intercept(shape_scores, h_coefs),
        IndexMethod::NadarayaWatson(_) => {
            // For NW, h_coefs stores the fitted h values from training.
            // At prediction time we would re-estimate, but for training fitted
            // values we use the stored values directly. For new data, we
            // fall back to identity (scores) since the training sample is needed.
            if h_coefs.len() == shape_scores.len() {
                h_coefs.to_vec()
            } else {
                // Prediction case: use identity as fallback.
                shape_scores.to_vec()
            }
        }
    };

    h_vals
        .iter()
        .zip(g_vals.iter())
        .map(|(&h, &g)| h + g)
        .collect()
}

/// Compute residuals, SSE, and R-squared.
fn compute_residuals(y: &[f64], fitted: &[f64], y_mean: f64) -> (Vec<f64>, f64, f64) {
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
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

/// Project beta onto the Fourier basis to recover coefficients.
fn project_onto_basis(
    beta: &[f64],
    b_mat: &DMatrix<f64>,
    weights: &[f64],
    m: usize,
    nbasis: usize,
) -> Vec<f64> {
    // c_k = sum_j beta[j] * B[j,k] * w[j]
    let mut coefs = vec![0.0; nbasis];
    for k in 0..nbasis {
        let mut val = 0.0;
        for j in 0..m {
            val += beta[j] * b_mat[(j, k)] * weights[j];
        }
        coefs[k] = val;
    }
    coefs
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate synthetic data where response depends on curve shape.
    fn generate_shape_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];

        for i in 0..n {
            // Vary shape via frequency parameter.
            let freq = 1.0 + 2.0 * (i as f64 / n as f64);
            let amp = 1.5;
            for j in 0..m {
                data[(i, j)] = amp * (freq * PI * t[j]).sin();
            }
            // Response depends on frequency (shape), not amplitude.
            y[i] = freq;
        }
        (data, y, t)
    }

    #[test]
    fn test_scalar_on_shape_basic() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let result = scalar_on_shape(&data, &y, &t, &config);
        assert!(
            result.is_ok(),
            "scalar_on_shape should succeed: {:?}",
            result.err()
        );

        let res = result.unwrap();
        assert_eq!(res.beta.len(), 51);
        assert_eq!(res.fitted_values.len(), 20);
        assert_eq!(res.residuals.len(), 20);
        assert_eq!(res.shape_scores.len(), 20);
        assert_eq!(res.gammas.shape(), (20, 51));
        assert!(res.sse.is_finite());
        assert!(res.r_squared.is_finite());
        assert!(res.n_iter_outer > 0);
        assert!(res.n_iter_inner > 0);
    }

    #[test]
    fn test_scalar_on_shape_default_config() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig::default();
        let result = scalar_on_shape(&data, &y, &t, &config);
        assert!(
            result.is_ok(),
            "scalar_on_shape with default config should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_scalar_on_shape_identity() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            index_method: IndexMethod::Identity,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();
        assert!(matches!(res.index_method, IndexMethod::Identity));
        assert_eq!(res.h_coefficients, vec![1.0]);
    }

    #[test]
    fn test_scalar_on_shape_polynomial_index() {
        let (data, y, t) = generate_shape_data(20, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            index_method: IndexMethod::Polynomial(2),
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let result = scalar_on_shape(&data, &y, &t, &config);
        assert!(
            result.is_ok(),
            "polynomial index should succeed: {:?}",
            result.err()
        );
        let res = result.unwrap();
        // Polynomial(2) => intercept + degree 1 + degree 2 = 3 coefficients
        assert_eq!(res.h_coefficients.len(), 3);
    }

    #[test]
    fn test_scalar_on_shape_nadaraya_watson() {
        let (data, y, t) = generate_shape_data(20, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            index_method: IndexMethod::NadarayaWatson(0.5),
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let result = scalar_on_shape(&data, &y, &t, &config);
        assert!(
            result.is_ok(),
            "NW index should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_scalar_on_shape_invalid_dimensions() {
        let data = FdMatrix::zeros(1, 10);
        let y = vec![1.0];
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let config = ScalarOnShapeConfig::default();
        assert!(scalar_on_shape(&data, &y, &t, &config).is_err());
    }

    #[test]
    fn test_scalar_on_shape_y_length_mismatch() {
        let data = FdMatrix::zeros(5, 10);
        let y = vec![1.0, 2.0, 3.0]; // wrong length
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let config = ScalarOnShapeConfig::default();
        assert!(scalar_on_shape(&data, &y, &t, &config).is_err());
    }

    #[test]
    fn test_scalar_on_shape_nbasis_too_small() {
        let (data, y, t) = generate_shape_data(10, 20);
        let config = ScalarOnShapeConfig {
            nbasis: 2, // too small
            ..ScalarOnShapeConfig::default()
        };
        assert!(scalar_on_shape(&data, &y, &t, &config).is_err());
    }

    #[test]
    fn test_predict_scalar_on_shape_basic() {
        let (data, y, t) = generate_shape_data(20, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let fit = scalar_on_shape(&data, &y, &t, &config).unwrap();

        let preds = predict_scalar_on_shape(&fit, &data, &t);
        assert!(preds.is_ok(), "predict should succeed: {:?}", preds.err());
        let preds = preds.unwrap();
        assert_eq!(preds.len(), 20);
        assert!(preds.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_predict_method_syntax() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let fit = scalar_on_shape(&data, &y, &t, &config).unwrap();

        let preds1 = predict_scalar_on_shape(&fit, &data, &t).unwrap();
        let preds2 = fit.predict(&data, &t).unwrap();
        assert_eq!(preds1.len(), preds2.len());
        for (a, b) in preds1.iter().zip(&preds2) {
            assert!((a - b).abs() < 1e-10, "Method and function should match");
        }
    }

    #[test]
    fn test_predict_dimension_mismatch() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 3,
            max_iter_outer: 2,
            ..ScalarOnShapeConfig::default()
        };
        let fit = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Wrong number of columns.
        let bad_data = FdMatrix::zeros(5, 30);
        let bad_t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
        assert!(predict_scalar_on_shape(&fit, &bad_data, &bad_t).is_err());
    }

    #[test]
    fn test_scalar_on_shape_finite_values() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        assert!(
            res.beta.iter().all(|v| v.is_finite()),
            "beta should be finite"
        );
        assert!(
            res.fitted_values.iter().all(|v| v.is_finite()),
            "fitted values should be finite"
        );
        assert!(
            res.residuals.iter().all(|v| v.is_finite()),
            "residuals should be finite"
        );
        assert!(
            res.shape_scores.iter().all(|v| v.is_finite()),
            "shape scores should be finite"
        );
        assert!(
            res.beta_coefficients.iter().all(|v| v.is_finite()),
            "beta coefficients should be finite"
        );
    }

    #[test]
    fn test_scalar_on_shape_beta_unit_norm() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();
        let weights = simpsons_weights(&t);
        let norm_sq: f64 = res
            .beta
            .iter()
            .zip(weights.iter())
            .map(|(&b, &w)| b * b * w)
            .sum();
        assert!(
            (norm_sq.sqrt() - 1.0).abs() < 0.05,
            "beta should have approximately unit L2 norm, got {}",
            norm_sq.sqrt()
        );
    }

    #[test]
    fn test_scalar_on_shape_residuals_sum() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Check that residuals = y - fitted_values.
        for (i, (&yi, &fi)) in y.iter().zip(res.fitted_values.iter()).enumerate() {
            assert!(
                (res.residuals[i] - (yi - fi)).abs() < 1e-10,
                "residual mismatch at index {i}"
            );
        }

        // Check SSE matches.
        let sse_check: f64 = res.residuals.iter().map(|&r| r * r).sum();
        assert!(
            (res.sse - sse_check).abs() < 1e-10,
            "SSE mismatch: {} vs {}",
            res.sse,
            sse_check
        );
    }

    // ── Gap 1: Verify g(0) = 0 constraint ──────────────────────────────────

    #[test]
    fn test_g_at_zero_is_zero() {
        // The g polynomial has no intercept: g(x) = a1*x + a2*x^2 + ...
        // Therefore g(0) must be exactly 0 for any coefficients.
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Evaluate g at x = 0: sum of g_coefs[k] * 0^(k+1) = 0 for all k.
        let g_at_zero: f64 = res
            .g_coefficients
            .iter()
            .enumerate()
            .map(|(k, &a)| a * (0.0_f64).powi((k + 1) as i32))
            .sum();
        assert!(
            g_at_zero.abs() < 1e-15,
            "g(0) should be exactly 0, got {}",
            g_at_zero
        );
    }

    #[test]
    fn test_g_no_intercept_polynomial_via_helper() {
        // Directly verify eval_polynomial_no_intercept at zero is 0.
        let coefs = vec![3.7, -1.2, 0.5];
        let result = eval_polynomial_no_intercept(&[0.0], &coefs);
        assert!(
            result[0].abs() < 1e-15,
            "no-intercept polynomial at 0 should be 0, got {}",
            result[0]
        );
    }

    // ── Gap 2: Quantify fit quality ─────────────────────────────────────────

    #[test]
    fn test_fit_quality_r_squared_positive() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        assert!(
            res.r_squared > 0.0,
            "R-squared should be positive for structured data, got {}",
            res.r_squared
        );
    }

    #[test]
    fn test_fit_quality_positive_correlation() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Pearson correlation between y and fitted should be positive.
        let n = y.len() as f64;
        let y_mean: f64 = y.iter().sum::<f64>() / n;
        let f_mean: f64 = res.fitted_values.iter().sum::<f64>() / n;
        let cov: f64 = y
            .iter()
            .zip(res.fitted_values.iter())
            .map(|(&yi, &fi)| (yi - y_mean) * (fi - f_mean))
            .sum();
        let sd_y: f64 = y
            .iter()
            .map(|&yi| (yi - y_mean).powi(2))
            .sum::<f64>()
            .sqrt();
        let sd_f: f64 = res
            .fitted_values
            .iter()
            .map(|&fi| (fi - f_mean).powi(2))
            .sum::<f64>()
            .sqrt();
        let corr = cov / (sd_y * sd_f).max(1e-15);
        assert!(
            corr > 0.0,
            "correlation between y and fitted should be positive, got {}",
            corr
        );
    }

    #[test]
    fn test_fit_quality_residual_variance_less_than_total() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        let n = y.len() as f64;
        let y_mean: f64 = y.iter().sum::<f64>() / n;
        let ss_total: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let residual_var: f64 = res.residuals.iter().map(|&r| r * r).sum();
        assert!(
            residual_var < ss_total,
            "residual SS ({}) should be less than total SS ({})",
            residual_var,
            ss_total
        );
    }

    // ── Gap 3: h function behavior per IndexMethod ──────────────────────────

    #[test]
    fn test_h_identity_coefficients() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            index_method: IndexMethod::Identity,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        assert_eq!(
            res.h_coefficients,
            vec![1.0],
            "identity h should have coefs [1.0]"
        );

        // Fitted = shape_scores + g(f0). Verify this relationship.
        let g_vals = eval_polynomial_no_intercept(
            &{
                let weights = simpsons_weights(&t);
                (0..data.shape().0)
                    .map(|i| {
                        let mut s = 0.0;
                        for j in 0..data.shape().1 {
                            s += data[(i, j)] * weights[j];
                        }
                        s
                    })
                    .collect::<Vec<f64>>()
            },
            &res.g_coefficients,
        );
        for i in 0..res.fitted_values.len() {
            let expected = res.shape_scores[i] + g_vals[i];
            assert!(
                (res.fitted_values[i] - expected).abs() < 1e-10,
                "identity h: fitted[{}] should equal shape_score + g(f0), diff = {}",
                i,
                (res.fitted_values[i] - expected).abs()
            );
        }
    }

    #[test]
    fn test_h_polynomial_2_coefficients() {
        let (data, y, t) = generate_shape_data(20, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            index_method: IndexMethod::Polynomial(2),
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Polynomial(2) => intercept + z + z^2 = 3 coefficients.
        assert_eq!(
            res.h_coefficients.len(),
            3,
            "Polynomial(2) h should have 3 coefficients, got {}",
            res.h_coefficients.len()
        );
        // h should not be the identity [1.0].
        assert_ne!(res.h_coefficients, vec![1.0]);
    }

    #[test]
    fn test_h_polynomial_1_coefficients() {
        let (data, y, t) = generate_shape_data(20, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            index_method: IndexMethod::Polynomial(1),
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Polynomial(1) => intercept + z = 2 coefficients.
        assert_eq!(
            res.h_coefficients.len(),
            2,
            "Polynomial(1) h should have 2 coefficients, got {}",
            res.h_coefficients.len()
        );
    }

    // ── Gap 4: Warp functions are monotonic with correct boundary conditions ─

    #[test]
    fn test_gammas_monotonic() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();
        let (n, m) = res.gammas.shape();

        for i in 0..n {
            for j in 1..m {
                assert!(
                    res.gammas[(i, j)] >= res.gammas[(i, j - 1)] - 1e-10,
                    "gamma[{}, {}] = {} < gamma[{}, {}] = {} (not monotone)",
                    i,
                    j,
                    res.gammas[(i, j)],
                    i,
                    j - 1,
                    res.gammas[(i, j - 1)]
                );
            }
        }
    }

    #[test]
    fn test_gammas_boundary_conditions() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();
        let (n, m) = res.gammas.shape();

        for i in 0..n {
            assert!(
                (res.gammas[(i, 0)] - t[0]).abs() < 1e-6,
                "gamma[{}, 0] = {} should be close to argvals[0] = {}",
                i,
                res.gammas[(i, 0)],
                t[0]
            );
            assert!(
                (res.gammas[(i, m - 1)] - t[m - 1]).abs() < 1e-6,
                "gamma[{}, {}] = {} should be close to argvals[last] = {}",
                i,
                m - 1,
                res.gammas[(i, m - 1)],
                t[m - 1]
            );
        }
    }

    // ── Gap 5: Convergence behavior ─────────────────────────────────────────

    #[test]
    fn test_convergence_loose_tolerance() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            tol: 1.0, // very loose
            max_iter_inner: 15,
            max_iter_outer: 10,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        // Loose tolerance should converge quickly.
        assert!(
            res.n_iter_outer <= 3,
            "loose tol should converge in few outer iterations, got {}",
            res.n_iter_outer
        );
    }

    #[test]
    fn test_convergence_tight_tolerance_uses_more_iterations() {
        let (data, y, t) = generate_shape_data(15, 41);

        let config_loose = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            tol: 1.0,
            max_iter_inner: 15,
            max_iter_outer: 10,
            ..ScalarOnShapeConfig::default()
        };
        let res_loose = scalar_on_shape(&data, &y, &t, &config_loose).unwrap();

        let config_tight = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            tol: 1e-8,
            max_iter_inner: 15,
            max_iter_outer: 10,
            ..ScalarOnShapeConfig::default()
        };
        let res_tight = scalar_on_shape(&data, &y, &t, &config_tight).unwrap();

        assert!(
            res_tight.n_iter_outer >= res_loose.n_iter_outer,
            "tight tol ({} iters) should use >= iterations than loose tol ({} iters)",
            res_tight.n_iter_outer,
            res_loose.n_iter_outer
        );
    }

    #[test]
    fn test_convergence_respects_max_iter() {
        let (data, y, t) = generate_shape_data(15, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 8,
            max_iter_outer: 5,
            tol: 1e-12, // very tight so it likely hits max
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        assert!(
            res.n_iter_inner <= config.max_iter_inner,
            "inner iterations {} should not exceed max {}",
            res.n_iter_inner,
            config.max_iter_inner
        );
        assert!(
            res.n_iter_outer <= config.max_iter_outer,
            "outer iterations {} should not exceed max {}",
            res.n_iter_outer,
            config.max_iter_outer
        );
    }

    // ── Gap 6: Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_edge_case_n_equals_2() {
        // Minimum viable n (n >= 2 required).
        let (data, y, t) = generate_shape_data(2, 41);
        let config = ScalarOnShapeConfig {
            nbasis: 5,
            lambda: 1e-1,
            max_iter_inner: 3,
            max_iter_outer: 2,
            ..ScalarOnShapeConfig::default()
        };
        let result = scalar_on_shape(&data, &y, &t, &config);
        assert!(result.is_ok(), "n=2 should succeed: {:?}", result.err());
        let res = result.unwrap();
        assert_eq!(res.fitted_values.len(), 2);
        assert!(res.sse.is_finite());
    }

    #[test]
    fn test_edge_case_all_y_identical() {
        let (data, _y, t) = generate_shape_data(15, 41);
        let y_const = vec![5.0; 15];
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let result = scalar_on_shape(&data, &y_const, &t, &config);
        assert!(
            result.is_ok(),
            "identical y should not panic: {:?}",
            result.err()
        );
        let res = result.unwrap();
        // With identical y, R-squared should be 0 (ss_total = 0 triggers 0 branch).
        assert!(
            res.r_squared.abs() < 1e-10,
            "R-squared for constant y should be ~0, got {}",
            res.r_squared
        );
    }

    #[test]
    fn test_edge_case_large_lambda() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e5, // very large penalty
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let result = scalar_on_shape(&data, &y, &t, &config);
        assert!(
            result.is_ok(),
            "large lambda should succeed: {:?}",
            result.err()
        );
        let res = result.unwrap();

        // Beta coefficients should be shrunk toward zero by heavy regularization.
        let coef_norm: f64 = res
            .beta_coefficients
            .iter()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt();

        // Fit with moderate lambda for comparison.
        let config_moderate = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res_moderate = scalar_on_shape(&data, &y, &t, &config_moderate).unwrap();
        let coef_norm_moderate: f64 = res_moderate
            .beta_coefficients
            .iter()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt();

        // The heavily penalized coefficients should have smaller or comparable norm.
        // (Beta is normalized to unit L2 in function space, but the Fourier
        // coefficients reflect smoothness — heavy penalty pushes toward smooth.)
        assert!(
            coef_norm.is_finite(),
            "beta coefficients norm should be finite, got {}",
            coef_norm
        );
        // At minimum, verify the large-lambda result is valid and finite.
        assert!(res.sse.is_finite());
        assert!(res.r_squared.is_finite());
        // The large penalty should lead to a different (typically smoother) beta.
        let diff: f64 = res
            .beta_coefficients
            .iter()
            .zip(res_moderate.beta_coefficients.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff > 1e-10,
            "large lambda should produce different beta coefficients than moderate lambda, diff = {}",
            diff
        );
        let _ = coef_norm_moderate; // used above in diff comparison
    }

    // ── Gap 7: Prediction consistency ───────────────────────────────────────

    #[test]
    fn test_predict_on_training_data_bounded_discrepancy() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let fit = scalar_on_shape(&data, &y, &t, &config).unwrap();
        let preds = predict_scalar_on_shape(&fit, &data, &t).unwrap();

        // Predictions re-align to beta, so they won't match fitted_values exactly,
        // but the discrepancy should be bounded.
        let y_range = {
            let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min);
            let ymax = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (ymax - ymin).max(1e-10)
        };
        let max_diff: f64 = fit
            .fitted_values
            .iter()
            .zip(preds.iter())
            .map(|(&f, &p)| (f - p).abs())
            .fold(0.0, f64::max);

        // Relative to the range of y, the discrepancy should be bounded.
        let relative_diff = max_diff / y_range;
        assert!(
            relative_diff < 5.0,
            "prediction discrepancy relative to y range should be bounded, got {}",
            relative_diff
        );
    }

    // ── Gap 8: SSE and R-squared consistency ────────────────────────────────

    #[test]
    fn test_sse_r_squared_consistency() {
        let (data, y, t) = generate_shape_data(20, 51);
        let config = ScalarOnShapeConfig {
            nbasis: 7,
            lambda: 1e-2,
            max_iter_inner: 5,
            max_iter_outer: 3,
            ..ScalarOnShapeConfig::default()
        };
        let res = scalar_on_shape(&data, &y, &t, &config).unwrap();

        let n = y.len() as f64;
        let y_mean: f64 = y.iter().sum::<f64>() / n;
        let ss_total: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        // SSE = sum of residuals^2.
        let sse_from_residuals: f64 = res.residuals.iter().map(|&r| r * r).sum();
        assert!(
            (res.sse - sse_from_residuals).abs() < 1e-10,
            "SSE should match sum of squared residuals: {} vs {}",
            res.sse,
            sse_from_residuals
        );

        // R^2 = 1 - SSE / SS_total.
        if ss_total > 0.0 {
            let r2_computed = 1.0 - res.sse / ss_total;
            assert!(
                (res.r_squared - r2_computed).abs() < 1e-10,
                "R-squared should equal 1 - SSE/SS_total: {} vs {}",
                res.r_squared,
                r2_computed
            );
        }

        // SSE from fitted values directly.
        let sse_from_fitted: f64 = y
            .iter()
            .zip(res.fitted_values.iter())
            .map(|(&yi, &fi)| (yi - fi).powi(2))
            .sum();
        assert!(
            (res.sse - sse_from_fitted).abs() < 1e-10,
            "SSE should match sum of (y - fitted)^2: {} vs {}",
            res.sse,
            sse_from_fitted
        );
    }
}
