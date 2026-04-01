use super::{
    build_design_matrix, cholesky_solve, compute_fitted, compute_r_squared, recover_beta_t,
    resolve_ncomp, validate_fregre_inputs, FregreRobustResult,
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

/// Maximum number of IRLS iterations.
const MAX_ITER: usize = 100;
/// Convergence tolerance on relative coefficient change.
const CONV_TOL: f64 = 1e-6;
/// Epsilon floor to avoid division by zero in L1 weights.
const L1_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Weighted least squares helper
// ---------------------------------------------------------------------------

/// Compute X'WX (symmetric, p×p stored flat row-major).
fn compute_xtwx(x: &FdMatrix, w: &[f64]) -> Vec<f64> {
    let (n, p) = x.shape();
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += w[i] * x[(i, k)] * x[(i, j)];
            }
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }
    xtwx
}

/// Compute X'Wy (length p).
fn compute_xtwy(x: &FdMatrix, y: &[f64], w: &[f64]) -> Vec<f64> {
    let (n, p) = x.shape();
    (0..p)
        .map(|k| {
            let mut s = 0.0;
            for i in 0..n {
                s += w[i] * x[(i, k)] * y[i];
            }
            s
        })
        .collect()
}

/// Solve weighted least squares: min Σ w_i (y_i - x_i'β)² via normal equations.
fn wls_solve(x: &FdMatrix, y: &[f64], w: &[f64]) -> Result<Vec<f64>, FdarError> {
    let p = x.ncols();
    let xtwx = compute_xtwx(x, w);
    let xtwy = compute_xtwy(x, y, w);
    cholesky_solve(&xtwx, &xtwy, p)
}

/// Solve unweighted OLS via Cholesky (for initialization).
fn ols_init(x: &FdMatrix, y: &[f64]) -> Result<Vec<f64>, FdarError> {
    let n = x.nrows();
    let w = vec![1.0; n];
    wls_solve(x, y, &w)
}

/// Relative change in coefficient vector: ||β_new - β_old|| / (||β_old|| + ε).
fn relative_change(beta_new: &[f64], beta_old: &[f64]) -> f64 {
    let diff_norm: f64 = beta_new
        .iter()
        .zip(beta_old)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let old_norm: f64 = beta_old.iter().map(|&b| b * b).sum::<f64>().sqrt();
    diff_norm / (old_norm + 1e-12)
}

// ---------------------------------------------------------------------------
// L1 (Least Absolute Deviations) regression
// ---------------------------------------------------------------------------

/// L1 (median) functional regression via FPCA + IRLS.
///
/// Uses iteratively reweighted least squares to minimise
/// `Σ|y_i - x_i'β|` instead of `Σ(y_i - x_i'β)²`.
/// More robust to outliers than OLS.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector (length n)
/// * `scalar_covariates` - Optional scalar covariates matrix (n × p)
/// * `ncomp` - Number of FPC components (if 0, selected by GCV)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, `y.len() != n`, or `scalar_covariates` row count differs from `n`.
/// Returns [`FdarError::InvalidParameter`] if auto-selected `ncomp` via CV fails.
/// Returns [`FdarError::ComputationFailed`] if FPCA fails or the normal equations
/// are singular.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_l1;
///
/// let (n, m) = (20, 30);
/// let data = FdMatrix::from_column_major(
///     (0..n * m).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(),
///     n, m,
/// ).unwrap();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
/// let fit = fregre_l1(&data, &y, None, 3).unwrap();
/// assert_eq!(fit.fitted_values.len(), 20);
/// assert_eq!(fit.beta_t.len(), 30);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_l1(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<FregreRobustResult, FdarError> {
    fregre_robust_irls(data, y, scalar_covariates, ncomp, RobustType::L1)
}

// ---------------------------------------------------------------------------
// Huber M-estimation
// ---------------------------------------------------------------------------

/// Huber M-estimation functional regression via FPCA + IRLS.
///
/// Uses the Huber loss: `L(r) = r²/2` for `|r| ≤ k`, `k|r| - k²/2` for `|r| > k`.
/// Combines the efficiency of OLS for small residuals with the robustness of L1
/// for large residuals.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector (length n)
/// * `scalar_covariates` - Optional scalar covariates matrix (n × p)
/// * `ncomp` - Number of FPC components (if 0, selected by GCV)
/// * `k` - Huber tuning constant (default 1.345 for 95% asymptotic efficiency
///   relative to OLS under Gaussian errors)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, `y.len() != n`, or `scalar_covariates` row count differs from `n`.
/// Returns [`FdarError::InvalidParameter`] if `k <= 0` or auto-selected `ncomp`
/// via CV fails.
/// Returns [`FdarError::ComputationFailed`] if FPCA fails or the normal equations
/// are singular.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_huber;
///
/// let (n, m) = (20, 30);
/// let data = FdMatrix::from_column_major(
///     (0..n * m).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(),
///     n, m,
/// ).unwrap();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
/// let fit = fregre_huber(&data, &y, None, 3, 1.345).unwrap();
/// assert_eq!(fit.fitted_values.len(), 20);
/// assert!(fit.converged);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_huber(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    k: f64,
) -> Result<FregreRobustResult, FdarError> {
    if k <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("Huber tuning constant must be positive, got {k}"),
        });
    }
    fregre_robust_irls(data, y, scalar_covariates, ncomp, RobustType::Huber(k))
}

// ---------------------------------------------------------------------------
// Unified IRLS engine
// ---------------------------------------------------------------------------

/// Internal enum to select the robust weight function.
enum RobustType {
    L1,
    Huber(f64),
}

/// Compute IRLS weights from residuals.
fn compute_weights(residuals: &[f64], robust_type: &RobustType) -> Vec<f64> {
    match robust_type {
        RobustType::L1 => residuals
            .iter()
            .map(|&r| 1.0 / r.abs().max(L1_EPS))
            .collect(),
        RobustType::Huber(k) => residuals
            .iter()
            .map(|&r| {
                let abs_r = r.abs();
                if abs_r <= *k {
                    1.0
                } else {
                    *k / abs_r
                }
            })
            .collect(),
    }
}

/// Unified IRLS implementation for L1 and Huber robust regression.
fn fregre_robust_irls(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    robust_type: RobustType,
) -> Result<FregreRobustResult, FdarError> {
    let (n, m) = data.shape();
    validate_fregre_inputs(n, m, y, scalar_covariates)?;

    let ncomp = resolve_ncomp(ncomp, data, y, scalar_covariates, n, m)?;

    // Step 1: FPCA
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;

    // Step 2: Build design matrix [1, scores, scalar_covariates]
    let design = build_design_matrix(&fpca.scores, ncomp, scalar_covariates, n);
    let p_total = design.ncols();

    // Step 3: Initialize with OLS solution
    let mut coeffs = ols_init(&design, y)?;

    // Step 4: IRLS loop
    let mut iterations = 0;
    let mut converged = false;
    let mut weights = vec![1.0; n];

    for iter in 0..MAX_ITER {
        // Compute residuals
        let fitted = compute_fitted(&design, &coeffs);
        let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(&yi, &yh)| yi - yh).collect();

        // Compute weights
        weights = compute_weights(&residuals, &robust_type);

        // Solve weighted least squares
        let new_coeffs = wls_solve(&design, y, &weights)?;

        // Check convergence
        let rel_change = relative_change(&new_coeffs, &coeffs);
        coeffs = new_coeffs;
        iterations = iter + 1;

        if rel_change < CONV_TOL {
            converged = true;
            break;
        }
    }

    // Step 5: Compute final fitted values and residuals
    let fitted_values = compute_fitted(&design, &coeffs);
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();

    // Recompute final weights from final residuals
    weights = compute_weights(&residuals, &robust_type);

    let (r_squared, _) = compute_r_squared(y, &residuals, p_total);

    // Recover β(t) from FPC coefficients
    let beta_t = recover_beta_t(&coeffs[1..=ncomp], &fpca.rotation, m);

    Ok(FregreRobustResult {
        intercept: coeffs[0],
        beta_t,
        fitted_values,
        residuals,
        coefficients: coeffs,
        ncomp,
        fpca,
        iterations,
        converged,
        weights,
        r_squared,
    })
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// Predict new responses using a fitted robust functional regression model.
///
/// Projects `new_data` onto the FPCA basis stored in `fit`, then applies
/// the estimated coefficients.
///
/// # Arguments
/// * `fit` - A fitted [`FregreRobustResult`]
/// * `new_data` - New functional predictor matrix (n_new × m)
/// * `new_scalar` - Optional new scalar covariates (n_new × p)
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::{fregre_l1, predict_fregre_robust};
///
/// let (n, m) = (20, 30);
/// let data = FdMatrix::from_column_major(
///     (0..n * m).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(),
///     n, m,
/// ).unwrap();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
/// let fit = fregre_l1(&data, &y, None, 3).unwrap();
/// let preds = predict_fregre_robust(&fit, &data, None);
/// assert_eq!(preds.len(), 20);
/// ```
pub fn predict_fregre_robust(
    fit: &FregreRobustResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Vec<f64> {
    let (n_new, m) = new_data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.coefficients.len() - 1 - ncomp;

    let mut predictions = vec![0.0; n_new];
    for i in 0..n_new {
        let mut yhat = fit.intercept;
        // Project onto FPC basis: ξ_k = Σ_j (x(t_j) - μ(t_j)) · φ_k(t_j) · w_j
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (new_data[(i, j)] - fit.fpca.mean[j])
                    * fit.fpca.rotation[(j, k)]
                    * fit.fpca.weights[j];
            }
            yhat += fit.coefficients[1 + k] * s;
        }
        // Add scalar covariates
        if let Some(sc) = new_scalar {
            for j in 0..p_scalar {
                yhat += fit.coefficients[1 + ncomp + j] * sc[(i, j)];
            }
        }
        predictions[i] = yhat;
    }
    predictions
}
